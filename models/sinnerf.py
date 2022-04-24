from torchvision import models as tmodels
import torch
import torch.nn as nn
from collections import defaultdict

from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.discriminator import Discriminator
from models.rendering import render_rays, eval_points
from models.diff_aug import DiffAugment
from models.extractor import VitExtractor


# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict
from kornia.losses import inverse_depth_smoothness_loss

# metrics
from metrics import *

from pytorch_lightning import LightningModule, Trainer
from einops import rearrange


class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None, useMask=True):
        if None == mask and useMask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask])  # * 2 ** (1 - 2)
        return loss


def project_with_depth(side_coord, depth_ref, proj_ref, proj_src):
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    y_ref, x_ref = torch.meshgrid(
        [side_coord[0, 0].float(), side_coord[0, 1].float()])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    xyz_ref = torch.stack((x_ref, y_ref, torch.ones_like(x_ref, device=x_ref.device))).unsqueeze(
        0) * (depth_ref.view(batchsize, -1).unsqueeze(1))  # [b, 1, n]
    xyz_homo_coord = torch.cat((xyz_ref, torch.ones_like(
        xyz_ref[:, :1, :], device=xyz_ref.device)), dim=1)
    K_xyz_src = torch.matmul(proj_src, torch.matmul(
        torch.inverse(proj_ref), xyz_homo_coord))[:, :3, :]  # [n, 3, n]
    depth_src = K_xyz_src[:, 2:3, :]
    xy_src = K_xyz_src[:, :2, :] / K_xyz_src[:, 2:3, :]
    x_src = xy_src[:, 0, :].view([batchsize, height, width])
    y_src = xy_src[:, 1, :].view([batchsize, height, width])
    return x_src, y_src, depth_src

# (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')


def forward_warp(ref_depth_full, side_coord, depth_ref, proj_ref, proj_src):
    # depth_ref is [b, 1, 64, 64]
    # print(depth_ref.shape, side_coord.shape, ref_depth_full.shape)
    x_res, y_res, depth_src = project_with_depth(
        side_coord, depth_ref, proj_ref, proj_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    depth_src = depth_src.reshape(height, width)
    new_depth = torch.zeros_like(
        ref_depth_full[0], device=ref_depth_full.device)
    yy_base, xx_base = torch.meshgrid([torch.arange(0, height, dtype=torch.long, device=depth_ref.device), torch.arange(
        0, width, dtype=torch.long, device=depth_ref.device)])
    y_res = torch.clamp(y_res, 0, 400 - 1).long()
    x_res = torch.clamp(x_res, 0, 400 - 1).long()
    new_depth[y_res, x_res] = depth_src[yy_base, xx_base]
    return new_depth


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        create_label = True
        if target_is_real:
            # create_label = ((self.real_label_var is None) or
            if create_label:
                self.real_label_var = torch.zeros(
                    input.shape, device=input.device, requires_grad=False).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            if create_label:
                self.fake_label_var = torch.zeros(
                    input.shape, device=input.device, requires_grad=False).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, mask=None):
        if mask is None:
            target_tensor = self.get_target_tensor(input, target_is_real)
        else:
            target_tensor = mask  # .detach()
        return self.loss(input, target_tensor)


class SinNeRF(LightningModule):
    def __init__(self, hparams):
        super(SinNeRF, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()
        self.patch_loss = loss_dict[hparams.patch_loss]()
        self.s1 = SL1Loss()

        self.embedding_xyz = Embedding(3, 10)  # 10 is the default number
        self.embedding_dir = Embedding(3, 4)  # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF(use_new_activation=True)
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(use_new_activation=True)
            self.models += [self.nerf_fine]

        if self.hparams.dis_weight > 0:
            self.D = Discriminator(
                conditional=False, policy='color,cutout', imsize=self.hparams.patch_size)

        if self.hparams.vit_weight > 0:
            self.ext = VitExtractor(
                model_name='dino_vits16', device=self.device)

        self.ref_ = None

        self.criterionGAN = GANLoss()
        self.l2 = nn.MSELoss()
        self.init_data()

    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        rgbs = batch['rgbs']  # (B, 3)
        return rays, rgbs

    def get_vit_feature(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).reshape(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        return self.ext.get_feature_from_input(x)[-1][0, 0, :]

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def init_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        # kwargs = vars(self.hparams)
        kwargs = self.hparams
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)

        scheduler = get_scheduler(self.hparams, self.optimizer)
        li = [self.optimizer]
        if self.hparams.dis_weight > 0:
            self.opt_d = get_optimizer(self.hparams, [self.D], rate=0.2)
            li.append(self.opt_d)
        return li, [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=2,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=False)

    def compute_grad2(self, d_outs, x_in):
        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        reg = 0
        for d_out in d_outs:
            batch_size = x_in.size(0)
            grad_dout = torch.autograd.grad(
                outputs=d_out.sum(), inputs=x_in,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_dout2 = grad_dout.pow(2)
            assert (grad_dout2.size() == x_in.size())
            reg += grad_dout2.view(batch_size, -1).sum(1)
        return reg / len(d_outs)

    def compute_loss(self, d_outs, target):
        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = torch.tensor(0.0, device=d_outs[0].device)

        for d_out in d_outs:

            targets = d_out.new_full(size=d_out.size(), fill_value=target)

            if self.hparams.dloss == 'standard':
                loss += F.binary_cross_entropy_with_logits(d_out, targets)
            elif self.hparams.dloss in ['wgan', 'wgan_gp']:
                loss += (2 * target - 1) * d_out.mean()
            else:
                raise NotImplementedError

        return loss / len(d_outs)

    def wgan_gp_reg(self, x_real, x_fake, y=1, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(
            batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.D(x_interp, y)

        reg = (self.compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        with torch.no_grad():
            if (self.ref_ is None or (np.random.random() > 0.95)) and self.hparams.vit_weight > 0:
                self.ref_ = self.get_vit_feature(batch['real_patch'])
                self.ref_.requires_grad = False

        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        w, h = self.hparams.img_wh
        psx, psy = batch['real_patch'].shape[-2:]
        ps = self.hparams.patch_size
        rays_full = batch['depth_ray']
        rgbs_full = rearrange(batch['depth_ray_rgb'],
                              'b (p q) c -> b c p q', c=3, p=psx, q=psy)
        rays_side = batch['rays_full']  # ref view
        real_patch = batch['real_patch']
        bs = real_patch.shape[0]
        rays = rearrange(rays, 'b n c -> (b n) c', c=8)
        depth = rearrange(batch['depth'], 'b n c -> (b n) c', c=1)
        if self.hparams.dataset_name != 'llff_dual_patch':
            rgbs = rearrange(rgbs, 'b n c -> (b n) c', c=3)
        rays_full = rearrange(rays_full, 'b n c -> (b n) c', c=8)
        rays_side = rearrange(rays_side, 'b n c -> (b n) c', c=8)

        rand_rays_proj = batch['rays_proj']
        rand_depth_proj = batch['depth_proj']
        rand_rays_proj = rearrange(rand_rays_proj, 'b n c -> (b n) c', c=8)
        rand_depth_proj = rearrange(rand_depth_proj, 'b n c -> (b n) c', c=1)

        mask = batch['warp_patch'].sum(1) > 0
        side_rgb = batch['warp_patch'] * (mask) + torch.ones_like(
            batch['warp_patch'], device=mask.device) * (~mask)

        results = self.forward(rays)
        results_full = self.forward(rays_full)
        results_side = self.forward(rays_side)
        results_proj = self.forward(rand_rays_proj)

        rand_depth_proj = rand_depth_proj.squeeze()  # [b, 1]
        loss_depth = self.s1(results_proj['depth_fine'], rand_depth_proj, useMask=False) + self.s1(
            results_proj['depth_coarse'], rand_depth_proj, useMask=False)
        if self.hparams.dataset_name == 'llff_dual_patch':
            results['rgb_coarse'] = rearrange(
                results['rgb_coarse'], '(b w h) c -> b c w h', c=3, w=ps, h=ps)
            results['rgb_fine'] = rearrange(
                results['rgb_fine'], '(b w h) c -> b c w h', c=3, w=ps, h=ps)
        loss_g = self.loss(results, rgbs)
        loss_depth += self.s1(results['depth_fine'], depth, useMask=False) + \
            self.s1(results['depth_coarse'], depth, useMask=False)
        need_zero = depth.view(-1, 1) == 0

        results_full['rgb_coarse'] = rearrange(
            results_full['rgb_coarse'], '(b p q) c -> b c p q', c=3, p=psx, q=psy)
        results_full['rgb_fine'] = rearrange(
            results_full['rgb_fine'], '(b p q) c -> b c p q', c=3, p=psx, q=psy)

        results_side['rgb_coarse'] = rearrange(
            results_side['rgb_coarse'], '(b p q) c -> b c p q', c=3, p=psx, q=psy)
        results_side['rgb_fine'] = rearrange(
            results_side['rgb_fine'], '(b p q) c -> b c p q', c=3, p=psx, q=psy)

        if self.hparams.vit_weight > 0:
            results_semantics_coarse = self.get_vit_feature(
                results_side['rgb_coarse'])
            results_semantics_fine = self.get_vit_feature(
                results_side['rgb_fine'])
            loss_vit = F.mse_loss(results_semantics_coarse, self.ref_) + \
                F.mse_loss(results_semantics_fine, self.ref_)
        else:
            loss_vit = 0

        if batch_idx % 10 == 0:
            self.logger.experiment.add_scalar(
                'train/depth_min', results_full['depth_fine'].min().detach(), self.global_step)
            self.logger.experiment.add_scalar(
                'train/depth_max', results_full['depth_fine'].max().detach(), self.global_step)

        rgb_loss = self.patch_loss(results_full, rgbs_full)
        for k, v in rgb_loss.items():
            if k in loss_g:
                loss_g[k] += v
            else:
                loss_g[k] = v
        depth_gt = rearrange(
            batch['depth_gt'], 'b (p q) c -> b c p q', c=1, p=psx, q=psy)
        results_full['depth_fine'] = rearrange(
            results_full['depth_fine'], '(b p q c) -> b c p q', c=1, p=psx, q=psy)
        results_full['depth_coarse'] = rearrange(
            results_full['depth_coarse'], '(b p q c) -> b c p q', c=1, p=psx, q=psy)
        if self.hparams.dataset_name == 'dtu_proj':
            loss_depth_patch = self.s1(
                results_full['depth_fine'].view(-1), depth_gt.view(-1))
            loss_depth_patch += self.s1(
                results_full['depth_coarse'].view(-1), depth_gt.view(-1))
            loss_depth += loss_depth_patch
        else:
            loss_depth_patch = self.patch_loss(
                {'rgb_fine': results_full['depth_fine'], 'rgb_coarse': results_full['depth_coarse']}, depth_gt)
            loss_depth += loss_depth_patch['tot']
        loss_depth_smooth = inverse_depth_smoothness_loss(
            results_full['depth_fine'], results_full['rgb_fine'])
        loss_depth_smooth += inverse_depth_smoothness_loss(
            results_full['depth_coarse'], results_full['rgb_fine'])
        if batch_idx % 10 == 0 and self.hparams.dataset_name != 'dtu_proj':
            if 'l2' in loss_depth_patch:
                self.logger.experiment.add_scalar(
                    'train/depth_l2', loss_depth_patch['l2'].detach(),  self.global_step)
            if 'ssim' in loss_depth_patch:
                self.logger.experiment.add_scalar(
                    'train/depth_ssim', loss_depth_patch['ssim'].detach(),  self.global_step)

        if 'blender' in self.hparams.dataset_name:
            need_zero = depth_gt.view(-1, 1) == 0
            loss_depth += self.s1(results_full['depth_coarse'].view(-1, 1),
                                  depth_gt.view(-1, 1), mask=need_zero.view(-1, 1)) * 2
            loss_depth += self.s1(results_full['depth_fine'].view(-1, 1),
                                  depth_gt.view(-1, 1), mask=need_zero.view(-1, 1)) * 2
        # only use warp_patch on side views
        if 'warp_patch' in batch:
            # pseudo label from warping
            results_side['depth_fine'] = rearrange(
                results_side['depth_fine'], '(b p q c) -> b c p q', c=1, p=psx, q=psy)
            results_side['depth_coarse'] = rearrange(
                results_side['depth_coarse'], '(b p q c) -> b c p q', c=1, p=psx, q=psy)
            loss_depth_smooth += inverse_depth_smoothness_loss(
                results_side['depth_coarse'], results_side['rgb_fine'])
            loss_depth_smooth += inverse_depth_smoothness_loss(
                results_side['depth_fine'], results_side['rgb_fine'])
            depth_mask = batch['warp_patch_depth'].view(psx, psy) > 0
            if depth_mask.sum() > 0:
                loss_side_depth = self.s1(
                    results_side['depth_coarse'][0][0], batch['warp_patch_depth'].view(psx, psy), depth_mask)
                loss_side_depth += self.s1(
                    results_side['depth_fine'][0][0], batch['warp_patch_depth'].view(psx, psy), depth_mask)
            else:
                loss_side_depth = 0

        if 'rgbs_proj' in batch:
            rand_rgbs_proj = batch['rgbs_proj']
            rand_rgbs_proj = rearrange(rand_rgbs_proj, 'b n c -> (b n) c', c=3)
        gradient_penalty = 0

        if batch_idx % 10 == 0:
            img_coarse = results_full[f'rgb_coarse'][0].detach().cpu()
            img_fine = results_full[f'rgb_fine'][0].detach().cpu()
            img_real = real_patch[0].detach().cpu()  # (3, H, W)
            if self.hparams.dataset_name == 'llff_dual_patch':
                img_fine2 = results[f'rgb_fine'][0].detach().cpu()
                stack = torch.stack(
                    [img_real, img_fine2, img_coarse, img_fine])  # 3, 3, h, w
            else:
                stack = torch.stack(
                    [img_real, img_coarse, img_fine])  # 3, 3, h, w
                # stack = torch.cat([stack, warp], dim=2)
            self.logger.experiment.add_images(
                'train/images', stack, self.global_step)
            if 'ssim' in loss_g:
                self.logger.experiment.add_scalar(
                    'train/ssim', loss_g['ssim'], self.global_step)

            # side
            img_coarse = results_side[f'rgb_coarse'][0].detach().cpu()
            img_fine = results_side[f'rgb_fine'][0].detach().cpu()
            img_real = side_rgb[0].detach().cpu()  # (3, H, W)
            depth_fine = visualize_depth(
                results_side[f'depth_fine'].view(psx, psy).detach())
            depth_coarse = visualize_depth(
                results_side[f'depth_coarse'].view(psx, psy).detach())
            depth_gt = visualize_depth(
                batch['warp_patch_depth'].view(psx, psy).detach())
            stack = torch.stack(
                [img_real, img_coarse, img_fine, depth_coarse, depth_fine, depth_gt])
            self.logger.experiment.add_images(
                'train/images_side', stack, self.global_step)
        if self.hparams.dis_weight > 0:
            if optimizer_idx == 0:
                pred_fake_full = self.D((results_side['rgb_fine']))
                if self.hparams.dloss == 'hinge':
                    loss_d = -torch.mean(pred_fake_full)
                    # loss_d += -torch.mean(pred_fake_full2)
                elif self.hparams.dloss == 'vanilla':
                    loss_d = self.criterionGAN(pred_fake_full, True)
                elif self.hparams.dloss == 'relavistic':
                    pred_real = self.D(DiffAugment(real_patch))
                    loss_d = (self.criterionGAN(pred_real - torch.mean(pred_fake_full), False) +
                              self.criterionGAN(pred_fake_full - torch.mean(pred_real), True)) / 2
                elif self.hparams.dloss == 'wgan':
                    loss_d = self.compute_loss(pred_fake_full, 1)
                elif self.hparams.dloss == 'wgan_gp':
                    loss_d = self.compute_loss(pred_fake_full, 1)

            else:
                # train discriminator

                pred_real = self.D((real_patch))

                pred_fake_full = self.D((results_side['rgb_fine'].detach()))
                if self.hparams.dloss == 'hinge':
                    loss_Dreal = (F.relu(torch.ones_like(
                        pred_real) - pred_real)).mean()
                    loss_Dgen = (F.relu(torch.ones_like(
                        pred_fake_full) + pred_fake_full)).mean()
                    loss_d = (loss_Dreal + loss_Dgen) / 2
                elif self.hparams.dloss == 'relavistic':
                    loss_d = (self.criterionGAN(pred_real - torch.mean(pred_fake_full), True) +
                              self.criterionGAN(pred_fake_full - torch.mean(pred_real), False)) / 2
                elif self.hparams.dloss == 'vanilla':
                    loss_d = (self.criterionGAN(pred_real, True) +
                              self.criterionGAN(pred_fake_full, False)) / 2
                elif self.hparams.dloss == 'wgan':
                    loss_d = self.compute_loss(
                        pred_fake_full, 0) + self.compute_loss(pred_real, 1)
                elif self.hparams.dloss == 'wgan_gp':
                    loss_d = self.compute_loss(
                        pred_fake_full, 0) + self.compute_loss(pred_real, 1)
                    loss_d += 10 * \
                        self.compute_grad2(pred_real, real_patch).mean()
        else:
            # disable discriminator
            loss_d = 0

        if self.hparams.load_depth:
            if self.hparams.depth_anneal:
                # linear decay to 0 at 2000 epoch
                dw = max(self.hparams.depth_weight - self.current_epoch /
                         (500 / self.hparams.depth_weight), 1)
            else:
                dw = self.hparams.depth_weight
            loss = loss_g['tot'] + loss_d * self.hparams.dis_weight + \
                loss_depth * dw
        else:
            raise NotImplementedError

        loss += self.hparams.proj_weight * \
            (self.hparams.depth_weight * loss_side_depth)

        loss += self.hparams.vit_weight * loss_vit

        loss += loss_depth_smooth * self.hparams.depth_smooth_weight

        log['train/loss'] = loss
        log['train/loss_g'] = loss_g
        log['train/loss_vit'] = loss_vit
        log['train/loss_d'] = loss_d
        log['train/loss_depth_smooth'] = loss_depth_smooth
        log['train/loss_side_depth'] = loss_side_depth

        if gradient_penalty != 0:
            log['train/gradient_penalty'] = gradient_penalty
        if self.hparams.load_depth:
            log['train/loss_depth'] = loss_depth

        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        pb = {
            'train_psnr': psnr_,
            'g': loss_g['tot'],
        }
        if self.hparams.vit_weight > 0:
            pb['vit'] = loss_vit
        if self.hparams.dis_weight > 0:
            pb['d'] = loss_d
        if self.hparams.depth_smooth_weight > 0:
            pb['smooth'] = loss_depth_smooth
        if gradient_penalty != 0:
            pb['gradient_penalty'] = gradient_penalty
        if loss_depth != 0:
            pb['depth'] = loss_depth
        if loss_side_depth != 0:
            pb['side_depth'] = loss_side_depth
        # for k, v in loss_g.items():
        #     if k == 'tot':
        #         continue
        #     log[f'train/loss_{k}'] = v.item()
        #     pb[f'loss_{k}'] = v.item()

        return {'loss': loss,
                'progress_bar': pb,
                'log': log
                }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        log = {}
        # log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb % 5 == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(
                results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        # mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        # return {'progress_bar': {'val_loss': mean_loss,
        return {'progress_bar': {'val_psnr': mean_psnr},
                'log': {'val/psnr': mean_psnr}
                }
        # 'log': {'val/loss': mean_loss,
