import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
import tqdm
from torchvision import transforms as T

from .ray_utils import *


import cv2


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    # x = depth.cpu().numpy()
    x = depth
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    return x_


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rot_z(th): return torch.Tensor([
    [np.cos(th), -np.sin(th), 0, 0],
    [np.sin(th), np.cos(th), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(c2w, theta, phi):
    # c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi)  # @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def rotate(angle):
    return rot_z(angle/180.*np.pi)
    # return rot_theta(angle/180.*np.pi)
    # return rot_phi(angle/180.*np.pi)


def flatten(pose):
    if pose.shape[0] != 4:
        pose = torch.cat([pose, torch.Tensor([[0, 0, 0, 1]])], dim=0)
    return torch.inverse(pose)[:3, :4]


def rotate_3d(c2w, x, y, z):
    rot = rot_phi(x/180.*np.pi) @ rot_theta(y/180.*np.pi) @ rot_z(z/180.*np.pi)
    return rot @ c2w


def convert(c2w):
    # return np.linalg.inv(c2w)
    R, T = c2w[:3, :3], c2w[:3, 3:]
    ww = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])
    #  [0, 0, 0, 1]])
    R_ = R.T
    T_ = -1 * R_ @ T
    R_ = ww @ R_
    T_ = ww @ T_
    # print(R_.shape, T_.shape)
    new = np.concatenate((R_, T_), axis=1)
    # new = torch.inverse(torch.from_numpy(ww @ c2w).float())
    new = np.concatenate((new, np.array([[0, 0, 0, 1]])), axis=0)
    return new


def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                                   torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), torch.stack(
        (x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * (depth_ref.view(batchsize, -1).unsqueeze(1)))

    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                           torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:, :3, :]
    # print(xyz_src.shape)  B*3*20480

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480
    depth_src = K_xyz_src[:, 2:3, :]
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:,2:3,:] + 1e-9)
    x_src = xy_src[:, 0, :].view([batchsize, height, width])
    y_src = xy_src[:, 1, :].view([batchsize, height, width])
    # print(x_src.shape) #B*128*160

    return x_src, y_src, depth_src
# (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')


def forward_warp(data, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]
    data = data[0].permute(1, 2, 0)
    new = np.zeros_like(data)
    depth_src = depth_src.reshape(height, width)
    new_depth = np.zeros_like(depth_src)
    yy_base, xx_base = torch.meshgrid([torch.arange(
        0, height, dtype=torch.long), torch.arange(0, width, dtype=torch.long)])
    y_res = np.clip(np.floor(y_res.numpy()), 0, width - 1)  # .astype(np.int64)
    x_res = np.clip(np.floor(x_res.numpy()), 0,
                    height - 1)  # .astype(np.int64)
    # print(x_res.min(), x_res.max(), y_res.min(), y_res.max())
    # be careful with nan
    x_res = x_res.astype(np.int64)
    y_res = y_res.astype(np.int64)
    new[y_res, x_res] = data[yy_base, xx_base]
    new_depth[y_res, x_res] = depth_src[yy_base, xx_base]
    return new, new_depth


class Blender_ray_patch_1image_rot3d_Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), patch_size=-1, factor=1, test_crop=False, with_ref=False, repeat=1, load_depth=False, depth_type='nerf', sH=1, sW=1, angle=30, **kwargs):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.factor = factor
        self.angle = angle
        self.with_ref = with_ref
        self.test_crop = test_crop
        self.repeat = repeat
        self.depth_type = depth_type
        self.load_depth = load_depth
        self.patch_size = patch_size
        self.sH = sH
        self.sW = sW
        if 'lego' in self.root_dir or 'hotdog' in self.root_dir:
            self.my_test = True
        else:
            self.my_test = False

        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        if self.split in ['test_train', 'test_train2']:
            json_name = f"transforms_train.json"
        elif self.split == 'val':
            if self.my_test:
                json_name = f"transforms_mytest.json"
            else:
                json_name = f"transforms_train.json"
            # json_name = f"transforms_test.json"
        else:
            json_name = f"transforms_{self.split}.json"
        with open(os.path.join(self.root_dir,
                               json_name), 'r') as f:
            self.meta = json.load(f)

        if self.split == 'val' and self.my_test:
            # json_name = f"transforms_mytest.json"
            self.meta['frames'] = self.meta['frames'][30 -
                                                      self.angle:30 + self.angle]

        w, h = self.img_wh
        # original focal length
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x'])
        # when W=800

        # modify focal length to match size self.img_wh
        self.focal *= self.img_wh[0]/800
        K = np.array([[self.focal, 0, (400 - 1) / 2],
                     [0, self.focal, (400 - 1) / 2], [0, 0, 1]])
        self.K = torch.from_numpy(K).float()
        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions_small = \
            get_ray_directions(int(self.img_wh[1] // self.factor), int(
                self.img_wh[0] // self.factor), self.focal)  # (H, W, 3)
        # if self.center_crop:
        self.directions = \
            get_ray_directions(
                self.img_wh[1], self.img_wh[0], self.focal)  # (h, w, 3)

        if 'lego' in self.root_dir:
            self.ref_idx = 20
        elif 'chair' in self.root_dir:
            # self.ref_idx = 31
            self.ref_idx = 99
        elif 'ship' in self.root_dir:
            self.ref_idx = 80
        elif 'hotdog' in self.root_dir:
            self.ref_idx = 3
        elif 'mic' in self.root_dir:
            self.ref_idx = 15
        elif 'ficus' in self.root_dir:
            self.ref_idx = 22
        elif 'drums' in self.root_dir:
            self.ref_idx = 19
        else:
            raise NotImplementedError
        # self.val_idx = 24 #on val set
        # self.val_idx = 59 # on test set
        if self.depth_type == 'gt':
            print('using gt depth')
            json_name = f"transforms_mytest.json"
            with open(os.path.join(self.root_dir,
                                   json_name), 'r') as f:
                self.meta = json.load(f)
            if 'lego' in self.root_dir or 'hotdog' in self.root_dir:
                self.ref_idx = 29  # blender/r_58
            else:
                raise NotImplementedError

        if self.split == 'train':  # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.poses_fake = []
            # self.all_rays_full = []
            self.imgs_2d = []
            self.ref_view = None
            for idx, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(
                    self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                # idx == 0: first image also provide full resolution
                if self.with_ref and idx == self.ref_idx:
                    first_train = True
                    self.ref_c2w = torch.from_numpy(
                        np.array(frame['transform_matrix'])).float()
                    self.poses_real = flatten(self.ref_c2w)
                    # print(self.poses_real.shape)
                else:
                    first_train = False
                h, w = img.size
                if not first_train:
                    img = img.crop((int((h - self.img_wh[0] // self.factor) // 2), int((w - self.img_wh[1] // self.factor) // 2), int(
                        (h + self.img_wh[0] // self.factor) // 2), int((w + self.img_wh[1] // self.factor) // 2)))
                    hh, ww = img.size

                img = self.transform(img)  # (4, h, w)
                img = img[:3, :, :] * img[-1:, :, :] + (1 - img[-1:, :, :])

                if first_train:
                    self.imgs_2d += [img] * self.repeat

                if first_train and self.ref_view is None:
                    # print(img.shape) # 3, 378, 504
                    self.ref_view = img.permute(1, 2, 0)
                # img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                if first_train:
                    self.all_rgbs += [img] * \
                        (self.repeat if first_train else 0)
                if first_train:
                    rays_o, rays_d = get_rays(
                        self.directions, c2w)  # both (h*w, 3)
                else:
                    rays_o, rays_d = get_rays(
                        self.directions_small, c2w)  # both (h*w, 3)
                rays_ = torch.cat([rays_o, rays_d,
                                   self.near*torch.ones_like(rays_o[:, :1]),
                                   self.far*torch.ones_like(rays_o[:, :1])],
                                  1)  # (h*w, 8)

                rays_o_full, rays_d_full = get_rays(self.directions, c2w)
                if first_train:
                    self.all_rays += [rays_] * \
                        (self.repeat if first_train else 0)
                rays_full = torch.cat([rays_o_full, rays_d_full, self.near * torch.ones_like(
                    rays_o_full[:, :1]), self.far * torch.ones_like(rays_o_full[:, :1])], 1)  # h * w, 8
                rays_full = rays_full.view(w, h, 8)
                # print('ray_full', h, w) # 504 378
                rays_full = rays_full.view(w, h, 8)
                if first_train:
                    self.ref_rays = rays_full

            self.all_rays = torch.cat(self.all_rays, 0)
            # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.nonzero_rays = self.all_rays[self.all_rgbs.sum(dim=-1) != 3]
            self.nonzero_rgbs = self.all_rgbs[self.all_rgbs.sum(dim=-1) != 3]

            # self.len_full = len(self.all_rays_full)
            self.rgb_num = len(self.nonzero_rgbs)
            self.all_rgb_num = len(self.all_rgbs)
            self.img_num = len(self.imgs_2d)

            if self.load_depth:
                if self.depth_type == 'nerf':
                    self.ref_depth = np.load(os.path.join(self.root_dir, 'depth_nerf', os.path.basename(
                        self.image_paths[self.ref_idx])).replace('.JPG', '.npy').replace('.png', '.npy'))
                    self.ref_depth = torch.from_numpy(
                        self.ref_depth).float()  # (378, 504)
                elif self.depth_type == 'gt':
                    self.ref_depth = torch.from_numpy(np.load(os.path.join(self.root_dir, 'my_testset', os.path.basename(
                        self.image_paths[self.ref_idx])).replace('.png', '_400.npy').replace('.JPG', '_400.npy'))).float()

                    self.ref_depth[self.ref_depth > 1000] = 0
                    self.ref_depth = self.ref_depth[:, :, 0]
                else:
                    self.ref_depth = np.load(os.path.join(self.root_dir, 'depth', os.path.basename(
                        self.image_paths[self.ref_idx])).replace('.JPG', '.JPG.npy').replace('.png', '.npy'))
                    self.ref_depth = torch.from_numpy(
                        self.ref_depth).float()  # (378, 504)
                    # ref image is flower/images/IMG_2981.JPG
                    # NOTE: need check carefully whether already divided by scale_factor
                self.all_depth = self.ref_depth.reshape(-1, 1)
                self.nonzero_depth = self.all_depth[self.all_rgbs.sum(
                    dim=-1) != 3]
                # reshape depth
                # self.ref_depth = self.ref_depth.reshape(-1, 1)
            print(self.all_rays.shape, self.all_rgbs.shape, self.all_depth.shape,
                  self.nonzero_rays.shape, self.nonzero_rgbs.shape, self.nonzero_depth.shape)

            test_c2w = []
            for x in range(-self.angle, self.angle + 1, self.angle // 2):
                for y in range(-self.angle, self.angle + 1, self.angle // 2):
                    for z in range(-self.angle, self.angle + 1, self.angle // 2):
                        test_c2w.append(
                            rotate_3d(self.ref_c2w, x, y, z)[:3, :4])

            self.all_rays_full = []
            self.all_rgbs_full = []
            self.all_depths_full = []
            self.proj_rays_full = []
            self.proj_rgbs_full = []
            self.proj_depths_full = []
            self.proj_mat = []
            self.ref_proj_mat = torch.FloatTensor(
                convert(self.ref_c2w.numpy())).clone()
            self.ref_proj_mat[:3, :4] = torch.matmul(
                self.K, self.ref_proj_mat[:3, :4])
            reffff = self.ref_view.permute((2, 0, 1)).unsqueeze(0)
            refff_depth = self.ref_depth.unsqueeze(0)
            refff_c2w = torch.FloatTensor(convert(self.ref_c2w.numpy()))
            idx = 0
            for c2w in tqdm.tqdm(test_c2w):
                c2w = torch.FloatTensor(c2w)
                rays_o, rays_d = get_rays(self.directions, c2w)
                rays_full = torch.cat([rays_o, rays_d,  self.near*torch.ones_like(
                    rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], 1)  # (h*w, 8)
                rays_full = rays_full.view(w, h, 8)
                self.poses_fake.append(flatten(c2w))
                # self.all_rays_full.append(rays_full)
                proj_mat = torch.FloatTensor(convert(c2w.numpy()))
                out, depth = forward_warp(
                    reffff, refff_depth, self.K, refff_c2w, self.K, proj_mat)
                proj_mat[:3, :4] = torch.matmul(self.K, proj_mat[:3, :4])
                idx += 1
                self.proj_mat.append(proj_mat)
                out = torch.FloatTensor(out).view(-1, 3)
                mask = out.sum(-1) != 0  # remove cant warped part
                # still has white area here
                self.proj_rgbs_full.append(out[mask])
                self.proj_depths_full.append(
                    torch.FloatTensor(depth).view(-1, 1)[mask])
                self.proj_rays_full.append(rays_full.view(-1, 8)[mask])
            self.proj_rays_full = torch.cat(self.proj_rays_full, 0)
            self.proj_rgbs_full = torch.cat(self.proj_rgbs_full, 0)
            self.proj_depths_full = torch.cat(self.proj_depths_full, 0)
            # """
            self.len_full = len(test_c2w)

        elif self.split == 'test_train2':
            pose = np.array(self.meta['frames'][self.ref_idx]
                            ['transform_matrix'])  # [:3, :4]
            c2w = torch.FloatTensor(pose)
            self.poses_test = [(rotate(angle) @ c2w)[:3, :4]
                               for angle in np.linspace(-self.angle, self.angle, 30)]
            for x in self.poses_test:
                print(x[:3, 3])

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            # return self.all_rays.shape[0]
            self.len = max(self.len_full, self.img_num)
            if self.load_depth:
                self.depth_sz = self.ref_depth.shape[0]  # // self.len
            return self.len
            # return len(self.all_rays)
        if self.split == 'val':
            if self.my_test:
                return len(self.meta['frames'])
            return 1  # only validate 8 images (to support <=8 gpus)
        if self.split == 'test_train2':
            self.len = len(self.poses_test)
            return self.len
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers

            # random select real sample
            new_idx = np.random.randint(0, self.img_num)
            im = self.imgs_2d[new_idx]
            _, w, h = im.shape
            if w > self.patch_size:
                while True:
                    ll = np.random.randint(
                        0, w - (self.patch_size - 1) * self.sW - 1)
                    up = np.random.randint(
                        0, h - (self.patch_size - 1) * self.sH - 1)
                    patch = im[:, ll:ll+(self.patch_size - 1) * self.sW + 1:self.sW,
                               up:up+(self.patch_size - 1) * self.sH + 1:self.sH]

                    if patch.max() != 0:
                        break
            else:
                patch = im

            num = 4096
            ray_idx = np.random.choice(self.all_rgb_num, num // 10)
            ray_idx2 = np.random.choice(self.rgb_num, num - num // 10)
            # lower -> avoid too much exceeding 3sigma
            x = np.random.normal(0, self.angle // 2)
            # lower -> avoid too much exceeding 3sigma
            y = np.random.normal(0, self.angle // 2)
            # lower -> avoid too much exceeding 3sigma
            z = np.random.normal(0, self.angle // 2)
            c2w = rotate_3d(self.ref_c2w, x, y, z)

            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_full = torch.cat([rays_o, rays_d,  self.near*torch.ones_like(
                rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], 1)  # (h*w, 8)
            rays_full = rays_full.view(w, h, 8)

            out, depth = forward_warp(self.ref_view.permute((2, 0, 1)).unsqueeze(0), self.ref_depth.unsqueeze(
                0), self.K, torch.FloatTensor(convert(self.ref_c2w.numpy())), self.K, torch.FloatTensor(convert(c2w.numpy())))
            out = torch.FloatTensor(out).contiguous()
            depth = torch.FloatTensor(depth).contiguous()

            while True:
                ll = np.random.randint(
                    0, w - (self.patch_size - 1) * self.sW - 1)
                up = np.random.randint(
                    0, h - (self.patch_size - 1) * self.sH - 1)
                warp_patch_depth = depth[ll:ll+(self.patch_size - 1) * self.sW + 1:self.sW, up:up+(
                    self.patch_size - 1) * self.sH + 1:self.sH]
                if warp_patch_depth.sum() != 0:  # only if not all holes
                    warp_patch = out[ll:ll+(self.patch_size - 1) * self.sW + 1:self.sW, up:up+(
                        self.patch_size - 1) * self.sH + 1:self.sH, :].permute(2, 0, 1)
                    fake_patch = rays_full[ll:ll+(self.patch_size - 1) * self.sW + 1:self.sW, up:up+(
                        self.patch_size - 1) * self.sH + 1:self.sH, :]
                    fake_patch = fake_patch.reshape(-1, 8)
                    break

            ray_idx_proj = np.random.choice(
                (self.proj_depths_full).shape[0], num)
            sample = {'rays': torch.cat([self.nonzero_rays[ray_idx2], self.all_rays[ray_idx]]),
                      'rgbs': torch.cat([self.nonzero_rgbs[ray_idx2], self.all_rgbs[ray_idx]]),
                      'depth': torch.cat([self.nonzero_depth[ray_idx2], self.all_depth[ray_idx]]),
                      'rays_proj': self.proj_rays_full[ray_idx_proj],
                      'depth_proj': self.proj_depths_full[ray_idx_proj],
                      'real_patch': patch,
                      'rays_full': fake_patch,
                      'warp_patch': warp_patch,
                      'warp_patch_depth': warp_patch_depth,
                      'side_proj': self.proj_mat[idx],
                      'ref_proj': self.ref_proj_mat,
                      'ref_depth_full': self.ref_depth,
                      'side_coord': torch.stack([torch.arange(0, self.patch_size) * self.sW + ll, torch.arange(0, self.patch_size) * self.sH + up])
                      }

            sample['pose_real'] = self.poses_real
            sample['pose_fake'] = self.poses_fake[idx %
                                                  self.len_full]  # same as patch data

            if self.load_depth:
                sample['depth_ray'] = self.ref_rays[ll:ll+(self.patch_size - 1) * self.sW + 1:self.sW, up:up+(
                    self.patch_size - 1) * self.sH + 1:self.sH, :].reshape(-1, 8)
                sample['depth_gt'] = self.ref_depth[ll:ll+(self.patch_size - 1) * self.sW + 1:self.sW, up:up+(
                    self.patch_size - 1) * self.sH + 1:self.sH].reshape(-1, 1)
                sample['depth_ray_rgb'] = self.ref_view[ll:ll+(self.patch_size - 1) * self.sW + 1:self.sW, up:up+(
                    self.patch_size - 1) * self.sH + 1:self.sH, :].reshape(-1, 3)
        else:  # create data for each image separately
            if self.split == 'val':
                if self.my_test:
                    frame = self.meta['frames'][idx]
                else:
                    frame = self.meta['frames'][self.ref_idx]
                    # frame = self.meta['frames'][self.val_idx]

            if self.split == 'test_train2':
                c2w = self.poses_test[idx]
            else:
                c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(
                self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            if self.test_crop:
                h, w = img.size
                img = img.crop((int((h - self.img_wh[0] // self.factor) // 2), int((w - self.img_wh[1] // self.factor) // 2), int(
                    (h + self.img_wh[0] // self.factor) // 2), int((w + self.img_wh[1] // self.factor) // 2)))
            img = self.transform(img)  # (4, H, W)
            # (H*W) valid color area # not used
            valid_mask = (img[-1] > 0).flatten()
            img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:])  # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)
            if self.test_crop:
                rays_o, rays_d = get_rays(self.directions_small, c2w)

            rays = torch.cat([rays_o, rays_d,
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                             1)  # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}
            if self.split.endswith('train'):
                sample['fname'] = frame['file_path']

        return sample


if __name__ == "__main__":
    dataset = Blender_ray_patch_1image_rot3d_Dataset('../../nerf_synthetic/lego', split='test_train2',
                                                     load_depth=True, depth_type='nerf', sH=2, sW=2, patch_size=180, img_wh=(400, 400), with_ref=True)

    item = dataset[0]
