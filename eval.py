import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
# from utils.visualization import visualize_depth
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

import cv2
from PIL import Image


def visualize_depth(x, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    # x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    # x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff',
                                 'blender_ray_patch_1image', 'dtu_proj'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--angle', type=int, default=64)
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--timestamp', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--depth_type', type=str,
                        default='nerf')  # depth supervision
    parser.add_argument('--save_depth', default=False, action="store_true")
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes', 'npy', 'png'],
                        help='which format to save')
    parser.add_argument('--model', type=str, default="nerf",
                        choices=['nerf', 'nerf_ft'])

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    # chunk = 1024*32 * 8
    chunk = 1024*32 * 16
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        # render_rays_hog(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=False)
        # test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    if args.timestamp == "":
        args.timestamp = args.ckpt_path.split('/')[1]
        print("[timestamp auto set]", args.timestamp)
    w, h = args.img_wh

    # kwargs = {'root_dir': args.root_dir,
    #           'split': args.split,
    #           'img_wh': tuple(args.img_wh)}
    # if args.dataset_name == 'llff':
    #     kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**vars(args))
    dic = torch.load(args.ckpt_path)
    # print(list(dic['callbacks'].values())[0]['best_model_score'])

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF(use_new_activation=True)
    nerf_fine = NeRF(use_new_activation=True)

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}/{args.timestamp}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        # for i in li:
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()

        if 'fname' in sample:
            fname = os.path.basename(sample['fname']).replace('.JPG', '')
        else:
            fname = f'{i:03d}'

        if args.save_depth:
            depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
            depth_pred = np.nan_to_num(depth_pred)
            # print(depth_pred.shape) # 378, 504
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(
                    dir_name, f'depth_{fname}.pfm'), depth_pred)
            elif args.depth_format == 'pfm':
                np.save(os.path.join(dir_name, f'{fname}.npy'), depth_pred)
            else:
                # with open(f'depth_{fname}', 'wb') as f:
                    # f.write(depth_pred.tobytes())
                visualize_depth(depth_pred).save(
                    os.path.join(dir_name, f'{fname}_depth.png'))

        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{fname}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

    imageio.mimsave(os.path.join(
        dir_name, f'{args.scene_name}.gif'), imgs, fps=5)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
