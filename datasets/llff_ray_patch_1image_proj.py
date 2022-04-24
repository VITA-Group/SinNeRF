import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
import tqdm
from torchvision import transforms as T

from .ray_utils import *
from einops import rearrange

import cv2


def assign_last(a, index, b):
    """a[index] = b
    """
    index = index[::-1]
    b = b[::-1]

    ix_unique, ix_first = np.unique(index, return_index=True)
    # np.unique: return index of first occurrence.
    # ix_unique = index[ix_first]

    a[ix_unique] = b[ix_first]
    return a


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

# llff uses right up back
# opencv: right, down, forward


def convert(c2w, scale_factor=1):
    # return np.linalg.inv(c2w)
    R, T = c2w[:3, :3], c2w[:3, 3:]
    # T *= scale_factor
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

    pts = torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(
        0) * (depth_ref.view(batchsize, -1).unsqueeze(1))

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), pts)

    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                           torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:, :3, :]

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480
    depth_src = K_xyz_src[:, 2:3, :]
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:, 2:3, :] + 1e-9)
    x_src = xy_src[:, 0, :].view([batchsize, height, width])
    y_src = xy_src[:, 1, :].view([batchsize, height, width])

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
        0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long)])
    y_res = np.clip(y_res.numpy(), 0, height - 1).astype(np.int64)
    x_res = np.clip(x_res.numpy(), 0, width - 1).astype(np.int64)
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    y_res = y_res.reshape(-1)
    x_res = x_res.reshape(-1)
    # painter's algo
    for i in range(yy_base.shape[0]):
        if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
            new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
            new[y_res[i], x_res[i]] = data[yy_base[i], xx_base[i]]
    return new, new_depth


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    # convert to homogeneous coordinate for faster computation
    pose_avg_homo[:3] = pose_avg
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]),
                       (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row],
                       1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(
        pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        def trans_t(t): return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9*t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        def rot_phi(phi): return np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        def rot_theta(th): return np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],
                       [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        # 36 degree view downwards
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)]
    return np.stack(spheric_poses, 0)


class LLFF_ray_patch_1image_proj_Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378), spheric_poses=False, val_num=1, patch_size=-1, factor=1, test_crop=False, with_ref=False, repeat=1, load_depth=False, depth_type='nerf', sH=1, sW=1, patch_size_x=-1, patch_size_y=-1, **kwargs):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num)  # at least 1

        # extra param
        self.factor = factor
        self.with_ref = with_ref
        self.test_crop = test_crop
        self.repeat = repeat
        self.depth_type = depth_type
        self.load_depth = load_depth
        # self.patch_size = patch_size
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.sH = sH
        self.sW = sW
        self.ndc = False
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(
            glob.glob(os.path.join(self.root_dir, 'images/*.JPG')))
        # load full resolution image then resize
        if self.split in ['train', 'val']:
            # print(len(poses_bounds), len(self.image_paths))
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        # original intrinsics, same for all images
        H, W, self.focal = poses[0, :, -1]
        assert H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'

        self.focal *= self.img_wh[0]/W

        K = np.array([[self.focal, 0, (self.img_wh[1] - 1) / 2],
                     [0, self.focal, (self.img_wh[0] - 1) / 2], [0, 0, 1]])
        self.K = torch.from_numpy(K).float()

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate(
            [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # choose val image as the closest to
        val_idx = np.argmin(distances_from_center)
        # center image
        ref_idx = val_idx - 1
        print('Val image is: ', val_idx, self.image_paths[val_idx])
        print('Ref image is: ', ref_idx, self.image_paths[ref_idx])

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        self.near = near_original * 0.9 / scale_factor
        self.far = self.bounds.max() * 1
        print('[near far]', self.near, self.far, scale_factor)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions_small = \
            get_ray_directions(int(self.img_wh[1] // self.factor), int(
                self.img_wh[0] // self.factor), self.focal)  # (H, W, 3)
        self.directions = \
            get_ray_directions(
                self.img_wh[1], self.img_wh[0], self.focal)  # (H, W, 3)

        if self.split == 'train':  # create buffer of all rays and rgb data
            if self.load_depth:
                if self.depth_type == 'nerf':
                    self.ref_depth = np.load(os.path.join(self.root_dir, 'depth_nerf', os.path.basename(
                        self.image_paths[ref_idx])).replace('.JPG', '.npy'))
                    self.ref_depth = torch.from_numpy(
                        self.ref_depth).float()  # (378, 504)
                    print(self.ref_depth.min(), self.ref_depth.max())
                else:
                    self.ref_depth = np.load(os.path.join(self.root_dir, 'depth', os.path.basename(
                        self.image_paths[ref_idx])).replace('.JPG', '.JPG.npy'))
                    self.ref_depth = torch.from_numpy(
                        self.ref_depth).float()  # (378, 504)
                    # ref image is flower/images/IMG_2981.JPG
                    # NOTE: need check carefully whether already divided by scale_factor

            self.all_rays = []
            self.all_rgbs = []
            # self.all_rays_full = []
            self.imgs_2d = []
            self.ref_view = None
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx:  # exclude the val image
                    continue
                if self.with_ref and i == ref_idx:  # new ref image
                    first_train = True
                else:
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                # print('ori_size', img.size) # 4032, 3024
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                h, w = img.size
                if not first_train:
                    img = img.crop((int((h - self.img_wh[0] // self.factor) // 2), int((w - self.img_wh[1] // self.factor) // 2), int(
                        (h + self.img_wh[0] // self.factor) // 2), int((w + self.img_wh[1] // self.factor) // 2)))
                    hh, ww = img.size
                img = self.transform(img)  # (3, h, w)
                if first_train:
                    self.imgs_2d += [img] * self.repeat
                    self.ref_c2w = c2w
                if first_train and self.ref_view is None:
                    # print(img.shape) # 3, 378, 504
                    self.ref_view = img.permute(1, 2, 0)
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
                rays_o_full, rays_d_full = get_rays(self.directions, c2w)
                if self.ndc:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                    # near plane is always at 1.0
                    # near and far in NDC are always 0 and 1
                    # See https://github.com/bmild/nerf/issues/34
                    rays_o_full, rays_d_full = get_ndc_rays(
                        self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o_full, rays_d_full)
                elif self.spheric_poses:
                    # won't happen, we don't use spheric poses for LLFF
                    near = self.bounds.min()
                    # focus on central object only
                    far = min(8 * near, self.bounds.max())
                else:
                    # DSNeRF
                    near = self.near
                    far = self.far

                rays_ = torch.cat([rays_o, rays_d,
                                   near*torch.ones_like(rays_o[:, :1]),
                                   far*torch.ones_like(rays_o[:, :1])],
                                  1)  # (h*w, 8)
                if first_train:
                    self.all_rays += [rays_] * \
                        (self.repeat if first_train else 0)

                rays_full = torch.cat([rays_o_full, rays_d_full, near * torch.ones_like(
                    rays_o_full[:, :1]), far * torch.ones_like(rays_o_full[:, :1])], 1)  # h * w, 8
                # print('ray_full', h, w) # 504 378
                rays_full = rays_full.view(w, h, 8)
                if first_train:
                    self.ref_rays = rays_full

                first_train = False

            # ((N_images-1)*h*w, 8)
            self.all_rays = torch.cat(self.all_rays, 0)
            # ((N_images-1)*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.nonzero_rgbs = self.all_rgbs[self.all_rgbs != 0]
            self.rgb_num = len(self.all_rgbs)
            self.img_num = len(self.imgs_2d)

            focus_depth = 3.5  # hardcoded, this is numerically close to the formula
            # given in the original repo. Mathematically if near=1
            # and far=infinity, then this number will converge to 4
            radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
            test_c2w = self.poses  # [:3]
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

            idx = 0
            reffff = self.ref_view.permute((2, 0, 1)).unsqueeze(0)
            refff_depth = self.ref_depth.unsqueeze(0)
            refff_c2w = torch.FloatTensor(convert(self.ref_c2w.numpy()))
            for c2w in tqdm.tqdm(test_c2w):
                c2w = torch.FloatTensor(c2w)
                rays_o, rays_d = get_rays(self.directions, c2w)
                if self.ndc:
                    rays_o, rays_d = get_ndc_rays(
                        self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d)
                # else:

                rays_full = torch.cat([rays_o, rays_d,  self.near*torch.ones_like(
                    rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], 1)  # (h*w, 8)
                rays_full = rays_full.view(w, h, 8)
                self.all_rays_full.append(rays_full)
                out, depth = forward_warp(
                    reffff, refff_depth, self.K, refff_c2w, self.K, torch.FloatTensor(convert(c2w.numpy())))
                im_gt = Image.open(self.image_paths[idx])
                im_gt = im_gt.resize(self.img_wh, Image.LANCZOS)
                idx += 1
                proj_mat = torch.FloatTensor(convert(c2w.numpy())).clone()
                proj_mat[:3, :4] = torch.matmul(self.K, proj_mat[:3, :4])
                self.proj_mat.append(proj_mat)
                self.all_rgbs_full.append(torch.FloatTensor(out).view(w, h, 3))
                self.all_depths_full.append(
                    torch.FloatTensor(depth).view(w, h, 1))
                out = torch.FloatTensor(out).view(-1, 3)
                mask = out.sum(-1) != 0  # remove cant warped part
                # still has white area here
                self.proj_rgbs_full.append(out[mask])
                # print(depth.min(), depth.max())
                self.proj_depths_full.append(
                    torch.FloatTensor(depth).view(-1, 1)[mask])
                self.proj_rays_full.append(rays_full.view(-1, 8)[mask])
            self.all_rays_full = torch.stack(self.all_rays_full, 0)
            self.all_rgbs_full = torch.stack(self.all_rgbs_full, 0)
            self.all_depths_full = torch.stack(self.all_depths_full, 0)
            self.proj_rays_full = torch.cat(self.proj_rays_full, 0)
            self.proj_rgbs_full = torch.cat(self.proj_rgbs_full, 0)
            self.proj_depths_full = torch.cat(self.proj_depths_full, 0)
            self.len_full = (self.all_rays_full.shape[0])

            self.all_depth = []
            self.all_depth.append(self.ref_depth.view(-1, 1))

            self.all_depth = torch.cat(
                self.all_depth, 0)  # ((N_images-1)*h*w, 3)

            print(self.all_rgbs_full.shape, self.all_rays_full.shape,
                  self.all_depths_full.shape)

        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]

        else:  # for testing, create a parametric rendering path
            if self.split.endswith('train'):  # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5  # hardcoded, this is numerically close to the formula
                # given in the original repo. Mathematically if near=1
                # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            self.len = max(self.len_full, self.img_num)
            if self.load_depth:
                self.depth_sz = self.ref_depth.shape[0]  # // self.len
            return self.len
        if self.split == 'val':
            return len(self.image_paths)
            # return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            # random select real sample
            new_idx = np.random.randint(0, self.img_num)
            im = self.imgs_2d[new_idx]
            _, w, h = im.shape
            ll = np.random.randint(
                0, w - (self.patch_size_x - 1) * self.sW - 1)
            up = np.random.randint(
                0, h - (self.patch_size_y - 1) * self.sH - 1)
            patch = im[:, ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW,
                       up:up+(self.patch_size_y - 1) * self.sH + 1:self.sH]

            ray = self.all_rays_full[idx % self.len_full]
            rgb = self.all_rgbs_full[idx % self.len_full]
            depth = self.all_depths_full[idx % self.len_full]
            w, h, _ = ray.shape
            ll = np.random.randint(
                0, w - (self.patch_size_x - 1) * self.sW - 1)
            up = np.random.randint(
                0, h - (self.patch_size_y - 1) * self.sH - 1)
            fake_patch = ray[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW,
                             up:up+(self.patch_size_y - 1) * self.sH + 1:self.sH, :]
            warp_patch = rgb[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                self.patch_size_y - 1) * self.sH + 1:self.sH, :].permute(2, 0, 1)  # [3, ps, ps]
            warp_patch_depth = depth[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                self.patch_size_y - 1) * self.sH + 1:self.sH, :]
            fake_patch = fake_patch.reshape(-1, 8)
            num = 4096
            ray_idx = np.random.choice(self.rgb_num, num)
            ray_idx_proj = np.random.choice(
                (self.proj_depths_full).shape[0], num)
            sample = {'rays': self.all_rays[ray_idx],
                      'rgbs': self.all_rgbs[ray_idx],
                      'depth': self.all_depth[ray_idx],
                      'rays_proj': self.proj_rays_full[ray_idx_proj],
                      'depth_proj': self.proj_depths_full[ray_idx_proj],
                      'real_patch': patch,
                      'warp_patch': warp_patch,
                      'warp_patch_depth': warp_patch_depth,
                      'side_proj': self.proj_mat[idx],
                      'ref_proj': self.ref_proj_mat,
                      'rays_full': fake_patch}

            if self.load_depth:
                sample['depth_ray'] = self.ref_rays[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                    self.patch_size_y - 1) * self.sH + 1:self.sH, :].reshape(-1, 8)
                sample['depth_gt'] = self.ref_depth[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                    self.patch_size_y - 1) * self.sH + 1:self.sH].reshape(-1, 1)
                sample['depth_ray_rgb'] = self.ref_view[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                    self.patch_size_y - 1) * self.sH + 1:self.sH, :].reshape(-1, 3)
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[idx])
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if self.test_crop:
                rays_o, rays_d = get_rays(self.directions_small, c2w)
            if self.ndc:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            elif self.spheric_poses:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())
            else:
                # DSNeRF
                near = self.near
                far = self.far

            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                             1)  # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_paths[idx]).convert('RGB')
                # img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                if self.test_crop:
                    h, w = img.size
                    img = img.crop((int((h - self.img_wh[0] // self.factor) // 2), int((w - self.img_wh[1] // self.factor) // 2), int(
                        (h + self.img_wh[0] // self.factor) // 2), int((w + self.img_wh[1] // self.factor) // 2)))
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
                sample['rgbs'] = img

        return sample


if __name__ == "__main__":
    dataset = LLFF_ray_patch_1image_proj_Dataset('../../nerf_llff_data/room', split='train',
                                                 load_depth=True, depth_type='nerf', sH=6, sW=6, patch_size_x=60, patch_size_y=80, with_ref=True)

    item = dataset[0]
