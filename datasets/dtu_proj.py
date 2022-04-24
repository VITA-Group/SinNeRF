
import torch.nn.functional as FF
from torch.utils.data import Dataset
# from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
from .ray_utils import *
import re
import shutil


def get_ray_directions_dtu(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def zoom(im):
    return im
    return cv2.resize(im, None, fx=4, fy=4,
                      interpolation=cv2.INTER_LINEAR)
    #    interpolation=cv2.INTER_NEAREST)


def look_at_rotation(camera_position, at=(0, 0, 0), up=(0., 0, 1), device: str = "cpu") -> torch.Tensor:
    # Format input and broadcast
    nbatch = camera_position.shape[0]
    camera_position = camera_position.to(device)
    if not torch.is_tensor(at):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    at = at.expand(nbatch, 3)
    if not torch.is_tensor(up):
        up = torch.tensor(up, dtype=torch.float32, device=device)
    up = up.expand(nbatch, 3)

    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = FF.normalize(camera_position - at, eps=1e-5)
    x_axis = FF.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = FF.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(
        0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        # print(f'warning: up vector {up[0].detach()} is close to x_axis {z_axis[0].detach()}')
        replacement = FF.normalize(
            torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :],
                  z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


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


def rot_theta2(th): return torch.Tensor([
    [np.cos(th), 0, np.sin(th), 0],
    [0, 1, 0, 0],
    [-np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rot_z(th): return torch.Tensor([
    [np.cos(th), -np.sin(th), 0, 0],
    [np.sin(th), np.cos(th), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def pose_spherical_dtu(radii, focus_depth, n_poses=120, world_center=np.array([0, 0, 0])):
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
    # rotate 4pi (2 rounds)
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center+world_center], 1)]  # (3, 4)

    # (n_poses, 3, 4)
    return np.stack(poses_spiral, 0) @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def pose_spherical(c2w, theta, phi):
    # c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi)  # @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def rotate(angle):
    return rot_theta2(angle/180.*np.pi)


def convert(c2w):
    return np.linalg.inv(c2w)


def rotate2(angle, c2w):
    w2c = np.linalg.inv(c2w)
    w2c = c2w
    T = rotate(angle) @ w2c[:, -1]
    T = (T[:-1]).reshape(1, 3).float()
    R = look_at_rotation(T, up=(0, 1, 0))
    # print(R.shape, T.shape)
    new = torch.cat([R[0], T.reshape(3, 1)], dim=1)
    # print(new.shape)
    new = torch.cat([new, torch.FloatTensor([[0, 0, 0, 1]])], dim=0)
    return np.linalg.inv(new.numpy())
    # return new.numpy()


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def warp_img_proj_numpy(data, depth_ref, ref_proj_mat, src_proj_mats):
    height = depth_ref.shape[0]
    width = depth_ref.shape[1]
    # print(height, width)
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # print("yy", yy.max(), yy.min())
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])
    X = np.vstack((xx, yy, np.ones_like(xx)))
    D = depth_ref.reshape([-1])
    X = np.vstack((X * D, np.ones_like(xx)))

    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mats, X)
    depth_src = X[2].reshape([height, width]).astype(np.float32)
    X /= X[2]
    X = X[:2]

    yy = X[0].reshape([height, width]).astype(np.float32)
    xx = X[1].reshape([height, width]).astype(np.float32)
    new = np.zeros_like(data)
    new_depth = np.zeros_like(depth_src)
    xx_base, yy_base = np.meshgrid(np.arange(0, width), np.arange(0, height))
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    yy = np.clip(yy, 0, width - 1).astype(np.int64)
    xx = np.clip(xx, 0, height - 1).astype(np.int64)
    # 反了一下
    y_res = xx.reshape(-1)
    x_res = yy.reshape(-1)
    # painter's algo
    for i in range(yy_base.shape[0]):
        # print(x_res.shape, y_res.shape, yy_base.shape)
        if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
            new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
            new[y_res[i], x_res[i]] = data[yy_base[i], xx_base[i]]

    return new, new_depth


class MVSDatasetDTU_proj(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=1, img_wh=None, downSample=1, max_len=-1, patch_size=-1, sW=1, sH=1, patch_size_x=-1, patch_size_y=-1, scan=4, **kwargs):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        self.split = split
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y

        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        # self.scale_factor = 1.0
        self.max_len = max_len
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'
        self.scan = scan  # 115

        self.light_idx = 3

        self.id_list = [2]
        self.build_metas()
        self.n_views = n_views
        self.levels = levels  # FPN levels
        self.get_src_views()
        self.build_proj_mats()
        self.define_transforms()
        print(f'==> image down scale: {self.downSample}')
        self.load()
        self.patch_size = patch_size
        self.sW = sW
        self.sH = sH
        self.white_back = True

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor()
                                    ])

    def get_src_views(self):
        # with open(f'configs/dtu_pairs.txt') as f:
        with open(os.path.join(self.root_dir, f'Cameras/pair.txt')) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            # print(self.id_list)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                # print(ref_view)
                # ref_view is 43
                # print('??')
                self.src_views = [int(x)
                                  for x in f.readline().rstrip().split()[1::2]]
                if ref_view in self.id_list:
                    break
        print(self.src_views)

    def build_metas(self):
        self.id_list = np.unique(self.id_list)
        self.build_remap()

    def build_proj_mats(self):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        for vid in self.id_list:
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(
                proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor
            intrinsics += [intrinsic.copy()]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]

            proj_mats += [(proj_mat_l, near_far)]
            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]

        self.proj_mats, self.intrinsics = np.stack(
            proj_mats), np.stack(intrinsics)
        self.world2cams, self.cam2worlds = np.stack(
            world2cams), np.stack(cam2worlds)

        self.src_data = {}
        for vid in self.src_views:
            self.src_data[vid] = {}
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(
                proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor

            intrinsics += [intrinsic.copy()]
            self.src_data[vid]['intrinsics'] = intrinsic

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]
            self.src_data[vid]['proj_mats'] = [proj_mat_l, near_far]
            self.src_data[vid]['world2cams'] = np.stack([extrinsic])
            self.src_data[vid]['cam2worlds'] = np.stack(
                [np.linalg.inv(extrinsic)])

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(
            ' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(
            ' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + \
            float(lines[11].split()[1]) * 192 * self.scale_factor
        self.depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, [depth_min, depth_max]

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[
                           0], dtype=np.float32)
        depth_h = cv2.resize(depth_h, None, fx=4, fy=4,
                             interpolation=cv2.INTER_LINEAR)
        mask = depth_h > 0

        return depth_h, mask, depth_h

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i

    def __len__(self):
        if self.split == 'train':
            return self.len
        return len(self.poses_test)

    def load(self):
        self.sample = {}

        affine_mat, affine_mat_inv = [], []
        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [
        ], [], [], []  # record proj mats between views
        for i, vid in enumerate(self.id_list):

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/scan{self.scan}_train/rect_{vid + 1:03d}_{self.light_idx}_r5000.png')
            depth_filename = os.path.join(self.root_dir, 'MVSNet_pytorch_outputs/',
                                          f'scan{self.scan}/depth_est/rect_{vid + 1:03d}_{self.light_idx}_r5000.pfm')
            # read 1/4 depth, its' aligned
            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            # img = img.view(3, -1).permute(1, 0)
            imgs += [img]
            # [3, w, h]
            self.ref_view = img.permute(1, 2, 0)
            # [w, h, 3]

            index_mat = self.remap[vid]
            proj_mat_ls, near_far = self.proj_mats[index_mat]
            self.near, self.far = near_far
            intrinsics.append(self.intrinsics[index_mat])
            w2cs.append(self.world2cams[index_mat])
            c2ws.append(self.cam2worlds[index_mat])
            # print(depth_filename)
            if os.path.exists(depth_filename):
                depth, mask, depth_h = self.read_depth(depth_filename)
                depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                if self.split == 'train':
                    raise NotImplementedError
                depth_h = np.zeros((1, 1))
                depths_h.append(depth_h)

            near_fars.append(near_far)
            depth_h = np.clip(depth_h, a_min=near_far[0], a_max=near_far[1])
            print(near_far, depth_h.min(), depth_h.max(), depth_h.mean())

        imgs = torch.stack(imgs).float()

        depths_h = np.stack(depths_h)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(
            w2cs), np.stack(c2ws), np.stack(near_fars)

        self.sample['images'] = imgs  # (V, H, W, 3)
        self.sample['depths'] = depths_h.astype(np.float32)  # (V, H, W)
        self.sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        self.sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        self.sample['near_fars'] = near_fars.astype(np.float32)
        # self.sample['proj_mats'] = proj_mats.astype(np.float32)
        self.sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)

        self.focal = [intrinsics[0][0][0], intrinsics[0][1][1]]
        print('focal', self.focal)
        center = [intrinsics[0][0, 2], intrinsics[0][1, 2]]
        print('center', center)
        self.directions = get_ray_directions_dtu(
            self.img_wh[1], self.img_wh[0], self.focal, center)  # (h, w, 3)
        print(self.directions.mean())
        self.ref_c2w = c2ws[0]
        print('ref', self.ref_c2w)
        rays_o, rays_d = get_rays(
            self.directions, torch.FloatTensor(c2ws[0][:3, :4]).float())
        rays_ = torch.cat([rays_o, rays_d,  self.near*torch.ones_like(rays_o[:, :1]),
                          self.far*torch.ones_like(rays_o[:, :1])], 1)  # (h*w, 8)
        w, h, _ = self.ref_view.shape
        self.ref_rays = rays_.view(w, h, 8)
        self.ref_depth = torch.from_numpy(zoom(depths_h).reshape(w, h)).float()
        print(self.ref_view.shape, self.ref_rays.shape, self.ref_depth.shape)
        self.all_rays_gt = [rays_]
        # print('imgs', imgs.shape)
        self.all_rgbs_gt = [imgs.reshape(3, -1).permute(1, 0)]
        print(self.all_rays_gt[0].shape, self.all_rgbs_gt[0].shape)
        self.all_depth = [torch.from_numpy(
            zoom(depths_h).reshape(-1, 1)).float()]

        idx = 0
        self.all_rays_full = []
        self.all_rgbs_full = []
        self.all_depths_full = []
        self.proj_rays_full = []
        self.proj_rgbs_full = []
        self.proj_depths_full = []
        self.poses_test = [self.ref_c2w]
        h, w = self.img_wh
        ww = []
        for cur_src_view in self.src_views:
            src_w2c = self.src_data[cur_src_view]['world2cams']
            src_c2w = self.src_data[cur_src_view]['cam2worlds'][0]
            intrinsics = self.src_data[cur_src_view]['intrinsics']
            ref_w2c = w2cs[0]
            ref_proj, _ = self.proj_mats[0]
            # print(ref_proj)
            src_proj, _ = self.src_data[cur_src_view]['proj_mats']
            warp_rgb, warp_depth = warp_img_proj_numpy((imgs[0].numpy().transpose(
                [1, 2, 0])), (depths_h[0]), (ref_proj.astype(np.float32)), (src_proj.astype(np.float32)))
            Image.fromarray((warp_rgb * 255).astype(np.uint8)
                            ).save(f'vis/{cur_src_view}_rgb.png')
            (visualize_depth(warp_depth[:, :])).save(
                f'vis/{cur_src_view}_depth.png')
            np.save(f'vis/{cur_src_view}_depth.npy', warp_depth)
            idx += 1

            rays_o, rays_d = get_rays(
                self.directions, torch.FloatTensor(src_c2w[:3, :4]).float())
            self.poses_test += [src_c2w]
            rays_ = torch.cat([rays_o, rays_d,  self.near*torch.ones_like(
                rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], 1)  # (h*w, 8)
            rgbs_ = zoom(warp_rgb)
            warp_depth = zoom(warp_depth)  # .reshape(-1, 1)
            self.all_rays_full.append(rays_.view(w, h, 8))
            self.all_rgbs_full.append(torch.FloatTensor(rgbs_).view(w, h, 3))
            self.all_depths_full.append(
                torch.FloatTensor(warp_depth).view(w, h, 1))
            # print('rgbs', rgbs_.shape)
            rgbs_ = torch.FloatTensor(rgbs_).view(-1, 3)
            warp_depth = torch.FloatTensor(warp_depth).view(-1, 1)

            rays_ = rays_[rgbs_.sum(dim=-1) != 0]
            warp_depth = warp_depth[rgbs_.sum(dim=-1) != 0]
            rgbs_ = rgbs_[rgbs_.sum(dim=-1) != 0]
            self.proj_rays_full += [rays_]
            self.proj_rgbs_full += [rgbs_]
            self.proj_depths_full += [(warp_depth).float()]

        self.all_rays_gt = torch.cat(self.all_rays_gt, 0)
        self.all_rgbs_gt = torch.cat(self.all_rgbs_gt, 0)
        self.all_depth = torch.cat(self.all_depth, 0)

        self.all_rays_full = torch.stack(self.all_rays_full, 0)
        self.all_rgbs_full = torch.stack(self.all_rgbs_full, 0)
        self.all_depths_full = torch.stack(self.all_depths_full, 0)
        self.proj_rays_full = torch.cat(self.proj_rays_full, 0)
        self.proj_rgbs_full = torch.cat(self.proj_rgbs_full, 0)
        self.proj_depths_full = torch.cat(self.proj_depths_full, 0)
        print(self.all_rays_gt.shape, self.all_rgbs_gt.shape, self.all_depth.shape)
        self.len = self.len_full = (self.all_rays_full.shape[0])
        self.all_rgb_num = self.all_rgbs_gt.shape[0]

        print(self.all_rgbs_full.shape, self.all_rays_full.shape,
              self.all_depths_full.shape)

        self.gt_img = [self.ref_view.permute(2, 0, 1)]
        # idx = 0
        for cur_src_view in self.src_views:
            src_c2w = self.src_data[cur_src_view]['cam2worlds'][0]
            vid = cur_src_view
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/scan{self.scan}_train/rect_{vid + 1:03d}_{self.light_idx}_r5000.png')
            # print(vid, img_filename)
            depth_filename = os.path.join(self.root_dir, 'MVSNet_pytorch_outputs/',
                                          f'scan{self.scan}/depth_est/rect_{vid + 1:03d}_{self.light_idx}_r5000.pfm')

            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            img.save(f'vis/{cur_src_view}_gt.png')
            img = self.transform(img)
            # [3, w, h]
            # img = img.view(3, -1).permute(1, 0)
            self.gt_img += [img]

            depth, mask, depth_h = self.read_depth(depth_filename)
            depth_h *= self.scale_factor
            np.save(f'vis/{cur_src_view}_depth_gt.npy', depth_h)
            (visualize_depth(depth_h)).save(f'vis/{cur_src_view}_depth_gt.png')

    def __getitem__(self, idx):
        if self.split == 'train':
            # new_idx = np.random.randint(0, self.img_num)
            im = self.gt_img[0]
            _, w, h = im.shape
            if w > self.patch_size:
                while True:
                    ll = np.random.randint(
                        0, w - (self.patch_size_x - 1) * self.sW - 1)
                    up = np.random.randint(
                        0, h - (self.patch_size_y - 1) * self.sH - 1)
                    patch = im[:, ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW,
                               up:up+(self.patch_size_y - 1) * self.sH + 1:self.sH]

                    if patch.mean() > 0.01:
                        break
            else:
                patch = im

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
            ray_idx = np.random.choice(self.all_rgb_num, num)
            ray_idx_proj = np.random.choice(
                (self.proj_depths_full).shape[0], num)

            sample = {
                'rays': self.all_rays_gt[ray_idx],
                'rgbs': self.all_rgbs_gt[ray_idx],
                'depth': self.all_depth[ray_idx],
                'rays_proj': self.proj_rays_full[ray_idx_proj],
                'rgbs_proj': self.proj_rgbs_full[ray_idx_proj],
                'depth_proj': self.proj_depths_full[ray_idx_proj],
                'real_patch': patch,
                'rays_full': fake_patch,
                'warp_patch': warp_patch,
                'warp_patch_depth': warp_patch_depth,
            }
            sample['depth_ray'] = self.ref_rays[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                self.patch_size_y - 1) * self.sH + 1:self.sH, :].reshape(-1, 8)
            sample['depth_gt'] = self.ref_depth[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                self.patch_size_y - 1) * self.sH + 1:self.sH].reshape(-1, 1)
            sample['depth_ray_rgb'] = self.ref_view[ll:ll+(self.patch_size_x - 1) * self.sW + 1:self.sW, up:up+(
                self.patch_size_y - 1) * self.sH + 1:self.sH, :].reshape(-1, 3)
            # for k, v in sample.items():
            #     print(k, v.shape)
            return sample

        if self.split in ['val', 'test']:
            c2w = self.poses_test[idx]
            rays_o, rays_d = get_rays(
                self.directions, torch.FloatTensor(c2w[:3, :4]).float())
            rays_ = torch.cat([rays_o, rays_d,  self.near*torch.ones_like(
                rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], 1)  # (h*w, 8)
            return {'rays': rays_, 'rgbs': self.gt_img[idx].view(3, -1).permute(1, 0)}


if __name__ == "__main__":
    dataset = MVSDatasetDTU_proj('../../mvs_training/dtu', split='train', load_depth=True,
                                 depth_type='nerf', sH=1, sW=1, patch_size_x=60, patch_size_y=80, with_ref=True, img_wh=[640, 512], scan=4)
    for i in range(10):
        item = dataset[i]
