# SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image

[[Paper]](https://arxiv.org/abs/2204.00928) [[Website]](https://vita-group.github.io/SinNeRF/)

<div>
<img src="https://user-images.githubusercontent.com/10709657/161841696-b88f5797-6fd0-4f5c-a661-dd2802a12ac7.gif" height="150"/>
<img src="https://user-images.githubusercontent.com/10709657/161841718-047678ab-4511-41ee-aed6-e716d370bfd9.gif" height="150"/>
<img src="https://user-images.githubusercontent.com/10709657/161841788-0baebb06-0375-48c6-8cc1-b00c796967db.gif" height="150"/>
</div>

## Pipeline

![](./docs/static/media/SinNeRF.drawio.01f837d9d69b1db62c00.jpg)

## Code

### Environment

```
pip install -r requirements.txt
```

### Dataset Preparation

Please download the datasets from these links:

- NeRF synthetic: Download `nerf_synthetic.zip` from https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
- LLFF: Download `nerf_llff_data.zip` from https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
- DTU: Download the preprocessed DTU training data from https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view

Please download the depth from here: https://drive.google.com/drive/folders/13Lc79Ox0k9Ih2o0Y9e_g_ky41Nx40eJw?usp=sharing

### Training

If you meet OOM issue, try:

1. enable `precision=16`
2. reduce the patch size `--patch_size` (or `--patch_size_x`, `--patch_size_y`) and enlarge the stride size `--sH`, `--sW`

<details>
  <summary>NeRF synthetic</summary>


- Step 1
  ```
  python train.py  --dataset_name blender_ray_patch_1image_rot3d  --root_dir  ../../dataset/nerf_synthetic/lego   --N_importance 64 --img_wh 400 400 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 2e-4  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name lego_s6 --with_ref --patch_size 64 --sW 6 --sH 6 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 10 --scan 4
  ```

- Step 2
  ```
  python train.py  --dataset_name blender_ray_patch_1image_rot3d  --root_dir  ../../dataset/nerf_synthetic/lego   --N_importance 64 --img_wh 400 400 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 5e-5  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name lego_s6_4ft --with_ref --patch_size 64 --sW 4 --sH 4 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0.01 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 0 --pt_model xxx.ckpt --nerf_only  --scan 4
  ```

</details>

<details>
  <summary>LLFF</summary>

- Step 1
  ```
  python train.py  --dataset_name llff_ray_patch_1image_proj  --root_dir  ../../dataset/nerf_llff_data/room   --N_importance 64 --img_wh 504 378 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 2e-4  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name llff_room_s4 --with_ref --patch_size_x 63 --patch_size_y 84 --sW 4 --sH 4 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 10
  ```

- Step 2
  ```
  python train.py  --dataset_name llff_ray_patch_1image_proj  --root_dir  ../../dataset/nerf_llff_data/room   --N_importance 64 --img_wh 504 378 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 5e-5  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name llff_room_s4_2ft --with_ref --patch_size_x 63 --patch_size_y 84 --sW 2 --sH 2 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0.01 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 0 --pt_model xxx.ckpt --nerf_only
  ```

</details>

<details>
  <summary>DTU</summary>

- Step 1
  ```
  python train.py  --dataset_name dtu_proj  --root_dir  ../../dataset/mvs_training/dtu   --N_importance 64 --img_wh 640 512 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 2e-4  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name dtu_scan4_s8 --with_ref --patch_size_y 70 --patch_size_x 56 --sW 8 --sH 8 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 10 --scan 4
  ```

- Step 2
  ```
  python train.py  --dataset_name dtu_proj  --root_dir  ../../dataset/mvs_training/dtu   --N_importance 64 --img_wh 640 512 --num_epochs 2000 --batch_size 1  --optimizer adam --lr 5e-5  --lr_scheduler steplr --decay_step 500 1000 --decay_gamma 0.5  --exp_name dtu_scan4_s8_4ft --with_ref --patch_size_y 70 --patch_size_x 56 --sW 4 --sH 4 --proj_weight 1 --depth_smooth_weight 0  --dis_weight 0.01 --num_gpus 4 --load_depth --depth_type nerf --model sinnerf --depth_weight 8 --vit_weight 0 --pt_model xxx.ckpt --nerf_only  --scan 4
  ```

More finetuning with smaller strides benefits reconstruction quality.

</details>


### Testing

```
python eval.py  --dataset_name llff  --root_dir /dataset/nerf_llff_data/room --N_importance 64 --img_wh 504 378 --model nerf --ckpt_path ckpts/room.ckpt --timestamp test
```

Please use `--split val` for NeRF synthetic dataset.

## Acknowledgement

Codebase based on https://github.com/kwea123/nerf_pl . Thanks for sharing!

## Citation

If you find this repo is helpful, please cite:

```

@InProceedings{Xu_2022_SinNeRF,
author = {Xu, Dejia and Jiang, Yifan and Wang, Peihao and Fan, Zhiwen and Shi, Humphrey and Wang, Zhangyang},
title = {SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image},
journal={arXiv preprint arXiv:2204.00928},
year={2022}
}

```
