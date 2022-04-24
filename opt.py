import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['llff_ray_patch_1image_proj', 'blender_ray_patch_1image_proj',
                                 'blender_ray_patch_1image_rot3d', 'dtu_proj'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    # params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    ###########################
    # my extra params
    parser.add_argument('--with_ref', default=False, action="store_true")
    parser.add_argument('--patch_size', type=int, default=-1)
    # for llff / dtu
    parser.add_argument('--patch_size_x', type=int, default=-1)
    parser.add_argument('--patch_size_y', type=int, default=-1)
    parser.add_argument('--pt_model', type=str, default=None)
    parser.add_argument('--model', type=str,
                        default="nerf", choices=['sinnerf'])
    parser.add_argument('--repeat', type=int, default=1)
    # change ray sampling sparsity when generating rays
    parser.add_argument('--nW', type=int, default=32)
    # change ray num per patch (not used)
    parser.add_argument('--nH', type=int, default=32)
    # change ray num per patch (not used)
    parser.add_argument('--sW', type=int, default=1)
    # change ray sampling stride
    parser.add_argument('--sH', type=int, default=1)
    # change ray sampling stride
    parser.add_argument('--dloss', type=str, default="hinge")
    # discriminator loss type
    parser.add_argument('--load_depth', default=False,
                        action="store_true")  # use depth
    # load weight of nerf only from checkpoint
    parser.add_argument('--nerf_only', default=False, action="store_true")
    parser.add_argument('--depth_type', type=str,
                        default='nerf')  # depth supervision type
    # weight of discriminator loss
    parser.add_argument('--dis_weight', type=float, default=0.001)
    # weight of loss on projected views
    parser.add_argument('--proj_weight', type=float, default=1)
    parser.add_argument('--angle', type=int, default=30)  # angle for rot3d
    parser.add_argument('--scan', type=int, default=4)  # for dtu dataset
    # weight for depth supervision
    parser.add_argument('--depth_weight', type=float, default=0.05)
    parser.add_argument('--vit_weight', type=float,
                        default=0)  # weight for vit loss

    parser.add_argument('--depth_smooth_weight', type=float, default=0)
    parser.add_argument('--depth_anneal', default=False, action="store_true")
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'ft', 'clip', 'l2_ssim', 'l2_vgg'], help='loss to use')
    parser.add_argument('--patch_loss', type=str, default='mse',
                        choices=['mse', 'ft', 'clip', 'l2_ssim', 'l2_vgg'], help='loss to use')

    return parser.parse_args()
