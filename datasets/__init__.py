from .llff import LLFFDataset
from .llff_ray_patch_1image_proj import LLFF_ray_patch_1image_proj_Dataset
from .blender_ray_patch_1image_rot3d import Blender_ray_patch_1image_rot3d_Dataset
from .blender_ray_patch_1image_proj import Blender_ray_patch_1image_proj_Dataset
from .dtu_proj import MVSDatasetDTU_proj

dataset_dict = {
    'llff': LLFFDataset,
    'blender_ray_patch_1image_proj': Blender_ray_patch_1image_proj_Dataset,
    'blender_ray_patch_1image_rot3d': Blender_ray_patch_1image_rot3d_Dataset,
    'dtu_proj': MVSDatasetDTU_proj, 'llff_ray_patch_1image_proj': LLFF_ray_patch_1image_proj_Dataset,

}
