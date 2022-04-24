import os
import sys
from opt import get_opts
import torch

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger
# from pytorch_lightning.plugins import DDPPlugin

from models.sinnerf import SinNeRF

if __name__ == '__main__':
    hparams = get_opts()
    print(str(hparams))

    if hparams.model == 'sinnerf':
        system = SinNeRF(hparams)
    else:
        raise NotImplementedError
    if hparams.pt_model != None:
        dic = torch.load(hparams.pt_model)
        if hparams.nerf_only:
            nerf_fine = {k.replace('nerf_fine.', ''): v for k,
                         v in dic['state_dict'].items() if 'nerf_fine' in k}
            nerf_coarse = {k.replace('nerf_coarse.', ''): v for k,
                           v in dic['state_dict'].items() if 'nerf_coarse' in k}
            system.nerf_coarse.load_state_dict(nerf_coarse)
            system.nerf_fine.load_state_dict(nerf_fine)
        else:
            system.load_state_dict(dic['state_dict'], strict=False)
        print(f"Loaded model from <{hparams.pt_model}>")
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(
        f'ckpts/{hparams.exp_name}', '{epoch:d}'), monitor='val/psnr', mode='max', save_last=True, save_top_k=2)

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary="full",
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus > 1 else None,
                      #   plugins=DDPPlugin(find_unused_parameters=False),
                      num_sanity_val_steps=1,
                      benchmark=True,
                      #   precision=16,
                      check_val_every_n_epoch=20,
                      #   prepare_data_per_node=True,
                      profiler=hparams.num_gpus == 1)

# with torch.autograd.detect_anomaly():
    trainer.fit(system)
