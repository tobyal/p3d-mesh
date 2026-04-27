import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from configs.gsdf_schema import GSDFConfig
from utils.config import load_config
from data.gsdf_datamodule import GSDFDataModule
from models.gsdf.gsdf_module import GSDFModule
from models.gsdf.utils.mesh_utils import GSDFMeshCallback


def main():
    cfg = load_config("configs/gsdf.yaml", GSDFConfig())

    cfg.exp.out_dir = os.path.join(cfg.exp.out_dir, cfg.exp.data_name, cfg.exp.exp_name)

    datamodule = GSDFDataModule(cfg)
    model = GSDFModule(cfg)

    logger = CSVLogger(save_dir=cfg.exp.out_dir, name="gsdf_log")

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.exp.out_dir, "ckpt"),
            filename="gsdf-{step}",
            every_n_train_steps=cfg.train.save_freq,
            save_top_k=-1,
            save_weights_only=True,
        ),
        GSDFMeshCallback(every_n_train_steps=cfg.train.val_freq),
    ]

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.train.devices,
        max_steps=cfg.train.max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.report_freq,
        default_root_dir=cfg.exp.out_dir,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()