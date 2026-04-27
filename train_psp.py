import os
import argparse
import torch
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from utils.config import load_config
from configs.psp_schema import PSPConfig
from data.psp_datamodule import PSPDataModule
from models.psp.psp_module import PSPModule

def main():
    cfg = load_config("configs/psp.yaml", PSPConfig())
    cfg.exp.out_dir = os.path.join(cfg.exp.out_dir, cfg.exp.data_name, cfg.exp.exp_name)
    cfg.data.train_dir = os.path.join(cfg.data.train_dir, cfg.exp.data_name)
    
    datamodule = PSPDataModule(cfg.data)
    model = PSPModule(cfg)
    if cfg.train.ckpt_dir:
        model.load_pretrained(cfg.train.ckpt_dir, strict=False)
    
    logger = CSVLogger(save_dir=cfg.exp.out_dir, name="psp_log")

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.exp.out_dir, "ckpt"),
            filename="psp-{epoch}",
            every_n_epochs=cfg.train.save_rate,
            save_top_k=-1,
            save_weights_only=True,
        ),
    ]

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[3],
        max_epochs=cfg.train.epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.print_rate,
        default_root_dir=cfg.exp.out_dir,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()