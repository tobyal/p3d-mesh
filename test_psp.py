import lightning as L
import torch

from data.psp_datamodule import PSPDataModule
from models.psp.psp_module import PSPModule
from utils.config import load_config
from configs.psp_schema import PSPConfig
import os


def main():
    cfg = load_config("configs/psp.yaml", PSPConfig())
    cfg.exp.out_dir = os.path.join(cfg.exp.out_dir, cfg.exp.data_name, cfg.exp.exp_name)
    cfg.data.test_dir = os.path.join(cfg.data.test_dir, cfg.exp.data_name)

    datamodule = PSPDataModule(cfg.data)
    
    model = PSPModule(cfg)
    model.load_pretrained(cfg.test.ckpt_dir, True)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[3],
    )

    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    main()