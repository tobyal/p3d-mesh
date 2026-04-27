import lightning as L
from torch.utils.data import DataLoader

from data.datasets.psp_dataset import Shape2PatchDataset, PSPInferDataset
import os


class PSPDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = args.train_dir
        self.test_dataset = args.test_dir

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = Shape2PatchDataset(
                input_path=os.path.join(self.train_dataset, "input"),
                gt_path=os.path.join(self.train_dataset, "gt"),
                normalize=True,
            )
            
        if stage in (None, "predict"):
            self.predict_dataset = PSPInferDataset(
                input_dir=os.path.join(self.test_dataset, "input"),
                gt_dir=None,
            )

        if stage in (None, "test"):
            self.test_dataset = PSPInferDataset(
                input_dir=os.path.join(self.test_dataset, "input"),
                gt_dir=os.path.join(self.test_dataset, "gt"),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,      # 保持你现有逻辑
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )