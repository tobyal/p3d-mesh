import lightning as L
from torch.utils.data import DataLoader
from data.datasets.gsdf_dataset import GSDFStepDataset, DatasetNP




class GSDFDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_np = None
        self.train_dataset = None
        self.point_size = None
        self.dataset_knn = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.dataset_np = DatasetNP(
                self.cfg.data.data_dir,
                self.cfg.data.data_name,
                self.cfg.data,
            )
            self.point_size = self.dataset_np.point_size
            self.dataset_knn = self.dataset_np.dataset_knn
            self.train_dataset = GSDFStepDataset(self.cfg.train.max_steps)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def sample_train_batch(self, iter_step: int):
        return self.dataset_np.sdf_train_data(
            self.cfg.train.batch_size,
            iter_step,
            self.cfg.data.data_type,
        )
        
        
