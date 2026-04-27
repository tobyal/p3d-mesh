import math
import torch
import torch.nn.functional as F
import lightning as L
from pathlib import Path
from models.gsdf.networks.gsdf_net import SDFNetwork
from models.gsdf.utils.mesh_utils import extract_geometry
from models.gsdf.utils.loss_utils import pull_knn_loss
from typing import Optional


class GSDFModule(L.LightningModule):
    def __init__(self, cfg, point_size: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.point_size = point_size
        self.sdf_network = None

        # 这里先不立即建网络，等 datamodule setup 后再通过 hook 设置
        self.save_hyperparameters(ignore=[])

    def setup(self, stage=None):
        if self.sdf_network is None:
            point_size = self.trainer.datamodule.point_size
            self.sdf_network = SDFNetwork(
                point_size,
                self.cfg.model.sdf_network
            )

    def forward(self, x):
        return self.sdf_network.sdf(x, self.global_step)

    def training_step(self, batch, batch_idx):
        dm = self.trainer.datamodule

        sample_near, points_near, normals_near, \
        sample_uniform, points_uniform, normals_uniform = dm.sample_train_batch(self.global_step)

        samples = torch.cat((sample_near, sample_uniform), dim=0)
        points = torch.cat((points_near, points_uniform), dim=0)

        gradients_samples, sdf_samples = self.sdf_network.gradient(samples, self.global_step)
        gradients_samples_norm = F.normalize(gradients_samples, dim=-1)
        samples_moved = samples - gradients_samples_norm * sdf_samples

        move_position = samples_moved.detach()
        gradients_samples_moved, _ = self.sdf_network.gradient(move_position, self.global_step)
        gradients_samples_moved_norm = F.normalize(gradients_samples_moved, dim=-1)
        loss_grad_consis = (
            1 - F.cosine_similarity(
                gradients_samples_moved_norm,
                gradients_samples_norm,
                dim=-1
            )
        ).mean()

        points_ = points.clone() if dm.dataset_knn == 1 else points[:, 0, :]
        sdf_points = self.sdf_network.sdf(points_, self.global_step)

        if dm.dataset_knn == 1:
            loss_pull = torch.linalg.norm((points - samples_moved), ord=2, dim=-1).mean()
        else:
            loss_pull = pull_knn_loss(points, samples_moved, samples)

        loss_sdf = torch.abs(sdf_points).mean()
        loss_inter = torch.exp(-1e2 * torch.abs(sdf_samples)).mean()

        if normals_near is not None and normals_uniform is not None:
            normals_gt = torch.cat((normals_near, normals_uniform), dim=0)
            loss_normal = (
                1 - F.cosine_similarity(normals_gt, gradients_samples, dim=-1)
            ).mean()
        else:
            loss_normal = torch.zeros((1,), device=loss_sdf.device)

        w = self.cfg.train.loss_weight
        total_loss = (
            w[0] * loss_pull
            + w[1] * loss_sdf
            + w[2] * loss_grad_consis
            + w[3] * loss_inter
            + 0.01 * loss_normal
        )

        self.log("train_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1)
        self.log("loss_pull", loss_pull, on_step=True, on_epoch=False, batch_size=1)
        self.log("loss_sdf", loss_sdf, on_step=True, on_epoch=False, batch_size=1)
        self.log("loss_grad_consis", loss_grad_consis, on_step=True, on_epoch=False, batch_size=1)
        self.log("loss_inter", loss_inter, on_step=True, on_epoch=False, batch_size=1)
        self.log("loss_normal", loss_normal, on_step=True, on_epoch=False, batch_size=1)

        return total_loss

    def _lr_lambda(self, step: int):
        warm_up_end = self.cfg.train.warm_up_end
        max_steps = self.cfg.train.max_steps

        if warm_up_end > 0 and step < warm_up_end:
            return step / warm_up_end

        if max_steps <= warm_up_end:
            return 1.0

        progress = (step - warm_up_end) / (max_steps - warm_up_end)
        return 0.5 * (math.cos(progress * math.pi) + 1.0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.sdf_network.parameters(),
            lr=self.cfg.train.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._lr_lambda,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        
    def export_mesh(self, resolution=512, threshold=0.0, real_world=True):
        dm = self.trainer.datamodule
        dataset_np = dm.dataset_np

        bound_min = torch.tensor(dataset_np.object_bbox_min, dtype=torch.float32, device=self.device)
        bound_max = torch.tensor(dataset_np.object_bbox_max, dtype=torch.float32, device=self.device)

        mesh = extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.sdf_network.sdf(pts, self.global_step),
        )

        if real_world:
            mesh.apply_scale(dataset_np.scale)
            mesh.apply_translation(dataset_np.loc)

        mesh_dir = Path(self.cfg.exp.out_dir) / "mesh_outputs"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = mesh_dir / f"step_{self.global_step:08d}_res{resolution}.ply"
        mesh.export(mesh_path)
        return str(mesh_path)