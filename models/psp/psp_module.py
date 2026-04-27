import os
import torch
import torch.optim as optim
import lightning as L
from einops import rearrange
from pathlib import Path
import open3d as o3d
import numpy as np

from models.psp import PSPNet
from models.psp.utils.psp_utils import get_cd_loss, normalize_point_cloud
from models.psp.utils.test_utils import load_point_cloud_as_tensor, save_point_cloud, upsampling, chamfer_sqrt


class PSPModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PSPNet(args.model)

        # 可选：只保存一部分超参
        self.save_hyperparameters()
        
        self.test_outputs = []

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

        print(f"[Info] Loading pretrained weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]


            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[len("model."):]] = v

            state_dict = new_state_dict

        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)

        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
        
    def forward(self, input_pts, patch_center, patch_radius):
        return self.model(input_pts, patch_center, patch_radius)

    def training_step(self, batch, batch_idx):
        input_pts, gt_pts, patch_center, patch_radius = batch

        input_pts = input_pts[0]
        gt_pts = gt_pts[0]
        patch_center = patch_center[0]
        patch_radius = patch_radius[0]

        input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float()
        gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float()
        patch_center = rearrange(patch_center, 'b n c -> b c n').contiguous().float()
        patch_radius = patch_radius.contiguous().float()

        gen_pts, reg_loss = self.model(input_pts, patch_center, patch_radius)

        loss = get_cd_loss(gen_pts, gt_pts)
        total_loss = loss + reg_loss if reg_loss is not None else loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        if reg_loss is not None:
            self.log("reg_loss", reg_loss, on_step=True, on_epoch=True, batch_size=1)

        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_path = batch["input_path"][0]

        input_pcd = load_point_cloud_as_tensor(input_path, self.device)
        input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)

        pcd_upsampled = upsampling(self.args.test, self.model, input_pcd)
        pcd_upsampled = centroid + pcd_upsampled * furthest_distance

        if self.args.test.up_rate == 16:
            pcd_upsampled, centroid, furthest_distance = normalize_point_cloud(pcd_upsampled)
            pcd_upsampled = upsampling(self.args.test, self.model, pcd_upsampled)
            pcd_upsampled = centroid + pcd_upsampled * furthest_distance

        # save ply
        file_name = Path(input_path).stem + ".ply"
        output_path = Path(self.args.exp.out_dir) / "ply" / Path(self.args.test.ckpt_dir).stem / file_name
        save_point_cloud(str(output_path), pcd_upsampled)
        
        return output_path

    def test_step(self, batch, batch_idx):
        input_path = batch["input_path"][0]
        gt_path = batch["gt_path"][0]

        input_pcd = load_point_cloud_as_tensor(input_path, self.device)
        input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)

        pcd_upsampled = upsampling(self.args.test, self.model, input_pcd)
        pcd_upsampled = centroid + pcd_upsampled * furthest_distance

        if self.args.test.up_rate == 16:
            pcd_upsampled, centroid, furthest_distance = normalize_point_cloud(pcd_upsampled)
            pcd_upsampled = upsampling(self.args.test, self.model, pcd_upsampled)
            pcd_upsampled = centroid + pcd_upsampled * furthest_distance

        # save ply
        file_name = Path(input_path).stem + ".ply"
        output_path = Path(self.args.exp.out_dir) / "ply" / Path(self.args.test.ckpt_dir).stem / file_name
        save_point_cloud(str(output_path), pcd_upsampled)
        
        # metric
        gt = o3d.io.read_point_cloud(gt_path)
        gt = torch.from_numpy(np.asarray(gt.points)).float().unsqueeze(0).to(self.device)

        cd = chamfer_sqrt(
            pcd_upsampled.permute(0, 2, 1).contiguous(),
            gt
        ) * 1e3

        self.log("test_cd", cd, prog_bar=True, batch_size=1)

        result = {
            "name": Path(input_path).name,
            "cd": cd.detach().cpu().item(),
        }
        self.test_outputs.append(result)
        
        return result
    def on_test_epoch_end(self):
        save_dir = Path(self.args.exp.out_dir) / "ply" / Path(self.args.test.ckpt_dir).stem
        save_dir.mkdir(parents=True, exist_ok=True)
        print(save_dir)
        total_cd = 0.0
        lines = []

        for item in self.test_outputs:
            lines.append(f"{item['name']}: {item['cd']}")
            total_cd += item["cd"]

        if len(self.test_outputs) > 0:
            lines.append(f"overall: {total_cd / len(self.test_outputs)}")

        with open(save_dir / "cd.txt", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        self.test_outputs.clear()
    def configure_optimizers(self):
        deform_params = [v for k, v in self.model.named_parameters() if "deform" in k]
        other_params = [v for k, v in self.model.named_parameters() if "deform" not in k]

        assert self.args.train.optim in ["adam", "sgd"]

        if self.args.train.optim == "adam":
            optimizer = optim.Adam(
                [
                    {"params": other_params},
                    {"params": deform_params, "lr": self.args.train.lr * 0.1},
                ],
                lr=self.args.train.lr,
                weight_decay=self.args.train.weight_decay,
            )
        else:
            sgd_lr = self.args.train.lr * 100
            optimizer = optim.SGD(
                [
                    {"params": other_params},
                    {"params": deform_params, "lr": sgd_lr * 0.1},
                ],
                lr=sgd_lr,
            )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.05 ** (1 / 150),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }