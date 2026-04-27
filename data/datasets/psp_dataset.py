import torch
from data.datasets.dataset_utils import load_patch_data
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class Shape2PatchDataset(Dataset):
    """
    每个样本对应一个完整 shape。
    一个 shape 文件中保存了该 shape 的所有 patch, 形状为 (P, N, 3)。

    input_dir/
        shape_000.npy
        shape_001.npy
        ...

    gt_dir/
        shape_000.npy
        shape_001.npy
        ...
    """
    def __init__(self, input_path, gt_path, normalize=True):
        super().__init__()
        self.normalize = normalize

        if not os.path.isdir(input_path):
            raise ValueError(f"input_path 不是目录: {input_path}")
        if not os.path.isdir(gt_path):
            raise ValueError(f"gt_path 不是目录: {gt_path}")

        self.input_files = sorted(glob.glob(os.path.join(input_path, "*.npy")))
        self.gt_files = sorted(glob.glob(os.path.join(gt_path, "*.npy")))

        if len(self.input_files) == 0:
            raise ValueError(f"目录 {input_path} 下未找到 .npy 文件")
        if len(self.gt_files) == 0:
            raise ValueError(f"目录 {gt_path} 下未找到 .npy 文件")

        assert len(self.input_files) == len(self.gt_files), \
            f"输入 shape 文件数 {len(self.input_files)} 与 GT shape 文件数 {len(self.gt_files)} 不一致"

        # 可选：检查文件名是否一一对应
        input_names = [os.path.basename(f) for f in self.input_files]
        gt_names = [os.path.basename(f) for f in self.gt_files]
        assert input_names == gt_names, \
            f"输入与GT文件名不一致，请检查目录。\ninput: {input_names[:5]}\ngt: {gt_names[:5]}"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """
        返回一个完整 shape 的所有 patch

        Returns:
            input_pts:   (P, Nin, 3)
            gt_pts:      (P, Ngt, 3)
            data_radius: (P, 1)
        """
        input_pts = load_patch_data(self.input_files[idx]).copy()   # (P, Nin, 3)
        gt_pts = load_patch_data(self.gt_files[idx]).copy()         # (P, Ngt, 3)

        assert input_pts.shape[0] == gt_pts.shape[0], \
            f"patch 数不一致: input {input_pts.shape}, gt {gt_pts.shape}"

        P = input_pts.shape[0]

        if self.normalize:
            # 每个 patch 独立归一化，保持和原 patch-based 网络一致
            # centroid: (P, 1, 3)
            input_centroid = np.mean(input_pts, axis=1, keepdims=True)

            # input 去中心
            input_pts = input_pts - input_centroid

            # furthest distance: (P, 1)
            input_furthest_distance = np.sqrt(np.sum(input_pts ** 2, axis=-1))   # (P, Nin)
            input_furthest_distance = np.amax(input_furthest_distance, axis=1, keepdims=True)  # (P, 1)
            input_furthest_distance = np.maximum(input_furthest_distance, 1e-8)

            # broadcast 到 (P, N, 3)
            scale = input_furthest_distance[:, :, None]  # (P, 1, 1)

            # input 归一化
            input_pts = input_pts / scale

            # gt 用同一套 centroid 和 distance
            gt_pts = gt_pts - input_centroid
            gt_pts = gt_pts / scale

            data_radius = np.ones((P, 1), dtype=np.float32)
        else:
            data_radius = np.ones((P, 1), dtype=np.float32)

        return (
            torch.from_numpy(input_pts).float(),     # (P, Nin, 3)
            torch.from_numpy(gt_pts).float(),        # (P, Ngt, 3)
            #torch.from_numpy(data_radius).float(),    # (P, 1)
            torch.from_numpy(input_centroid).float(),            # (P, 1, 3)
            torch.from_numpy(input_furthest_distance).float()   # (P, 1)
        )


class PSPInferDataset(Dataset):
    def __init__(self, input_dir: str, gt_dir: Optional[str] = None, suffix: str = "*.ply"):
        self.input_paths = sorted(glob.glob(str(Path(input_dir) / suffix)))
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]

        sample = {
            "input_path": input_path,
        }

        if self.gt_dir is not None:
            gt_path = str(Path(self.gt_dir) / Path(input_path).name)
            sample["gt_path"] = gt_path

        return sample