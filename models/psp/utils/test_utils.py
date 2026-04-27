import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from einops import rearrange

from .psp_utils import FPS, extract_knn_patch, normalize_point_cloud
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()



def load_point_cloud_as_tensor(path: str, device: torch.device):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    tensor = torch.from_numpy(points).float().to(device)
    tensor = rearrange(tensor, "n c -> c n").contiguous().unsqueeze(0)
    return tensor


def save_point_cloud(path: str, points_tensor: torch.Tensor):
    saved_pcd = rearrange(points_tensor.squeeze(0), "c n -> n c").contiguous()
    saved_pcd = saved_pcd.detach().cpu().numpy()

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(saved_pcd)
    o3d.io.write_point_cloud(path, pcd_o3d, write_ascii=True)


def upsampling(cfg, model, input_pcd, target_num_points = None, up_rate = 4):
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = cfg.patch_pts_num
    sample_num = int(pcd_pts_num / patch_pts_num * cfg.patch_rate)

    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)

    coarse_pts, _ = model(patches, centroid, furthest_distance)
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, "b c n -> c (b n)").contiguous()

    
    final_num_points = target_num_points if target_num_points is not None else pcd_pts_num * up_rate
    coarse_pts = FPS(coarse_pts.unsqueeze(0), final_num_points)

    return coarse_pts

def _normalize_point_cloud(pc):
    # b, n, 3
    centroid = torch.mean(pc, dim=1, keepdim = True) # b, 1, 3
    pc = pc - centroid # b, n, 3
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0] # b, 1, 1
    pc = pc / furthest_distance
    return pc

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)