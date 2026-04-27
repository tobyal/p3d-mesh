import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
import os
from models.cpplib.libkdtree import KDTree
import trimesh
import fpsample

####### For PSP ###########
def load_patch_data(path):
    """
    读取一个 shape 的 patch 数据

    支持：
    - 单个 .npy 文件：形状必须为 (P, N, 3)

    返回：
    - patches: np.ndarray, shape (P, N, 3), dtype float32
    """
    if os.path.isfile(path):
        if path.endswith(".npy"):
            arr = np.load(path)
            assert arr.ndim == 3 and arr.shape[-1] == 3, \
                f"{path} 形状应为 (P,N,3)，当前 {arr.shape}"
            return arr.astype(np.float32)
        else:
            raise ValueError(f"暂不支持该文件格式: {path}")

    else:
        raise ValueError(f"路径不存在或不是文件: {path}")


####### For GSDF ###########
def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx

def process_data(data_dir, dataname, conf, with_normal=False):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        if with_normal:
            pcd = o3d.io.read_point_cloud(os.path.join(data_dir, dataname) + '.ply')
            pointcloud = np.array(pcd.points)
            pointnormal = np.array(pcd.normals)
            pointnormal = pointnormal / np.linalg.norm(pointnormal, axis=-1, keepdims=True)
        else:
            pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
            pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'):
        pointcloud = np.load(os.path.join(data_dir, dataname) + '.xyz', allow_pickle=True)
    else:
        print('Only support .xyz or .ply data. Please make adjust your data.')
        exit()
    shape_scale = np.max(
        [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
         np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
    shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                    (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                    (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    queries_size = conf.queries_size
    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = queries_size // POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace=False)
    pointcloud = pointcloud[point_idx, :]
    if with_normal:
        pointnormal = pointnormal[point_idx, :]

    ptree = KDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud, 100, axis=0):
        d = ptree.query(p, 51)
        sigmas.append(d[0][:, -1])

    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []
    sample_near_normal = []
    kdtree = KDTree(pointcloud)
    knn = conf.pull_knn
    for i in range(QUERY_EACH):
        scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale * np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1, 3)

        _, nearest_idx = kdtree.query(tt, knn)
        nearest_points = pointcloud[nearest_idx]
        nearest_points = np.asarray(nearest_points).reshape(-1, 3)
        sample_near.append(nearest_points)
        if with_normal:
            nearest_points_normals = np.asarray(pointnormal[nearest_idx]).reshape(-1, 3)
            sample_near_normal.append(nearest_points_normals)

    sample = np.asarray(sample).reshape(-1, 3)
    sample_near = np.asarray(sample_near).reshape(-1, 3)
    cube_boxsize = 1.1
    sample_uniform = np.random.rand(sample.shape[0] // 10, 3)
    sample_uniform = cube_boxsize * (sample_uniform - 0.5)  # [-0.55,0.55]
    _, nearest_idx = kdtree.query(sample_uniform, knn)
    nearest_points = pointcloud[nearest_idx]
    nearest_points = np.asarray(nearest_points).reshape(-1, 3)
    sample_uniform_near = nearest_points

    if with_normal:
        sample_near_normal = np.asarray(sample_near_normal).reshape(-1, 3)
        sample_uniform_near_normal = np.asarray(pointnormal[nearest_idx]).reshape(-1, 3)
    else:
        sample_near_normal = None
        sample_uniform_near_normal = None

    np.savez(os.path.join(data_dir, dataname) + '.npz', sample=sample, loc=shape_center, scale=shape_scale,
             sample_near=sample_near, sample_near_normal=sample_near_normal,
             sample_uniform=sample_uniform, sample_uniform_near=sample_uniform_near, sample_uniform_near_normal=sample_uniform_near_normal,
             point=pointcloud, knn=knn)
    
def get_neighbor_idx_noself(pc, query_pts, k):
    kdtree = KDTree(pc)
    (x, idx) = kdtree.query(query_pts, k + 1)
    idx = idx[:, 1:]
    idx = torch.from_numpy(idx.astype(int)).long()

    return idx.squeeze(-1)

def guassian_kernel(points, queries):  # input n,k,3/n,3
    pts_dist = torch.linalg.norm((points - queries.unsqueeze(1)), ord=2, dim=-1)  # n,k
    h = pts_dist.mean(dim=-1, keepdim=True)  # n,1
    dist_exp = torch.exp(-pts_dist ** 2 / h ** 2)  # n,k
    gaussian_weight = dist_exp / dist_exp.sum(dim=-1).unsqueeze(-1)  # n,k

    return gaussian_weight