import numpy as np
import torch
import trimesh
import mcubes
from tqdm import tqdm
from lightning.pytorch.callbacks import Callback


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 128
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in tqdm(enumerate(X), total=len(X), desc="GSDF Mesh Inference"):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).to(bound_min.device)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[
                        xi * N: xi * N + len(xs),
                        yi * N: yi * N + len(ys),
                        zi * N: zi * N + len(zs)
                    ] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    return trimesh.Trimesh(vertices, triangles)



class GSDFMeshCallback(Callback):
    def __init__(self, every_n_train_steps: int):
        self.every_n_train_steps = every_n_train_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every_n_train_steps == 0:
            pl_module.export_mesh()