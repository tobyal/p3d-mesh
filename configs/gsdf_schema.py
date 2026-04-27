from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ExpConfig:
    data_name: str = ""
    exp_name: str = ""
    out_dir: str = ""


@dataclass
class TrainConfig:
    max_steps: int = 6000
    save_freq: int = 500
    report_freq: int = 100
    val_freq: int = 1000
    batch_size: int = 5000
    learning_rate: float = 1e-4
    warm_up_end: float = 0.0
    loss_weight: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    accelerator: str = "gpu"
    devices: int = 1
    seed: int = 123456


@dataclass
class DataConfig:
    data_dir: str = ""
    data_name: str = ""

    data_type: str = ""
    queries_size: int = 1_000_000
    pull_knn: int = 1
    surface_queries: str = ""
    project_sdf_level: str = ""
    noisy_pts: str = ""

@dataclass
class SDFNetworkConfig:
    d_in: int = 3
    d_out: int = 1
    d_hidden: int = 256
    n_layers: int = 8
    skip_in: Tuple[int, ...] = (4,)
    multires: int = 0
    geometric_init: bool = True
    weight_norm: bool = True
    inside_outside: bool = False


@dataclass
class ModelConfig:
    sdf_network: SDFNetworkConfig = field(default_factory=SDFNetworkConfig)



@dataclass
class PathConfig:
    ckpt: str = ""
    recording: List[str] = field(default_factory=list)


@dataclass
class GSDFConfig:
    exp: ExpConfig = field(default_factory=ExpConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)