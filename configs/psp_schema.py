from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpConfig:
    data_name: str = ""
    exp_name: str = ""
    out_dir: str = ""


@dataclass
class TrainConfig:
    seed: int = 21
    optim: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_decay_step: int = 20
    gamma: float = 0.5
    num_workers: int = 4
    epochs: int = 100
    batch_size: int = 256
    print_rate: int = 200
    save_rate: int = 200
    ckpt_dir: str = ""
    

@dataclass
class TestConfig:
    patch_rate: int = 3
    patch_pts_num: int = 256
    up_rate: int = 4
    ckpt_dir: str = ""


@dataclass
class DataConfig:
    train_dir: str = ""
    test_dir: str = ""


@dataclass
class EncoderConfig:
    k: int = 16
    encoder_dim: int = 64
    out_dim: int = 64
    encoder_bn: bool = True
    global_mlp: bool = True


@dataclass
class DecoderConfig:
    out_dim: int = 64

    up_rate: int = 4
    simple: bool = True
    
    conv_radius: float = 0.8
    neighbor_limits: int = 30
    kernel_radius: float = 0.3
    kernel_point_receptive_radius: float = 0.2
    num_kernel_points: int = 15
    in_dim: int = 64
    kp_dim: int = 64
    is_kp_bn: bool = True
    is_kp_bias: bool = False
    rigid_scale: float = 0.625
    query_scale: float = 1.0
    
    head_num: int = 4
    trans_num: int = 2
    trans_dim: int = 128
    is_attn_bn: bool = False


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)




@dataclass
class PSPConfig:
    exp: ExpConfig = field(default_factory=ExpConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    test:TestConfig = field(default_factory=TestConfig)