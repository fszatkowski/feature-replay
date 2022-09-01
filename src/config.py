from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Benchmark:
    name: str
    input_size: tuple[int, int, int]
    n_classes: int
    n_experiences: int
    augmentations: bool


@dataclass
class Strategy:
    name: str

    # BasicBuffer strategy
    memory_size: Union[int, list[int]]

    # FeatureBuffer strategy
    replay_mb_size: Union[int, list[int]]
    replay_prob: Union[float, list[float]]
    replay_slowdown: float

    # LwF
    alpha: float
    temperature: float

    # EWC
    ewc_lambda: float


@dataclass
class Model:
    name: str

    # MLP
    hidden_sizes: list[int]
    dropout_ratio: float

    # ConvMLP
    channels: list[int]
    kernel_size: int
    pooling: bool


@dataclass
class Optimizer:
    name: str
    lr: float
    momentum: float
    l2: float


@dataclass
class Training:
    train_epochs: int
    train_mb_size: int
    eval_mb_size: int
    optimizer: Optimizer


@dataclass
class Config:
    benchmark: Benchmark
    strategy: Strategy
    model: Model
    training: Training

    seed: Optional[int]
    device: str
    wandb_entity: str
    wandb_project: str
    output_model_path: Optional[str]
