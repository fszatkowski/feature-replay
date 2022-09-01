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
class GenerativeModel:
    name: str
    channels: list[int]
    strides: list[int]
    kernel_size: int
    hidden_sizes: list[int]
    nhid: int


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
class Generator:
    training: Training


@dataclass
class Strategy:
    name: str

    # Buffer strategies strategy
    memory_size: int

    # FeatureBuffer strategy
    replay_mb_size: Union[int, list[int]]
    update_strategy: str
    replay_slowdown: float

    # Generative
    generator: Generator
    increasing_replay_size: bool

    # LwF
    alpha: float
    temperature: float

    # EWC
    ewc_lambda: float


@dataclass
class Config:
    benchmark: Benchmark
    strategy: Strategy
    model: Model
    generative_model: GenerativeModel
    training: Training

    seed: Optional[int]
    device: str
    wandb_entity: str
    wandb_project: str
    output_model_path: Optional[str]
