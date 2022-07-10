from dataclasses import dataclass
from typing import Optional


@dataclass
class Benchmark:
    name: str
    input_size: int
    n_classes: int
    n_experiences: int


@dataclass
class Strategy:
    name: str

    # Buffer strategies
    buffer_size: int


@dataclass
class Model:
    name: str

    # MLP
    hidden_sizes: list[int]
    dropout_ratio: float


@dataclass
class Optimizer:
    name: str
    lr: float
    momentum: float


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
    wandb_project: str
