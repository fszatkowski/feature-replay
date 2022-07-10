from dataclasses import dataclass
from typing import Optional


@dataclass
class Benchmark:
    name: str
    input_size: int
    n_classes: int
    n_experiences: int


@dataclass
class Replay:
    buffer_size: int


@dataclass
class Model:
    name: str
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


@dataclass
class TDictConfig:
    # basic parameters
    benchmark: Benchmark
    strategy: str
    replay: Replay
    model: Model
    optimizer: Optimizer
    training: Training
    seed: Optional[int]
    device: str
    device_id: int
    wandb_project: str
