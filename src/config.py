from dataclasses import dataclass
from typing import Optional, Union


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

    # ResNet
    num_blocks: list[int]
    slim: bool


@dataclass
class TrainingHyperParameters:
    # Training params
    train_epochs: int
    train_mb_size: int
    replay_mb_size: Optional[int]
    eval_mb_size: int

    # Optimizer
    optimizer: str
    lr: float
    b1: float
    b2: float
    momentum: float
    l2: float


@dataclass
class Dataset:
    """
    Dataset parameters
    """

    name: str
    augmentations: bool
    padding: Optional[Union[int, tuple[int, int], tuple[int, int, int, int]]]
    input_size: tuple[int, int, int]
    train_per_class_sample_limit: Optional[int]
    test_per_class_sample_limit: Optional[int]


@dataclass
class Benchmark:
    """
    Benchmark to use.
    """

    name: str
    dataset: Dataset
    n_experiences: int
    model: Model
    hparams: TrainingHyperParameters


@dataclass
class Strategy:
    """
    Parameters of continual learning strategy
    """

    base: str
    plugins: list[str]

    # Memory size for strategies using replay
    memory_size: int
    constant_memory: bool  # If True, memory_size is per experience, otherwise it's constant

    # EWC
    ewc_lambda: float

    # LwF
    lwf_alpha: float
    lwf_temperature: float

    # RandomPartialFreezing
    rpf_probs: list[float]


@dataclass
class Wandb:
    """
    Wandb configuration
    """

    enable: bool
    entity: str
    project: str
    tags: Optional[list[str]]
    name: Optional[str]


@dataclass
class Config:
    """
    Experiment configuration
    """

    benchmark: Benchmark
    strategy: Strategy
    wandb: Wandb

    seed: Optional[int]
    device: str
    device_id: int
    output_dir: Optional[str]
    save_model: bool
