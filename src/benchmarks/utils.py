import logging
from typing import Union

from avalanche.benchmarks import (
    PermutedMNIST,
    SplitCIFAR100,
    SplitFMNIST,
    SplitMNIST,
    SplitOmniglot,
    SplitTinyImageNet,
)
from avalanche.benchmarks.classic.ccifar100 import (
    _default_cifar100_eval_transform,
    _default_cifar100_train_transform,
)
from avalanche.benchmarks.classic.cfashion_mnist import (
    _default_fmnist_eval_transform,
    _default_fmnist_train_transform,
)
from avalanche.benchmarks.classic.cmnist import (
    _default_mnist_eval_transform,
    _default_mnist_train_transform,
)
from avalanche.benchmarks.classic.comniglot import (
    _default_omniglot_eval_transform,
    _default_omniglot_train_transform,
)
from avalanche.benchmarks.classic.ctiny_imagenet import (
    _default_eval_transform as _default_imagenet_eval_transform,
)
from avalanche.benchmarks.classic.ctiny_imagenet import (
    _default_train_transform as _default_tinyimagenet_train_transform,
)
from torchvision.transforms import Pad

from config import Config

TBenchmark = Union[SplitCIFAR100, SplitMNIST, SplitOmniglot, SplitTinyImageNet]


def get_benchmark(cfg: Config) -> TBenchmark:
    if cfg.strategy.name == "JointTraining":
        n_experiences = 1
    else:
        n_experiences = cfg.benchmark.n_experiences

    if cfg.benchmark.name == "CIFAR100":
        benchmark_class = SplitCIFAR100
        train_transform = _default_cifar100_train_transform
        eval_transform = _default_cifar100_eval_transform
    elif cfg.benchmark.name == "PermutedMNIST":
        benchmark_class = PermutedMNIST
        train_transform = _default_mnist_train_transform
        eval_transform = _default_mnist_eval_transform
    elif cfg.benchmark.name == "SplitFMNIST":
        benchmark_class = SplitFMNIST
        train_transform = _default_fmnist_train_transform
        eval_transform = _default_fmnist_eval_transform
    elif cfg.benchmark.name == "SplitMNIST":
        benchmark_class = SplitMNIST
        train_transform = _default_mnist_train_transform
        eval_transform = _default_mnist_eval_transform
    elif cfg.benchmark.name == "SplitOmniglot":
        benchmark_class = SplitOmniglot
        train_transform = _default_omniglot_train_transform
        eval_transform = _default_omniglot_eval_transform
    elif cfg.benchmark.name == "SplitTinyImageNet":
        benchmark_class = SplitTinyImageNet
        train_transform = _default_tinyimagenet_train_transform
        eval_transform = _default_imagenet_eval_transform
    else:
        raise NotImplementedError()

    model_has_pooling_layers = hasattr(cfg.model, "pooling") and cfg.model.pooling
    benchmark_requires_padding = cfg.benchmark.name in [
        "PermutedMNIST",
        "SplitFMNIST",
        "SplitMNIST",
        "SplitOmniglot",
    ]
    if model_has_pooling_layers and benchmark_requires_padding:
        logging.warning(
            "Using ConvMLP with pooling layers requires input sizes to be powers of "
            "2. Datasets will be padded to required size."
        )

        padding: Union[int, tuple[int, int, int, int]]
        if cfg.benchmark.name in [
            "PermutedMNIST",
            "SplitFMNIST",
            "SplitMNIST",
        ]:
            padding = 2
            input_size = (1, 32, 32)
        elif cfg.benchmark.name == "SplitOmniglot":
            padding = (11, 11, 12, 12)
            input_size = (1, 128, 128)
        else:
            raise NotImplementedError()

        train_transform.transforms.append(Pad(padding, padding_mode="edge"))
        eval_transform.transforms.append(Pad(padding, padding_mode="edge"))
        cfg.benchmark.input_size = input_size

    if cfg.benchmark.augmentations:
        train_transform = eval_transform

    return benchmark_class(
        n_experiences=n_experiences, seed=cfg.seed, train_transform=train_transform
    )
