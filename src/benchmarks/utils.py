from typing import Union

from avalanche.benchmarks import (
    SplitCIFAR100,
    SplitMNIST,
    SplitOmniglot,
    SplitTinyImageNet,
)
from avalanche.benchmarks.classic.ccifar100 import (
    _default_cifar100_eval_transform,
    _default_cifar100_train_transform,
)
from avalanche.benchmarks.classic.cmnist import (
    PermutedMNIST,
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
    if cfg.benchmark.name == "CIFAR100":
        if cfg.benchmark.augmentations:
            train_transform = _default_cifar100_train_transform
        else:
            train_transform = _default_cifar100_eval_transform
        benchmark = SplitCIFAR100(
            n_experiences=cfg.benchmark.n_experiences,
            seed=cfg.seed,
            train_transform=train_transform,
        )

    elif cfg.benchmark.name == "SplitMNIST":
        if cfg.model.pooling:
            _default_mnist_train_transform.transforms.append(
                Pad(2, padding_mode="edge")
            )
            _default_mnist_eval_transform.transforms.append(Pad(2, padding_mode="edge"))
            cfg.benchmark.input_size = (1, 32, 32)
        benchmark = SplitMNIST(n_experiences=cfg.benchmark.n_experiences, seed=cfg.seed)

    elif cfg.benchmark.name == "SplitOmniglot":
        if cfg.model.pooling:
            _default_omniglot_train_transform.transforms.append(
                Pad((11, 11, 12, 12), padding_mode="edge")
            )
            _default_omniglot_eval_transform.transforms.append(
                Pad((11, 11, 12, 12), padding_mode="edge")
            )
            cfg.benchmark.input_size = (1, 128, 128)
        benchmark = SplitOmniglot(
            n_experiences=cfg.benchmark.n_experiences, seed=cfg.seed
        )

    elif cfg.benchmark.name == "SplitTinyImageNet":
        if cfg.benchmark.augmentations:
            train_transform = _default_tinyimagenet_train_transform
        else:
            train_transform = _default_imagenet_eval_transform
        benchmark = SplitTinyImageNet(
            n_experiences=cfg.benchmark.n_experiences,
            seed=cfg.seed,
            train_transform=train_transform,
        )

    else:
        raise NotImplementedError()

    return benchmark


def get_benchmark_for_joint_training(cfg: Config) -> TBenchmark:
    if cfg.benchmark.name == "CIFAR100":
        if cfg.benchmark.augmentations:
            train_transform = _default_cifar100_train_transform
        else:
            train_transform = _default_cifar100_eval_transform
        benchmark = SplitCIFAR100(
            n_experiences=1,
            seed=cfg.seed,
            train_transform=train_transform,
        )

    elif cfg.benchmark.name == "SplitMNIST":
        if cfg.model.pooling:
            _default_mnist_train_transform.transforms.append(
                Pad(2, padding_mode="edge")
            )
            _default_mnist_eval_transform.transforms.append(Pad(2, padding_mode="edge"))
            cfg.benchmark.input_size = (1, 32, 32)
        benchmark = SplitMNIST(n_experiences=1, seed=cfg.seed)

    elif cfg.benchmark.name == "PermutedMNIST":
        if cfg.model.pooling:
            _default_mnist_train_transform.transforms.append(
                Pad(2, padding_mode="edge")
            )
            _default_mnist_eval_transform.transforms.append(Pad(2, padding_mode="edge"))
            cfg.benchmark.input_size = (1, 32, 32)
        benchmark = PermutedMNIST(n_experiences=1, seed=cfg.seed)

    elif cfg.benchmark.name == "SplitOmniglot":
        if cfg.model.pooling:
            _default_omniglot_train_transform.transforms.append(
                Pad((11, 11, 12, 12), padding_mode="edge")
            )
            _default_omniglot_eval_transform.transforms.append(
                Pad((11, 11, 12, 12), padding_mode="edge")
            )
            cfg.benchmark.input_size = (1, 128, 128)
        benchmark = SplitOmniglot(n_experiences=1, seed=cfg.seed)

    elif cfg.benchmark.name == "SplitTinyImageNet":
        if cfg.benchmark.augmentations:
            train_transform = _default_tinyimagenet_train_transform
        else:
            train_transform = _default_imagenet_eval_transform
        benchmark = SplitTinyImageNet(
            n_experiences=1,
            seed=cfg.seed,
            train_transform=train_transform,
        )

    else:
        raise NotImplementedError()

    return benchmark
