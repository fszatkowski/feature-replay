from copy import deepcopy
from random import randint

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
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.scenarios.classification_scenario import ClassificationStream
from avalanche.benchmarks.utils import AvalancheSubset
from torchvision.transforms import Pad

from config import Config


class ClassIncrementalBenchmark:
    """
    Wrapper class for Class Incremental data stream.
    """

    def __init__(self, cfg: Config):
        if cfg.benchmark.dataset.name == "CIFAR100":
            scenario = SplitCIFAR100
            input_size = (3, 32, 32)
            train_transform = deepcopy(_default_cifar100_train_transform)
            eval_transform = deepcopy(_default_cifar100_eval_transform)
            fixed_class_order = list(range(100))
        elif cfg.benchmark.dataset.name == "PermutedMNIST":
            scenario = PermutedMNIST
            input_size = (1, 28, 28)
            train_transform = deepcopy(_default_mnist_train_transform)
            eval_transform = deepcopy(_default_mnist_eval_transform)
            fixed_class_order = None
        elif cfg.benchmark.dataset.name == "SplitFMNIST":
            scenario = SplitFMNIST
            input_size = (1, 28, 28)
            train_transform = deepcopy(_default_fmnist_train_transform)
            eval_transform = deepcopy(_default_fmnist_eval_transform)
            fixed_class_order = None
        elif cfg.benchmark.dataset.name == "SplitMNIST":
            scenario = SplitMNIST
            input_size = (1, 28, 28)
            train_transform = deepcopy(_default_mnist_train_transform)
            eval_transform = deepcopy(_default_mnist_eval_transform)
            fixed_class_order = None
        elif cfg.benchmark.dataset.name == "SplitOmniglot":
            scenario = SplitOmniglot
            input_size = (1, 105, 105)
            train_transform = deepcopy(_default_omniglot_train_transform)
            eval_transform = deepcopy(_default_omniglot_eval_transform)
            fixed_class_order = None
        elif cfg.benchmark.dataset.name == "SplitTinyImageNet":
            scenario = SplitTinyImageNet
            input_size = (3, 64, 64)
            train_transform = deepcopy(_default_tinyimagenet_train_transform)
            eval_transform = deepcopy(_default_imagenet_eval_transform)
            fixed_class_order = None
        else:
            raise NotImplementedError()

        padding = cfg.benchmark.dataset.padding
        if padding is not None:
            train_transform.transforms.append(Pad(padding, padding_mode="edge"))
            eval_transform.transforms.append(Pad(padding, padding_mode="edge"))
            input_size = cfg.benchmark.dataset.input_size

        if cfg.benchmark.dataset.augmentations:
            train_transform = eval_transform

        self.name = cfg.benchmark.name
        self.input_size = input_size

        kwargs = dict(
            n_experiences=cfg.benchmark.n_experiences,
            seed=cfg.seed,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        if cfg.benchmark.name == "PermutedMNIST":
            kwargs["fixed_class_order"] = fixed_class_order

        benchmark = scenario(**kwargs)
        train_sample_limit = cfg.benchmark.dataset.train_per_class_sample_limit
        test_sample_limit = cfg.benchmark.dataset.test_per_class_sample_limit
        if train_sample_limit is not None or test_sample_limit is not None:
            train_set = benchmark.original_train_dataset
            test_set = benchmark.original_test_dataset
            if train_sample_limit is not None:
                train_set = self._downsample_dataset(train_set, train_sample_limit)
            if test_sample_limit is not None:
                test_set = self._downsample_dataset(test_set, test_sample_limit)

            benchmark = nc_benchmark(
                train_dataset=train_set,
                test_dataset=test_set,
                n_experiences=cfg.benchmark.n_experiences,
                one_dataset_per_exp=False,
                task_labels=False,
            )

        self.benchmark = benchmark

    @property
    def train_stream(self) -> ClassificationStream:
        return self.benchmark.train_stream

    @property
    def test_stream(self) -> ClassificationStream:
        return self.benchmark.test_stream

    @property
    def n_classes(self) -> int:
        return self.benchmark.n_classes

    def _downsample_dataset(
        self, dataset: AvalancheSubset, n_examples_per_class: int
    ) -> AvalancheSubset:
        unique_classes = set(dataset.targets)
        class_to_indices: dict[int, list[int]] = {c: [] for c in unique_classes}

        while not all(
            len(v) == n_examples_per_class for v in class_to_indices.values()
        ):
            idx = randint(0, len(dataset))
            class_idx_list = class_to_indices[dataset.targets[idx]]
            if len(class_idx_list) < n_examples_per_class and idx not in class_idx_list:
                class_idx_list.append(idx)

        indices = [
            idx for class_indices in class_to_indices.values() for idx in class_indices
        ]
        return AvalancheSubset(
            dataset,
            indices,
        )
