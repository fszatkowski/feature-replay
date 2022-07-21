import random
from abc import ABC, abstractmethod
from typing import Sized, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from models.feature_replay_model import FeatureReplayModel


class DatasetSampler(ABC):
    def __init__(self, feature_level: int, device: str):
        """
        Replay data sampler.
        :param feature_level Level at which features at sampled. 0 means that raw examples are
        sampled, 1 means that we sampled features after first network layer etc.
        :param device: Device to use when running model inference.
        """
        self.feature_level = feature_level
        self.device = device

    def sample(
        self, sample_size: int, dataset: Dataset, model: FeatureReplayModel
    ) -> tuple[Tensor, Tensor]:
        """
        Sample examples from dataset at given feature level.
        :param sample_size: Number of replay examples sampled with `sample` method.
        :param dataset: Dataset to sample examples from.
        :param model: Model used for sampling.
        :return: Tuple containing sampled features and sampled labels.
        """
        assert len(cast(Sized, dataset)) >= sample_size, (
            "Dataset size should be greater then or equal to sample size,"
            f"but got dataset of size {len(cast(Sized, dataset))} with sample size {sample_size}"
        )
        assert self.feature_level <= model.n_layers(), (
            f"Cannot sample at feature level {self.feature_level} "
            f"from model with {model.n_layers()} layers."
        )

        return self._sample(sample_size=sample_size, dataset=dataset, model=model)

    @abstractmethod
    def _sample(
        self, sample_size: int, dataset: Dataset, model: FeatureReplayModel
    ) -> tuple[Tensor, Tensor]:
        ...


class RandomDatasetSampler(DatasetSampler):
    def __init__(self, feature_level: int, device: str):
        """
        Samples data randomly from new experience. Samples are not guaranteed to be evenly
        distributed across classes.
        """
        super().__init__(feature_level=feature_level, device=device)

    def _sample(
        self, sample_size: int, dataset: Dataset, model: FeatureReplayModel
    ) -> tuple[Tensor, Tensor]:
        sampled_features = []
        sampled_labels = []
        with torch.no_grad():
            model.eval()
            replay_data_indices = random.sample(
                range(len(cast(Sized, dataset))), sample_size
            )
            replay_dataset = Subset(dataset, replay_data_indices)
            skip_last = model.n_layers() - self.feature_level

            for batch in iter(replay_dataset):
                inputs, labels = batch[0], batch[1]
                if skip_last != model.n_layers():
                    inputs = inputs.unsqueeze(dim=0).to(self.device)
                    features = model(inputs, skip_last=skip_last).squeeze()
                else:
                    features = inputs
                sampled_features.append(features.cpu())
                sampled_labels.append(labels)
        return torch.stack(sampled_features), torch.tensor(sampled_labels)
