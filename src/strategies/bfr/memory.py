from abc import ABC, abstractmethod
from math import ceil

import torch
from torch.utils.data import Dataset

from models.feature_replay_model import FeatureReplayModel
from strategies.bfr.dataset_sampler import DatasetSampler


class ReplayMemory(ABC):
    def __init__(self, memory_size: int, sampler: DatasetSampler):
        """
        Memory for feature-level replay.
        :param memory_size: Size of the memory.
        :param sampler: Sampler used to update the memory.
        """
        self.memory_size = memory_size
        self.sampler = sampler
        self.features = torch.zeros(())
        self.labels = torch.zeros(())
        self.experience_ids = torch.zeros(())

    @abstractmethod
    def update(
        self, dataset: Dataset, model: FeatureReplayModel, experience_id: int
    ) -> None:
        ...


class FixedSizeReplayMemory(ReplayMemory):
    def __init__(self, memory_size: int, sampler: DatasetSampler, shuffle: bool = True):
        """
        Replay memory with fixed size. At each update, some samples from previous experiences are
        discarded.
        :param shuffle Whether to shuffler memory after each update.
        """
        super().__init__(memory_size=memory_size, sampler=sampler)
        self.shuffle = shuffle

    def update(
        self, dataset: Dataset, model: FeatureReplayModel, experience_id: int
    ) -> None:
        unique_experience_ids = self.experience_ids.unique()
        if self.features.shape == ():
            new_n_experiences = 1
        else:
            assert (
                experience_id not in self.experience_ids
            ), "Update should be only called when encountering new experiences."
            new_n_experiences = len(unique_experience_ids) + 1

        memory_per_experience = split_memory_between_experiences(
            self.memory_size, new_n_experiences
        )

        new_features, new_labels = self.sampler.sample(
            sample_size=memory_per_experience[-1], dataset=dataset, model=model
        )
        new_experience_ids = torch.tensor(
            [experience_id for _ in range(len(new_features))]
        )

        if self.features.shape == ():
            self.features = new_features
            self.labels = new_labels
            self.experience_ids = new_experience_ids
        else:
            remaining_indices = []
            for experience_id, experience_memory_size in zip(
                unique_experience_ids, memory_per_experience
            ):
                sample_indices = (
                    (self.experience_ids == experience_id).nonzero().squeeze()
                )
                remaining_indices.extend(
                    sample_indices[:experience_memory_size].tolist()
                )

            remaining_features = self.features[remaining_indices]
            remaining_labels = self.labels[remaining_indices]
            remaining_experience_ids = self.experience_ids[remaining_indices]

            self.features = torch.cat((remaining_features, new_features))
            self.labels = torch.cat((remaining_labels, new_labels))
            self.experience_ids = torch.cat(
                (remaining_experience_ids, new_experience_ids)
            )


def split_memory_between_experiences(memory_size: int, n_experiences: int) -> list[int]:
    """
    Generate memory sizes per experience so that all sizes sum to memory size and differ by 0 or 1.
    :param memory_size: Size of the memory.
    :param n_experiences: Number of experience between which the examples are split.
    :return: List containing number of samples per experience.
    """
    memory_split = [ceil(memory_size / n_experiences) for _ in range(n_experiences)]
    total_memory = sum(memory_split)
    excess_samples = total_memory - memory_size
    for i in range(n_experiences - 1, n_experiences - 1 - excess_samples, -1):
        memory_split[i] -= 1

    return memory_split
