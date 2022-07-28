import random
from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor

from strategies.bfr.memory import ReplayMemory


class MemorySampler(ABC):
    def __init__(
        self,
        memory_size: int,
        batch_size: int,
    ):
        """
        This class handles sampling from replay memory.
        :param memory_size: Size of the memory from which the sampling is done.
        :param batch_size: Batch size to use when sampling.
        """
        assert batch_size <= memory_size, (
            f"Batch size cannot exceed memory size, "
            f"but got batch size: {batch_size} while the memory size is {memory_size}"
        )

        self.memory_size = memory_size
        self.batch_size = batch_size

    def sample_batch_from_memory(self, memory: ReplayMemory) -> tuple[Tensor, Tensor]:
        """
        Wrapper method to directly sample batch from memory instead of using separate features,
        labels and experience_ids.
        :param memory: Memory from which the batch is sampled.
        :return Sampled batch of features and labels.
        """
        return self.sample_batch(
            features=memory.features,
            labels=memory.labels,
            experience_ids=memory.experience_ids,
        )

    def sample_batch(
        self,
        features: Tensor,
        labels: Tensor,
        experience_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample batch of features and labels from given memory.
        :param features: Tensor containing features. Should be of size [M, ...] where M is memory
        size.
        :param labels: Tensor containing labels. Should be of size [M, ...] where M is memory
        size.
        :param experience_ids: Tensor containing experience_ids of the examples. Should be of size
        [M, ...] where M is memory size.
        :return Sampled batch of features and labels.
        """

        assert len(features) == self.memory_size, (
            f"Size of the features memory should match sampler memory size, "
            f"but got features shape: {features.shape} and sampler memory size {self.memory_size}."
        )
        assert len(labels) == self.memory_size, (
            f"Size of the labels memory should match sampler memory size, "
            f"but got labels shape: {labels.shape} and sampler memory size {self.memory_size}."
        )
        if experience_ids is not None and experience_ids.shape:
            assert len(experience_ids) == self.memory_size, (
                f"Size of the experience_ids memory should match sampler memory size, "
                f"but got experience_ids shape: {experience_ids.shape} "
                f"and sampler memory size {self.memory_size}."
            )

        return self._sample_batch(
            features=features, labels=labels, experience_ids=experience_ids
        )

    @abstractmethod
    def _sample_batch(
        self,
        features: Tensor,
        labels: Tensor,
        experience_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset sampler state. Should be called after each training epoch if the sampler has any
        state (eg. if sampler tracks class / task ids of replayed samples).
        """
        ...


class RandomMemorySampler(MemorySampler):
    """
    Memory sampler that performs random sampling, but guarantees that each example can only be seen
    once until all other examples were sampled.
    """

    def __init__(self, memory_size: int, batch_size: int):
        super().__init__(memory_size, batch_size)
        self.current_permutation = random.sample(
            range(self.memory_size), self.memory_size
        )
        self.current_idx = 0

    def _sample_batch(
        self,
        features: Tensor,
        labels: Tensor,
        experience_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_indices = self.current_permutation[
            self.current_idx : self.current_idx + self.batch_size
        ]
        self.current_idx += self.batch_size

        if len(batch_indices) < self.batch_size:
            size_diff = self.batch_size - len(batch_indices)
            self.reset()
            batch_indices += self.current_permutation[:size_diff]
            self.current_idx += size_diff

        return features[batch_indices], labels[batch_indices]

    def reset(self) -> None:
        self.current_permutation = random.sample(
            range(self.memory_size), self.memory_size
        )
        self.current_idx = 0
