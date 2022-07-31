import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from avalanche.training.plugins.clock import Clock
from torch import Tensor

from strategies.bfr.buffer import FeatureReplayBuffer

if TYPE_CHECKING:
    from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy


@dataclass
class FeatureReplaySamplingResult:
    replay: bool
    feature_level: int = 0
    features: Tensor = torch.zeros(())
    labels: Tensor = torch.zeros(())


class FeatureReplayManager(ABC):
    def __init__(self, buffers: list[FeatureReplayBuffer], clock: Clock):
        """
        This class manages replay from feature buffers.
        :param buffers: List of replay buffers to manage.
        :param clock: Strategy Clock.
        """
        self.buffers = buffers
        self.clock = clock

    def after_training_exp(self, strategy: "BufferedFeatureReplayStrategy") -> None:
        pass
        for buffer in self.buffers:
            buffer.after_training_exp(strategy=strategy)

    @abstractmethod
    def step(self) -> FeatureReplaySamplingResult:
        """
        Returns the result of single manger step - whether to replay at all, at what level and what
        features. Should be used in every training step.
        """
        ...


class RandomFeatureReplayManager(FeatureReplayManager):
    def __init__(
        self,
        memory_sizes: list[int],
        batch_sizes: list[int],
        probs: list[float],
        clock: Clock,
        device: str,
        memory_size_incremental_per_class: bool = False,
    ):
        assert len(memory_sizes) == len(batch_sizes) == len(probs)
        assert sum(probs) <= 1

        self.device = device

        buffers = []
        replay_probs = []
        for feature_level, (memory_size, batch_size, prob) in enumerate(
            zip(memory_sizes, batch_sizes, probs)
        ):
            if memory_size == 0:
                continue
            if batch_size == 0 or prob == 0:
                logging.warning(
                    f"Batch size or replay probability set to 0 for feature level {feature_level} "
                    f"with non-zero memory size. Skipping buffer creation for this level. "
                    f"This is likely a bug."
                )
                continue

            if memory_size_incremental_per_class:
                # TODO implement incremental memory
                raise NotImplementedError()
            else:
                buffer = FeatureReplayBuffer(
                    memory_size=memory_size,
                    feature_level=feature_level,
                    batch_size=batch_size,
                )

                buffers.append(buffer)
                replay_probs.append(prob)

        total_replay_prob = sum(replay_probs)
        no_replay_prob = 1.0 - total_replay_prob
        replay_probs.append(no_replay_prob)
        self.replay_probs = replay_probs
        self.no_replay_idx = len(replay_probs) - 1

        super().__init__(buffers=buffers, clock=clock)

    def step(self) -> FeatureReplaySamplingResult:
        if self.clock.train_exp_counter == 0:
            return FeatureReplaySamplingResult(replay=False)
        else:
            replay_choice = np.random.choice(
                np.arange(len(self.replay_probs)), p=self.replay_probs
            )
            if replay_choice == self.no_replay_idx:
                return FeatureReplaySamplingResult(replay=False)

            buffer = self.buffers[replay_choice]
            features, labels = buffer.sample()
            return FeatureReplaySamplingResult(
                replay=True,
                feature_level=buffer.feature_level,
                features=features.to(self.device),
                labels=labels.to(self.device),
            )
