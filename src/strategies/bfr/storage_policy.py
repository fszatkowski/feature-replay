from typing import TYPE_CHECKING

import torch
from avalanche.benchmarks import AvalancheSubset
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training import BalancedExemplarsBuffer, ExemplarsBuffer
from torch.utils.data import TensorDataset

if TYPE_CHECKING:
    from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy


"""
Modified code samples from `avalanche.training.storage_policy` that allow buffers on feature level.
"""


class ExperienceBalancedFeatureBuffer(BalancedExemplarsBuffer):
    """Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(
        self,
        max_size: int,
        feature_level: int,
        adaptive_size: bool = True,
        num_experiences=None,
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param feature_level: level at which sample exemplars
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self.feature_level = feature_level

    def update(self, strategy: "BufferedFeatureReplayStrategy", **kwargs):
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = ReservoirFeatureSamplingBuffer(
            lens[-1], feature_level=self.feature_level
        )
        new_buffer.update(strategy)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)


class ReservoirFeatureSamplingBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int, feature_level: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)
        self.feature_level = feature_level

    def update(self, strategy: "BufferedFeatureReplayStrategy", **kwargs):
        """Update buffer."""
        new_data = strategy.experience.dataset
        new_weights = torch.rand(len(new_data))

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = AvalancheConcatDataset([new_data, self.buffer])  # type: ignore
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = AvalancheSubset(cat_data, buffer_idxs)

        x_ = []
        y_ = []
        t_ = []

        strategy.model.eval()
        with torch.no_grad():
            for x, y, t in self.buffer:
                x = x.unsqueeze(0).to(strategy.device)
                features = strategy.model(
                    x, skip_last=strategy.model.n_layers() - self.feature_level
                )
                x_.append(features.cpu())
                y_.append(y)
                t_.append(t)
        self.buffer = AvalancheDataset(
            TensorDataset(torch.cat(x_), torch.tensor(y_), torch.tensor(t_))
        )
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = AvalancheSubset(self.buffer, torch.arange(self.max_size))
        self._buffer_weights = self._buffer_weights[: self.max_size]
