from typing import TYPE_CHECKING, Iterator

from torch import Tensor
from torch.utils.data import DataLoader

from strategies.bfr.storage_policy import ExperienceBalancedFeatureBuffer

if TYPE_CHECKING:
    from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy


class FeatureReplayBuffer:
    def __init__(self, memory_size: int, feature_level: int, batch_size: int):
        """
        Buffer containing features for replay. After each experience, can add features obtained for
        this experience and balances the buffer so that each experience is represented by roughly
        the same number of examples (can vary if ratio of memory size to number of experiences is
        not integer).
        :param memory_size: Memory size for buffer.
        :param feature_level: Feature level at which the examples will be sampled.
        :param batch_size: Batch size to sample.
        """
        self.memory_size = memory_size
        self.feature_level = feature_level
        self.batch_size = batch_size

        self.buffer = ExperienceBalancedFeatureBuffer(
            max_size=memory_size, feature_level=feature_level
        )
        self.dataloader: DataLoader
        self.dl_iterator: Iterator

    def after_training_exp(self, strategy: "BufferedFeatureReplayStrategy") -> None:
        """
        Update the buffer on the last experience.
        """
        self.buffer.update(strategy)
        self.dataloader = DataLoader(
            self.buffer.buffer, batch_size=self.batch_size, shuffle=True
        )
        self.dl_iterator = iter(self.dataloader)

    def sample(self) -> tuple[Tensor, Tensor]:
        """
        Return random batch from buffer.
        """
        try:
            x, y, t, e = next(self.dl_iterator)
        except StopIteration:
            self.dl_iterator = iter(self.dataloader)
            x, y, t, e = next(self.dl_iterator)
        return x, y
