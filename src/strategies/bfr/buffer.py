from torch import Tensor
from torch.utils.data import Dataset

from models.feature_replay_model import FeatureReplayModel
from strategies.bfr.memory import ReplayMemory
from strategies.bfr.memory_sampler import MemorySampler


class FeatureReplayBuffer:
    def __init__(
        self,
        memory: ReplayMemory,
        memory_sampler: MemorySampler,
    ):
        """
        Buffer containing features for replay. After each experience, can add features obtained for
        this experience and balances the buffer so that each experience is represented by roughly
        the same number of examples (can vary if ratio of memory size to number of experiences is
        not integer).
        :param memory: Memory for buffer.
        :param memory_sampler: Sampler used to get replayed batches.
        """
        super().__init__()
        self.memory = memory
        self.memory_sampler = memory_sampler

    @property
    def feature_level(self) -> int:
        return self.memory.sampler.feature_level

    @property
    def memory_size(self) -> int:
        return self.memory.memory_size

    @property
    def batch_size(self) -> int:
        return self.memory_sampler.batch_size

    def after_training_exp(
        self, dataset: Dataset, model: FeatureReplayModel, experience_id: int
    ) -> None:
        """
        Update the buffer on the last experience.
        """
        self.memory.update(dataset, model, experience_id=experience_id)

    def before_training_epoch(self) -> None:
        """
        Reset the state of memory sampler at the start of each epoch.
        """
        self.memory_sampler.reset()

    def sample(self) -> tuple[Tensor, Tensor]:
        """
        Return random batch from buffer.
        """
        return self.memory_sampler.sample_batch_from_memory(memory=self.memory)
