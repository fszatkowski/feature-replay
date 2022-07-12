import random
from math import ceil
from typing import Iterator, Optional

import torch
from avalanche.core import SupervisedPlugin
from torch import Tensor
from torch.utils.data import DataLoader, Subset, TensorDataset

from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy


class BufferedFeatureReplayPlugin(SupervisedPlugin):
    def __init__(self, memory_size: int, batch_size: int, feature_level: int):
        """
        Buffer containing features for replay. After each experience, adds features obtained for
        this experience and balances the buffer so that each experience is represented by roughly
        the same number of examples (can vary if ratio of memory size to number of experiences is
        not integer).
        :param memory_size: Total size of the buffer containing all examples.
        :param feature_level: Level at which features are obtained - eg. 0 means standard buffer,
        1 means features obtained after running inference through the first layer etc.
        """
        super().__init__()
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.feature_level = feature_level

        self.experience_id_to_examples: dict[int, list[tuple[Tensor, Tensor]]] = {}
        self.current_data_loader_iterator: Optional[Iterator] = None
        self.main_data_loader_step = 0
        self.main_data_loader_replay_steps: set[int] = set()

    def after_training_iteration(
        self, strategy: BufferedFeatureReplayStrategy, *args, **kwargs
    ) -> None:
        """Replay data on selected training steps"""
        # If DataLoader is None no buffer was yet obtained
        if self.current_data_loader_iterator is None:
            return

        if self.main_data_loader_step in self.main_data_loader_replay_steps:
            # Run manual optimization step
            features, labels = next(self.current_data_loader_iterator)
            features, labels = features.to(strategy.device), labels.to(strategy.device)

            strategy.model.train()
            outputs = strategy.model(features, skip_first=self.feature_level)
            loss = strategy._criterion(outputs, labels)

            strategy.optimizer.zero_grad()
            loss.backward()
            strategy.optimizer.step()

        self.main_data_loader_step += 1

    def after_training_exp(
        self, strategy: BufferedFeatureReplayStrategy, *args, **kwargs
    ) -> None:
        """Update the buffer"""
        dataset = strategy.experience.dataset
        buffer_size_per_experience = ceil(
            self.memory_size / (len(self.experience_id_to_examples) + 1)
        )
        experience_id = len(self.experience_id_to_examples)

        # Drop examples from previous experiences to keep the buffer size constant
        self.experience_id_to_examples = {
            exp_id: exp_examples[:buffer_size_per_experience]
            for exp_id, exp_examples in self.experience_id_to_examples.items()
        }

        # Sample features from last experience
        new_features = []
        new_labels = []
        with torch.no_grad():
            strategy.model.eval()
            replay_data_indices = random.sample(
                range(len(dataset)), buffer_size_per_experience
            )
            replay_dataset = Subset(dataset, replay_data_indices)
            skip_last = strategy.model.n_layers() - self.feature_level

            for batch in iter(replay_dataset):
                inputs, labels = batch[0], batch[1]
                if skip_last != strategy.model.n_layers():
                    inputs = inputs.unsqueeze(dim=0).to(strategy.device)
                    features = strategy.model(inputs, skip_last=skip_last)
                else:
                    features = inputs
                new_features.append(features)
                new_labels.append(labels)
        self.experience_id_to_examples[experience_id] = [
            (features, labels) for features, labels in zip(new_features, new_labels)
        ]

        # Drop single examples from the latest experiences if total memory size is exceeded
        total_examples = sum(
            len(examples) for _, examples in self.experience_id_to_examples.items()
        )
        n_examples_over_memory_size = total_examples - self.memory_size
        for i in range(n_examples_over_memory_size):
            self.experience_id_to_examples[
                experience_id - i
            ] = self.experience_id_to_examples[experience_id - i][:-1]

    def before_training_epoch(
        self, strategy: BufferedFeatureReplayStrategy, *args, **kwargs
    ) -> None:
        """
        Create Iterator over replay DataLoader after each epoch.
        """
        # If examples buffer is empty, it means we are at experience 0 so return
        if len(self.experience_id_to_examples) == 0:
            return

        # Generate DataLoader from buffer
        replay_features = [
            example[0]
            for examples in self.experience_id_to_examples.values()
            for example in examples
        ]
        replay_labels = [
            torch.tensor(example[1])
            for examples in self.experience_id_to_examples.values()
            for example in examples
        ]
        replay_dataset = TensorDataset(
            torch.stack(replay_features, dim=0), torch.stack(replay_labels, dim=0)
        )
        replay_data_loader = DataLoader(
            replay_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Reinitialize DataLoader for the next epoch
        self.current_data_loader_iterator = iter(replay_data_loader)
        # Sample training steps at which to run replay and zero training step counter
        # `strategy.dataloader` is dataloader used to sample training examples in the main loop
        self.main_data_loader_replay_steps = set(
            random.sample(range(len(strategy.dataloader)), len(replay_data_loader))
        )
        self.main_data_loader_step = 0
