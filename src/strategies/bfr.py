import random
from math import ceil
from typing import List, Optional, Union, cast

import torch
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import Subset

from models.feature_replay_model import FeatureReplayModel


class BufferedFeatureReplayStrategy(SupervisedTemplate):
    def __init__(
        self,
        model: FeatureReplayModel,
        replay_memory_sizes: Union[int, list[int]],
        replay_mb_sizes: Optional[Union[int, list[int]]] = None,
        replay_probs: Optional[Union[float, list[float]]] = None,
        replay_slowdown: float = 0.01,
        criterion: Optional[torch.nn.Module] = None,
        lr: float = 0.001,
        momentum=0.9,
        l2=0.0005,
        train_epochs: int = 4,
        train_mb_size: int = 128,
        eval_mb_size: int = 128,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
    ):
        """
        Buffered replay strategy done at multiple feature levels.
        :param model: Any model implementing FeatureReplayModel interface.
        :param replay_memory_sizes: Buffer sizes for each feature level in model. If set to 0,
        buffer is not created for given level.
        :param replay_mb_sizes: Replay minibatch sizes for each layer. Defaults to `train_mb_size`
        for each feature level.
        :param replay_probs: Replay probability for feature levels. At each training step, replay
        level will be sampled based on these probabilities. Sum of the probabilities can not exceed
        1., and if probabilities do not sum to 1., some training steps might skip replay.
        :param replay_slowdown: Learning rate multiplier for layers below feature replay level.
        This update is slowed down to prevent changing earlier layers too fast.
        """
        if plugins is None:
            plugins = []

        self.optimizer = SGD(
            [{"params": layer.parameters(), "lr": lr} for layer in model.layers],
            lr=lr,
            momentum=momentum,
            weight_decay=l2,
        )
        if criterion is None:
            criterion = CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2

        self.replay_memory_sizes = self.get_replay_memory_sizes(
            replay_memory_sizes, n_layers=model.n_layers()
        )
        self.replay_mb_sizes = self.get_replay_mb_sizes(
            replay_mb_sizes, n_layers=model.n_layers(), train_mb_size=train_mb_size
        )
        self.buffers = self.get_buffers()
        self.replay_probs = self.get_replay_probs(
            replay_probs,
            replay_memory_sizes=self.replay_memory_sizes,
            n_layers=model.n_layers(),
        )
        self.replay_slowdown = replay_slowdown

        super().__init__(
            model,
            self.optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

    @staticmethod
    def get_replay_memory_sizes(
        replay_memory_sizes: Union[int, list[int]], n_layers: int
    ) -> list[int]:
        if isinstance(replay_memory_sizes, int):
            replay_memory_sizes = [replay_memory_sizes for _ in range(n_layers)]
        else:
            assert len(replay_memory_sizes) == n_layers
            replay_memory_sizes = replay_memory_sizes
        return replay_memory_sizes

    @staticmethod
    def get_replay_mb_sizes(
        replay_mb_sizes: Optional[Union[int, list[int]]],
        n_layers: int,
        train_mb_size: int,
    ) -> list[int]:
        if replay_mb_sizes is not None:
            if isinstance(replay_mb_sizes, int):
                replay_mb_sizes = [replay_mb_sizes for _ in range(n_layers)]
            else:
                assert len(replay_mb_sizes) == n_layers
                replay_mb_sizes = replay_mb_sizes
        else:
            replay_mb_sizes = [train_mb_size for _ in range(n_layers)]
        return replay_mb_sizes

    def get_buffers(self) -> list[Optional["FeatureBuffer"]]:
        buffers: list[Optional["FeatureBuffer"]] = []
        for feature_level, (memory_size, batch_size) in enumerate(
            zip(self.replay_memory_sizes, self.replay_mb_sizes)
        ):
            if memory_size > 0:
                buffers.append(
                    FeatureBuffer(
                        strategy=self,
                        memory_size=memory_size,
                        batch_size=batch_size,
                        feature_level=feature_level,
                    )
                )
            else:
                buffers.append(None)
        return buffers

    @staticmethod
    def get_replay_probs(
        replay_probs: Optional[Union[float, list[float]]],
        replay_memory_sizes: list[int],
        n_layers: int,
    ) -> list[Optional[float]]:
        if replay_probs is not None:
            if isinstance(replay_probs, float):
                output_replay_probs = [
                    replay_probs if mem_size != 0 else None
                    for mem_size in replay_memory_sizes
                ]
            else:
                assert len(cast(list[float], replay_probs)) == n_layers
                output_replay_probs = [
                    prob if mem_size != 0 else None
                    for prob, mem_size in zip(
                        cast(list[float], replay_probs), replay_memory_sizes
                    )
                ]
        else:
            tmp_probs = [
                1.0 if mem_size != 0 else None for mem_size in replay_memory_sizes
            ]
            norm_factor = sum(prob for prob in tmp_probs if prob is not None)
            output_replay_probs = [
                prob / norm_factor if prob is not None else None for prob in tmp_probs
            ]

        probs_sum = sum(p for p in output_replay_probs if p is not None)
        assert 0.0 <= probs_sum <= 1.0, (
            f"Sum of replay probabilities should be between 0 and 1, but with "
            f"`replay_probs`={replay_probs} obtained `self.replay_probs`={output_replay_probs} "
            f"which sum to {probs_sum}."
        )

        return output_replay_probs

    def _before_training_exp(self, **kwargs):
        # TODO AR1 reinitializes model and optimizer, do we also have to do this ?
        if self.clock.train_exp_counter > 0:
            for feature_buffer in self.buffers:
                feature_buffer.update()

        super()._before_training_exp(**kwargs)

    def make_optimizer(self):
        # TODO without this override make_optimizer crashes the strategy
        self.optimizer = SGD(
            [
                {"params": layer.parameters(), "lr": self.lr}
                for layer in self.model.layers
            ],
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.l2,
        )

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()

            self._before_forward(**kwargs)
            latent_minibatch = self.sample_latent_minibatch()
            if latent_minibatch is not None:
                feature_level, latent_features, latent_labels = latent_minibatch
                latent_features, latent_labels = latent_features.to(
                    self.device
                ), latent_labels.to(self.device)
                mb_latent_features = self.model(
                    self.mb_x, skip_last=self.model.n_layers() - feature_level
                )
                mb_latent_features = torch.cat([mb_latent_features, latent_features])
                self.mb_output = self.model(
                    mb_latent_features, skip_first=feature_level
                )
                mb_y = torch.cat([self.mb_y, latent_labels])
                self.mbatch[1] = mb_y

                for param_group in self.optimizer.param_groups[:feature_level]:
                    param_group["lr"] *= self.replay_slowdown

            else:
                self.mb_output = self.model(self.mb_x)

            self._after_forward(**kwargs)

            self.loss = self._criterion(self.mb_output, self.mb_y)
            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

            self._after_training_iteration(**kwargs)

    def sample_latent_minibatch(self) -> Union[tuple[int, Tensor, Tensor], None]:
        # Only return sample if buffer is not empty
        if self.clock.train_exp_counter > 0:
            sample = random.random()
            current_cumulative_sum = 0.0
            for i, replay_prob in enumerate(self.replay_probs):
                if replay_prob is None:
                    continue
                else:
                    current_cumulative_sum += replay_prob
                    if sample <= current_cumulative_sum:
                        if self.buffers[i] is not None:
                            features, labels = cast(
                                FeatureBuffer, self.buffers[i]
                            ).sample()
                            return i, features, labels
                        else:
                            raise ValueError("Buffer sampled for replay is None.")
        return None


class FeatureBuffer:
    def __init__(
        self,
        strategy: BufferedFeatureReplayStrategy,
        memory_size: int,
        batch_size: int,
        feature_level: int,
    ):
        """
        Buffer containing features for replay. After each experience, can add features obtained for
        this experience and balances the buffer so that each experience is represented by roughly
        the same number of examples (can vary if ratio of memory size to number of experiences is
        not integer).
        :param memory_size: Total size of the buffer containing all examples.
        :param batch_size: Size of the batch sampled from the buffer.
        :param feature_level: Level at which features are obtained - eg. 0 means standard buffer,
        1 means features obtained after running inference through the first layer etc.
        """
        super().__init__()
        self.strategy = strategy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.feature_level = feature_level

        self.experience_id_example_ids: dict[int, list[int]] = {}
        self.features = torch.zeros(())
        self.labels = torch.zeros(())

    def update(self) -> None:
        """Update the buffer"""
        if self.strategy.clock.train_exp_counter > 0:
            dataset = self.strategy.experience.dataset
            model = self.strategy.model
            device = self.strategy.device

            current_experience_id = len(self.experience_id_example_ids)
            memory_size_per_experience = ceil(
                self.memory_size / (current_experience_id + 1)
            )

            # Drop examples from previous experiences to keep the buffer size constant
            new_memory_sizes = [
                memory_size_per_experience for _ in range(current_experience_id + 1)
            ]
            size_idx = len(new_memory_sizes) - 1
            while sum(new_memory_sizes) > self.memory_size:
                new_memory_sizes[size_idx] -= 1

            new_features = []
            new_labels = []
            for experience_id, memory_size in enumerate(new_memory_sizes[:-1]):
                kept_indices = self.experience_id_example_ids[experience_id][
                    :memory_size
                ]
                new_features.append(self.features[kept_indices])
                new_labels.append(self.labels[kept_indices])
                self.experience_id_example_ids[experience_id] = kept_indices

            # Sample features from last experience
            new_exp_features = []
            new_exp_labels = []
            with torch.no_grad():
                model.eval()
                replay_data_indices = random.sample(
                    range(len(dataset)), new_memory_sizes[-1]
                )
                replay_dataset = Subset(dataset, replay_data_indices)
                skip_last = model.n_layers() - self.feature_level

                for batch in iter(replay_dataset):
                    inputs, labels = batch[0], batch[1]
                    if skip_last != model.n_layers():
                        inputs = inputs.unsqueeze(dim=0).to(device)
                        features = model(inputs, skip_last=skip_last).squeeze()
                    else:
                        features = inputs
                    new_exp_features.append(features)
                    new_exp_labels.append(labels)
            self.experience_id_example_ids[current_experience_id] = list(
                range(sum(new_memory_sizes[:-1]), self.memory_size)
            )

            new_features.append(torch.stack(new_exp_features))
            new_labels.append(torch.tensor(new_exp_labels))
            self.features = torch.cat(new_features)
            self.labels = torch.cat(new_labels)

    def sample(self) -> tuple[Tensor, Tensor]:
        """
        Return random batch from buffer.
        """
        if len(self.experience_id_example_ids) == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_indices = random.sample(range(self.memory_size), self.batch_size)
        return self.features[batch_indices], self.labels[batch_indices]
