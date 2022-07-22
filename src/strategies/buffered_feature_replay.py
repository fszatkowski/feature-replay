from typing import List, Optional, Union, cast

import torch
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from models.feature_replay_model import FeatureReplayModel
from strategies.bfr.buffer_manager import RandomFeatureReplayManager


class BufferedFeatureReplayStrategy(SupervisedTemplate):
    def __init__(
        self,
        model: FeatureReplayModel,
        replay_memory_sizes: Union[int, list[int]],
        replay_mb_sizes: Optional[Union[int, list[int]]],
        replay_probs: Optional[Union[float, list[float]]],
        replay_slowdown: float,
        criterion: Optional[torch.nn.Module],
        lr: float,
        momentum: float,
        l2: float,
        train_epochs: int,
        train_mb_size: int,
        eval_mb_size: int,
        device: str,
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

        self.replay_slowdown = replay_slowdown
        self.buffers = RandomFeatureReplayManager(
            memory_sizes=self.get_replay_memory_sizes(
                replay_memory_sizes, n_layers=model.n_layers()
            ),
            batch_sizes=self.get_replay_mb_sizes(
                replay_mb_sizes, n_layers=model.n_layers(), train_mb_size=train_mb_size
            ),
            probs=self.get_replay_probs(
                replay_probs,
                replay_memory_sizes=replay_memory_sizes,
                n_layers=model.n_layers(),
            ),
            clock=self.clock,
            device=device,
        )

    def _after_training_exp(self, **kwargs):
        self.buffers.after_training_exp(
            dataset=self.experience.dataset,
            model=self.model,
        )

        super()._after_training_exp(**kwargs)

    def _before_training_epoch(self, **kwargs):
        self.buffers.before_training_epoch()

        super()._before_training_epoch(**kwargs)

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
            replay_sample = self.buffers.step()
            if replay_sample.replay:
                mb_latent_features = self.model(
                    self.mb_x,
                    skip_last=self.model.n_layers() - replay_sample.feature_level,
                )
                mb_latent_features = torch.cat(
                    [mb_latent_features, replay_sample.features]
                )
                self.mb_output = self.model(
                    mb_latent_features, skip_first=replay_sample.feature_level
                )
                mb_y = torch.cat([self.mb_y, replay_sample.labels])
                self.mbatch[1] = mb_y

                for param_group in self.optimizer.param_groups[
                    : replay_sample.feature_level
                ]:
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

    @staticmethod
    def get_replay_probs(
        replay_probs: Optional[Union[float, list[float]]],
        replay_memory_sizes: Union[int, list[int]],
        n_layers: int,
    ) -> list[float]:
        replay_memory_sizes = BufferedFeatureReplayStrategy.get_replay_memory_sizes(
            replay_memory_sizes, n_layers
        )

        if replay_probs is not None:
            if isinstance(replay_probs, float):
                output_replay_probs = [
                    replay_probs if mem_size != 0 else 0
                    for mem_size in replay_memory_sizes
                ]
            else:
                assert len(cast(list[float], replay_probs)) == n_layers
                output_replay_probs = [
                    prob if mem_size != 0 else 0
                    for prob, mem_size in zip(
                        cast(list[float], replay_probs), replay_memory_sizes
                    )
                ]
        else:
            tmp_probs = [
                1.0 if mem_size != 0 else 0 for mem_size in replay_memory_sizes
            ]
            norm_factor = sum(prob for prob in tmp_probs)
            output_replay_probs = [
                prob / norm_factor if prob is not None else None for prob in tmp_probs
            ]

        probs_sum = sum(p for p in output_replay_probs)
        assert 0.0 <= probs_sum <= 1.0, (
            f"Sum of replay probabilities should be between 0 and 1, but with "
            f"`replay_probs`={replay_probs} obtained `self.replay_probs`={output_replay_probs} "
            f"which sum to {probs_sum}."
        )

        return output_replay_probs
