from typing import List, Optional, Union

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
        replay_mb_sizes: Union[int, list[int]],
        update_strategy: str,
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
        :param update_strategy: Replay update strategy.
        :param replay_slowdown: Learning rate multiplier for layers below feature replay level.
        This update is slowed down to prevent changing earlier layers too fast.
        """
        optimizer = SGD(
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
            optimizer,
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
                replay_mb_sizes, n_layers=model.n_layers()
            ),
            probs=self.get_replay_probs(
                update_strategy=update_strategy,
                n_layers=model.n_layers(),
            ),
            clock=self.clock,
            device=device,
        )

    def _after_training_exp(self, **kwargs):
        self.buffers.after_training_exp(strategy=self)
        super()._after_training_exp(**kwargs)

    def make_optimizer(self):
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
            self.loss = 0

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
                if self.clock.train_exp_counter > 0:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] *= self.replay_slowdown
                self.mb_output = self.model(self.mb_x)

            self._after_forward(**kwargs)

            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

            self._after_training_iteration(**kwargs)

    @staticmethod
    def get_replay_memory_sizes(
        replay_memory_sizes: Union[int, list[int]], n_layers: int
    ) -> list[int]:
        if isinstance(replay_memory_sizes, int):
            return [replay_memory_sizes for _ in range(n_layers)]
        else:
            assert len(replay_memory_sizes) == n_layers
            return replay_memory_sizes

    @staticmethod
    def get_replay_mb_sizes(
        replay_mb_sizes: Union[int, list[int]],
        n_layers: int,
    ) -> list[int]:
        if isinstance(replay_mb_sizes, int):
            return [replay_mb_sizes for _ in range(n_layers)]
        else:
            assert len(replay_mb_sizes) == n_layers
            return replay_mb_sizes

    @staticmethod
    def get_replay_probs(
        update_strategy: str,
        n_layers: int,
    ) -> list[float]:
        """
        Computes replay probs for given strategy.
        :param update_strategy: Name of the update strategy. Either `linear`, `reverse_linear` or
        `geometric`. Use `geometric` with `geo_strategy_base=1` to obtain equal probs.
        :param n_layers: Number of model layers.
        :return: Replay probabilities.
        """
        weights: list[Union[int, float]]
        if update_strategy == "equal":
            probs = [1 / n_layers for _ in range(n_layers)]
        elif update_strategy == "linear_ascending":
            weights = [i + 1 for i in range(n_layers)]
            probs = [weight / sum(weights) for weight in weights]
        elif update_strategy == "linear_descending":
            weights = [i + 1 for i in range(n_layers)]
            probs = [weight / sum(weights) for weight in reversed(weights)]
        elif update_strategy == "geometric_ascending":
            weights = [2**i for i in range(n_layers)]
            probs = [weight / sum(weights) for weight in weights]
        elif update_strategy == "geometric_descending":
            weights = [0.5**i for i in range(n_layers)]
            probs = [weight / sum(weights) for weight in weights]
        else:
            raise NotImplementedError()

        return probs
