from typing import List, Optional, Union

import torch
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate

from models.feature_replay_model import FeatureReplayModel
from strategies.bfr.buffer_manager import RandomFeatureReplayManager


class BufferedFeatureReplayStrategy(SupervisedTemplate):
    def __init__(
        self,
        model: FeatureReplayModel,
        update_strategy: str,
        total_memory_size: int,
        replay_mb_sizes: Union[int, list[int]],
        replay_slowdown: float,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[torch.nn.Module],
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
        :param update_strategy: Replay update strategy.
        :param total_memory_size: Buffer sizes for each feature level in model. If set to 0,
        buffer is not created for given level.
        :param replay_mb_sizes: Replay minibatch sizes for each layer. Defaults to `train_mb_size`
        for each feature level.
        :param replay_slowdown: Learning rate multiplier for layers below feature replay level.
        This update is slowed down to prevent changing earlier layers too fast.
        """
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
        probs, memory_sizes = self.get_replay_params(
            update_strategy=update_strategy,
            total_memory_size=total_memory_size,
            n_layers=model.n_layers(),
        )
        self.buffers = RandomFeatureReplayManager(
            memory_sizes=memory_sizes,
            batch_sizes=self.get_replay_mb_sizes(
                replay_mb_sizes, n_layers=model.n_layers()
            ),
            probs=probs,
            clock=self.clock,
            device=device,
        )

    def _after_training_exp(self, **kwargs):
        self.buffers.after_training_exp(strategy=self)
        super()._after_training_exp(**kwargs)

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
    def get_replay_params(
        update_strategy: str,
        total_memory_size: int,
        n_layers: int,
    ) -> tuple[list[float], list[int]]:
        """
        Computes replay probs for given strategy.
        :param update_strategy: Name of the update strategy.
        :param total_memory_size: Memory size to distribute across strategies.
        :param n_layers: Number of model layers.
        :return: Replay probabilities.
        """
        weights: list[Union[int, float]]
        if update_strategy == "equal":
            probs = [1 / n_layers for _ in range(n_layers)]
        elif update_strategy == "only_first_layer":
            probs = [0.0 for _ in range(n_layers)]
            probs[0] = 1.0
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

        memory_sizes = [int(p * total_memory_size) for p in probs]
        return probs, memory_sizes
