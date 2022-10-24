from typing import Iterator, List, Optional

import numpy as np
import torch
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.utils.data import DataLoader

from models.feature_replay_model import FeatureReplayModel


class RandomPartialFreezingStrategy(SupervisedTemplate):
    def __init__(
        self,
        model: FeatureReplayModel,
        freezing_probs: list[float],
        memory_size: int,
        constant_memory_size: bool,
        n_classes: Optional[int],
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
        Partial freezing strategy.
        :param model: Any model implementing FeatureReplayModel interface.
        :param freezing_probs: Probabilities for freezing the network at each layer. Should sum
        to 1 and have size equal to the number of model layers.
        :param memory_size: Size of the buffer.
        :param constant_memory_size: If True, memory size will be constant and equally divided
        over all experiences seen so far. Otherwise, memory size will be equally divided between
        each class.
        :param n_classes: Number of classes, required if constant_memory_size is False.
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
        assert abs(sum(freezing_probs) - 1.0) < 0.01, (
            f"Freezing probs should sum to 1, " f"but sum to {sum(freezing_probs)}."
        )
        assert len(freezing_probs) == model.n_layers(), (
            "Freezing probs should be defined for " "each layer."
        )

        self.probs = freezing_probs
        if constant_memory_size:
            buffer = ClassBalancedBuffer(
                max_size=memory_size,
                adaptive_size=True,
            )
        else:
            buffer = ClassBalancedBuffer(
                max_size=memory_size,
                adaptive_size=False,
                total_num_classes=n_classes,
            )
        self.buffer = buffer
        self.replay_mb_size = train_mb_size
        self.replay_dataloader: Optional[DataLoader] = None
        self.replay_dl_iterator: Optional[Iterator] = None

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        self.buffer.update(strategy=self)
        self.replay_dataloader = DataLoader(
            self.buffer.buffer, batch_size=self.replay_mb_size, shuffle=True
        )
        self.replay_dl_iterator = iter(self.replay_dataloader)

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            self.optimizer.zero_grad()

            self.loss = 0

            self._before_forward(**kwargs)

            # Get output for real data
            x_data, y_data, t_data = self.mbatch[0], self.mbatch[1], self.mbatch[2]
            p_data = self.model(x_data)

            # Get output for data from memory if buffer was initialized
            if self.replay_dataloader is not None:
                try:
                    mbatch_mem = next(self.replay_dl_iterator)
                except StopIteration:
                    self.replay_dl_iterator = iter(self.replay_dataloader)
                    mbatch_mem = next(self.replay_dl_iterator)

                for i in range(len(mbatch_mem)):
                    mbatch_mem[i] = mbatch_mem[i].to(self.device)
                x_mem, y_mem, t_mem = mbatch_mem[0], mbatch_mem[1], mbatch_mem[2]

                # Randomly select layer at which the network will be frozen
                skip_layer_idx = np.random.choice(
                    range(self.model.n_layers()), 1, p=self.probs
                ).item()

                with torch.no_grad():
                    f_mem = self.model(
                        x_mem, skip_last=self.model.n_layers() - skip_layer_idx
                    )
                p_mem = self.model(f_mem, skip_first=skip_layer_idx)

                self.mbatch[0] = torch.cat([x_data, x_mem])
                self.mbatch[1] = torch.cat([y_data, y_mem])
                self.mbatch[2] = torch.cat([t_data, t_mem])
                self.mb_output = torch.cat([p_data, p_mem])
            else:
                self.mb_output = p_data

            self._after_forward(**kwargs)

            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
