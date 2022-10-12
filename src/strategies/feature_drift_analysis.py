from typing import List, Optional

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins.ewc import EWCPlugin
from avalanche.training.plugins.lwf import LwFPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import TensorDataset

from models.feature_replay_model import FeatureReplayModel
from strategies.drift.buffer import DriftAnalysisBuffer


class FeatureDriftAnalysisStrategy(SupervisedTemplate):
    def __init__(
        self,
        model: FeatureReplayModel,
        criterion: Optional[torch.nn.Module],
        memory_size: int,
        n_experiences: int,
        n_classes: int,
        replay: bool,
        ewc_lambda: float,
        lwf_alpha: float,
        lwf_temperature: float,
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
        Strategy that saves original images along with the representations to analyse the feature
        drift.
        :param model: Any model implementing FeatureReplayModel interface.
        :param memory_size: Buffer size for drift evaluation.
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

        if ewc_lambda > 0:
            ewc = EWCPlugin(ewc_lambda, mode="separate")
            if plugins is None:
                plugins = [ewc]
            else:
                plugins.append(ewc)

        if lwf_alpha > 0:
            lwf = LwFPlugin(lwf_alpha, lwf_temperature)
            if plugins is None:
                plugins = [lwf]
            else:
                plugins.append(lwf)

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

        self.drift_buffer = DriftAnalysisBuffer(
            memory_size=memory_size,
            n_experiences=n_experiences,
            n_classes=n_classes,
            expanding_buffer=True,
            n_layers=model.n_layers(),
        )
        self.replay = replay

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)
        if self.replay:
            if len(self.drift_buffer.buffer[0]) > 0:
                x_ds, y_ds, t_ds = [], [], []
                for exp_id, buffer in self.drift_buffer.buffer.items():
                    for x, y in buffer:
                        x_ds.append(x)
                        y_ds.append(y)
                        t_ds.append(exp_id)
                x_ds = torch.stack(x_ds)
                y_ds = torch.tensor(y_ds)
                t_ds = torch.tensor(t_ds)
                buffer_dataset = AvalancheDataset(
                    TensorDataset(x_ds, y_ds), task_labels=t_ds
                )
                batch_size = self.dataloader._dl.dataloaders[0].batch_size
                self.dataloader = ReplayDataLoader(
                    self.adapted_dataset,
                    buffer_dataset,
                    oversample_small_tasks=True,
                    batch_size=batch_size,
                    batch_size_mem=batch_size,
                    shuffle=True,
                )

    def _after_training_exp(self, **kwargs):
        self.drift_buffer.after_training_exp(strategy=self)
        super()._after_training_exp(**kwargs)

    def _after_training_epoch(self, **kwargs):
        self.drift_buffer.after_training_epoch(strategy=self)
        super()._after_training_epoch(**kwargs)

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
