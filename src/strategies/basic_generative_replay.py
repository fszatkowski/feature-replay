from typing import List, Optional

import torch
from avalanche.training.plugins import (
    EvaluationPlugin,
    GenerativeReplayPlugin,
    SupervisedPlugin,
    TrainGeneratorAfterExpPlugin,
)
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.nn import Module
from torch.optim import Optimizer


class BasicGenerativeReplay(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: Optional[torch.nn.Module],
        train_mb_size: int,
        train_epochs: int,
        eval_mb_size: int,
        device=str,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        generator_strategy: BaseTemplate = None,
        replay_mb_size: int = None,
        increasing_replay_size: bool = False,
        **base_kwargs
    ):
        """
        Creates an instance of Generative Replay Strategy
        for a solver-generator pair.

        :param model: The solver model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param generator_strategy: A trainable strategy with a generative model,
            which employs GenerativeReplayPlugin. Defaults to None.
        :param **base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        rp = GenerativeReplayPlugin(
            generator_strategy=generator_strategy,
            replay_size=replay_mb_size,
            increasing_replay_size=increasing_replay_size,
        )

        tgp = TrainGeneratorAfterExpPlugin()

        if plugins is None:
            plugins = [tgp, rp]
        else:
            plugins.append(tgp)
            plugins.append(rp)

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
            **base_kwargs
        )
