import os
import time

import omegaconf
import wandb
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import CSVLogger, InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from config import Config


def get_eval_plugin(cfg: Config) -> EvaluationPlugin:
    strategy_name = cfg.strategy.base
    if cfg.strategy.plugins:
        strategy_name = "_" + "_".join(sorted(cfg.strategy.plugins))
    run_name = f"{cfg.benchmark.name}_{strategy_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    loggers = [InteractiveLogger(), CSVLogger(log_folder=cfg.output_dir)]

    if cfg.wandb.enable:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        params = {}
        if cfg.wandb.tags is not None:
            params["tags"] = cfg.wandb.tags
        wandb_logger = WandBLogger(
            project_name=cfg.wandb.project, run_name=run_name, config=cfg_dict, params=params
        )
        loggers.append(wandb_logger)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=False,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        loggers=loggers,
    )

    return eval_plugin
