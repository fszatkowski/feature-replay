import os

import omegaconf
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    confusion_matrix_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import CSVLogger, InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from config import Config


def get_eval_plugin(cfg: Config) -> EvaluationPlugin:
    run_name = f"{cfg.benchmark.name}_{cfg.strategy.name}_{cfg.model.name}"
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    loggers = [InteractiveLogger(), CSVLogger(log_folder=cfg.output_dir)]

    if cfg.wandb.enable:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        wandb_logger = WandBLogger(
            project_name=cfg.wandb.project, run_name=run_name, config=cfg_dict
        )
        loggers.append(wandb_logger)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(
            stream=True,
            wandb=cfg.wandb.enable,
            class_names=[str(i) for i in range(cfg.benchmark.n_classes)],
        ),
        loggers=loggers,
    )

    return eval_plugin
