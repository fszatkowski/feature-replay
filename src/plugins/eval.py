import omegaconf
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    confusion_matrix_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from config import Config


def get_eval_plugin(cfg: Config) -> EvaluationPlugin:
    run_name = f"{cfg.benchmark.name}_{cfg.strategy.name}_{cfg.model.name}"
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    interactive_logger = InteractiveLogger()
    wandb_logger = WandBLogger(
        project_name=cfg.wandb_project, run_name=run_name, config=cfg_dict
    )
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
            wandb=True,
            class_names=[str(i) for i in range(cfg.benchmark.n_classes)],
        ),
        loggers=[interactive_logger, wandb_logger],
    )

    return eval_plugin
