import torch.nn
from avalanche.training import EWC, CWRStar, GDumb, LwF
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer
from avalanche.training.supervised import JointTraining, Naive
from avalanche.training.templates import SupervisedTemplate

from config import Config
from models.feature_replay_model import FeatureReplayModel
from plugins.eval import get_eval_plugin
from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy


def get_training_strategy(
    cfg: Config,
    model: FeatureReplayModel,
) -> SupervisedTemplate:
    criterion = torch.nn.CrossEntropyLoss()
    if cfg.training.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            weight_decay=cfg.training.optimizer.l2,
        )
    else:
        raise NotImplementedError()

    common_args = dict(
        criterion=criterion,
        train_epochs=cfg.training.train_epochs,
        train_mb_size=cfg.training.train_mb_size,
        eval_mb_size=cfg.training.eval_mb_size,
        device=cfg.device,
        evaluator=get_eval_plugin(cfg),
    )

    if cfg.strategy.name == "JointTraining":
        strategy = JointTraining(model=model, optimizer=optimizer, **common_args)

    elif cfg.strategy.name == "Naive":
        strategy = Naive(model=model, optimizer=optimizer, **common_args)

    elif cfg.strategy.name == "BasicBuffer":
        replay_plugin = ReplayPlugin(
            mem_size=cfg.strategy.memory_size,
            batch_size_mem=cfg.strategy.replay_mb_size,
            storage_policy=ExperienceBalancedBuffer(max_size=cfg.strategy.memory_size),
        )
        strategy = Naive(
            model=model, optimizer=optimizer, plugins=[replay_plugin], **common_args
        )

    elif cfg.strategy.name == "FeatureBuffer":
        strategy = BufferedFeatureReplayStrategy(
            model=model,
            total_memory_size=cfg.strategy.memory_size,
            replay_mb_sizes=cfg.strategy.replay_mb_size,
            update_strategy=cfg.strategy.update_strategy,
            replay_slowdown=cfg.strategy.replay_slowdown,
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            l2=cfg.training.optimizer.l2,
            **common_args
        )

    elif cfg.strategy.name == "CWRStar":
        strategy = CWRStar(
            model=model,
            optimizer=optimizer,
            cwr_layer_name=None,  # type: ignore
            **common_args
        )

    elif cfg.strategy.name == "EWC":
        strategy = EWC(
            model=model,
            ewc_lambda=cfg.strategy.ewc_lambda,
            optimizer=optimizer,
            **common_args
        )

    elif cfg.strategy.name == "GDumb":
        strategy = GDumb(
            model=model,
            mem_size=cfg.strategy.memory_size,
            optimizer=optimizer,
            **common_args
        )

    elif cfg.strategy.name == "LwF":
        strategy = LwF(
            model=model,
            alpha=cfg.strategy.alpha,
            temperature=cfg.strategy.temperature,
            optimizer=optimizer,
            **common_args
        )

    else:
        raise NotImplementedError()

    return strategy
