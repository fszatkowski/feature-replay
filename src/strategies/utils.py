import torch.nn
from avalanche.training import AR1, LwF, EWC, CWRStar
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

    if cfg.strategy.name == "Naive":
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    elif cfg.strategy.name == "BasicBuffer":
        replay_plugin = ReplayPlugin(
            mem_size=cfg.strategy.memory_size,
            batch_size_mem=cfg.strategy.replay_mb_size,
            storage_policy=ExperienceBalancedBuffer(max_size=cfg.strategy.memory_size),
        )
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            plugins=[replay_plugin],
            evaluator=get_eval_plugin(cfg),
        )

    elif cfg.strategy.name == "JointTraining":
        strategy = JointTraining(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    elif cfg.strategy.name == "FeatureBuffer":
        strategy = BufferedFeatureReplayStrategy(
            model=model,
            replay_memory_sizes=cfg.strategy.memory_size,
            replay_mb_sizes=cfg.strategy.replay_mb_size,
            replay_probs=cfg.strategy.replay_prob,
            replay_slowdown=cfg.strategy.replay_slowdown,
            criterion=criterion,
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            l2=cfg.training.optimizer.l2,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    elif cfg.strategy.name == "AR1":
        strategy = AR1(
            criterion=criterion,
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            l2=cfg.training.optimizer.l2,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    elif cfg.strategy.name == "LwF":
        strategy = LwF(
            model=model,
            alpha=cfg.strategy.alpha,
            temperature=cfg.strategy.temperature,
            criterion=criterion,
            optimizer=optimizer,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    elif cfg.strategy.name == "EWC":
        strategy = EWC(
            model=model,
            ewc_lambda=cfg.strategy.ewc_lambda,
            criterion=criterion,
            optimizer=optimizer,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    elif cfg.strategy.name == "CWRStar":
        strategy = CWRStar(
            model=model,
            ewc_lambda=cfg.strategy.ewc_lambda,
            criterion=criterion,
            optimizer=optimizer,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    else:
        raise NotImplementedError()

    return strategy
