import torch.nn
from avalanche.training.plugins import EWCPlugin, GDumbPlugin, LwFPlugin, ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.supervised import Cumulative, Naive
from avalanche.training.templates import SupervisedTemplate

from benchmarks.ci import ClassIncrementalBenchmark
from config import Config
from models.feature_replay_model import FeatureReplayModel
from plugins.eval import get_eval_plugin
from plugins.selective_freezing import RandomPartialFreezingPlugin
from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy

AVAILABLE_PLUGINS = ["replay", "gdumb", "ewc", "lwf", "rpf"]


def get_training_strategy(
    cfg: Config,
    benchmark: ClassIncrementalBenchmark,
    model: FeatureReplayModel,
) -> SupervisedTemplate:
    """
    Utility function to initialize training strategy.
    :param cfg: Experiment configuration.
    :param benchmark: Tested benchmark.
    :param model: Model to train.
    :return: Initialized class incremental strategy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = _get_optimizer(model, cfg)

    common_args = dict(
        criterion=criterion,
        optimizer=optimizer,
        train_epochs=cfg.benchmark.hparams.train_epochs,
        train_mb_size=cfg.benchmark.hparams.train_mb_size,
        eval_mb_size=cfg.benchmark.hparams.eval_mb_size,
        device=cfg.device,
        evaluator=get_eval_plugin(cfg),
    )

    if cfg.strategy.base == "Cumulative":
        strategy = Cumulative(model=model, **common_args)

    elif cfg.strategy.base == "Naive":
        plugins = []

        for plugin_name in cfg.strategy.plugins:
            not_implemented_plugins = []
            if plugin_name not in AVAILABLE_PLUGINS:
                not_implemented_plugins.append(plugin_name)
            if len(not_implemented_plugins) > 0:
                raise NotImplementedError(
                    f"Plugins: {not_implemented_plugins} are not supported"
                )

        if "replay" in cfg.strategy.plugins:
            if cfg.strategy.constant_memory:
                buffer = ClassBalancedBuffer(
                    max_size=cfg.strategy.memory_size,
                    adaptive_size=True,
                )
            else:
                buffer = ClassBalancedBuffer(
                    max_size=cfg.strategy.memory_size,
                    adaptive_size=False,
                    total_num_classes=benchmark.n_classes,
                )
            replay_plugin = ReplayPlugin(
                mem_size=cfg.strategy.memory_size,
                batch_size_mem=cfg.benchmark.hparams.replay_mb_size,
                storage_policy=buffer,
            )
            plugins.append(replay_plugin)
        if "gdumb" in cfg.strategy.plugins:
            if "replay" in cfg.strategy.plugins:
                raise ValueError("GDumb can't be used with replay plugin.")
            plugins.append(GDumbPlugin(mem_size=cfg.strategy.memory_size))
        if "ewc" in cfg.strategy.plugins:
            plugins.append(EWCPlugin(ewc_lambda=cfg.strategy.ewc_lambda))
        if "lwf" in cfg.strategy.plugins:
            plugins.append(
                LwFPlugin(
                    alpha=cfg.strategy.lwf_alpha,
                    temperature=cfg.strategy.lwf_temperature,
                )
            )
        if "rpf" in cfg.strategy.plugins:
            plugins.append(RandomPartialFreezingPlugin(probs=cfg.strategy.rpf_probs))

        strategy = Naive(model=model, plugins=plugins, **common_args)

    elif cfg.strategy.base == "FeatureBuffer":
        strategy = BufferedFeatureReplayStrategy(
            model=model,
            total_memory_size=cfg.strategy.memory_size,
            replay_mb_sizes=cfg.strategy.replay_mb_size,
            update_strategy=cfg.strategy.update_strategy,
            replay_slowdown=cfg.strategy.replay_slowdown,
            **common_args,
        )
    else:
        raise NotImplementedError(
            "Currently, only `Cumulative`, `Naive` and `FeatureBuffer` are supported as base "
            "strategies."
        )

    return strategy


def _get_optimizer(model: FeatureReplayModel, cfg: Config) -> torch.optim.Optimizer:
    optimizer: torch.optim.Optimizer
    if cfg.benchmark.hparams.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.benchmark.hparams.lr,
            momentum=cfg.benchmark.hparams.momentum,
            weight_decay=cfg.benchmark.hparams.l2,
        )
    elif cfg.benchmark.hparams.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.benchmark.hparams.lr,
            betas=(cfg.benchmark.hparams.b1, cfg.benchmark.hparams.b2),
        )
    else:
        raise NotImplementedError()

    return optimizer
