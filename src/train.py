import random
from pathlib import Path

import hydra
import numpy as np
import torch
from avalanche.training import GDumb
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer
from avalanche.training.supervised import JointTraining, Naive

from benchmarks.utils import get_benchmark, get_benchmark_for_joint_training
from config import Config
from models.conv_mlp import ConvMLP
from models.mlp import MLP
from plugins.eval import get_eval_plugin
from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy

ROOT = Path(__file__).parent.parent


@hydra.main(
    config_path=str(ROOT / "config"), config_name="config.yaml", version_base="1.2"
)
def run(cfg: Config):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    if cfg.strategy.name != "JointTraining":
        benchmark = get_benchmark(cfg)
    else:
        benchmark = get_benchmark_for_joint_training(cfg)

    if cfg.model.name == "MLP":
        input_size = 1
        for size in cfg.benchmark.input_size:
            input_size *= size
        model = MLP(
            input_size=input_size,
            num_classes=cfg.benchmark.n_classes,
            hidden_sizes=cfg.model.hidden_sizes,
            dropout_ratio=cfg.model.dropout_ratio,
        )

    elif cfg.model.name == "ConvMLP":
        model = ConvMLP(
            input_size=cfg.benchmark.input_size,
            num_classes=cfg.benchmark.n_classes,
            channels=cfg.model.channels,
            kernel_size=cfg.model.kernel_size,
            pooling=cfg.model.pooling,
            hidden_sizes=cfg.model.hidden_sizes,
            dropout_ratio=cfg.model.dropout_ratio,
        )

    else:
        raise NotImplementedError()

    if cfg.training.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            weight_decay=cfg.training.optimizer.l2,
        )
    else:
        raise NotImplementedError()
    criterion = torch.nn.CrossEntropyLoss()

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

    elif cfg.strategy.name == "GDumb":
        strategy = GDumb(
            model=model,
            mem_size=cfg.strategy.memory_size,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=cfg.training.train_epochs,
            train_mb_size=cfg.training.train_mb_size,
            eval_mb_size=cfg.training.eval_mb_size,
            device=cfg.device,
            evaluator=get_eval_plugin(cfg),
        )

    else:
        raise NotImplementedError()

    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)

    if cfg.output_model_path is not None:
        Path(cfg.output_model_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(model, cfg.output_model_path)


if __name__ == "__main__":
    run()
