from pathlib import Path

import hydra
import torch
from avalanche.benchmarks import SplitCIFAR100, SplitMNIST
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.supervised import Naive

from config import Config
from models.mlp import MLP
from plugins.eval import get_eval_plugin

ROOT = Path(__file__).parent.parent


@hydra.main(
    config_path=str(ROOT / "config"), config_name="config.yaml", version_base="1.2"
)
def run(cfg: Config):
    if cfg.benchmark.name == "CIFAR100":
        benchmark = SplitCIFAR100(
            n_experiences=cfg.benchmark.n_experiences, seed=cfg.seed
        )
    elif cfg.benchmark.name == "SplitMNIST":
        benchmark = SplitMNIST(n_experiences=cfg.benchmark.n_experiences, seed=cfg.seed)
    else:
        raise NotImplementedError()

    if cfg.model.name == "MLP":
        model = MLP(
            input_size=cfg.benchmark.input_size,
            num_classes=cfg.benchmark.n_classes,
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
            mem_size=cfg.strategy.buffer_size,
            storage_policy=ReservoirSamplingBuffer(max_size=cfg.strategy.buffer_size),
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
    else:
        raise NotImplementedError()

    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    run()
