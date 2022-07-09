from pathlib import Path

import hydra
import torch
from avalanche.benchmarks import SplitCIFAR100, SplitMNIST
from avalanche.training.supervised import Naive

from config import TDictConfig
from plugins.eval import get_eval_plugin
from models.mlp import MLP

ROOT = Path(__file__).parent.parent


@hydra.main(
    config_path=str(ROOT / "config"), config_name="config.yaml", version_base="1.2"
)
def run(cfg: TDictConfig):
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

    if cfg.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum
        )
    else:
        raise NotImplementedError()
    criterion = torch.nn.CrossEntropyLoss()

    if cfg.strategy == "Naive":
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
    elif cfg.strategy == "BasicBufferedReplay":
        ...
    else:
        raise NotImplementedError()

    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    run()
