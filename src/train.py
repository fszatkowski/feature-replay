import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from omegaconf import OmegaConf

from benchmarks.ci import ClassIncrementalBenchmark
from config import Config
from models.utils import get_model
from strategies.utils import get_training_strategy

ROOT = Path(__file__).parent.parent
os.environ["WANDB_START_METHOD"] = "thread"


@hydra.main(
    config_path=str(ROOT / "config"), config_name="config.yaml", version_base="1.2"
)
def run_main(cfg: Config):
    run(cfg)


def run(cfg: Config) -> None:
    if cfg.wandb.enable:
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    benchmark = ClassIncrementalBenchmark(cfg)
    model = get_model(cfg, benchmark)
    strategy = get_training_strategy(cfg=cfg, benchmark=benchmark, model=model)

    for experience in benchmark.train_stream:
        strategy.train(experience)
        metrics = strategy.eval(benchmark.test_stream)

    if cfg.output_dir is None:
        output_dir = Path("results")
    else:
        output_dir = Path(cfg.output_dir)
    strategy_name = cfg.strategy.base
    if cfg.strategy.plugins:
        strategy_name = cfg.strategy.base + "_" + "_".join(sorted(cfg.strategy.plugins))
    output_dir = (
        output_dir / benchmark.name / strategy_name / time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    with output_dir.joinpath("config.yml").open("w+") as f:
        OmegaConf.save(cfg, f)
    with output_dir.joinpath("results.json").open("w+") as f:
        metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
        json.dump(metrics, f, indent=2)
    if cfg.save_model:
        model_path = output_dir / "model.pt"
        torch.save(model, str(model_path))


if __name__ == "__main__":
    run_main()
