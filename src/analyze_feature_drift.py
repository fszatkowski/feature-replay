import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from benchmarks.utils import get_benchmark
from config import Config
from models.utils import get_model
from strategies.drift.plot_drift import save_drift_plot
from strategies.feature_drift_analysis import FeatureDriftAnalysisStrategy

ROOT = Path(__file__).parent.parent


@hydra.main(
    config_path=str(ROOT / "config"),
    config_name="feature_drift.yaml",
    version_base="1.2",
)
def run(cfg: Config):
    if cfg.output_model_path is not None and Path(cfg.output_model_path).exists():
        logging.info(f"Model at {cfg.output_model_path} already exists, skipping training.")
        exit()

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    benchmark = get_benchmark(cfg)

    model = get_model(cfg)

    strategy = FeatureDriftAnalysisStrategy(
        model=model,
        replay=cfg.replay,
        ewc_lambda=cfg.ewc_lambda,
        memory_size=cfg.strategy.memory_size,
        n_experiences=benchmark.n_experiences,
        n_classes=benchmark.n_classes,
        criterion=torch.nn.CrossEntropyLoss(),
        lr=cfg.training.optimizer.lr,
        momentum=cfg.training.optimizer.momentum,
        l2=cfg.training.optimizer.l2,
        train_epochs=cfg.training.train_epochs,
        train_mb_size=cfg.training.train_mb_size,
        eval_mb_size=cfg.training.eval_mb_size,
        device=cfg.device,
        # evaluator=get_eval_plugin(cfg),
        # plugins=[replay_plugin]
    )

    metrics = []
    for experience in benchmark.train_stream:
        strategy.train(experience)
        metrics.append(strategy.eval(benchmark.test_stream))

    if cfg.output_model_path is not None:
        output_dir = Path(cfg.output_model_path).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        drift_stats = strategy.drift_buffer.per_exp_drift

        with output_dir.joinpath("drift_stats.json").open("w+") as f:
            json.dump(drift_stats, f, indent=2)
        with output_dir.joinpath("metrics.json").open("w+") as f:
            json.dump(metrics, f, indent=2)
        with output_dir.joinpath("config.yml").open("w+") as f:
            OmegaConf.save(cfg, f)
        save_drift_plot(
            drift_stats,
            output_dir / "drift_plot.png",
            benchmark=cfg.benchmark.name,
            model_name=cfg.model.name,
            replay=cfg.replay,
            ewc_lambda=cfg.ewc_lambda,
            memory_size=cfg.strategy.memory_size,
        )
        torch.save(model, cfg.output_model_path)


if __name__ == "__main__":
    run()
