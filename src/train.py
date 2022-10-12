import random
from pathlib import Path

import hydra
import numpy as np
import torch

from benchmarks.utils import get_benchmark
from config import Config
from models.utils import get_model
from strategies.utils import get_training_strategy

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

    benchmark = get_benchmark(cfg)

    model = get_model(cfg)
    strategy = get_training_strategy(cfg=cfg, model=model)

    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)

    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        model_path = output_dir / "model.pt"
        torch.save(model, str(model_path))


if __name__ == "__main__":
    run()
