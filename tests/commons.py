from typing import cast

from hydra import compose, initialize

from config import Config
from train import run


def run_with_overrides(overrides: list[str]) -> None:
    overrides += [
        "benchmark.hparams.train_epochs=1",
        "benchmark.n_experiences=2",
        "benchmark.dataset.train_per_class_sample_limit=20",
        "benchmark.dataset.test_per_class_sample_limit=2",
        "wandb.enable=false",
    ]
    with initialize(version_base="1.2", config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)
        run(cast(Config, cfg))
