from benchmarks.ci import ClassIncrementalBenchmark
from config import Config
from models.conv_mlp import ConvMLP
from models.feature_replay_model import FeatureReplayModel
from models.mlp import MLP
from models.resnet import ResNet


def get_model(cfg: Config, benchmark: ClassIncrementalBenchmark) -> FeatureReplayModel:
    if cfg.benchmark.model.name == "MLP":
        input_size = 1
        for size in benchmark.input_size:
            input_size *= size
        model = MLP(
            input_size=input_size,
            num_classes=benchmark.n_classes,
            hidden_sizes=cfg.benchmark.model.hidden_sizes,
            dropout_ratio=cfg.benchmark.model.dropout_ratio,
        )

    elif cfg.benchmark.model.name == "ConvMLP":
        model = ConvMLP(
            input_size=benchmark.input_size,
            num_classes=benchmark.n_classes,
            channels=cfg.benchmark.model.channels,
            kernel_size=cfg.benchmark.model.kernel_size,
            pooling=cfg.benchmark.model.pooling,
            hidden_sizes=cfg.benchmark.model.hidden_sizes,
            dropout_ratio=cfg.benchmark.model.dropout_ratio,
        )

    elif cfg.benchmark.model.name == "ResNet":
        if cfg.benchmark.dataset.name != "CIFAR100":
            raise ValueError("ResNet can only be used on CIFAR100 benchmarks.")
        model = ResNet(
            num_classes=benchmark.n_classes,
            num_blocks=cfg.benchmark.model.num_blocks,
            slim=cfg.benchmark.model.slim,
        )

    else:
        raise NotImplementedError()

    return model
