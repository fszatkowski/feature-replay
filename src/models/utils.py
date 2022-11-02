from benchmarks.ci import ClassIncrementalBenchmark
from config import Config
from models.conv_mlp import ConvMLP
from models.feature_replay_model import FeatureReplayModel
from models.mlp import MLP
from models.resnet import ResNet18, ResNet32, SlimResNet18


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

    elif cfg.benchmark.model.name == "ResNet18":
        model = ResNet18(
            num_classes=benchmark.n_classes,
        )

    elif cfg.benchmark.model.name == "SlimResNet18":
        model = SlimResNet18(
            num_classes=benchmark.n_classes,
        )

    elif cfg.benchmark.model.name == "ResNet32":
        model = ResNet32(
            num_classes=benchmark.n_classes,
        )

    else:
        raise NotImplementedError()

    return model
