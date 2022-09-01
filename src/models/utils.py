from config import Config
from models.conv_mlp import ConvMLP
from models.feature_replay_model import FeatureReplayModel
from models.mlp import MLP
from models.resnet import ResNet18


def get_model(cfg: Config) -> FeatureReplayModel:
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

    elif cfg.model.name == "ResNet":
        if cfg.benchmark.name != "CIFAR100":
            raise ValueError("ResNet can only be used on CIFAR100 benchmarks.")
        model = ResNet18(num_classes=cfg.benchmark.n_classes)

    else:
        raise NotImplementedError()

    return model
