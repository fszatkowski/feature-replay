from typing import Optional

from torch import Tensor

from models.feature_replay_model import FeatureReplayModel
from models.layers.conv import ConvLayer
from models.layers.dense import DenseLayer


class ConvMLP(FeatureReplayModel):
    def __init__(
        self,
        num_classes: int,
        input_size: tuple[int, int, int],
        channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        pooling: bool = False,
        hidden_sizes: Optional[list[int]] = None,
        dropout_ratio: float = 0.5,
    ):
        """
        Sequential network with Convolutional layers followed by Multi-Layer Perceptron with
        custom parameters and support for partial inference from intermediate layers.
        :param num_classes: Output size.
        :param input_size: Input size.
        :param channels: List of channels for convolutional layers.
        :param kernel_size: Kernel size for convolutional layers.
        :param pooling: Whether to apply pooling after each conv channel.
        :param hidden_sizes: Hidden layer sizes
        :param dropout_ratio: Dropout rate. 0 to disable.
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512]
        if channels is None:
            channels = [64]

        for layer_idx, (in_channels, out_channels) in enumerate(
            zip([input_size[0]] + channels, channels)
        ):
            self.layers.add_module(
                f"conv{layer_idx}",
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pooling=pooling,
                    dropout_ratio=dropout_ratio,
                    flatten=layer_idx == (len(channels) - 1),
                ),
            )

        hidden_input_size = input_size[1] * input_size[2] * channels[-1]
        if pooling:
            hidden_input_size = int(hidden_input_size / 4 ** len(channels))
        for layer_idx, (in_size, out_size) in enumerate(
            zip([hidden_input_size] + hidden_sizes, hidden_sizes)
        ):
            self.layers.add_module(
                f"fc{layer_idx}",
                DenseLayer(
                    in_size=in_size, out_size=out_size, dropout_ratio=dropout_ratio
                ),
            )
        self.layers.add_module(
            "classifier",
            DenseLayer(
                in_size=hidden_sizes[-1],
                out_size=num_classes,
                dropout_ratio=dropout_ratio,
            ),
        )

    def forward(self, x: Tensor, skip_first: int = 0, skip_last: int = 0) -> Tensor:
        if len(self.layers) - skip_last - skip_first < 0:
            raise ValueError()

        used_layers = self.layers[skip_first : len(self.layers) - skip_last]
        if len(used_layers) > 0:
            for layer in used_layers:
                x = layer(x)
        return x

    def get_features(self, x: Tensor) -> Tensor:
        return self.forward(x, skip_last=1)

    def n_layers(self) -> int:
        return len(self.layers)
