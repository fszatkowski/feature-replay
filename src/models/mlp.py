from typing import Optional

import torch.nn as nn
from torch import Tensor

from models.feature_replay_model import FeatureReplayModel


class MLP(FeatureReplayModel):
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 28 * 28,
        hidden_sizes: Optional[list[int]] = None,
        dropout_ratio: float = 0.5,
    ):
        """
        Multi-Layer Perceptron with custom parameters and support for partial inference from
        intermediate layers.
        :param num_classes: Output size.
        :param input_size: Input size.
        :param hidden_sizes: Hidden layer sizes
        :param dropout_ratio: Dropout rate. 0 to disable.
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512]

        layers = nn.Sequential()
        for layer_idx, (in_size, out_size) in enumerate(
            zip([input_size] + hidden_sizes, hidden_sizes)
        ):
            layers.add_module(
                f"fc{layer_idx}",
                HiddenLayer(
                    in_size=in_size, out_size=out_size, dropout_ratio=dropout_ratio
                ),
            )
        layers.add_module("classifier", nn.Linear(hidden_sizes[-1], num_classes))
        self.layers = layers

    def forward(self, x: Tensor, skip_first: int = 0, skip_last: int = 0) -> Tensor:
        if len(self.layers) - skip_last - skip_first < 0:
            raise ValueError()

        x = x.contiguous()
        x = x.view(x.size(0), -1)

        for layer in self.layers[skip_first : len(self.layers) - skip_last]:
            x = layer(x)
        return x

    def get_features(self, x: Tensor) -> Tensor:
        return self.forward(x, skip_last=1)

    def n_layers(self) -> int:
        return len(self.layers)


class HiddenLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout_ratio: float):
        """
        Single dense hidden layer with ReLU activation and dropout.
        :param in_size: Input size.
        :param out_size: Output size.
        :param dropout_ratio: Dropout ratio. 0 to disable.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
