from typing import Optional

from torch import nn

from models.feature_replay_model import FeatureReplayModel
from models.layers.dense import DenseLayer


class MLP(FeatureReplayModel):
    def __init__(
        self,
        num_classes: int,
        input_size: int,
        hidden_sizes: Optional[list[int]] = None,
        dropout_ratio: float = 0.0,
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

        for layer_idx, (in_size, out_size) in enumerate(
            zip([input_size] + hidden_sizes, hidden_sizes)
        ):
            if layer_idx == 0:
                self.layers.add_module(
                    f"fc{layer_idx}",
                    nn.Sequential(
                        nn.Flatten(),
                        DenseLayer(
                            in_size=in_size,
                            out_size=out_size,
                            activation=True,
                            dropout_ratio=dropout_ratio,
                        ),
                    ),
                )
            else:
                self.layers.add_module(
                    f"fc{layer_idx}",
                    DenseLayer(
                        in_size=in_size,
                        out_size=out_size,
                        activation=True,
                        dropout_ratio=dropout_ratio,
                    ),
                )
        self.layers.add_module(
            "classifier",
            DenseLayer(
                in_size=hidden_sizes[-1],
                out_size=num_classes,
                activation=False,
                dropout_ratio=dropout_ratio,
            ),
        )
