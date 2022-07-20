from torch import Tensor, nn


class DenseLayer(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, activation: bool, dropout_ratio: float
    ):
        """
        Single dense hidden layer with ReLU activation and dropout.
        :param in_size: Input size.
        :param out_size: Output size.
        :param activation: Whether to add ReLU activation.
        :param dropout_ratio: Dropout ratio. 0 to disable.
        """
        super().__init__()
        layers = [nn.Dropout(p=dropout_ratio), nn.Linear(in_size, out_size)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
