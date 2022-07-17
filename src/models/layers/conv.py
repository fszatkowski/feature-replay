from torch import Tensor, nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pooling: bool,
        dropout_ratio: float,
        flatten: bool = False,
    ):
        """
        Single conv hidden layer with ReLU activation and dropout.
        :param in_channels: Input channels.
        :param out_channels: Output channels.
        :param kernel_size: Kernel size.
        :param pooling: Whether to apply max pooling. Without this option, preserves the size.
        :param dropout_ratio: Dropout ratio. 0 to disable.
        :param flatten: Whether to flatten the output.
        """
        super().__init__()
        layers = [
            nn.Dropout(p=dropout_ratio),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.ReLU(inplace=True),
        ]
        if pooling:
            layers.append(nn.MaxPool2d(kernel_size=2))
        if flatten:
            layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
