from torch import Tensor, nn


class ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        flatten: bool = False,
    ):
        """
        Single conv hidden layer with ReLU activation and dropout.
        :param in_channels: Input channels.
        :param out_channels: Output channels.
        :param kernel_size: Kernel size.
        :param stride:  Stride of the convolution
        :param flatten: Whether to flatten the output.
        """
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                output_padding=0
            ),
            nn.ReLU(inplace=True),
        ]

        if flatten:
            layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
