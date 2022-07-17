from torch import Tensor, nn


class DenseLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout_ratio: float):
        """
        Single dense hidden layer with ReLU activation and dropout.
        :param in_size: Input size.
        :param out_size: Output size.
        :param dropout_ratio: Dropout ratio. 0 to disable.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout_ratio),
            nn.Linear(in_size, out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
