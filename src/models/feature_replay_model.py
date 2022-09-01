from avalanche.models import BaseModel
from torch import Tensor, nn


class FeatureReplayModel(nn.Module, BaseModel):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()

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
