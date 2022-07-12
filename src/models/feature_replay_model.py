from abc import abstractmethod

from avalanche.models import BaseModel
from torch import Tensor, nn


class FeatureReplayModel(nn.Module, BaseModel):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, skip_first: int = 0, skip_last: int = 0) -> Tensor:
        ...

    @abstractmethod
    def get_features(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def n_layers(self) -> int:
        ...
