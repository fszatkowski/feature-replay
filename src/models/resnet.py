from typing import Optional

from avalanche.models.slim_resnet18 import BasicBlock
from torch import nn

from models.feature_replay_model import FeatureReplayModel


class ResNet(FeatureReplayModel):
    def __init__(
        self,
        num_classes: int,
        num_blocks: Optional[list[int]] = None,
        slim: bool = False,
    ):
        """
        Avalanche ResNet implementation adapted for feature replay, with defaults resulting in
        ResNet18 configuration.
        :param num_classes: Number of classes to use.
        :param num_blocks: Number of ResNet blocks to use per ResNet layer. Defaults to ResNet18.
        :param slim: Whether to use slim variant with 20 input channels or default with 64..
        """
        super(ResNet, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        if slim:
            nf = 20
        else:
            nf = 64
        self.in_planes = nf

        self.layers.add_module(
            "conv0",
            nn.Sequential(
                nn.Conv2d(3, nf * 1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(nf * 1),
            ),
        )
        self.layers.add_module(
            "resnet0", self._make_layer(nf * 1, num_blocks[0], stride=1)
        )
        self.layers.add_module(
            "resnet1", self._make_layer(nf * 2, num_blocks[1], stride=2)
        )
        self.layers.add_module(
            "resnet2", self._make_layer(nf * 4, num_blocks[2], stride=2)
        )
        self.layers.add_module(
            "resnet3",
            nn.Sequential(
                self._make_layer(nf * 8, num_blocks[3], stride=2),
                nn.AvgPool2d(4),
                nn.Flatten(),
            ),
        )
        self.layers.add_module(
            "classifier", nn.Linear(nf * 8 * BasicBlock.expansion, num_classes)
        )

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
