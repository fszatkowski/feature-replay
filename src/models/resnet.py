from avalanche.models.slim_resnet18 import BasicBlock
from torch import nn

from models.feature_replay_model import FeatureReplayModel


class ResNet(FeatureReplayModel):
    def __init__(
        self,
        num_classes: int,
        num_blocks: list[int],
        in_planes: int,
        pool_size: int,
    ):
        """
        Avalanche ResNet implementation adapted for feature replay, with defaults resulting in
        ResNet18 configuration.
        :param num_classes: Number of classes to use.
        :param num_blocks: Number of ResNet blocks to use per ResNet layer. Defaults to ResNet18.
        :param in_planes: Number of feature maps.
        :param pool_size: Pooling kernel size for last layer.
        """
        super(ResNet, self).__init__()

        self.in_planes = in_planes

        self.layers.add_module(
            "conv0",
            nn.Sequential(
                nn.Conv2d(
                    3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(),
            ),
        )
        planes = self.in_planes
        for i in range(len(num_blocks)):
            if i == 0:
                stride = 1
            else:
                stride = 2

            if i == len(num_blocks) - 1:
                module = nn.Sequential(
                    self._make_layer(planes, num_blocks[i], stride=stride),
                    nn.AvgPool2d(pool_size),
                    nn.Flatten(),
                )
            else:
                module = self._make_layer(planes, num_blocks[i], stride=stride)
                planes = 2 * planes

            self.layers.add_module(f"resnet{i}", module)

        self.layers.add_module(
            "classifier",
            nn.Linear(planes * BasicBlock.expansion, num_classes),
        )

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)


class ResNet18(ResNet):
    def __init__(self, num_classes: int):
        super().__init__(
            num_classes=num_classes, num_blocks=[2, 2, 2, 2], in_planes=64, pool_size=4
        )


class SlimResNet18(ResNet):
    def __init__(self, num_classes: int):
        super().__init__(
            num_classes=num_classes, num_blocks=[2, 2, 2, 2], in_planes=20, pool_size=4
        )


class ResNet32(ResNet):
    def __init__(self, num_classes: int):
        super().__init__(
            num_classes=num_classes, num_blocks=[5, 5, 5], in_planes=16, pool_size=8
        )
