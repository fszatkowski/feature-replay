import torch

from models.resnet import ResNet18


def test_resnet_skip_first_inference() -> None:
    num_classes = 10
    batch_size = 8
    inputs = [
        torch.rand((batch_size, 3, 32, 32)),
        torch.rand((batch_size, 20, 32, 32)),
        torch.rand((batch_size, 20, 32, 32)),
        torch.rand((batch_size, 40, 16, 16)),
        torch.rand((batch_size, 80, 8, 8)),
        torch.rand((batch_size, 160)),
    ]
    resnet = ResNet18(num_classes=num_classes)

    for skip_idx, input_vector in enumerate(inputs):
        y = resnet(input_vector, skip_first=skip_idx)
        assert y.shape == (batch_size, num_classes)


def test_resnet_skip_last_inference() -> None:
    num_classes = 5
    batch_size = 4
    input_vector = torch.rand((batch_size, 3, 32, 32))
    output_sizes = [
        (batch_size, 3, 32, 32),
        (batch_size, 20, 32, 32),
        (batch_size, 20, 32, 32),
        (batch_size, 40, 16, 16),
        (batch_size, 80, 8, 8),
        (batch_size, 160),
        (batch_size, 5),
    ]
    resnet = ResNet18(num_classes=num_classes)

    for skip_idx, size in enumerate(reversed(output_sizes)):
        y = resnet(input_vector, skip_last=skip_idx)
        assert y.shape == size


def test_resnet_features() -> None:
    num_classes = 3
    batch_size = 2
    input_vector = torch.rand((batch_size, 3, 32, 32))
    resnet = ResNet18(num_classes=num_classes)

    y = resnet.get_features(input_vector)
    assert y.shape == (batch_size, 160)
