import torch

from models.resnet import ResNet, ResNet18, ResNet32, SlimResNet18

BATCH_SIZE = 3


def _test_skip_first_inference(
    model: ResNet, feature_shapes: list[tuple[int, ...]], num_classes: int
) -> None:
    inputs = [torch.rand((BATCH_SIZE, *shape)) for shape in feature_shapes]

    for skip_idx, input_vector in enumerate(inputs):
        y = model(input_vector, skip_first=skip_idx)
        assert y.shape == (BATCH_SIZE, num_classes)


def test_slimresnet18_skip_first_inference() -> None:
    num_classes = 8
    model = SlimResNet18(num_classes=num_classes)

    _test_skip_first_inference(
        model,
        num_classes=num_classes,
        feature_shapes=[
            (3, 32, 32),
            (20, 32, 32),
            (20, 32, 32),
            (40, 16, 16),
            (80, 8, 8),
            (160,),
        ],
    )


def test_resnet18_skip_first_inference() -> None:
    num_classes = 10
    model = ResNet18(num_classes=num_classes)

    _test_skip_first_inference(
        model,
        num_classes=num_classes,
        feature_shapes=[
            (3, 32, 32),
            (64, 32, 32),
            (64, 32, 32),
            (128, 16, 16),
            (256, 8, 8),
            (512,),
        ],
    )


def test_resnet32_skip_first_inference() -> None:
    num_classes = 9
    model = ResNet32(num_classes=num_classes)

    _test_skip_first_inference(
        model,
        num_classes=num_classes,
        feature_shapes=[
            (3, 32, 32),
            (16, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
            (64,),
        ],
    )


def _test_skip_last_inference(
    model: ResNet,
    feature_shapes: list[tuple[int, ...]],
) -> None:
    input_vector = torch.rand((BATCH_SIZE, 3, 32, 32))
    output_sizes = [(BATCH_SIZE, *shape) for shape in feature_shapes]

    for skip_idx, size in enumerate(reversed(output_sizes)):
        y = model(input_vector, skip_last=skip_idx)
        assert y.shape == size


def test_resnet18_skip_last_inference() -> None:
    num_classes = 9
    model = ResNet18(num_classes=num_classes)

    _test_skip_last_inference(
        model,
        feature_shapes=[
            (3, 32, 32),
            (64, 32, 32),
            (64, 32, 32),
            (128, 16, 16),
            (256, 8, 8),
            (512,),
            (num_classes,),
        ],
    )


def test_slimresnet18_skip_last_inference() -> None:
    num_classes = 7
    model = SlimResNet18(num_classes=num_classes)

    _test_skip_last_inference(
        model,
        feature_shapes=[
            (3, 32, 32),
            (20, 32, 32),
            (20, 32, 32),
            (40, 16, 16),
            (80, 8, 8),
            (160,),
            (num_classes,),
        ],
    )


def test_resnet32_skip_last_inference() -> None:
    num_classes = 9
    model = ResNet32(num_classes=num_classes)

    _test_skip_last_inference(
        model,
        feature_shapes=[
            (3, 32, 32),
            (16, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
            (64,),
            (num_classes,),
        ],
    )


def _test_get_features(model: ResNet, features_size: int) -> None:
    input_vector = torch.rand((BATCH_SIZE, 3, 32, 32))

    y = model.get_features(input_vector)
    assert y.shape == (BATCH_SIZE, features_size)


def test_resnet18_get_features() -> None:
    model = ResNet18(num_classes=7)

    _test_get_features(model, features_size=512)


def test_slimresnet18_get_features() -> None:
    model = SlimResNet18(num_classes=7)

    _test_get_features(model, features_size=160)


def test_resnet32_get_features() -> None:
    model = ResNet32(num_classes=7)

    _test_get_features(model, features_size=64)
