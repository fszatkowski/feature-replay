import torch

from models.conv_mlp import ConvMLP


def test_conv_mlp_inference_pooling() -> None:
    model = ConvMLP(
        input_size=(3, 32, 32),
        num_classes=10,
        channels=[8, 16, 32],
        kernel_size=3,
        hidden_sizes=[1024, 512],
        pooling=True,
        dropout_ratio=0.5,
    )
    x = torch.rand((8, 3, 32, 32))

    y = model(x)

    assert y.shape == (8, 10)


def test_conv_mlp_inference_no_pooling() -> None:
    model = ConvMLP(
        input_size=(1, 28, 28),
        num_classes=10,
        channels=[8, 4, 2],
        kernel_size=3,
        hidden_sizes=[1024, 512],
        pooling=False,
        dropout_ratio=0.5,
    )
    x = torch.rand((8, 1, 28, 28))

    y = model(x)

    assert y.shape == (8, 10)


def test_conv_mlp_skipped_inference_pooling() -> None:
    model = ConvMLP(
        input_size=(3, 32, 32),
        num_classes=10,
        channels=[8, 16, 32],
        kernel_size=3,
        hidden_sizes=[256],
        pooling=True,
        dropout_ratio=0.5,
    )
    feature_vectors = [
        torch.rand((8, 3, 32, 32)),
        torch.rand((8, 8, 16, 16)),
        torch.rand((8, 16, 8, 8)),
        torch.rand((8, 512)),
        torch.rand((8, 256)),
    ]

    for feature_level, feature_vector in enumerate(feature_vectors):
        y = model(feature_vector, skip_first=feature_level)
        assert y.shape == (8, 10)

        y = model(feature_vectors[0], skip_last=model.n_layers() - feature_level)
        assert y.shape == feature_vector.shape


def test_conv_mlp_skipped_inference_no_pooling() -> None:
    model = ConvMLP(
        input_size=(1, 28, 28),
        num_classes=10,
        channels=[8, 4, 2],
        kernel_size=3,
        hidden_sizes=[1024, 512],
        pooling=False,
        dropout_ratio=0.5,
    )
    feature_vectors = [
        torch.rand((8, 1, 28, 28)),
        torch.rand((8, 8, 28, 28)),
        torch.rand((8, 4, 28, 28)),
        torch.rand((8, 1568)),
        torch.rand(8, 1024),
        torch.rand(8, 512),
    ]

    for feature_level, feature_vector in enumerate(feature_vectors):
        y = model(feature_vector, skip_first=feature_level)
        assert y.shape == (8, 10)

        y = model(feature_vectors[0], skip_last=model.n_layers() - feature_level)
        assert y.shape == feature_vector.shape
