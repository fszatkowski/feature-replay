import torch
from torch.utils.data import TensorDataset

from models.mlp import MLP
from strategies.bfr.dataset_sampler import RandomDatasetSampler


def test_random_dataset_sampler_examples_match() -> None:
    sampler = RandomDatasetSampler(feature_level=0, device="cpu")
    model = MLP(num_classes=8, input_size=1, hidden_sizes=[2])
    features = torch.tensor([0, 1, 5, 3, 2, 6, 8, 7])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    dataset = TensorDataset(features, labels)

    sampled_features, sampled_labels = sampler.sample(4, dataset, model)
    for sampled_feature, sampled_label in zip(sampled_features, sampled_labels):
        org_feature_idx = (features == sampled_feature).nonzero()
        assert labels[org_feature_idx] == sampled_label


def test_random_sampler_on_feature_level() -> None:
    features = torch.stack([torch.rand((1, 28, 28)) for _ in range(10)])
    labels = torch.tensor(list(range(10)))
    dataset = TensorDataset(features, labels)
    model = MLP(num_classes=10, input_size=784, hidden_sizes=[512, 256, 128, 64])
    sample_size = 5
    expected_shapes = [
        (sample_size, 1, 28, 28),
        (sample_size, 512),
        (sample_size, 256),
        (sample_size, 128),
        (sample_size, 64),
    ]

    samplers = [
        RandomDatasetSampler(feature_level=feature_level, device="cpu")
        for feature_level in range(model.n_layers())
    ]

    for sampler, expected_shape in zip(samplers, expected_shapes):
        sampled_features, sampled_labels = sampler.sample(
            sample_size=sample_size, dataset=dataset, model=model
        )
        assert len(sampled_features) == len(sampled_labels) == sample_size
        assert sampled_features.shape == expected_shape
