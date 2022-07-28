import pytest
import torch

from strategies.bfr.memory_sampler import RandomMemorySampler


def test_random_memory_sampler() -> None:
    sampler = RandomMemorySampler(batch_size=5, memory_size=10)
    features = torch.tensor([0, 1, 5, 3, 2, 6, 8, 7, 4, 9])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    sampled_features, sampled_labels = sampler.sample_batch(
        features=features, labels=labels
    )
    for sampled_feature, sampled_label in zip(sampled_features, sampled_labels):
        org_feature_idx = (features == sampled_feature).nonzero()
        assert labels[org_feature_idx] == sampled_label


def test_random_memory_sampler_in_loop() -> None:
    batch_size = 3
    sampler = RandomMemorySampler(batch_size=batch_size, memory_size=7)
    features = torch.tensor([0, 4, 1, 5, 3, 2, 6])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])

    for _ in range(10):
        sampled_features, sampled_labels = sampler.sample_batch(
            features=features, labels=labels
        )
        assert len(sampled_features) == len(sampled_labels) == batch_size
        for sampled_feature, sampled_label in zip(sampled_features, sampled_labels):
            org_feature_idx = (features == sampled_feature).nonzero()
            assert labels[org_feature_idx] == sampled_label


def test_random_memory_sampler_raises_on_non_matching_memory_size() -> None:
    sampler = RandomMemorySampler(memory_size=10, batch_size=8)
    features = torch.tensor([0, 1, 5, 3, 2, 6, 8, 7])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    with pytest.raises(AssertionError):
        sampler.sample_batch(features=features, labels=labels)


def test_random_memory_sampler_raises_on_exceeded_batch_size() -> None:
    with pytest.raises(AssertionError):
        RandomMemorySampler(memory_size=10, batch_size=16)
