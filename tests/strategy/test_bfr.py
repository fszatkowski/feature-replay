import pytest

from strategies.buffered_feature_replay import BufferedFeatureReplayStrategy


def test_get_replay_mb_sizes_from_list() -> None:
    replay_mb_sizes = BufferedFeatureReplayStrategy.get_replay_mb_sizes(
        [64, 64, 64], n_layers=3
    )

    assert replay_mb_sizes == [64, 64, 64]


def test_get_replay_mb_sizes_from_list_raises() -> None:
    with pytest.raises(AssertionError):
        BufferedFeatureReplayStrategy.get_replay_mb_sizes([64], n_layers=3)


def test_get_replay_mb_sizes_from_int() -> None:
    replay_mb_sizes = BufferedFeatureReplayStrategy.get_replay_mb_sizes(64, n_layers=3)

    assert replay_mb_sizes == [64, 64, 64]


def get_replay_memory_sizes_from_list() -> None:
    replay_memory_sizes = BufferedFeatureReplayStrategy.get_replay_memory_sizes(
        [64, 64, 64], 3
    )

    assert replay_memory_sizes == [64, 64, 64]


def get_replay_memory_sizes_from_list_raises() -> None:
    with pytest.raises(AssertionError):
        BufferedFeatureReplayStrategy.get_replay_memory_sizes([64, 64, 64], 2)


def get_replay_memory_sizes_from_int() -> None:
    replay_memory_sizes = BufferedFeatureReplayStrategy.get_replay_memory_sizes(32, 2)

    assert replay_memory_sizes == [32, 32]
