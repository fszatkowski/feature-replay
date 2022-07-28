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


def test_get_replay_probs_from_list() -> None:
    replay_probs = BufferedFeatureReplayStrategy.get_replay_probs(
        [0.1, 0.2, 0.3, 0.4],
        replay_memory_sizes=[32, 32, 32, 32],
        n_layers=4,
    )

    assert replay_probs == [0.1, 0.2, 0.3, 0.4]


def test_get_replay_probs_from_list_raises_on_sum() -> None:
    with pytest.raises(AssertionError):
        BufferedFeatureReplayStrategy.get_replay_probs(
            [0.1, 0.2, 0.3, 0.5],
            replay_memory_sizes=[32, 32, 32, 32],
            n_layers=4,
        )


def test_get_replay_probs_from_list_raises_on_n_layers() -> None:
    with pytest.raises(AssertionError):
        BufferedFeatureReplayStrategy.get_replay_probs(
            [0.1, 0.2, 0.3],
            replay_memory_sizes=[32, 32, 32, 32],
            n_layers=4,
        )


def test_get_replay_probs_from_int() -> None:
    replay_probs = BufferedFeatureReplayStrategy.get_replay_probs(
        0.25,
        replay_memory_sizes=[32, 32, 32, 32],
        n_layers=4,
    )

    assert replay_probs == [0.25, 0.25, 0.25, 0.25]


def test_get_replay_probs_from_int_rasies_on_sum() -> None:
    with pytest.raises(AssertionError):
        BufferedFeatureReplayStrategy.get_replay_probs(
            0.3,
            replay_memory_sizes=[32, 32, 32, 32],
            n_layers=4,
        )
