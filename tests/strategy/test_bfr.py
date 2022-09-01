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


def test_get_weights_for_equal_update_strategy() -> None:
    replay_probs = BufferedFeatureReplayStrategy.get_replay_probs(
        update_strategy="equal", n_layers=4
    )

    assert replay_probs == [0.25, 0.25, 0.25, 0.25]


def test_get_weights_for_lin_asc_update_strategy() -> None:
    replay_probs = BufferedFeatureReplayStrategy.get_replay_probs(
        update_strategy="linear_ascending", n_layers=3
    )

    base = replay_probs[0]
    for i, prob in enumerate(replay_probs):
        assert (i + 1) * base == prob


def test_get_weights_for_lin_desc_update_strategy() -> None:
    replay_probs = BufferedFeatureReplayStrategy.get_replay_probs(
        update_strategy="linear_descending", n_layers=5
    )

    base = replay_probs[-1]
    for i, prob in enumerate(list(reversed(replay_probs))):
        assert (i + 1) * base == prob


def test_get_weights_for_geo_asc_strategy() -> None:
    replay_probs = BufferedFeatureReplayStrategy.get_replay_probs(
        update_strategy="geometric_ascending", n_layers=2
    )

    for prob, next_prob in zip(replay_probs, replay_probs[1:]):
        assert next_prob == 2 * prob


def test_get_weights_for_geo_desc_update_strategy() -> None:
    replay_probs = BufferedFeatureReplayStrategy.get_replay_probs(
        update_strategy="geometric_descending", n_layers=6
    )

    for prob, next_prob in zip(replay_probs, replay_probs[1:]):
        assert next_prob == 0.5 * prob
