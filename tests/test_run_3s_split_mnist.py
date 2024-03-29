import pytest
from commons import run_with_overrides


def run_3s_split_mnist_test(overrides: list[str]) -> None:
    run_with_overrides(["benchmark=3s_split_mnist"] + overrides)


@pytest.mark.e2e
def test_3s_split_mnist_default() -> None:
    run_3s_split_mnist_test([])


@pytest.mark.e2e
def test_3s_split_mnist_naive() -> None:
    run_3s_split_mnist_test(["strategy.base=Naive"])


@pytest.mark.e2e
def test_3s_split_mnist_rpf() -> None:
    run_3s_split_mnist_test(
        ["strategy.base=RandomPartialFreezing", "strategy.rpf_probs=[0.5,0.25,0.25]"]
    )


@pytest.mark.e2e
def test_3s_split_mnist_cumulative() -> None:
    run_3s_split_mnist_test(["strategy.base=Cumulative"])


@pytest.mark.e2e
def test_3s_split_mnist_replay_const_memory() -> None:
    run_3s_split_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[replay]",
            "strategy.constant_memory=true",
        ]
    )


@pytest.mark.e2e
def test_3s_split_mnist_replay_adaptive_memory() -> None:
    run_3s_split_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[replay]",
            "strategy.constant_memory=false",
        ]
    )


@pytest.mark.e2e
def test_3s_split_mnist_gdumb() -> None:
    run_3s_split_mnist_test(
        [
            "benchmark=3s_split_mnist",
            "strategy.base=Naive",
            "strategy.plugins=[gdumb]",
        ]
    )


@pytest.mark.e2e
def test_3s_split_mnist_ewc() -> None:
    run_3s_split_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[ewc]",
            "strategy.ewc_lambda=1.0",
        ]
    )


@pytest.mark.e2e
def test_3s_split_mnist_lwf() -> None:
    run_3s_split_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[lwf]",
            "strategy.lwf_alpha=1.0",
            "strategy.lwf_temperature=1.0",
        ]
    )
