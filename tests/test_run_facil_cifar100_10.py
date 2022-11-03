import pytest
from commons import run_with_overrides


def run_3s_facil_cifar100_10_test(overrides: list[str]) -> None:
    run_with_overrides(["benchmark=facil_cifar100_10"] + overrides)


@pytest.mark.e2e
def test_3s_facil_cifar100_10_default() -> None:
    run_3s_facil_cifar100_10_test([])


@pytest.mark.e2e
def test_3s_facil_cifar100_10_naive() -> None:
    run_3s_facil_cifar100_10_test(["strategy.base=Naive"])


@pytest.mark.e2e
def test_3s_facil_cifar100_10_rpf() -> None:
    run_3s_facil_cifar100_10_test(
        [
            "strategy.base=RandomPartialFreezing",
            "strategy.rpf_probs=[0.6,0.1,0.1,0.1,0.1]",
        ]
    )


@pytest.mark.e2e
def test_3s_facil_cifar100_10_cumulative() -> None:
    run_3s_facil_cifar100_10_test(["strategy.base=Cumulative"])


@pytest.mark.e2e
def test_3s_facil_cifar100_10_replay_const_memory() -> None:
    run_3s_facil_cifar100_10_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[replay]",
            "strategy.constant_memory=true",
        ]
    )


@pytest.mark.e2e
def test_3s_facil_cifar100_10_replay_adaptive_memory() -> None:
    run_3s_facil_cifar100_10_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[replay]",
            "strategy.constant_memory=false",
        ]
    )


@pytest.mark.e2e
def test_3s_facil_cifar100_10_gdumb() -> None:
    run_3s_facil_cifar100_10_test(
        [
            "benchmark=3s_split_mnist",
            "strategy.base=Naive",
            "strategy.plugins=[gdumb]",
        ]
    )


@pytest.mark.e2e
def test_3s_facil_cifar100_10_ewc() -> None:
    run_3s_facil_cifar100_10_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[ewc]",
            "strategy.ewc_lambda=1.0",
        ]
    )


@pytest.mark.e2e
def test_3s_facil_cifar100_10_lwf() -> None:
    run_3s_facil_cifar100_10_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[lwf]",
            "strategy.lwf_alpha=1.0",
            "strategy.lwf_temperature=1.0",
        ]
    )
