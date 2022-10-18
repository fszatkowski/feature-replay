from commons import run_with_overrides


def run_3s_permuted_mnist_test(overrides: list[str]) -> None:
    run_with_overrides(["benchmark=3s_permuted_mnist"] + overrides)


def test_3s_permuted_mnist_default() -> None:
    run_3s_permuted_mnist_test([])


def test_3s_permuted_mnist_naive() -> None:
    run_3s_permuted_mnist_test(["strategy.base=Naive"])


def test_3s_permuted_mnist_cumulative() -> None:
    run_3s_permuted_mnist_test(["strategy.base=Cumulative"])


def test_3s_permuted_mnist_replay_const_memory() -> None:
    run_3s_permuted_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[replay]",
            "strategy.constant_memory=true",
        ]
    )


def test_3s_permuted_mnist_replay_adaptive_memory() -> None:
    run_3s_permuted_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[replay]",
            "strategy.constant_memory=false",
        ]
    )


def test_3s_permuted_mnist_gdumb() -> None:
    run_3s_permuted_mnist_test(
        [
            "benchmark=3s_split_mnist",
            "strategy.base=Naive",
            "strategy.plugins=[gdumb]",
        ]
    )


def test_3s_permuted_mnist_ewc() -> None:
    run_3s_permuted_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[ewc]",
            "strategy.ewc_lambda=1.0",
        ]
    )


def test_3s_permuted_mnist_lwf() -> None:
    run_3s_permuted_mnist_test(
        [
            "strategy.base=Naive",
            "strategy.plugins=[lwf]",
            "strategy.lwf_alpha=1.0",
            "strategy.lwf_temperature=1.0",
        ]
    )
