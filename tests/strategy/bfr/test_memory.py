import pytest

from strategies.bfr.memory import split_memory_between_experiences


@pytest.mark.parametrize(
    "memory_size,n_exps,expected_split",
    [
        (100, 10, [10 for _ in range(10)]),
        (102, 5, [21, 21, 20, 20, 20]),
        (16, 3, [6, 5, 5]),
    ],
)
def test_split_memory_between_experiences(
    memory_size: int, n_exps: int, expected_split: list[int]
) -> None:
    split = split_memory_between_experiences(memory_size, n_exps)

    assert split == expected_split
