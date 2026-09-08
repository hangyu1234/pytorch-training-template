import pytest

from utils.selection import (
    initialize_best_metric,
    is_better,
)


def test_initialize_best_metric_min():
    best = initialize_best_metric(
        "min"
    )
    assert best == float("inf")

def test_initialize_best_metric_max():
    best = initialize_best_metric(
        "max"
    )
    assert best == float("-inf")

def test_initialize_best_metric_invalid():
    with pytest.raises(ValueError):
        initialize_best_metric(
            "invalid"
        )

def test_is_better_min():
    assert is_better(
        current=0.4,
        best=0.5,
        mode="min",
    )
    assert not is_better(
        current=0.6,
        best=0.5,
        mode="min",
    )

def test_is_better_max():
    assert is_better(
        current=0.9,
        best=0.8,
        mode="max",
    )
    assert not is_better(
        current=0.7,
        best=0.8,
        mode="max",
    )

def test_is_better_equal_is_not_better():
    assert not is_better(
        current=0.5,
        best=0.5,
        mode="min",
    )
    assert not is_better(
        current=0.5,
        best=0.5,
        mode="max",
    )

def test_is_better_invalid_mode():
    with pytest.raises(ValueError):
        is_better(
            current=1.0,
            best=0.0,
            mode="invalid",
        )