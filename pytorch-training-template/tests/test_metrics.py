import pytest
import torch

from utils.metrics import (
    classification_accuracy,
)


def test_classification_accuracy():
    logits = torch.tensor([
        [5.0, 1.0, 0.0],
        [0.0, 4.0, 1.0],
        [3.0, 2.0, 1.0],
    ])
    labels = torch.tensor([
        0,
        1,
        2,
    ])
    accuracy = classification_accuracy(
        logits,
        labels,
    )
    assert accuracy == pytest.approx(
        2 / 3
    )

def test_classification_accuracy_perfect():
    logits = torch.tensor([
        [5.0, 1.0],
        [1.0, 5.0],
        [4.0, 0.0],
    ])
    labels = torch.tensor([
        0,
        1,
        0,
    ])
    accuracy = classification_accuracy(
        logits,
        labels,
    )
    assert accuracy == 1.0


def test_classification_accuracy_returns_float():
    logits = torch.randn(
        8,
        3,
    )
    labels = torch.randint(
        0,
        3,
        (8,),
    )
    accuracy = classification_accuracy(
        logits,
        labels,
    )
    assert isinstance(
        accuracy,
        float,
    )

def test_classification_accuracy_invalid_logits_shape():
    logits = torch.randn(
        8,
    )
    labels = torch.randint(
        0,
        3,
        (8,),
    )
    with pytest.raises(
        ValueError
    ):
        classification_accuracy(
            logits,
            labels,
        )


def test_classification_accuracy_invalid_labels_shape():
    logits = torch.randn(
        8,
        3,
    )
    labels = torch.randint(
        0,
        3,
        (8, 1),
    )
    with pytest.raises(
        ValueError
    ):
        classification_accuracy(
            logits,
            labels,
        )