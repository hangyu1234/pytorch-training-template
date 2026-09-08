import torch
import pytest

from data.dataset import (
    RegressionDataset,
    generate_regression_data,
)

from data.split import (
    train_val_split,
)


def test_regression_dataset_length():
    features = torch.randn(10, 3)
    targets = torch.randn(10, 1)
    dataset = RegressionDataset(
        features,
        targets,
    )
    assert len(dataset) == 10

def test_regression_dataset_item_shape():
    features = torch.randn(10, 3)
    targets = torch.randn(10, 1)
    dataset = RegressionDataset(
        features,
        targets,
    )
    x, y = dataset[0]
    assert x.shape == torch.Size([3])
    assert y.shape == torch.Size([1])

def test_regression_dataset_dtype():
    features = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    targets = [
        [1],
        [2],
    ]
    dataset = RegressionDataset(
        features,
        targets,
    )
    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_regression_dataset_length_mismatch():
    features = torch.randn(10, 3)
    targets = torch.randn(8, 1)
    with pytest.raises(ValueError):
        RegressionDataset(
            features,
            targets,
        )


def test_generate_regression_data_shape():
    X, y, weight, bias = generate_regression_data(
        num_samples=100,
        num_features=4,
        seed=42,
    )
    assert X.shape == torch.Size([100, 4])
    assert y.shape == torch.Size([100, 1])
    assert weight.shape == torch.Size([4, 1])
    assert bias.shape == torch.Size([1])


def test_generate_regression_data_reproducible():
    X1, y1, w1, b1 = generate_regression_data(
        num_samples=100,
        num_features=3,
        seed=42,
    )
    X2, y2, w2, b2 = generate_regression_data(
        num_samples=100,
        num_features=3,
        seed=42,
    )
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)
    assert torch.equal(w1, w2)
    assert torch.equal(b1, b2)

def test_train_val_split_size():
    features = torch.randn(100, 3)
    targets = torch.randn(100, 1)
    (
        train_X,
        train_y,
        val_X,
        val_y,
    ) = train_val_split(
        features,
        targets,
        val_ratio=0.2,
        seed=42,
    )
    assert len(train_X) == 80
    assert len(train_y) == 80
    assert len(val_X) == 20
    assert len(val_y) == 20
    assert (
        len(train_X) + len(val_X)
        == len(features)
    )


def test_train_val_split_reproducible():
    features = torch.randn(100, 3)
    targets = torch.randn(100, 1)
    split1 = train_val_split(
        features,
        targets,
        val_ratio=0.2,
        seed=42,
    )
    split2 = train_val_split(
        features,
        targets,
        val_ratio=0.2,
        seed=42,
    )
    for tensor1, tensor2 in zip(
        split1,
        split2,
    ):
        assert torch.equal(
            tensor1,
            tensor2,
        )

def test_train_val_split_different_seed():
    features = torch.arange(
        100,
        dtype=torch.float32,
    ).reshape(100, 1)
    targets = features.clone()
    split1 = train_val_split(
        features,
        targets,
        val_ratio=0.2,
        seed=42,
    )
    split2 = train_val_split(
        features,
        targets,
        val_ratio=0.2,
        seed=123,
    )
    train_X1 = split1[0]
    train_X2 = split2[0]
    assert not torch.equal(
        train_X1,
        train_X2,
    )