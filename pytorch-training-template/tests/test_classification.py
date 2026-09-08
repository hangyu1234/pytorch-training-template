import torch
import pytest

from data.classification import (
    ClassificationDataset,
    generate_classification_data,
)


def test_classification_dataset_length():
    features = torch.randn(10, 4)
    labels = torch.randint(
        0,
        3,
        (10,),
    )
    dataset = ClassificationDataset(
        features,
        labels,
    )
    assert len(dataset) == 10

def test_classification_dataset_item_shape():
    features = torch.randn(10, 4)
    labels = torch.randint(
        0,
        3,
        (10,),
    )
    dataset = ClassificationDataset(
        features,
        labels,
    )
    x, y = dataset[0]
    assert x.shape == torch.Size([4])
    assert y.shape == torch.Size([])

def test_classification_dataset_dtype():
    features = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    labels = [
        0,
        2,
    ]
    dataset = ClassificationDataset(
        features,
        labels,
    )
    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.long

def test_classification_dataset_length_mismatch():
    features = torch.randn(10, 4)
    labels = torch.randint(
        0,
        3,
        (8,),
    )
    with pytest.raises(ValueError):
        ClassificationDataset(
            features,
            labels,
        )

def test_classification_dataset_label_shape():
    features = torch.randn(10, 4)
    labels = torch.randint(
        0,
        3,
        (10, 1),
    )
    with pytest.raises(ValueError):
        ClassificationDataset(
            features,
            labels,
        )

def test_generate_classification_data_shape():
    X, y, centers = generate_classification_data(
        num_samples=100,
        num_features=5,
        num_classes=4,
        seed=42,
    )
    assert X.shape == torch.Size(
        [100, 5]
    )
    assert y.shape == torch.Size(
        [100]
    )
    assert centers.shape == torch.Size(
        [4, 5]
    )

def test_generate_classification_data_reproducible():
    X1, y1, centers1 = (
        generate_classification_data(
            num_samples=100,
            num_features=4,
            num_classes=3,
            seed=42,
        )
    )
    X2, y2, centers2 = (
        generate_classification_data(
            num_samples=100,
            num_features=4,
            num_classes=3,
            seed=42,
        )
    )
    assert torch.equal(
        X1,
        X2,
    )
    assert torch.equal(
        y1,
        y2,
    )
    assert torch.equal(
        centers1,
        centers2,
    )

def test_generate_classification_data_balanced():
    _, y, _ = generate_classification_data(
        num_samples=1000,
        num_features=4,
        num_classes=3,
        seed=42,
    )
    counts = []
    for class_id in range(3):
        count = (
            y == class_id
        ).sum().item()
        counts.append(count)
    assert max(counts) - min(counts) <= 1