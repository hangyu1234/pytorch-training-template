import torch
import torch.nn as nn

from data.classification import (
    ClassificationDataset,
    generate_classification_data,
)

from data.dataloader import (
    create_dataloader,
)

from models.mlp import (
    MLPClassifier,
)

from engine.trainer import (
    train_one_epoch_classification,
    evaluate_classification,
)


def create_test_components():
    X, y, _ = (
        generate_classification_data(
            num_samples=120,
            num_features=4,
            num_classes=3,
            seed=42,
        )
    )
    dataset = ClassificationDataset(
        X,
        y,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=16,
        shuffle=False,
    )
    model = MLPClassifier(
        input_dim=4,
        hidden_dim=16,
        num_classes=3,
    )
    loss_fn = (
        nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )
    return (
        model,
        dataloader,
        loss_fn,
        optimizer,
    )

def test_classification_train_returns_metrics():
    (
        model,
        dataloader,
        loss_fn,
        optimizer,
    ) = create_test_components()
    loss, accuracy = (
        train_one_epoch_classification(
            model,
            dataloader,
            loss_fn,
            optimizer,
            device="cpu",
        )
    )
    assert isinstance(
        loss,
        float,
    )
    assert isinstance(
        accuracy,
        float,
    )
    assert loss >= 0
    assert 0.0 <= accuracy <= 1.0

def test_classification_train_updates_parameters():
    (
        model,
        dataloader,
        loss_fn,
        optimizer,
    ) = create_test_components()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }
    train_one_epoch_classification(
        model,
        dataloader,
        loss_fn,
        optimizer,
        device="cpu",
    )
    parameters_after = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }
    changed = False
    for name in parameters_before:
        if not torch.equal(
            parameters_before[name],
            parameters_after[name],
        ):
            changed = True
    assert changed

def test_classification_evaluate_returns_metrics():
    (
        model,
        dataloader,
        loss_fn,
        _,
    ) = create_test_components()
    loss, accuracy = (
        evaluate_classification(
            model,
            dataloader,
            loss_fn,
            device="cpu",
        )
    )
    assert isinstance(
        loss,
        float,
    )
    assert isinstance(
        accuracy,
        float,
    )
    assert loss >= 0
    assert 0.0 <= accuracy <= 1.0

def test_classification_evaluate_does_not_update_parameters():
    (
        model,
        dataloader,
        loss_fn,
        _,
    ) = create_test_components()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }
    evaluate_classification(
        model,
        dataloader,
        loss_fn,
        device="cpu",
    )
    parameters_after = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }
    for name in parameters_before:
        assert torch.equal(
            parameters_before[name],
            parameters_after[name],
        )

def test_classification_evaluate_no_gradients():
    (
        model,
        dataloader,
        loss_fn,
        _,
    ) = create_test_components()
    for parameter in model.parameters():
        parameter.grad = None
    evaluate_classification(
        model,
        dataloader,
        loss_fn,
        device="cpu",
    )
    for parameter in model.parameters():
        assert parameter.grad is None