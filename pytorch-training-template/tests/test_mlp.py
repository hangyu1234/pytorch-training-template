import torch
import torch.nn as nn

from models.mlp import (
    MLPClassifier,
)


def test_mlp_is_module():
    model = MLPClassifier(
        input_dim=4,
        hidden_dim=16,
        num_classes=3,
    )
    assert isinstance(
        model,
        nn.Module,
    )

def test_mlp_output_shape():
    model = MLPClassifier(
        input_dim=4,
        hidden_dim=16,
        num_classes=3,
    )
    x = torch.randn(
        8,
        4,
    )
    output = model(x)
    assert output.shape == torch.Size(
        [8, 3]
    )

def test_mlp_has_trainable_parameters():
    model = MLPClassifier(
        input_dim=4,
        hidden_dim=16,
        num_classes=3,
    )
    parameters = list(
        model.parameters()
    )
    assert len(parameters) > 0
    for parameter in parameters:
        assert parameter.requires_grad

def test_mlp_forward_is_finite():
    model = MLPClassifier(
        input_dim=4,
        hidden_dim=16,
        num_classes=3,
    )
    x = torch.randn(
        8,
        4,
    )
    output = model(x)
    assert torch.isfinite(
        output
    ).all()