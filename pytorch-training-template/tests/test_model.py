import torch
import torch.nn as nn

from models.linear import (
    LinearRegressionModel,
)


def test_linear_model_is_module():
    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    assert isinstance(
        model,
        nn.Module,
    )

def test_linear_model_output_shape():
    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    x = torch.randn(
        8,
        3,
    )
    output = model(x)
    assert output.shape == torch.Size(
        [8, 1]
    )

def test_linear_model_parameter_shape():
    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    assert model.linear.weight.shape == torch.Size(
        [1, 3]
    )
    assert model.linear.bias.shape == torch.Size(
        [1]
    )

def test_linear_model_has_trainable_parameters():
    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    parameters = list(
        model.parameters()
    )
    assert len(parameters) > 0
    for parameter in parameters:
        assert parameter.requires_grad