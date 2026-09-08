import torch
import torch.nn as nn

from models.linear import (
    LinearRegressionModel,
)

from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def create_components():
    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
    )
    generator = torch.Generator()
    generator.manual_seed(42)
    return (
        model,
        optimizer,
        generator,
    )

def run_one_optimization_step(
    model,
    optimizer,
):
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(
        prediction,
        y,
    )
    loss.backward()
    optimizer.step()

def test_checkpoint_restores_model(tmp_path):
    (
        model,
        optimizer,
        generator,
    ) = create_components()
    run_one_optimization_step(
        model,
        optimizer,
    )
    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_metric=0.123,
        train_generator=generator,
    )
    model_loaded = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    optimizer_loaded = torch.optim.SGD(
        model_loaded.parameters(),
        lr=0.01,
        momentum=0.9,
    )
    generator_loaded = torch.Generator()
    generator_loaded.manual_seed(999)
    load_checkpoint(
        checkpoint_path,
        model_loaded,
        optimizer_loaded,
        train_generator=generator_loaded,
    )
    for (
        parameter_original,
        parameter_loaded,
    ) in zip(
        model.parameters(),
        model_loaded.parameters(),
    ):
        assert torch.equal(
            parameter_original,
            parameter_loaded,
        )

def test_checkpoint_restores_metadata(tmp_path):
    (
        model,
        optimizer,
        generator,
    ) = create_components()
    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=7,
        best_metric=0.456,
        config={
            "seed": 42,
        },
        train_generator=generator,
        selection_metric="val_loss",
        selection_mode="min",
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        train_generator=generator,
    )
    assert checkpoint["epoch"] == 7
    assert checkpoint["best_metric"] == 0.456
    assert checkpoint["selection_metric"] == "val_loss"
    assert checkpoint["selection_mode"] == "min"
    assert checkpoint["config"] == {
        "seed": 42,
    }

def test_checkpoint_restores_optimizer(tmp_path):
    (
        model,
        optimizer,
        generator,
    ) = create_components()
    # 创建 momentum buffer
    run_one_optimization_step(
        model,
        optimizer,
    )
    optimizer_state_before = (
        optimizer.state_dict()
    )
    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        best_metric=0.5,
        train_generator=generator,
    )
    model_loaded = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    optimizer_loaded = torch.optim.SGD(
        model_loaded.parameters(),
        lr=0.01,
        momentum=0.9,
    )
    load_checkpoint(
        checkpoint_path,
        model_loaded,
        optimizer_loaded,
    )
    optimizer_state_after = (
        optimizer_loaded.state_dict()
    )
    assert (
        optimizer_state_before["param_groups"]
        == optimizer_state_after["param_groups"]
    )
    for parameter_id in optimizer_state_before["state"]:
        buffer_before = (
            optimizer_state_before["state"]
            [parameter_id]["momentum_buffer"]
        )
        buffer_after = (
            optimizer_state_after["state"]
            [parameter_id]["momentum_buffer"]
        )
        assert torch.equal(
            buffer_before,
            buffer_after,
        )

def test_checkpoint_restores_generator_state(
    tmp_path,
):
    (
        model,
        optimizer,
        generator,
    ) = create_components()
    # 先消耗一些随机数，让 RNG state 向前移动
    torch.randperm(
        20,
        generator=generator,
    )
    generator_state_before = (
        generator.get_state().clone()
    )
    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=4,
        best_metric=0.25,
        train_generator=generator,
    )
    generator_loaded = (
        torch.Generator()
    )
    generator_loaded.manual_seed(
        999
    )
    load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        train_generator=generator_loaded,
    )
    assert torch.equal(
        generator_state_before,
        generator_loaded.get_state(),
    )