import math
import pytest
import torch
import torch.nn as nn

from data.dataset import (
    RegressionDataset,
    generate_regression_data,
)

from data.split import (
    train_val_split,
)

from data.dataloader import (
    create_dataloader,
)

from models.linear import (
    LinearRegressionModel,
)

from engine.trainer import (
    train_one_epoch,
    evaluate,
)

from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

from utils.seed import (
    set_seed,
)


def test_training_pipeline_smoke(tmp_path):
    # =====================
    # 1. Reproducibility
    # =====================
    set_seed(42)
    train_generator = torch.Generator()
    train_generator.manual_seed(42)
    # =====================
    # 2. Data
    # =====================
    X, y, _, _ = generate_regression_data(
        num_samples=100,
        num_features=3,
        seed=42,
    )
    (
        train_X,
        train_y,
        val_X,
        val_y,
    ) = train_val_split(
        X,
        y,
        val_ratio=0.2,
        seed=42,
    )
    train_dataset = RegressionDataset(
        train_X,
        train_y,
    )
    val_dataset = RegressionDataset(
        val_X,
        val_y,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )
    # =====================
    # 3. Model
    # =====================
    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )
    # =====================
    # 4. Train
    # =====================
    initial_val_loss = evaluate(
        model,
        val_loader,
        loss_fn,
        device="cpu",
    )
    for _ in range(3):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device="cpu",
        )
    final_val_loss = evaluate(
        model,
        val_loader,
        loss_fn,
        device="cpu",
    )
    assert math.isfinite(train_loss)
    assert math.isfinite(final_val_loss)
    assert final_val_loss < initial_val_loss
    # =====================
    # 5. Save checkpoint
    # =====================
    checkpoint_path = (
        tmp_path / "last.pt"
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        best_metric=final_val_loss,
        train_generator=train_generator,
        selection_metric="val_loss",
        selection_mode="min",
    )
    assert checkpoint_path.exists()
    # =====================
    # 6. Rebuild model
    # =====================
    loaded_model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    loaded_optimizer = torch.optim.SGD(
        loaded_model.parameters(),
        lr=0.01,
    )
    loaded_generator = torch.Generator()
    loaded_generator.manual_seed(999)
    # =====================
    # 7. Load checkpoint
    # =====================
    checkpoint = load_checkpoint(
        checkpoint_path,
        loaded_model,
        loaded_optimizer,
        train_generator=loaded_generator,
    )
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_metric"] == final_val_loss
    assert checkpoint["selection_metric"] == "val_loss"
    assert checkpoint["selection_mode"] == "min"
    # =====================
    # 8. Evaluate loaded model
    # =====================
    loaded_val_loss = evaluate(
        loaded_model,
        val_loader,
        loss_fn,
        device="cpu",
    )
    assert loaded_val_loss == pytest.approx(
        final_val_loss,
        abs=1e-12,
    )
    # =====================
    # 9. Parameters identical
    # =====================
    for original, loaded in zip(
        model.parameters(),
        loaded_model.parameters(),
    ):
        assert torch.equal(
            original,
            loaded,
        )