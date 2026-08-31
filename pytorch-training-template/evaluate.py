import torch
import torch.nn as nn

from data.dataset import (
    generate_regression_data,
    RegressionDataset,
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
    evaluate,
)

from utils.checkpoint import (
    load_checkpoint,
)

from utils.seed import (
    set_seed,
)


def main():
    # =====================
    # 1. Reproducibility
    # =====================
    set_seed(42)
    # =====================
    # 2. Prepare data
    # =====================
    X, y, _, _ = generate_regression_data(
        num_samples=1000,
        num_features=3,
    )
    (
        _,
        _,
        val_X,
        val_y,
    ) = train_val_split(
        X,
        y,
    )
    val_dataset = RegressionDataset(
        val_X,
        val_y,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )
    # =====================
    # 3. Create model
    # =====================
    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )
    # =====================
    # 4. Load checkpoint
    # =====================
    checkpoint = load_checkpoint(
        "checkpoints/best.pt",
        model,
    )
    print("Loaded checkpoint:")
    print(
        "Epoch:",
        checkpoint["epoch"]
    )
    print(
        "Best validation loss:",
        checkpoint["best_val_loss"]
    )
    # =====================
    # 5. Evaluate
    # =====================
    loss_fn = nn.MSELoss()
    val_loss = evaluate(
        model,
        val_loader,
        loss_fn,
    )
    print(
        "Evaluation loss:",
        val_loss
    )


if __name__ == "__main__":
    main()