import torch
import torch.nn as nn
import argparse
import os

from utils.config import load_config
from utils.logger import get_logger
from utils.metrics import save_history
from utils.visualize import plot_loss_curve

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
    train_one_epoch,
    evaluate,
)

from utils.seed import (
    set_seed,
)

from utils.checkpoint import (
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )
    return parser.parse_args()

def main(config):
    logger = get_logger(
        log_dir=config["logging"]["dir"]
    )
    checkpoint_dir = config["checkpoint"]["dir"]
    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )
    # =====================
    # 1. Reproducibility
    # =====================
    set_seed(
        config["seed"]
    )
    device_config = config["training"]["device"]
    if device_config == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_config)
    logger.info(
        f"Using device: {device}"
    )
    # =====================
    # 2. Prepare data
    # =====================
    X, y, true_weight, true_bias = generate_regression_data(
        num_samples=config["data"]["num_samples"],
        num_features=config["data"]["num_features"],
    )
    (
        train_X,
        train_y,
        val_X,
        val_y
    ) = train_val_split(
        X,
        y,
        val_ratio=config["data"]["val_ratio"],
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
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    # =====================
    # 3. Model
    # =====================
    model = LinearRegressionModel(
        input_dim=config["model"]["input_dim"],
        output_dim=config["model"]["output_dim"],
    )
    model = model.to(device)
    # =====================
    # 4. Loss
    # =====================
    loss_fn = nn.MSELoss()
    # =====================
    # 5. Optimizer
    # =====================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["lr"],
    )
    # =====================
    # 6. Training
    # =====================
    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    logger.info("===== Training Started =====")
    logger.info(f"Config: {config}")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device=device,
        )
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device=device,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f} "
            f"LR: {current_lr:.6f}"
        )
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                f"{checkpoint_dir}/best.pt",
                model,
                optimizer,
                epoch + 1,
                best_val_loss,
            )
        # save latest
        save_checkpoint(
            f"{checkpoint_dir}/last.pt",
            model,
            optimizer,
            epoch + 1,
            best_val_loss,
        )
    logger.info("===== Training Finished =====")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    history_path = save_history(
        history,
        output_dir=config["output"]["dir"],
    )
    logger.info(
        f"Training history saved to: {history_path}"
    )
    figure_dir = os.path.join(
        config["output"]["dir"],
        "figures",
    )
    loss_curve_path = plot_loss_curve(
        history,
        output_dir=figure_dir,
    )
    logger.info(
        f"Loss curve saved to: {loss_curve_path}"
    )
    with torch.no_grad():
        print("Learned weight:")
        print(model.linear.weight.detach().cpu())
        print("Learned bias:")
        print(model.linear.bias.detach().cpu())

if __name__ == "__main__":
    args = parse_args()
    config = load_config(
        args.config
    )
    main(config)