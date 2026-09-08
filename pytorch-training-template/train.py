import torch
import torch.nn as nn
import argparse
from pathlib import Path

from utils.experiment import (
    prepare_run_dir,
    save_config_snapshot,
    save_resume_config_snapshot,
)

from utils.config import load_config
from utils.logger import get_logger

from utils.visualize import (
    plot_loss_curve,
    plot_accuracy_curve,
)

from utils.metrics import (
    save_history,
    load_history,
)

from utils.selection import (
    initialize_best_metric,
    is_better,
)

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

from utils.seed import (
    set_seed,
)

from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

from data.classification import (
    ClassificationDataset,
    generate_classification_data,
)

from models.mlp import (
    MLPClassifier,
)

from engine.trainer import (
    train_one_epoch,
    evaluate,
    train_one_epoch_classification,
    evaluate_classification,
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
    # =====================
    # 0. Experiment directory
    # =====================
    run_dir = prepare_run_dir(config)
    checkpoint_dir = run_dir / "checkpoints"
    figure_dir = run_dir / "figures"
    history_path = run_dir / "history.json"
    task_type = config["task"]["type"]
    if task_type not in {
        "regression",
        "classification",
    }:
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )
    logger = get_logger(
            log_dir=str(run_dir)
        )
    logger.info(
        f"Task type: {task_type}"
    )
    # Fresh run: save configuration snapshot
    if not config["resume"]["enabled"]:
        config_path = save_config_snapshot(
            config,
            run_dir,
        )
    else:
        config_path = save_resume_config_snapshot(
            config,
            run_dir,
        )
    # Logger is stored inside this run
    logger.info(
        f"Run directory: {run_dir}"
    )
    if not config["resume"]["enabled"]:
        logger.info(
            f"Config snapshot saved to: {config_path}"
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
    if task_type == "regression":
        X, y, true_weight, true_bias = (
            generate_regression_data(
                num_samples=config["data"]["num_samples"],
                num_features=config["data"]["num_features"],
            )
        )
    elif task_type == "classification":
        X, y, class_centers = (
            generate_classification_data(
                num_samples=config["data"]["num_samples"],
                num_features=config["data"]["num_features"],
                num_classes=config["data"]["num_classes"],
                noise_std=config["data"]["noise_std"],
                seed=config["seed"],
            )
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
    if task_type == "regression":
        train_dataset = RegressionDataset(
            train_X,
            train_y,
        )
        val_dataset = RegressionDataset(
            val_X,
            val_y,
        )
    elif task_type == "classification":
        train_dataset = ClassificationDataset(
            train_X,
            train_y,
        )
        val_dataset = ClassificationDataset(
            val_X,
            val_y,
        )
    train_generator = torch.Generator()
    train_generator.manual_seed(
        config["seed"]
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=train_generator,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    # =====================
    # 3. Model
    # =====================
    if task_type == "regression":
        model = LinearRegressionModel(
            input_dim=config["model"]["input_dim"],
            output_dim=config["model"]["output_dim"],
        )
    elif task_type == "classification":
        model = MLPClassifier(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )
    model = model.to(device)
    # =====================
    # 4. Loss
    # =====================
    if task_type == "regression":
        loss_fn = nn.MSELoss()
    elif task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
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
    start_epoch = 0
    selection_metric = config["selection"]["metric"]
    selection_mode = config["selection"]["mode"]
    best_metric = initialize_best_metric(
        selection_mode
    )
    logger.info(
        f"Selection metric: "
        f"{selection_metric} ({selection_mode})"
    )
    # history.json 的位置
    history_path = run_dir / "history.json"
    # 默认：从空 history 开始
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    if task_type == "classification":
        history["train_accuracy"] = []
        history["val_accuracy"] = []
    # 如果启用了 Resume
    if config["resume"]["enabled"]:
        # 1. 恢复模型和 optimizer
        checkpoint = load_checkpoint(
            config["resume"]["path"],
            model,
            optimizer,
            train_generator=train_generator,
        )
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_metric"]
        logger.info(
            f"Resumed from checkpoint: {config['resume']['path']}"
        )
        logger.info(
            f"Completed epochs: {start_epoch}"
        )
        logger.info(
            f"Best Val Loss: {best_metric:.6f}"
        )
        # 2. 恢复训练历史
        if history_path.exists():
            history = load_history(
                history_path
            )
            logger.info(
                f"Loaded training history: {history_path}"
            )
    logger.info("===== Training Started =====")
    logger.info(f"Config: {config}")
    for epoch in range(start_epoch, epochs):
        if task_type == "regression":
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
        elif task_type == "classification":
            (
                train_loss,
                train_accuracy,
            ) = train_one_epoch_classification(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device=device,
            )
            (
                val_loss,
                val_accuracy,
            ) = evaluate_classification(
                model,
                val_loader,
                loss_fn,
                device=device,
            )
        # =====================
        # Select current metric
        # =====================
        if selection_metric == "val_loss":
            current_metric = val_loss
        elif (
            selection_metric == "val_accuracy"
            and task_type == "classification"
        ):
            current_metric = val_accuracy
        else:
            raise ValueError(
                f"Unsupported selection metric: "
                f"{selection_metric}"
            )    
        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        if task_type == "classification":
            history["train_accuracy"].append(
                train_accuracy
            )
            history["val_accuracy"].append(
                val_accuracy
            )
        if task_type == "regression":
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f} "
                f"LR: {current_lr:.6f}"
            )
        elif task_type == "classification":
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Val Loss: {val_loss:.6f} "
                f"Val Acc: {val_accuracy:.4f} "
                f"LR: {current_lr:.6f}"
            )
        # save best model
        if is_better(
            current_metric,
            best_metric,
            selection_mode,
        ):
            best_metric = current_metric
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch + 1,
                best_metric,
                config=config,
                train_generator=train_generator,
                selection_metric=selection_metric,
                selection_mode=selection_mode,
            )
        # save latest
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch + 1,
            best_metric,
            config=config,
            train_generator=train_generator,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
        )
    logger.info("===== Training Finished =====")
    logger.info(
        f"Best {selection_metric}: "
        f"{best_metric:.6f}"
    )
    saved_history_path = save_history(
        history,
        output_dir=str(run_dir),
    )
    logger.info(
        f"Training history saved to: {saved_history_path}"
    )
    loss_curve_path = plot_loss_curve(
        history,
        output_dir=str(figure_dir),
    )
    if task_type == "classification":
        accuracy_curve_path = (
            plot_accuracy_curve(
                history,
                output_dir=str(
                    figure_dir
                ),
            )
        )
        logger.info(
            f"Accuracy curve saved to: "
            f"{accuracy_curve_path}"
        )
    logger.info(
        f"Loss curve saved to: {loss_curve_path}"
    )
    if task_type == "regression":
        with torch.no_grad():
            print("Learned weight:")
            print(
                model.linear.weight
                .detach()
                .cpu()
            )
            print("Learned bias:")
            print(
                model.linear.bias
                .detach()
                .cpu()
            )

if __name__ == "__main__":
    args = parse_args()
    config = load_config(
        args.config
    )
    main(config)