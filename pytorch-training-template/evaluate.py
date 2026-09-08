import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from data.dataset import (
    generate_regression_data,
    RegressionDataset,
)

from data.classification import (
    generate_classification_data,
    ClassificationDataset,
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

from models.mlp import (
    MLPClassifier,
)

from engine.trainer import (
    evaluate,
    evaluate_classification,
)

from utils.checkpoint import (
    load_checkpoint,
)

from utils.seed import (
    set_seed,
)

from utils.config import (
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to experiment run directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Checkpoint to evaluate",
    )
    return parser.parse_args()

def main(
    run_dir,
    checkpoint_name,
):
    # =====================
    # 0. Run directory
    # =====================
    run_dir = Path(run_dir)
    config_path = (
        run_dir / "config.yaml"
    )
    checkpoint_path = (
        run_dir
        / "checkpoints"
        / f"{checkpoint_name}.pt"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )
    # =====================
    # 1. Load config
    # =====================
    config = load_config(
        config_path
    )
    task_type = config["task"]["type"]
    if task_type not in {
        "regression",
        "classification",
    }:
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )
    print(
        "Task type:",
        task_type,
    )
    # =====================
    # 2. Reproducibility
    # =====================
    set_seed(
        config["seed"]
    )
    # =====================
    # 3. Device
    # =====================
    device_config = (
        config["training"]["device"]
    )
    if device_config == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            device_config
        )
    print(
        "Using device:",
        device,
    )
    # =====================
    # 4. Prepare data
    # =====================
    if task_type == "regression":
        X, y, _, _ = (
            generate_regression_data(
                num_samples=config["data"]["num_samples"],
                num_features=config["data"]["num_features"],
                seed=config["seed"],
            )
        )
    elif task_type == "classification":
        X, y, _ = (
            generate_classification_data(
                num_samples=config["data"]["num_samples"],
                num_features=config["data"]["num_features"],
                num_classes=config["data"]["num_classes"],
                noise_std=config["data"]["noise_std"],
                seed=config["seed"],
            )
        )
    (
        _,
        _,
        val_X,
        val_y,
    ) = train_val_split(
        X,
        y,
        val_ratio=config["data"]["val_ratio"],
        seed=config["seed"],
    )
    # =====================
    # 5. Dataset
    # =====================
    if task_type == "regression":
        val_dataset = RegressionDataset(
            val_X,
            val_y,
        )
    elif task_type == "classification":
        val_dataset = ClassificationDataset(
            val_X,
            val_y,
        )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    # =====================
    # 6. Model + Loss
    # =====================
    if task_type == "regression":
        model = LinearRegressionModel(
            input_dim=config["model"]["input_dim"],
            output_dim=config["model"]["output_dim"],
        )
        loss_fn = nn.MSELoss()
    elif task_type == "classification":
        model = MLPClassifier(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )
        loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    # =====================
    # 7. Load checkpoint
    # =====================
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
    )
    print(
        "Loaded checkpoint:",
        checkpoint_path,
    )
    print(
        "Epoch:",
        checkpoint["epoch"],
    )
    print(
    "Selection metric:",
    checkpoint["selection_metric"],
    )
    print(
        "Selection mode:",
        checkpoint["selection_mode"],
    )
    print(
        "Best metric:",
        checkpoint["best_metric"],
    )
    # =====================
    # 8. Evaluate
    # =====================
    if task_type == "regression":
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device=device,
        )
        print(
            "Evaluation loss:",
            val_loss,
        )
        metrics = {
            "split": "validation",
            "task_type": task_type,
            "checkpoint": str(
                checkpoint_path.relative_to(
                    run_dir
                )
            ),
            "epoch": checkpoint["epoch"],
            "best_val_loss_so_far": (
                checkpoint["best_val_loss"]
            ),
            "metrics": {
                "mse": val_loss,
            },
        }
    elif task_type == "classification":
        (
            val_loss,
            val_accuracy,
        ) = evaluate_classification(
            model,
            val_loader,
            loss_fn,
            device=device,
        )
        print(
            "Evaluation loss:",
            val_loss,
        )
        print(
            "Evaluation accuracy:",
            val_accuracy,
        )
        metrics = {
            "split": "validation",
            "task_type": task_type,
            "checkpoint": str(
                checkpoint_path.relative_to(
                    run_dir
                )
            ),
            "epoch": checkpoint["epoch"],
            "selection": {
                "metric": checkpoint["selection_metric"],
                "mode": checkpoint["selection_mode"],
                "best_metric": checkpoint["best_metric"],
            },
            "metrics": {
                "cross_entropy_loss": val_loss,
                "accuracy": val_accuracy,
            },
        }
    # =====================
    # 9. Save metrics
    # =====================
    evaluation_dir = (
        run_dir / "evaluation"
    )
    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    metrics_path = (
        evaluation_dir
        / f"{checkpoint_name}_metrics.json"
    )
    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )
    print(
        "Evaluation metrics saved to:",
        metrics_path,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.run,
        args.checkpoint,
    )