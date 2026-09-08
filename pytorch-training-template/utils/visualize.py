import json
import os

import matplotlib.pyplot as plt


def load_history(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        history = json.load(f)
    return history

def plot_loss_curve(
    history,
    output_dir="outputs/figures",
    filename="loss_curve.png",
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    plt.figure(
        figsize=(8, 5)
    )
    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
    )
    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(
        output_dir,
        filename,
    )
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    return save_path

def plot_accuracy_curve(
    history,
    output_dir="outputs/figures",
    filename="accuracy_curve.png",
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )
    if (
        "train_accuracy" not in history
        or "val_accuracy" not in history
    ):
        raise ValueError(
            "history does not contain "
            "classification accuracy"
        )
    epochs = history["epoch"]
    train_accuracy = (
        history["train_accuracy"]
    )
    val_accuracy = (
        history["val_accuracy"]
    )
    plt.figure(
        figsize=(8, 5)
    )
    plt.plot(
        epochs,
        train_accuracy,
        label="Train Accuracy",
    )
    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy",
    )
    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "Accuracy"
    )
    plt.title(
        "Training and Validation Accuracy"
    )
    plt.legend()
    plt.grid(
        True
    )
    save_path = os.path.join(
        output_dir,
        filename,
    )
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    return save_path


if __name__ == "__main__":

    history = load_history(
        "outputs/history.json"
    )

    save_path = plot_loss_curve(
        history
    )

    print(
        f"Loss curve saved to: {save_path}"
    )