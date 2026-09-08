import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_metric,
    config=None,
    train_generator=None,
    selection_metric=None,
    selection_mode=None,
):
    """
    Save training checkpoint.
    Args:
        path:
            checkpoint save path
        model:
            PyTorch model
        optimizer:
            optimizer
        epoch:
            current epoch
        best_val_loss:
            best validation loss
        config:
            experiment configuration
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "config": config,
    }
    if train_generator is not None:
        checkpoint["train_generator_state"] = (
            train_generator.get_state()
        )
    torch.save(
        checkpoint,
        path,
    )

def load_checkpoint(
    path,
    model,
    optimizer=None,
    train_generator=None,
):
    """
    Load training checkpoint.
    """
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
    if (
        train_generator is not None
        and "train_generator_state" in checkpoint
    ):
        train_generator.set_state(
            checkpoint["train_generator_state"]
        )
    return checkpoint


if __name__ == "__main__":

    import torch.nn as nn

    from models.linear import LinearRegressionModel


    model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )


    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )


    save_checkpoint(
        "test_checkpoint.pt",
        model,
        optimizer,
        epoch=10,
        best_val_loss=0.123,
        config={
            "lr":0.01
        },
    )


    print("Checkpoint saved.")


    new_model = LinearRegressionModel(
        input_dim=3,
        output_dim=1,
    )


    new_optimizer = torch.optim.SGD(
        new_model.parameters(),
        lr=0.01,
    )


    checkpoint = load_checkpoint(
        "test_checkpoint.pt",
        new_model,
        new_optimizer,
    )


    print("Checkpoint loaded.")

    print(
        checkpoint["epoch"]
    )

    print(
        checkpoint["best_val_loss"]
    )