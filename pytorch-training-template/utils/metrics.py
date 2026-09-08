import json
import os


def save_history(
    history,
    output_dir="outputs",
    filename="history.json",
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )
    path = os.path.join(
        output_dir,
        filename,
    )
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=4,
        )
    return path

def load_history(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        history = json.load(f)
    return history

def classification_accuracy(
    logits,
    labels,
):
    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[batch_size, num_classes]"
        )
    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape "
            "[batch_size]"
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits and labels must have "
            "the same batch size"
        )
    predictions = logits.argmax(
        dim=1
    )
    accuracy = (
        predictions == labels
    ).float().mean().item()
    return accuracy