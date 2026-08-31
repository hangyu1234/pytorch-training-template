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