import subprocess
import sys
from pathlib import Path

import yaml


def test_classification_cli_smoke(
    tmp_path,
):
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )
    runs_dir = (
        tmp_path / "runs"
    )
    config = {
        "seed": 42,
        "task": {
            "type": "classification",
        },
        "experiment": {
            "name": "cli_smoke",
            "runs_dir": str(runs_dir),
        },
        "data": {
            "num_samples": 60,
            "num_features": 4,
            "num_classes": 3,
            "noise_std": 0.8,
            "val_ratio": 0.2,
        },
        "model": {
            "input_dim": 4,
            "hidden_dim": 8,
            "num_classes": 3,
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "lr": 0.05,
            "device": "cpu",
        },
        "resume": {
            "enabled": False,
            "path": None,
        },
        "selection": {
            "metric": "val_accuracy",
            "mode": "max",
        },
    }
    config_path = (
        tmp_path / "config.yaml"
    )
    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        result.stdout
        + "\n"
        + result.stderr
    )
    run_dirs = list(
        runs_dir.glob(
            "*_cli_smoke"
        )
    )
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (
        run_dir / "config.yaml"
    ).exists()
    assert (
        run_dir / "history.json"
    ).exists()
    assert (
        run_dir
        / "checkpoints"
        / "best.pt"
    ).exists()
    assert (
        run_dir
        / "checkpoints"
        / "last.pt"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "loss_curve.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "accuracy_curve.png"
    ).exists()