from pathlib import Path

import pytest
import yaml

from utils.experiment import (
    prepare_run_dir,
    save_config_snapshot,
    save_resume_config_snapshot,
)


def create_config(
    runs_dir,
    resume_enabled=False,
    resume_path=None,
):
    return {
        "experiment": {
            "name": "linear_regression",
            "runs_dir": str(runs_dir),
        },
        "resume": {
            "enabled": resume_enabled,
            "path": resume_path,
        },
        "seed": 42,
    }

def test_prepare_run_dir_creates_new_run(tmp_path):
    runs_dir = tmp_path / "runs"
    config = create_config(
        runs_dir=runs_dir,
        resume_enabled=False,
    )
    run_dir = prepare_run_dir(
        config
    )
    assert run_dir.exists()
    assert (
        run_dir / "checkpoints"
    ).exists()
    assert (
        run_dir / "figures"
    ).exists()
    assert "linear_regression" in run_dir.name

def test_prepare_run_dir_resume_uses_existing_run(
    tmp_path,
):
    run_dir = (
        tmp_path
        / "runs"
        / "existing_run"
    )
    checkpoint_dir = (
        run_dir / "checkpoints"
    )
    checkpoint_dir.mkdir(
        parents=True
    )
    checkpoint_path = (
        checkpoint_dir / "last.pt"
    )
    checkpoint_path.touch()
    config = create_config(
        runs_dir=tmp_path / "runs",
        resume_enabled=True,
        resume_path=str(checkpoint_path),
    )
    resumed_run_dir = prepare_run_dir(
        config
    )
    assert resumed_run_dir == run_dir

def test_prepare_run_dir_missing_checkpoint(
    tmp_path,
):
    missing_checkpoint = (
        tmp_path
        / "runs"
        / "missing_run"
        / "checkpoints"
        / "last.pt"
    )
    config = create_config(
        runs_dir=tmp_path / "runs",
        resume_enabled=True,
        resume_path=str(missing_checkpoint),
    )
    with pytest.raises(
        FileNotFoundError
    ):
        prepare_run_dir(
            config
        )

def test_save_config_snapshot(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = create_config(
        runs_dir=tmp_path / "runs",
    )
    config_path = save_config_snapshot(
        config,
        run_dir,
    )
    assert config_path.exists()
    assert config_path.name == "config.yaml"
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == config

def test_save_resume_config_snapshot(
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = create_config(
        runs_dir=tmp_path / "runs",
        resume_enabled=True,
        resume_path="some/checkpoint.pt",
    )
    config_path = save_resume_config_snapshot(
        config,
        run_dir,
    )
    assert config_path.exists()
    assert (
        config_path.parent.name
        == "resume_configs"
    )
    assert config_path.suffix == ".yaml"
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == config