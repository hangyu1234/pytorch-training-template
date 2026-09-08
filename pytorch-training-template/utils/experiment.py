from datetime import datetime
from pathlib import Path

import yaml


def prepare_run_dir(config):
    """
    Fresh run:
        create a new timestamped run directory

    Resume:
        reuse the run directory that contains the checkpoint
    """
    if config["resume"]["enabled"]:
        checkpoint_path = Path(
            config["resume"]["path"]
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        # Example:
        #
        # runs/20260905_140000_linear_regression/
        #     checkpoints/
        #         last.pt
        #
        # checkpoint_path.parent
        #     -> checkpoints/
        #
        # checkpoint_path.parent.parent
        #     -> experiment run directory
        run_dir = checkpoint_path.parent.parent
        return run_dir
    # Fresh experiment
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    experiment_name = config["experiment"]["name"]
    runs_dir = Path(
        config["experiment"]["runs_dir"]
    )
    run_name = (
        f"{timestamp}_{experiment_name}"
    )
    run_dir = runs_dir / run_name
    # Create experiment directories
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    (run_dir / "checkpoints").mkdir()
    (run_dir / "figures").mkdir()
    return run_dir


def save_config_snapshot(
    config,
    run_dir,
):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    return config_path

def save_resume_config_snapshot(
    config,
    run_dir,
):
    run_dir = Path(run_dir)
    resume_config_dir = (
        run_dir / "resume_configs"
    )
    resume_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    config_path = (
        resume_config_dir
        / f"{timestamp}.yaml"
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
            allow_unicode=True,
        )
    return config_path


if __name__ == "__main__":
    from utils.config import load_config

    config = load_config(
        "configs/base.yaml"
    )

    run_dir = prepare_run_dir(
        config
    )

    config_path = save_config_snapshot(
        config,
        run_dir,
    )

    print("Run directory:")
    print(run_dir)

    print("Config snapshot:")
    print(config_path)