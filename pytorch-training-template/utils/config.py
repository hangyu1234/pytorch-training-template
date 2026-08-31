import yaml


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":

    config = load_config(
        "configs/base.yaml"
    )

    print(config)

    print(
        config["training"]["lr"]
    )