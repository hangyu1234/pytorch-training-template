import torch


def train_val_split(
    features,
    targets,
    val_ratio=0.2,
    seed=42,
):
    """
    Split data into training and validation sets.
    Args:
        features:
            input features
        targets:
            labels
        val_ratio:
            validation proportion
        seed:
            random seed
    Returns:
        train_features,
        train_targets,
        val_features,
        val_targets
    """
    torch.manual_seed(seed)
    num_samples = len(features)
    indices = torch.randperm(
        num_samples
    )
    val_size = int(
        num_samples * val_ratio
    )
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_features = features[train_indices]
    train_targets = targets[train_indices]
    val_features = features[val_indices]
    val_targets = targets[val_indices]
    return (
        train_features,
        train_targets,
        val_features,
        val_targets,
    )

if __name__ == "__main__":

    from dataset import generate_regression_data


    X, y, _, _ = generate_regression_data(
        num_samples=1000,
        num_features=3,
    )


    (
        train_X,
        train_y,
        val_X,
        val_y,
    ) = train_val_split(
        X,
        y,
    )


    print("Train:")
    print(train_X.shape)
    print(train_y.shape)


    print("Validation:")
    print(val_X.shape)
    print(val_y.shape)