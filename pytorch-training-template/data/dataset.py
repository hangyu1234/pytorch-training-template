import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

        if len(self.features) != len(self.targets):
            raise ValueError("features and targets must have the same number of samples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def generate_regression_data(
    num_samples=1000,
    num_features=3,
    noise_std=0.1,
    seed=42,
):
    """
    Generate synthetic linear regression data:
        y = Xw + b + noise
    Args:
        num_samples:
            number of samples
        num_features:
            number of input features
        noise_std:
            standard deviation of noise
        seed:
            random seed
    """
    torch.manual_seed(seed)
    # input features
    features = torch.randn(
        num_samples,
        num_features
    )
    # ground truth parameters
    true_weight = torch.randn(
        num_features,
        1
    )
    true_bias = torch.randn(1)
    # random noise
    noise = noise_std * torch.randn(
        num_samples,
        1
    )
    # generate targets
    targets = (
        features @ true_weight
        + true_bias
        + noise
    )
    return (
        features,
        targets,
        true_weight,
        true_bias,
    )


if __name__ == "__main__":

    X, y, w, b = generate_regression_data(
        num_samples=1000,
        num_features=3,
    )


    dataset = RegressionDataset(
        X,
        y
    )


    print("Dataset size:", len(dataset))

    print("Feature shape:", X.shape)

    print("Target shape:", y.shape)

    print("True weight:")
    print(w)

    print("True bias:")
    print(b)

    print("First sample:")
    print(dataset[0])