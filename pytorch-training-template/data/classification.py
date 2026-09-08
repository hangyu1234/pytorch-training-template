import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
    ):
        self.features = torch.as_tensor(
            features,
            dtype=torch.float32,
        )
        self.labels = torch.as_tensor(
            labels,
            dtype=torch.long,
        )
        if len(self.features) != len(self.labels):
            raise ValueError(
                "features and labels must have "
                "the same number of samples"
            )
        if self.labels.ndim != 1:
            raise ValueError(
                "labels must have shape [N]"
            )
    def __len__(self):
        return len(self.features)
    def __getitem__(
        self,
        index,
    ):
        return (
            self.features[index],
            self.labels[index],
        )


def generate_classification_data(
    num_samples=1000,
    num_features=4,
    num_classes=3,
    noise_std=0.8,
    seed=42,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    # =====================
    # 1. Create class labels
    # =====================
    labels = (
        torch.arange(num_samples)
        % num_classes
    )
    # =====================
    # 2. Create class centers
    # =====================
    class_centers = (
        torch.randn(
            num_classes,
            num_features,
            generator=generator,
        )
        * 3.0
    )
    # =====================
    # 3. Generate samples
    # =====================
    noise = (
        noise_std
        * torch.randn(
            num_samples,
            num_features,
            generator=generator,
        )
    )
    features = (
        class_centers[labels]
        + noise
    )
    # =====================
    # 4. Shuffle samples
    # =====================
    permutation = torch.randperm(
        num_samples,
        generator=generator,
    )
    features = features[
        permutation
    ]
    labels = labels[
        permutation
    ]
    return (
        features,
        labels,
        class_centers,
    )