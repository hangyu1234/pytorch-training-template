from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    generator=None,
):
    """
    Create PyTorch DataLoader.
    Args:
        dataset:
            PyTorch Dataset object
        batch_size:
            number of samples in one batch
        shuffle:
            whether shuffle data every epoch
    Returns:
        DataLoader
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator
    )
    return loader


if __name__ == "__main__":

    from dataset import (
        RegressionDataset,
        generate_regression_data,
    )


    X, y, _, _ = generate_regression_data(
        num_samples=1000,
        num_features=3,
    )


    dataset = RegressionDataset(
        X,
        y,
    )


    loader = create_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
    )


    for batch_x, batch_y in loader:

        print("Batch X shape:")
        print(batch_x.shape)

        print("Batch y shape:")
        print(batch_y.shape)

        break