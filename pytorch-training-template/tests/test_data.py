import torch

from data import RegressionDataset, split_dataset


X = torch.randn(100, 5)
y = torch.randn(100)

dataset = RegressionDataset(X, y)

# 1. Dataset 长度
assert len(dataset) == 100

# 2. 单个样本形状
x0, y0 = dataset[0]

assert x0.shape == torch.Size([5])
assert y0.shape == torch.Size([1])

# 3. train / val / test 大小
train_set, val_set, test_set = split_dataset(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)

assert len(train_set) == 70
assert len(val_set) == 15
assert len(test_set) == 15

# 4. 固定 seed 应该产生相同划分
train_set_2, val_set_2, test_set_2 = split_dataset(
    dataset,
    seed=42,
)

assert train_set.indices == train_set_2.indices
assert val_set.indices == val_set_2.indices
assert test_set.indices == test_set_2.indices

print("All data tests passed!")