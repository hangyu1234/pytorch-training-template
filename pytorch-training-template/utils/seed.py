import random
import numpy as np
import torch



def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed:
            random seed value
    """
    # Python random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # PyTorch CPU random
    torch.manual_seed(seed)
    # PyTorch GPU random
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    set_seed(42)


    print("Random:")
    print(random.random())


    print("NumPy:")
    print(np.random.rand())


    print("Torch:")
    print(torch.randn(3))