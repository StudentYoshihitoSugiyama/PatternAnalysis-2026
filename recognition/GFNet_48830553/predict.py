import torch


def get_device():
    """
    Selects the optimal hardware device available.
    Attempts to use GPU (CUDA / DirectML / MPS), falling back to CPU if unavailable.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")