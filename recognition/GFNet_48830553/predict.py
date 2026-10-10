import torch
import copy


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



class EarlyStopping:
    """
    Early stopping monitor to halt training when validation loss stops improving,
    preventing overfitting and restoring the optimal model weights.
    """
    def __init__(self, patience=5, min_delta=1e-3, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = torch.inf
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f" -> Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}. Stashing best weights.")
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f" -> EarlyStopping counter: {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.4f} at epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored best model weights from epoch {self.best_epoch} (Val Loss: {self.best_loss:.4f}).")