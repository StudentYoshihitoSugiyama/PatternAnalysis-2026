import torch
import copy
import os
import time
import argparse
import torch.optim as optim
from modules import ResNet18, GFNet, criterion, load_checkpoint, get_device, WEIGHT, LABEL_SMOOTHING
from dataset import data_loaders

# Global variable definitions for file paths
PATH_LATEST_MODEL = "Models/latest"
PATH_BEST_MODEL = "Models/best"

# Hyperparameters
EPOCHS = 5
PATIENCE = 10
ETA_MIN = 1e-5
LR = 0.001
WD = 0.01

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

def save_checkpoint(path, model, information):
    """
    Saves model state dictionary and hyperparameter dictionary locally.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "information": information,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Successfully saved checkpoint to '{path}'.")

def train(
    model,
    criterion,
    optimizer,
    scheduler,
    device, 
    train_loader, 
    val_loader
):
    """
    Trains the provided model on the ADNI dataset
    """
    print(f"\nBeginning training on device '{device}' for {EPOCHS} epochs...")

    early_stopping = EarlyStopping()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):      
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # --- Validation Phase (Running Loss & Validation Accuracy) ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # Step Learning Rate Scheduler
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Check Early Stopping
        if early_stopping is not None:
            early_stopping(epoch_val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}. Stopping training early.")
                break

    if early_stopping is not None:
        early_stopping.restore_best_weights(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_training_time = time.time() - start_time

    print("Training finished.")
    print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time / 60:.2f} minutes)")
    return model, total_training_time

def evaluate_model(model,test_loader, criterion, device):
    """
    Evaluates the model on the test dataset
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            test_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_accuracy = test_correct / test_total
    return test_accuracy

def consolidate_information(model, test_accuracy, criterion, optimiser, scheduler, device):
    information = {
        "model": str(model),
        "criterion": f"Cross-Entropy: (weight: {WEIGHT}), (label smoothing: {LABEL_SMOOTHING})",
        "optimiser": "ADAMW: " + str(optimiser.state_dict()["param_groups"]),
        "test_accuracy": str(test_accuracy),
        "scheduler": f"Cosine Annealing LR: (T_MAX: {EPOCHS}), (eta_min: {ETA_MIN})",
        "device": str(device)
    }
    return information
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-18 model and GFNet model training")
    parser.add_argument("--model", default="RN18", choices=["RN18", "GFNET"], help="Model to train: 'RN18' or 'GFNET'")
    args = parser.parse_args()
    mode = args.model

    if mode == "RN18":
        model = ResNet18()
    else:
        model = GFNet()

    latest_path = PATH_LATEST_MODEL + "_" + str(mode)
    best_path = PATH_BEST_MODEL + "_" + str(mode)
    train_loader, val_loader, test_loader = data_loaders()
    optimiser = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=ETA_MIN)
    device = get_device()
    train(model=model, criterion=criterion, optimizer=optimiser, scheduler=scheduler,
            device=device, train_loader=train_loader, val_loader=val_loader)
    current_model_accuracy = evaluate_model(model=model, criterion=criterion, device=device)
    
    information = consolidate_information(
        model=model, test_accuracy=current_model_accuracy, 
        criterion=criterion, optimiser=optimiser,
        scheduler=scheduler, device=device)
    
    save_checkpoint(path=latest_path, model=model, information=information)
    best_model = load_checkpoint(best_path, device=device)

    if best_model == None:
        print("No best model found. Automatically saving as best.")
        save_checkpoint(path=best_path, model=model, information=information)
    elif best_model.get("test_accuracy", -1.0) < current_model_accuracy:
        print(f"Current model ({current_model_accuracy:.4f}%) did beat best model ({best_model:.4f}%).")
        save_checkpoint(path=best_path, model=model, information=information)
    else:
        print(f"Current model ({current_model_accuracy:.4f}%) did not beat best model ({best_model:.4f}%).")

