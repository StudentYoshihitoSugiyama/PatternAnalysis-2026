import torch
import time
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryConfusionMatrix



def evaluate_model(model,test_loader, criterion, device):
    """
    Evaluates the model on the test dataset
    """
    precision_metric = BinaryPrecision(num_classes=2, average="macro")
    recall_metric = BinaryRecall(num_classes=2, average="macro")
    confusion_matrix_metric = BinaryConfusionMatrix(num_classes=2)

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            test_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)

            precision_metric.update(preds, batch_y.to(device))
            recall_metric.update(preds, batch_y.to(device))
            confusion_matrix_metric.update(preds, batch_y.to(device))

            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)
    end_time = time.time()

    precision = precision_metric.compute()
    recall = recall_metric.compute()
    conf_matrix = confusion_matrix_metric.compute()
    total_time = end_time - start_time
    test_accuracy = test_correct / test_total

