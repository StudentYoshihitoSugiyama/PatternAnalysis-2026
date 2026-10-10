import torch
import time
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryConfusionMatrix


def fast_evaluate_model(model, test_loader, device):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

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
    acc_tp_conf = 0.0
    acc_tn_conf = 0.0
    acc_fp_conf = 0.0
    acc_fn_conf = 0.0
    batch_count = 0

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

            probs = torch.softmax(outputs, dim=1)
            # Get the probability assigned to the positive class (class 1)
            prob_positive = probs[:, 1]
            # Get the probability assigned to the negative class (class 0)
            prob_negative = probs[:, 0]

            # 2. Create boolean masks for each outcome category (Binary classification example)
            tp_mask = (preds == 1) & (batch_y == 1)  # True Positive
            tn_mask = (preds == 0) & (batch_y == 0)  # True Negative
            fp_mask = (preds == 1) & (batch_y == 0)  # False Positive
            fn_mask = (preds == 0) & (batch_y == 1)  # False Negative

            # 3. Calculate average probabilities (with checks to avoid NaN if a batch lacks a category)
            avg_tp_conf = prob_positive[tp_mask].mean() if tp_mask.any() else torch.tensor(0.0)
            avg_fp_conf = prob_positive[fp_mask].mean() if fp_mask.any() else torch.tensor(0.0)
            avg_fn_conf = prob_positive[fn_mask].mean() if fn_mask.any() else torch.tensor(0.0)
            avg_tn_conf = prob_negative[tn_mask].mean() if tn_mask.any() else torch.tensor(0.0)

            acc_tp_conf += avg_tp_conf
            acc_tn_conf += avg_tn_conf
            acc_fp_conf += avg_fp_conf
            acc_fn_conf += avg_fn_conf
            batch_count += 1

            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)

    total_avg_tp_conf = acc_tp_conf / batch_count
    total_avg_tn_conf = acc_tn_conf/ batch_count
    total_avg_fp_conf = acc_fp_conf / batch_count
    total_avg_fn_conf = acc_fn_conf / batch_count

    precision = precision_metric.compute()
    recall = recall_metric.compute()
    conf_matrix = confusion_matrix_metric.compute()
    total_time = fast_evaluate_model(model=model, test_loader=test_loader, device=device)
    test_accuracy = test_correct / test_total

