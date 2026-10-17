import torch
import time
import argparse
from modules import load_checkpoint, get_device, ResNet18, GFNet
from dataset import data_loaders
from train import PATH_BEST_MODEL, PATH_LATEST_MODEL
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy
import numpy as np
import matplotlib.pyplot as plt


def fast_evaluate_model(model, test_loader, device):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, _ = torch.max(outputs, 1)
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

def evaluate_model(model, test_loader, device, output_path):
    """
    Evaluates the model on the test dataset
    """
    precision_metric = BinaryPrecision(average="macro", device=device)
    recall_metric = BinaryRecall(average="macro", device=device)
    confusion_matrix_metric = BinaryConfusionMatrix(device=device)
    accuracy_metric = BinaryAccuracy(device=device)

    model.eval()
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

            _, preds = torch.max(outputs, 1)

            precision_metric.update(preds, batch_y)
            recall_metric.update(preds, batch_y)
            confusion_matrix_metric.update(preds, batch_y)
            accuracy_metric.update(preds, batch_y)

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
    total_true_conf = torch.mean((total_avg_tp_conf, total_avg_tn_conf))
    total_false_conf = torch.mean((total_avg_fp_conf, total_avg_fn_conf))

    precision = precision_metric.compute()
    recall = recall_metric.compute()
    conf_matrix = confusion_matrix_metric.compute()
    test_accuracy = accuracy_metric.compute()
    total_time = fast_evaluate_model(model=model, test_loader=test_loader, device=device)

    print(f"True positive confidence: {total_avg_tp_conf:.2f}")
    print(f"True negative confidence: {total_avg_tn_conf:.2f}")
    print(f"False positive confidence: {total_avg_fp_conf:.2f}")
    print(f"False negative confidence: {total_avg_fn_conf:.2f}")
    print(f"True confidence: {total_true_conf:.2f}")
    print(f"False confidence: {total_false_conf:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Inference time: {total_time:.2f}")
    print(f"Confusion Matrix: {conf_matrix}")

    with open(output_path, "w") as f:
        f.write(f"True positive confidence: {total_avg_tp_conf:.2f}\n")
        f.write(f"True negative confidence: {total_avg_tn_conf:.2f}\n")
        f.write(f"False positive confidence: {total_avg_fp_conf:.2f}\n")
        f.write(f"False negative confidence: {total_avg_fn_conf:.2f}\n")
        f.write(f"True confidence: {total_true_conf:.2f}\n")
        f.write(f"False confidence: {total_false_conf:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}\n")
        f.write(f"Inference time: {total_time:.2f}\n")
        f.write(f"Confusion Matrix: {conf_matrix}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-18 model and GFNet model testing")
    parser.add_argument("--model", default="RN18", choices=["RN18", "GFNET"], help="Model to test: 'RN18' or 'GFNET'")
    parser.add_argument("--version", default="BEST", choices=["BEST", "LATEST"], help="Version to test: 'BEST' or 'LATEST'")
    args = parser.parse_args()
    model_version = args.model
    version = args.version

    print("Obtaining device...")
    device = get_device()

    print("Obtaining model path...")
    if version == "BEST" and model_version == "RN18":
        path = PATH_BEST_MODEL + "_" + model_version
        model = ResNet18()
    elif version == "LATEST" and model_version == "RN18":
        path = PATH_LATEST_MODEL + "_" + model_version
        model = ResNet18()
    elif version == "BEST" and model_version == "GFNET":
        path = PATH_BEST_MODEL + "_" + model_version
        model = GFNet()
    else:
        path = PATH_LATEST_MODEL + "_" + model_version
        model = GFNet()

    output_path = path + "_information.txt"
    graph_path = path + "_graph.pdf"

    print("Loading model...")
    model_info = load_checkpoint(path, device=device)
    if model_info == None:
        print("Model is None")
        raise ValueError
    
    model_weights = model_info.get("model_state_dict", None)
    if model_weights == None:
        print("Model weights is None")
        raise ValueError
    model.load_state_dict(model_weights)
    model.to(device=device)

    print("Constructing data loaders...")
    _, _, test_loader = data_loaders()

    print("Begin testing...")
    evaluate_model(model=model, test_loader=test_loader, device=device, output_path=output_path)
    print("Testing Visualisation")
    
    loss_info = model_info.get("information").get("losses")
    train_losses, val_losses, epochs = loss_info
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    epochs = np.array(epochs)
    plt.plot(train_losses, epochs)
    plt.plot(val_losses, epochs)
    plt.savefig(graph_path)