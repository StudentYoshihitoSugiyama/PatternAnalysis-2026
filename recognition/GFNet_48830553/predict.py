import torch
import time
import argparse
import os
from modules import load_checkpoint, get_device, ResNet18, GFNet
from dataset import data_loaders
from train import PATH_BEST_MODEL, PATH_LATEST_MODEL
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy
import matplotlib.pyplot as plt


def fast_evaluate_model(model, test_loader, device):
    """
    Computes total inference latency across the test dataset with CUDA acceleration.
    """
    model.eval()
    use_cuda = device.type == "cuda"
    device_type = "cuda" if use_cuda else "cpu"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if use_cuda:
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(batch_X)
                _, _ = torch.max(outputs, 1)
    if use_cuda:
        torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

def evaluate_model(model, test_loader, device, output_path):
    """
    Evaluates the model on the test dataset with CUDA and mixed-precision acceleration.
    """
    precision_metric = BinaryPrecision(device=device)
    recall_metric = BinaryRecall(device=device)
    confusion_matrix_metric = BinaryConfusionMatrix(device=device)
    accuracy_metric = BinaryAccuracy(device=device)

    use_cuda = device.type == "cuda"
    device_type = "cuda" if use_cuda else "cpu"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

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
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(batch_X)

            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)

            precision_metric.update(preds, batch_y)
            recall_metric.update(preds, batch_y)
            confusion_matrix_metric.update(preds, batch_y)
            accuracy_metric.update(preds, batch_y)

            # 2. Create boolean masks for each outcome category
            tp_mask = (preds == 1) & (batch_y == 1)  # True Positive
            tn_mask = (preds == 0) & (batch_y == 0)  # True Negative
            fp_mask = (preds == 1) & (batch_y == 0)  # False Positive
            fn_mask = (preds == 0) & (batch_y == 1)  # False Negative

            # 3. Calculate average probabilities (with checks to avoid NaN if a batch lacks a category)
            avg_tp_conf = max_probs[tp_mask].mean().item() if tp_mask.any() else 0.0
            avg_fp_conf = max_probs[fp_mask].mean().item() if fp_mask.any() else 0.0
            avg_fn_conf = max_probs[fn_mask].mean().item() if fn_mask.any() else 0.0
            avg_tn_conf = max_probs[tn_mask].mean().item() if tn_mask.any() else 0.0

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
    total_true_conf = (total_avg_tp_conf + total_avg_tn_conf) / 2.0
    total_false_conf = (total_avg_fp_conf + total_avg_fn_conf) / 2.0

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
    if model_version == "RN18":
        output_path = version + "_" + model_version
        model = ResNet18()
    else:
        output_path = version + "_" + model_version
        model = GFNet()

    if version == "BEST":
            model_path = PATH_BEST_MODEL + "_" + model_version + ".pt"
    else:
        model_path = PATH_LATEST_MODEL + "_" + model_version + ".pt"

    result_path = "Results/" + output_path + "_information.txt"
    graph_path = "Results/" + output_path + "_graph.pdf"

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(graph_path) or ".", exist_ok=True)

    print("Loading model...")
    model_info = load_checkpoint(model_path, device=device)
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
    evaluate_model(model=model, test_loader=test_loader, device=device, output_path=result_path)
    
    print("Testing Visualisation")
    print(f"Result Path: {result_path}")
    print(f"Graph Path: {graph_path}")
    
    loss_info = model_info.get("information").get("losses")
    train_losses, val_losses, epochs = loss_info
    train_losses = train_losses
    val_losses = val_losses
    epochs = epochs
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title(f"{model_version} ({version}) Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(graph_path)
    plt.close()
