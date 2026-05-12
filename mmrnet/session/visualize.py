import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import torch

def plot_confusion_and_metrics(Y_pred, Y_true, output_dir, class_labels=None, display=False):
    """
    Generates a normalized confusion matrix and per-class metrics plot, saving them to the given directory.

    Args:
        Y_pred (np.ndarray): Model predictions, either as class indices or logits/probabilities.
        Y_true (np.ndarray): Ground truth labels, as class indices.
        output_dir (str): Path to the output folder where plots will be saved.
        class_labels (list of str, optional): List of class label names.
    """
    _, top3_indices = torch.topk(Y_pred, k=3, dim=1)
    top3_correct = (top3_indices == Y_true.unsqueeze(-1)).any(dim=1).float()
    top3_accuracy = top3_correct.mean().item()

    if isinstance(Y_pred, torch.Tensor):
        Y_pred = Y_pred.clone().detach().cpu().numpy()
    if isinstance(Y_true, torch.Tensor):
        Y_true = Y_true.clone().detach().cpu().numpy()
    os.makedirs(output_dir, exist_ok=True)

    # Convert predicted probabilities to class labels if needed
    if len(Y_pred.shape) > 1:
        Y_pred = np.argmax(Y_pred, axis=1)

    if len(Y_true.shape) > 1:
        Y_true = np.argmax(Y_true, axis=1)

    # 1. Normalized Confusion Matrix
    cm = confusion_matrix(Y_true, Y_pred, normalize='true')
    # Determine tick labels
    if class_labels is None:
        num_classes = cm.shape[0]
        class_labels = list(range(num_classes))
    
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(cm, annot=False, cmap='Blues', square=True,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"))
    plt.close()

    # 2. Per-class precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0)
    classes = class_labels if class_labels else list(range(len(precision)))
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1 Score')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision, Recall, and F1 Score')
    ax.set_ylim(0, 1)  # Set y-axis range to [0, 1]
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)  # Enable grid for y-axis
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"))
    plt.close()

    # 3. Classification Report (console + optional file)
    report = classification_report(
        Y_true, Y_pred, digits=3, zero_division=0,
        target_names=[str(label) for label in class_labels]
    )

    # Calculate top-3 accuracy
    
    report += f"\nTop-3 Accuracy: {top3_accuracy:.3f}\n"

    if display:
        print("\nClassification Report:\n")
        print(report)

    # Save report to a text file too
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
