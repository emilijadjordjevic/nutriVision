from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config
from src.data import IMAGENET_MEAN, IMAGENET_STD


def _logits(output):
    return output.logits if hasattr(output, "logits") else output


@torch.no_grad()
def _predict_all(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        probs = F.softmax(_logits(model(images)), dim=1)
        all_probs.append(probs.cpu())
        all_preds.append(probs.argmax(dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_preds), torch.cat(all_labels)


def evaluate(model, test_loader, class_names):
    device = Config.DEVICE
    model.to(device)

    _, preds, labels = _predict_all(model, test_loader, device)
    preds, labels = preds.numpy(), labels.numpy()

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(class_names)), zero_division=0
    )
    per_class = pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, class_names, save_path="outputs/confusion_matrix.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(18, 15))
    sns.heatmap(
        cm_norm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Recall (row-normalized)"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def _denormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def show_worst_predictions(model, test_loader, k=10, class_names=None,
                           save_path="outputs/worst_predictions.png"):
    device = Config.DEVICE
    model.to(device)
    model.eval()

    records = []
    for images, labels in tqdm(test_loader, desc="Scanning", leave=False):
        probs = F.softmax(_logits(model(images.to(device))), dim=1).cpu()
        confs, preds = probs.max(dim=1)
        wrong = preds != labels
        for img, label, pred, conf in zip(
            images[wrong], labels[wrong], preds[wrong], confs[wrong]
        ):
            records.append((conf.item(), img, label.item(), pred.item()))

    records.sort(key=lambda r: r[0], reverse=True)
    records = records[:k]

    cols = 5
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax in axes[len(records):]:
        ax.axis("off")

    for ax, (conf, img, label, pred) in zip(axes, records):
        ax.imshow(_denormalize(img))
        true_name = class_names[label] if class_names else str(label)
        pred_name = class_names[pred] if class_names else str(pred)
        ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.2f})",
                     fontsize=9, color="red")
        ax.axis("off")

    fig.suptitle(f"Top {len(records)} most confidently wrong predictions", fontsize=14)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    return save_path
