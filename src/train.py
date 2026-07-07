import csv
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config


def _logits(output):
    return output.logits if hasattr(output, "logits") else output


def _run_epoch(model, loader, criterion, optimizer, device, desc):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = _logits(model(images))
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, epochs, lr,
          scheduler_type="none", patience=5,
          checkpoint_path="checkpoints/model.pt"):
    device = Config.DEVICE
    model.to(device)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = checkpoint_path.with_suffix(".csv")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=0.01,
    )

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"])

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch}/{epochs} [train]",
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device,
            desc=f"Epoch {epoch}/{epochs} [val]",
        )

        if scheduler_type == "cosine":
            scheduler.step()
        elif scheduler_type == "plateau":
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                 f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.2e}",
                 f"{epoch_time:.1f}"]
            )

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_acc": val_acc, "val_loss": val_loss},
                checkpoint_path,
            )
            print(f"  -> saved best checkpoint (val_acc {val_acc:.4f})")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {patience} epochs)")
                break

    return history


def plot_curves(history, out_path="outputs/curves.png", title=""):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
