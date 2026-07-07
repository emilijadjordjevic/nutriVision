import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.data import FOOD40
from src.evaluate import evaluate, plot_confusion_matrix
from src.gradcam import visualize_gradcam_grid
from src.model_a import count_parameters
from src.nutrition import compute_meal_nutrition, grams_from_volume
from src.uncertainty import predict_with_uncertainty
from src.volume import estimate_depth, estimate_food_mask, estimate_volume

FIG_DIR = Config.OUTPUTS_DIR / "figures"
DPI = 250


def _ensure_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def _logits(output):
    return output.logits if hasattr(output, "logits") else output


def summarize_csv(csv_path):
    df = pd.read_csv(csv_path)
    best_idx = df["val_acc"].idxmax()
    summary = {
        "epochs": len(df),
        "best_val_acc": float(df["val_acc"].max()),
        "best_val_loss": float(df["val_loss"].min()),
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "final_train_acc": float(df["train_acc"].iloc[-1]),
    }
    if "epoch_time_s" in df.columns:
        summary["total_time_min"] = float(df["epoch_time_s"].sum() / 60)
    return summary


def fig_training_curves(csv_a, csv_b, save_name="fig1_training_curves.png"):
    _ensure_dir()
    df_a, df_b = pd.read_csv(csv_a), pd.read_csv(csv_b)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (df, name) in enumerate([(df_a, "Model A — CNN"),
                                      (df_b, "Model B — Swin")]):
        axes[0, col].plot(df["epoch"], df["train_loss"], label="train")
        axes[0, col].plot(df["epoch"], df["val_loss"], label="val")
        axes[0, col].set_title(f"{name}: loss")
        axes[0, col].set_xlabel("epoch")
        axes[0, col].legend()
        axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(df["epoch"], df["train_acc"], label="train")
        axes[1, col].plot(df["epoch"], df["val_acc"], label="val")
        axes[1, col].set_title(f"{name}: accuracy")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].legend()
        axes[1, col].grid(alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / save_name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_confusion_matrix(results_b, save_name="fig2_confusion_matrix.png"):
    _ensure_dir()
    return plot_confusion_matrix(
        results_b["confusion_matrix"], FOOD40, FIG_DIR / save_name
    )


def fig_per_class_f1(results_b, save_name="fig3_per_class_f1.png"):
    _ensure_dir()
    df = results_b["per_class"].sort_values("f1")

    plt.figure(figsize=(10, 12))
    colors = plt.cm.RdYlGn(df["f1"].values)
    plt.barh([c.replace("_", " ") for c in df["class"]], df["f1"], color=colors)
    plt.xlabel("F1 score")
    plt.title("Per-class F1 — Model B (test set)")
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = FIG_DIR / save_name
    plt.savefig(path, dpi=DPI)
    plt.close()
    return path


def fig_gradcam_correct_incorrect(model, test_loader, n_correct=10, n_incorrect=5):
    _ensure_dir()
    device = Config.DEVICE
    model.to(device).eval()

    correct_imgs, correct_labels = [], []
    incorrect_imgs, incorrect_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            preds = _logits(model(images.to(device))).argmax(dim=1).cpu()
            for img, label, pred in zip(images, labels, preds):
                if pred == label and len(correct_imgs) < n_correct:
                    correct_imgs.append(img)
                    correct_labels.append(label)
                elif pred != label and len(incorrect_imgs) < n_incorrect:
                    incorrect_imgs.append(img)
                    incorrect_labels.append(label)
            if len(correct_imgs) >= n_correct and len(incorrect_imgs) >= n_incorrect:
                break

    paths = []
    if correct_imgs:
        paths.append(visualize_gradcam_grid(
            model, torch.stack(correct_imgs),
            FIG_DIR / "fig4a_gradcam_correct.png",
            class_names=FOOD40, labels=torch.stack(correct_labels),
        ))
    if incorrect_imgs:
        paths.append(visualize_gradcam_grid(
            model, torch.stack(incorrect_imgs),
            FIG_DIR / "fig4b_gradcam_incorrect.png",
            class_names=FOOD40, labels=torch.stack(incorrect_labels),
        ))
    return paths


def _table_png(df, save_path, title):
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(df.columns)),
                                    0.6 * len(df) + 1.8))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     rowLabels=df.index, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4c72b0")
            cell.set_text_props(color="white", weight="bold")
    ax.set_title(title, pad=20, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def table_model_comparison(results_a, results_b, model_a, model_b,
                           csv_a=None, csv_b=None,
                           save_name="fig5_model_comparison.png"):
    _ensure_dir()
    time_a = time_b = "—"
    if csv_a is not None:
        s = summarize_csv(csv_a)
        time_a = f"{s['total_time_min']:.0f} min" if "total_time_min" in s else "—"
    if csv_b is not None:
        s = summarize_csv(csv_b)
        time_b = f"{s['total_time_min']:.0f} min" if "total_time_min" in s else "—"

    df = pd.DataFrame({
        "Model A (CNN)": [
            f"{results_a['accuracy']:.4f}",
            f"{results_a['macro_f1']:.4f}",
            f"{results_a['weighted_f1']:.4f}",
            f"{count_parameters(model_a):,}",
            time_a,
        ],
        "Model B (Swin)": [
            f"{results_b['accuracy']:.4f}",
            f"{results_b['macro_f1']:.4f}",
            f"{results_b['weighted_f1']:.4f}",
            f"{count_parameters(model_b):,}",
            time_b,
        ],
    }, index=["Accuracy", "Macro F1", "Weighted F1", "Parameters", "Training time"])

    df.to_csv(FIG_DIR / "table5_model_comparison.csv")
    return _table_png(df, FIG_DIR / save_name, "Model A vs Model B (test set)")


def table_scheduler_experiment(csv_paths: dict,
                               save_name="fig6_scheduler_experiment.png"):
    _ensure_dir()
    rows = {}
    for name, path in csv_paths.items():
        s = summarize_csv(path)
        rows[name] = [
            f"{s['best_val_acc']:.4f}",
            f"{s['best_val_loss']:.4f}",
            s["best_epoch"],
            s["epochs"],
            f"{s['total_time_min']:.0f} min" if "total_time_min" in s else "—",
        ]

    df = pd.DataFrame(
        rows,
        index=["Best val acc", "Best val loss", "Best epoch",
               "Epochs run", "Training time"],
    )
    df.to_csv(FIG_DIR / "table6_scheduler_experiment.csv")
    return _table_png(df, FIG_DIR / save_name,
                      "Model B: LR schedule comparison (validation set)")


def fig_sample_app_outputs(model, raw_test_ds, label_map, n=5,
                           scale_cm_per_pixel=0.05,
                           save_name="fig7_app_outputs.png"):
    _ensure_dir()
    from src.data import EVAL_TRANSFORM

    inv_map = {v: k for k, v in label_map.items()}
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.2 * n),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    for i in range(n):
        item = raw_test_ds[i]
        pil_img = item["image"].convert("RGB")
        true_name = FOOD40[label_map[item["label"]]]

        tensor = EVAL_TRANSFORM(pil_img)
        unc = predict_with_uncertainty(model, tensor, n_samples=15)
        dish = FOOD40[unc["predicted_class"]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            depth = estimate_depth(pil_img)
            mask = estimate_food_mask(pil_img)
            vol = estimate_volume(pil_img, food_mask=mask, depth_map=depth,
                                  reference_scale_cm_per_pixel=scale_cm_per_pixel)

        grams = grams_from_volume(dish, vol["volume_cm3"])
        meal = compute_meal_nutrition(dish, grams)

        lines = [
            f"true: {true_name.replace('_', ' ')}",
            f"pred: {dish.replace('_', ' ')}  "
            f"(p={unc['predicted_prob']:.2f}, conf: {unc['confidence']})",
            f"portion: ~{grams:.0f} g  ({vol['volume_cm3']:.0f} cm³)",
        ]
        if meal is not None:
            lines += [
                f"kcal: {meal['kcal']:.0f}",
                f"protein: {meal['protein_g']:.1f} g   "
                f"carbs: {meal['carbs_g']:.1f} g   fat: {meal['fat_g']:.1f} g",
                f"source: {meal['source'][:50]}",
            ]
        else:
            lines.append("nutrition: unavailable (no USDA_API_KEY)")

        axes[i, 0].imshow(pil_img)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        axes[i, 1].text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                        fontsize=12, family="monospace",
                        transform=axes[i, 1].transAxes)

    fig.suptitle("End-to-end NutriVision outputs", fontsize=15)
    fig.tight_layout()
    path = FIG_DIR / save_name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def generate_all_figures(model_a, model_b, test_loader, raw_test_ds, label_map,
                         csv_a, csv_b, scheduler_csvs=None):
    _ensure_dir()
    paths = []

    results_a = evaluate(model_a, test_loader, FOOD40)
    results_b = evaluate(model_b, test_loader, FOOD40)

    paths.append(fig_training_curves(csv_a, csv_b))
    paths.append(fig_confusion_matrix(results_b))
    paths.append(fig_per_class_f1(results_b))
    paths.extend(fig_gradcam_correct_incorrect(model_b, test_loader))
    paths.append(table_model_comparison(results_a, results_b, model_a, model_b,
                                        csv_a, csv_b))
    if scheduler_csvs:
        paths.append(table_scheduler_experiment(scheduler_csvs))
    paths.append(fig_sample_app_outputs(model_b, raw_test_ds, label_map))

    print(f"{len(paths)} figures written to {FIG_DIR}")
    return paths
