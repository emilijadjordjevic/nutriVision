from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.data import IMAGENET_MEAN, IMAGENET_STD


class _LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


def _swin_reshape_transform(tensor):
    if tensor.dim() == 4:
        return tensor
    b, n, c = tensor.shape
    h = w = int(n ** 0.5)
    return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2)


def _get_cam_config(model):
    if hasattr(model, "swin"):
        return [model.swin.layernorm], _swin_reshape_transform
    if hasattr(model, "features"):
        return [model.features[-1]], None
    raise ValueError("Unsupported model type for Grad-CAM")


def _denormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float32)


def generate_gradcam(model, image_tensor, target_class=None):
    device = Config.DEVICE
    model.to(device).eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    if target_class is None:
        with torch.no_grad():
            out = model(image_tensor)
            logits = out.logits if hasattr(out, "logits") else out
            target_class = int(logits.argmax(dim=1).item())

    target_layers, reshape_transform = _get_cam_config(model)
    wrapped = _LogitsWrapper(model)

    with GradCAM(model=wrapped, target_layers=target_layers,
                 reshape_transform=reshape_transform) as cam:
        grayscale = cam(input_tensor=image_tensor,
                        targets=[ClassifierOutputTarget(target_class)])

    return grayscale[0]


def visualize_gradcam_grid(model, images, save_path="outputs/gradcam_grid.png",
                           class_names=None, labels=None):
    device = Config.DEVICE
    model.to(device).eval()

    images = images[:10]
    n = images.shape[0]
    cols = min(n, 5)
    blocks = (n + cols - 1) // cols

    with torch.no_grad():
        out = model(images.to(device))
        logits = out.logits if hasattr(out, "logits") else out
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

    fig, axes = plt.subplots(blocks * 2, cols, figsize=(cols * 3.2, blocks * 7))
    axes = np.array(axes).reshape(blocks * 2, cols)

    for ax in axes.flat:
        ax.axis("off")

    for i in range(n):
        rgb = _denormalize(images[i])
        heatmap = generate_gradcam(model, images[i], int(preds[i]))
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)

        block, col = divmod(i, cols)
        ax_orig = axes[block * 2, col]
        ax_over = axes[block * 2 + 1, col]

        pred_name = class_names[int(preds[i])] if class_names else str(int(preds[i]))
        title = f"pred: {pred_name} ({confs[i]:.2f})"
        color = "black"
        if labels is not None and class_names is not None:
            true_name = class_names[int(labels[i])]
            title = f"true: {true_name}\n" + title
            color = "green" if int(labels[i]) == int(preds[i]) else "red"

        ax_orig.imshow(rgb)
        ax_orig.set_title(title, fontsize=9, color=color)
        ax_over.imshow(overlay)

    fig.suptitle("Grad-CAM — originals (upper row) and heatmap overlays (lower row)",
                 fontsize=13)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    return save_path
