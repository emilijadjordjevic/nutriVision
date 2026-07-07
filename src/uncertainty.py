import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


def enable_mc_dropout(model) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _logits(output):
    return output.logits if hasattr(output, "logits") else output


def confidence_label(entropy: float) -> str:
    if entropy < 0.5:
        return "high"
    if entropy < 1.0:
        return "medium"
    return "low"


@torch.no_grad()
def predict_with_uncertainty(model, image_tensor, n_samples: int = 20):
    device = Config.DEVICE
    model.to(device)
    enable_mc_dropout(model)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    probs = torch.stack([
        F.softmax(_logits(model(image_tensor)), dim=1).squeeze(0).cpu()
        for _ in range(n_samples)
    ])

    mean_probs = probs.mean(dim=0)
    std_probs = probs.std(dim=0)
    entropy = float(-(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum())

    model.eval()

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "entropy": entropy,
        "predicted_class": int(mean_probs.argmax()),
        "predicted_prob": float(mean_probs.max()),
        "predicted_std": float(std_probs[mean_probs.argmax()]),
        "confidence": confidence_label(entropy),
    }
