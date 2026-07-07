import glob
import os
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.data import EVAL_TRANSFORM, FOOD40
from src.gradcam import generate_gradcam
from src.model_b import get_model_b
from src.nutrition import compute_meal_nutrition, grams_from_volume
from src.uncertainty import predict_with_uncertainty
from src.volume import estimate_depth, estimate_food_mask, estimate_volume

CHECKPOINT_CANDIDATES = [
    Config.CHECKPOINTS_DIR / "best_swin.pt",
    Config.CHECKPOINTS_DIR / "model_b.pt",
]

UNCERTAINTY_PCT = {"high": 0.10, "medium": 0.20, "low": 0.35}

MODEL = None
CHECKPOINT_LOADED = False


def load_model():
    global MODEL, CHECKPOINT_LOADED
    if MODEL is not None:
        return MODEL
    model, _ = get_model_b()
    for ckpt_path in CHECKPOINT_CANDIDATES:
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=Config.DEVICE)
            model.load_state_dict(state.get("model_state_dict", state))
            CHECKPOINT_LOADED = True
            print(f"Loaded checkpoint: {ckpt_path}")
            break
    if not CHECKPOINT_LOADED:
        print("WARNING: no checkpoint found, using untrained head.")
    model.to(Config.DEVICE).eval()
    MODEL = model
    return MODEL


def _overlay_heatmap(rgb_float, heatmap):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    return show_cam_on_image(rgb_float, heatmap, use_rgb=True)


def _depth_to_rgb(depth):
    import matplotlib.cm as cm
    return (cm.inferno(depth)[:, :, :3] * 255).astype(np.uint8)


def analyze(image: Image.Image, scale_cm_per_px: float, use_reference: bool):
    if image is None:
        return "Please upload an image.", None, None

    model = load_model()
    image = image.convert("RGB")

    tensor = EVAL_TRANSFORM(image)
    unc = predict_with_uncertainty(model, tensor, n_samples=15)
    dish = FOOD40[unc["predicted_class"]]
    dish_pretty = dish.replace("_", " ").title()

    heatmap = generate_gradcam(model, tensor, unc["predicted_class"])
    rgb_224 = np.asarray(
        image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    ).astype(np.float32) / 255.0
    cam_img = _overlay_heatmap(rgb_224, heatmap)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        depth = estimate_depth(image)
        mask = estimate_food_mask(image)
        scale = scale_cm_per_px if use_reference else None
        vol = estimate_volume(image, food_mask=mask, depth_map=depth,
                              reference_scale_cm_per_pixel=scale)

    depth_img = _depth_to_rgb(depth)

    pct = UNCERTAINTY_PCT[unc["confidence"]]
    lines = [
        f"## {dish_pretty}",
        f"**Model confidence:** {unc['confidence']} "
        f"(p = {unc['predicted_prob']:.2f} ± {unc['predicted_std']:.3f}, "
        f"entropy = {unc['entropy']:.2f})",
        "",
    ]

    if vol["is_metric"]:
        grams = grams_from_volume(dish, vol["volume_cm3"])
        delta = grams * pct
        lines.append(
            f"**Estimated portion:** ~{grams:.0f} g "
            f"(± {delta:.0f} g) — from {vol['volume_cm3']:.0f} cm³"
        )
        meal = compute_meal_nutrition(dish, grams)
        if meal is not None:
            lines += [
                "",
                "| | |",
                "|---|---|",
                f"| **Calories** | {meal['kcal']:.0f} kcal |",
                f"| **Protein** | {meal['protein_g']:.1f} g |",
                f"| **Carbs** | {meal['carbs_g']:.1f} g |",
                f"| **Fat** | {meal['fat_g']:.1f} g |",
                "",
                f"*Nutrition source: USDA — {meal['source']}*",
            ]
        else:
            lines.append(
                "\n*Nutrition lookup unavailable — set the `USDA_API_KEY` "
                "environment variable.*"
            )
    else:
        lines.append(
            f"**Relative volume:** {vol['relative_volume']:.0f} a.u. "
            f"(no reference scale — enable it and set cm/pixel for grams & kcal)"
        )

    if not CHECKPOINT_LOADED:
        lines.append("\n⚠️ *No trained checkpoint found — predictions are random.*")

    return "\n".join(lines), cam_img, depth_img


def build_demo():
    example_files = sorted(
        glob.glob("examples/*.jpg") + glob.glob("examples/*.jpeg")
        + glob.glob("examples/*.png")
    )
    examples = [[f, 0.05, True] for f in example_files] or None

    with gr.Blocks(title="NutriVision", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🍽️ NutriVision\n"
            "Estimate a meal's nutrition from a single photo — dish "
            "classification (Swin-Tiny), MC-dropout confidence, depth-based "
            "portion size, and USDA nutrition facts."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Food photo")
                use_reference = gr.Checkbox(
                    value=True, label="I know the scale (cm per pixel)"
                )
                scale_in = gr.Slider(
                    0.01, 0.20, value=0.05, step=0.005,
                    label="Scale (cm per pixel)",
                    info="≈ plate diameter in cm ÷ plate width in pixels",
                )
                run_btn = gr.Button("Analyze", variant="primary")
                if examples:
                    gr.Examples(examples=examples,
                                inputs=[image_in, scale_in, use_reference])
            with gr.Column(scale=1):
                result_md = gr.Markdown(label="Results")
                with gr.Row():
                    cam_out = gr.Image(label="Grad-CAM — what the model looked at")
                    depth_out = gr.Image(label="Estimated depth map")

        run_btn.click(analyze, inputs=[image_in, scale_in, use_reference],
                      outputs=[result_md, cam_out, depth_out])
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
