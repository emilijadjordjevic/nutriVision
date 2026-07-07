import warnings

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from src.config import Config

DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"
SAM2_CHECKPOINT = "facebook/sam2-hiera-tiny"

_depth_model = None
_depth_processor = None
_sam2_model = None
_sam2_processor = None


def _get_depth_model():
    global _depth_model, _depth_processor
    if _depth_model is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        _depth_processor = AutoImageProcessor.from_pretrained(DEPTH_CHECKPOINT)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_CHECKPOINT)
        _depth_model.to(Config.DEVICE).eval()
    return _depth_model, _depth_processor


def _get_sam2():
    global _sam2_model, _sam2_processor
    if _sam2_model is None:
        from transformers import Sam2Model, Sam2Processor
        _sam2_processor = Sam2Processor.from_pretrained(SAM2_CHECKPOINT)
        _sam2_model = Sam2Model.from_pretrained(SAM2_CHECKPOINT)
        _sam2_model.to(Config.DEVICE).eval()
    return _sam2_model, _sam2_processor


@torch.no_grad()
def estimate_depth(image: Image.Image) -> np.ndarray:
    model, processor = _get_depth_model()
    inputs = processor(images=image, return_tensors="pt").to(Config.DEVICE)
    depth = model(**inputs).predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def _otsu_threshold(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)

    cum_mean = np.cumsum(hist * bin_centers)
    mean_bg = np.where(weight_bg > 0, cum_mean / np.maximum(weight_bg, 1e-12), 0)
    mean_fg = np.where(
        weight_fg > 0,
        (cum_mean[-1] - cum_mean) / np.maximum(weight_fg, 1e-12),
        0,
    )

    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_var[~valid] = -1
    return float(bin_centers[np.argmax(between_var)])


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == (np.argmax(sizes) + 1))


def _otsu_saturation_mask(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    saturation = hsv[:, :, 1]
    threshold = _otsu_threshold(saturation)
    return _clean_mask(saturation > threshold)


@torch.no_grad()
def _sam2_mask(image: Image.Image) -> np.ndarray:
    model, processor = _get_sam2()
    w, h = image.size
    center = [[[[w / 2, h / 2]]]]

    inputs = processor(
        images=image, input_points=center, input_labels=[[[1]]],
        return_tensors="pt",
    ).to(Config.DEVICE)
    outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    scores = outputs.iou_scores.cpu().squeeze()
    best = masks.squeeze(0)[scores.argmax()].numpy() > 0.5
    return _clean_mask(best)


def estimate_food_mask(image: Image.Image, use_sam2: bool = True) -> np.ndarray:
    if use_sam2:
        try:
            return _sam2_mask(image)
        except Exception as e:
            warnings.warn(
                f"SAM2 unavailable ({type(e).__name__}: {e}); "
                "falling back to Otsu saturation mask."
            )
    return _otsu_saturation_mask(image)


def estimate_volume(image: Image.Image, food_mask: np.ndarray = None,
                    reference_scale_cm_per_pixel: float = None,
                    h_max_cm: float = 5.0, depth_map: np.ndarray = None):
    if food_mask is None:
        food_mask = estimate_food_mask(image)
    food_mask = food_mask.astype(bool)

    if depth_map is None:
        depth_map = estimate_depth(image)

    if not food_mask.any():
        warnings.warn("Empty food mask; volume is zero.")
        return {"relative_volume": 0.0, "volume_cm3": None, "is_metric": False,
                "mask_area_px": 0, "mean_height_rel": 0.0}

    ring = ndimage.binary_dilation(food_mask, iterations=15) & ~food_mask
    background_depth = float(np.median(depth_map[ring])) if ring.any() \
        else float(np.median(depth_map[~food_mask]))

    height_rel = np.clip(depth_map - background_depth, 0.0, None) * food_mask
    relative_volume = float(height_rel.sum())
    mean_height_rel = float(height_rel[food_mask].mean())
    peak = float(height_rel.max())

    result = {
        "relative_volume": relative_volume,
        "mask_area_px": int(food_mask.sum()),
        "mean_height_rel": mean_height_rel,
        "is_metric": False,
        "volume_cm3": None,
    }

    if reference_scale_cm_per_pixel is None:
        warnings.warn(
            "No reference scale provided; returning relative volume "
            "(arbitrary units), not cm³."
        )
        return result

    s = reference_scale_cm_per_pixel
    height_cm = (height_rel / peak * h_max_cm) if peak > 1e-8 else height_rel
    result["volume_cm3"] = float((height_cm * s * s).sum())
    result["is_metric"] = True
    return result