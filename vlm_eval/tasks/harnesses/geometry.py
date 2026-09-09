"""Utility helpers for projecting bounding boxes into the model's preprocessing frame."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, val))


def _as_resolution(transform_info: Dict[str, Sequence[int]], image_wh: Tuple[int, int]) -> Tuple[float, float]:
    target = transform_info.get("target_resolution")
    if isinstance(target, Sequence) and len(target) == 2:
        h, w = target
        if h and w:
            return float(h), float(w)
    # Fallback to original resolution if metadata is missing
    return float(image_wh[1]), float(image_wh[0])


def _normalize_letterbox(bbox_xyxy: Sequence[float], image_wh: Tuple[int, int]) -> List[float]:
    width, height = map(float, image_wh)
    max_wh = max(width, height)
    pad_x = (max_wh - width) / 2 if width < max_wh else 0.0
    pad_y = (max_wh - height) / 2 if height < max_wh else 0.0

    transformed = [
        (bbox_xyxy[0] + pad_x) / max_wh,
        (bbox_xyxy[1] + pad_y) / max_wh,
        (bbox_xyxy[2] + pad_x) / max_wh,
        (bbox_xyxy[3] + pad_y) / max_wh,
    ]
    return [_clamp01(coord) for coord in transformed]


def _normalize_resize_naive(bbox_xyxy: Sequence[float], image_wh: Tuple[int, int]) -> List[float]:
    width, height = map(float, image_wh)
    transformed = [
        bbox_xyxy[0] / width,
        bbox_xyxy[1] / height,
        bbox_xyxy[2] / width,
        bbox_xyxy[3] / height,
    ]
    return [_clamp01(coord) for coord in transformed]


def _normalize_resize_crop(
    bbox_xyxy: Sequence[float], image_wh: Tuple[int, int], transform_info: Dict[str, Sequence[int]]
) -> List[float]:
    width, height = map(float, image_wh)
    target_h, target_w = _as_resolution(transform_info, image_wh)

    # Uniformly scale until both dimensions exceed the model's target resolution
    scale = max(target_w / width, target_h / height)
    scaled_w, scaled_h = width * scale, height * scale

    scaled_bbox = [coord * scale for coord in bbox_xyxy]
    offset_x = max(0.0, (scaled_w - target_w) / 2.0)
    offset_y = max(0.0, (scaled_h - target_h) / 2.0)

    transformed = [
        (scaled_bbox[0] - offset_x) / target_w,
        (scaled_bbox[1] - offset_y) / target_h,
        (scaled_bbox[2] - offset_x) / target_w,
        (scaled_bbox[3] - offset_y) / target_h,
    ]
    return [_clamp01(coord) for coord in transformed]


def normalize_bbox_for_transform(
    bbox_xyxy: Sequence[float], image_wh: Tuple[int, int], transform_info: Dict[str, Sequence[int]] | None
) -> List[float]:
    """Project an absolute COCO/OCID box into the coordinate frame the MODEL EMITS.

    That frame is ALWAYS the centered pad-to-square ("letterbox") frame, for every config, and it is
    deliberately independent of `transform_info["policy"]`. `transform_info` is accepted (and ignored)
    only to keep the call signature stable for the two scorers.

    WHY (verified 2026-09-08, see docs/reference/GROUNDING_COORDINATE_FRAME.md):

    A VLM's box coordinates are learned from the TEXT of the instruction-tuning mixture, not from how
    its vision backbone happens to pad pixels. In `llava_v1_5_mix665k.json` (RefCOCO 48K + VG 86K) the
    box strings are normalized in the centered pad-to-square frame: across 33,521 sampled boxes the
    observed y-range matches the letterbox band exactly in every aspect-ratio bucket and never reaches
    0.0/1.0 (640x480 -> [0.12, 0.88] vs the predicted [0.125, 0.875]; 640x360 -> [0.22, 0.78] vs
    [0.219, 0.781]). `prismatic/preprocessing/datasets/datasets.py` never touches those strings -- they
    are opaque text -- so the convention is identical for EVERY config regardless of backbone.

    Keying the projection off the pixel policy (as this function did until 2026-09-08) therefore
    mis-scored every non-`letterbox` config. Empirically, on real predictions the centered frame beat
    every alternative for all 8 policy families tested, and the margin GROWS with model quality
    (RefCOCO: +3.4 pts at 5-10% accuracy, +13.2 pts at 40%+; 0 of 429 rows negative). The blind
    (no-vision) baseline -- whose outputs are a pure language prior -- also scores highest under the
    centered frame (9.74% vs 7.32% naive), which is a direct read-out of the training convention.

    Note the corollary: a backbone that pads at the CORNER (ViTDet's `PadToSquare`) or onto a
    NON-SQUARE canvas (vmamba-detection, 800x1333) still emits centered-frame coordinates -- the pixel
    padding is a fixed affine relation the model learns through, not the output convention.
    """
    del transform_info  # intentionally unused; see docstring
    return _normalize_letterbox(bbox_xyxy, image_wh)
