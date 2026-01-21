
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, feature, morphology, segmentation
from skimage.restoration import denoise_nl_means, estimate_sigma


ArrayLike = np.ndarray

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _require_2d_or_3d(x: ArrayLike) -> None:
    if x.ndim not in (2, 3):
        raise ValueError(f"Only 2D/3D supported, got shape={x.shape} (ndim={x.ndim}).")

def _as_float(img: ArrayLike) -> np.ndarray:
    """Convert to float32 without changing dynamic range."""
    if np.issubdtype(img.dtype, np.floating):
        return img.astype(np.float32, copy=False)
    return img.astype(np.float32)


def _as_float01(img: ArrayLike) -> np.ndarray:
    """Convert to float32 in [0,1] (recommended for NLM and some filters)."""
    _require_2d_or_3d(img)
    if np.issubdtype(img.dtype, np.floating):
        x = img.astype(np.float32, copy=False)
        mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
        if mn >= 0.0 and mx <= 1.0:
            return x
        return (x - mn) / (mx - mn + 1e-12)

    info = np.iinfo(img.dtype)
    return img.astype(np.float32) / float(info.max)


def _restore_dtype_range(x01: np.ndarray, ref: ArrayLike) -> np.ndarray:
    """Map float [0,1] back to the reference dtype range (if ref is integer)."""
    if np.issubdtype(ref.dtype, np.floating):
        return x01.astype(np.float32, copy=False)
    info = np.iinfo(ref.dtype)
    y = np.clip(x01, 0.0, 1.0) * float(info.max)
    return y.round().astype(ref.dtype)


def _structuring_element(ndim: int, radius: int):
    if ndim == 2:
        return morphology.disk(radius)
    if ndim == 3:
        return morphology.ball(radius)
    raise ValueError("Only 2D/3D supported.")



def _cleanup_mask(
    mask: np.ndarray,
    *,
    min_size: int = 0,
    opening_radius: int = 0,
    closing_radius: int = 0,
) -> np.ndarray:
    """Optional morphological cleanup for 2D/3D boolean masks."""
    if mask.dtype != bool:
        mask = mask.astype(bool, copy=False)

    selem_open = _structuring_element(mask.ndim, opening_radius) if opening_radius > 0 else None
    selem_close = _structuring_element(mask.ndim, closing_radius) if closing_radius > 0 else None

    out = mask
    if min_size > 0:
        out = morphology.remove_small_objects(out, min_size=min_size)
    if selem_open is not None:
        out = morphology.binary_opening(out, footprint=selem_open)
    if selem_close is not None:
        out = morphology.binary_closing(out, footprint=selem_close)
    return out
