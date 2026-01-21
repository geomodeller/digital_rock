"""
Digital Rock Segmentation Utilities
- Iterative selection thresholding
- Otsu thresholding
- Watershed segmentation (marker-based) using distance transform

All functions accept a NumPy array that can be either 2D (H,W) or 3D (Z,Y,X).
Return types:
- iterative_selection_threshold / otsu_threshold: float threshold
- segment_iterative_selection_mask / segment_otsu: boolean pore mask
- segment_watershed_labels: integer-labeled segmentation (0=background, 1..N regions)

Dependencies:
    numpy
    scipy
    scikit-image

Install:
    pip install numpy scipy scikit-image
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, util

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "scripts"))
from helper import _structuring_element, _as_float, _cleanup_mask

ArrayLike = np.ndarray

def iterative_selection_threshold(
    img: ArrayLike,
    *,
    pore_is_dark: bool = True,
    max_iter: int = 100,
    tol: float = 0.5,
    init: str = "midrange",
) -> float:
    """
    Iterative selection thresholding (a.k.a. iterative mean / Ridler-Calvard style).

    Parameters
    ----------
    img : np.ndarray
        2D or 3D grayscale image.
    pore_is_dark : bool
        If True, pore phase assumed darker (lower intensity).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance in intensity units (works for uint8, uint16, float).
    init : str
        "midrange" or "mean". Sets initial threshold.

    Returns
    -------
    float
        Converged threshold value.
    """
    x = _as_float(img)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if init == "midrange":
        T = 0.5 * (vmin + vmax)
    elif init == "mean":
        T = float(np.mean(x))
    else:
        raise ValueError("init must be 'midrange' or 'mean'")

    for _ in range(max_iter):
        if pore_is_dark:
            g1 = x[x < T]
            g2 = x[x >= T]
        else:
            g1 = x[x > T]
            g2 = x[x <= T]

        # If one group becomes empty, stop (cannot refine)
        if g1.size == 0 or g2.size == 0:
            break

        mu1 = float(np.mean(g1))
        mu2 = float(np.mean(g2))
        T_new = 0.5 * (mu1 + mu2)

        if abs(T_new - T) < tol:
            T = T_new
            break
        T = T_new

    return float(T)


def segment_iterative_selection_mask(
    img: ArrayLike,
    *,
    pore_is_dark: bool = True,
    max_iter: int = 100,
    tol: float = 0.5,
    init: str = "midrange",
    min_size: int = 0,
    opening_radius: int = 0,
    closing_radius: int = 0,
) -> np.ndarray:
    """
    Binary mask from iterative selection thresholding.

    Returns
    -------
    np.ndarray (bool)
        Pore mask (True=pore).
    """
    T = iterative_selection_threshold(
        img, pore_is_dark=pore_is_dark, max_iter=max_iter, tol=tol, init=init
    )
    x = _as_float(img)
    pore = (x < T) if pore_is_dark else (x > T)

    pore = _cleanup_mask(
        pore,
        min_size=min_size,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
    )
    return pore


def otsu_threshold(img: ArrayLike) -> float:
    """
    Otsu threshold (global) for 2D or 3D image.

    Returns
    -------
    float
        Otsu threshold value.
    """
    x = _as_float(img)
    # skimage expects finite values
    if not np.isfinite(x).all():
        raise ValueError("Image contains NaN/Inf; please clean before thresholding.")
    return float(filters.threshold_otsu(x))


def segment_otsu(
    img: ArrayLike,
    *,
    pore_is_dark: bool = True,
    min_size: int = 0,
    opening_radius: int = 0,
    closing_radius: int = 0,
) -> np.ndarray:
    """
    Binary pore mask using Otsu threshold.

    Returns
    -------
    np.ndarray (bool)
        Pore mask (True=pore).
    """
    T = otsu_threshold(img)
    x = _as_float(img)
    pore = (x < T) if pore_is_dark else (x > T)

    pore = _cleanup_mask(
        pore,
        min_size=min_size,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
    )
    return pore




def segment_watershed_labels(
    img_or_mask: ArrayLike,
    *,
    mask_is_pore: bool = False,
    pore_is_dark: bool = True,
    threshold_method: str = "otsu",
    min_size: int = 0,
    opening_radius: int = 0,
    closing_radius: int = 0,
    # Watershed specifics
    peak_footprint_radius: int = 1,
    min_peak_distance: int = 1,
    compactness: float = 0.0,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Marker-based watershed to split connected pore regions.

    Typical workflow:
        labels = segment_watershed_labels(img, threshold_method="otsu", pore_is_dark=True)

    Parameters
    ----------
    img_or_mask : np.ndarray
        2D/3D grayscale image OR a boolean pore mask.
    mask_is_pore : bool
        If True, img_or_mask is interpreted as pore mask (bool or 0/1).
    pore_is_dark : bool
        Only used when mask_is_pore=False (thresholding step).
    threshold_method : str
        "otsu" or "iterative" (only used when mask_is_pore=False).
    min_size, opening_radius, closing_radius : int
        Cleanup options applied to the pore mask.
    peak_footprint_radius : int
        Radius for footprint used to find local maxima in distance transform.
        Larger -> fewer markers -> less over-segmentation.
    min_peak_distance : int
        Minimum distance between peaks (in pixels/voxels).
    compactness : float
        Watershed compactness parameter (0.0 = classic watershed).
    connectivity : int
        Connectedness for watershed (1 = 4-neigh in 2D / 6-neigh in 3D, 2 = 8/18/26 etc.).

    Returns
    -------
    np.ndarray (int)
        Watershed labels with 0=background (non-pore), 1..N pore regions.
    """
    if mask_is_pore:
        pore = np.asarray(img_or_mask).astype(bool)
        pore = _cleanup_mask(
            pore, min_size=min_size, opening_radius=opening_radius, closing_radius=closing_radius
        )
    else:
        method = threshold_method.lower().strip()
        if method == "otsu":
            pore = segment_otsu(
                img_or_mask,
                pore_is_dark=pore_is_dark,
                min_size=min_size,
                opening_radius=opening_radius,
                closing_radius=closing_radius,
            )
        elif method in {"iter", "iterative", "iterative_selection"}:
            pore = segment_iterative_selection_mask(
                img_or_mask,
                pore_is_dark=pore_is_dark,
                min_size=min_size,
                opening_radius=opening_radius,
                closing_radius=closing_radius,
            )
        else:
            raise ValueError("threshold_method must be 'otsu' or 'iterative'")

    if pore.ndim not in (2, 3):
        raise ValueError(f"Only 2D/3D supported, got ndim={pore.ndim}")

    # Distance transform: inside pores, distance to nearest solid boundary
    dist = ndi.distance_transform_edt(pore)

    # Find markers (seeds) from local maxima of distance map
    fp = _structuring_element(pore.ndim, peak_footprint_radius)
    coords = feature.peak_local_max(
        dist,
        footprint=fp,
        labels=pore.astype(np.uint8),
        min_distance=min_peak_distance,
        exclude_border=False,
    )

    markers = np.zeros(pore.shape, dtype=np.int32)
    for i, idx in enumerate(coords, start=1):
        markers[tuple(idx)] = i

    # If no peaks found, fallback: single marker at global max (if any pore exists)
    if markers.max() == 0 and pore.any():
        max_idx = np.unravel_index(np.argmax(dist), dist.shape)
        markers[max_idx] = 1

    # Watershed over negative distance (so maxima become basins)
    labels = segmentation.watershed(
        -dist,
        markers=markers,
        mask=pore,
        compactness=compactness,
        connectivity=connectivity,
    ).astype(np.int32)

    return labels


# ------------------------
# Optional quick demo usage
# ------------------------
if __name__ == "__main__":
    # Synthetic example (2D). Replace with your own ndarray.
    rng = np.random.default_rng(0)
    img = rng.normal(120, 25, size=(256, 256)).astype(np.float32)
    img = np.clip(img, 0, 255)

    T_i = iterative_selection_threshold(img, tol=0.1)
    T_o = otsu_threshold(img)

    pore_i = segment_iterative_selection_mask(img, min_size=64, opening_radius=1, closing_radius=1)
    pore_o = segment_otsu(img, min_size=64, opening_radius=1, closing_radius=1)

    labels = segment_watershed_labels(img, threshold_method="otsu", min_size=64, opening_radius=1)

    print("Iterative threshold:", T_i)
    print("Otsu threshold:", T_o)
    print("Watershed regions (nonzero labels):", labels.max())
