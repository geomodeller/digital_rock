"""
Digital Rock Denoising Utilities (2D / 3D)
Includes:
  1) Gaussian smoothing
  2) Median filtering
  3) Bilateral filtering (edge-preserving)
  4) Non-Local Means (patch-based)
  5) Fourier-based filtering (low-pass / high-pass / band-pass)
  6) Wavelet-based denoising (shrinkage)

All functions take a NumPy array that can be either 2D (H,W) or 3D (Z,Y,X).
Notes:
- For 3D bilateral, scikit-image's denoise_bilateral is primarily 2D-oriented; this code applies it slice-wise.
- Fourier filtering is global. For pore-scale work, low-pass may blur fine pores—use carefully.
- Wavelet denoising here uses skimage's denoise_wavelet (supports 2D/3D).

Dependencies:
  numpy
  scipy
  scikit-image

Install:
  pip install numpy scipy scikit-image
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage import util
from skimage.filters import gaussian, median
from skimage.morphology import disk, ball
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    estimate_sigma,
    denoise_wavelet,
    denoise_tv_chambolle
)


ArrayLike = np.ndarray


def _require_2d_or_3d(img: ArrayLike) -> None:
    if img.ndim not in (2, 3):
        raise ValueError(f"Only 2D/3D supported, got shape={img.shape} (ndim={img.ndim}).")


def _as_float01(img: ArrayLike) -> np.ndarray:
    """
    Convert to float32 in [0,1] using dtype-aware scaling.
    This is recommended for bilateral/NLM/wavelet APIs.
    """
    _require_2d_or_3d(img)
    if np.issubdtype(img.dtype, np.floating):
        x = img.astype(np.float32, copy=False)
        # If already looks like [0,1], keep; else min-max to [0,1]
        if np.nanmin(x) >= 0.0 and np.nanmax(x) <= 1.0:
            return x
        mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
        return (x - mn) / (mx - mn + 1e-12)

    # Integer types
    info = np.iinfo(img.dtype)
    return img.astype(np.float32) / float(info.max)


def _restore_dtype_range(x01: np.ndarray, ref: ArrayLike) -> np.ndarray:
    """
    Map float [0,1] result back to reference dtype range if ref is integer,
    otherwise keep float32.
    """
    if np.issubdtype(ref.dtype, np.floating):
        return x01.astype(np.float32, copy=False)

    info = np.iinfo(ref.dtype)
    y = np.clip(x01, 0.0, 1.0) * float(info.max)
    return y.round().astype(ref.dtype)


# -----------------------------------------------------------------------------
# 1) Gaussian smoothing
# -----------------------------------------------------------------------------
def denoise_gaussian(
    img: ArrayLike,
    *,
    sigma: Union[float, Tuple[float, ...]] = 1.0,
    preserve_range: bool = True,
) -> np.ndarray:
    """
    Gaussian smoothing for 2D/3D.

    sigma:
      - float -> isotropic smoothing
      - tuple -> anisotropic (e.g., (0.5, 1.0, 1.0) for (Z,Y,X) in 3D)

    preserve_range=True keeps the numeric range of the input.
    """
    _require_2d_or_3d(img)
    out = gaussian(img, sigma=sigma, preserve_range=preserve_range, channel_axis=None)
    return out.astype(img.dtype, copy=False) if preserve_range and out.dtype != img.dtype else out


# -----------------------------------------------------------------------------
# 2) Median filtering
# -----------------------------------------------------------------------------
def denoise_median(
    img: ArrayLike,
    *,
    radius: int = 1,
    preserve_range: bool = True,
) -> np.ndarray:
    """
    Median filtering for 2D/3D.

    radius:
      - 2D uses a disk(radius)
      - 3D uses a ball(radius)

    Good for salt-and-pepper / speckle-like outliers; can remove tiny pores if radius is large.
    """
    _require_2d_or_3d(img)
    footprint = disk(radius) if img.ndim == 2 else ball(radius)
    out = median(img, footprint=footprint, mode="nearest")
    return out.astype(img.dtype, copy=False) if preserve_range and out.dtype != img.dtype else out


# -----------------------------------------------------------------------------
# 3) Bilateral filtering (edge-preserving)
# -----------------------------------------------------------------------------
def denoise_bilateral_filter(
    img: ArrayLike,
    *,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
    bins: int = 1000,
    mode: str = "constant",
    cval: float = 0.0,
    slice_wise_for_3d: bool = True,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Bilateral denoising (edge-preserving).

    Works best on float images scaled to [0,1], so we internally convert to [0,1].
    For 3D: scikit-image bilateral is primarily 2D; default applies slice-wise along Z.

    Parameters
    ----------
    sigma_color : float
        Intensity similarity. Smaller -> stronger edge preservation.
    sigma_spatial : float
        Spatial smoothing strength in pixels.
    slice_wise_for_3d : bool
        If True and img is 3D, apply bilateral to each z-slice.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    _require_2d_or_3d(img)
    x01 = _as_float01(img)

    if img.ndim == 2:
        y01 = denoise_bilateral(
            x01,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            bins=bins,
            mode=mode,
            cval=cval,
            channel_axis=None,
        )
    else:
        if not slice_wise_for_3d:
            # If you want true 3D bilateral, you'd need a different library/implementation.
            raise NotImplementedError(
                "True 3D bilateral is not provided here. Use slice_wise_for_3d=True."
            )
        y01 = np.empty_like(x01, dtype=np.float32)
        for z in range(x01.shape[0]):
            y01[z] = denoise_bilateral(
                x01[z],
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                bins=bins,
                mode=mode,
                cval=cval,
                channel_axis=None,
            )

    return _restore_dtype_range(y01, img) if preserve_dtype else y01


# -----------------------------------------------------------------------------
# 4) Non-Local Means (patch-based)
# -----------------------------------------------------------------------------
def denoise_non_local_means(
    img: ArrayLike,
    *,
    patch_size: int = 5,
    patch_distance: int = 6,
    h: Optional[float] = None,
    fast_mode: bool = True,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Non-Local Means denoising for 2D/3D.

    Parameters
    ----------
    patch_size : int
        Size of patches used for similarity.
    patch_distance : int
        Max search distance for similar patches.
    h : float or None
        Filter strength. If None, estimated from noise sigma.
        Larger h -> stronger smoothing.
    fast_mode : bool
        True is faster, slight quality tradeoff.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    _require_2d_or_3d(img)
    x01 = _as_float01(img)

    # Estimate sigma on [0,1] scale
    sigma_est = float(np.mean(estimate_sigma(x01, channel_axis=None)))
    if h is None:
        h = 0.8 * sigma_est  # common rule-of-thumb

    y01 = denoise_nl_means(
        x01,
        h=h,
        fast_mode=fast_mode,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
    ).astype(np.float32, copy=False)

    return _restore_dtype_range(y01, img) if preserve_dtype else y01


# -----------------------------------------------------------------------------
# gaussian-based biased correction filtering
# -----------------------------------------------------------------------------
def denoise_bias_field_correct(
    img: ArrayLike,
    *,
    sigma=30, 
    eps=1e-6, 
    preserve_dtype=True
    ):
    img_f = img.astype(np.float32, copy=False)

    bg = gaussian_filter(img_f, sigma=sigma)
    bg = np.maximum(bg, eps)
    
    out = img_f / bg * np.mean(bg)

    if preserve_dtype:
        # clip back to original dtype range if integer
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            out = np.clip(out, info.min, info.max).astype(img.dtype)
        else:
            out = out.astype(img.dtype)
    return out

# -----------------------------------------------------------------------------
# 5) Fourier-based filtering (global frequency domain)
# -----------------------------------------------------------------------------
def denoise_fourier(
    img: ArrayLike,
    *,
    kind: Literal["lowpass", "highpass", "bandpass"] = "lowpass",
    cutoff_low: float = 0.15,
    cutoff_high: float = 0.35,
    transition: float = 0.02,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Fourier-domain filtering for 2D/3D using a smooth radial filter (raised-cosine transition).

    Cutoffs are fractions of the Nyquist radius (0..0.5), expressed in normalized frequency units.
    - lowpass: keep <= cutoff_high
    - highpass: keep >= cutoff_low
    - bandpass: keep between [cutoff_low, cutoff_high]

    Notes:
    - This is GLOBAL filtering: it can blur pore edges if lowpass is too aggressive.
    - For ring/stripe artifacts, specialized CT methods (polar filtering / sinogram correction) often work better.

    Parameters
    ----------
    cutoff_low, cutoff_high : float
        Normalized cutoffs (0..0.5). Typical: lowpass cutoff_high ~ 0.10-0.25
    transition : float
        Width of smooth transition band.
    """
    _require_2d_or_3d(img)
    x = img.astype(np.float32, copy=False)

    def _make_radial_filter(shape):
        # Create normalized frequency grid in [-0.5, 0.5)
        coords = [np.fft.fftfreq(n) for n in shape]
        grids = np.meshgrid(*coords, indexing="ij")
        r = np.sqrt(sum(g**2 for g in grids))  # radial frequency
        # r is in [0, ~0.5*sqrt(ndim)] but max relevant is 0.5 along axes.

        # Helper: smooth step from 0->1 around edge with transition width
        def smooth_step(xv, edge, w):
            # 0 below edge-w/2, 1 above edge+w/2, cosine blend in between
            lo = edge - 0.5 * w
            hi = edge + 0.5 * w
            out = np.zeros_like(xv, dtype=np.float32)
            out[xv >= hi] = 1.0
            mid = (xv > lo) & (xv < hi)
            out[mid] = 0.5 - 0.5 * np.cos(np.pi * (xv[mid] - lo) / (hi - lo + 1e-12))
            return out

        # Build masks
        kind_l = kind.lower()
        if kind_l == "lowpass":
            # pass low freq -> 1 below cutoff_high, 0 above
            s = smooth_step(r, cutoff_high, transition)
            H = 1.0 - s
        elif kind_l == "highpass":
            s = smooth_step(r, cutoff_low, transition)
            H = s
        elif kind_l == "bandpass":
            s1 = smooth_step(r, cutoff_low, transition)   # 0->1 at low cutoff
            s2 = smooth_step(r, cutoff_high, transition)  # 0->1 at high cutoff
            H = s1 * (1.0 - s2)
        else:
            raise ValueError("kind must be 'lowpass', 'highpass', or 'bandpass'")
        return H.astype(np.float32)

    F = np.fft.fftn(x)
    H = _make_radial_filter(x.shape)
    y = np.fft.ifftn(F * H).real.astype(np.float32)

    if preserve_dtype:
        # Preserve input range approximately
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            y = np.clip(y, 0, float(info.max))
            return y.round().astype(img.dtype)
        return y.astype(np.float32, copy=False)
    return y


# -----------------------------------------------------------------------------
# 6) Wavelet-based denoising (shrinkage)
# -----------------------------------------------------------------------------
def denoise_wavelet_filter(
    img: ArrayLike,
    *,
    method: Literal["BayesShrink", "VisuShrink"] = "BayesShrink",
    mode: Literal["soft", "hard"] = "soft",
    wavelet: str = "db2",
    rescale_sigma: bool = True,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Wavelet denoising via coefficient shrinkage, works for 2D/3D.

    method:
      - "BayesShrink": adaptive thresholding, usually good default
      - "VisuShrink": universal threshold, often stronger smoothing

    mode:
      - "soft": smoother results, common choice
      - "hard": preserves edges more but can introduce artifacts

    wavelet:
      - Common choices: "db2", "db4", "sym4", "coif1"

    Internally works best on float [0,1], so we rescale.
    """
    _require_2d_or_3d(img)
    x01 = _as_float01(img)

    y01 = denoise_wavelet(
        x01,
        method=method,
        mode=mode,
        wavelet=wavelet,
        rescale_sigma=rescale_sigma,
        channel_axis=None,
    ).astype(np.float32, copy=False)

    return _restore_dtype_range(y01, img) if preserve_dtype else y01


# -----------------------------------------------------------------------------
# Convenience: apply method by name
# -----------------------------------------------------------------------------
DenoiseMethod = Literal["gaussian", "median", "bilateral", "nlm", "fourier", "wavelet"]


def denoise(
    img: ArrayLike,
    method: DenoiseMethod,
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point.

    Examples:
        img2 = denoise(img, "gaussian", sigma=1.2)
        img2 = denoise(img, "nlm", patch_size=5, patch_distance=7)
        img2 = denoise(img, "fourier", kind="lowpass", cutoff_high=0.18)
        img2 = denoise(img, "wavelet", method="BayesShrink", wavelet="db2")

    """
    m = method.lower().strip()
    if m == "gaussian":
        return denoise_gaussian(img, **kwargs)
    if m == "median":
        return denoise_median(img, **kwargs)
    if m == "bilateral":
        return denoise_bilateral_filter(img, **kwargs)
    if m in ("nlm", "nonlocal", "non-local"):
        return denoise_non_local_means(img, **kwargs)
    if m == "fourier":
        return denoise_fourier(img, **kwargs)
    if m == "wavelet":
        return denoise_wavelet_filter(img, **kwargs)
    raise ValueError(f"Unknown method: {method}")


# -----------------------------------------------------------------------------
# Quick demo (safe defaults)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic example (replace with your 2D slice or 3D volume ndarray)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)

    g = denoise_gaussian(img, sigma=1.0)
    m = denoise_median(img, radius=1)
    b = denoise_bilateral_filter(img, sigma_color=0.08, sigma_spatial=3)
    n = denoise_non_local_means(img)
    f = denoise_fourier(img, kind="lowpass", cutoff_high=0.18)
    w = denoise_wavelet_filter(img, method="BayesShrink", wavelet="db2")

    print("Done:", g.shape, m.shape, b.shape, n.shape, f.shape, w.shape)


def denoise_tv(
    img: np.ndarray,
    *,
    weight: float = 0.1,
    max_num_iter: int = 200,
    eps: float = 2e-4,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Total Variation (TV) denoising for 2D or 3D images.

    Parameters
    ----------
    img : np.ndarray
        Input image (2D or 3D), integer or float.
    weight : float
        Denoising strength.
        - Larger -> stronger smoothing
        - Typical for CT: 0.05 ~ 0.2
    max_num_iter : int
        Maximum number of iterations.
    eps : float
        Convergence tolerance.
    preserve_dtype : bool
        If True, output is returned in the same dtype/range as input.

    Returns
    -------
    np.ndarray
        TV-denoised image.
    """

    if img.ndim not in (2, 3):
        raise ValueError(f"Only 2D or 3D supported, got shape={img.shape}")

    # ---- Convert to float [0,1] ----
    if np.issubdtype(img.dtype, np.floating):
        x = img.astype(np.float32, copy=False)
        minv, maxv = float(x.min()), float(x.max())
        if minv < 0.0 or maxv > 1.0:
            x = (x - minv) / (maxv - minv + 1e-12)
    else:
        info = np.iinfo(img.dtype)
        x = img.astype(np.float32) / float(info.max)

    # ---- TV denoising ----
    y = denoise_tv_chambolle(
        x,
        weight=weight,
        max_num_iter=max_num_iter,
        eps=eps,
        channel_axis=None,
    ).astype(np.float32, copy=False)

    # ---- Restore dtype & range ----
    if not preserve_dtype:
        return y

    if np.issubdtype(img.dtype, np.floating):
        return y.astype(np.float32, copy=False)

    y = np.clip(y, 0.0, 1.0)
    return (y * info.max).round().astype(img.dtype)