import numpy as np
import matplotlib.pyplot as plt

from skimage import io, filters, morphology, measure, exposure, util
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy import ndimage as ndi

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "scripts"))
from scripts.visualization import show_slices

def normalize(img):
    """Robust intensity normalization to [0,1]."""
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img


def denoise(img):
    """Non-local means denoising (works for 2D or 3D)."""
    img_u8 = util.img_as_ubyte(img)  # NLM works well in ubyte scale
    sigma_est = np.mean(estimate_sigma(img_u8, channel_axis=None))
    patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=None)

    den = denoise_nl_means(
        img_u8,
        h=0.8 * sigma_est,
        fast_mode=True,
        **patch_kw
    )
    den = util.img_as_float32(den)
    return den


def segment_otsu(img, pore_is_dark=True):
    """Otsu segmentation with cleanup. Returns pore mask = True."""
    thr = filters.threshold_otsu(img)
    if pore_is_dark:
        pore = img < thr
    else:
        pore = img > thr

    # Remove small speckles + smooth edges
    pore = morphology.remove_small_objects(pore, min_size=64)
    pore = morphology.binary_opening(pore, morphology.ball(1) if img.ndim == 3 else morphology.disk(1))
    pore = morphology.binary_closing(pore, morphology.ball(1) if img.ndim == 3 else morphology.disk(1))
    return pore


def porosity(mask):
    return float(mask.mean())


def connectivity_stats(pore_mask, connectivity=1):
    """Connected components: count + largest cluster fraction."""
    lbl = measure.label(pore_mask, connectivity=connectivity)
    props = measure.regionprops(lbl)
    if len(props) == 0:
        return {"n_components": 0, "largest_fraction": 0.0}

    sizes = np.array([p.area for p in props], dtype=np.float64)
    largest = sizes.max()
    total = sizes.sum()
    return {
        "n_components": int(len(sizes)),
        "largest_fraction": float(largest / (total + 1e-12))
    }


# ------------------------------
# DEMO ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    # OPTION A) Load a 2D image (png/tif)
    # img = io.imread("your_slice.tif")

    # OPTION B) Load a 3D stack from a folder of images (z-slices)
    # from pathlib import Path
    # folder = Path("your_stack_folder")
    # files = sorted(folder.glob("*.tif"))
    # vol = np.stack([io.imread(f) for f in files], axis=0)
    # img = vol

    # ---- For demonstration without your data (synthetic placeholder) ----
    # Replace this block with your actual load code above.
    rng = np.random.default_rng(0)
    img = rng.normal(0.5, 0.12, size=(256, 256)).astype(np.float32)
    img = np.clip(img, 0, 1)

    # 1) Normalize
    img_n = normalize(img)
    show_slices(img_n, "Raw (normalized)")

    # 2) Denoise
    img_d = denoise(img_n)
    show_slices(img_d, "Denoised")

    # 3) Segment pores (assume pores are darker in CT)
    pore = segment_otsu(img_d, pore_is_dark=True)

    # 4) Metrics
    phi = porosity(pore)
    stats = connectivity_stats(pore, connectivity=1)

    print(f"Porosity = {phi:.4f}")
    print(f"Connected components = {stats['n_components']}")
    print(f"Largest pore cluster fraction = {stats['largest_fraction']:.4f}")

    # 5) Visualize segmentation overlay (2D case)
    if img.ndim == 2:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_d, cmap="gray")
        plt.title("Denoised")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_d, cmap="gray")
        plt.imshow(pore, alpha=0.35)  # overlay mask
        plt.title("Pore mask overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
