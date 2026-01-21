import numpy as np
import matplotlib.pyplot as plt

def show_slices(arr, title="", n=6):
    """Quick viewer for 2D image or 3D volume."""
    plt.figure(figsize=(14, 3))
    if arr.ndim == 2:
        plt.subplot(1, 2, 1)
        plt.imshow(arr, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.hist(arr.ravel(), bins=64)
        plt.title("Histogram")
        plt.tight_layout()
        plt.show()
        return

    # 3D volume
    z = arr.shape[0]
    idxs = np.linspace(0, z - 1, n, dtype=int)
    for i, k in enumerate(idxs):
        plt.subplot(1, n, i + 1)
        plt.imshow(arr[k], cmap="gray")
        plt.title(f"z={k}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def show_pipeline_results(
    original,
    result,
    pipeline,
    slice_idx=0,
    figsize_per_panel=(3.5, 3.5),
):
    """
    Visualize original image + pipeline results side by side.

    Parameters
    ----------
    original : np.ndarray
        Input image (2D or 3D)
    result : dict
        Output from run_pipeline(...)
    pipeline : list
        PIPELINE list (with .name attributes)
    slice_idx : int
        Slice index for 3D volumes
    figsize_per_panel : tuple
        Size per subplot (width, height)
    """

    images = result["images"]
    n_panels = len(images) + 1

    # Pick slice if 3D
    def get_view(x):
        return x if x.ndim == 2 else x[slice_idx]

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        constrained_layout=True,
    )

    # ---- Original ----
    axes[0].imshow(get_view(original), cmap="gray")
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # ---- Pipeline steps ----
    for i, (ax, img, step) in enumerate(zip(axes[1:], images, pipeline)):
        view = get_view(img)

        if step.name.lower() == "watershed":
            im = ax.imshow(view, cmap="tab20")
        else:
            im = ax.imshow(view, cmap="gray")

        ax.set_title(step.name.capitalize(), fontsize=12)
        ax.axis("off")

        # Optional colorbar for watershed
        if step.name.lower() == "watershed":
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    return fig

def plot_pipeline_with_hists(
    original,
    result,
    pipeline,
    *,
    slice_idx=0,
    bins_intensity=80,
    bins_poresize=50,
    pore_size_mode="voxel",   # "voxel" or "equiv_diameter"
    log_poresize=True,
    hist_use_full_volume=True,  # if False, histogram uses displayed slice only
    figsize_per_col=(3.4, 3.2),
):
    """
    2-row figure:
      Row 1: images (original + each pipeline outcome)
      Row 2: histogram for each outcome
          - For grayscale: intensity histogram
          - For boolean mask: 0/1 histogram + pore fraction annotation
          - For watershed labels: pore size distribution (region sizes)

    Works for 2D (H,W) or 3D (Z,Y,X).

    Expected:
      result["images"] is list of intermediate outputs in pipeline order
      result["meta"] may include {"otsu": threshold} (optional)
    """

    images = result["images"]
    step_names = [s.name for s in pipeline]
    cols = 1 + len(images)  # include original

    def _view(x):
        return x if x.ndim == 2 else x[slice_idx]

    def _hist_array(x):
        # Either full volume or displayed slice for histogram
        return x.ravel() if (hist_use_full_volume or x.ndim == 2) else _view(x).ravel()

    def _pore_sizes_from_labels(labels: np.ndarray):
        """Compute pore sizes from labels: sizes in voxels/pixels, excluding label 0."""
        if labels.size == 0:
            return np.array([], dtype=np.float64)

        flat = labels.ravel()
        flat = flat[flat > 0]
        if flat.size == 0:
            return np.array([], dtype=np.float64)

        counts = np.bincount(flat.astype(np.int64))
        # bincount includes 0..maxLabel; we removed zeros already so counts[0] might be 0.
        sizes = counts[counts > 0].astype(np.float64)  # sizes per label, in voxels/pixels

        if pore_size_mode == "voxel":
            return sizes

        # Equivalent diameter from area/volume (pixel/voxel units)
        # 2D: d_eq = 2*sqrt(A/pi)
        # 3D: d_eq = 2*(3V/4pi)^(1/3)
        if labels.ndim == 2:
            return 2.0 * np.sqrt(sizes / np.pi)
        elif labels.ndim == 3:
            return 2.0 * ((3.0 * sizes) / (4.0 * np.pi)) ** (1.0 / 3.0)
        else:
            raise ValueError("labels must be 2D or 3D")

    fig, axes = plt.subplots(
        2,
        cols,
        figsize=(figsize_per_col[0] * cols, figsize_per_col[1] * 2),
        constrained_layout=True,
    )

    # ---- Column 0: Original ----
    orig_view = _view(original)
    axes[0, 0].imshow(orig_view, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")

    # Histogram: original intensity
    orig_hist = _hist_array(original).astype(np.float32)
    axes[1, 0].hist(orig_hist, bins=bins_intensity, color="black", alpha=0.85)
    axes[1, 0].set_title("Intensity hist", fontsize=10)
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Count")

    # ---- Remaining columns: pipeline outputs + their histograms ----
    for j, (out, name) in enumerate(zip(images, step_names), start=1):
        name_l = name.lower()

        # --- Row 1: image ---
        out_view = _view(out)

        if name_l == "watershed":
            im = axes[0, j].imshow(out_view, cmap="tab20")
            axes[0, j].set_title("Watershed", fontsize=11)
            plt.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.02)
        else:
            axes[0, j].imshow(out_view, cmap="gray")
            axes[0, j].set_title(name.capitalize(), fontsize=11)

        axes[0, j].axis("off")

        # --- Row 2: histogram ---
        axh = axes[1, j]

        if name_l == "watershed":
            labels = np.asarray(out)
            sizes = _pore_sizes_from_labels(labels)

            if sizes.size == 0:
                axh.text(0.5, 0.5, "No labeled pores", ha="center", va="center")
                axh.set_axis_off()
            else:
                # Optionally plot on log-scale (common for pore size distributions)
                if log_poresize:
                    # avoid log(0)
                    sizes_plot = sizes[sizes > 0]
                    axh.hist(np.log10(sizes_plot), bins=bins_poresize, color="black", alpha=0.85)
                    xlabel = "log10(size)"
                else:
                    axh.hist(sizes, bins=bins_poresize, color="black", alpha=0.85)
                    xlabel = "size"

                unit = "voxels" if pore_size_mode == "voxel" else "equiv. diameter (px/vox)"
                axh.set_title(f"Pore size dist ({unit})", fontsize=10)
                axh.set_xlabel(xlabel)
                axh.set_ylabel("Count")

                axh.text(
                    0.02,
                    0.95,
                    f"N={sizes.size}",
                    transform=axh.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                )

        else:
            arr = np.asarray(out)
            if arr.dtype == bool:
                # 0/1 histogram + pore fraction
                vals = _hist_array(arr).astype(np.uint8)
                counts = np.bincount(vals, minlength=2)
                axh.bar([0, 1], counts, width=0.6, color="black", alpha=0.85)
                axh.set_xticks([0, 1])
                axh.set_xticklabels(["solid(0)", "pore(1)"])
                axh.set_title("Mask counts", fontsize=10)
                axh.set_ylabel("Count")
                phi = float(arr.mean())
                axh.text(0.02, 0.95, f"ϕ={phi:.3f}", transform=axh.transAxes, ha="left", va="top", fontsize=9)
            else:
                vals = _hist_array(arr).astype(np.float32)
                axh.hist(vals, bins=bins_intensity, color="black", alpha=0.85)
                axh.set_title("Intensity hist", fontsize=10)
                axh.set_xlabel("Intensity")
                axh.set_ylabel("Count")

                # If Otsu threshold was stored in result["meta"], draw it on the Otsu step’s hist
                if name_l == "otsu" and "otsu" in result.get("meta", {}):
                    T = float(result["meta"]["otsu"])
                    axh.axvline(T, linestyle="--", linewidth=2, color="red")
                    axh.text(0.02, 0.95, f"T={T:.2f}", transform=axh.transAxes, ha="left", va="top", fontsize=9)

    return fig

import matplotlib.pyplot as plt
import numpy as np


def show_raw_filtered_binary(
    raw,
    filtered,
    binary,
    *,
    slice_idx=0,
    figsize=(12, 4),
    cmap_img="gray",
):
    """
    Visualize raw → filtered → binary side by side.
    Works for 2D or 3D arrays.

    Parameters
    ----------
    raw, filtered, binary : np.ndarray
        2D (H,W) or 3D (Z,H,W)
    slice_idx : int
        Slice index if inputs are 3D
    """

    def view(x):
        return x if x.ndim == 2 else x[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # ---- Raw ----
    im0 = axes[0].imshow(view(raw), cmap=cmap_img)
    axes[0].set_title("Raw CT", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    # ---- Filtered ----
    im1 = axes[1].imshow(view(filtered), cmap=cmap_img)
    axes[1].set_title("Filtered", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    # ---- Binary ----
    axes[2].imshow(view(binary), cmap="gray")
    axes[2].set_title("Binary (Pore Mask)", fontsize=12)
    axes[2].axis("off")

    return fig

def show_signed_difference(image_1, image_2, vlim=None):
    """
    Signed difference: image_1 - image_2
    """
    diff = image_1.astype(np.float32) - image_2.astype(np.float32)

    if vlim is None:
        vmax = np.percentile(np.abs(diff), 99)
        vlim = (-vmax, vmax)

    plt.figure(figsize=(5, 5))
    im = plt.imshow(diff, cmap="seismic", vmin=vlim[0], vmax=vlim[1])
    plt.title("Signed Difference (image_1 − image_2)")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.02)
    plt.show()


def show_signed_difference_3col(image_1, image_2, vlim=None, cmap_img="gray"):
    """
    1×3 visualization:
    [ image_1 | signed difference (image_1 - image_2) | image_2 ]

    Parameters
    ----------
    image_1, image_2 : np.ndarray
        2D images with same shape
    vlim : tuple or None
        (vmin, vmax) for difference image
    """

    diff = image_1.astype(np.float32) - image_2.astype(np.float32)

    if vlim is None:
        vmax = np.percentile(np.abs(diff), 99)
        vlim = (-vmax, vmax)

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    # ---- Left: image_1 ----
    im0 = axes[0].imshow(image_1, cmap=cmap_img)
    axes[0].set_title("Image 1", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    # ---- Middle: signed difference ----
    im1 = axes[1].imshow(diff, cmap="seismic", vmin=vlim[0], vmax=vlim[1])
    axes[1].set_title("Signed Difference (Image 1 − Image 2)", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    # ---- Right: image_2 ----
    im2 = axes[2].imshow(image_2, cmap=cmap_img)
    axes[2].set_title("Image 2", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

    plt.show()

# Example usage:
# fig = plot_pipeline_with_hists(
#     original=img,
#     result=result,
#     pipeline=PIPELINE,
#     slice_idx=0,
#     hist_use_full_volume=True,
#     pore_size_mode="voxel",    # or "equiv_diameter"
#     log_poresize=True,
# )
# plt.show()
