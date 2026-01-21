from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
ArrayLike = np.ndarray

# # -----------------------------------------------------------------------------
# # Minimal user-style loop 
# # -----------------------------------------------------------------------------
# def apply_image_processing_pipeline(
#     input_image: ArrayLike,
#     pipeline: List[Callable[..., ArrayLike]],
#     function_kwargs: Dict[Callable[..., Dict[str, Any]]],
#     return_intermediate_results: bool = False,
# ) -> Tuple[ArrayLike, List[ArrayLike]] | ArrayLike:
#     """
#     Apply a sequence of image processing functions to an image.

#     Parameters
#     ----------
#     input_image : ArrayLike
#         Input image.
#     pipeline : List[Callable[..., ArrayLike]]
#         Sequence of image processing functions.
#     function_kwargs : Dict[Callable[..., Dict[str, Any]]]
#         Dictionary mapping each function to its associated keyword arguments.
#     return_intermediate_results : bool
#         If True, return a tuple containing the final output image and a list of intermediate output images.

#     Returns
#     -------
#     Tuple[ArrayLike, List[ArrayLike]] | ArrayLike
#         If return_intermediate_results is False, returns the final output image.
#         If return_intermediate_results is True, returns a tuple containing the final output image and a list of intermediate output images.
#     """
#     current_image = input_image
#     intermediate_results = []
#     for func in pipeline:
#         kwargs = function_kwargs.get(func, {})
#         current_image = func(current_image, **kwargs)
#         if return_intermediate_results:
#             intermediate_results.append(current_image)
#     if return_intermediate_results:
#         return current_image, intermediate_results
#     else:
#         return current_image


# -----------------------------------------------------------------------------
# Pipeline runner
# -----------------------------------------------------------------------------
StepFn = Callable[[Any], Any]


@dataclass
class Step:
    name: str
    fn: StepFn


def run_pipeline(
    img: ArrayLike,
    pipeline: List[Step],
    params: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Runs a pipeline of named steps. Each step can have parameters in params[step.name].

    The step functions can return:
      - ndarray (image/mask/labels)
      - or (ndarray, meta) tuple, where meta is stored

    Returns a dict with intermediates and metadata.
    """
    x: Any = img
    out_images: List[Any] = []
    out_steps: List[str] = []
    meta: Dict[str, Any] = {}

    for step in pipeline:
        kw = params.get(step.name, {})
        y = step.fn(x, **kw)

        # capture tuple returns like (mask, threshold)
        if isinstance(y, tuple) and len(y) == 2 and isinstance(y[0], np.ndarray):
            x = y[0]
            meta[step.name] = y[1]
        else:
            x = y

        out_steps.append(step.name)
        out_images.append(x)

    return {
        "steps": out_steps,
        "images": out_images,  # intermediate outputs in order
        "final": x,
        "meta": meta,          # e.g., thresholds
    }


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


def show_pipeline_results_with_hist(
    original,
    result,
    pipeline,
    slice_idx=0,
    bins=80,
    figsize_per_panel=(3.5, 3.5),
):
    """
    Show:
    [ Original | Histogram | Pipeline steps... ]
    """

    images = result["images"]
    n_panels = len(images) + 2  # +1 for original, +1 for histogram

    def get_view(x):
        return x if x.ndim == 2 else x[slice_idx]

    original_view = get_view(original)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        constrained_layout=True,
    )

    # ---- 1) Original image ----
    axes[0].imshow(original_view, cmap="gray")
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # ---- 2) Histogram ----
    axes[1].hist(original_view.ravel(), bins=bins, color="black", alpha=0.8)
    axes[1].set_title("Histogram", fontsize=12)
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Frequency")

    # Optional: show Otsu threshold if available
    if "otsu" in result["meta"]:
        T = result["meta"]["otsu"]
        axes[1].axvline(T, color="red", linestyle="--", linewidth=2, label=f"Otsu T={T:.2f}")
        axes[1].legend()

    # ---- 3) Pipeline steps ----
    for ax, img, step in zip(axes[2:], images, pipeline):
        view = get_view(img)

        if step.name.lower() == "watershed":
            im = ax.imshow(view, cmap="tab20")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.imshow(view, cmap="gray")

        ax.set_title(step.name.capitalize(), fontsize=12)
        ax.axis("off")

    return fig