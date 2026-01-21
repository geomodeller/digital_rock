import math
from typing import Optional, Tuple, Literal, List

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedPatchDataset(Dataset):
    """
    Paired 2D patch dataset from 3D volumes (Z, H, W):
      X: filtered image patch (float32)
      Y: binary mask patch (float32, 0/1)

    Supports:
      - grid sampling (deterministic) or random sampling
      - stride control
      - limiting dataset size (max_patches)
    """

    def __init__(
        self,
        vol_x: np.ndarray,
        vol_y: np.ndarray,
        *,
        patch_size: int = 128,
        stride: int = 128,
        sampling: Literal["grid", "random"] = "grid",
        max_patches: Optional[int] = None,
        seed: int = 0,
        # normalization for X
        x_norm: Literal["none", "minmax01", "uint8_01"] = "uint8_01",
        # output layout for PyTorch
        channels_first: bool = True,  # True -> (1,H,W), False -> (H,W,1)
        return_indices: bool = False, # optionally return (z,y,x) patch origin
    ):
        """
        Parameters
        ----------
        vol_x, vol_y : np.ndarray
            3D arrays with shape (Z, H, W). vol_y should be binary (bool or {0,1}).
        patch_size : int
            Patch height/width (square).
        stride : int
            Sliding window stride for "grid" sampling.
        sampling : "grid" or "random"
            - "grid": enumerate all valid patch positions and optionally sub-sample.
            - "random": sample random patch positions on the fly (length = max_patches required).
        max_patches : int or None
            If None:
              - grid: use all patches
              - random: ERROR (needs a defined length)
            If int: limit dataset length to this many patches (sub-sample if grid).
        seed : int
            RNG seed (for shuffling/subsampling/random sampling).
        x_norm : "none" | "minmax01" | "uint8_01"
            - "uint8_01": if vol_x is uint8, divide by 255; otherwise falls back to minmax01
            - "minmax01": min-max normalize within the *dataset volume* to [0,1]
            - "none": no normalization (still float32)
        channels_first : bool
            Output shape: (1,H,W) if True else (H,W,1)
        return_indices : bool
            If True, __getitem__ returns (x, y, (z, y0, x0)).
        """
        if vol_x.ndim != 3 or vol_y.ndim != 3:
            raise ValueError(f"vol_x and vol_y must be 3D. Got {vol_x.shape}, {vol_y.shape}")
        if vol_x.shape != vol_y.shape:
            raise ValueError(f"vol_x and vol_y must match shape. Got {vol_x.shape} vs {vol_y.shape}")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.vol_x = vol_x
        self.vol_y = vol_y
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.sampling = sampling
        self.max_patches = max_patches
        self.channels_first = channels_first
        self.return_indices = return_indices

        self.Z, self.H, self.W = vol_x.shape
        ps = self.patch_size
        if self.H < ps or self.W < ps:
            raise ValueError(f"patch_size={ps} is larger than image size {(self.H, self.W)}")

        self.rng = np.random.default_rng(seed)

        # Precompute normalization stats for X if needed
        self.x_norm = x_norm
        self._x_scale_mode = "none"
        if x_norm == "uint8_01":
            if vol_x.dtype == np.uint8:
                self._x_scale_mode = "uint8_01"
                self._x_min = 0.0
                self._x_max = 255.0
            else:
                # fallback
                self._x_scale_mode = "minmax01"
                self._x_min = float(np.min(vol_x))
                self._x_max = float(np.max(vol_x))
        elif x_norm == "minmax01":
            self._x_scale_mode = "minmax01"
            self._x_min = float(np.min(vol_x))
            self._x_max = float(np.max(vol_x))
        elif x_norm == "none":
            self._x_scale_mode = "none"
        else:
            raise ValueError("x_norm must be one of: 'none', 'minmax01', 'uint8_01'")

        # Build index list for grid sampling
        if sampling == "grid":
            self._indices: List[Tuple[int, int, int]] = self._build_grid_indices()
            # Subsample / shuffle if max_patches is specified
            if self.max_patches is not None:
                if self.max_patches <= 0:
                    raise ValueError("max_patches must be > 0 when provided")
                if self.max_patches < len(self._indices):
                    self.rng.shuffle(self._indices)
                    self._indices = self._indices[: self.max_patches]
        elif sampling == "random":
            if self.max_patches is None:
                raise ValueError("For sampling='random', you must set max_patches (dataset length).")
            if self.max_patches <= 0:
                raise ValueError("max_patches must be > 0 when provided")
            self._indices = []  # generated on the fly
        else:
            raise ValueError("sampling must be 'grid' or 'random'")

    def _build_grid_indices(self) -> List[Tuple[int, int, int]]:
        ps = self.patch_size
        st = self.stride
        ys = list(range(0, self.H - ps + 1, st))
        xs = list(range(0, self.W - ps + 1, st))
        idx = []
        for z in range(self.Z):
            for y0 in ys:
                for x0 in xs:
                    idx.append((z, y0, x0))
        return idx

    def __len__(self) -> int:
        if self.sampling == "grid":
            return len(self._indices)
        return int(self.max_patches)  # random sampling uses this as length

    def _norm_x(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self._x_scale_mode == "none":
            return x
        if self._x_scale_mode == "uint8_01":
            return x / 255.0
        # minmax01
        denom = (self._x_max - self._x_min) + 1e-12
        return (x - self._x_min) / denom

    def __getitem__(self, i: int):
        ps = self.patch_size

        if self.sampling == "grid":
            z, y0, x0 = self._indices[i]
        else:
            # random
            z = int(self.rng.integers(0, self.Z))
            y0 = int(self.rng.integers(0, self.H - ps + 1))
            x0 = int(self.rng.integers(0, self.W - ps + 1))

        x_patch = self.vol_x[z, y0 : y0 + ps, x0 : x0 + ps]
        y_patch = self.vol_y[z, y0 : y0 + ps, x0 : x0 + ps]

        # Normalize / convert
        x_patch = self._norm_x(x_patch)

        # y to float 0/1
        if y_patch.dtype != np.float32:
            y_patch = y_patch.astype(np.float32, copy=False)
        # if it was bool, astype already -> {0,1}; if it was 0/255, map to {0,1}
        if y_patch.max() > 1.0:
            y_patch = (y_patch > 0).astype(np.float32)

        # To torch tensors
        x_t = torch.from_numpy(x_patch)
        y_t = torch.from_numpy(y_patch)

        # Add channel dimension
        if self.channels_first:
            x_t = x_t.unsqueeze(0)  # (1,H,W)
            y_t = y_t.unsqueeze(0)
        else:
            x_t = x_t.unsqueeze(-1)  # (H,W,1)
            y_t = y_t.unsqueeze(-1)

        if self.return_indices:
            return x_t, y_t, (z, y0, x0)
        return x_t, y_t


def make_train_test_paired_datasets(
    filtered_vol: np.ndarray,
    binary_vol: np.ndarray,
    *,
    train_slices: int = 700,
    test_slices: int = 300,  # optional sanity check; can be None if you don't want to enforce
    patch_size: int = 128,
    stride: int = 64,
    sampling: Literal["grid", "random"] = "grid",
    train_max_patches: Optional[int] = None,
    test_max_patches: Optional[int] = None,
    seed: int = 0,
    x_norm: Literal["none", "minmax01", "uint8_01"] = "uint8_01",
    channels_first: bool = True,
    return_indices: bool = False,
) -> Tuple[Dataset, Dataset]:
    """
    Split along Z (first train_slices for train, next for test), then create paired patch datasets.

    Example:
        train_ds, test_ds = make_train_test_paired_datasets(
            filtered, binary, train_slices=700, patch_size=128, stride=64,
            sampling="grid", train_max_patches=50000
        )
    """
    if filtered_vol.shape != binary_vol.shape:
        raise ValueError(f"filtered_vol and binary_vol must match. Got {filtered_vol.shape} vs {binary_vol.shape}")
    if filtered_vol.ndim != 3:
        raise ValueError(f"Expected 3D volumes (Z,H,W). Got {filtered_vol.shape}")

    Z = filtered_vol.shape[0]
    if train_slices <= 0 or train_slices >= Z:
        raise ValueError(f"train_slices must be in [1, Z-1]. Got train_slices={train_slices}, Z={Z}")

    z_train_end = train_slices
    z_test_end = Z if test_slices is None else min(Z, train_slices + test_slices)

    if test_slices is not None and (train_slices + test_slices) > Z:
        raise ValueError(f"train_slices + test_slices exceeds Z. Got {train_slices}+{test_slices} > {Z}")

    train_x = filtered_vol[:z_train_end]
    train_y = binary_vol[:z_train_end]
    test_x = filtered_vol[z_train_end:z_test_end]
    test_y = binary_vol[z_train_end:z_test_end]

    # Use different seeds so train/test subsampling differs but stays reproducible
    train_ds = PairedPatchDataset(
        train_x, train_y,
        patch_size=patch_size,
        stride=stride,
        sampling=sampling,
        max_patches=train_max_patches,
        seed=seed,
        x_norm=x_norm,
        channels_first=channels_first,
        return_indices=return_indices,
    )

    test_ds = PairedPatchDataset(
        test_x, test_y,
        patch_size=patch_size,
        stride=stride,
        sampling=sampling,
        max_patches=test_max_patches,
        seed=seed + 1,
        x_norm=x_norm,
        channels_first=channels_first,
        return_indices=return_indices,
    )

    return train_ds, test_ds
