"""
Workflow: Train 2D ResUNet++ on paired patch datasets (train_ds, test_ds)

Assumptions
-----------
- train_ds and test_ds are torch.utils.data.Dataset returning:
    x: (1, 128, 128) float32 in [0,1]  (filtered CT patch)
    y: (1, 128, 128) float32 in {0,1}  (Otsu mask patch)
- Model: ResUNetPlusPlus2D from earlier message.

Best practice for binary mask prediction:
- Use model out_activation="none" (logits)
- Use BCEWithLogitsLoss (numerically stable)
- Track Dice score / IoU during validation

If you want regression (e.g., porosity), swap loss to MSE and activation to tanh/linear.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Dice score for binary segmentation.
    logits: (B,1,H,W)
    target: (B,1,H,W) in {0,1}
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    # Flatten per-batch
    pred_f = pred.view(pred.size(0), -1)
    targ_f = target.view(target.size(0), -1)

    inter = (pred_f * targ_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + targ_f.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


@torch.no_grad()
def iou_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    pred_f = pred.view(pred.size(0), -1)
    targ_f = target.view(target.size(0), -1)

    inter = (pred_f * targ_f).sum(dim=1)
    union = pred_f.sum(dim=1) + targ_f.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    dice_sum = 0.0
    iou_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item())
        dice_sum += dice_score_from_logits(logits, y)
        iou_sum += iou_score_from_logits(logits, y)
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "dice": dice_sum / denom,
        "iou": iou_sum / denom,
    }


# -----------------------------
# Full training workflow
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    amp: bool = True  # mixed precision on CUDA
    save_dir: str = "checkpoints_resunetpp_2d"
    save_best: bool = True


def train_resunetpp_2d(
    model: nn.Module,
    train_ds,
    test_ds,
    cfg: TrainConfig,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Trains model and returns (best_model, best_metrics).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)




    # DataLoaders
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    model = model.to(device)

    # Loss (binary segmentation): logits + BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Optional scheduler (simple + effective)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best = {"loss": float("inf"), "dice": 0.0, "iou": 0.0, "epoch": -1}
    best_path = os.path.join(cfg.save_dir, "best.pt")
    last_path = os.path.join(cfg.save_dir, "last.pt")


    history = {
    "train_loss": [],
    "val_loss": [],
    "val_dice": [],
    "val_iou": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()

            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)

        # ---- Evaluate ----
        val_metrics = evaluate(model, test_loader, loss_fn, device)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"dice={val_metrics['dice']:.4f} | "
            f"iou={val_metrics['iou']:.4f}"
        )

        # ---- Save last checkpoint ----
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "val_metrics": val_metrics,
            },
            last_path,
        )

        # ---- Save best checkpoint (by val_loss) ----
        if cfg.save_best and val_loss < best["loss"]:
            best = {**val_metrics, "epoch": epoch}
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        # ---- Update history ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])


    # Load best weights (if saved)
    if cfg.save_best and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best = ckpt.get("val_metrics", best) | {"epoch": ckpt.get("epoch", best["epoch"])}

    return model, best, history


import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path: Optional[str] = None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # ---- Loss ----
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ---- Dice ----
    axes[1].plot(epochs, history["val_dice"], color="tab:green", linewidth=2)
    axes[1].set_title("Dice Score (Validation)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].grid(alpha=0.3)

    # ---- IoU ----
    axes[2].plot(epochs, history["val_iou"], color="tab:orange", linewidth=2)
    axes[2].set_title("IoU (Validation)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].grid(alpha=0.3)

    plt.show()
    if save_path is not None:
        fig.savefig(save_path)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Import your model definition (from your earlier converted code)
    # from scripts.models import ResUNetPlusPlus2D
    # model = ResUNetPlusPlus2D(in_channels=1, out_channels=1, out_activation="none")
    #
    # train_ds, test_ds = ...  # created from your PairedPatchDataset split function

    cfg = TrainConfig(
        epochs=30,
        batch_size=16,
        num_workers=4,
        lr=2e-4,
        grad_clip=1.0,
        amp=True,
        save_dir="checkpoints_resunetpp_2d",
    )

    # model, best, history = train_resunetpp_2d(model, train_ds, test_ds, cfg)
    # plot_training_history(history)
    # print("Best:", best)
    pass


