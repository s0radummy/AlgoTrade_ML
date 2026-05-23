"""
Evaluates the trained TFT checkpoint on the held-out validation set.

Metrics reported:
  - Directional accuracy   : did P50 prediction get the sign of log-return right?
  - MAE / RMSE             : on P50 (median) forecast vs actual log-return
  - Quantile calibration   : what fraction of actuals fall below each quantile?
                             (P10 should cover ~10%, P90 should cover ~90%, etc.)
  - Per-stock breakdown    : directional accuracy per symbol

Walk-forward mode (--walk_forward N):
  Tests the model on N consecutive non-overlapping 14-day windows ending at the
  most recent data. Reports per-window directional accuracy to reveal whether
  model performance is consistent across different market regimes or luck-of-draw
  on one particular validation period.

Usage:
    py scripts/evaluate_model.py
    py scripts/evaluate_model.py --checkpoint models/tft_v1.pth
    py scripts/evaluate_model.py --val_days 14 --batch_size 512
    py scripts/evaluate_model.py --walk_forward 6   # test 6 consecutive 14-day windows
"""

import argparse
import os
import sys
import pathlib

# Allow imports from project root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TFTDataset, create_dataloaders
from src.models.tft_model import TemporalFusionTransformer


QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
Q_LABELS  = ["P10", "P30", "P50", "P70", "P90"]
P50_IDX   = 2   # index of median in the output


def evaluate(checkpoint_path: str, val_days: int, batch_size: int, data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Val days: {val_days}  |  Batch size: {batch_size}\n")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss = {ckpt['val_loss']:.6f}\n")

    target_stats = ckpt.get("target_stats", {})
    if target_stats:
        print(f"target_stats loaded for {len(target_stats)} stocks (will denormalize predictions).")
    else:
        print("No target_stats in checkpoint — metrics will be in z-scored space.")

    model = TemporalFusionTransformer()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    # ── Build val dataset ──────────────────────────────────────────────────────
    print("Loading validation dataset...")
    val_ds = TFTDataset(data_dir=data_dir, split="val", val_days=val_days)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  {len(val_ds):,} windows for evaluation\n")

    if len(val_ds) == 0:
        print("ERROR: validation set is empty - check val_days and data_dir")
        sys.exit(1)

    # ── Run inference ──────────────────────────────────────────────────────────
    all_preds   = []   # (N, 5, 5)  — 5 steps × 5 quantiles
    all_targets = []   # (N, 5, 1)

    print("Running inference over validation set...")
    with torch.no_grad():
        for i, (static_cov, past_inputs, future_inputs, targets) in enumerate(val_loader, 1):
            static_cov    = static_cov.to(device)
            past_inputs   = past_inputs.to(device)
            future_inputs = future_inputs.to(device)

            preds = model(static_cov, past_inputs, future_inputs)  # (B, 5, 5)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

            if i % 50 == 0:
                print(f"  batch {i}/{len(val_loader)}")

    preds   = np.concatenate(all_preds,   axis=0)   # (N, 5, 5)
    targets = np.concatenate(all_targets, axis=0)   # (N, 5, 1)

    # Denormalize from z-score space back to raw log-return scale.
    # Targets are z-scored per stock during training; model output is also in z-score space.
    # We use global median stats as an approximation (per-sample stock ID is not tracked here).
    if target_stats:
        global_mean = float(np.mean([v["mean"] for v in target_stats.values()]))
        global_std  = float(np.mean([v["std"]  for v in target_stats.values()]))
        print(f"Denormalizing: global_mean={global_mean:.6f}  global_std={global_std:.6f}")
        preds   = preds   * global_std + global_mean
        targets = targets * global_std + global_mean

    print(f"\nInference done. Samples: {len(preds):,}\n")
    print("=" * 60)

    # ── Flatten steps for global metrics ─────────────────────────────────────
    # (N, 5, 5) → (N*5, 5) predictions vs (N*5,) targets
    preds_flat   = preds.reshape(-1, 5)       # (N*5, 5)
    targets_flat = targets.reshape(-1)         # (N*5,)

    p50 = preds_flat[:, P50_IDX]              # median forecast

    # ── MAE / RMSE ────────────────────────────────────────────────────────────
    mae  = np.abs(p50 - targets_flat).mean()
    rmse = np.sqrt(((p50 - targets_flat) ** 2).mean())

    print(f"Median (P50) forecast - MAE : {mae:.6f}  |  RMSE : {rmse:.6f}")

    # ── Directional accuracy ──────────────────────────────────────────────────
    correct     = (np.sign(p50) == np.sign(targets_flat))
    # Exclude flat actuals (log-return == 0 exactly) from direction calc
    nonzero     = targets_flat != 0.0
    dir_acc     = correct[nonzero].mean() * 100
    n_nonzero   = nonzero.sum()

    print(f"Directional accuracy       : {dir_acc:.2f}%  ({n_nonzero:,} non-zero samples)")

    # ── Quantile calibration ──────────────────────────────────────────────────
    print("\nQuantile calibration (target ~= label%):")
    print(f"  {'Quantile':<10}  {'Expected':>10}  {'Observed':>10}  {'Gap':>8}")
    for q_idx, (q_val, q_label) in enumerate(zip(QUANTILES, Q_LABELS)):
        q_preds  = preds_flat[:, q_idx]
        observed = (targets_flat <= q_preds).mean() * 100
        expected = q_val * 100
        gap      = observed - expected
        print(f"  {q_label:<10}  {expected:>9.1f}%  {observed:>9.2f}%  {gap:>+7.2f}%")

    # -- Distribution sanity checks -----------------------------------------------
    print(f"\nTarget log-return stats (val set):")
    print(f"  mean  = {targets_flat.mean():.6f}")
    print(f"  std   = {targets_flat.std():.6f}")
    print(f"  min   = {targets_flat.min():.6f}")
    print(f"  max   = {targets_flat.max():.6f}")
    print(f"  % zero = {(targets_flat == 0).mean()*100:.1f}%")

    print(f"\nP50 prediction stats:")
    print(f"  mean  = {p50.mean():.6f}")
    print(f"  std   = {p50.std():.6f}")
    print(f"  min   = {p50.min():.6f}")
    print(f"  max   = {p50.max():.6f}")

    # ── Step-by-step breakdown ────────────────────────────────────────────────
    print("\nPer-step directional accuracy (step 1 = next candle, step 5 = 5 candles ahead):")
    print(f"  {'Step':<8}  {'Dir Acc':>10}  {'MAE':>12}  {'RMSE':>12}")
    for step in range(5):
        p50_step = preds[:, step, P50_IDX]          # (N,)
        tgt_step = targets[:, step, 0]               # (N,)
        nonzero_s = tgt_step != 0.0
        da = (np.sign(p50_step[nonzero_s]) == np.sign(tgt_step[nonzero_s])).mean() * 100
        mae_s  = np.abs(p50_step - tgt_step).mean()
        rmse_s = np.sqrt(((p50_step - tgt_step) ** 2).mean())
        print(f"  Step {step+1:<3}  {da:>9.2f}%  {mae_s:>12.6f}  {rmse_s:>12.6f}")

    # ── Monotonicity check ────────────────────────────────────────────────────
    # For valid quantile outputs: P10 <= P30 <= P50 <= P70 <= P90 should hold
    monotone = np.all(np.diff(preds_flat, axis=1) >= 0, axis=1)
    print(f"\nQuantile monotonicity (P10<=...<=P90): {monotone.mean()*100:.1f}% of samples")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


def _run_batch(model, val_ds, batch_size, device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a TFTDataset; return (preds, targets) arrays."""
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    all_preds, all_targets = [], []
    with torch.no_grad():
        for static_cov, past_inputs, future_inputs, targets in loader:
            preds = model(static_cov.to(device), past_inputs.to(device),
                          future_inputs.to(device))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate_walk_forward(
    checkpoint_path: str,
    n_windows: int,
    window_days: int,
    batch_size: int,
    data_dir: str,
):
    """
    Evaluate the model on N consecutive non-overlapping windows of window_days each,
    working backwards from the most recent data.

    This guards against the model being lucky (or unlucky) on one specific val period.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_stats = ckpt.get("target_stats", {})

    model = TemporalFusionTransformer()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    global_mean = float(np.mean([v["mean"] for v in target_stats.values()])) if target_stats else 0.0
    global_std  = float(np.mean([v["std"]  for v in target_stats.values()])) if target_stats else 1.0

    print(f"\n{'='*60}")
    print(f"Walk-forward evaluation: {n_windows} windows x {window_days} days")
    print(f"{'='*60}\n")
    print(f"  {'Window':<20}  {'Samples':>8}  {'Dir Acc':>9}  {'MAE':>12}  {'RMSE':>12}")
    print(f"  {'-'*70}")

    dir_accs = []

    for w in range(n_windows, 0, -1):
        # Window ends (n_windows - w) * window_days ago, starts window_days earlier
        days_back_end   = (w - 1) * window_days
        days_back_start = w * window_days

        # Build a TFTDataset where the "val" split corresponds to this window.
        # We exploit val_days by temporarily treating the window as val.
        # To get window [start, end]: set val_days = days_back_end..days_back_start
        # This requires a custom _load_stock split — instead, we filter by date
        # using the raw parquet files directly (lightweight approach).
        try:
            val_ds = _build_window_dataset(data_dir, days_back_start, days_back_end)
        except Exception as e:
            print(f"  Window {w}: skipped ({e})")
            continue

        if len(val_ds) == 0:
            print(f"  Window {w}: empty — skipped")
            continue

        preds, targets = _run_batch(model, val_ds, batch_size, device)

        if target_stats:
            preds   = preds   * global_std + global_mean
            targets = targets * global_std + global_mean

        preds_flat   = preds.reshape(-1, 5)
        targets_flat = targets.reshape(-1)
        p50          = preds_flat[:, P50_IDX]
        nonzero      = targets_flat != 0.0
        da           = (np.sign(p50[nonzero]) == np.sign(targets_flat[nonzero])).mean() * 100
        mae          = np.abs(p50 - targets_flat).mean()
        rmse         = np.sqrt(((p50 - targets_flat) ** 2).mean())

        end_dt   = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back_end)
        start_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back_start)
        label    = f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"

        dir_accs.append(da)
        flag = "  <-- best" if da == max(dir_accs) else ("  <-- worst" if da == min(dir_accs) and len(dir_accs) > 1 else "")
        print(f"  {label:<20}  {nonzero.sum():>8,}  {da:>8.2f}%  {mae:>12.6f}  {rmse:>12.6f}{flag}")

    if dir_accs:
        print(f"\n  Mean directional accuracy: {np.mean(dir_accs):.2f}%  "
              f"(std: {np.std(dir_accs):.2f}%  range: {min(dir_accs):.2f}–{max(dir_accs):.2f}%)")
    print(f"\n{'='*60}")
    print("Walk-forward evaluation complete.")


def _build_window_dataset(data_dir: str, days_back_start: int, days_back_end: int) -> TFTDataset:
    """
    Returns a TFTDataset covering exactly the window
    [series_end - days_back_start, series_end - days_back_end].
    Uses TFTDataset.val_start_days to bound the window on both sides.

    TFTDataset split logic: win_end >= (series_end - val_days) AND win_end < (series_end - val_start_days)
    So val_days must be the OLDER (larger) boundary and val_start_days the NEWER (smaller) boundary.
    """
    return TFTDataset(
        data_dir=data_dir,
        split="val",
        val_days=days_back_start,      # older boundary — more days back from series end
        val_start_days=days_back_end,  # newer boundary — fewer days back from series end
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFT checkpoint on validation set.")
    parser.add_argument("--checkpoint",   type=str, default="models/tft_v1.pth")
    parser.add_argument("--val_days",     type=int, default=14)
    parser.add_argument("--batch_size",   type=int, default=512)
    parser.add_argument("--data_dir",     type=str, default="data/historical")
    parser.add_argument("--walk_forward", type=int, default=0,
                        help="If > 0, run walk-forward eval over N consecutive 14-day windows")
    args = parser.parse_args()

    if args.walk_forward > 0:
        evaluate_walk_forward(
            checkpoint_path=args.checkpoint,
            n_windows=args.walk_forward,
            window_days=args.val_days,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )
    else:
        evaluate(args.checkpoint, args.val_days, args.batch_size, args.data_dir)


if __name__ == "__main__":
    main()
