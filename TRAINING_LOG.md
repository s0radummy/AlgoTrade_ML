# TFT Training Log

_Last updated: 2026-05-20 (post-fix)_

This document is a full post-mortem of all TFT training attempts. Every run, every config change, every failure, and the current recommended path forward.

---

## Model Architecture (unchanged throughout)

| Parameter | Value |
|-----------|-------|
| Parameters | 462K |
| past_inputs | (B, 60, 11) — log-ret OHLCV cols 0–6, RSI/MACD/ATR cols 7–9, Nifty50 col 10 |
| future_inputs | (B, 5, 4) — cyclical time features |
| static_covariates | (B, 32) — sector one-hot, vol profile, cap bucket |
| output | (B, 5, 5) — 5 quantiles × 5 future steps |
| loss | Pinball (quantile) loss, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9] |
| optimizer | Adam, LR=3e-4, CosineAnnealingLR(T_max=25, eta_min=1e-6) |

---

## Dataset

| Parameter | Value |
|-----------|-------|
| Stocks | 47 Nifty 50 stocks + NIFTY50 index |
| History | 3 years 1-min OHLCV |
| Train windows | ~10.7M (47 stocks × sliding 65-bar windows) |
| Val windows | ~146K (last 14 days per stock) |
| Batches/epoch | 20,827 @ batch_size=512 |
| Epoch duration | ~52–60 min (RTX 3050 Laptop) |
| Target | Per-stock z-scored log_ret (col 0) |

---

## Health Metrics

| Metric | What it measures | Healthy sign |
|--------|-----------------|--------------|
| `val_loss` | Generalization to held-out 14 days | Decreasing each epoch |
| `p50_std` | Std of P50 predictions across all val samples | Growing toward 0.01–0.02 |
| `train/val gap` | Overfitting proxy | Flat or shrinking |
| `grad_norm` | Gradient health (clipped at 1.0) | Stable 0.02–0.35 |

**Critical:** `p50_std < 1e-4` triggers collapse alarm. If all quantile predictions converge to the same value, the model is useless for uncertainty-aware trading signals even if val loss looks acceptable.

---

## Run History

### Diagnostic 1 — 2026-05-18

| Field | Value |
|-------|-------|
| Stocks | 3 (RELIANCE, INFY, HDFCBANK) |
| Epochs | 6 (early stop) |
| weight_decay | 0 |
| patience | 5 |
| Init | From scratch |
| Best val | 0.221692 @ epoch 1 |
| p50_std range | 0.001–0.007 |
| Verdict | **FAILED** |

**Root cause:** Val never improved past epoch 1. Model learned something in epoch 1 then stalled. Likely: only 3 stocks is insufficient signal diversity; patience=5 fired before CosineAnnealingLR had any effect.

---

### Diagnostic 2 — 2026-05-19 ✅

| Field | Value |
|-------|-------|
| Stocks | 10 (all sectors + cap buckets) |
| Epochs | 10 |
| weight_decay | 0 |
| patience | 15 |
| Init | From scratch |
| Best val | 0.203295 @ epoch 7 |
| p50_std range | 0.002 → 0.015 |
| Verdict | **PASSED** |

**Checkpoint saved:** `models/tft_v1.pth` — epoch 7, 10 target_stats.

This is the only confirmed-passing configuration. Key properties: no weight decay, 10+ diverse stocks, p50_std growing steadily.

---

### Run 1 (Full attempt 1) — 2026-05-19/20

| Field | Value |
|-------|-------|
| Stocks | 47 |
| Epochs | killed @ epoch 4 |
| weight_decay | 0 |
| patience | 5 |
| Init | **Warm-start from Diagnostic 2 checkpoint** |
| Best val | — |
| Verdict | **KILLED** |

**What happened:** Patience counter hit 3/5 by epoch 4. Val was worsening.

**Root cause identified (incorrectly):** Blamed overfitting. Real issue was patience=5 was too low — CosineAnnealingLR(T_max=25) doesn't meaningfully decay LR until epoch 10–15.

**Fix applied:** Increase patience to 15. Restart.

---

### Run 2 (Full attempt 2) — 2026-05-19/20

| Field | Value |
|-------|-------|
| Stocks | 47 |
| Epochs | killed @ epoch 10 |
| weight_decay | 0 |
| patience | 15 |
| Init | **Warm-start from Run 1 epoch-1 checkpoint** |

| Epoch | train | val | gap |
|-------|-------|-----|-----|
| 1 | ~0.190 | ~0.217 | ~0.027 |
| 4 | improving | worsening | widening |
| 9+ | still improving | still worsening | still widening |

**Verdict:** **KILLED** — clear train/val divergence over 9+ epochs.

**Root cause identified:** Attributed to overfitting (no L2 regularization). Also noted: warm-start from biased 10-stock checkpoint may have contributed.

**Fix applied:** Add `weight_decay=1e-4`, delete checkpoint, train from scratch.

**Note (in hindsight):** The val divergence may have been partially caused by warm-start bias (biased initial weights from 10-stock data) rather than pure overfitting. This hypothesis was not tested.

---

### Run 3 (Full attempt 3) — 2026-05-20 ← Current

| Field | Value |
|-------|-------|
| Stocks | 47 |
| Epochs | killed @ epoch 9 batch 6000 |
| weight_decay | **1e-4** |
| patience | 15 |
| Init | From scratch (checkpoint deleted) |

| Ep | train | val | p50_std | patience |
|----|-------|-----|---------|---------|
| 1 | 0.189784 | 0.214411 | 0.001588 | 0 ← checkpoint |
| 2 | 0.189535 | 0.214395 | 0.001127 | 0 ← checkpoint |
| 3 | 0.189502 | 0.214319 | 0.000439 | 0 ← checkpoint |
| 4 | 0.189485 | **0.214295** | 0.000385 | 0 ← **best** |
| 5 | 0.189478 | 0.214362 | 0.000302 | 1 |
| 6 | 0.189471 | 0.214332 | 0.000195 | 2 |
| 7 | 0.189464 | 0.214482 | 0.000189 | 3 |
| 8 | 0.189462 | 0.214335 | 0.000188 | 4 |
| 9 | killed mid-epoch | — | — | — |

**Verdict:** **FAILED**

**Root cause:** `weight_decay=1e-4` is too aggressive for this 462K-param model. The L2 penalty pushed all weights toward zero faster than the model could learn diverse predictions, causing p50_std to collapse from 0.001588 → 0.000188 (stuck near the alarm threshold of 1e-4) by epoch 8.

**Observation:** The train/val gap stayed flat at ~0.0248 across all epochs. This means overfitting was NOT actually happening — the regularization was unnecessary and harmful.

**Checkpoint state:** `models/tft_v1.pth` currently contains epoch 4 weights from this failed run (val=0.214295, p50_std=0.000385 — predictions collapsed, DO NOT USE FOR INFERENCE).

---

## Lessons Learned

### 1. patience=5 is wrong for CosineAnnealingLR(T_max=25)
LR stays at ~95%+ of initial value through epoch 5. Early stopping fires before the LR schedule has any effect. **Use patience=15 minimum.**

### 2. Never warm-start from a mismatched checkpoint
Diagnostic 2 (10-stock) checkpoint has target_stats for 10 stocks. Loading it for a 47-stock run creates biased initial weights. Always train from scratch when changing the stock universe.

### 3. weight_decay=1e-4 collapses p50_std in this model
The 462K-param TFT with quantile output is sensitive to L2 regularization strength. 1e-4 killed prediction variance without preventing overfitting. The train/val gap was flat — there was no overfitting to prevent.

### 4. p50_std growing is the primary health signal
A shrinking p50_std (even slowly) means the model is converging to constant predictions. For a trading system that needs P10–P90 quantile spread, this is a fatal failure even if val loss looks acceptable.

### 5. Val divergence in Run 2 may have been warm-start bias, not overfitting
Run 2 showed classic "train improves, val worsens" divergence — but it was warm-started from a biased checkpoint. A from-scratch run without weight decay (Diagnostic 2 configuration) passed cleanly. The regularization added in response to Run 2 may have been the wrong diagnosis.

### 6. Always run a diagnostic before committing to a full 22-hour run
Each full epoch takes ~52 min × 25 epochs ≈ 22 hours. A 10-epoch diagnostic with 10 stocks takes ~8–10 hours and catches hyperparameter failures early.

---

## Code Fixes Applied — 2026-05-20

Root causes identified via code analysis + research on pytorch-forecasting reference implementation. Three confirmed bugs, all in `src/models/training.py`:

### Fix 1: Adam → AdamW (line 64)
`Adam(weight_decay=1e-4)` applies L2 regularization INSIDE the adaptive update. For parameters with small gradients (output head after predictions start converging), Adam's `v_t → 0` amplifies the effective L2 force, creating a feedback loop: predictions constant → gradients small → effective L2 large → weights → 0 → predictions more constant. `AdamW` decouples weight decay from the adaptive step, breaking this loop. Validated by Loshchilov & Hutter (ICLR 2019) and consistent with pytorch-forecasting's default of `weight_decay=0.0`.

### Fix 2: Gradient clip 1.0 → 0.1 (line 96)
pytorch-forecasting recommends `gradient_clip_val=0.1`. Clipping at 1.0 allows 10x larger gradient steps through the attention layers, which can trigger attention entropy collapse (Zhai et al., ICML 2023) — compounding the output collapse.

### Fix 3: Soft quantile spread diversity loss (lines 40–44)
Added `0.01 * relu(0.05 - (P90 - P10)).mean()` to `QuantileLoss.forward()`. Penalizes spread below 0.05 z-score units without distorting the primary pinball signal.

### Additional safeguards added:
- Monotonic p50_std decline detection (3 consecutive epochs → immediate warning)
- Checkpoint p50_std guard (auto-skips warm-start if checkpoint has collapsed predictions)
- `--no-warmstart` CLI flag for explicit from-scratch control
- `p50_std` now saved in checkpoint dict

### Checkpoint state:
`models/tft_v1.pth` **deleted** (Run 3 epoch-4, collapsed, unsafe).

---

## Current State (as of 2026-05-20 post-fix)

| Item | State |
|------|-------|
| `models/tft_v1.pth` | **Deleted** |
| `src/models/training.py` | Fixed: AdamW, grad_clip=0.1, spread loss, collapse monitoring |
| Training process | Ready to run |

---

## Next Run — Diagnostic 4

```
py -m src.models.training --diagnostic --epochs 10 --lr 3e-4 --batch_size 256
```

**Pass criteria (by epoch 7):**
- p50_std increases or stays stable — does NOT monotonically decline
- p50_std reaches ≥ 0.005 by epoch 7
- No "declining 3+ consecutive epochs" warning
- val_loss improves between epochs 1 and 7

If Diagnostic 4 passes → full run:
```
py -m src.models.training --epochs 25 --lr 3e-4 --batch_size 512
```

### What NOT to do
- Do NOT warm-start from any checkpoint with unknown p50_std
- Do NOT use patience < 12 with CosineAnnealingLR(T_max=25)
- Do NOT reintroduce `Adam(weight_decay=...)` — use AdamW if regularization is needed

---

## Time Spent

| Run | Duration | Outcome |
|-----|----------|---------|
| Diagnostic 1 | ~5 hrs | Failed |
| Diagnostic 2 | ~10 hrs | Passed |
| Run 1 | ~4 hrs (killed ep4) | Killed |
| Run 2 | ~10 hrs (killed ep10) | Killed |
| Run 3 | ~8 hrs (killed ep9) | Failed |
| **Total** | **~37 hrs** | **0 usable full-run checkpoints** |
