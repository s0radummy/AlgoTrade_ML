"""
Technical indicator functions for TFT feature engineering.

All functions operate on the full per-stock 1-minute price series (not just
the 60-tick window) so indicators near window edges have full warmup history.

Used by:
  - src/data/dataset.py            (offline training dataset)
  - src/consumers/inference_consumer.py  (live inference deque)

Feature column order added to past_inputs (cols 7, 8, 9):
  7: rsi_14        — RSI(14) in [0, 1]
  8: macd_hist_norm — MACD histogram / close (dimensionless)
  9: atr_norm       — ATR(14) / close (dimensionless)
"""

import numpy as np


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed at index period-1. Values at 0..period-2 are 0."""
    n = len(arr)
    alpha = 2.0 / (period + 1)
    ema = np.zeros(n, dtype=np.float64)
    if n >= period:
        ema[period - 1] = arr[:period].mean()
        for i in range(period, n):
            ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI with Wilder's smoothing (alpha = 1/period).
    Output is in [0, 1] (RSI / 100).
    Rows 0..period-1 are filled with 0.5 (neutral).
    """
    n = len(close)
    rsi = np.full(n, 0.5, dtype=np.float32)

    if n <= period:
        return rsi

    c = close.astype(np.float64)
    deltas = np.diff(c)
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    alpha    = 1.0 / period
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, n):
        avg_gain = alpha * gains[i - 1]  + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i - 1] + (1.0 - alpha) * avg_loss
        if avg_loss < 1e-10:
            rsi[i] = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = float(rs / (1.0 + rs))

    return rsi


def compute_macd_hist(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """
    MACD histogram = (EMA_fast - EMA_slow) - signal_EMA, normalized by close.
    Rows 0..slow+signal-2 are 0 (warmup). Output is dimensionless.
    """
    n = len(close)
    result = np.zeros(n, dtype=np.float32)

    first_valid = slow + signal - 2
    if n <= first_valid:
        return result

    c      = close.astype(np.float64)
    ema_f  = _ema(c, fast)
    ema_s  = _ema(c, slow)
    macd_line = ema_f - ema_s

    # Signal line: EMA of the valid portion of MACD (starting at index slow-1)
    sig = np.zeros(n, dtype=np.float64)
    sig_vals = _ema(macd_line[slow - 1:], signal)
    sig[slow - 1:] = sig_vals

    hist = macd_line - sig

    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(c > 0, hist / c, 0.0)

    result[:] = norm.astype(np.float32)
    result[:first_valid] = 0.0
    return result


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    ATR(period) using Wilder's smoothing, normalized by close.
    Dimensionless relative volatility. Rows 0..period-1 are SMA-seeded.
    """
    n = len(close)
    result = np.zeros(n, dtype=np.float32)

    if n < 2:
        return result

    h = high.astype(np.float64)
    l = low.astype(np.float64)
    c = close.astype(np.float64)

    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )

    atr = np.zeros(n)
    if n >= period:
        alpha       = 1.0 / period
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(c > 0, atr / c, 0.0)

    result[:] = norm.astype(np.float32)
    return result
