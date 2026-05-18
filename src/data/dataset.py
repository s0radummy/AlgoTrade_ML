"""
PyTorch Dataset for TFT training on Nifty 50 historical 1-minute OHLCV candles.

Each sample is a (past_60, future_5) sliding window. Windows that cross a
session boundary (end-of-day gap) are automatically excluded.

Tensor shapes returned by __getitem__:
  static_cov:    (32,)    — stock ID, sector one-hot, market-cap, volatility profile
  past_inputs:   (60, 11) — 60-min history: 7 log-return features + RSI + MACD hist + ATR
                             + Nifty50 index log-return (col 10)
  future_inputs: (5,  4)  — calendar features for the 5 target minutes
  targets:       (5,  1)  — z-scored log returns for those 5 minutes

Target standardization:
  log_ret targets are z-scored per stock (mean/std computed over full training series).
  The col-0 (log_ret) channel of past_inputs is also z-scored for consistency.
  Per-stock (mean, std) are stored in each stock's dict and saved in the checkpoint
  so inference can denormalize predictions back to raw log-return scale.

Nifty50 feature (col 10):
  data/historical/NIFTY50.parquet must exist. Fetch with:
    py scripts/fetch_historical_data.py --symbol NIFTY50
  If missing, dataset raises a clear error rather than silently training with zeros.
"""

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.features import compute_rsi, compute_macd_hist, compute_atr

# ── Stock metadata ─────────────────────────────────────────────────────────────

SECTOR_MAP = {
    "IT": 0, "Finance": 1, "Energy": 2, "Auto": 3,
    "Consumer": 4, "Healthcare": 5, "Industrials": 6,
    "Materials": 7, "Telecom": 8, "Utilities": 9,
}

# (sector, approximate market cap in INR)
STOCK_META: dict[str, tuple[str, int]] = {
    "ADANIENT":   ("Energy",       4_500_000_000_000),
    "ADANIPORTS": ("Industrials",  2_800_000_000_000),
    "APOLLOHOSP": ("Healthcare",   1_000_000_000_000),
    "ASIANPAINT": ("Consumer",     2_400_000_000_000),
    "AXISBANK":   ("Finance",      3_800_000_000_000),
    "BAJAJ-AUTO": ("Auto",         1_800_000_000_000),
    "BAJFINANCE": ("Finance",      4_500_000_000_000),
    "BAJAJFINSV": ("Finance",      2_800_000_000_000),
    "BEL":        ("Industrials",    900_000_000_000),
    "BHARTIARTL": ("Telecom",      8_500_000_000_000),
    "BRITANNIA":  ("Consumer",     1_200_000_000_000),
    "CIPLA":      ("Healthcare",   1_100_000_000_000),
    "COALINDIA":  ("Energy",       2_400_000_000_000),
    "DIVISLAB":   ("Healthcare",     900_000_000_000),
    "DRREDDY":    ("Healthcare",   1_200_000_000_000),
    "EICHERMOT":  ("Auto",         1_400_000_000_000),
    "GRASIM":     ("Materials",    1_500_000_000_000),
    "HCLTECH":    ("IT",           4_500_000_000_000),
    "HDFCBANK":   ("Finance",     12_000_000_000_000),
    "HDFCLIFE":   ("Finance",      1_500_000_000_000),
    "HEROMOTOCO": ("Auto",         1_000_000_000_000),
    "HINDALCO":   ("Materials",    1_400_000_000_000),
    "HINDUNILVR": ("Consumer",     5_500_000_000_000),
    "ICICIBANK":  ("Finance",      8_500_000_000_000),
    "ITC":        ("Consumer",     5_500_000_000_000),
    "INDUSINDBK": ("Finance",      1_200_000_000_000),
    "INFY":       ("IT",           8_000_000_000_000),
    "JSWSTEEL":   ("Materials",    2_000_000_000_000),
    "KOTAKBANK":  ("Finance",      3_500_000_000_000),
    "LT":         ("Industrials",  4_500_000_000_000),
    "LTIM":       ("IT",           1_500_000_000_000),
    "M&M":        ("Auto",         3_500_000_000_000),
    "MARUTI":     ("Auto",         3_800_000_000_000),
    "NTPC":       ("Utilities",    3_000_000_000_000),
    "NESTLEIND":  ("Consumer",     2_400_000_000_000),
    "ONGC":       ("Energy",       3_500_000_000_000),
    "POWERGRID":  ("Utilities",    2_500_000_000_000),
    "RELIANCE":   ("Energy",      16_000_000_000_000),
    "SBILIFE":    ("Finance",      1_600_000_000_000),
    "SBIN":       ("Finance",      7_000_000_000_000),
    "SUNPHARMA":  ("Healthcare",   3_500_000_000_000),
    "TCS":        ("IT",          14_000_000_000_000),
    "TATACONSUM": ("Consumer",     1_200_000_000_000),
    "TATAMOTORS": ("Auto",         3_500_000_000_000),
    "TATASTEEL":  ("Materials",    1_800_000_000_000),
    "TECHM":      ("IT",           1_200_000_000_000),
    "TITAN":      ("Consumer",     2_800_000_000_000),
    "ULTRACEMCO": ("Materials",    2_400_000_000_000),
    "WIPRO":      ("IT",           2_800_000_000_000),
    "ZOMATO":     ("Consumer",     2_000_000_000_000),
}

_LOG_CAP_MIN = math.log10(100_000_000)
_LOG_CAP_MAX = math.log10(20_000_000_000_000)

# log10(INR) boundaries for 8 market-cap buckets
_CAP_BUCKET_EDGES = [12.2, 12.4, 12.6, 12.8, 13.0, 13.3, 13.6]


def _build_static_cov(symbol: str, target_std: float = 0.0007) -> np.ndarray:
    """
    Build the 32-dim static covariate vector for a stock.

    Layout (22 of 32 dims are meaningful):
      0:      stock identity (hash scalar)
      1:      sector ordinal (0–1 scalar, kept for legacy)
      2:      log-normalized market cap (–1 to 1)
      3–12:   sector one-hot (10 dims) — removes false ordinal relationship
      13:     historical return volatility (target_std / 0.002, clipped [0,1])
      14–21:  market-cap bucket one-hot (8 buckets by log10 INR)
      22–31:  reserved zeros
    """
    sector, cap = STOCK_META.get(symbol, ("IT", 1_000_000_000_000))
    sector_idx  = SECTOR_MAP.get(sector, 0)
    log_cap     = math.log10(max(cap, 1e8))
    cap_norm    = 2.0 * ((log_cap - _LOG_CAP_MIN) / (_LOG_CAP_MAX - _LOG_CAP_MIN)) - 1.0
    cap_norm    = float(np.clip(cap_norm, -1.0, 1.0))

    vec = np.zeros(32, dtype=np.float32)

    vec[0] = hash(symbol) % 1000 / 1000.0   # stock identity
    vec[1] = sector_idx / 9.0               # sector ordinal (legacy)
    vec[2] = cap_norm                        # log-normalised market cap

    # sector one-hot (dims 3–12)
    vec[3 + sector_idx] = 1.0

    # historical return volatility (dim 13)
    vec[13] = float(np.clip(target_std / 0.002, 0.0, 1.0))

    # market-cap bucket one-hot (dims 14–21)
    bucket = int(np.searchsorted(_CAP_BUCKET_EDGES, log_cap))
    vec[14 + min(bucket, 7)] = 1.0

    # dims 22–31: reserved zeros
    return vec


# ── Dataset ────────────────────────────────────────────────────────────────────

class TFTDataset(Dataset):
    """
    Sliding-window dataset over Nifty 50 1-minute OHLCV candles.

    Price features are log-returns (stationarity). Technical indicators
    (RSI, MACD hist, ATR) are appended as cols 7–9. Log-return targets
    are z-scored per stock to eliminate the degenerate zero-prediction
    minimum of the pinball loss.
    """

    PAST_LEN   = 60
    FUTURE_LEN = 5
    WIN_LEN    = PAST_LEN + FUTURE_LEN  # 65 candles per sample

    # NSE market hours in UTC (09:15–15:30 IST = 03:45–10:00 UTC)
    _MARKET_OPEN_UTC_H,  _MARKET_OPEN_UTC_M  = 3, 45
    _MARKET_CLOSE_UTC_H, _MARKET_CLOSE_UTC_M = 10, 0

    def __init__(
        self,
        data_dir: str = "data/historical",
        split: str = "train",
        val_days: int = 14,
        val_start_days: Optional[int] = None,
        symbols: Optional[list[str]] = None,
    ):
        """
        val_days:       Windows whose last candle is within the last val_days days → val split.
        val_start_days: If provided (must be > val_days), val windows are restricted to
                        [now - val_start_days, now - val_days]. Used by walk-forward evaluation
                        to select non-overlapping windows.
        """
        self.data_dir      = data_dir
        self.split         = split
        self.val_days      = val_days
        self.val_start_days = val_start_days

        # Preload Nifty50 1-min log returns indexed by UTC timestamp.
        # Must be loaded before _load_stock is called for any individual stock.
        self._nifty_log_ret: Optional[pd.Series] = self._load_nifty(data_dir)

        self._stocks: list[dict] = []
        all_stock_idxs: list[np.ndarray] = []
        all_start_idxs: list[np.ndarray] = []

        available = sorted(
            f[:-8] for f in os.listdir(data_dir)
            if f.endswith(".parquet") and f[:-8] != "NIFTY50"
        )
        if symbols:
            available = [s for s in available if s in symbols]

        print(f"[{split}] Loading {len(available)} stocks...")
        for symbol in available:
            stock = self._load_stock(symbol, val_days)
            if stock is None:
                continue
            idx = len(self._stocks)
            self._stocks.append(stock)
            starts = stock.pop("valid_starts")
            all_stock_idxs.append(np.full(len(starts), idx, dtype=np.int32))
            all_start_idxs.append(starts)

        self._win_stock = np.concatenate(all_stock_idxs) if all_stock_idxs else np.array([], dtype=np.int32)
        self._win_start = np.concatenate(all_start_idxs) if all_start_idxs else np.array([], dtype=np.int32)

        print(f"  => {len(self._win_stock):,} windows across {len(self._stocks)} stocks")

    @staticmethod
    def _load_nifty(data_dir: str) -> Optional[pd.Series]:
        """Load NIFTY50 1-min log returns indexed by UTC timestamp."""
        path = os.path.join(data_dir, "NIFTY50.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"NIFTY50.parquet not found at {path}. "
                "Fetch it first:\n"
                "  py scripts/fetch_historical_data.py --symbol NIFTY50"
            )
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        close = df.set_index("timestamp")["close"].astype(np.float64)
        log_ret = np.log(close / close.shift(1)).fillna(0.0)
        return log_ret

    # ── Per-stock loading ──────────────────────────────────────────────────────

    def _load_stock(self, symbol: str, val_days: int) -> Optional[dict]:
        path = os.path.join(self.data_dir, f"{symbol}.parquet")
        df = pd.read_parquet(path)
        if len(df) < self.WIN_LEN + 1:
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Filter to market hours (avoids pytz dependency; 03:45–10:00 UTC = 09:15–15:30 IST)
        utc_h = df["timestamp"].dt.hour
        utc_m = df["timestamp"].dt.minute
        utc_minutes = utc_h * 60 + utc_m
        open_min  = self._MARKET_OPEN_UTC_H  * 60 + self._MARKET_OPEN_UTC_M   # 225
        close_min = self._MARKET_CLOSE_UTC_H * 60 + self._MARKET_CLOSE_UTC_M  # 600
        df = df[utc_minutes.between(open_min, close_min - 1)].reset_index(drop=True)

        if len(df) < self.WIN_LEN + 1:
            return None

        # ── Features ──────────────────────────────────────────────────────────
        close  = df["close"].to_numpy(dtype=np.float64)
        open_  = df["open"].to_numpy(dtype=np.float64)
        high   = df["high"].to_numpy(dtype=np.float64)
        low    = df["low"].to_numpy(dtype=np.float64)
        vol    = df["volume"].to_numpy(dtype=np.float64)

        prev_close = np.empty_like(close)
        prev_close[0] = open_[0]
        prev_close[1:] = close[:-1]

        pos_vol  = vol[vol > 0]
        mean_vol = pos_vol.mean() if len(pos_vol) else 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret      = np.log(close  / prev_close)
            open_ret     = np.log(open_  / prev_close)
            high_ret     = np.log(high   / prev_close)
            low_ret      = np.log(low    / prev_close)
            intraday_ret = np.log(close  / np.where(open_ > 0, open_, 1.0))
            intraday_rng = np.log(high   / np.where(low   > 0, low,   1.0))
            vol_norm     = np.log1p(vol  / mean_vol)

        # Technical indicators (cols 7, 8, 9)
        rsi       = compute_rsi(close)             # [0, 1]
        macd_hist = compute_macd_hist(close)       # dimensionless, ~(-0.01, 0.01)
        atr_norm  = compute_atr(high, low, close)  # dimensionless, ~(0, 0.02)

        # Nifty50 index return (col 10) — separates idiosyncratic from market-wide moves
        if self._nifty_log_ret is not None:
            nifty_col = df["timestamp"].map(self._nifty_log_ret).to_numpy(dtype=np.float64)
            nifty_col = np.where(np.isfinite(nifty_col), nifty_col, 0.0)
        else:
            nifty_col = np.zeros(len(df), dtype=np.float64)

        # (n, 11) — feature order must match inference_consumer.py
        features = np.column_stack([
            log_ret, open_ret, high_ret, low_ret,
            intraday_ret, intraday_rng, vol_norm,
            rsi, macd_hist, atr_norm, nifty_col,
        ]).astype(np.float32)
        # nan/inf cleanup must happen AFTER all 11 columns are assembled
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Per-stock target normalization stats ───────────────────────────────
        # Computed from the full series so distribution is stable.
        # Used to z-score col-0 (log_ret) in both past_inputs and targets,
        # eliminating the degenerate pinball-loss minimum at zero.
        log_ret_series = features[:, 0]
        target_mean = float(log_ret_series.mean())
        target_std  = float(log_ret_series.std())
        if target_std < 1e-8:
            target_std = 1.0  # degenerate stock (constant price) — skip

        # ── Calendar features (future inputs) ─────────────────────────────────
        utc_min_of_day  = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).to_numpy(np.float32)
        mins_since_open = utc_min_of_day - open_min
        frac = mins_since_open / 375.0

        calendar = np.column_stack([
            np.sin(2 * np.pi * frac),
            np.cos(2 * np.pi * frac),
            (df["timestamp"].dt.dayofweek.to_numpy(np.float32)) / 4.0,
            (df["timestamp"].dt.day.to_numpy(np.float32))       / 31.0,
        ]).astype(np.float32)

        # ── Gap detection ──────────────────────────────────────────────────────
        ts_us    = df["timestamp"].astype(np.int64).to_numpy()
        diff_min = np.diff(ts_us) / 60_000_000
        is_gap   = np.zeros(len(df), dtype=bool)
        is_gap[1:] = diff_min > 2

        gap_cumsum = np.cumsum(is_gap.astype(np.int32))

        # ── Valid window starts ────────────────────────────────────────────────
        n, w = len(df), self.WIN_LEN
        if n < w:
            return None

        end_idx   = np.arange(w - 1, n)
        start_idx = np.arange(0, n - w + 1)
        gap_free  = (gap_cumsum[end_idx] - gap_cumsum[start_idx]) == 0

        # ── Time-based train / val split ───────────────────────────────────────
        # Anchor on the series max so splits are reproducible per stock, not wall-clock.
        series_end_us = int(df["timestamp"].max().timestamp() * 1e6)
        cutoff_us     = series_end_us - val_days * 86_400 * 1_000_000
        win_end_us    = ts_us[end_idx]

        if self.split == "train":
            split_mask = win_end_us < cutoff_us
        elif self.val_start_days is not None:
            # Restrict val to a specific window: [series_end - val_start_days, series_end - val_days]
            start_cutoff_us = series_end_us - self.val_start_days * 86_400 * 1_000_000
            split_mask = (win_end_us >= cutoff_us) & (win_end_us < start_cutoff_us)
        else:
            split_mask = win_end_us >= cutoff_us

        valid_starts = start_idx[gap_free & split_mask].astype(np.int32)

        if len(valid_starts) == 0:
            return None

        return {
            "symbol":       symbol,
            "features":     features,                                   # (n, 11) float32
            "calendar":     calendar,                                   # (n,  4) float32
            "static_cov":   _build_static_cov(symbol, target_std),     # (32,)   float32
            "target_mean":  target_mean,
            "target_std":   target_std,
            "valid_starts": valid_starts,                               # (k,)    int32
        }

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._win_stock)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s     = self._stocks[self._win_stock[idx]]
        start = int(self._win_start[idx])
        p, f  = self.PAST_LEN, self.FUTURE_LEN

        # Z-score only col 0 (log_ret) of past_inputs; other cols are already well-scaled
        past_raw       = s["features"][start : start + p].copy()
        past_raw[:, 0] = (past_raw[:, 0] - s["target_mean"]) / s["target_std"]
        past_inputs    = torch.from_numpy(past_raw).float()           # (60, 11)

        future_inputs  = torch.from_numpy(
            s["calendar"][start + p : start + p + f]
        ).float()                                                      # (5,  4)

        raw_targets    = s["features"][start + p : start + p + f, :1] # (5,  1)
        targets        = torch.from_numpy(
            (raw_targets - s["target_mean"]) / s["target_std"]
        ).float()                                                      # (5,  1) z-scored

        static_cov     = torch.from_numpy(s["static_cov"]).float()    # (32,)

        return static_cov, past_inputs, future_inputs, targets


# ── Factory ────────────────────────────────────────────────────────────────────

def create_dataloaders(
    data_dir: str = "data/historical",
    batch_size: int = 256,
    val_days: int = 14,
    num_workers: int = 0,
    symbols: Optional[list[str]] = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TFTDataset(data_dir, split="train", val_days=val_days, symbols=symbols)
    val_ds   = TFTDataset(data_dir, split="val",   val_days=val_days, symbols=symbols)
    # val_start_days not set here — training uses the standard single-window val split

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_dl, val_dl = create_dataloaders(batch_size=256)

    print("\nSampling one batch from train loader...")
    static, past, future, target = next(iter(train_dl))
    print(f"  static_cov:    {tuple(static.shape)}   dtype={static.dtype}")
    print(f"  past_inputs:   {tuple(past.shape)}  dtype={past.dtype}")
    print(f"  future_inputs: {tuple(future.shape)}   dtype={future.dtype}")
    print(f"  targets:       {tuple(target.shape)}   dtype={target.dtype}")
    print(f"\n  past log_ret (z-scored) — mean: {past[:,:,0].mean():.4f}  std: {past[:,:,0].std():.4f}")
    print(f"  past RSI                — mean: {past[:,:,7].mean():.4f}  std: {past[:,:,7].std():.4f}")
    print(f"  past nifty_log_ret      — mean: {past[:,:,10].mean():.4f}  std: {past[:,:,10].std():.4f}")
    print(f"  target (z-scored)       — mean: {target.mean():.4f}        std: {target.std():.4f}")
    print(f"\nNaN in batch: {any(t.isnan().any().item() for t in [static, past, future, target])}")
