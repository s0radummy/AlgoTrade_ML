"""
PyTorch Dataset for TFT training on Nifty 50 historical 1-minute OHLCV candles.

Each sample is a (past_60, future_5) sliding window. Windows that cross a
session boundary (end-of-day gap) are automatically excluded.

Tensor shapes returned by __getitem__:
  static_cov:    (32,)   — stock ID, sector, market-cap embeddings + zero-pad
  past_inputs:   (60, 7) — 60-min history, all log-based features
  future_inputs: (5,  4) — calendar features for the 5 target minutes
  targets:       (5,  1) — log returns for those 5 minutes
"""

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

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


def _build_static_cov(symbol: str) -> np.ndarray:
    sector, cap = STOCK_META.get(symbol, ("IT", 1_000_000_000_000))
    log_cap = math.log10(max(cap, 1e8))
    cap_norm = 2.0 * ((log_cap - _LOG_CAP_MIN) / (_LOG_CAP_MAX - _LOG_CAP_MIN)) - 1.0
    cap_norm = float(np.clip(cap_norm, -1.0, 1.0))

    vec = np.zeros(32, dtype=np.float32)
    vec[0] = hash(symbol) % 1000 / 1000.0          # stock identity
    vec[1] = SECTOR_MAP.get(sector, 0) / 9.0        # sector (0–1)
    vec[2] = cap_norm                               # log-normalised market cap
    return vec


# ── Dataset ────────────────────────────────────────────────────────────────────

class TFTDataset(Dataset):
    """
    Sliding-window dataset over Nifty 50 1-minute OHLCV candles.
    All price features are expressed as log-returns for stationarity.
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
        split: str = "train",       # "train" or "val"
        val_days: int = 14,
        symbols: Optional[list[str]] = None,
    ):
        self.data_dir = data_dir
        self.split    = split

        self._stocks: list[dict] = []
        all_stock_idxs: list[np.ndarray] = []
        all_start_idxs: list[np.ndarray] = []

        available = sorted(
            f[:-8] for f in os.listdir(data_dir) if f.endswith(".parquet")
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
            starts = stock.pop("valid_starts")        # (k,) int32
            all_stock_idxs.append(np.full(len(starts), idx, dtype=np.int32))
            all_start_idxs.append(starts)

        # Flat index arrays — much cheaper than a Python list of tuples
        self._win_stock = np.concatenate(all_stock_idxs) if all_stock_idxs else np.array([], dtype=np.int32)
        self._win_start = np.concatenate(all_start_idxs) if all_start_idxs else np.array([], dtype=np.int32)

        print(f"  → {len(self._win_stock):,} windows across {len(self._stocks)} stocks")

    # ── Per-stock loading ──────────────────────────────────────────────────────

    def _load_stock(self, symbol: str, val_days: int) -> Optional[dict]:
        path = os.path.join(self.data_dir, f"{symbol}.parquet")
        df = pd.read_parquet(path)
        if len(df) < self.WIN_LEN + 1:
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Filter to market hours using UTC time directly
        # (avoids pytz dependency; 03:45–10:00 UTC = 09:15–15:30 IST)
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

        # prev_close: for row 0 use its own open; otherwise use previous close
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

        # (n, 7) — feature order must match inference_consumer.py when updated
        features = np.column_stack([
            log_ret, open_ret, high_ret, low_ret,
            intraday_ret, intraday_rng, vol_norm,
        ]).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Calendar features (future inputs) ─────────────────────────────────
        # Position within the 375-minute trading day — cyclically encoded.
        utc_min_of_day = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).to_numpy(np.float32)
        mins_since_open = utc_min_of_day - open_min           # 0 … 374
        frac = mins_since_open / 375.0

        calendar = np.column_stack([
            np.sin(2 * np.pi * frac),                          # cyclical time (sin)
            np.cos(2 * np.pi * frac),                          # cyclical time (cos)
            (df["timestamp"].dt.dayofweek.to_numpy(np.float32)) / 4.0,   # Mon=0 … Fri=1
            (df["timestamp"].dt.day.to_numpy(np.float32))      / 31.0,   # day of month
        ]).astype(np.float32)

        # ── Gap detection ──────────────────────────────────────────────────────
        # Timestamps are datetime64[us] — integer values are microseconds.
        # A gap > 2 minutes between consecutive candles means a session boundary.
        ts_us = df["timestamp"].astype(np.int64).to_numpy()  # microseconds since epoch
        diff_min = np.diff(ts_us) / 60_000_000               # us → minutes
        is_gap = np.zeros(len(df), dtype=bool)
        is_gap[1:] = diff_min > 2

        gap_cumsum = np.cumsum(is_gap.astype(np.int32))

        # ── Valid window starts ────────────────────────────────────────────────
        n, w = len(df), self.WIN_LEN
        if n < w:
            return None

        # Window [i … i+w-1] is valid if no gap exists inside it
        end_idx   = np.arange(w - 1, n)
        start_idx = np.arange(0, n - w + 1)
        gap_free  = (gap_cumsum[end_idx] - gap_cumsum[start_idx]) == 0

        # ── Time-based train / val split ───────────────────────────────────────
        cutoff_us = int(
            (df["timestamp"].max() - pd.Timedelta(days=val_days)).timestamp() * 1e6
        )
        win_end_us = ts_us[end_idx]

        if self.split == "train":
            split_mask = win_end_us < cutoff_us
        else:
            split_mask = win_end_us >= cutoff_us

        valid_starts = start_idx[gap_free & split_mask].astype(np.int32)

        if len(valid_starts) == 0:
            return None

        return {
            "symbol":       symbol,
            "features":     features,       # (n, 7)  float32
            "calendar":     calendar,       # (n, 4)  float32
            "static_cov":   _build_static_cov(symbol),  # (32,) float32
            "valid_starts": valid_starts,   # (k,)    int32
        }

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._win_stock)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s      = self._stocks[self._win_stock[idx]]
        start  = int(self._win_start[idx])
        p, f   = self.PAST_LEN, self.FUTURE_LEN

        past_inputs   = torch.from_numpy(s["features"][start     : start + p])         # (60, 7)
        future_inputs = torch.from_numpy(s["calendar"][start + p : start + p + f])     # (5,  4)
        targets       = torch.from_numpy(s["features"][start + p : start + p + f, :1]) # (5,  1) log_return
        static_cov    = torch.from_numpy(s["static_cov"])                               # (32,)

        return static_cov, past_inputs, future_inputs, targets


# ── Factory ────────────────────────────────────────────────────────────────────

def create_dataloaders(
    data_dir: str = "data/historical",
    batch_size: int = 256,
    val_days: int = 14,
    num_workers: int = 0,       # keep 0 on Windows; increase on Linux/Mac
    symbols: Optional[list[str]] = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TFTDataset(data_dir, split="train", val_days=val_days, symbols=symbols)
    val_ds   = TFTDataset(data_dir, split="val",   val_days=val_days, symbols=symbols)

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
    print(f"\n  past log_return  — mean: {past[:,:,0].mean():.5f}  std: {past[:,:,0].std():.5f}")
    print(f"  target log_return — mean: {target.mean():.5f}  std: {target.std():.5f}")
    print(f"\nNaN in batch: {any(t.isnan().any().item() for t in [static, past, future, target])}")
