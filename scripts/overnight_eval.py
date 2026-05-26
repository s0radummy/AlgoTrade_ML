"""
overnight_eval.py — Evaluate TFT model on today's live InfluxDB tick data.

Queries InfluxDB for today's session ticks, reconstructs 1-min OHLCV bars,
runs batched TFT inference, and reports directional accuracy vs the
52.65% walk-forward benchmark from historical data.

Usage:
    py scripts/overnight_eval.py
    py scripts/overnight_eval.py --date 2026-05-25
    py scripts/overnight_eval.py --output results/eval_custom.csv

InfluxDB note: timestamps stored with a -5:30h offset (timezone bug in
persistence_consumer.py). This script corrects for it automatically.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.data.dataset import _build_static_cov
from src.data.features import compute_rsi, compute_macd_hist, compute_atr
from src.models.tft_model import TemporalFusionTransformer

# ── Constants ────────────────────────────────────────────────────────────────

MIN_BARS     = 60
FUTURE_STEPS = 5
IST_OFFSET   = timedelta(hours=5, minutes=30)
OPEN_MIN_UTC = 3 * 60 + 45     # 09:15 IST = 03:45 UTC (used in future_inputs)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_checkpoint(model_path: str):
    """Load TFT checkpoint. Returns (model, target_stats)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TemporalFusionTransformer()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    target_stats = ckpt.get("target_stats", {})
    print(f"Checkpoint loaded: {model_path} ({len(target_stats)} stocks in target_stats)")
    return model, target_stats


# ── InfluxDB data fetch ───────────────────────────────────────────────────────

def fetch_bars(symbol: str, date) -> pd.DataFrame:
    """
    Query InfluxDB for a full session's tick data for one symbol, aggregate
    into 1-min OHLCV bars, and return bars for the 9:15–15:30 IST window.

    Handles two timestamp storage regimes automatically:
      Correct UTC (2026-05-27+): session 03:45–10:00 UTC stored as-is.
      Bugged  UTC (2026-05-26):  same ticks stored at UTC-5:30 (02:15–04:30 UTC).
    Tries the correct window first; falls back to the bugged window if empty.

    Returns DataFrame with columns [open, high, low, close, volume] indexed
    by true UTC timestamps. Empty DataFrame if no data found.
    """
    from influxdb_client import InfluxDBClient

    client = InfluxDBClient(
        url=f"http://{settings.influxdb_host}:{settings.influxdb_port}",
        org=settings.influxdb_org,
        token=settings.influxdb_token,
    )

    # Correct UTC  (2026-05-27+): full session 03:30–10:05 UTC
    # Bugged  UTC  (2026-05-26):  same session stored at -5:30h → 22:00 prev–04:35 UTC
    prev = date - timedelta(days=1)
    windows = [
        (f"{date.isoformat()}T03:30:00Z",  f"{date.isoformat()}T10:05:00Z",  False),
        (f"{prev.isoformat()}T22:00:00Z",  f"{date.isoformat()}T04:35:00Z",  True),
    ]

    def _query_window(start_utc: str, stop_utc: str, apply_correction: bool) -> pd.DataFrame:
        query = f'''
from(bucket: "{settings.influxdb_bucket}")
  |> range(start: {start_utc}, stop: {stop_utc})
  |> filter(fn: (r) => r["_measurement"] == "ticks" and r["symbol"] == "{symbol}")
  |> filter(fn: (r) => r["_field"] == "last_price" or r["_field"] == "volume")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
'''
        try:
            result = client.query_api().query_data_frame(query)
        except Exception as e:
            print(f"  [{symbol}] InfluxDB error: {e}")
            return pd.DataFrame()

        if result is None or (isinstance(result, list) and all(r.empty for r in result)):
            return pd.DataFrame()
        if isinstance(result, list):
            result = pd.concat(result, ignore_index=True)
        if result.empty or "_time" not in result.columns:
            return pd.DataFrame()
        if "last_price" not in result.columns or "volume" not in result.columns:
            return pd.DataFrame()

        times = pd.to_datetime(result["_time"], utc=True)
        if apply_correction:
            times = times + IST_OFFSET          # undo the -5:30h storage bug → true UTC
        result["_time"] = times
        result = result.set_index("_time").sort_index()

        lp  = result["last_price"].dropna()
        vol = result["volume"].dropna()

        bars = pd.DataFrame({
            "open":    lp.resample("1min").first(),
            "high":    lp.resample("1min").max(),
            "low":     lp.resample("1min").min(),
            "close":   lp.resample("1min").last(),
            "vol_max": vol.resample("1min").max(),
            "vol_min": vol.resample("1min").min(),
        })
        bars["volume"] = (bars["vol_max"] - bars["vol_min"]).clip(lower=0)
        bars = bars[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"])

        # Filter to IST market hours 9:15–15:30 (index is true UTC after correction)
        ist_idx = bars.index + IST_OFFSET
        ist_min = ist_idx.hour * 60 + ist_idx.minute
        bars    = bars[(ist_min >= 9 * 60 + 15) & (ist_min <= 15 * 60 + 30)]
        return bars

    try:
        for start_utc, stop_utc, apply_correction in windows:
            bars = _query_window(start_utc, stop_utc, apply_correction)
            if not bars.empty and len(bars) >= 10:
                return bars
        return pd.DataFrame()
    finally:
        client.close()


# ── Feature construction ──────────────────────────────────────────────────────

def build_feature_matrix(bars: pd.DataFrame) -> np.ndarray:
    """
    Build (N, 11) feature matrix matching inference_consumer._bars_to_features
    exactly. Col 10 (nifty_log_ret) is zero — NIFTY50 not in InfluxDB.
    """
    close = bars["close"].to_numpy(dtype=np.float64)
    open_ = bars["open"].to_numpy(dtype=np.float64)
    high  = bars["high"].to_numpy(dtype=np.float64)
    low   = bars["low"].to_numpy(dtype=np.float64)
    vol   = bars["volume"].to_numpy(dtype=np.float64)
    n     = len(close)

    prev_close    = np.empty_like(close)
    prev_close[0] = open_[0]
    prev_close[1:] = close[:-1]

    pos_vol  = vol[vol > 0]
    mean_vol = pos_vol.mean() if len(pos_vol) > 0 else 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret      = np.log(close / prev_close)
        open_ret     = np.log(open_ / prev_close)
        high_ret     = np.log(high  / prev_close)
        low_ret      = np.log(low   / prev_close)
        intraday_ret = np.log(close / np.where(open_ > 0, open_, 1.0))
        intraday_rng = np.log(high  / np.where(low   > 0, low,   1.0))
        vol_norm     = np.log1p(vol / mean_vol)

    rsi_vals  = compute_rsi(close).astype(np.float64)
    macd_vals = compute_macd_hist(close).astype(np.float64)
    atr_vals  = compute_atr(high, low, close).astype(np.float64)
    nifty_col = np.zeros(n, dtype=np.float64)

    mat = np.column_stack([
        log_ret, open_ret, high_ret, low_ret,
        intraday_ret, intraday_rng, vol_norm,
        rsi_vals, macd_vals, atr_vals, nifty_col,
    ])
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_future_inputs_batch(timestamps: pd.DatetimeIndex, n_windows: int) -> np.ndarray:
    """
    Build (n_windows, 5, 4) future calendar inputs.
    timestamps[i+MIN_BARS-1] is the last input bar for window i.
    """
    feats = np.zeros((n_windows, FUTURE_STEPS, 4), dtype=np.float32)
    for i in range(n_windows):
        ts = timestamps[i + MIN_BARS - 1].to_pydatetime()
        mins_since_open = (ts.hour * 60 + ts.minute) - OPEN_MIN_UTC
        for step in range(FUTURE_STEPS):
            frac = (mins_since_open + step + 1) / 375.0
            feats[i, step, 0] = float(np.sin(2 * np.pi * frac))
            feats[i, step, 1] = float(np.cos(2 * np.pi * frac))
            feats[i, step, 2] = float(ts.weekday()) / 4.0
            feats[i, step, 3] = float(ts.day) / 31.0
    return feats


# ── Per-stock evaluation ──────────────────────────────────────────────────────

def evaluate_stock(
    model: torch.nn.Module,
    symbol: str,
    bars: pd.DataFrame,
    target_stats: dict,
) -> list:
    """
    Batched rolling inference for one stock.
    Returns list of result dicts, one per prediction step.
    """
    stats  = target_stats.get(symbol, {"mean": 0.0, "std": 0.0007})
    t_mean = float(stats.get("mean", 0.0))
    t_std  = float(stats.get("std",  0.0007))

    feat_mat = build_feature_matrix(bars)   # (N, 11)
    if t_std > 0:
        feat_mat[:, 0] = (feat_mat[:, 0] - t_mean) / t_std   # z-score col 0

    closes     = bars["close"].to_numpy(dtype=np.float64)
    timestamps = bars.index   # actual UTC timestamps (post +5:30h correction)
    n          = len(feat_mat)
    n_windows  = n - MIN_BARS  # need MIN_BARS input + 1 future bar for ground truth

    if n_windows <= 0:
        return []

    # Build batched tensors — (n_windows, 60, 11)
    past_batch = np.stack([feat_mat[i : i + MIN_BARS] for i in range(n_windows)])

    # Static covariates — broadcast same vector across all windows
    static_np  = _build_static_cov(symbol, target_std=t_std)
    static_t   = torch.from_numpy(static_np).unsqueeze(0).expand(n_windows, -1).float()  # (n_windows, 32)

    # Future inputs — (n_windows, 5, 4)
    future_batch = build_future_inputs_batch(timestamps, n_windows)

    past_t   = torch.from_numpy(past_batch).float()
    future_t = torch.from_numpy(future_batch).float()

    with torch.no_grad():
        output = model(static_t, past_t, future_t)   # (n_windows, 5, 5) in z-score space

    preds_z  = output.cpu().numpy()              # (n_windows, 5, 5)
    preds    = preds_z * t_std + t_mean          # denormalize to raw log-return scale

    # Ground truth: log return at bar (i+MIN_BARS) vs bar (i+MIN_BARS-1)
    # close[i+MIN_BARS] / close[i+MIN_BARS-1] for i=0..n_windows-1
    actual_lrs = np.log(
        closes[MIN_BARS : MIN_BARS + n_windows] /
        closes[MIN_BARS - 1 : MIN_BARS + n_windows - 1]
    )

    results = []
    for i in range(n_windows):
        p10, p30, p50, p70, p90 = preds[i, 0]   # step+1 only
        actual   = float(actual_lrs[i])
        dir_ok   = int(p50 * actual > 0)

        # Label: IST time of last input bar (when prediction is made)
        bar_ts   = timestamps[i + MIN_BARS - 1]
        bar_ist  = (bar_ts + IST_OFFSET).strftime("%H:%M")

        results.append({
            "symbol":      symbol,
            "bar_time_ist": bar_ist,
            "P10":         round(float(p10), 7),
            "P30":         round(float(p30), 7),
            "P50":         round(float(p50), 7),
            "P70":         round(float(p70), 7),
            "P90":         round(float(p90), 7),
            "actual":      round(actual, 7),
            "dir_correct": dir_ok,
        })

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Overnight TFT evaluation on live InfluxDB session data")
    parser.add_argument("--date",   default="2026-05-26",
                        help="Session date in IST, YYYY-MM-DD (default: today)")
    parser.add_argument("--output", default="results/live_eval_2026-05-26.csv")
    parser.add_argument("--model",  default=settings.model_path)
    args = parser.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d").date()

    print(f"\nOvernight TFT Evaluation — {args.date}")
    print(f"  Tries correct-UTC window first ({date}T03:30Z→10:05Z),")
    print(f"  falls back to bugged-UTC window (prev-dayT22:00Z→{date}T04:35Z + +5:30h).")
    print(f"  Model:  {args.model}")
    print(f"  Output: {args.output}\n")

    if not os.path.exists(args.model):
        print(f"ERROR: model checkpoint not found at {args.model}")
        sys.exit(1)

    model, target_stats = load_checkpoint(args.model)

    stocks = [s for s in settings.stock_list if s in target_stats]
    print(f"Evaluating {len(stocks)}/{len(settings.stock_list)} stocks (those in target_stats)...\n")
    print(f"  {'Symbol':12s}  {'n':>4s}  {'DirAcc':>7s}  {'MAE':>10s}  {'Spread':>8s}  {'P10cov':>6s}  {'P90cov':>6s}")
    print(f"  {'-'*70}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    all_results     = []
    stock_summaries = []

    for symbol in stocks:
        bars = fetch_bars(symbol, date)

        if bars.empty or len(bars) < MIN_BARS + 2:
            print(f"  [{symbol:12s}]  SKIP — {len(bars)} bars (need ≥{MIN_BARS + 2})")
            continue

        results = evaluate_stock(model, symbol, bars, target_stats)
        if not results:
            print(f"  [{symbol:12s}]  SKIP — no inference results")
            continue

        n_pairs  = len(results)
        dir_acc  = sum(r["dir_correct"] for r in results) / n_pairs * 100
        mae      = np.mean([abs(r["P50"] - r["actual"]) for r in results])
        spread   = np.mean([r["P90"] - r["P10"] for r in results])
        p10_cov  = np.mean([r["actual"] < r["P10"] for r in results]) * 100
        p90_cov  = np.mean([r["actual"] < r["P90"] for r in results]) * 100

        print(f"  {symbol:12s}  {n_pairs:4d}  {dir_acc:6.1f}%  {mae:10.7f}  {spread:8.5f}  {p10_cov:5.0f}%  {p90_cov:5.0f}%")

        stock_summaries.append({
            "symbol":           symbol,
            "n_pairs":          n_pairs,
            "dir_accuracy_pct": round(dir_acc,  2),
            "mae":              round(mae,       7),
            "mean_spread":      round(spread,    6),
            "p10_coverage_pct": round(p10_cov,  1),
            "p90_coverage_pct": round(p90_cov,  1),
        })
        all_results.extend(results)

    if not all_results:
        print("\nNo results — check InfluxDB connection and date range.")
        return

    # ── Aggregate summary ─────────────────────────────────────────────────────
    total_pairs  = len(all_results)
    overall_dir  = sum(r["dir_correct"] for r in all_results) / total_pairs * 100
    overall_mae  = np.mean([abs(r["P50"] - r["actual"]) for r in all_results])
    overall_sprd = np.mean([r["P90"] - r["P10"] for r in all_results])
    p10_overall  = np.mean([r["actual"] < r["P10"] for r in all_results]) * 100
    p90_overall  = np.mean([r["actual"] < r["P90"] for r in all_results]) * 100

    print(f"\n  {'-'*70}")
    print(f"  {'AGGREGATE':12s}  {total_pairs:4d}  {overall_dir:6.1f}%  {overall_mae:10.7f}  {overall_sprd:8.5f}  {p10_overall:5.0f}%  {p90_overall:5.0f}%")
    print(f"\n  Walk-forward benchmark (historical): 52.65% dir accuracy")
    delta = overall_dir - 52.65
    sign  = "+" if delta >= 0 else ""
    print(f"  Live delta vs benchmark:             {sign}{delta:.2f}pp")
    print(f"\n  Quantile calibration:")
    print(f"    P10 empirical coverage: {p10_overall:.1f}%  (ideal: 10%)")
    print(f"    P90 empirical coverage: {p90_overall:.1f}%  (ideal: 90%)")
    print(f"    Mean quantile spread:   {overall_sprd:.5f}  (training diversity floor: 0.05)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nDetailed results -> {args.output}")

    summary_path = args.output.replace(".csv", "_summary.csv")
    pd.DataFrame(stock_summaries).sort_values("dir_accuracy_pct", ascending=False).to_csv(summary_path, index=False)
    print(f"Per-stock summary -> {summary_path}")


if __name__ == "__main__":
    main()
