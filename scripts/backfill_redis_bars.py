"""
backfill_redis_bars.py — Rebuild 1-min bar history in Redis from InfluxDB tick data.

Queries a full trading session's raw ticks from InfluxDB, builds 1-min OHLCV bars
per symbol using the same volume-delta method as VizConsumer, and writes them to
STOCK:{sym}:bars (up to 300 bars, newest at index 0) and STOCK:{sym}:current_bar.

This allows InferenceConsumer to warm-start immediately the next morning without
waiting 60+ minutes for live bars to accumulate.

Usage:
    py scripts/backfill_redis_bars.py                    # previous trading day
    py scripts/backfill_redis_bars.py --date 2026-05-27
    py scripts/backfill_redis_bars.py --date 2026-05-27 --flush
"""

import argparse
import json
import os
import sys
from datetime import date, timedelta

import pandas as pd
import redis as redis_lib
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()

INFLUX_URL    = f"http://{os.getenv('INFLUXDB_HOST', 'localhost')}:{os.getenv('INFLUXDB_PORT', '8086')}"
INFLUX_ORG    = os.getenv("INFLUXDB_ORG",   "algotrade")
INFLUX_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "algotrade-dev-token-2024")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "stocks")
REDIS_HOST    = os.getenv("REDIS_HOST",      "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT",  "6379"))

BAR_CAP = 300
# NSE session in UTC: 09:15 IST = 03:45 UTC, 15:30 IST = 10:00 UTC
SESSION_START_UTC = 3 * 60 + 45   # minutes since midnight UTC
SESSION_END_UTC   = 10 * 60 + 0


def prev_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:   # skip Sat/Sun
        d -= timedelta(days=1)
    return d


def build_bars(sym_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tick DataFrame into 1-min OHLCV bars filtered to the NSE session window.
    Volume is the per-bar delta of KiteConnect's cumulative daily volume.
    """
    sym_df = sym_df.copy()
    sym_df.index = pd.to_datetime(sym_df["_time"], utc=True)
    sym_df = sym_df.sort_index()

    lp  = sym_df["last_price"].dropna()
    vol = sym_df["volume"].dropna()

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

    # Filter to NSE session (UTC)
    ut_min = bars.index.hour * 60 + bars.index.minute
    bars = bars[(ut_min >= SESSION_START_UTC) & (ut_min <= SESSION_END_UTC)]
    return bars


def main():
    parser = argparse.ArgumentParser(description="Backfill Redis bar lists from InfluxDB")
    parser.add_argument("--date",  default=None,
                        help="Session date YYYY-MM-DD (default: previous trading day)")
    parser.add_argument("--flush", action="store_true",
                        help="Delete existing STOCK:*:bars keys before writing")
    args = parser.parse_args()

    session_date = (
        date.fromisoformat(args.date) if args.date
        else prev_trading_day(date.today())
    )
    print(f"Backfilling bars from session: {session_date}")

    # ── Query InfluxDB ──────────────────────────────────────────────────────────
    start = f"{session_date}T03:30:00Z"
    stop  = f"{session_date}T10:10:00Z"
    print(f"Querying InfluxDB  {start} to {stop} ...")

    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r["_measurement"] == "ticks")
  |> filter(fn: (r) => r["_field"] == "last_price" or r["_field"] == "volume")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
'''

    client = InfluxDBClient(url=INFLUX_URL, org=INFLUX_ORG, token=INFLUX_TOKEN)
    try:
        result = client.query_api().query_data_frame(flux)
        if isinstance(result, list):
            result = pd.concat(result, ignore_index=True) if result else pd.DataFrame()
    finally:
        client.close()

    if result is None or result.empty:
        print(f"ERROR: No data in InfluxDB for {session_date}.")
        sys.exit(1)

    required = {"_time", "last_price", "volume", "symbol"}
    if not required.issubset(result.columns):
        print(f"ERROR: Missing columns. Got: {list(result.columns)}")
        sys.exit(1)

    print(f"  {len(result):,} tick rows retrieved.")

    # ── Connect Redis ───────────────────────────────────────────────────────────
    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    try:
        r.ping()
    except Exception as e:
        print(f"ERROR: Cannot reach Redis at {REDIS_HOST}:{REDIS_PORT} — {e}")
        sys.exit(1)
    print(f"  Redis connected ({REDIS_HOST}:{REDIS_PORT})")

    if args.flush:
        keys = r.keys("STOCK:*:bars")
        if keys:
            r.delete(*keys)
            print(f"  Flushed {len(keys)} existing STOCK:*:bars keys")

    # ── Build and write bars per symbol ────────────────────────────────────────
    symbols = sorted(s for s in result["symbol"].unique() if s and s != "UNKNOWN")
    written, skipped = 0, 0

    for symbol in symbols:
        sym_df = result[result["symbol"] == symbol]
        bars   = build_bars(sym_df)

        if len(bars) < 5:
            print(f"  SKIP {symbol}: only {len(bars)} bars")
            skipped += 1
            continue

        bars = bars.tail(BAR_CAP)

        # Pipeline: delete → LPUSH oldest-first → LTRIM
        # LPUSH prepends, so pushing oldest→newest puts newest at index 0
        pipe = r.pipeline()
        pipe.delete(f"STOCK:{symbol}:bars")
        for ts, row in bars.iterrows():
            bar_json = json.dumps({
                "open":   round(float(row["open"]),   2),
                "high":   round(float(row["high"]),   2),
                "low":    round(float(row["low"]),    2),
                "close":  round(float(row["close"]),  2),
                "volume": round(float(row["volume"]), 2),
                "ts":     ts.isoformat(),
            })
            pipe.lpush(f"STOCK:{symbol}:bars", bar_json)
        pipe.ltrim(f"STOCK:{symbol}:bars", 0, BAR_CAP - 1)
        pipe.execute()

        # Write current_bar from last completed bar (for dashboard + InferenceConsumer LTP)
        last_ts  = bars.index[-1]
        last_row = bars.iloc[-1]
        r.hset(f"STOCK:{symbol}:current_bar", mapping={
            "bar_open":     str(round(float(bars.iloc[0]["open"]), 2)),  # session open
            "bar_high":     str(round(float(last_row["high"]),  2)),
            "bar_low":      str(round(float(last_row["low"]),   2)),
            "bar_close":    str(round(float(last_row["close"]), 2)),
            "bar_vol":      str(round(float(last_row["volume"]), 2)),
            "bar_ts":       last_ts.isoformat(),
            "LTP":          str(round(float(last_row["close"]), 2)),
            "Open":         str(round(float(bars.iloc[0]["open"]), 2)),
            "High":         str(round(float(bars["high"].max()),   2)),
            "Low":          str(round(float(bars["low"].min()),    2)),
            "Close":        "0",
            "Volume":       "0",
            "Change":       "0",
            "Last_Updated": last_ts.isoformat(),
        })
        r.expire(f"STOCK:{symbol}:current_bar", 86400)  # 24h

        written += 1

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'-'*50}")
    print(f"Written : {written} symbols")
    print(f"Skipped : {skipped} symbols (< 5 bars)")

    sample = "RELIANCE"
    n = r.llen(f"STOCK:{sample}:bars")
    print(f"\nSTOCK:{sample}:bars -> {n} bars")
    if n > 0:
        oldest = json.loads(r.lindex(f"STOCK:{sample}:bars", -1))
        newest = json.loads(r.lindex(f"STOCK:{sample}:bars",  0))
        print(f"  Oldest: {oldest['ts']}  close={oldest['close']}")
        print(f"  Newest: {newest['ts']}  close={newest['close']}")
    print(f"\nInferenceConsumer will warm-start immediately tomorrow - no 60-min wait.")


if __name__ == "__main__":
    main()
