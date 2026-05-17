"""
Bulk-fetches up to 3 years of 1-minute OHLCV candles for Nifty 50 stocks
from KiteConnect historical data API. Saves one Parquet file per stock.

Resumable: if a stock's file already exists, continues from the last saved date.
Rate-limited to ≤3 requests/second (KiteConnect enforced limit).

Usage:
    py scripts/fetch_historical_data.py                   # full 3-year fetch, all stocks
    py scripts/fetch_historical_data.py --days 90         # last 90 days only
    py scripts/fetch_historical_data.py --symbol RELIANCE # single stock

Output: data/historical/<SYMBOL>.parquet
Columns: timestamp (UTC, tz-aware), open, high, low, close, volume, symbol
"""

import argparse
import json
import os
import sys
import time
import threading
import webbrowser
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("KITE_API_KEY", "")
API_SECRET   = os.getenv("KITE_API_SECRET", "")
SESSION_FILE = ".kite_session.json"
DATA_DIR     = "data/historical"
CHUNK_DAYS   = 59       # KiteConnect limit is 60 days per request; 59 is safe
RATE_SLEEP   = 0.35     # seconds between requests; keeps us under 3 req/sec
HISTORY_DAYS = 365 * 3  # how far back to go on a fresh fetch

# Current Nifty 50 constituents (KiteConnect / NSE symbols).
# Update this list if the index composition changes.
NIFTY50_SYMBOLS = [
    "ADANIENT",   "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL",        "BHARTIARTL",
    "BRITANNIA",  "CIPLA",      "COALINDIA",  "DIVISLAB",   "DRREDDY",
    "EICHERMOT",  "GRASIM",     "HCLTECH",    "HDFCBANK",   "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO",   "HINDUNILVR", "ICICIBANK",  "ITC",
    "INDUSINDBK", "INFY",       "JSWSTEEL",   "KOTAKBANK",  "LT",
    "LTIM",       "M&M",        "MARUTI",     "NTPC",       "NESTLEIND",
    "ONGC",       "POWERGRID",  "RELIANCE",   "SBILIFE",    "SBIN",
    "SUNPHARMA",  "TCS",        "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM",      "TITAN",      "ULTRACEMCO", "WIPRO",      "ZOMATO",
]

# ── Auth (reuses .kite_session.json cache from live_terminal.py) ──────────────
_captured_token: dict = {}


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/kite/callback":
            token = parse_qs(parsed.query).get("request_token", [None])[0]
            if token:
                _captured_token["value"] = token
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>Authorized! Return to the terminal.</h2>")
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def _browser_login() -> str:
    server = HTTPServer(("localhost", 8000), _CallbackHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    login_url = KiteConnect(api_key=API_KEY).login_url()
    print("Opening KiteConnect login in your browser...")
    print("  1. Log in with Zerodha credentials + TOTP")
    print("  2. Click Authorize — token is captured automatically\n")
    webbrowser.open(login_url)

    print("Waiting for authorization", end="", flush=True)
    while "value" not in _captured_token:
        time.sleep(0.5)
        print(".", end="", flush=True)
    server.shutdown()
    print(" done.\n")

    kite = KiteConnect(api_key=API_KEY)
    data = kite.generate_session(_captured_token["value"], api_secret=API_SECRET)
    access_token = data["access_token"]

    with open(SESSION_FILE, "w") as f:
        json.dump({"access_token": access_token, "date": date.today().isoformat()}, f)

    print("Session cached.\n")
    return access_token


def get_access_token() -> str:
    today = date.today().isoformat()
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            cached = json.load(f)
        if cached.get("date") == today:
            print(f"Using cached session ({today}).")
            return cached["access_token"]
    return _browser_login()


# ── Instrument token resolution ───────────────────────────────────────────────

def build_token_map(kite: KiteConnect, symbols: list[str]) -> dict[str, int]:
    """
    Returns {symbol: instrument_token} for each symbol found on NSE.
    Downloads the full NSE instrument dump once; no per-symbol requests.
    """
    print("Downloading NSE instruments list...", flush=True)
    instruments = kite.instruments("NSE")  # list of dicts
    nse_map = {inst["tradingsymbol"]: inst["instrument_token"] for inst in instruments}

    token_map = {}
    for sym in symbols:
        token = nse_map.get(sym)
        if token:
            token_map[sym] = token
        else:
            print(f"  [WARN] {sym} not found in NSE instruments — skipping")

    print(f"Resolved {len(token_map)}/{len(symbols)} symbols.\n")
    return token_map


# ── Per-stock fetch ───────────────────────────────────────────────────────────

def _parquet_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}.parquet")


def _last_saved_date(symbol: str) -> date | None:
    path = _parquet_path(symbol)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=["timestamp"])
    if df.empty:
        return None
    max_ts = df["timestamp"].max()
    if pd.isna(max_ts):
        return None
    return pd.Timestamp(max_ts).date()


def _fetch_chunk(
    kite: KiteConnect,
    token: int,
    from_dt: date,
    to_dt: date,
) -> pd.DataFrame:
    """Fetch one ≤60-day chunk of 1-minute candles. Returns empty DataFrame on error."""
    try:
        candles = kite.historical_data(
            instrument_token=token,
            from_date=datetime.combine(from_dt, datetime.min.time()),
            to_date=datetime.combine(to_dt, datetime.max.time().replace(microsecond=0)),
            interval="minute",
        )
    except Exception as exc:
        print(f"      [ERROR] API call failed: {exc}")
        return pd.DataFrame()

    if not candles:
        return pd.DataFrame()

    # kiteconnect returns dicts with key "date", not "timestamp"
    df = pd.DataFrame(candles)
    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_and_save(
    kite: KiteConnect,
    symbol: str,
    token: int,
    start_date: date,
    end_date: date,
) -> int:
    """
    Fetches candles for one stock from start_date to end_date in 59-day chunks.
    Appends to existing Parquet file if present. Returns total new candles written.
    """
    last = _last_saved_date(symbol)
    if last is not None:
        # Resume from the day after the last saved candle
        actual_start = last + timedelta(days=1)
        if actual_start >= end_date:
            print(f"  {symbol:<14} already up to date (last: {last}).")
            return 0
        print(f"  {symbol:<14} resuming from {actual_start} (last saved: {last})")
    else:
        actual_start = start_date
        print(f"  {symbol:<14} fetching {actual_start} → {end_date}")

    chunks = []
    cursor = actual_start
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end_date)
        chunk_df = _fetch_chunk(kite, token, cursor, chunk_end)
        if not chunk_df.empty:
            chunks.append(chunk_df)
        print(
            f"    {cursor} → {chunk_end}  "
            f"({len(chunk_df)} candles){'  [empty]' if chunk_df.empty else ''}",
            flush=True,
        )
        cursor = chunk_end + timedelta(days=1)
        time.sleep(RATE_SLEEP)

    if not chunks:
        print(f"  {symbol:<14} no new data.\n")
        return 0

    new_df = pd.concat(chunks, ignore_index=True)
    new_df["symbol"] = symbol

    path = _parquet_path(symbol)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["timestamp", "symbol"], inplace=True)
        combined.sort_values("timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        combined.to_parquet(path, index=False)
        written = len(combined) - len(existing)
    else:
        new_df.sort_values("timestamp", inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        new_df.to_parquet(path, index=False)
        written = len(new_df)

    print(f"  {symbol:<14} +{written:,} new candles saved → {path}\n")
    return written


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch Nifty 50 historical 1-min candles.")
    parser.add_argument("--days",   type=int, default=HISTORY_DAYS,
                        help=f"How many days of history to fetch (default: {HISTORY_DAYS})")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Fetch a single symbol instead of all Nifty 50")
    args = parser.parse_args()

    if not API_KEY or not API_SECRET:
        print("ERROR: KITE_API_KEY and KITE_API_SECRET must be set in .env")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    access_token = get_access_token()
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(access_token)

    symbols = [args.symbol.upper()] if args.symbol else NIFTY50_SYMBOLS
    token_map = build_token_map(kite, symbols)

    end_date   = date.today() - timedelta(days=1)  # up to yesterday
    start_date = end_date - timedelta(days=args.days)

    print(f"Fetching {start_date} → {end_date}  ({args.days} days)\n")
    print("=" * 60)

    total_candles = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        token = token_map.get(symbol)
        if token is None:
            failed.append(symbol)
            continue

        print(f"[{i}/{len(symbols)}] {symbol}")
        try:
            written = fetch_and_save(kite, symbol, token, start_date, end_date)
            total_candles += written
        except Exception as exc:
            print(f"  [ERROR] {symbol}: {exc}\n")
            failed.append(symbol)

    print("=" * 60)
    print(f"Done. Total new candles written: {total_candles:,}")
    if failed:
        print(f"Failed / skipped: {', '.join(failed)}")
    print(f"\nData saved to: {os.path.abspath(DATA_DIR)}/")


if __name__ == "__main__":
    main()
