"""
Live stock data terminal display using KiteConnect WebSocket.

Auth flow (fully automated — no browser required):
  1. Logs in with KITE_USER_ID + KITE_PASSWORD from .env
  2. Submits TOTP using KITE_TWO_FA secret from .env
  3. Exchanges request_token for access_token
  4. Token is cached in .kite_session.json for the rest of the trading day

Run: py scripts/live_terminal.py
"""

import json
import os
import sys
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

load_dotenv()

API_KEY     = os.getenv("KITE_API_KEY", "")
API_SECRET  = os.getenv("KITE_API_SECRET", "")
USER_ID     = os.getenv("KITE_USER_ID", "")
PASSWORD    = os.getenv("KITE_PASSWORD", "")
TOTP_SECRET = os.getenv("KITE_TWO_FA", "").strip()
STOCKS      = [s.strip() for s in os.getenv("STOCKS", "RELIANCE,INFY,TCS,WIPRO,LT,ASIANPAINT").split(",")]
SESSION_CACHE = ".kite_session.json"


def auto_login() -> str:
    """Login with user_id + password + TOTP — returns a fresh access_token."""
    print("Logging in to KiteConnect...")

    session = requests.Session()

    r = session.post(
        "https://kite.zerodha.com/api/login",
        data={"user_id": USER_ID, "password": PASSWORD},
    )
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"Login failed: {body.get('message', body)}")
    request_id = body["data"]["request_id"]
    print("  [1/3] Password OK")

    totp_code = pyotp.TOTP(TOTP_SECRET).now()
    r = session.post(
        "https://kite.zerodha.com/api/twofa",
        data={
            "user_id":     USER_ID,
            "request_id":  request_id,
            "twofa_value": totp_code,
            "twofa_type":  "totp",
        },
    )
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"TOTP failed: {body.get('message', body)}")
    print("  [2/3] TOTP OK")

    login_url = f"https://kite.zerodha.com/connect/login?api_key={API_KEY}&v=3"
    request_token = None
    try:
        r = session.get(login_url, allow_redirects=True, timeout=10)
        request_token = parse_qs(urlparse(r.url).query).get("request_token", [None])[0]
    except requests.exceptions.ConnectionError as e:
        if e.request is not None:
            request_token = parse_qs(urlparse(str(e.request.url)).query).get("request_token", [None])[0]
    if not request_token:
        raise RuntimeError(
            "Could not extract request_token from the login redirect.\n"
            "Check that KITE_API_KEY in .env matches your KiteConnect app."
        )
    print("  [3/3] request_token captured")

    kite = KiteConnect(api_key=API_KEY)
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    with open(SESSION_CACHE, "w") as f:
        json.dump({"access_token": access_token, "date": datetime.today().strftime("%Y-%m-%d")}, f)

    print("  Session cached for today.\n")
    return access_token


def get_access_token() -> str:
    today = datetime.today().strftime("%Y-%m-%d")
    if os.path.exists(SESSION_CACHE):
        with open(SESSION_CACHE) as f:
            cached = json.load(f)
        if cached.get("date") == today:
            token = cached["access_token"]
            try:
                test = KiteConnect(api_key=API_KEY)
                test.set_access_token(token)
                test.profile()
                print(f"Cached session valid ({today}) — skipping login.")
                return token
            except Exception:
                print("Cached token rejected — re-authenticating...")
                os.remove(SESSION_CACHE)
    return auto_login()


def resolve_tokens(kite: KiteConnect, symbols: list) -> dict:
    """Return {instrument_token: symbol} for each stock."""
    qualified = [f"NSE:{s}" for s in symbols]
    quotes = kite.quote(qualified)
    token_map = {}
    for sym in symbols:
        key = f"NSE:{sym}"
        if key in quotes:
            token_map[quotes[key]["instrument_token"]] = sym
        else:
            print(f"  [WARN] {sym} not found — skipping")
    return token_map


def print_tick(symbol: str, tick: dict):
    ts = datetime.now().strftime("%H:%M:%S")
    ltp        = tick.get("last_price", 0)
    volume     = tick.get("volume", 0)
    ohlc       = tick.get("ohlc", {})
    open_      = ohlc.get("open", 0)
    high       = ohlc.get("high", 0)
    low        = ohlc.get("low", 0)
    close      = ohlc.get("close", 0)
    change_pct = ((ltp - close) / close * 100) if close else 0
    arrow      = "+" if change_pct >= 0 else "-"

    print(
        f"[{ts}] {symbol:<14} "
        f"LTP: {ltp:>10.2f}  {arrow}{abs(change_pct):>5.2f}%  "
        f"O:{open_:.2f} H:{high:.2f} L:{low:.2f} C:{close:.2f}  "
        f"Vol:{volume:>10,}"
    )


def main():
    if not all([API_KEY, API_SECRET]):
        print("ERROR: KITE_API_KEY and KITE_API_SECRET must be set in .env")
        sys.exit(1)

    access_token = get_access_token()

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(access_token)

    print(f"\nResolving instrument tokens for: {', '.join(STOCKS)}")
    token_map = resolve_tokens(kite, STOCKS)  # {token: symbol}

    # Print current snapshot — single batched LTP call
    print("\n--- Current Quotes ---")
    qualified = [f"NSE:{s}" for s in token_map.values()]
    ltps = kite.ltp(qualified)
    for sym in token_map.values():
        key = f"NSE:{sym}"
        ltp = ltps.get(key, {}).get("last_price", 0)
        print(f"  {sym:<16} {ltp:.2f}")

    print("\n--- Live Feed (Ctrl+C to stop) ---\n")

    # Silence Twisted's own error logger — our callbacks handle messaging
    import logging
    logging.getLogger("twisted").setLevel(logging.CRITICAL)

    ticker = KiteTicker(API_KEY, access_token, reconnect=True)

    def on_ticks(ws, ticks):
        for tick in ticks:
            symbol = token_map.get(tick["instrument_token"], "UNKNOWN")
            print_tick(symbol, tick)

    def on_connect(ws, response):
        tokens = list(token_map.keys())
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        print(f"Subscribed to {len(tokens)} instruments.\n")

    def on_close(ws, code, reason):
        print(f"\nWebSocket closed ({code}): {reason}")

    def on_error(ws, code, reason):
        print(f"\nWebSocket error ({code}): {reason}")

    ticker.on_ticks   = on_ticks
    ticker.on_connect = on_connect
    ticker.on_close   = on_close
    ticker.on_error   = on_error

    import time
    ticker.connect(threaded=True)  # reactor runs in background thread

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        ticker.close()


if __name__ == "__main__":
    main()
