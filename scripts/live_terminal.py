"""
Live stock data terminal display using KiteConnect WebSocket.

Auth flow (fully automatic after first browser login):
  1. Starts a local server on http://localhost:8000 to catch the callback
  2. Opens the KiteConnect login page in your browser
  3. You log in once (user ID + password + TOTP from your app)
  4. Click Authorize — the callback is captured automatically
  5. Access token is cached for the rest of the trading day

Run: py scripts/live_terminal.py
"""

import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

load_dotenv()

API_KEY  = os.getenv("KITE_API_KEY", "")
API_SECRET = os.getenv("KITE_API_SECRET", "")
STOCKS   = [s.strip() for s in os.getenv("STOCKS", "RELIANCE,INFY,TCS,WIPRO,LT,ASIANPAINT").split(",")]

SESSION_CACHE  = ".kite_session.json"
CALLBACK_PORT  = 8000
CALLBACK_PATH  = "/kite/callback"

# Shared state between HTTP handler and main thread
_captured_token: dict = {}


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == CALLBACK_PATH:
            params = parse_qs(parsed.query)
            token = params.get("request_token", [None])[0]
            if token:
                _captured_token["value"] = token
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>Authorized! You can close this tab and return to the terminal.</h2>")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing request_token")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress server access logs


def browser_login() -> str:
    """Open browser login, capture request_token via local callback server."""
    server = HTTPServer(("localhost", CALLBACK_PORT), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    login_url = KiteConnect(api_key=API_KEY).login_url()
    print(f"\nOpening KiteConnect login in your browser...")
    print(f"  1. Log in with your Zerodha credentials + TOTP")
    print(f"  2. Click Authorize")
    print(f"  3. Return here — the token is captured automatically\n")
    webbrowser.open(login_url)

    print("Waiting for authorization...", end="", flush=True)
    while "value" not in _captured_token:
        import time; time.sleep(0.5)
        print(".", end="", flush=True)
    server.shutdown()
    print(" done.\n")

    request_token = _captured_token["value"]
    kite = KiteConnect(api_key=API_KEY)
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    with open(SESSION_CACHE, "w") as f:
        json.dump({"access_token": access_token, "date": datetime.today().strftime("%Y-%m-%d")}, f)

    print("Session cached for today.")
    return access_token


def get_access_token() -> str:
    today = datetime.today().strftime("%Y-%m-%d")
    if os.path.exists(SESSION_CACHE):
        with open(SESSION_CACHE) as f:
            cached = json.load(f)
        if cached.get("date") == today:
            print(f"Using cached session ({today}).")
            return cached["access_token"]
    return browser_login()


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
