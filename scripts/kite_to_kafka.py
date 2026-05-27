"""
kite_to_kafka.py — Bridge: KiteConnect WebSocket → Kafka

Fully automated auth: logs in with user_id + password + TOTP (no browser).
Subscribes to all STOCKS from .env and forwards every live tick to the
stock-quotes Kafka topic. A consumer thread reads them back to confirm
end-to-end round-trip latency.

Usage:
    py scripts/kite_to_kafka.py
    py scripts/kite_to_kafka.py --bootstrap localhost:29092
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from kiteconnect import KiteConnect, KiteTicker

load_dotenv()

# ── Config from .env ──────────────────────────────────────────────────────────
API_KEY     = os.getenv("KITE_API_KEY", "")
API_SECRET  = os.getenv("KITE_API_SECRET", "")
USER_ID     = os.getenv("KITE_USER_ID", "")
PASSWORD    = os.getenv("KITE_PASSWORD", "")
TOTP_SECRET = os.getenv("KITE_TWO_FA", "").strip()
_STOCKS_DEFAULT = (
    "ADANIENT,ADANIPORTS,APOLLOHOSP,ASIANPAINT,AXISBANK,"
    "BAJAJ-AUTO,BAJAJFINSV,BAJFINANCE,BEL,BHARTIARTL,"
    "BRITANNIA,CIPLA,COALINDIA,DIVISLAB,DRREDDY,"
    "EICHERMOT,GRASIM,HCLTECH,HDFCBANK,HDFCLIFE,"
    "HEROMOTOCO,HINDALCO,HINDUNILVR,ICICIBANK,ITC,"
    "INDUSINDBK,INFY,JSWSTEEL,KOTAKBANK,LT,"
    "M&M,MARUTI,NESTLEIND,NTPC,ONGC,"
    "POWERGRID,RELIANCE,SBILIFE,SBIN,SUNPHARMA,"
    "TATACONSUM,TATASTEEL,TCS,TECHM,TITAN,"
    "ULTRACEMCO,WIPRO"
)
STOCKS        = [s.strip() for s in os.getenv("STOCKS", _STOCKS_DEFAULT).split(",")]
BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC         = os.getenv("KAFKA_TOPIC", "stock-quotes")
SESSION_CACHE = ".kite_session.json"
NIFTY50_TOKEN = 256265   # NSE:NIFTY 50 — stable across all KiteConnect accounts

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
WHITE   = "\033[97m"

# ── Shared state ──────────────────────────────────────────────────────────────
_lock      = threading.Lock()
_sent      = 0
_recv      = 0
_latencies: list[float] = []
_stop      = threading.Event()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ── Automated auth (no browser) ───────────────────────────────────────────────
def auto_login() -> str:
    """Login with user_id + password + TOTP — returns a fresh access_token."""
    print("Logging in to KiteConnect...")

    session = requests.Session()

    # Step 1: password login
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

    # Step 2: TOTP
    totp_code = pyotp.TOTP(TOTP_SECRET).now()
    r = session.post(
        "https://kite.zerodha.com/api/twofa",
        data={
            "user_id":    USER_ID,
            "request_id": request_id,
            "twofa_value": totp_code,
            "twofa_type": "totp",
        },
    )
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"TOTP failed: {body.get('message', body)}")
    print("  [2/3] TOTP OK")

    # Step 3: follow connect/login → Zerodha redirects to your app's callback URL.
    # Nothing is listening there, so requests raises ConnectionError — but the
    # failed request URL contains the request_token we need.
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

    # Step 4: exchange for access token
    kite = KiteConnect(api_key=API_KEY)
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    with open(SESSION_CACHE, "w") as f:
        json.dump({"access_token": access_token, "date": datetime.today().strftime("%Y-%m-%d")}, f)

    print("  Session cached for today.\n")
    return access_token


def get_access_token() -> str:
    """Return a valid access token, re-authenticating if the cache is stale."""
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


# ── Token resolution ──────────────────────────────────────────────────────────
def resolve_tokens(kite: KiteConnect) -> dict:
    """Return {instrument_token: symbol} for each stock in STOCKS."""
    qualified = [f"NSE:{s}" for s in STOCKS]
    quotes = kite.quote(qualified)
    token_map = {}
    for sym in STOCKS:
        key = f"NSE:{sym}"
        if key in quotes:
            token_map[quotes[key]["instrument_token"]] = sym
        else:
            print(f"  {YELLOW}[WARN]{RESET} {sym} not found — skipping")
    return token_map


# ── Consumer thread (round-trip verification) ─────────────────────────────────
def consumer_loop(bootstrap: str):
    global _recv

    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=[bootstrap],
            group_id="kite-bridge-verify-grp",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=2000,
            max_poll_records=100,
        )
    except Exception as e:
        print(f"{RED}[CONSUMER] Failed to connect: {e}{RESET}")
        _stop.set()
        return

    print(f"{MAGENTA}{BOLD}[CONSUMER]{RESET} Subscribed to '{TOPIC}' — watching for real ticks\n")

    while not _stop.is_set():
        try:
            for msg in consumer:
                if _stop.is_set():
                    break

                tick = msg.value
                sent_at = tick.get("_sent_at")
                rtt_ms  = (time.time() - sent_at) * 1000 if sent_at else None
                sym     = tick.get("symbol", "?")
                ltp     = tick.get("last_price", 0.0)
                vol     = tick.get("volume", 0)

                if rtt_ms is not None:
                    colour = GREEN if rtt_ms < 50 else (YELLOW if rtt_ms < 200 else RED)
                    lat_str = f"{colour}{rtt_ms:>6.1f}ms{RESET}"
                else:
                    lat_str = "    ---"

                with _lock:
                    _recv += 1
                    if rtt_ms is not None:
                        _latencies.append(rtt_ms)

                print(
                    f"{MAGENTA}[CONSUMER]{RESET} {ts()}  "
                    f"{WHITE}{BOLD}{sym:<16}{RESET}"
                    f"  LTP {ltp:>10.2f}"
                    f"  vol {vol:>10,}"
                    f"  RTT {lat_str}"
                    f"  {DIM}part {msg.partition}  off {msg.offset}{RESET}"
                )
        except Exception:
            pass  # consumer_timeout_ms fired — poll again

    consumer.close()
    print(f"{MAGENTA}[CONSUMER]{RESET} Stopped. Total received: {_recv}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KiteConnect → Kafka live bridge")
    parser.add_argument("--bootstrap", default=BOOTSTRAP,
                        help=f"Kafka bootstrap server (default: {BOOTSTRAP})")
    args = parser.parse_args()
    bootstrap = args.bootstrap

    required = [API_KEY, API_SECRET, USER_ID, PASSWORD, TOTP_SECRET]
    if not all(required):
        print("ERROR: KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_PASSWORD, KITE_TWO_FA must all be set in .env")
        sys.exit(1)

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )

    logging.getLogger("twisted").setLevel(logging.CRITICAL)

    access_token = get_access_token()

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(access_token)

    print(f"Resolving instrument tokens for {len(STOCKS)} stocks...")
    token_map = resolve_tokens(kite)
    if not token_map:
        print("No instruments resolved — check your STOCKS list in .env")
        sys.exit(1)
    token_map[NIFTY50_TOKEN] = "NIFTY50"
    print(f"  → {len(token_map)} instruments resolved (incl. NIFTY50)\n")

    # Connect Kafka producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            request_timeout_ms=10_000,
        )
        print(f"{CYAN}{BOLD}[PRODUCER]{RESET} Connected to Kafka at {bootstrap}")
    except Exception as e:
        print(f"{RED}[PRODUCER] Failed to connect to Kafka: {e}{RESET}")
        print("  → Is Docker running? Try: docker-compose up -d")
        sys.exit(1)

    # Start consumer verification thread
    consumer_thread = threading.Thread(target=consumer_loop, args=(bootstrap,), daemon=True)
    consumer_thread.start()

    # ── KiteTicker callbacks ──────────────────────────────────────────────────
    ticker = KiteTicker(API_KEY, access_token, reconnect=True)

    def on_ticks(ws, ticks):
        # on_ticks runs on the Twisted reactor thread — must never block.
        # Use async send with callbacks; .get() would freeze the WebSocket.
        for tick in ticks:
            token = tick["instrument_token"]
            sym   = token_map.get(token, "UNKNOWN")
            ltp   = tick.get("last_price", 0)
            vol   = tick.get("volume", 0)
            ohlc  = tick.get("ohlc", {})
            chg   = tick.get("change", 0)

            payload = {
                "symbol":           sym,
                "instrument_token": token,
                "last_price":       ltp,
                "volume":           vol,
                "ohlc": {
                    "open":  ohlc.get("open", 0),
                    "high":  ohlc.get("high", 0),
                    "low":   ohlc.get("low", 0),
                    "close": ohlc.get("close", 0),
                },
                "change":    chg,
                "timestamp": datetime.utcnow().isoformat(),
                "_sent_at":  time.time(),
            }

            def on_success(meta, sym=sym, ltp=ltp, chg=chg, vol=vol):
                global _sent
                with _lock:
                    _sent += 1
                arrow = f"{GREEN}+{RESET}" if chg >= 0 else f"{RED}-{RESET}"
                print(
                    f"{CYAN}[PRODUCER]{RESET} {ts()}  "
                    f"{WHITE}{BOLD}{sym:<16}{RESET}"
                    f"  LTP {ltp:>10.2f}  {arrow}{abs(chg):>5.2f}%"
                    f"  vol {vol:>10,}"
                    f"  {DIM}→ part {meta.partition}  off {meta.offset}{RESET}"
                )

            def on_error(exc, sym=sym):
                print(f"{RED}[PRODUCER] Kafka error for {sym}: {exc}{RESET}")

            producer.send(
                TOPIC,
                value=payload,
                key=str(token).encode(),
            ).add_callback(on_success).add_errback(on_error)

    def on_connect(ws, response):
        tokens = list(token_map.keys())
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, [t for t in tokens if t != NIFTY50_TOKEN])
        ws.set_mode(ws.MODE_QUOTE, [NIFTY50_TOKEN])
        print(f"\nSubscribed to {len(tokens)} instruments ({len(tokens)-1} stocks + NIFTY50). Live ticks incoming...\n")
        print(f"{'─'*80}\n")

    def on_close(ws, code, reason):
        print(f"\nWebSocket closed ({code}): {reason}")

    def on_error(ws, code, reason):
        print(f"\nWebSocket error ({code}): {reason}")

    ticker.on_ticks   = on_ticks
    ticker.on_connect = on_connect
    ticker.on_close   = on_close
    ticker.on_error   = on_error

    print(f"\n{BOLD}KiteConnect → Kafka Bridge{RESET}")
    print(f"  Kafka:   {bootstrap}  →  topic: {TOPIC}")
    print(f"  Stocks:  {len(token_map)} instruments")
    print(f"  Press {BOLD}Ctrl+C{RESET} to stop.\n")

    ticker.connect(threaded=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{BOLD}Shutting down...{RESET}")
        _stop.set()
        ticker.close()
        producer.flush()
        producer.close()

    consumer_thread.join(timeout=5)

    with _lock:
        lats = list(_latencies)
    if lats:
        avg = sum(lats) / len(lats)
        p50 = sorted(lats)[len(lats) // 2]
        p99 = sorted(lats)[int(len(lats) * 0.99)]
        print(f"\n{BOLD}Final RTT stats:{RESET}  avg {avg:.1f}ms  p50 {p50:.1f}ms  p99 {p99:.1f}ms")
    print(f"{BOLD}Done.{RESET}  Sent {_sent}  |  Received {_recv}")


if __name__ == "__main__":
    main()
