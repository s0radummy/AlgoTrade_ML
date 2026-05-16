"""
Live Kafka round-trip viewer — producer and consumer in one terminal.

Shows every message the producer sends and every message the consumer
receives, with round-trip latency per message.

Usage:
    python scripts/kafka_live_view.py
    python scripts/kafka_live_view.py --bootstrap localhost:29092
    python scripts/kafka_live_view.py --interval 0.3   # ticks faster
    python scripts/kafka_live_view.py --stocks RELIANCE,TCS
"""

import argparse
import json
import random
import sys
import threading
import time
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
MAGENTA= "\033[35m"
WHITE  = "\033[97m"

# ── Defaults ──────────────────────────────────────────────────────────────────
BOOTSTRAP_DEFAULT = "localhost:29092"
TOPIC             = "stock-quotes"
STOCKS            = {
    "RELIANCE":   2500.0,
    "INFY":       1800.0,
    "TCS":        4200.0,
    "WIPRO":       550.0,
    "LT":         2100.0,
    "ASIANPAINT": 3100.0,
}

# Shared state
_lock           = threading.Lock()
_sent_count     = 0
_recv_count     = 0
_latencies_ms   = []
_stop_event     = threading.Event()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def fmt_price(p: float) -> str:
    return f"{p:>10.2f}"


def fmt_latency(ms: float) -> str:
    colour = GREEN if ms < 50 else YELLOW if ms < 200 else RED
    return f"{colour}{ms:>6.1f}ms{RESET}"


# ── Producer thread ───────────────────────────────────────────────────────────
def producer_loop(bootstrap: str, interval: float, stocks: dict):
    global _sent_count

    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            request_timeout_ms=10000,
        )
    except Exception as e:
        print(f"{RED}[PRODUCER] Failed to connect: {e}{RESET}")
        _stop_event.set()
        return

    print(f"{CYAN}{BOLD}[PRODUCER]{RESET} Connected to {bootstrap}")

    prices = dict(stocks)

    while not _stop_event.is_set():
        for symbol, base in stocks.items():
            if _stop_event.is_set():
                break

            change_pct = random.uniform(-0.015, 0.015)
            prices[symbol] = prices[symbol] * (1 + change_pct)
            ltp    = prices[symbol]
            volume = random.randint(500, 80_000)
            ohlc   = {
                "open":  base,
                "high":  ltp * 1.005,
                "low":   ltp * 0.995,
                "close": ltp,
            }
            token = abs(hash(symbol)) % 1_000_000

            payload = {
                "symbol":            symbol,
                "instrument_token":  token,
                "last_price":        ltp,
                "volume":            volume,
                "ohlc":              ohlc,
                "change":            change_pct,
                "_sent_at":          time.time(),           # for RTT calc
                "timestamp":         datetime.utcnow().isoformat(),
            }

            try:
                meta = producer.send(
                    TOPIC, value=payload, key=symbol.encode()
                ).get(timeout=5)

                with _lock:
                    _sent_count += 1

                arrow = (f"{GREEN}+{RESET}" if change_pct >= 0 else f"{RED}-{RESET}")
                print(
                    f"{CYAN}[PRODUCER]{RESET} {ts()}  "
                    f"{WHITE}{BOLD}{symbol:<12}{RESET}"
                    f"  LTP {BOLD}{fmt_price(ltp)}{RESET}"
                    f"  {arrow}{abs(change_pct)*100:4.2f}%"
                    f"  vol {volume:>7,}"
                    f"  {DIM}→ partition {meta.partition}  offset {meta.offset}{RESET}"
                )

            except KafkaError as e:
                print(f"{RED}[PRODUCER] Send error: {e}{RESET}")

        time.sleep(interval)

    producer.flush()
    producer.close()
    print(f"{CYAN}[PRODUCER]{RESET} Stopped.  Total sent: {_sent_count}")


# ── Consumer thread ───────────────────────────────────────────────────────────
def consumer_loop(bootstrap: str):
    global _recv_count

    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=[bootstrap],
            group_id="live-view-grp",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=2000,
            max_poll_records=50,
        )
    except Exception as e:
        print(f"{MAGENTA}[CONSUMER]{RESET} {RED}Failed to connect: {e}{RESET}")
        _stop_event.set()
        return

    print(f"{MAGENTA}{BOLD}[CONSUMER]{RESET} Subscribed to '{TOPIC}'  (group: live-view-grp)")

    while not _stop_event.is_set():
        try:
            for msg in consumer:
                if _stop_event.is_set():
                    break

                tick = msg.value
                sent_at = tick.get("_sent_at")
                rtt_ms  = (time.time() - sent_at) * 1000 if sent_at else None
                symbol  = tick.get("symbol", "?")
                ltp     = tick.get("last_price", 0.0)
                volume  = tick.get("volume", 0)

                with _lock:
                    _recv_count += 1
                    if rtt_ms is not None:
                        _latencies_ms.append(rtt_ms)

                lat_str = fmt_latency(rtt_ms) if rtt_ms is not None else "  ---  "

                print(
                    f"{MAGENTA}[CONSUMER]{RESET} {ts()}  "
                    f"{WHITE}{BOLD}{symbol:<12}{RESET}"
                    f"  LTP {BOLD}{fmt_price(ltp)}{RESET}"
                    f"  vol {volume:>7,}"
                    f"  RTT {lat_str}"
                    f"  {DIM}partition {msg.partition}  offset {msg.offset}{RESET}"
                )

        except Exception:
            # consumer_timeout_ms fired — just poll again
            pass

    consumer.close()
    print(f"{MAGENTA}[CONSUMER]{RESET} Stopped.  Total received: {_recv_count}")


# ── Stats ticker ─────────────────────────────────────────────────────────────
def stats_loop(interval_s: int = 10):
    while not _stop_event.is_set():
        time.sleep(interval_s)
        if _stop_event.is_set():
            break
        with _lock:
            sent = _sent_count
            recv = _recv_count
            lats = list(_latencies_ms)

        if lats:
            avg  = sum(lats) / len(lats)
            p99  = sorted(lats)[int(len(lats) * 0.99)]
            lat_str = f"avg {avg:.1f}ms  p99 {p99:.1f}ms"
        else:
            lat_str = "no latency data yet"

        lag = sent - recv
        print(
            f"\n{YELLOW}{'─'*60}{RESET}\n"
            f"{YELLOW}[STATS]{RESET}  sent {sent}  recv {recv}  "
            f"lag {lag}  |  {lat_str}\n"
            f"{YELLOW}{'─'*60}{RESET}\n"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default=BOOTSTRAP_DEFAULT)
    parser.add_argument("--interval",  type=float, default=0.5,
                        help="Seconds between producer batches (default 0.5)")
    parser.add_argument("--stocks",    default=",".join(STOCKS.keys()),
                        help="Comma-separated stock symbols")
    args = parser.parse_args()

    stocks = {s.strip(): STOCKS.get(s.strip(), 1000.0)
              for s in args.stocks.split(",") if s.strip()}

    # Windows: enable ANSI codes
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    print(
        f"\n{BOLD}Kafka Live View{RESET}  |  broker: {args.bootstrap}"
        f"  |  topic: {TOPIC}  |  interval: {args.interval}s\n"
        f"Stocks: {', '.join(stocks.keys())}\n"
        f"Press {BOLD}Ctrl+C{RESET} to stop.\n"
        f"{'─'*70}\n"
    )

    threads = [
        threading.Thread(target=producer_loop, args=(args.bootstrap, args.interval, stocks), daemon=True),
        threading.Thread(target=consumer_loop, args=(args.bootstrap,), daemon=True),
        threading.Thread(target=stats_loop, args=(10,), daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print(f"\n{BOLD}Shutting down...{RESET}")
        _stop_event.set()

    for t in threads:
        t.join(timeout=5)

    with _lock:
        lats = list(_latencies_ms)
    if lats:
        avg = sum(lats) / len(lats)
        p50 = sorted(lats)[len(lats) // 2]
        p99 = sorted(lats)[int(len(lats) * 0.99)]
        print(f"\n{BOLD}Final latency stats:{RESET}  avg {avg:.1f}ms  p50 {p50:.1f}ms  p99 {p99:.1f}ms")

    print(f"{BOLD}Done.{RESET}  Sent {_sent_count}  |  Received {_recv_count}")


if __name__ == "__main__":
    main()
