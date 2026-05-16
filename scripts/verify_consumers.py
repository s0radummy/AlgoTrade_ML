"""
verify_consumers.py — Verify Redis + InfluxDB consumers against the Kafka feed.

Runs two consumers in parallel threads:
  - Redis consumer:   stock-quotes → STOCK:<symbol> hashes in Redis
  - InfluxDB consumer: stock-quotes → batched points in InfluxDB

Every 30s prints a verification summary by reading back from both stores.

Run this alongside kite_to_kafka.py (live) or kafka_live_view.py (mock data).
Uses auto_offset_reset="earliest" so it also picks up any messages already
sitting in the topic from a previous run.

Usage:
    py scripts/verify_consumers.py
"""

import json
import os
import sys
import threading
import time
from datetime import datetime

import redis as redis_lib
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from kafka import KafkaConsumer
from kafka.errors import KafkaError

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BOOTSTRAP      = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC          = os.getenv("KAFKA_TOPIC", "stock-quotes")
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
INFLUX_HOST    = os.getenv("INFLUXDB_HOST", "localhost")
INFLUX_PORT    = int(os.getenv("INFLUXDB_PORT", "8086"))
INFLUX_TOKEN   = os.getenv("INFLUXDB_TOKEN", "algotrade-dev-token-2024")
INFLUX_ORG     = os.getenv("INFLUXDB_ORG", "algotrade")
INFLUX_BUCKET  = os.getenv("INFLUXDB_BUCKET", "stocks")

INFLUX_BATCH_SIZE    = 50
INFLUX_FLUSH_INTERVAL = 5  # seconds

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
BLUE    = "\033[34m"
WHITE   = "\033[97m"

# ── Shared state ──────────────────────────────────────────────────────────────
_lock        = threading.Lock()
_redis_count = 0
_influx_count = 0
_stop        = threading.Event()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ── Redis consumer ────────────────────────────────────────────────────────────
def redis_consumer_loop():
    global _redis_count

    # Connect to Redis
    try:
        r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        print(f"{GREEN}{BOLD}[REDIS]{RESET} Connected to {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"{RED}[REDIS] Failed to connect: {e}{RESET}")
        _stop.set()
        return

    # Connect to Kafka — group_id=None means no offset tracking, always reads
    # from auto_offset_reset="earliest" on every run (no stale committed offsets)
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=[BOOTSTRAP],
            group_id=None,
            auto_offset_reset="earliest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=3000,
        )
        print(f"{GREEN}{BOLD}[REDIS]{RESET} Kafka consumer ready (no group — reads all from start)")
    except Exception as e:
        print(f"{RED}[REDIS] Kafka connect failed: {e}{RESET}")
        _stop.set()
        return

    while not _stop.is_set():
        try:
            for msg in consumer:
                if _stop.is_set():
                    break

                tick = msg.value
                sym  = tick.get("symbol", "UNKNOWN")
                ltp  = tick.get("last_price", 0)
                ohlc = tick.get("ohlc", {})

                state = {
                    "LTP":          str(ltp),
                    "Open":         str(ohlc.get("open", 0)),
                    "High":         str(ohlc.get("high", 0)),
                    "Low":          str(ohlc.get("low", 0)),
                    "Close":        str(ohlc.get("close", 0)),
                    "Volume":       str(tick.get("volume", 0)),
                    "Change":       str(tick.get("change", 0)),
                    "Last_Updated": datetime.utcnow().isoformat(),
                }

                try:
                    r.hset(f"STOCK:{sym}", mapping=state)
                    r.expire(f"STOCK:{sym}", 3600)

                    with _lock:
                        _redis_count += 1

                    print(
                        f"{GREEN}[REDIS]{RESET} {ts()}  "
                        f"{WHITE}{BOLD}{sym:<16}{RESET}"
                        f"  LTP {ltp:>10.2f}"
                        f"  {DIM}STOCK:{sym}{RESET}"
                    )
                except Exception as e:
                    print(f"{RED}[REDIS] Write error for {sym}: {e}{RESET}")

        except Exception:
            pass  # consumer_timeout_ms fired — loop again

    consumer.close()
    print(f"{GREEN}[REDIS]{RESET} Stopped. Total written: {_redis_count}")


# ── InfluxDB consumer ─────────────────────────────────────────────────────────
def influx_consumer_loop():
    global _influx_count

    # Connect to InfluxDB
    try:
        client = InfluxDBClient(
            url=f"http://{INFLUX_HOST}:{INFLUX_PORT}",
            token=INFLUX_TOKEN,
            org=INFLUX_ORG,
        )
        health = client.health()
        if health.status != "pass":
            raise RuntimeError(f"InfluxDB unhealthy: {health.status}")
        write_api = client.write_api(write_options=SYNCHRONOUS)
        query_api = client.query_api()
        print(f"{BLUE}{BOLD}[INFLUX]{RESET} Connected to {INFLUX_HOST}:{INFLUX_PORT}  bucket: {INFLUX_BUCKET}")
    except Exception as e:
        print(f"{RED}[INFLUX] Failed to connect: {e}{RESET}")
        _stop.set()
        return

    # Connect to Kafka — group_id=None means no offset tracking, always reads
    # from auto_offset_reset="earliest" on every run (no stale committed offsets)
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=[BOOTSTRAP],
            group_id=None,
            auto_offset_reset="earliest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=3000,
        )
        print(f"{BLUE}{BOLD}[INFLUX]{RESET} Kafka consumer ready (no group — reads all from start)")
    except Exception as e:
        print(f"{RED}[INFLUX] Kafka connect failed: {e}{RESET}")
        _stop.set()
        return

    batch: list[Point] = []
    last_flush = time.time()

    def flush():
        nonlocal batch, last_flush
        global _influx_count
        if not batch:
            return
        try:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=batch)
            with _lock:
                _influx_count += len(batch)
            print(
                f"{BLUE}[INFLUX]{RESET} {ts()}  "
                f"Flushed {BOLD}{len(batch)}{RESET} points  "
                f"{DIM}(total: {_influx_count}){RESET}"
            )
            batch = []
            last_flush = time.time()
        except Exception as e:
            print(f"{RED}[INFLUX] Write error: {e}{RESET}")

    while not _stop.is_set():
        try:
            for msg in consumer:
                if _stop.is_set():
                    break

                tick = msg.value
                sym  = tick.get("symbol", "UNKNOWN")
                ohlc = tick.get("ohlc", {})

                point = (
                    Point("ticks")
                    .tag("symbol", sym)
                    .tag("instrument_token", str(tick.get("instrument_token", 0)))
                    .field("last_price", float(tick.get("last_price", 0)))
                    .field("volume",     int(tick.get("volume", 0)))
                    .field("change",     float(tick.get("change", 0)))
                    .field("open",       float(ohlc.get("open", 0)))
                    .field("high",       float(ohlc.get("high", 0)))
                    .field("low",        float(ohlc.get("low", 0)))
                    .field("close",      float(ohlc.get("close", 0)))
                )
                batch.append(point)

                if len(batch) >= INFLUX_BATCH_SIZE or (time.time() - last_flush) >= INFLUX_FLUSH_INTERVAL:
                    flush()

        except Exception:
            # consumer_timeout_ms fired — flush whatever we have
            flush()

    flush()
    consumer.close()
    write_api.close()
    client.close()
    print(f"{BLUE}[INFLUX]{RESET} Stopped. Total written: {_influx_count}")


# ── Verification loop ─────────────────────────────────────────────────────────
def verify_loop():
    """Every 30s read back from Redis and InfluxDB to confirm data is there."""
    time.sleep(15)  # give consumers time to write something first

    try:
        r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    except Exception:
        return

    try:
        client = InfluxDBClient(url=f"http://{INFLUX_HOST}:{INFLUX_PORT}", token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
    except Exception:
        query_api = None

    while not _stop.is_set():
        print(f"\n{YELLOW}{'─'*70}{RESET}")
        print(f"{YELLOW}{BOLD}[VERIFY]{RESET} {ts()}")

        # Redis: show a sample of STOCK:* keys
        try:
            keys = r.keys("STOCK:*")
            print(f"{GREEN}[REDIS]{RESET}   {len(keys)} STOCK:* keys present")
            for key in sorted(keys)[:3]:
                ltp = r.hget(key, "LTP")
                updated = r.hget(key, "Last_Updated")
                print(f"  {key:<24} LTP={ltp}  updated={updated}")
            if len(keys) > 3:
                print(f"  ... and {len(keys) - 3} more")
        except Exception as e:
            print(f"{RED}[REDIS]   Query error: {e}{RESET}")

        # InfluxDB: count records written in last hour
        if query_api:
            try:
                flux = f'''
                from(bucket: "{INFLUX_BUCKET}")
                  |> range(start: -1h)
                  |> filter(fn: (r) => r._measurement == "ticks")
                  |> filter(fn: (r) => r._field == "last_price")
                  |> count()
                  |> sum()
                '''
                result = query_api.query(flux)
                total = sum(r.get_value() for table in result for r in table.records)
                print(f"{BLUE}[INFLUX]{RESET}  {total} tick records in InfluxDB (last 1h)")
            except Exception as e:
                print(f"{RED}[INFLUX]  Query error: {e}{RESET}")

        with _lock:
            print(f"{DIM}  Redis writes: {_redis_count}  |  InfluxDB writes: {_influx_count}{RESET}")
        print(f"{YELLOW}{'─'*70}{RESET}\n")

        for _ in range(30):
            if _stop.is_set():
                break
            time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )

    print(f"\n{BOLD}Consumer Verification{RESET}")
    print(f"  Kafka:   {BOOTSTRAP}  →  topic: {TOPIC}")
    print(f"  Redis:   {REDIS_HOST}:{REDIS_PORT}")
    print(f"  InfluxDB: {INFLUX_HOST}:{INFLUX_PORT}  bucket: {INFLUX_BUCKET}")
    print(f"  Offset:  earliest (picks up existing + new messages)")
    print(f"  Press {BOLD}Ctrl+C{RESET} to stop.\n")

    threads = [
        threading.Thread(target=redis_consumer_loop,  daemon=True, name="redis"),
        threading.Thread(target=influx_consumer_loop, daemon=True, name="influx"),
        threading.Thread(target=verify_loop,          daemon=True, name="verify"),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n{BOLD}Shutting down...{RESET}")
        _stop.set()

    for t in threads:
        t.join(timeout=8)

    print(f"\n{BOLD}Final counts:{RESET}  Redis {_redis_count}  |  InfluxDB {_influx_count}")


if __name__ == "__main__":
    main()
