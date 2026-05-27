import json
import signal
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config.settings import settings
from src.core.redis_client import redis_client
from src.data.tick_validator import tick_validator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BAR_CAP = 300  # max completed bars stored per symbol in Redis


class BarAccumulator:
    """
    Converts a KiteConnect tick stream into completed 1-minute OHLCV bars.

    KiteConnect sends cumulative daily volume, so bar volume is the delta between
    the cumulative volume at bar-open and the current tick.
    """

    def __init__(self):
        self._bar: Optional[Dict] = None
        self._bar_key: Optional[Tuple] = None  # (year, month, day, hour, minute)
        self._bar_start_vol: float = 0.0

    def update(self, tick: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Ingest one tick.
        Returns (bar_completed, completed_bar): bar_completed is True when a minute
        boundary is crossed and completed_bar holds the just-finished bar dict.
        """
        ts_str = tick.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            ts = datetime.now(tz=timezone.utc)

        bar_key = (ts.year, ts.month, ts.day, ts.hour, ts.minute)
        ltp     = float(tick.get("last_price", 0.0))
        cum_vol = float(tick.get("volume_traded", tick.get("volume", 0.0)))

        if bar_key != self._bar_key:
            completed_bar = dict(self._bar) if self._bar is not None else None

            bar_ts = ts.replace(second=0, microsecond=0).isoformat()
            self._bar = {
                "open":   ltp,
                "high":   ltp,
                "low":    ltp,
                "close":  ltp,
                "volume": 0.0,
                "ts":     bar_ts,
            }
            self._bar_key       = bar_key
            self._bar_start_vol = cum_vol
            return (completed_bar is not None, completed_bar)
        else:
            if ltp > 0:
                self._bar["high"]  = max(self._bar["high"], ltp)
                self._bar["low"]   = min(self._bar["low"],  ltp)
                self._bar["close"] = ltp
            self._bar["volume"] = max(0.0, cum_vol - self._bar_start_vol)
            return (False, None)

    def current_bar(self) -> Optional[Dict]:
        return self._bar


class VizConsumer:
    """
    Single bar-builder for the entire system.

    Per tick  → writes STOCK:{sym}:current_bar (running OHLCV + day-level KiteConnect data)
    Per bar   → LPUSH STOCK:{sym}:bars + LTRIM to 300 (newest at index 0)

    Replaces the old STOCK:{sym} tick-only hash entirely.
    Both stocks and NIFTY50 (symbol="NIFTY50") are handled identically.
    """

    def __init__(self, group_id: str = "viz-grp"):
        self.group_id      = group_id
        self.running       = False
        self.consumer      = None
        self.message_count = 0

        self.accumulators: Dict[str, BarAccumulator] = {}

        self.initialize_consumer()
        self._register_signal_handlers()

    def initialize_consumer(self):
        try:
            self.consumer = KafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_servers,
                group_id=self.group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_poll_records=100,
                session_timeout_ms=30000,
                consumer_timeout_ms=1000,
            )
            logger.info(f"Viz consumer initialized (group: {self.group_id})")
        except Exception as e:
            logger.error(f"Failed to initialize consumer: {e}")
            raise

    def _register_signal_handlers(self):
        try:
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        except (OSError, ValueError):
            pass
        signal.signal(signal.SIGINT, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def _process_tick(self, tick: dict):
        symbol = tick.get("symbol", "")
        if not symbol:
            return

        if symbol not in self.accumulators:
            self.accumulators[symbol] = BarAccumulator()

        acc = self.accumulators[symbol]
        bar_completed, completed_bar = acc.update(tick)

        if bar_completed and completed_bar is not None:
            self._push_completed_bar(symbol, completed_bar)

        self._write_current_bar(symbol, tick, acc)

    def _write_current_bar(self, symbol: str, tick: dict, acc: BarAccumulator):
        """Write live bar state + day-level tick data to STOCK:{sym}:current_bar every tick."""
        try:
            ltp  = tick.get("last_price", 0)
            ohlc = tick.get("ohlc", {})
            bar  = acc.current_bar() or {}

            state = {
                # Current-minute bar fields (updated on every tick)
                "bar_open":  str(bar.get("open",   ltp)),
                "bar_high":  str(bar.get("high",   ltp)),
                "bar_low":   str(bar.get("low",    ltp)),
                "bar_close": str(bar.get("close",  ltp)),
                "bar_vol":   str(bar.get("volume", 0)),
                "bar_ts":    str(bar.get("ts", "")),

                # Day-level from KiteConnect OHLC (backward-compatible field names)
                "LTP":          str(ltp),
                "Open":         str(ohlc.get("open",  0)),
                "High":         str(ohlc.get("high",  0)),
                "Low":          str(ohlc.get("low",   0)),
                "Close":        str(ohlc.get("close", 0)),
                "Volume":       str(tick.get("volume", 0)),
                "Change":       str(tick.get("change", 0)),
                "Last_Updated": datetime.utcnow().isoformat(),
            }

            redis_client.hset(f"STOCK:{symbol}:current_bar", state, ttl=3600)
            redis_client.publish(f"STOCK_UPDATE:{symbol}", json.dumps({"LTP": str(ltp)}))

        except Exception as e:
            logger.error(f"Error writing current_bar for {symbol}: {e}")

    def _push_completed_bar(self, symbol: str, bar: dict):
        """Push completed 1-min bar to STOCK:{sym}:bars (newest at index 0, capped at BAR_CAP)."""
        try:
            bar_json = json.dumps({
                "open":   bar["open"],
                "high":   bar["high"],
                "low":    bar["low"],
                "close":  bar["close"],
                "volume": bar["volume"],
                "ts":     bar["ts"],
            })
            redis_client.lpush(f"STOCK:{symbol}:bars", bar_json)
            redis_client.ltrim(f"STOCK:{symbol}:bars", 0, BAR_CAP - 1)
            logger.debug(f"Bar pushed for {symbol}: close={bar['close']:.2f}")
        except Exception as e:
            logger.error(f"Error pushing completed bar for {symbol}: {e}")

    def consume(self):
        self.running = True
        logger.info("Starting viz consumer (bar builder)...")

        try:
            while self.running:
                for message in self.consumer:
                    if not self.running:
                        break
                    try:
                        tick = message.value

                        is_valid, error = tick_validator.validate_tick(tick)
                        if not is_valid:
                            logger.warning(f"Invalid tick: {error}")
                            continue

                        self._process_tick(tick)

                        self.message_count += 1
                        if self.message_count % 500 == 0:
                            logger.info(
                                f"Processed {self.message_count} ticks (Viz) — "
                                f"{len(self.accumulators)} symbols tracked"
                            )

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
        finally:
            self.stop()

    def stop(self):
        if self.running:
            logger.info("Stopping viz consumer...")
            self.running = False
            if self.consumer:
                self.consumer.close()
            logger.info("Viz consumer stopped")


if __name__ == "__main__":
    consumer = VizConsumer()
    consumer.consume()
