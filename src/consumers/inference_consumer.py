import json
import os
import signal
import torch
import pandas as pd
from collections import deque
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import settings
from src.utils.logger import setup_logger
from src.core.redis_client import redis_client
from src.models.model_manager import model_manager
from src.data.instrument_loader import instrument_loader
from src.data.tick_validator import tick_validator
from src.data.dataset import _build_static_cov
from src.data.features import compute_rsi, compute_macd_hist, compute_atr

logger = setup_logger(__name__)

# Nifty50 index instrument token (stable across all KiteConnect accounts)
NIFTY50_TOKEN = 256265

# Minimum completed 1-minute bars required before running inference
MIN_BARS = 60


class BarAccumulator:
    """
    Converts a stream of raw KiteConnect ticks into completed 1-minute OHLCV bars.

    KiteConnect ticks carry cumulative daily volume (volume_traded), so bar volume
    is computed as the delta between the first and last tick in each minute.

    Maintains a deque of completed bars. Call get_bars() to get completed bars
    plus the current in-progress bar for immediate use in inference.
    """

    def __init__(self, maxlen: int = 60):
        self.completed: deque = deque(maxlen=maxlen)
        self._bar: Optional[Dict] = None          # current in-progress bar
        self._bar_key: Optional[Tuple] = None     # (year, month, day, hour, minute)
        self._bar_start_vol: float = 0.0          # cumulative volume at bar open

    def update(self, tick: Dict) -> bool:
        """
        Ingest one tick. Returns True when a bar is completed (minute boundary crossed).
        Ticks from KiteConnect carry exchange_timestamp in milliseconds.
        """
        ts_ms = tick.get("exchange_timestamp") or tick.get("timestamp", 0)
        if ts_ms:
            try:
                if isinstance(ts_ms, str):
                    ts = datetime.fromisoformat(ts_ms).replace(tzinfo=timezone.utc)
                else:
                    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                ts = datetime.now(tz=timezone.utc)
        else:
            ts = datetime.now(tz=timezone.utc)

        bar_key = (ts.year, ts.month, ts.day, ts.hour, ts.minute)
        ltp     = float(tick.get("last_price", 0.0))
        cum_vol = float(tick.get("volume_traded", tick.get("volume", 0.0)))

        if bar_key != self._bar_key:
            completed_bar = False
            if self._bar is not None:
                self.completed.append(self._bar)
                completed_bar = True
            self._bar = {
                "open":   ltp,
                "high":   ltp,
                "low":    ltp,
                "close":  ltp,
                "volume": 0.0,
                "ts":     ts,
            }
            self._bar_key    = bar_key
            self._bar_start_vol = cum_vol
            return completed_bar
        else:
            if ltp > 0:
                self._bar["high"]  = max(self._bar["high"], ltp)
                self._bar["low"]   = min(self._bar["low"],  ltp)
                self._bar["close"] = ltp
            self._bar["volume"] = max(0.0, cum_vol - self._bar_start_vol)
            return False

    def get_bars(self) -> List[Dict]:
        """Completed bars + current in-progress bar (most recent last)."""
        bars = list(self.completed)
        if self._bar is not None:
            bars.append(self._bar)
        return bars

    def ready(self) -> bool:
        return len(self.completed) >= MIN_BARS


class InferenceConsumer:
    """
    Consumes ticks from Kafka, accumulates them into 1-minute OHLCV bars,
    runs TFT inference after each completed bar, and caches predictions in Redis.

    Feature layout for past_inputs (60, 11) — matches src/data/dataset.py exactly:
      col  0: log_ret          (z-scored per stock)
      col  1: open_ret         = log(open / prev_close)
      col  2: high_ret         = log(high / prev_close)
      col  3: low_ret          = log(low  / prev_close)
      col  4: intraday_ret     = log(close / open)
      col  5: intraday_rng     = log(high / low)
      col  6: vol_norm         = log1p(bar_vol / mean_bar_vol)
      col  7: rsi_14           (compute_rsi from features.py)
      col  8: macd_hist_norm   (compute_macd_hist from features.py)
      col  9: atr_norm         (compute_atr from features.py)
      col 10: nifty_log_ret    (Nifty50 index 1-min log return, raw)

    Predictions are denormalized from z-score space using per-stock target_stats
    loaded from the checkpoint before writing to Redis.
    """

    def __init__(self, group_id: str = "inference-grp"):
        self.group_id = group_id
        self.running  = False
        self.consumer = None

        # Per-stock bar accumulators: {instrument_token: BarAccumulator}
        self.accumulators: Dict[str, BarAccumulator] = {}

        # Shared Nifty50 bar accumulator (instrument_token == NIFTY50_TOKEN)
        self.nifty_acc = BarAccumulator(maxlen=MIN_BARS)

        self.message_count = 0
        self.target_stats: dict = {}

        self.initialize_consumer()
        self._load_target_stats()
        self._register_signal_handlers()
        self.warm_bars: Dict[str, List[Dict]] = self._warm_start_from_parquet()

    def initialize_consumer(self):
        try:
            self.consumer = KafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_servers,
                group_id=self.group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                max_poll_records=100,
                session_timeout_ms=30000,
            )
            logger.info(f"Inference consumer initialized (group: {self.group_id})")
        except Exception as e:
            logger.error(f"Failed to initialize consumer: {e}")
            raise

    def _load_target_stats(self):
        """Load per-stock target normalization stats from the model checkpoint."""
        try:
            if os.path.exists(settings.model_path):
                ckpt = torch.load(settings.model_path, map_location="cpu", weights_only=False)
                self.target_stats = ckpt.get("target_stats", {})
                logger.info(f"Loaded target_stats for {len(self.target_stats)} stocks")
            else:
                logger.warning("No checkpoint found — predictions will not be denormalized")
        except Exception as e:
            logger.warning(f"Could not load target_stats from checkpoint: {e}")

    def _register_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def _bars_to_features(self, bars: List[Dict]) -> Optional[np.ndarray]:
        """
        Convert a list of 1-min OHLCV bars to the (N, 11) feature matrix
        that matches dataset.py _load_stock exactly.

        Returns None if fewer than 2 bars (need prev_close for log returns).
        """
        n = len(bars)
        if n < 2:
            return None

        close = np.array([b["close"] for b in bars], dtype=np.float64)
        open_ = np.array([b["open"]  for b in bars], dtype=np.float64)
        high  = np.array([b["high"]  for b in bars], dtype=np.float64)
        low   = np.array([b["low"]   for b in bars], dtype=np.float64)
        vol   = np.array([b["volume"] for b in bars], dtype=np.float64)

        prev_close    = np.empty_like(close)
        prev_close[0] = open_[0]
        prev_close[1:] = close[:-1]

        pos_vol  = vol[vol > 0]
        mean_vol = pos_vol.mean() if len(pos_vol) > 0 else 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret      = np.log(close  / prev_close)
            open_ret     = np.log(open_  / prev_close)
            high_ret     = np.log(high   / prev_close)
            low_ret      = np.log(low    / prev_close)
            intraday_ret = np.log(close  / np.where(open_ > 0, open_, 1.0))
            intraday_rng = np.log(high   / np.where(low   > 0, low,   1.0))
            vol_norm     = np.log1p(vol  / mean_vol)

        return np.column_stack([
            log_ret, open_ret, high_ret, low_ret,
            intraday_ret, intraday_rng, vol_norm,
        ]).astype(np.float32)

    def construct_input_tensor(self, instrument_token: str, symbol: str) -> Optional[Dict]:
        """
        Build TFT input tensors from the accumulated 1-min bars for this stock.
        Returns {static_cov, past_inputs, future_inputs} or None if not ready.
        """
        try:
            acc = self.accumulators.get(instrument_token)
            if acc is None or not acc.ready():
                logger.debug(f"[{symbol}] construct: acc not ready (completed={len(acc.completed) if acc else 'None'})")
                return None

            bars = acc.get_bars()
            if len(bars) < MIN_BARS + 1:
                logger.debug(f"[{symbol}] construct: not enough bars ({len(bars)} < {MIN_BARS + 1})")
                return None

            # Use the last MIN_BARS+1 bars so we have prev_close for bar[0]
            bars = bars[-(MIN_BARS + 1):]
            feat_mat = self._bars_to_features(bars)  # (MIN_BARS+1, 7) or None
            if feat_mat is None:
                return None

            # Take the last MIN_BARS rows (drop the leading row used only for prev_close)
            feat_mat = feat_mat[-MIN_BARS:]  # (60, 7)
            closes = np.array([b["close"] for b in bars], dtype=np.float64)[-MIN_BARS:]
            highs  = np.array([b["high"]  for b in bars], dtype=np.float64)[-MIN_BARS:]
            lows   = np.array([b["low"]   for b in bars], dtype=np.float64)[-MIN_BARS:]

            # Technical indicators (cols 7–9)
            rsi_vals  = compute_rsi(closes).astype(np.float32)[-MIN_BARS:]
            macd_vals = compute_macd_hist(closes).astype(np.float32)[-MIN_BARS:]
            atr_vals  = compute_atr(highs, lows, closes).astype(np.float32)[-MIN_BARS:]

            # Nifty50 log returns (col 10)
            nifty_bars = self.nifty_acc.get_bars()
            if len(nifty_bars) >= 2:
                nifty_close = np.array([b["close"] for b in nifty_bars], dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    nifty_lr = np.log(nifty_close[1:] / nifty_close[:-1])
                nifty_lr = np.nan_to_num(nifty_lr, nan=0.0, posinf=0.0, neginf=0.0)
                # Align length to MIN_BARS; pad front with zeros if shorter
                if len(nifty_lr) < MIN_BARS:
                    nifty_lr = np.concatenate([np.zeros(MIN_BARS - len(nifty_lr)), nifty_lr])
                else:
                    nifty_lr = nifty_lr[-MIN_BARS:]
            else:
                nifty_lr = np.zeros(MIN_BARS, dtype=np.float32)

            nifty_lr = nifty_lr.astype(np.float32)

            # Assemble (60, 11) — col order matches dataset.py
            past_mat = np.column_stack([
                feat_mat,                          # cols 0–6
                rsi_vals[:, None],                 # col 7
                macd_vals[:, None],                # col 8
                atr_vals[:, None],                 # col 9
                nifty_lr[:, None],                 # col 10
            ])
            past_mat = np.nan_to_num(past_mat, nan=0.0, posinf=0.0, neginf=0.0)

            # Z-score col 0 (log_ret) per stock
            stats       = self.target_stats.get(symbol, {"mean": 0.0, "std": 0.0007})
            target_mean = stats.get("mean", 0.0)
            target_std  = stats.get("std",  0.0007)
            if target_std > 0:
                past_mat[:, 0] = (past_mat[:, 0] - target_mean) / target_std

            # Static covariates
            static_np  = _build_static_cov(symbol, target_std=target_std)
            static_cov = torch.from_numpy(static_np).unsqueeze(0).float()  # (1, 32)

            past_inputs = torch.from_numpy(past_mat).unsqueeze(0).float()  # (1, 60, 11)

            # Future inputs — cyclical time + calendar (matches dataset.py calendar encoding)
            last_bar_ts = bars[-1].get("ts", datetime.now(tz=timezone.utc))
            open_min_utc   = 3 * 60 + 45          # 09:15 IST = 03:45 UTC
            mins_since_open = (last_bar_ts.hour * 60 + last_bar_ts.minute) - open_min_utc

            future_feats = []
            for step in range(5):
                frac = (mins_since_open + step + 1) / 375.0
                future_feats.append([
                    float(np.sin(2 * np.pi * frac)),
                    float(np.cos(2 * np.pi * frac)),
                    float(last_bar_ts.weekday()) / 4.0,
                    float(last_bar_ts.day) / 31.0,
                ])
            future_inputs = torch.tensor(future_feats, dtype=torch.float32).unsqueeze(0)  # (1, 5, 4)

            return {
                "static_cov":    static_cov,
                "past_inputs":   past_inputs,
                "future_inputs": future_inputs,
            }

        except Exception as e:
            logger.error(f"Error constructing input tensor for {symbol}: {e}")
            return None

    def run_inference(self, instrument_token: str, symbol: str):
        """Run TFT inference and cache denormalized predictions in Redis."""
        try:
            inputs = self.construct_input_tensor(instrument_token, symbol)
            if inputs is None:
                return

            predictions = model_manager.predict(
                inputs["static_cov"],
                inputs["past_inputs"],
                inputs["future_inputs"],
            )

            preds = predictions[0].cpu().numpy()  # (5 steps, 5 quantiles) — z-score space

            stats        = self.target_stats.get(symbol, {"mean": 0.0, "std": 1.0})
            preds_denorm = preds * stats["std"] + stats["mean"]

            quantiles_dict = {
                "P10": float(preds_denorm[0, 0]),
                "P30": float(preds_denorm[0, 1]),
                "P50": float(preds_denorm[0, 2]),
                "P70": float(preds_denorm[0, 3]),
                "P90": float(preds_denorm[0, 4]),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

            logger.debug(f"[{symbol}] Writing prediction P50={quantiles_dict['P50']:.6f} to Redis (ttl={settings.model_cache_ttl}s)")
            model_manager.cache_prediction(
                symbol,
                quantiles_dict,
                ttl=settings.model_cache_ttl,
            )
            logger.debug(f"[{symbol}] Prediction cached OK")

        except Exception as e:
            logger.error(f"Inference error for {symbol}: {e}")
            model_manager.circuit_breaker_fallback(symbol)

    def _warm_start_from_parquet(self) -> Dict[str, List[Dict]]:
        """Pre-load last MIN_BARS+5 bars per stock from parquet so inference starts immediately."""
        parquet_dir = "data/historical"
        warm_bars: Dict[str, List[Dict]] = {}

        def _parquet_to_bars(path: str) -> List[Dict]:
            df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.tail(MIN_BARS + 5)
            bars = []
            for _, row in df.iterrows():
                ts = pd.to_datetime(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                bars.append({
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row["volume"]),
                    "ts":     ts.to_pydatetime(),
                })
            return bars

        # Per-stock warm-start
        for symbol in settings.stock_list:
            path = os.path.join(parquet_dir, f"{symbol}.parquet")
            if not os.path.exists(path):
                continue
            try:
                warm_bars[symbol] = _parquet_to_bars(path)
            except Exception as e:
                logger.warning(f"Could not warm-start {symbol} from parquet: {e}")

        # Nifty50 index warm-start
        nifty_path = os.path.join(parquet_dir, "NIFTY50.parquet")
        if os.path.exists(nifty_path):
            try:
                for bar in _parquet_to_bars(nifty_path):
                    self.nifty_acc.completed.append(bar)
                logger.info(f"Nifty50 warm-started with {len(self.nifty_acc.completed)} bars")
            except Exception as e:
                logger.warning(f"Could not warm-start Nifty50 from parquet: {e}")

        logger.info(f"Parquet warm-start complete: {len(warm_bars)}/{len(settings.stock_list)} stocks ready")
        return warm_bars

    def consume(self):
        self.running = True
        logger.info("Starting inference consumer (bar-based, 1-min resolution)...")

        try:
            for message in self.consumer:
                try:
                    tick = message.value

                    is_valid, error = tick_validator.validate_tick(tick)
                    if not is_valid:
                        logger.warning(f"Invalid tick: {error}")
                        continue

                    instrument_token = str(tick.get("instrument_token", ""))
                    symbol           = tick.get("symbol", "")

                    # Route Nifty50 index ticks to the shared accumulator
                    if instrument_token == str(NIFTY50_TOKEN):
                        self.nifty_acc.update(tick)
                        continue

                    # Per-stock bar accumulation
                    if instrument_token not in self.accumulators:
                        acc = BarAccumulator(maxlen=MIN_BARS)
                        if symbol in self.warm_bars:
                            for bar in self.warm_bars.pop(symbol):
                                acc.completed.append(bar)
                            logger.info(f"Warm-started {symbol} with {len(acc.completed)} historical bars")
                        self.accumulators[instrument_token] = acc

                    bar_completed = self.accumulators[instrument_token].update(tick)

                    if bar_completed:
                        logger.debug(f"Bar completed for {symbol} (ready={self.accumulators[instrument_token].ready()})")

                    # Run inference whenever a bar is completed and we have enough history
                    if bar_completed and self.accumulators[instrument_token].ready():
                        instrument_data = instrument_loader.get_instrument(symbol)
                        if instrument_data:
                            self.run_inference(instrument_token, symbol)

                    self.message_count += 1
                    if self.message_count % 100 == 0:
                        logger.info(f"Processed {self.message_count} ticks")

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
        finally:
            self.stop()

    def stop(self):
        if self.running:
            logger.info("Stopping inference consumer...")
            self.running = False
            if self.consumer:
                self.consumer.close()
            logger.info("Inference consumer stopped")


if __name__ == "__main__":
    consumer = InferenceConsumer()
    consumer.consume()
