import json
import os
import signal
import time
import torch
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config.settings import settings
from src.core.redis_client import redis_client
from src.data.dataset import _build_static_cov
from src.data.features import compute_atr, compute_macd_hist, compute_rsi
from src.data.instrument_loader import instrument_loader
from src.data.tick_validator import tick_validator
from src.models.model_manager import model_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Minimum completed 1-minute bars required before running inference
MIN_BARS = 60


class InferenceConsumer:
    """
    Reads ticks from Kafka to detect minute-boundary crossings, then reads the
    completed bars written by VizConsumer from Redis and runs TFT inference.

    Feature layout for past_inputs (60, 11) — matches src/data/dataset.py exactly:
      col  0: log_ret          (z-scored per stock)
      col  1: open_ret         = log(open / prev_close)
      col  2: high_ret         = log(high / prev_close)
      col  3: low_ret          = log(low  / prev_close)
      col  4: intraday_ret     = log(close / open)
      col  5: intraday_rng     = log(high / low)
      col  6: vol_norm         = log1p(bar_vol / mean_bar_vol)
      col  7: rsi_14
      col  8: macd_hist_norm
      col  9: atr_norm
      col 10: nifty_log_ret    (from STOCK:NIFTY50:bars in Redis)
    """

    def __init__(self, group_id: str = "inference-grp"):
        self.group_id      = group_id
        self.running       = False
        self.consumer      = None
        self.message_count = 0
        self.target_stats: dict = {}

        # Last-seen minute per symbol for boundary detection: {symbol: (y,m,d,h,min)}
        self._last_minute: Dict[str, Tuple] = {}

        self.initialize_consumer()
        self._load_target_stats()
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
            logger.info(f"Inference consumer initialized (group: {self.group_id})")
        except Exception as e:
            logger.error(f"Failed to initialize consumer: {e}")
            raise

    def _load_target_stats(self):
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
        try:
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        except (OSError, ValueError):
            pass
        signal.signal(signal.SIGINT, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def _bars_to_features(self, bars: List[Dict]) -> Optional[np.ndarray]:
        """Convert list of OHLCV bar dicts to (N, 7) feature array (cols 0–6)."""
        n = len(bars)
        if n < 2:
            return None

        close = np.array([b["close"]  for b in bars], dtype=np.float64)
        open_ = np.array([b["open"]   for b in bars], dtype=np.float64)
        high  = np.array([b["high"]   for b in bars], dtype=np.float64)
        low   = np.array([b["low"]    for b in bars], dtype=np.float64)
        vol   = np.array([b["volume"] for b in bars], dtype=np.float64)

        prev_close     = np.empty_like(close)
        prev_close[0]  = open_[0]
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

    def construct_input_tensor(
        self,
        symbol: str,
        bars: List[Dict],
        nifty_bars: List[Dict],
    ) -> Optional[Dict]:
        """
        Build TFT input tensors from Redis bar lists (chronological, oldest first).
        Returns {static_cov, past_inputs, future_inputs} or None if not enough data.
        """
        try:
            if len(bars) < MIN_BARS + 1:
                logger.debug(f"[{symbol}] Not enough bars ({len(bars)} < {MIN_BARS + 1})")
                return None

            # Feature matrix: need MIN_BARS+1 bars to compute first prev_close
            bars_slice = bars[-(MIN_BARS + 1):]
            feat_mat   = self._bars_to_features(bars_slice)
            if feat_mat is None:
                return None
            feat_mat = feat_mat[-MIN_BARS:]  # (60, 7)

            # Technical indicators computed on ALL available bars for full MACD warmup
            all_closes = np.array([b["close"] for b in bars], dtype=np.float64)
            all_highs  = np.array([b["high"]  for b in bars], dtype=np.float64)
            all_lows   = np.array([b["low"]   for b in bars], dtype=np.float64)

            rsi_vals  = compute_rsi(all_closes).astype(np.float32)[-MIN_BARS:]
            macd_vals = compute_macd_hist(all_closes).astype(np.float32)[-MIN_BARS:]
            atr_vals  = compute_atr(all_highs, all_lows, all_closes).astype(np.float32)[-MIN_BARS:]

            # Nifty50 log returns (col 10)
            if len(nifty_bars) >= 2:
                nifty_close = np.array([b["close"] for b in nifty_bars], dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    nifty_lr = np.log(nifty_close[1:] / nifty_close[:-1])
                nifty_lr = np.nan_to_num(nifty_lr, nan=0.0, posinf=0.0, neginf=0.0)
                if len(nifty_lr) < MIN_BARS:
                    nifty_lr = np.concatenate([np.zeros(MIN_BARS - len(nifty_lr)), nifty_lr])
                else:
                    nifty_lr = nifty_lr[-MIN_BARS:]
            else:
                nifty_lr = np.zeros(MIN_BARS, dtype=np.float32)
            nifty_lr = nifty_lr.astype(np.float32)

            # Assemble (60, 11)
            past_mat = np.column_stack([
                feat_mat,
                rsi_vals[:, None],
                macd_vals[:, None],
                atr_vals[:, None],
                nifty_lr[:, None],
            ])
            past_mat = np.nan_to_num(past_mat, nan=0.0, posinf=0.0, neginf=0.0)

            # Z-score col 0 (log_ret) per stock
            stats       = self.target_stats.get(symbol, {"mean": 0.0, "std": 0.0007})
            target_mean = stats.get("mean", 0.0)
            target_std  = stats.get("std",  0.0007)
            if target_std > 0:
                past_mat[:, 0] = (past_mat[:, 0] - target_mean) / target_std

            static_np   = _build_static_cov(symbol, target_std=target_std)
            static_cov  = torch.from_numpy(static_np).unsqueeze(0).float()
            past_inputs = torch.from_numpy(past_mat).unsqueeze(0).float()

            # Future inputs built from last completed bar's timestamp
            last_bar_ts_str = bars[-1].get("ts", "")
            try:
                last_bar_ts = datetime.fromisoformat(last_bar_ts_str)
                if last_bar_ts.tzinfo is None:
                    last_bar_ts = last_bar_ts.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                last_bar_ts = datetime.now(tz=timezone.utc)

            open_min_utc    = 3 * 60 + 45   # 09:15 IST = 03:45 UTC
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
            future_inputs = torch.tensor(future_feats, dtype=torch.float32).unsqueeze(0)

            return {
                "static_cov":    static_cov,
                "past_inputs":   past_inputs,
                "future_inputs": future_inputs,
            }

        except Exception as e:
            logger.error(f"Error constructing input tensor for {symbol}: {e}")
            return None

    def run_inference(self, symbol: str, bars: List[Dict], nifty_bars: List[Dict]):
        """Run TFT inference and write denormalized predictions to Redis."""
        try:
            inputs = self.construct_input_tensor(symbol, bars, nifty_bars)
            if inputs is None:
                return

            predictions  = model_manager.predict(
                inputs["static_cov"],
                inputs["past_inputs"],
                inputs["future_inputs"],
            )

            preds = predictions[0].cpu().numpy()  # (5 steps, 5 quantiles) in z-score space

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

            all_steps = [
                {
                    "P10": float(preds_denorm[s, 0]),
                    "P30": float(preds_denorm[s, 1]),
                    "P50": float(preds_denorm[s, 2]),
                    "P70": float(preds_denorm[s, 3]),
                    "P90": float(preds_denorm[s, 4]),
                }
                for s in range(preds_denorm.shape[0])
            ]

            # Current LTP from the live bar written by VizConsumer
            current_bar  = redis_client.hgetall(f"STOCK:{symbol}:current_bar")
            current_ltp  = float(current_bar.get("LTP", 0) or 0)

            redis_client.set(
                f"PRED:{symbol}:steps",
                json.dumps({
                    "steps":     all_steps,
                    "ltp":       current_ltp,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }),
                ttl=settings.model_cache_ttl,
            )

            logger.debug(f"[{symbol}] P50={quantiles_dict['P50']:.6f}")
            model_manager.cache_prediction(symbol, quantiles_dict, ttl=settings.model_cache_ttl)

        except Exception as e:
            logger.error(f"Inference error for {symbol}: {e}")
            model_manager.circuit_breaker_fallback(symbol)

    def _trigger_inference(self, symbol: str):
        """
        Read bars from Redis and run inference after a minute-boundary is detected.
        100ms retry guards against the narrow race where VizConsumer hasn't yet written
        the completed bar (almost never fires in practice).
        """
        newest_raw = redis_client.lindex(f"STOCK:{symbol}:bars", 0)
        if newest_raw is None:
            time.sleep(0.1)
            newest_raw = redis_client.lindex(f"STOCK:{symbol}:bars", 0)
            if newest_raw is None:
                logger.debug(f"[{symbol}] No bars in Redis yet — skipping")
                return

        bars_raw  = redis_client.lrange(f"STOCK:{symbol}:bars", 0, 99)
        nifty_raw = redis_client.lrange("STOCK:NIFTY50:bars",   0, 99)

        bars = []
        for b in reversed(bars_raw):
            try:
                bars.append(json.loads(b))
            except Exception:
                pass

        nifty_bars = []
        for b in reversed(nifty_raw):
            try:
                nifty_bars.append(json.loads(b))
            except Exception:
                pass

        if len(bars) < MIN_BARS:
            logger.debug(f"[{symbol}] Not enough bars ({len(bars)} < {MIN_BARS})")
            return

        if instrument_loader.get_instrument(symbol):
            self.run_inference(symbol, bars, nifty_bars)

    def consume(self):
        self.running = True
        logger.info("Starting inference consumer (Redis-backed, 1-min bars)...")

        try:
            while self.running:
                for message in self.consumer:
                    if not self.running:
                        break
                    try:
                        tick = message.value

                        is_valid, error = tick_validator.validate_tick(tick)
                        if not is_valid:
                            continue

                        symbol = tick.get("symbol", "")
                        if not symbol or symbol == "NIFTY50":
                            continue

                        try:
                            ts = datetime.fromisoformat(tick.get("timestamp", ""))
                            current_minute = (ts.year, ts.month, ts.day, ts.hour, ts.minute)
                        except (ValueError, AttributeError):
                            continue

                        last_minute = self._last_minute.get(symbol)
                        self._last_minute[symbol] = current_minute

                        if last_minute is not None and current_minute != last_minute:
                            self._trigger_inference(symbol)

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
