import json
import signal
import torch
from collections import deque
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from datetime import datetime
from typing import Dict, Optional
import numpy as np

from config.settings import settings
from src.utils.logger import setup_logger
from src.core.redis_client import redis_client
from src.models.model_manager import model_manager
from src.data.instrument_loader import instrument_loader
from src.data.tick_validator import tick_validator

logger = setup_logger(__name__)

class InferenceConsumer:
    """
    Consumes ticks from Kafka, runs TFT inference, and caches predictions in Redis.
    LATENCY-CRITICAL: Target <400ms per tick.
    """

    def __init__(self, group_id: str = "inference-grp"):
        self.group_id = group_id
        self.running = False
        self.consumer = None
        self.ticks_deque = {}  # {instrument_token: deque(60)}
        self.message_count = 0

        self.initialize_consumer()
        self._register_signal_handlers()

    def initialize_consumer(self):
        """Initialize Kafka consumer."""
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

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def hydrate_deque(self, instrument_token: str):
        """
        Initialize deque with latest 60 ticks from InfluxDB.
        In production, query InfluxDB; for now, initialize empty.
        """
        self.ticks_deque[instrument_token] = deque(maxlen=60)
        logger.debug(f"Hydrated deque for instrument {instrument_token}")

    def update_deque(self, instrument_token: str, tick: Dict):
        """Update deque with new tick (FIFO, max 60)."""
        if instrument_token not in self.ticks_deque:
            self.hydrate_deque(instrument_token)

        self.ticks_deque[instrument_token].append(tick)

    def get_log_returns(self, prices: list) -> list:
        """Compute log returns from prices."""
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                log_ret = np.log(prices[i] / prices[i-1])
                log_returns.append(log_ret)
        return log_returns

    def construct_input_tensor(self, instrument_token: str, tick: Dict) -> Optional[Dict]:
        """
        Construct TFT input tensor from tick and deque.
        Returns: {static_cov, past_inputs, future_inputs} or None if not ready
        """
        try:
            instrument_data = instrument_loader.get_instrument(tick.get("symbol", ""))
            if not instrument_data:
                return None

            deque_ticks = list(self.ticks_deque.get(instrument_token, []))
            if len(deque_ticks) < 10:  # Need at least 10 ticks to start
                return None

            # 1. Static Covariates
            symbol = tick.get("symbol", "")
            sector = instrument_data.get("sector", "Unknown")
            market_cap = instrument_data.get("market_cap", 1000000000)

            static_cov = torch.tensor([
                hash(symbol) % 100 / 100,  # Stock ID embedding (normalized hash)
                float(instrument_loader.get_sector_embedding(sector)),  # Sector
                instrument_loader.normalize_market_cap(market_cap),  # Market cap
            ], dtype=torch.float32)

            # Pad to 32 dims
            static_cov = torch.cat([static_cov, torch.zeros(32 - 3)])
            static_cov = static_cov.unsqueeze(0)  # (1, 32)

            # 2. Past Inputs (60 ticks)
            prices = [t.get("last_price", 0) for t in deque_ticks]
            volumes = [t.get("volume", 0) for t in deque_ticks]
            log_rets = self.get_log_returns(prices)

            # Normalize
            if volumes:
                max_vol = max(volumes)
                volumes_norm = [v / max_vol if max_vol > 0 else 0 for v in volumes]
            else:
                volumes_norm = [0] * len(volumes)

            # Construct features: [log_ret, volume, open, high, low, close, ...]
            past_features = []
            for i, t in enumerate(deque_ticks):
                feat = [
                    log_rets[i] if i < len(log_rets) else 0,
                    volumes_norm[i],
                    t.get("ohlc", {}).get("open", 0),
                    t.get("ohlc", {}).get("high", 0),
                    t.get("ohlc", {}).get("low", 0),
                    t.get("ohlc", {}).get("close", 0),
                    t.get("last_price", 0),
                ]
                past_features.append(feat)

            past_inputs = torch.tensor(past_features, dtype=torch.float32).unsqueeze(0)  # (1, 60, 7)

            # 3. Future Inputs (time features for next 5 ticks)
            now = datetime.utcnow()
            hour = float(now.hour)
            day_of_week = float(now.weekday())
            is_trading_hours = 1.0 if 9.25 <= hour < 15.5 else 0.0
            day_of_month = float(now.day)

            future_features = []
            for step in range(5):
                feat = [hour, day_of_week, is_trading_hours, day_of_month]
                future_features.append(feat)

            future_inputs = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0)  # (1, 5, 4)

            return {
                "static_cov": static_cov,
                "past_inputs": past_inputs,
                "future_inputs": future_inputs,
            }

        except Exception as e:
            logger.error(f"Error constructing input tensor: {e}")
            return None

    def run_inference(self, instrument_token: str, tick: Dict):
        """Run TFT inference and cache predictions."""
        try:
            inputs = self.construct_input_tensor(instrument_token, tick)
            if inputs is None:
                return

            # Get predictions
            predictions = model_manager.predict(
                inputs["static_cov"],
                inputs["past_inputs"],
                inputs["future_inputs"]
            )

            # Convert to quantiles (5 quantiles for each of 5 future steps)
            preds = predictions[0].cpu().numpy()  # (5 steps, 5 quantiles)

            # Cache only P50 (median) for now
            quantiles_dict = {
                "P10": float(preds[0, 0]),
                "P30": float(preds[0, 1]),
                "P50": float(preds[0, 2]),
                "P70": float(preds[0, 3]),
                "P90": float(preds[0, 4]),
                "timestamp": datetime.utcnow().isoformat(),
            }

            model_manager.cache_prediction(
                tick.get("symbol", ""),
                quantiles_dict,
                ttl=settings.model_cache_ttl
            )

            logger.debug(f"Cached predictions for {tick.get('symbol', '')} (P50: {quantiles_dict['P50']:.6f})")

        except Exception as e:
            logger.error(f"Inference error: {e}")
            # Circuit breaker: use cached prediction
            model_manager.circuit_breaker_fallback(tick.get("symbol", ""))

    def consume(self):
        """Main consumption loop."""
        self.running = True
        logger.info("Starting inference consumer...")

        try:
            for message in self.consumer:
                try:
                    tick = message.value

                    # Validate tick
                    is_valid, error = tick_validator.validate_tick(tick)
                    if not is_valid:
                        logger.warning(f"Invalid tick: {error}")
                        continue

                    symbol = tick.get("symbol", "")
                    instrument_token = tick.get("instrument_token")

                    # Update deque
                    self.update_deque(instrument_token, tick)

                    # Run inference
                    self.run_inference(instrument_token, tick)

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
        """Stop consumer gracefully."""
        if self.running:
            logger.info("Stopping inference consumer...")
            self.running = False
            if self.consumer:
                self.consumer.close()
            logger.info("Inference consumer stopped")

if __name__ == "__main__":
    consumer = InferenceConsumer()
    consumer.consume()
