import json
import signal
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from datetime import datetime
from config.settings import settings
from src.utils.logger import setup_logger
from src.core.redis_client import redis_client
from src.data.tick_validator import tick_validator

logger = setup_logger(__name__)

class VizConsumer:
    """
    Consumes ticks and updates real-time state in Redis for visualization.
    Aggregates OHLCV data per stock for UI polling/Pub-Sub delivery.
    """

    def __init__(self, group_id: str = "viz-grp"):
        self.group_id = group_id
        self.running = False
        self.consumer = None
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
            logger.info(f"Viz consumer initialized (group: {self.group_id})")
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

    def update_stock_state(self, tick: dict):
        """
        Update real-time stock state in Redis.
        Key: STOCK:<SYMBOL>
        Fields: LTP, OHLC, Volume, Last_Updated
        """
        try:
            symbol = tick.get("symbol", "")
            if not symbol:
                return

            cache_key = f"STOCK:{symbol}"
            state = {
                "LTP": str(tick.get("last_price", 0)),
                "Open": str(tick.get("ohlc", {}).get("open", 0)),
                "High": str(tick.get("ohlc", {}).get("high", 0)),
                "Low": str(tick.get("ohlc", {}).get("low", 0)),
                "Close": str(tick.get("ohlc", {}).get("close", 0)),
                "Volume": str(tick.get("volume", 0)),
                "Change": str(tick.get("change", 0)),
                "Last_Updated": datetime.utcnow().isoformat(),
            }

            redis_client.hset(cache_key, state, ttl=3600)  # 1 hour retention

            # Publish update to Redis Pub/Sub for real-time clients
            redis_client.publish(f"STOCK_UPDATE:{symbol}", json.dumps(state))

            logger.debug(f"Updated state for {symbol}")

        except Exception as e:
            logger.error(f"Error updating stock state: {e}")

    def consume(self):
        """Main consumption loop."""
        self.running = True
        logger.info("Starting viz consumer...")

        try:
            for message in self.consumer:
                try:
                    tick = message.value

                    # Validate tick
                    is_valid, error = tick_validator.validate_tick(tick)
                    if not is_valid:
                        logger.warning(f"Invalid tick: {error}")
                        continue

                    # Update state
                    self.update_stock_state(tick)

                    self.message_count += 1
                    if self.message_count % 500 == 0:
                        logger.info(f"Processed {self.message_count} ticks (Viz)")

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
            logger.info("Stopping viz consumer...")
            self.running = False
            if self.consumer:
                self.consumer.close()
            logger.info("Viz consumer stopped")

if __name__ == "__main__":
    consumer = VizConsumer()
    consumer.consume()
