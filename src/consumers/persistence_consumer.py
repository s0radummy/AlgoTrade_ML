import json
import signal
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from datetime import datetime
from typing import List
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from config.settings import settings
from src.utils.logger import setup_logger
from src.data.tick_validator import tick_validator

logger = setup_logger(__name__)

class PersistenceConsumer:
    """
    Consumes ticks and writes to InfluxDB in batches.
    Persists all historical tick data for backtesting and analysis.
    """

    def __init__(self, group_id: str = "persistence-grp", batch_size: int = 1000):
        self.group_id = group_id
        self.running = False
        self.consumer = None
        self.batch_size = batch_size
        self.batch = []
        self.message_count = 0
        self.write_count = 0

        self.initialize_influxdb()
        self.initialize_consumer()
        self._register_signal_handlers()

    def initialize_influxdb(self):
        """Initialize InfluxDB client."""
        try:
            self.client = InfluxDBClient(
                url=f"http://{settings.influxdb_host}:{settings.influxdb_port}",
                org=settings.influxdb_org,
                token="dummy_token",  # Change to real token in production
            )
            self.write_api = self.client.write_api(write_type=SYNCHRONOUS)
            logger.info("InfluxDB client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB: {e}")
            # Continue without InfluxDB for dev purposes
            self.client = None
            self.write_api = None

    def initialize_consumer(self):
        """Initialize Kafka consumer."""
        try:
            self.consumer = KafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_servers,
                group_id=self.group_id,
                auto_offset_reset="earliest",  # Process all messages
                enable_auto_commit=False,  # Manual commit after write
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                max_poll_records=self.batch_size,
                session_timeout_ms=30000,
            )
            logger.info(f"Persistence consumer initialized (group: {self.group_id})")
        except Exception as e:
            logger.error(f"Failed to initialize consumer: {e}")
            raise

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, flushing batch and shutting down...")
        self.flush_batch()
        self.stop()

    def tick_to_line_protocol(self, tick: dict) -> str:
        """Convert tick to InfluxDB line protocol format."""
        try:
            symbol = tick.get("symbol", "UNKNOWN")
            timestamp = tick.get("timestamp", datetime.utcnow().isoformat())

            # Convert ISO timestamp to epoch nanoseconds
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp_ns = int(dt.timestamp() * 1e9)

            # InfluxDB line protocol: measurement,tags fields timestamp
            line = (
                f"ticks,"
                f"symbol={symbol},"
                f"instrument_token={tick.get('instrument_token', 0)} "
                f"last_price={tick.get('last_price', 0)},"
                f"volume={tick.get('volume', 0)},"
                f"change={tick.get('change', 0)} "
                f"{timestamp_ns}"
            )
            return line
        except Exception as e:
            logger.error(f"Error converting tick to line protocol: {e}")
            return None

    def add_to_batch(self, tick: dict):
        """Add tick to batch for writing."""
        line = self.tick_to_line_protocol(tick)
        if line:
            self.batch.append(line)

    def flush_batch(self):
        """Write batch to InfluxDB."""
        if not self.batch or not self.write_api:
            return

        try:
            batch_data = "\n".join(self.batch)
            self.write_api.write(
                bucket=settings.influxdb_bucket,
                org=settings.influxdb_org,
                write_precision="ns",
                record=batch_data,
            )
            self.write_count += len(self.batch)
            logger.info(
                f"Flushed {len(self.batch)} ticks to InfluxDB "
                f"(total: {self.write_count})"
            )
            self.batch = []

        except Exception as e:
            logger.error(f"Error writing to InfluxDB: {e}")
            # Don't clear batch on error; retry on next flush

    def consume(self):
        """Main consumption loop."""
        self.running = True
        logger.info("Starting persistence consumer...")

        try:
            for message in self.consumer:
                try:
                    tick = message.value

                    # Validate tick
                    is_valid, error = tick_validator.validate_tick(tick)
                    if not is_valid:
                        logger.warning(f"Invalid tick: {error}")
                        tick_validator.send_to_dlq(tick, error)
                        continue

                    # Add to batch
                    self.add_to_batch(tick)

                    self.message_count += 1

                    # Flush when batch is full or periodically
                    if len(self.batch) >= self.batch_size:
                        self.flush_batch()

                    if self.message_count % 500 == 0:
                        logger.info(f"Processed {self.message_count} ticks (Persistence)")

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
        finally:
            self.flush_batch()
            self.stop()

    def stop(self):
        """Stop consumer gracefully."""
        if self.running:
            logger.info("Stopping persistence consumer...")
            self.running = False
            if self.consumer:
                self.consumer.close()
            if self.client:
                self.client.close()
            logger.info("Persistence consumer stopped")

if __name__ == "__main__":
    consumer = PersistenceConsumer()
    consumer.consume()
