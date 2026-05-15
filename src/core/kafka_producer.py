import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError
from typing import Dict, Any, Optional
from datetime import datetime
import threading
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class KafkaProducerService:
    def __init__(self):
        self.producer = None
        self.running = False
        self.initialize_producer()

    def initialize_producer(self):
        """Initialize Kafka producer with error handling."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
                request_timeout_ms=30000,
            )
            self.running = True
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def send_tick(self, instrument_token: str, tick_data: Dict[str, Any]) -> bool:
        """Send a tick to Kafka topic."""
        try:
            # Add metadata to tick
            payload = {
                "instrument_token": instrument_token,
                "timestamp": datetime.utcnow().isoformat(),
                **tick_data
            }

            future = self.producer.send(
                settings.kafka_topic,
                value=payload,
                key=str(instrument_token).encode('utf-8')
            )

            # Wait for confirmation with timeout
            record_metadata = future.get(timeout=5)
            logger.debug(
                f"Sent tick for {instrument_token} to partition "
                f"{record_metadata.partition} at offset {record_metadata.offset}"
            )
            return True

        except KafkaError as e:
            logger.error(f"Kafka error sending tick: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending tick: {e}")
            return False

    def send_batch(self, ticks: list) -> int:
        """Send multiple ticks in batch."""
        success_count = 0
        for tick in ticks:
            if self.send_tick(tick.get("instrument_token"), tick):
                success_count += 1
        return success_count

    def reconnect_with_backoff(self, max_retries: int = 5):
        """Reconnect with exponential backoff."""
        retry_count = 0
        base_wait = 1

        while retry_count < max_retries:
            try:
                self.initialize_producer()
                logger.info("Reconnected to Kafka producer")
                return True
            except Exception as e:
                retry_count += 1
                wait_time = min(base_wait * (2 ** retry_count), 60)
                logger.warning(
                    f"Reconnection attempt {retry_count}/{max_retries} failed. "
                    f"Retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

        logger.error(f"Failed to reconnect after {max_retries} attempts")
        return False

    def close(self):
        """Close Kafka producer connection."""
        try:
            if self.producer:
                self.producer.flush(timeout_ms=5000)
                self.producer.close()
                self.running = False
                logger.info("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")

    def is_connected(self) -> bool:
        """Check if producer is connected."""
        return self.running and self.producer is not None

# Singleton instance
kafka_producer = KafkaProducerService()
