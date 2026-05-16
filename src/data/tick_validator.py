import json
from typing import Dict, Tuple, Optional
from config.settings import settings
from src.utils.logger import setup_logger
from src.core.kafka_producer import kafka_producer

logger = setup_logger(__name__)

class TickValidator:
    """Validate incoming market ticks and handle outliers."""

    def __init__(self, outlier_threshold: float = 0.05):
        """
        Args:
            outlier_threshold: Reject ticks with price change > threshold (5% default)
        """
        self.outlier_threshold = outlier_threshold
        self.previous_prices = {}  # Track previous prices for outlier detection

    def validate_tick(self, tick: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a single tick.
        Returns: (is_valid, error_message)
        """
        # Check required fields
        required_fields = ["instrument_token", "last_price", "volume", "ohlc"]
        for field in required_fields:
            if field not in tick:
                return False, f"Missing required field: {field}"

        # Validate data types
        try:
            instrument_token = int(tick["instrument_token"])
            last_price = float(tick["last_price"])
            volume = int(tick["volume"])
        except (ValueError, TypeError) as e:
            return False, f"Invalid data type: {e}"

        # Validate price range (positive and reasonable)
        if last_price <= 0:
            return False, "Price must be positive"

        if last_price > 1_000_000:
            return False, "Price too high (> 1M)"

        # Validate volume
        if volume < 0:
            return False, "Volume cannot be negative"

        # Check for outliers (e.g., >5% price jump)
        if instrument_token in self.previous_prices:
            prev_price = self.previous_prices[instrument_token]
            price_change = abs((last_price - prev_price) / prev_price)

            if price_change > self.outlier_threshold:
                return False, f"Price outlier detected: {price_change*100:.2f}% change"

        # Update previous price
        self.previous_prices[instrument_token] = last_price

        return True, None

    def validate_batch(self, ticks: list) -> Tuple[list, list]:
        """
        Validate a batch of ticks.
        Returns: (valid_ticks, invalid_ticks_with_errors)
        """
        valid_ticks = []
        invalid_ticks = []

        for tick in ticks:
            is_valid, error_msg = self.validate_tick(tick)
            if is_valid:
                valid_ticks.append(tick)
            else:
                invalid_ticks.append({
                    "tick": tick,
                    "error": error_msg
                })
                logger.warning(f"Invalid tick: {error_msg}")

        return valid_ticks, invalid_ticks

    def send_to_dlq(self, invalid_tick: Dict, error: str):
        """Send invalid tick to Dead Letter Queue."""
        try:
            dlq_message = {
                "tick": invalid_tick,
                "error": error,
                "timestamp": str(datetime.utcnow())
            }
            # In production, write to a DLQ topic
            logger.error(f"DLQ: {json.dumps(dlq_message)}")
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")

# Singleton instance
from datetime import datetime
tick_validator = TickValidator()
