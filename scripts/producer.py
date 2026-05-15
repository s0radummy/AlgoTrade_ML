"""
Sample producer script that generates mock market data for testing.
In production, replace with real KiteConnect WebSocket consumer.
"""

import json
import time
import random
from datetime import datetime
from config.settings import settings
from src.core.kafka_producer import kafka_producer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_mock_tick(symbol: str, base_price: float, variation: float = 0.02) -> dict:
    """Generate mock OHLCV tick data."""
    price_change = random.uniform(-variation, variation)
    new_price = base_price * (1 + price_change)

    return {
        "symbol": symbol,
        "instrument_token": hash(symbol) % 1000000,
        "last_price": new_price,
        "volume": random.randint(1000, 100000),
        "ohlc": {
            "open": base_price,
            "high": max(base_price, new_price) * 1.01,
            "low": min(base_price, new_price) * 0.99,
            "close": new_price
        },
        "change": (new_price - base_price) / base_price,
        "timestamp": datetime.utcnow().isoformat()
    }

def run_mock_producer():
    """Run mock producer."""
    logger.info("Starting mock Kafka producer...")

    # Stock prices (base for simulation)
    stocks = {
        "RELIANCE": 2500,
        "INFY": 1800,
        "TCS": 4200,
        "WIPRO": 550,
        "LT": 2100,
        "ASIANPAINT": 3100
    }

    tick_count = 0

    try:
        while kafka_producer.is_connected():
            for symbol, base_price in stocks.items():
                tick = generate_mock_tick(symbol, base_price)
                success = kafka_producer.send_tick(
                    tick["instrument_token"],
                    tick
                )

                if success:
                    tick_count += 1
                    if tick_count % 100 == 0:
                        logger.info(f"Sent {tick_count} ticks")
                else:
                    logger.warning(f"Failed to send tick for {symbol}")

            time.sleep(0.5)  # Simulate 2 ticks/sec per stock

    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
    except Exception as e:
        logger.error(f"Producer error: {e}")
    finally:
        kafka_producer.close()
        logger.info(f"Producer stopped. Total ticks sent: {tick_count}")

if __name__ == "__main__":
    run_mock_producer()
