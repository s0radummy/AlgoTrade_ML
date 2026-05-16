import json
from typing import Dict, Optional, List
from config.settings import settings
from src.utils.logger import setup_logger
from src.core.redis_client import redis_client

logger = setup_logger(__name__)

class InstrumentLoader:
    """Load and cache instrument metadata (Symbol, Sector, Market Cap)."""

    def __init__(self):
        self.cache = {}
        self.load_default_instruments()

    def load_default_instruments(self):
        """Load default instrument metadata."""
        # This is a placeholder - in production, this would come from KiteConnect API
        default_instruments = {
            "RELIANCE": {
                "symbol": "RELIANCE",
                "token": 738561,
                "sector": "Energy",
                "market_cap": 15000000000000,  # 15T INR
                "lot_size": 1
            },
            "INFY": {
                "symbol": "INFY",
                "token": 1346649,
                "sector": "IT",
                "market_cap": 8500000000000,  # 8.5T INR
                "lot_size": 1
            },
            "TCS": {
                "symbol": "TCS",
                "token": 1314561,
                "sector": "IT",
                "market_cap": 13200000000000,  # 13.2T INR
                "lot_size": 1
            },
            "WIPRO": {
                "symbol": "WIPRO",
                "token": 1769169,
                "sector": "IT",
                "market_cap": 3200000000000,  # 3.2T INR
                "lot_size": 1
            },
            "LT": {
                "symbol": "LT",
                "token": 1064961,
                "sector": "Industrials",
                "market_cap": 2800000000000,  # 2.8T INR
                "lot_size": 1
            },
            "ASIANPAINT": {
                "symbol": "ASIANPAINT",
                "token": 855169,
                "sector": "Consumer",
                "market_cap": 2400000000000,  # 2.4T INR
                "lot_size": 1
            }
        }

        self.cache = default_instruments
        self._cache_to_redis()
        logger.info(f"Loaded {len(self.cache)} default instruments")

    def _cache_to_redis(self):
        """Cache instruments to Redis (best-effort — Redis may not be up yet)."""
        try:
            for symbol, data in self.cache.items():
                redis_client.set(
                    f"INSTRUMENT:{symbol}",
                    json.dumps(data),
                    ttl=86400  # 1 day
                )
        except Exception as e:
            logger.warning(f"Could not cache instruments to Redis: {e} — will retry on first access")

    def get_instrument(self, symbol: str) -> Optional[Dict]:
        """Get instrument metadata by symbol."""
        if symbol not in self.cache:
            # Try to load from Redis
            cached = redis_client.get(f"INSTRUMENT:{symbol}")
            if cached:
                self.cache[symbol] = json.loads(cached)
                return self.cache[symbol]
            logger.warning(f"Instrument {symbol} not found")
            return None
        return self.cache[symbol]

    def get_all_instruments(self) -> Dict:
        """Get all cached instruments."""
        return self.cache

    def get_sector_embedding(self, sector: str) -> int:
        """Get sector embedding index (for TFT model)."""
        sector_map = {
            "IT": 0,
            "Energy": 1,
            "Industrials": 2,
            "Consumer": 3,
            "Finance": 4,
            "Healthcare": 5,
            "Utilities": 6,
            "Telecom": 7,
        }
        return sector_map.get(sector, 8)  # 8 for unknown

    def normalize_market_cap(self, market_cap: float) -> float:
        """Normalize market cap to [-1, 1] range."""
        # Assume min: 100M, max: 100T INR
        min_cap = 100_000_000  # 100M
        max_cap = 100_000_000_000_000  # 100T

        import math
        log_cap = math.log10(market_cap)
        log_min = math.log10(min_cap)
        log_max = math.log10(max_cap)

        normalized = 2 * ((log_cap - log_min) / (log_max - log_min)) - 1
        return max(-1, min(1, normalized))

# Singleton instance
instrument_loader = InstrumentLoader()
