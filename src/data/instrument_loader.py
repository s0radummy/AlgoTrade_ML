import json
from typing import Dict, Optional, List
from config.settings import settings
from src.utils.logger import setup_logger
from src.core.redis_client import redis_client
from src.data.dataset import STOCK_META

logger = setup_logger(__name__)

class InstrumentLoader:
    """Load and cache instrument metadata (Symbol, Sector, Market Cap)."""

    def __init__(self):
        self.cache = {}
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_default_instruments()
            self._loaded = True

    def load_default_instruments(self):
        """Load instrument metadata from STOCK_META for all tracked stocks."""
        new_instruments = {}
        for symbol in settings.stock_list:
            if symbol not in STOCK_META:
                logger.warning(f"Symbol {symbol} not in STOCK_META — skipping")
                continue
            sector, market_cap = STOCK_META[symbol]
            new_instruments[symbol] = {
                "symbol": symbol,
                "token": 0,  # placeholder — real token from KiteConnect API
                "sector": sector,
                "market_cap": market_cap,
                "lot_size": 1,
            }
        self.cache = new_instruments
        self._cache_to_redis()
        logger.info(f"Loaded {len(self.cache)} instruments from STOCK_META")

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
        self._ensure_loaded()
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
        self._ensure_loaded()
        return self.cache

    def get_sector_embedding(self, sector: str) -> int:
        """Get sector embedding index (for TFT model)."""
        sector_map = {
            "IT": 0, "Finance": 1, "Energy": 2, "Auto": 3,
            "Consumer": 4, "Healthcare": 5, "Industrials": 6,
            "Materials": 7, "Telecom": 8, "Utilities": 9,
        }
        return sector_map.get(sector, 10)  # 10 for unknown

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
