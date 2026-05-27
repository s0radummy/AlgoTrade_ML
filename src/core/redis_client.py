import redis
import threading
from typing import Optional
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RedisClient:
    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._connected = False
                    instance._conn_lock = threading.Lock()
                    instance.pool = None
                    instance.client = None
                    cls._instance = instance
        return cls._instance

    def _ensure_connected(self):
        """Connect on first use — safe to call from any thread."""
        if not self._connected:
            with self._conn_lock:
                if not self._connected:
                    self._initialize()
                    self._connected = True

    def _initialize(self):
        self.pool = redis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
            max_connections=10,
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self.client.ping()
        logger.info(f"Redis connected to {settings.redis_host}:{settings.redis_port}")

    def get(self, key: str) -> Optional[str]:
        self._ensure_connected()
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        self._ensure_connected()
        try:
            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def hset(self, key: str, mapping: dict, ttl: Optional[int] = None) -> bool:
        self._ensure_connected()
        try:
            self.client.hset(key, mapping=mapping)
            if ttl:
                self.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting hash {key}: {e}")
            return False

    def hget(self, key: str, field: str) -> Optional[str]:
        self._ensure_connected()
        try:
            return self.client.hget(key, field)
        except Exception as e:
            logger.error(f"Error getting hash field {key}:{field}: {e}")
            return None

    def hgetall(self, key: str) -> dict:
        self._ensure_connected()
        try:
            return self.client.hgetall(key)
        except Exception as e:
            logger.error(f"Error getting hash {key}: {e}")
            return {}

    def delete(self, key: str) -> bool:
        self._ensure_connected()
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    def lpush(self, key: str, *values: str) -> int:
        self._ensure_connected()
        try:
            return self.client.lpush(key, *values)
        except Exception as e:
            logger.error(f"Error lpush on key {key}: {e}")
            return 0

    def ltrim(self, key: str, start: int, end: int) -> bool:
        self._ensure_connected()
        try:
            self.client.ltrim(key, start, end)
            return True
        except Exception as e:
            logger.error(f"Error ltrim on key {key}: {e}")
            return False

    def lrange(self, key: str, start: int, end: int) -> list:
        self._ensure_connected()
        try:
            return self.client.lrange(key, start, end)
        except Exception as e:
            logger.error(f"Error lrange on key {key}: {e}")
            return []

    def lindex(self, key: str, index: int) -> Optional[str]:
        self._ensure_connected()
        try:
            return self.client.lindex(key, index)
        except Exception as e:
            logger.error(f"Error lindex on key {key}: {e}")
            return None

    def publish(self, channel: str, message: str) -> int:
        self._ensure_connected()
        try:
            return self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0

    def close(self):
        if self._connected and self.pool:
            try:
                self.pool.disconnect()
                self._connected = False
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")


# Singleton instance — safe to import without Redis running
redis_client = RedisClient()
