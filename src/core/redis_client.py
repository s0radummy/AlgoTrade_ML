import redis
from typing import Any, Optional
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RedisClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Redis connection with pooling."""
        try:
            self.pool = redis.ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                max_connections=10,
            )
            self.client = redis.Redis(connection_pool=self.pool)
            # Test connection
            self.client.ping()
            logger.info(f"Redis connected to {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """Get a value from Redis."""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis."""
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
        """Set a hash in Redis."""
        try:
            self.client.hset(key, mapping=mapping)
            if ttl:
                self.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting hash {key}: {e}")
            return False

    def hget(self, key: str, field: str) -> Optional[str]:
        """Get a field from a Redis hash."""
        try:
            return self.client.hget(key, field)
        except Exception as e:
            logger.error(f"Error getting hash field {key}:{field}: {e}")
            return None

    def hgetall(self, key: str) -> dict:
        """Get all fields from a Redis hash."""
        try:
            return self.client.hgetall(key)
        except Exception as e:
            logger.error(f"Error getting hash {key}: {e}")
            return {}

    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    def publish(self, channel: str, message: str) -> int:
        """Publish a message to a Redis channel."""
        try:
            return self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0

    def close(self):
        """Close Redis connection."""
        try:
            self.pool.disconnect()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

# Singleton instance
redis_client = RedisClient()
