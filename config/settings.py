from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # KiteConnect API
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_user_id: str = ""
    kite_password: str = ""
    kite_two_fa: str = ""

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "stock-quotes"
    kafka_partitions: int = 25

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # InfluxDB
    influxdb_host: str = "localhost"
    influxdb_port: int = 8086
    influxdb_database: str = "stock_data"
    influxdb_username: str = "admin"
    influxdb_password: str = "admin_password"
    influxdb_bucket: str = "stocks"
    influxdb_org: str = "algotrade"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "your_api_key_for_external_access"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Environment
    environment: str = "development"

    # Model
    model_path: str = "models/tft_v1.pth"
    model_cache_ttl: int = 2

    # Stocks
    stocks: str = "RELIANCE,INFY,TCS,WIPRO,LT,ASIANPAINT"

    # Market Hours
    market_open: str = "09:15"
    market_close: str = "15:30"
    pre_market_open: str = "09:00"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def stock_list(self) -> list:
        return [s.strip() for s in self.stocks.split(",")]

    @property
    def kafka_servers(self) -> list:
        return [s.strip() for s in self.kafka_bootstrap_servers.split(",")]

settings = Settings()
