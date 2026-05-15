import json
import logging
import logging.handlers
import os
from datetime import datetime
from config.settings import settings

def setup_logger(name: str) -> logging.Logger:
    """Setup structured JSON logging for a module."""
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, settings.log_level))

    # Create logs directory if it doesn't exist
    os.makedirs(settings.log_dir, exist_ok=True)

    # Create rotating file handler
    log_file = os.path.join(settings.log_dir, f"{name.replace('.', '_')}.log")
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=7,
    )

    # JSON formatter
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)

    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Also add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
