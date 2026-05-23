import torch
import json
import os
from typing import Dict, Optional, Tuple
from datetime import datetime
from config.settings import settings
from src.utils.logger import setup_logger
from src.core.redis_client import redis_client
from src.models.tft_model import TemporalFusionTransformer

logger = setup_logger(__name__)

class ModelManager:
    """Load, cache, and serve TFT model predictions."""

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_metadata = {}
        self.model_version = "v1"
        self.cache_ttl = settings.model_cache_ttl
        self._model_loaded = False

    def _ensure_model_loaded(self):
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True

    def load_model(self) -> bool:
        """Load TFT model from checkpoint."""
        try:
            if not os.path.exists(settings.model_path):
                logger.warning(f"Model not found at {settings.model_path}. Using random model for dev.")
                self.model = TemporalFusionTransformer()
                self.model.to(self.device)
                self.model.eval()
                return False

            checkpoint = torch.load(settings.model_path, map_location=self.device)

            # Load model state
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model = TemporalFusionTransformer()
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model_metadata = checkpoint.get("metadata", {})
            else:
                self.model = checkpoint

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {settings.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to random model for development
            self.model = TemporalFusionTransformer()
            self.model.to(self.device)
            self.model.eval()
            return False

    def preprocess_inputs(
        self,
        static_covariates: torch.Tensor,
        past_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess inputs to model format."""
        static_covariates = static_covariates.to(self.device)
        past_inputs = past_inputs.to(self.device)
        future_inputs = future_inputs.to(self.device)
        return static_covariates, past_inputs, future_inputs

    def predict(
        self,
        static_covariates: torch.Tensor,
        past_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Get predictions from model."""
        self._ensure_model_loaded()
        try:
            with torch.no_grad():
                static_covariates, past_inputs, future_inputs = self.preprocess_inputs(
                    static_covariates, past_inputs, future_inputs
                )
                output = self.model(static_covariates, past_inputs, future_inputs)
            return output

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def cache_prediction(
        self, symbol: str, prediction: Dict, ttl: Optional[int] = None
    ) -> bool:
        """Cache prediction in Redis."""
        try:
            cache_key = f"PRED:{symbol}:quantiles"
            ttl = ttl or self.cache_ttl

            quantiles_dict = {
                "P10": float(prediction.get("P10", 0)),
                "P30": float(prediction.get("P30", 0)),
                "P50": float(prediction.get("P50", 0)),
                "P70": float(prediction.get("P70", 0)),
                "P90": float(prediction.get("P90", 0)),
                "timestamp": datetime.utcnow().isoformat(),
            }

            redis_client.hset(cache_key, quantiles_dict, ttl=ttl)
            return True

        except Exception as e:
            logger.error(f"Failed to cache prediction: {e}")
            return False

    def get_cached_prediction(self, symbol: str) -> Optional[Dict]:
        """Retrieve cached prediction from Redis."""
        try:
            cache_key = f"PRED:{symbol}:quantiles"
            prediction = redis_client.hgetall(cache_key)
            return prediction if prediction else None

        except Exception as e:
            logger.error(f"Failed to get cached prediction: {e}")
            return None

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> bool:
        """Save model checkpoint."""
        self._ensure_model_loaded()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "metadata": metadata or {
                    "saved_at": datetime.utcnow().isoformat(),
                    "version": self.model_version,
                },
            }

            torch.save(checkpoint, path)
            logger.info(f"Model checkpoint saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get model information."""
        self._ensure_model_loaded()
        return {
            "version": self.model_version,
            "device": str(self.device),
            "metadata": self.model_metadata,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def circuit_breaker_fallback(self, symbol: str) -> Optional[Dict]:
        """Fallback to cached prediction if inference fails."""
        logger.warning(f"Using circuit breaker fallback for {symbol}")
        return self.get_cached_prediction(symbol)

# Singleton instance
model_manager = ModelManager()
