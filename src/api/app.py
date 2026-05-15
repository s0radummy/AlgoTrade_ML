from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from datetime import datetime
import signal

from config.settings import settings
from src.utils.logger import setup_logger
from src.core.redis_client import redis_client
from src.models.model_manager import model_manager
from src.data.instrument_loader import instrument_loader

logger = setup_logger(__name__)

app = FastAPI(
    title="AlgoTrading API",
    description="Real-time stock prediction API",
    version="1.0.0"
)

# Middleware for authentication
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_status = redis_client.client.ping()

        return {
            "status": "healthy" if redis_status else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "redis": "connected" if redis_status else "disconnected",
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/model/version")
async def model_version(x_api_key: str = Header(None)) -> Dict[str, Any]:
    """Get model metadata."""
    try:
        verify_api_key(x_api_key)
        model_info = model_manager.get_model_info()
        return {
            "version": model_info["version"],
            "device": model_info["device"],
            "parameters": model_info["parameters"],
            "metadata": model_info["metadata"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{symbol}")
async def predict(symbol: str, x_api_key: str = Header(None)) -> Dict[str, Any]:
    """
    Get latest prediction for a symbol.
    Returns quantile forecasts (P10, P50, P90).
    """
    try:
        verify_api_key(x_api_key)

        # Get cached prediction from Redis
        prediction = redis_client.hgetall(f"PRED:{symbol}:quantiles")
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction found for symbol {symbol}"
            )

        # Get current stock state
        stock_state = redis_client.hgetall(f"STOCK:{symbol}")
        current_price = float(stock_state.get("LTP", 0))

        return {
            "symbol": symbol,
            "timestamp": prediction.get("timestamp"),
            "current_price": current_price,
            "quantile_forecast": {
                "P10": float(prediction.get("P10", 0)),
                "P30": float(prediction.get("P30", 0)),
                "P50": float(prediction.get("P50", 0)),
                "P70": float(prediction.get("P70", 0)),
                "P90": float(prediction.get("P90", 0)),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{symbol}")
async def history(
    symbol: str,
    hours: Optional[int] = 24,
    x_api_key: str = Header(None)
) -> Dict[str, Any]:
    """
    Get historical data and predictions for a symbol.
    In production, queries InfluxDB.
    """
    try:
        verify_api_key(x_api_key)

        # Get current state
        stock_state = redis_client.hgetall(f"STOCK:{symbol}")
        prediction = redis_client.hgetall(f"PRED:{symbol}:quantiles")

        if not stock_state:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {symbol}"
            )

        return {
            "symbol": symbol,
            "hours": hours,
            "current_state": stock_state,
            "latest_prediction": prediction if prediction else {},
            "note": "Full historical data from InfluxDB to be implemented"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks")
async def list_stocks(x_api_key: str = Header(None)) -> Dict[str, Any]:
    """List all tracked stocks."""
    try:
        verify_api_key(x_api_key)
        instruments = instrument_loader.get_all_instruments()
        return {
            "stocks": list(instruments.keys()),
            "count": len(instruments),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(x_api_key: str = Header(None)) -> Dict[str, Any]:
    """Get system statistics."""
    try:
        verify_api_key(x_api_key)

        # Get Redis memory
        try:
            info = redis_client.client.info("memory")
            redis_memory = info.get("used_memory_human", "N/A")
        except:
            redis_memory = "N/A"

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "redis_memory": redis_memory,
            "model_info": model_manager.get_model_info(),
            "stocks_tracked": len(instrument_loader.get_all_instruments()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("AlgoTrading API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("AlgoTrading API shutting down...")
    redis_client.close()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
    )
