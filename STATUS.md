# AlgoTrading Project — Current Status

_Last updated: 2026-05-16_

---

## What Is Actually Done (Verified Working)

### 1. KiteConnect WebSocket — Live Market Data
- **Script:** `scripts/live_terminal.py`
- **What it does:** Authenticates with KiteConnect via browser OAuth + TOTP, subscribes to 50 Nifty 50 stocks, and streams live tick data (LTP, OHLCV, volume, % change) to the terminal in real time.
- **Auth flow:** First run opens a browser login; the session token is cached in `.kite_session.json` for the rest of the trading day so subsequent runs skip the login step.
- **Status:** Working. Tested live during market hours. Receives ticks for all 50 stocks listed in `.env`.

### 2. Kafka Producer + Consumer — Message Pipeline
- **Scripts:** `scripts/kafka_live_view.py` (combined view), `scripts/test_kafka.py` (round-trip test)
- **What it does:** A mock producer generates synthetic OHLCV ticks for 6 stocks and publishes them to the `stock-quotes` Kafka topic. A consumer in the same process reads them back. Both are visible in one terminal with color-coded output and round-trip latency per message.
- **Infrastructure:** Docker Compose brings up `algotrade-kafka` + `algotrade-zookeeper`. Kafka is configured with a dual-listener setup — `kafka:9092` for internal container-to-container traffic, `localhost:29092` for scripts running on the host machine.
- **Status:** Working. Round-trip latency confirmed well under 100ms locally.

---

## What Is Written But Not Yet Tested

These files exist and are structurally complete, but have not been run end-to-end:

| File | Purpose |
|------|---------|
| `src/core/kafka_producer.py` | KafkaProducerService class (used by real pipeline, not yet wired to KiteConnect) |
| `src/core/redis_client.py` | Redis connection singleton |
| `src/consumers/inference_consumer.py` | Kafka → TFT model → Redis predictions |
| `src/consumers/viz_consumer.py` | Kafka → Redis live state for UI |
| `src/consumers/persistence_consumer.py` | Kafka → InfluxDB batch writes |
| `src/data/tick_validator.py` | Tick validation + outlier detection + DLQ |
| `src/data/instrument_loader.py` | Stock metadata cache |
| `src/models/tft_model.py` | PyTorch TFT architecture (designed, not trained) |
| `src/models/model_manager.py` | Model loading, inference, Redis caching |
| `src/models/training.py` | Offline training pipeline |
| `src/api/app.py` | FastAPI endpoints (/predict, /history, /health) |

---

## What Does Not Exist Yet

- No trained TFT model weights (`models/tft_v1.pth` does not exist)
- KiteConnect WebSocket is not wired into `src/core/kafka_producer.py` — the live terminal script and the Kafka producer are two separate, unconnected scripts right now
- Docker images for producer/consumers have not been built or tested
- Redis and InfluxDB containers have not been started or verified
- No unit or integration tests have been run

---

## Environment

- **Python:** 3.12 (use `py` and `py -m pip`, not `python`/`pip` — those aren't in PATH)
- **Kafka host port:** `localhost:29092` (changed from 9092 — updated in `.env`)
- **Docker:** Installed and working
- **KiteConnect credentials:** In `.env` (rotate before each trading session)

---

## The Immediate Next Step

Wire the KiteConnect WebSocket tick stream into the Kafka producer so live market data flows into the `stock-quotes` topic. That connects the two verified pieces and makes the pipeline real.

```
live_terminal.py (working)          kafka_live_view.py (working)
       ↓                    →              ↓
  KiteConnect ticks       BRIDGE     Kafka producer → consumer
```

The bridge is `src/core/kafka_producer.py` — it just needs to be called from inside the `on_ticks` callback in the WebSocket script.
