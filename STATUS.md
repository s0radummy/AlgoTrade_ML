# AlgoTrading Project — Current Status

_Last updated: 2026-05-17_

---

## What Is Actually Done (Verified Working)

### 1. KiteConnect WebSocket — Live Market Data
- **Scripts:** `scripts/live_terminal.py`, `scripts/kite_to_kafka.py`
- **Auth:** Fully automated — logs in with `KITE_USER_ID` + `KITE_PASSWORD` + TOTP (generated from `KITE_TWO_FA` secret). No browser interaction. Session cached in `.kite_session.json` for the day; cache is validated against KiteConnect `profile()` before use.
- **Status:** Working. Tested live during market hours. Streams 50 Nifty stocks in `MODE_FULL`.

### 2. Kafka Producer + Consumer — Message Pipeline
- **Scripts:** `scripts/kafka_live_view.py` (mock data), `scripts/test_kafka.py` (round-trip diagnostic), `scripts/kite_to_kafka.py` (live bridge)
- **Infrastructure:** Docker Compose brings up `algotrade-kafka` + `algotrade-zookeeper`. Dual-listener: `kafka:9092` (container-to-container), `localhost:29092` (host scripts).
- **Live bridge:** `on_ticks` callback uses async `producer.send()` with callbacks — never blocks the Twisted reactor thread. Confirmed 49 snapshot ticks flowing on connect.
- **Status:** Working. Round-trip latency confirmed well under 100ms locally.

### 3. Redis Consumer — Live State
- **Script:** `scripts/verify_consumers.py` (Redis thread)
- **What it does:** Reads from `stock-quotes` Kafka topic, writes `STOCK:<SYMBOL>` hashes (LTP, OHLC, Volume, Change, Last_Updated) with 1-hour TTL.
- **Status:** Working. 49 snapshot ticks written and verified via `redis-cli HGETALL STOCK:RELIANCE`.

### 4. InfluxDB Consumer — Time-Series Persistence
- **Script:** `scripts/verify_consumers.py` (InfluxDB thread)
- **What it does:** Reads from `stock-quotes` Kafka topic, writes batched `Point` records to the `stocks` bucket using synchronous write API.
- **Status:** Working. 49 snapshot ticks written and verified via Flux query (`from(bucket:"stocks") |> range(start:-1h) |> count()`).

### 5. Unit Tests
- **`tests/test_tft_model.py`:** Tests TFT forward pass, output shapes `(batch, 5, 5)`, no NaN values, CPU inference. All pass.
- **`tests/test_validators.py`:** Tests valid tick acceptance, missing fields, negative price, outlier detection (>5% jump), batch validation. All pass.

### 6. Historical Data — KiteConnect API Fetch
- **Script:** `scripts/fetch_historical_data.py`
- **What it does:** Bulk-fetches up to 3 years of 1-minute OHLCV candles for Nifty 50 stocks via KiteConnect historical data API. Resumable (skips already-fetched data). Rate-limited to ≤3 req/sec. Saves one Parquet file per stock to `data/historical/`.
- **Result:** 13,060,164 candles saved across 47 stocks. 3 stocks not fetched: LTIM, TATAMOTORS, ZOMATO (likely symbol resolution failure during the run).
- **Status:** Completed. Data is on disk and excluded from git (per `.gitignore`).

### 7. TFT Model Architecture — Full Paper Implementation
- **File:** `src/models/tft_model.py`
- **What it does:** Full TFT from Lim et al. (2021). Implements GRN (Gated Residual Network), VSN (Variable Selection Network), LSTM encoder-decoder initialised from static context vectors, temporal self-attention, and a quantile regression output head.
- **Input:** `static_covariates (B,32)`, `past_inputs (B,60,7)`, `future_inputs (B,5,4)`
- **Output:** `(B, 5, 5)` — 5 quantiles (P10/P30/P50/P70/P90) for 5 future steps
- **Status:** Architecture complete, sanity-checked (forward pass, output shape, no NaN, parameter count).

### 8. Dataset — Sliding Window Loader
- **File:** `src/data/dataset.py`
- **What it does:** PyTorch `Dataset` over 1-minute OHLCV Parquet files. Filters to market hours (09:15–15:30 IST), detects session gaps (excludes cross-day windows), computes 7 log-return features and 4 cyclical calendar features, splits train/val by date (last 14 days = val).
- **Status:** Implemented and structurally verified. Not yet run against the actual 13M-candle dataset end-to-end in a benchmark.

### 9. TFT Model Training — First Run Completed
- **File:** `src/models/training.py`
- **What happened:** Training was run overnight on the 13M-candle dataset. The training script used pinball (quantile) loss, Adam optimizer, ReduceLROnPlateau scheduler, early stopping (patience=5).
- **Checkpoint:** `models/tft_v1.pth` — epoch 1, val_loss = 0.000180 (4.87 MB on disk, git-ignored).
- **Caveat:** Checkpoint was saved at epoch 1, meaning epoch 1 was the best validation loss achieved. Training may have triggered early stopping after epochs 2–6 without improvement — this is worth investigating. We do not currently know the training/validation loss curve beyond epoch 1's best.
- **Status:** A trained checkpoint exists. It is the first real run; we have not yet validated whether the model is learning meaningfully or overfitting.

---

## What Is Written and Structurally Complete (But Not End-to-End Tested)

These files are fully implemented and reviewed. They will work once all Docker services are running, but have not been run as part of a live pipeline yet.

| File | Purpose | Blocker |
|------|---------|---------|
| `src/core/redis_client.py` | Singleton Redis connection pool (10 connections, hset/get/pub-sub) | Crashes on import if Redis unavailable (singleton instantiated at module load) |
| `src/core/kafka_producer.py` | KafkaProducerService with exponential backoff reconnect | Crashes on import if Kafka unavailable (singleton instantiated at module load) |
| `src/consumers/inference_consumer.py` | Kafka → per-stock 60-tick deque → TFT tensor → Redis quantile cache | Cascading singleton deps (redis_client, model_manager, tick_validator, instrument_loader) |
| `src/consumers/viz_consumer.py` | Kafka → Redis STOCK:* hashes + Pub/Sub publish | Depends on redis_client singleton |
| `src/consumers/persistence_consumer.py` | Kafka → InfluxDB batch writes (1000-tick batches, manual offset commit) | Depends on influxdb_client; soft failure if unavailable |
| `src/data/instrument_loader.py` | Stock metadata cache with sector embeddings and market-cap normalization | Only 6 hardcoded stocks; placeholder for full KiteConnect instrument API |
| `src/data/tick_validator.py` | Required field checks, price range validation, per-stock outlier detection (5% threshold) | None — runs standalone |
| `src/models/model_manager.py` | Load/serve TFT weights, cache predictions in Redis, circuit-breaker fallback | Depends on redis_client; falls back to random model if weights missing |
| `src/api/app.py` | FastAPI: /health, /predict/{symbol}, /history/{symbol}, /stocks, /stats, /model/version | Cascading singleton deps; /history InfluxDB query not yet implemented |
| `src/utils/logger.py` | Structured JSON logging with rotating file handler (10MB, 7 backups) | None — runs standalone |
| `config/settings.py` | Pydantic BaseSettings loading all env vars from .env | None |

---

## What Does Not Exist Yet

- **Validated model** — `models/tft_v1.pth` exists but has not been evaluated. No directional accuracy, RMSE, or MAE numbers. No loss curve analysis. The best checkpoint is from epoch 1, which is suspicious.
- **3 missing stocks in historical data** — LTIM, TATAMOTORS, ZOMATO not fetched (47/50 stocks present). Reason unknown — likely a symbol resolution failure. Easy to re-fetch.
- **`instrument_loader.py` full stock list** — Currently hardcodes 6 stocks. Needs to be wired to KiteConnect's instrument API to serve all 50 Nifty stocks.
- **`/history` endpoint** — Stub exists in `src/api/app.py` but the InfluxDB Flux query is not implemented.
- **Dead Letter Queue** — `tick_validator.send_to_dlq()` logs to `logger.error` instead of publishing to an actual Kafka DLQ topic.
- **Docker images for producer/consumers** — `docker/Dockerfile.producer` and `docker/Dockerfile.consumer` referenced in `docker-compose.yml` have not been created or tested. The full `docker-compose up` stack (including producer and consumer services) has not been run.
- **Real inference pipeline** — `inference_consumer.py` exists but `hydrate_deque()` initializes empty (no InfluxDB backfill of 60-tick history on startup). Inference won't run until each stock accumulates ≥10 ticks post-connect.

---

## Critical Design Issues to Fix Before Full Pipeline

### 1. Module-Load Singletons (Most Urgent)
`redis_client`, `kafka_producer`, `model_manager`, and `instrument_loader` are all instantiated at the bottom of their modules. Any script that imports these crashes immediately if the backing service is unavailable — even for unrelated reasons. This makes the consumers fragile to start-order issues.

**Files affected:** `src/core/redis_client.py:109`, `src/core/kafka_producer.py:112`, `src/models/model_manager.py:161`, `src/data/instrument_loader.py:122`

**Fix:** Wrap in a lazy factory or move instantiation to `if __name__ == "__main__"` / application startup.

### 2. Kafka Offset Commit in Persistence Consumer
`persistence_consumer.py` sets `enable_auto_commit=False` but never calls `consumer.commit()` after a successful InfluxDB write. Messages will be re-processed on restart.

### 3. Outlier Threshold on First Run
`tick_validator.py` rejects ticks with >5% price change from the previous seen price. On the initial 60-tick warm-up window, stocks with high volatility at open could be silently dropped by the inference consumer.

---

## Environment

- **Python:** 3.12 (use `py` and `py -m pip`)
- **Kafka host port:** `localhost:29092`
- **Docker:** All four core services verified healthy:
  - `algotrade-kafka` ✅ healthy
  - `algotrade-zookeeper` ✅ up
  - `algotrade-redis` ✅ healthy
  - `algotrade-influxdb` ✅ healthy
- **KiteConnect credentials:** In `.env` (rotate before each trading session)
- **InfluxDB token:** `algotrade-dev-token-2024` (set via `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN`)

---

## How to Run (Current State)

### Live pipeline (market hours)
```
# Terminal 1 — KiteConnect → Kafka
py scripts/kite_to_kafka.py

# Terminal 2 — Kafka → Redis + InfluxDB (verification)
py scripts/verify_consumers.py
```

### Off-market testing (mock data)
```
# Terminal 1 — mock producer + consumer
py scripts/kafka_live_view.py

# Terminal 2 — Redis + InfluxDB verification
py scripts/verify_consumers.py
```

### Diagnostics
```
py scripts/test_kafka.py          # Kafka connectivity + round-trip test
py -m pytest tests/ -v            # Unit tests (TFT model + validators)
```

### Historical data fetch (run once, off-market hours)
```
py scripts/fetch_historical_data.py           # full Nifty 50
py scripts/fetch_historical_data.py --symbol LTIM   # re-fetch missing stocks
```

### TFT training (run off-market hours)
```
py -m src.models.training          # trains on data/historical/, saves models/tft_v1.pth
```

---

## Immediate Next Steps (Model Validation Phase)

1. **Evaluate the trained checkpoint** — run `scripts/evaluate_model.py` (to be created): compute directional accuracy, MAE, RMSE, and quantile calibration on the held-out 14-day val set. Inspect the loss curve to understand why epoch 1 was the best.
2. **Re-fetch missing stocks** — run `fetch_historical_data.py --symbol LTIM`, `--symbol TATAMOTORS`, `--symbol ZOMATO` to complete the 50-stock dataset.
3. **Retrain if needed** — based on evaluation results, decide whether to adjust architecture (hidden_dim, num_heads, dropout), training config (LR, batch_size), or feature engineering.
4. **Wire inference consumer** — connect `inference_consumer.py` to the live Kafka topic using the validated checkpoint; verify predictions appear in Redis under `PRED:<SYMBOL>:quantiles`.
5. **Build Docker images** — create `docker/Dockerfile.producer` and `docker/Dockerfile.consumer` so the full `docker-compose up` stack can run.
