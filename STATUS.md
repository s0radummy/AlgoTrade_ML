# AlgoTrading Project — Current Status

_Last updated: 2026-05-18_

---

## What Is Actually Done (Verified Working)

### 1. KiteConnect WebSocket — Live Market Data
- **Scripts:** `scripts/live_terminal.py`, `scripts/kite_to_kafka.py`
- **Auth:** Fully automated — logs in with `KITE_USER_ID` + `KITE_PASSWORD` + TOTP (generated from `KITE_TWO_FA` secret). No browser interaction. Session cached in `.kite_session.json` for the day; cache is validated against KiteConnect `profile()` before use.
- **Status:** Working. Tested live during market hours. Streams 50 Nifty stocks in `MODE_FULL`.

### 2. Kafka Producer + Consumer — Message Pipeline
- **Scripts:** `scripts/kafka_live_view.py` (mock data), `scripts/test_kafka.py` (round-trip diagnostic), `scripts/kite_to_kafka.py` (live bridge)
- **Infrastructure:** Docker Compose brings up `algotrade-kafka` + `algotrade-zookeeper`. Dual-listener: `kafka:9092` (container-to-container), `localhost:29092` (host scripts).
- **Live bridge:** `on_ticks` callback uses async `producer.send()` with callbacks — never blocks the Twisted reactor thread. Confirmed 50 stocks subscribed with snapshot burst on connect.
- **Measured RTT (2026-05-18, live market hours):** avg 27.6ms · p50 15.3ms · p99 28.8ms — well inside 100ms SLA.
- **Note:** On fresh Docker start, `stock-quotes` is auto-created with 1 partition. Must be manually altered to 25 before running: `docker exec algotrade-kafka kafka-topics --bootstrap-server localhost:9092 --alter --topic stock-quotes --partitions 25`
- **Status:** Working. 1576 ticks produced and confirmed in a 10-second live test.

### 3. Redis Consumer — Live State
- **Script:** `scripts/verify_consumers.py` (Redis thread)
- **What it does:** Reads from `stock-quotes` Kafka topic, writes `STOCK:<SYMBOL>` hashes (LTP, OHLC, Volume, Change, Last_Updated) with 1-hour TTL.
- **Status:** Working. 49 `STOCK:*` keys written and verified via `redis-cli KEYS "STOCK:*"` and `redis-cli HGETALL STOCK:RELIANCE` (2026-05-18).

### 4. InfluxDB Consumer — Time-Series Persistence
- **Script:** `scripts/verify_consumers.py` (InfluxDB thread)
- **What it does:** Reads from `stock-quotes` Kafka topic, writes batched `Point` records (batch size 50) to the `stocks` bucket using synchronous write API.
- **Bug fixed (2026-05-18, commit f809ecb):** Points were not setting `.time()`, so all points in a batch received the same nanosecond server write timestamp. Multiple ticks for the same stock within a batch were silently deduplicated by InfluxDB — only 22% of written points (411/1850) survived. Fix: each `Point` now uses the tick's own UTC timestamp from the KiteConnect payload. Survival rate is now 97% (1796/1850); the remaining 3% are genuine same-microsecond duplicate ticks.
- **Status:** Working. 1796 records confirmed in InfluxDB after a 1576-message replay (2026-05-18).

### 5. Unit Tests
- **`tests/test_tft_model.py`:** Tests TFT forward pass, output shapes `(batch, 5, 5)`, no NaN values, CPU inference. Uses `past_inputs (batch, 60, 11)`. All 3 pass.
- **`tests/test_validators.py`:** Tests valid tick acceptance, missing fields, negative price, outlier detection (>5% jump), batch validation. All pass.

### 6. Historical Data — KiteConnect API Fetch
- **Script:** `scripts/fetch_historical_data.py`
- **What it does:** Bulk-fetches up to 3 years of 1-minute OHLCV candles for Nifty 50 stocks via KiteConnect historical data API. Resumable (skips already-fetched data). Rate-limited to ≤3 req/sec. Saves one Parquet file per stock to `data/historical/`. Also supports index symbols (NIFTY50) via hardcoded instrument tokens.
- **Result:** 13,060,164 candles saved across 47 stocks. 3 stocks not fetched: LTIM, TATAMOTORS, ZOMATO (likely symbol resolution failure). NIFTY50 index must be fetched separately (see Next Steps).
- **Status:** Completed for equities. NIFTY50 index parquet pending.

### 7. TFT Model Architecture — Full Paper Implementation
- **File:** `src/models/tft_model.py`
- **What it does:** Full TFT from Lim et al. (2021). Implements GRN (Gated Residual Network), VSN (Variable Selection Network), LSTM encoder-decoder initialised from static context vectors, temporal self-attention, and a quantile regression output head.
- **Input:** `static_covariates (B,32)`, `past_inputs (B,60,11)`, `future_inputs (B,5,4)`
- **Output:** `(B, 5, 5)` — 5 quantiles (P10/P30/P50/P70/P90) for 5 future steps
- **past_inputs col layout:**
  - 0: log_ret (z-scored per stock)
  - 1: open_ret, 2: high_ret, 3: low_ret, 4: intraday_ret, 5: intraday_rng
  - 6: vol_norm = log1p(vol / mean_vol)
  - 7: RSI(14) — [0,1] Wilder smoothing
  - 8: MACD histogram (12,26,9) normalized by close
  - 9: ATR(14) normalized by close
  - 10: Nifty50 1-min log return (raw, not z-scored)
- **Status:** Architecture complete, sanity-checked (forward pass, output shape (4,5,5), no NaN, 462,273 parameters).

### 8. Dataset — Sliding Window Loader
- **File:** `src/data/dataset.py`
- **What it does:** PyTorch `Dataset` over 1-minute OHLCV Parquet files. Filters to market hours (09:15–15:30 IST = 03:45–10:00 UTC), detects session gaps (excludes cross-day windows), computes 11 features per candle (7 OHLCV log-returns + RSI + MACD hist + ATR + Nifty50 return), and 4 cyclical calendar features. Splits train/val by date (last 14 days = val by default).
- **Target standardization:** log_ret targets and col-0 of past_inputs are z-scored per stock to eliminate the degenerate pinball-loss minimum. Per-stock stats saved in checkpoint.
- **Static covariates (32 dims):** stock identity hash, sector ordinal + one-hot (10 dims), log-normalized market cap, historical volatility (dim 13), market-cap bucket one-hot (8 dims). 22/32 dims are meaningful.
- **Walk-forward support:** `val_start_days` parameter restricts val to a specific window `[series_end - val_start_days, series_end - val_days]` for non-overlapping evaluation folds.
- **Nifty50 feature:** requires `data/historical/NIFTY50.parquet`. Raises `FileNotFoundError` if missing (fail loud, not silently zero).
- **Status:** Implemented and structurally verified. Pending full run with NIFTY50.parquet.

### 9. TFT Model Training — Diagnostic Run Completed; Full Run Pending
- **File:** `src/models/training.py`
- **Key features:** CosineAnnealingLR (T_max=max_epochs, eta_min=1e-6), p50_std collapse monitoring per epoch, loss curve CSV at `models/tft_v1_loss_curve.csv`, warm-start from checkpoint (RuntimeError fallback on shape mismatch), diagnostic mode (`--diagnostic`: 3 stocks, 10 epochs, batch=256).
- **Round 1 diagnostic results (2026-05-18, 3 stocks, 10 epochs, lr=3e-4):**
  - Best val: 0.221698 at epoch 6. Val improved every epoch 1–6, then plateaued.
  - Train loss: 0.18273 → 0.18229 (steady, healthy decline over 10 epochs).
  - p50_std: 0.001–0.006 — still low, but model is learning (not collapsed to constant).
  - Gradient norms: 0.01–0.05 — healthy throughout.
- **Current checkpoint:** `models/tft_v1.pth` (epoch 6, val=0.221698, 47 target_stats). Git-ignored.
- **Full 47-stock run status:** Was started (`--epochs 25 --lr 3e-4 --batch_size 512`) but killed after epoch 1 batch ~7500 to incorporate Round 2 architecture upgrade (Nifty50 feature). Must be restarted after fetching NIFTY50.parquet.
- **Status:** Checkpoint exists from diagnostic. Needs NIFTY50 data fetch + full retrain.

### 10. Inference Consumer — 1-Minute Bar Accumulator
- **File:** `src/consumers/inference_consumer.py`
- **Previous state:** Used a raw-tick deque (60 raw ticks ≈ 15–20 seconds) — severe train/inference mismatch (training uses 60 one-minute bars = 60 minutes).
- **Current state:** `BarAccumulator` class converts raw KiteConnect ticks into completed 1-minute OHLCV bars using cumulative volume deltas. Inference runs only when ≥60 completed bars exist (60-minute lookback), matching training exactly.
- **Feature parity:** Bar-derived features (cols 0–10) match `dataset.py _load_stock` exactly: log-return OHLCV, RSI/MACD/ATR computed on bar closes/highs/lows, Nifty50 log return from shared bar accumulator.
- **Nifty50 routing:** Ticks with `instrument_token == 256265` (NIFTY50 NSE index) are routed to a shared `nifty_acc` accumulator; its bars provide col 10 for all stock inferences.
- **Status:** Implemented. Awaiting live test (requires model retrain with Nifty50 feature).

---

## What Is Written and Structurally Complete (But Not End-to-End Tested)

These files are fully implemented and reviewed. They will work once all Docker services are running, but have not been run as part of a live pipeline yet.

| File | Purpose | Blocker |
|------|---------|---------|
| `src/core/redis_client.py` | Singleton Redis connection pool (10 connections, hset/get/pub-sub) | Crashes on import if Redis unavailable (singleton instantiated at module load) |
| `src/core/kafka_producer.py` | KafkaProducerService with exponential backoff reconnect | Crashes on import if Kafka unavailable (singleton instantiated at module load) |
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

- **NIFTY50.parquet** — Required before next training run. Fetch with: `py scripts/fetch_historical_data.py --symbol NIFTY50`
- **Retrained model** — Current `models/tft_v1.pth` is from a 3-stock diagnostic (10 epochs). The 47-stock full training (25 epochs, past_dim=11) has not yet completed.
- **Walk-forward evaluation results** — `scripts/evaluate_model.py --walk_forward 6` exists but hasn't been run against a fully-trained checkpoint yet.
- **3 missing equity stocks** — LTIM, TATAMOTORS, ZOMATO not fetched (47/50 present). Re-fetch with `--symbol` flag.
- **`instrument_loader.py` full stock list** — Currently hardcodes 6 stocks. Needs to be wired to KiteConnect's instrument API.
- **`/history` endpoint** — Stub exists in `src/api/app.py` but the InfluxDB Flux query is not implemented.
- **Dead Letter Queue** — `tick_validator.send_to_dlq()` logs to `logger.error` instead of publishing to an actual Kafka DLQ topic.
- **Docker images for producer/consumers** — `docker/Dockerfile.producer` and `docker/Dockerfile.consumer` referenced in `docker-compose.yml` have not been created or tested.

---

## Critical Design Issues to Fix Before Full Pipeline

### 1. Module-Load Singletons (Most Urgent)
`redis_client`, `kafka_producer`, `model_manager`, and `instrument_loader` are all instantiated at the bottom of their modules. Any script that imports these crashes immediately if the backing service is unavailable — even for unrelated reasons.

**Files affected:** `src/core/redis_client.py:109`, `src/core/kafka_producer.py:112`, `src/models/model_manager.py:161`, `src/data/instrument_loader.py:122`

**Fix:** Wrap in a lazy factory or move instantiation to `if __name__ == "__main__"` / application startup.

### 2. Kafka Offset Commit in Persistence Consumer
`persistence_consumer.py` sets `enable_auto_commit=False` but never calls `consumer.commit()` after a successful InfluxDB write. Messages will be re-processed on restart.

### 3. Kafka Topic Auto-Created with Wrong Partition Count
When Docker starts fresh and a producer connects before the topic is manually created, Kafka auto-creates `stock-quotes` with 1 partition (broker default). All ticks then land on partition 0, breaking the per-stock ordering guarantee.

**Workaround:** Add topic pre-creation to a startup script or docker-compose `command` hook.

### 4. Outlier Threshold on First Run
`tick_validator.py` rejects ticks with >5% price change from the previous seen price. On the initial warm-up window (before 60 bars accumulate), high-volatility stocks at open could be silently dropped by the inference consumer.

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
py scripts/fetch_historical_data.py                  # full Nifty 50
py scripts/fetch_historical_data.py --symbol NIFTY50 # Nifty50 index (required for training)
py scripts/fetch_historical_data.py --symbol LTIM    # re-fetch missing stocks
```

### TFT training (run off-market hours)
```
# Diagnostic first (3 stocks, ~35 min)
py -m src.models.training --diagnostic --epochs 10 --lr 3e-4 --batch_size 256

# Full training (47 stocks, ~46 hrs at current GPU speed)
py -m src.models.training --epochs 25 --lr 3e-4 --batch_size 512
```

### Model evaluation
```
# Standard val set (last 14 days)
py scripts/evaluate_model.py --checkpoint models/tft_v1.pth

# Walk-forward (6 consecutive 14-day windows — checks consistency across regimes)
py scripts/evaluate_model.py --walk_forward 6
```

---

## Model Evaluation — Historical Findings (2026-05-17, pre-fix checkpoint)

This section records what we learned from the first evaluation run before the collapse fix.
Kept as a reference baseline — do not delete.

### Hard Numbers (Measured)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Directional accuracy (P50) | 50.20% | Statistically indistinguishable from coin flip |
| MAE (P50 vs actual) | 0.000471 | Equal to predicting zero every time |
| RMSE (P50 vs actual) | 0.000688 | Matches actual log-return std — no better than naive mean |
| **P50 prediction std** | **0.000008** | **86x less variable than actual data — model collapsed to near-constant** |
| Val windows evaluated | 146,170 | 47 stocks, last 14 days |
| Best checkpoint epoch | 1 | Never improved past epoch 1 |

**Root cause confirmed:** The pinball loss has a degenerate minimum for near-zero-mean targets. Model learned to predict ≈0 for P50, which technically minimizes the loss. Fixed by: (a) per-stock z-scoring of targets so the model must predict deviations from mean, and (b) adding RSI/MACD/ATR to give regime context, and (c) Nifty50 return to separate idiosyncratic from market-wide moves.

---

## Immediate Next Steps

1. **Fetch Nifty50 index data** (required before retraining):
   ```
   py scripts/fetch_historical_data.py --symbol NIFTY50
   ```

2. **Run diagnostic** (3 stocks, 10 epochs, ~35 min — verify learning before committing to full run):
   ```
   py -m src.models.training --diagnostic --epochs 10 --lr 3e-4 --batch_size 256
   ```

3. **Full training** (47 stocks, 25 epochs, ~46 hrs):
   ```
   py -m src.models.training --epochs 25 --lr 3e-4 --batch_size 512
   ```

4. **Walk-forward evaluation** (after training completes):
   ```
   py scripts/evaluate_model.py --walk_forward 6
   ```

5. **Wire live inference** — run KiteConnect producer + inference consumer together, verify `PRED:<SYMBOL>:quantiles` keys appear in Redis within 60 minutes of market open (60-bar warmup period).
