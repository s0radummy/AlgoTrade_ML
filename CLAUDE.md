# AlgoTrading: Real-Time Predictive Stock Engine

## Project Overview

Building a real-time stock price prediction system that ingests live market data via KiteConnect WebSocket, processes it through Kafka, and powers three parallel consumer systems: a Temporal Fusion Transformer (TFT) inference engine, real-time visualization dashboard, and time-series persistence layer.

### Project Constraints

- **Scale**: Small (100 stocks max, <1K ticks/sec)
- **Latency SLA**: 100-500ms end-to-end (intraday trading)
- **Deployment**: On-premise only, single machine via Docker Compose
- **Timeline**: 6-week implementation plan (sequential phases)
- **Model Status**: PyTorch TFT architecture designed but not yet implemented

---

## Architecture Overview

### Data Flow

```
KiteConnect API (WebSocket)
    ↓
[Kafka Producer] — logs 7 variables (LTP, OHLCV, Volume, Change)
    ↓ (binary → JSON → UTF-8 bytes)
[Kafka Broker] — stock-quotes topic, 25 partitions (2:1 stock-to-partition ratio)
    ↓ (Fan-out via 3 Consumer Groups)
    ├→ [Inference Consumer] → TFT Model → Redis predictions
    ├→ [Viz Consumer] → Redis state aggregation (UI polling/Pub-Sub)
    └→ [Persistence Consumer] → InfluxDB time-series writes
```

### Services

- **Kafka**: Message broker (25 partitions for strict per-stock ordering)
- **Redis**: Shared state (predictions, live ticks)
- **InfluxDB**: Time-series database (historical tick persistence)
- **API**: FastAPI wrapper for external prediction queries
- **Docker Compose**: Orchestrates all services locally

---

## TFT Model Specification

### Input Architecture

1. **Static Covariates** (Per Stock)
   - Stock ID embedding (learned)
   - Sector embedding (8-16 dims, categorical)
   - Market Cap (normalized to [-1, 1])
   - Total: ~24-32 dimensions

2. **Past Inputs** (Time Series, 60-tick window)
   - Log Returns: `log(P_t / P_t-1)` for stationarity
   - Volume (normalized per stock)
   - OHLCV features
   - Sequence length: 60 ticks
   - Per-tick features: ~5-7
   - Tensor shape: (batch, 60, 7)

3. **Known Future Inputs** (Calendar/Time Features)
   - Hour of day: 0-23 (cyclical encoding)
   - Day of week: 0-4
   - Market phase: pre-market/trading/post-market
   - Day of month
   - Total: ~4-6 dimensions

### Output Specification

- **Prediction Horizon**: Multi-step forward (5 ticks ahead)
- **Target Variable**: Log returns (not raw prices)
- **Output Format**: 5-quantile forecasts (P10, P30, P50, P70, P90) per future tick
- **Output Shape**: (batch, N_steps, 5) where N_steps=5
- **Loss Function**: Quantile loss (pinball loss) for multi-quantile regression
- **Retraining**: Daily overnight batch training; inference uses latest weights

### Model Architecture Components

- Embedding layers for static covariates
- Temporal Fusion Transformer blocks (multi-head attention)
- Variable selection networks for feature importance
- Quantile regression head (output 5 quantiles per step)
- Adam optimizer, learning rate scheduler
- Validation: Hold-out test set (last 2 weeks of historical data)

---

## Critical Implementation Sequence

### Phase 0: TFT Model (Week 0)

- [ ] PyTorch TFT architecture implementation
- [ ] Training pipeline (data loader, preprocessing, training loop)
- [ ] Model checkpointing (best model by validation loss)
- [ ] ModelManager class (loading, inference, caching in Redis)

### Phase 1: Foundation (Week 1)

- [ ] Project structure: `src/`, `tests/`, `config/`, `deployment/`, `docs/`
- [ ] `config/settings.py` — Environment-based configs (dev/prod)
- [ ] `src/utils/logger.py` — Structured JSON logging
- [ ] `src/data/instrument_loader.py` — Stock metadata cache
- [ ] `src/core/kafka_producer.py` — KiteConnect → Kafka with reconnect logic
- [ ] `src/core/redis_client.py` — Redis pooling singleton
- [ ] `src/models/model_manager.py` — TFT serving layer

### Phase 2: Data Pipeline (Week 2)

- [ ] `src/data/tick_validator.py` — Input validation, outlier detection
- [ ] `src/consumers/inference_consumer.py` — Kafka → TFT → Redis (latency-critical)
- [ ] `src/consumers/viz_consumer.py` — Kafka → Redis state aggregation
- [ ] `src/consumers/persistence_consumer.py` — Kafka → InfluxDB batch writes

### Phase 3: API & Deployment (Week 3)

- [ ] `src/api/app.py` — FastAPI endpoints (/predict, /history, /health, /model/version)
- [ ] `docker-compose.yml` — All services (Kafka, Redis, InfluxDB, producers, consumers, API)
- [ ] Health checks and graceful shutdown handlers

### Phase 4: Testing & Ops (Weeks 4-6)

- [ ] Unit tests (TFT forward pass, validators, Redis ops)
- [ ] Integration tests (end-to-end mock pipeline)
- [ ] Load testing (500 ticks/sec, target <400ms P50 latency)
- [ ] Backtesting (directional accuracy >50%)
- [ ] `docs/DEPLOYMENT.md` — Setup and verification
- [ ] `docs/TROUBLESHOOTING.md` — Common issues

---

## Key Files & Priorities

| File                                    | Purpose                      | Priority | Phase |
| --------------------------------------- | ---------------------------- | -------- | ----- |
| `config/settings.py`                    | Environment configs, secrets | P0       | 1     |
| `src/models/tft_model.py`               | TFT architecture             | P0       | 0     |
| `src/models/model_manager.py`           | Model loading/caching        | P0       | 1     |
| `src/core/kafka_producer.py`            | Market data ingestion        | P0       | 1     |
| `src/core/redis_client.py`              | Shared state                 | P0       | 1     |
| `src/utils/logger.py`                   | Structured logging           | P0       | 1     |
| `src/data/instrument_loader.py`         | Metadata cache               | P0       | 1     |
| `src/consumers/inference_consumer.py`   | TFT inference loop           | P0       | 2     |
| `docker-compose.yml`                    | Local orchestration          | P0       | 3     |
| `src/data/tick_validator.py`            | Input validation             | P1       | 2     |
| `src/models/training.py`                | Offline training             | P1       | 0     |
| `src/consumers/viz_consumer.py`         | Viz state                    | P1       | 2     |
| `src/consumers/persistence_consumer.py` | InfluxDB writes              | P1       | 2     |
| `src/api/app.py`                        | REST API                     | P1       | 3     |
| `tests/test_*.py`                       | Unit/integration tests       | P2       | 4     |

---

## Critical Design Decisions

### 1. Kafka Partitioning

- **25 partitions** (2:1 stock-to-partition ratio for ~50 stocks)
- **Stock ID/Instrument Token as key** ensures strict per-stock ordering
- **Fan-out via 3 consumer groups** (inference-grp, viz-grp, archive-grp)

### 2. Data Representation

- **Log returns** (not raw prices) for statistical stationarity
- **60-tick history window** for TFT (balance between latency and pattern capture)
- **5-step quantile forecast** for uncertainty quantification in predictions

### 3. Redis Usage

- **Shared predictions**: `PRED:<SYMBOL>:quantiles` hash with P10/P30/P50/P70/P90
- **Live ticks**: `STOCK:<SYMBOL>` hash with LTP, OHLCV, Volume, Last_Updated
- **TTL**: 1-2 sec for predictions, older for tick history

### 4. Deployment

- **Docker Compose** (not Kubernetes) for on-premise simplicity
- **Single machine** assumption (vertical scaling via resource tuning)
- **Graceful shutdown** handlers for clean Kafka offset commits

### 5. Error Handling

- **Exponential backoff** for KiteConnect WebSocket reconnection
- **Dead Letter Queue** (DLQ) for invalid/unparseable Kafka messages
- **Circuit breaker** for TFT inference failures (fallback to previous prediction)

---

## Performance Targets

| Metric             | Target                      | Method                       |
| ------------------ | --------------------------- | ---------------------------- |
| End-to-end latency | <400ms P50, <500ms P99      | Mock 1000 ticks, measure     |
| Memory usage       | <2GB (single machine)       | Monitor during load test     |
| Redis memory       | <500MB (1-min retention)    | `redis-cli INFO memory`      |
| Model accuracy     | >50% directional (baseline) | Backtest on held-out 2 weeks |
| Docker startup     | <30 seconds all healthy     | `docker-compose up -d && ps` |
| Kafka lag          | <10 messages (per consumer) | Monitor offset commits       |

---

## Testing Strategy

### Unit Tests

- TFT model forward pass (random tensors)
- Tick validators (range, missing fields)
- Redis client (connection, key operations)
- Model manager (versioning, loading)

### Integration Tests

- Mock Kafka producer → consumer pipeline
- End-to-end latency measurement
- Graceful shutdown verification

### Load Tests

- Simulate 500 ticks/sec (5x expected load) for 5 minutes
- Measure P50/P99 inference latencies
- Monitor memory stability

### Backtesting

- Test on 2-week held-out dataset
- Measure directional accuracy, MAE, RMSE
- Log confusion matrix

---

## Operational Considerations

### Logging

- Structured JSON logging (python-json-logger or similar)
- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Rotation: Daily, 7-day retention

### Monitoring

- Prometheus metrics (optional): inference latency, Kafka lag, Redis memory
- Alert thresholds:
  - Inference latency > 400ms
  - Kafka lag > 10 messages
  - Redis disconnections
  - No ticks for >5 seconds

### Graceful Shutdown

- Signal handlers (SIGTERM, SIGINT)
- Close Kafka connections (commit offsets)
- Flush Redis cache
- Exit within 10 seconds

### Deployment Steps

1. `docker-compose up -d` (start all services)
2. Verify Kafka producer is sending ticks
3. Check Redis for predictions updating
4. Query InfluxDB for historical data
5. Test API endpoints

---

## What Was Missing from Original Process_Flow.md

Your original Process_Flow.md covered **only** the core data pipeline. This CLAUDE.md adds:

1. ✅ **TFT Model Architecture** (fully specified)
2. ✅ **Configuration Management** (environment-based)
3. ✅ **Structured Logging** (JSON format)
4. ✅ **Data Validation** (validators, DLQ)
5. ✅ **Monitoring & Alerting** (metrics, thresholds)
6. ✅ **Error Handling & Resilience** (backoff, circuit breaker)
7. ✅ **Model Management** (versioning, retraining strategy)
8. ✅ **API Layer** (FastAPI endpoints)
9. ✅ **Testing Strategy** (unit, integration, load, backtest)
10. ✅ **Deployment Automation** (Docker Compose)
11. ✅ **Operational Runbooks** (shutdown, monitoring, scaling)
12. ✅ **Documentation** (deployment, troubleshooting)

---

## Useful Commands

### Docker

```bash
docker-compose up -d              # Start all services
docker-compose ps                 # Check service health
docker-compose logs -f <service>  # Follow logs
docker-compose down               # Stop all services
```

### Kafka

```bash
kafka-console-consumer --topic stock-quotes --from-beginning --max-messages 10
kafka-consumer-groups --list --bootstrap-server localhost:9092
```

### Redis

```bash
redis-cli HGET PRED:RELIANCE:quantiles P50
redis-cli INFO memory
redis-cli FLUSHALL
```

### InfluxDB

```bash
influx query "SELECT * FROM ticks WHERE time > now() - 1h"
```

### API

```bash
curl http://localhost:8000/predict/RELIANCE
curl http://localhost:8000/health
curl http://localhost:8000/model/version
```

---

## Next Steps

1. **Create project structure** (Week 1)
2. **Implement TFT model** (Week 0 parallel)
3. **Build configuration layer** (Week 1)
4. **Implement core consumers** (Week 2)
5. **Add API layer** (Week 3)
6. **Deploy & test** (Weeks 4-6)

See `/right-now-this-folder-elegant-clock.md` in `.claude/plans/` for detailed implementation plan.
