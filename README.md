# AlgoTrading: Real-Time Predictive Stock Engine

A production-ready real-time stock price prediction system using Temporal Fusion Transformer, Kafka, Redis, and InfluxDB.

## Quick Start

### 1. Setup

```bash
cp .env.example .env
# Edit .env with your KiteConnect API credentials
nano .env
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Verify Health

```bash
curl http://localhost:8000/health
```

### 4. Get Prediction

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/predict/RELIANCE
```

## Architecture

```
KiteConnect WebSocket
    ↓
[Kafka Producer] → Stock-Quotes Topic (25 partitions)
    ↓
    ├→ [Inference Consumer] → TFT Model → Redis
    ├→ [Viz Consumer] → Redis (UI state)
    └→ [Persistence Consumer] → InfluxDB
```

## Services

| Service  | Port | Purpose               |
| -------- | ---- | --------------------- |
| Kafka    | 9092 | Message broker        |
| Redis    | 6379 | Caching & predictions |
| InfluxDB | 8086 | Time-series storage   |
| API      | 8000 | FastAPI REST server   |

## Features

- **Real-Time Inference**: <400ms latency from tick to prediction
- **Multi-Quantile Forecasts**: P10, P30, P50, P70, P90 predictions
- **Scalable Architecture**: Kafka fan-out to 3 independent consumers
- **Fault Tolerant**: Graceful shutdown, exponential backoff, circuit breaker
- **Production Ready**: Structured logging, monitoring, health checks

## Project Structure

```
AlgoTrading/
├── config/              # Configuration management
├── src/
│   ├── core/           # Kafka producer, Redis client
│   ├── data/           # Validators, metadata loader
│   ├── models/         # TFT architecture, serving
│   ├── consumers/      # Inference, Viz, Persistence
│   ├── api/            # FastAPI REST layer
│   └── utils/          # Logging, helpers
├── docker/             # Docker build files
├── scripts/            # Utilities, mock producer
├── tests/              # Unit & integration tests
├── docs/               # Deployment, troubleshooting
└── docker-compose.yml  # Multi-service orchestration
```

## Configuration

Copy `.env.example` to `.env` and set:

- KiteConnect API credentials
- Kafka bootstrap servers
- Redis host/port
- InfluxDB credentials
- Model path
- Stock list to track

## Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
pytest tests/
```

### Train Model

```bash
python scripts/train_model.py
```

### Start Individual Services

```bash
# Producer (mock data)
python scripts/producer.py

# Inference consumer
python -m src.consumers.inference_consumer

# API server
python -m src.api.app
```

## API Documentation

### Endpoints

- `GET /health` — System health check
- `GET /predict/{symbol}` — Latest quantile forecast
- `GET /history/{symbol}` — Historical data (24 hours default)
- `GET /stocks` — List tracked stocks
- `GET /model/version` — Model metadata
- `GET /stats` — System statistics

### Authentication

All endpoints (except `/health`) require `X-API-Key` header.

## Monitoring

### View Logs

```bash
docker-compose logs -f inference-consumer
docker-compose logs -f api
```

### Check Performance

```bash
docker stats
redis-cli INFO memory
kafka-consumer-groups --group inference-grp --describe --bootstrap-server localhost:9092
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment steps.

## Performance Targets

- **Latency**: <400ms P50, <500ms P99
- **Throughput**: 1000 ticks/sec (5x expected load)
- **Memory**: <2GB per container
- **Accuracy**: >50% directional accuracy (baseline)

## Testing

### Unit Tests

```bash
pytest tests/test_tft_model.py
pytest tests/test_validators.py
```

### Load Testing

```bash
# Simulate 500 ticks/sec for 5 minutes
python scripts/load_test.py
```

### Backtesting

```bash
python scripts/backtest.py --start-date 2024-01-01 --end-date 2024-06-30
```

## Model Training

The TFT model is trained offline daily:

```python
from src.models.training import TFTTrainer, create_dummy_data

train_loader, val_loader = create_dummy_data(num_samples=10000)
trainer = TFTTrainer(model)
train_losses, val_losses = trainer.fit(train_loader, val_loader, epochs=50)
```

**Input Format**: (batch, 60, 7) where 7 = [log_return, volume, OHLC, ...]
**Output Format**: (batch, 5, 5) where 5 steps × 5 quantiles

## Graceful Shutdown

Services handle `SIGTERM` and `SIGINT`:

```bash
# Clean shutdown
docker-compose stop

# Or Ctrl+C in terminal
^C
```

- Flushes Kafka offsets
- Commits pending writes to InfluxDB
- Closes Redis connections
- Logs summary statistics

## Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Make changes and test: `pytest tests/`
3. Commit with descriptive message
4. Push and open PR

## License

Proprietary - AlgoTrading Team

## Support

For issues:

1. Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
2. Review service logs: `docker-compose logs`
3. Contact the development team

---

**Last Updated**: 2026-05-15
**Version**: 1.0.0-alpha
