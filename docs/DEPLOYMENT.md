# Deployment Guide

## Prerequisites

- Docker & Docker Compose (v20.10+)
- Python 3.11+ (for local development)
- Git
- At least 8GB RAM, 4 CPU cores recommended
- KiteConnect API credentials

## Setup

### 1. Clone Repository

```bash
git clone <repo-url>
cd AlgoTrading
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your credentials
nano .env
```

### 3. Directory Structure

```bash
mkdir -p logs models/checkpoints data/backtest
chmod 755 logs models data
```

### 4. Start Services

```bash
docker-compose up -d
```

### 5. Verify Health

```bash
# Check all containers
docker-compose ps

# Check logs
docker-compose logs -f api
docker-compose logs -f inference-consumer
docker-compose logs -f kafka

# Health check
curl http://localhost:8000/health
```

## Services

| Service   | Port | Purpose            |
| --------- | ---- | ------------------ |
| Kafka     | 9092 | Message broker     |
| Zookeeper | 2181 | Kafka coordination |
| Redis     | 6379 | Caching & state    |
| InfluxDB  | 8086 | Time-series DB     |
| API       | 8000 | FastAPI server     |

## API Endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Get Prediction

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/predict/RELIANCE
```

### List Stocks

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/stocks
```

### Model Info

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/model/version
```

## Troubleshooting

### Kafka Connection Issues

```bash
# Check Kafka broker
kafka-broker-api-versions --bootstrap-server localhost:9092

# List topics
kafka-topics --list --bootstrap-server localhost:9092

# Check consumer lag
kafka-consumer-groups --group inference-grp --describe --bootstrap-server localhost:9092
```

### Redis Connection Issues

```bash
redis-cli ping
redis-cli INFO memory
redis-cli FLUSHALL  # WARNING: Clears all data
```

### InfluxDB Issues

```bash
# Check health
curl http://localhost:8086/health

# Query data
influx query 'SELECT * FROM ticks WHERE time > now() - 1h'
```

### Container Restart

```bash
# Restart specific service
docker-compose restart inference-consumer

# Restart all
docker-compose restart

# Full rebuild
docker-compose down -v
docker-compose up -d
```

## Monitoring

### View Logs

```bash
# Real-time logs
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last N lines
docker-compose logs --tail=100 producer
```

### Memory Usage

```bash
# Check container resources
docker stats

# Redis memory
redis-cli INFO memory
```

## Stopping Services

### Graceful Shutdown

```bash
docker-compose stop
```

### Hard Shutdown

```bash
docker-compose down
```

### Remove Volumes

```bash
docker-compose down -v  # WARNING: Deletes data
```

## Performance Tuning

### Kafka

- Increase `max_poll_records` in consumers for higher throughput
- Tune `batch.size` in producers

### Redis

- Monitor with `redis-cli --bigkeys`
- Adjust `maxmemory-policy` if memory constrained

### InfluxDB

- Adjust batch write size in persistence consumer
- Check disk I/O performance

## Production Checklist

- [ ] Set strong API keys and passwords
- [ ] Configure TLS/SSL for external connections
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation and retention
- [ ] Backup InfluxDB and Redis regularly
- [ ] Test disaster recovery procedures
- [ ] Configure firewall rules
- [ ] Set resource limits in docker-compose

## Support

For issues, check TROUBLESHOOTING.md or contact the development team.
