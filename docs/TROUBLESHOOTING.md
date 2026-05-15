# Troubleshooting Guide

## Common Issues

### 1. Kafka Connection Refused

**Error**: `Connection refused (Errno 111)`

**Causes**:

- Kafka service not running
- Wrong bootstrap servers address
- Network connectivity issue

**Solutions**:

```bash
# Check if Kafka is running
docker-compose ps kafka

# Check Kafka logs
docker-compose logs kafka

# Restart Kafka
docker-compose restart kafka

# Verify connectivity
telnet localhost 9092
```

### 2. Redis Connection Failed

**Error**: `Error connecting to redis://localhost:6379`

**Causes**:

- Redis service down
- Wrong host/port configuration
- Memory full

**Solutions**:

```bash
# Check Redis status
docker-compose ps redis

# Ping Redis
redis-cli ping

# Check memory
redis-cli INFO memory

# Restart Redis
docker-compose restart redis

# Clear memory (WARNING: data loss)
redis-cli FLUSHALL
```

### 3. InfluxDB Write Failures

**Error**: `Failed to connect to InfluxDB` or `Bucket not found`

**Causes**:

- InfluxDB service down
- Wrong credentials
- Bucket doesn't exist
- Disk full

**Solutions**:

```bash
# Check health
curl http://localhost:8086/health

# Check logs
docker-compose logs influxdb

# Verify bucket exists
influx bucket list

# Check disk space
docker exec algotrade-influxdb df -h

# Restart InfluxDB
docker-compose restart influxdb
```

### 4. Model Loading Failed

**Error**: `Model not found at models/tft_v1.pth` or `CUDA out of memory`

**Causes**:

- Model file missing
- Incompatible PyTorch version
- GPU memory insufficient
- Device mismatch

**Solutions**:

```bash
# Check model exists
ls -la models/

# Verify PyTorch version
python -c "import torch; print(torch.__version__)"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Generate dummy model for testing
python scripts/generate_model.py
```

### 5. High Inference Latency

**Error**: Predictions take >400ms or system slowdown

**Causes**:

- Model too large
- Batch size too large
- I/O bottleneck
- CPU throttling

**Solutions**:

```bash
# Monitor latency
docker-compose logs inference-consumer | grep "Inference"

# Check resource usage
docker stats

# Reduce batch size in docker-compose.yml
# Increase CPU/memory allocation
docker-compose.yml: resources: limits: cpus: "2" memory: 2G
```

### 6. Kafka Lag Building Up

**Error**: Consumer lag continuously increasing

**Causes**:

- Consumer slower than producer
- Partition rebalancing
- Processing errors

**Solutions**:

```bash
# Check consumer lag
kafka-consumer-groups --group inference-grp --describe --bootstrap-server localhost:9092

# Check partition distribution
kafka-topics --describe --topic stock-quotes --bootstrap-server localhost:9092

# Restart consumer
docker-compose restart inference-consumer

# Increase consumer parallelism
docker-compose up -d --scale inference-consumer=2
```

### 7. API Returning 401 Errors

**Error**: `HTTP 401: Invalid API key`

**Causes**:

- Wrong API key in header
- API key changed in .env
- Header name case mismatch

**Solutions**:

```bash
# Verify API key in .env
cat .env | grep API_KEY

# Test with correct key
curl -H "X-API-Key: your_api_key" http://localhost:8000/health
```

### 8. Out of Memory Errors

**Error**: `MemoryError` or `Killed (OOM)`

**Causes**:

- Container memory limit too low
- Data accumulation (deques, caches)
- Memory leak in model

**Solutions**:

```bash
# Check memory usage
docker stats

# Increase limits in docker-compose.yml:
# services:
#   inference-consumer:
#     deploy:
#       resources:
#         limits:
#           memory: 4G

# Check for memory leaks
docker exec algotrade-inference-consumer python -m memory_profiler script.py
```

### 9. Ticks Not Flowing Through Pipeline

**Error**: No data in Redis or InfluxDB after starting producer

**Causes**:

- Producer not connected to KiteConnect
- Kafka topic not created
- Consumer lag
- Message validation failures

**Solutions**:

```bash
# Check producer logs
docker-compose logs producer

# Check if messages in Kafka
kafka-console-consumer --topic stock-quotes --from-beginning --bootstrap-server localhost:9092 --max-messages 10

# Check validation errors
docker-compose logs inference-consumer | grep "Invalid"

# Verify topic exists
kafka-topics --list --bootstrap-server localhost:9092
```

### 10. Docker Compose Build Failures

**Error**: `failed to solve with frontend dockerfile.v0`

**Causes**:

- Missing Dockerfile
- Corrupted docker cache
- Resource limitations

**Solutions**:

```bash
# Clean build cache
docker-compose build --no-cache

# Prune unused images
docker image prune -a

# Check disk space
df -h /var/lib/docker

# Try again
docker-compose up -d
```

## Performance Diagnostics

### Profile Inference Latency

```bash
# Add timing logs (edit inference_consumer.py)
start = time.time()
# ... inference code ...
latency = (time.time() - start) * 1000
logger.info(f"Inference latency: {latency:.2f}ms")

# Analyze logs
docker-compose logs inference-consumer | grep "latency" | tail -100
```

### Monitor Resource Spikes

```bash
# Real-time stats
docker stats --no-stream

# Historical stats
docker stats --no-stream > stats.log &
# ... let it run ...
# kill %1

# Parse logs
awk '{print $1, $3}' stats.log | sort -u
```

## Debug Mode

Enable verbose logging:

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Restart services
docker-compose up -d
```

## Useful Commands

```bash
# Exec into container
docker exec -it algotrade-api bash

# View specific logs
docker-compose logs --since 10m inference-consumer

# Tail real-time
docker-compose logs -f --tail=50 api

# Check network connectivity
docker exec algotrade-api ping kafka

# Inspect container
docker inspect algotrade-redis
```

## Getting Help

1. Check logs: `docker-compose logs`
2. Check this guide
3. Check service health endpoints: `/health`, `/model/version`
4. Review CLAUDE.md for architecture details
