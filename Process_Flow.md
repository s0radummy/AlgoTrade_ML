# Real-Time Predictive Stock Engine: Technical Workflow

---

## Phase 1: Ingestion & Serialization

- **Source**: Establish a persistent WebSocket connection via KiteConnect API.

- **Decoding**: Parse incoming binary tick packets into Python dictionaries.

- **Transformation**: Package the 7 core variables (LTP, OHLCV, Volume, Change) into a structured JSON object.

- **Serialization**: Convert JSON to UTF-8 Byte-stream for Kafka compatibility.

- **Error Handling**: Implement exponential backoff for socket reconnection and heartbeat monitoring.

---

## Phase 2: Brokerage & Distribution (The Pipe)

- **Producer**: Execute `producer.produce()` using the Stock-ID/Instrument Token as the Message Key.

- **Partitioning**: Configure the `stock-quotes` topic with 25 partitions (2:1 Stock-to-Partition ratio) to ensure strict chronological ordering per stock via Key-hashing.

- **Distribution Architecture**: Utilize Kafka’s Fan-out capability by deploying three independent Consumer Groups (`inference-grp`, `viz-grp`, `archive-grp`).

---

## Phase 3: The Consumer Ecosystem

### Consumer 1: Inference Engine (TFT)

- **Bootstrap**: On initialization, query InfluxDB for the most recent 60 ticks per stock to hydrate local Deques.

- **Preprocessing**: Convert raw price data into Log Returns for statistical stationarity.

- **Inference Loop**:
  - Update stock-specific deque with the latest Kafka tick.
  - Construct the 3-Part Input Tensor:
    - Static Metadata: Stock ID, Sector, Market Cap (via local lookup).
    - Past Inputs: The 60-tick window from the deque.
    - Future Context: Normalized Time/Day features.

  - Execution: Pass tensor through the Temporal Fusion Transformer.

- **Output**: Transmit a 5-step Quantile Forecast (`$P_{10}, P_{50}, P_{90}$`) to the shared Redis Hash.

---

### Consumer 2: Visualization Manager

- **State Aggregation**: Consume raw ticks from Kafka and perform an atomic HMSET to the Redis Hash (Key: `STOCK_ID`).

- **Payload**: Update real-time fields: LTP, OHLCV, Volume, and Last_Updated.

- **UI Delivery**: Decouple the frontend (Streamlit/Dash) by allowing it to poll Redis or receive updates via Redis Pub/Sub for millisecond-level responsiveness.

---

### Consumer 3: Persistence Layer (Persistence)

- **Storage**: Direct raw Kafka ticks to InfluxDB (Time-Series DB).

- **Optimization**: Implement Batch Writing (e.g., every 1000 ticks or 5 seconds) to minimize Disk I/O overhead.

- **Schema**: Utilize Stock_ID and Sector as Tags for high-speed indexing; store price variables as Fields.

---
