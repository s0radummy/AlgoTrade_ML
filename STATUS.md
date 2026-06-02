# AlgoTrading — Status

_Last updated: 2026-06-02 (session 5)_

---

## 1. What This Project Is For

The end goal is precise and narrow: **for each of 47 Nifty 50 stocks, produce a calibrated probability distribution over where the stock's price is likely to land one minute from now** — continuously, in real time, throughout the entire trading session (09:15–15:30 IST).

This is not a binary "up or down" signal. The system outputs five quantile forecasts — P10, P30, P50, P70, P90 — for each of the next five one-minute bars. Reading them together tells you not just the most probable direction, but also how wide the range of outcomes is, and how asymmetric the distribution is. A forecast where P50 = +0.4% and P10 = +0.1% is a high-confidence bullish signal with a tight lower bound. A forecast where P50 = +0.4% but P10 = −0.6% means the upside is the median expectation but there is significant downside tail risk — a very different trading situation, even though the P50 is identical.

Concretely: every time a one-minute bar completes for a stock, the system reads the last 60 completed bars from Redis, constructs an 11-feature time-series tensor, attaches static stock metadata (sector, market cap) and forward-looking calendar features (time-of-day, day-of-week), and passes the whole thing through a 462,000-parameter Temporal Fusion Transformer. The model returns five quantile log-return forecasts for each of the next five minutes. These are denormalized back to absolute ₹ price space and written to Redis. The process repeats for all 47 stocks, independently, on every minute boundary — meaning the entire prediction surface across the portfolio refreshes every minute.

The underlying model was trained on three years of 1-minute OHLCV data for all 47 stocks simultaneously. It learned price dynamics, volume patterns, sector co-movement, and intraday time structure. The Nifty 50 index (NIFTY50) is ingested as a market-wide feature (column 10 of every stock's input tensor), giving the model a live read on broad market direction during inference.

The practical output: at any moment during market hours, for any of the 47 tracked stocks, you can see exactly what the model believes the next five minutes will look like — with calibrated confidence bands. Walk-forward validation on the trained checkpoint confirms 52.65% directional accuracy across six non-overlapping 14-day windows. Quantile calibration is excellent: P10 empirically covers 10.7% of outcomes, P90 covers 89.6%.

---

## 2. Accessing the System

The entire system runs locally. Start all services with `docker-compose up -d` and run `kite_to_kafka.py` to begin the live feed. Two interfaces are available:

### Streamlit Dashboard — `localhost:8501`

Run with `streamlit run scripts/dashboard.py`. Auto-refreshes every 2 seconds by default (adjustable 1–30s slider in the sidebar, pauseable).

**Market Overview → Table View**
A live table of all 47 stocks showing LTP, open/high/low/close, volume, and percentage change. Rows are color-coded green (gaining) or red (losing) per cell. Filterable by All / Gainers / Losers. Sortable by sector, symbol, LTP, or change. Clicking any row selects that stock and syncs the sidebar and Stock Detail tab.

**Market Overview → Heatmap View**
A Plotly treemap of all 47 stocks. Cell size is proportional to trading volume; cell color is the percentage change on a red–green diverging scale. Sectors are grouped. Gives an instant visual read on where money is flowing in the session. Clicking a cell selects the stock.

**Stock Detail Tab**
A per-stock drill-down showing:
- A 30-bar 1-minute candlestick chart sourced from Redis (`STOCK:{sym}:bars`). Completed bars are green (up) or red (down); the live in-progress bar is rendered in blue (up) or orange (down) and updates every refresh cycle.
- A session sparkline tracking LTP from session start, accumulated in browser state.
- Day-level OHLCV stats (open, high, low, close, volume, change%) from KiteConnect.

**TFT Predictions Tab**
Refreshes every 60 seconds (aligned to the 1-minute bar cadence, not the 2-second market data refresh):
- A full P10/P30/P50/P70/P90 table for all 47 stocks simultaneously.
- A stock-specific forecast chart in ΔPrice space: Y-axis is ₹ change from current LTP, zero line is the current price. Bars extend up (bullish) or down (bearish). Green = positive P50, red = negative P50.
- A per-step quantile table below the chart showing compounded implied prices for each of the five forecast steps.
- A market-wide P50 direction strip at the bottom showing the directional lean across all 47 stocks at a glance.

**Sidebar**
Stock selector dropdown (all 47), LTP, full OHLCV detail, sector badge, and a compact colored prediction pill (↑ Bullish / ↓ Bearish + magnitude %) when predictions are available.

### FastAPI REST API — `localhost:8000`

| Endpoint | Purpose |
|---|---|
| `GET /health` | System health (Redis, model loaded, last tick timestamp) |
| `GET /predict/{symbol}` | Latest P10–P90 quantile forecast for a stock |
| `GET /model/version` | Checkpoint metadata (epoch, val loss, p50_std) |
| `GET /history/{symbol}` | Last 24h tick history from InfluxDB _(stub — incomplete)_ |

Authentication via `X-API-Key` header (set in `.env`).

---

## 3. Project Architecture

### Data Ingestion

`scripts/kite_to_kafka.py` authenticates with Zerodha KiteConnect fully automatically: POST to `/api/login` (password), POST to `/api/twofa` (TOTP generated from the base32 secret in `.env`), follow the OAuth redirect to capture `request_token`, exchange for `access_token`. The session is cached in `.kite_session.json` and reused for the rest of the trading day.

After auth, it resolves instrument tokens for all 47 stocks via `kite.quote()` and subscribes via `KiteTicker`. Stocks are set to `MODE_FULL` (full tick data: LTP, OHLCV, volume, change). The Nifty 50 index (token `256265`) is added separately in `MODE_QUOTE`. Every incoming tick is serialized to JSON and produced to Kafka with the instrument token as the partition key, ensuring strict chronological ordering per stock across the session.

A background consumer thread reads back from Kafka and computes round-trip latency. Measured live: avg 27.6ms RTT.

### Message Bus (Kafka)

Topic: `stock-quotes`, 25 partitions (2:1 stock-to-partition ratio for 47 stocks). Three independent consumer groups create a fan-out: `viz-grp`, `inference-grp`, `persistence-grp`. Each consumer group processes every message independently. Within each group, messages for the same stock always land on the same partition (key-hash routing), guaranteeing per-stock FIFO ordering.

Kafka is run via `confluentinc/cp-kafka:7.5.0`. Zookeeper handles coordination. The topic is auto-created at broker startup by docker-compose (`KAFKA_CREATE_TOPICS: "stock-quotes:25:1"`).

### Consumer 1 — VizConsumer (`viz-grp`)

The single bar-builder for the entire system. Owns a `BarAccumulator` instance per stock. For each incoming tick:

1. Extracts LTP and cumulative daily volume from the tick.
2. Checks whether the tick's minute bucket `(year, month, day, hour, minute)` has changed vs the previous tick for that stock.
3. If the minute is new: the previous bar is completed and pushed to Redis. A new bar is opened with the current LTP as open/high/low/close and the current cumulative volume as the bar-open baseline.
4. If the minute is the same: the running bar's high, low, close, and volume delta are updated in memory.
5. Every tick (regardless of bar completion): writes the full live bar state to `STOCK:{sym}:current_bar` as a Redis hash. Also writes day-level KiteConnect OHLCV (the daily open/high/low/close, not the 1-minute bar) for use by the dashboard.

On bar completion:
- `LPUSH STOCK:{sym}:bars <bar_json>` — newest bar at index 0.
- `LTRIM STOCK:{sym}:bars 0 299` — caps the list at 300 bars.

Both NIFTY50 and all 47 stocks go through the same `BarAccumulator` path. Redis is configured with AOF persistence, so `STOCK:*:bars` lists survive container restarts and are available at next-day startup without any warm-start query.

### Consumer 2 — InferenceConsumer (`inference-grp`)

Latency-critical path. Does not build bars — it only uses ticks to detect minute-boundary crossings (by comparing the current tick's `(year, month, day, hour, minute)` against the last seen minute for that symbol). On a boundary crossing for stock `S`:

1. `LINDEX STOCK:S:bars 0` — checks whether VizConsumer has already written the completed bar. If not, waits 100ms and retries once (guards against the narrow race window).
2. `LRANGE STOCK:S:bars 0 99` — reads up to 100 most recent completed bars (newest first).
3. `LRANGE STOCK:NIFTY50:bars 0 99` — reads Nifty50 bars for col 10 of the feature tensor.
4. Reverses both lists (oldest first) and constructs the input tensors.
5. Runs TFT inference (`model_manager.predict()`).
6. Denormalizes predictions from z-score space: `pred_denorm = pred_zscore × std + mean` using per-stock stats loaded from the checkpoint at startup.
7. Writes results to Redis: `PRED:{sym}:quantiles` (hash with P10–P90 for step 1) and `PRED:{sym}:steps` (JSON with all 5 steps).

**Feature tensor construction (past_inputs: 60 × 11):**

| Col | Feature | How computed |
|-----|---------|-------------|
| 0 | `log_ret` | `log(close / prev_close)`, then z-scored per stock |
| 1 | `open_ret` | `log(open / prev_close)` |
| 2 | `high_ret` | `log(high / prev_close)` |
| 3 | `low_ret` | `log(low / prev_close)` |
| 4 | `intraday_ret` | `log(close / open)` |
| 5 | `intraday_rng` | `log(high / low)` |
| 6 | `vol_norm` | `log1p(bar_vol / mean_bar_vol)` |
| 7 | `rsi_14` | Wilder RSI on all available bars |
| 8 | `macd_hist_norm` | MACD histogram / close, on all available bars |
| 9 | `atr_norm` | ATR(14) / close, on all available bars |
| 10 | `nifty_log_ret` | `log(nifty_close[t] / nifty_close[t-1])` |

Cols 7–9 are computed on all available bars (not just the 60-bar window) to ensure MACD has a full 26-bar slow EMA warmup. The result is then trimmed to the last 60 values.

**Static covariates (32 dims):** stock identity hash, sector ordinal, log-normalized market cap, sector one-hot (10 sectors), historical return volatility, market-cap bucket one-hot (8 buckets), reserved zeros.

**Future inputs (5 × 4):** sin/cos of intraday time fraction, weekday normalized (0–1), day-of-month normalized (0–1).

NIFTY50 ticks are filtered out before inference (only used as a market feature via `STOCK:NIFTY50:bars`, never run through the TFT directly).

### Consumer 3 — PersistenceConsumer (`persistence-grp`)

Buffers ticks in memory (default batch size: 1000) and flushes to InfluxDB as line protocol when the batch is full or on shutdown. Converts tick timestamp from ISO 8601 to epoch nanoseconds (UTC-aware, no IST offset shift). Commits Kafka offsets only after a successful InfluxDB write, giving at-least-once delivery semantics. On write failure, the batch is retained and retried on the next flush.

InfluxDB measurement: `ticks`. Tags: `symbol`. Fields: `last_price`, `volume`, `change`, `instrument_token` (int).

### Redis Key Schema

| Key | Type | Written by | Read by | Notes |
|-----|------|-----------|---------|-------|
| `STOCK:{sym}:current_bar` | Hash | VizConsumer (every tick) | Dashboard, InferenceConsumer | LTP, bar OHLCV, day OHLCV, Last_Updated |
| `STOCK:{sym}:bars` | List (max 300) | VizConsumer (every bar) | InferenceConsumer, Dashboard | Newest at index 0; AOF-persistent |
| `PRED:{sym}:quantiles` | Hash | InferenceConsumer | Dashboard, API | P10–P90 for step 1, timestamp |
| `PRED:{sym}:steps` | String (JSON) | InferenceConsumer | Dashboard, API | All 5 steps × 5 quantiles + LTP snapshot |

### TFT Model

462,464-parameter Temporal Fusion Transformer — strict implementation of Lim et al. (2021).

**Input:** `static_cov (B, 32)`, `past_inputs (B, 60, 11)`, `future_inputs (B, 5, 4)`.  
**Output:** `(B, 5, 5)` — 5 forecast steps × 5 quantiles (P10, P30, P50, P70, P90) in z-scored log-return space.

Internal components:
- **GRN (Gated Residual Network):** ELU activation → linear → GLU gate → residual add → LayerNorm. Optional context vector injection. Used throughout as the core non-linear unit.
- **VSN (Variable Selection Network):** per-variable GRN + softmax importance weighting → learned feature selection.
- **Static covariate encoder:** produces four context vectors from static inputs — used to initialize LSTM hidden/cell states, enrich temporal features, and weight variable selection.
- **LSTM encoder (past, 60 steps) + decoder (future, 5 steps):** initialized from static context vectors.
- **Post-LSTM gating + skip connection.**
- **Static enrichment layer:** GRN with static context injected.
- **Temporal multi-head self-attention (4 heads):** attends across the full 65-step sequence (60 past + 5 future).
- **Post-attention gating + skip connection.**
- **Position-wise GRN feed-forward.**
- **Pre-output gating + skip connection.**
- **Linear quantile head:** projects to 5 quantiles per step.

Loss function: quantile (pinball) loss across all five quantiles, summed. A soft diversity penalty is added to prevent quantile collapse (penalizes P90 − P10 < 0.05).

### Docker Compose

Eight services on a shared `algotrade-network` bridge network:

| Service | Image | Port | Role |
|---------|-------|------|------|
| `zookeeper` | confluentinc/cp-zookeeper:7.5.0 | 2181 | Kafka coordination |
| `kafka` | confluentinc/cp-kafka:7.5.0 | 9092, 29092 | Message broker |
| `redis` | redis:7.2-alpine | 6379 | Shared bar cache + prediction state |
| `influxdb` | influxdb:2.7-alpine | 8086 | Time-series tick persistence |
| `viz-consumer` | Dockerfile.consumer | — | Bar builder (VizConsumer) |
| `inference-consumer` | Dockerfile.consumer | — | TFT inference (InferenceConsumer) |
| `persistence-consumer` | Dockerfile.consumer | — | InfluxDB writer (PersistenceConsumer) |
| `api` | Dockerfile.api | 8000 | FastAPI REST server |

All services have health checks. Redis and InfluxDB use named Docker volumes for persistence. `kite_to_kafka.py` runs outside Docker on the host, connecting to Kafka via `localhost:29092`.

---

## Training History

| Run | Stocks | Epochs | Best Val | p50_std | Verdict |
|-----|--------|--------|----------|---------|---------|
| Diagnostic 1 (2026-05-18) | 3 | 6 (early stop) | 0.221692 @ ep1 | 0.001–0.007 | Failed — val stalled after ep1 |
| Diagnostic 2 (2026-05-19) | 10 | 10 | 0.203295 @ ep7 | 0.002→0.015 | Passed |
| Full Run 1 (2026-05-19) | 47 | killed @ ep4 | — | — | Killed — patience=5 too low for CosineAnnealingLR |
| Full Run 2 (2026-05-19) | 47 | killed @ ep10 | — | — | Killed — train/val divergence (misdiagnosed as overfitting; was warm-start bias from 10-stock checkpoint) |
| Full Run 3 (2026-05-20) | 47 | killed @ ep9 | 0.214295 @ ep4 | 0.001→0.0002 | Failed — p50_std collapsed; root cause: Adam+weight_decay adaptive L2 amplification loop |
| Diagnostic 4 (2026-05-20) | 10 | 10 | 0.203264 @ ep6 | 0.010→0.025 | Passed — confirmed AdamW+grad_clip=0.1 fixes collapse |
| **Full Run 4 (2026-05-20/21)** | **47** | **19 (early stop @ patience=15)** | **0.213697 @ ep4** | **0.032→0.095** | ✅ **Passed — first clean full run** |

**Current checkpoint:** `models/tft_v1.pth` — epoch 4, val=0.213697, p50_std=0.033, 47 target_stats. **Safe for inference.**

### What was broken and how it was fixed (2026-05-20)

Three confirmed bugs in `src/models/training.py` across all previous runs:

1. **Adam + weight_decay = silent prediction collapse.** `Adam(weight_decay=1e-4)` applies L2 regularization inside the adaptive update. Once output predictions converge, output-head gradients shrink, Adam's internal momentum amplifies the L2 force, and weights are pushed toward zero — making predictions even more constant. A feedback loop. Fix: switched to `AdamW` (decoupled weight decay), which breaks the loop.

2. **Gradient clipping at 1.0 instead of 0.1.** The reference library recommends `gradient_clip_val=0.1`. Clipping at 1.0 allows 10× larger gradient steps through attention layers, which can cause attention entropy collapse. Fix: `max_norm=1.0` → `max_norm=0.1`.

3. **No collapse early-warning.** The alarm only fired at `p50_std < 1e-4` — far too late (Run 3 collapse started at epoch 1). Fix: added monotonic decline detection (3 consecutive drops triggers immediate warning).

Additional safeguards: soft quantile diversity loss (penalizes P90−P10 < 0.05), checkpoint p50_std guard (auto-skips warm-start if saved checkpoint has collapsed predictions), `--no-warmstart` CLI flag.

### Why early stopping at epoch 19 is expected, not a bug

Full Run 4 best val was epoch 4 (0.213697). Val loss then oscillated in a 0.0003 band for 15 epochs without improving — patience=15 fired correctly. The tight band means the model reached its genuine generalization ceiling early. This is normal for stock log-return prediction. The checkpoint is at the best generalization point.

---

## Session History

**Session 3 (2026-05-26) — First live run.** Full pipeline ran end-to-end for the first time: 143K+ ticks, all 47 stocks, inference consumer predicting live. `jinja2` version mismatch found and fixed (pandas 3.0.3 requires `>=3.1.5`). Dashboard redesigned: heatmap fixed (equal-area sizing), table color coding fixed, TFT Predictions tab rebuilt with ΔPrice forecast chart and per-step quantile table. Overnight eval: 47.6% directional accuracy. Root causes identified: NIFTY50 not subscribed (col 10 was zeros throughout), and two prior days of training data had mock-producer contamination.

**Session 4 (2026-05-28) — Architecture overhaul.** VizConsumer made the sole owner of `BarAccumulator`; writes `STOCK:{sym}:bars` (300-bar AOF-persistent list) and `STOCK:{sym}:current_bar` (live tick hash) to Redis. InferenceConsumer refactored to read bars from Redis — `_warm_start_from_influxdb` (~160 lines) deleted entirely. NIFTY50 token 256265 subscribed; col 10 is now real data. Mock producer infrastructure removed. Dashboard Stock Detail rebuilt with 30-bar candlestick from Redis.

**Session 5 (2026-06-02) — InfluxDB series fragmentation fix.** Diagnosed that `instrument_token` stored as an InfluxDB tag caused series splits whenever Zerodha reassigns token values — confirmed across 6 stocks (ASIANPAINT, INFY, LT, RELIANCE, TCS, WIPRO) which had accumulated 3 series each (mock-producer era, sessions 3–4, session 5). Fixed by moving `instrument_token` from tag to field in `PersistenceConsumer.tick_to_line_protocol()` — `symbol` is now the sole series key. Added a PID lockfile to `kite_to_kafka.py` to prevent double-run scenarios that exacerbate the issue.

---

## Known Issues / Open Questions

- **47.6% live accuracy vs 52.65% benchmark** — first live session underperformed due to col 10 zeros and mock-producer data contamination. Both fixed as of session 4. Session 5 (today) is the first clean live run.
- **Fan chart not implemented** — TFT Predictions tab uses a bar/whisker chart in ΔPrice space. A proper P10–P90 filled cone with historical bar context to the left has not been built. Redis bar history is available.
- **3 missing stocks** — LTIM, TATAMOTORS, ZOMATO have no `target_stats` in the checkpoint. Do not add until data is fetched and model is retrained.
- **`/history` endpoint stub** — `src/api/app.py` has the route wired but the InfluxDB query is incomplete.
- **Docker Compose end-to-end untested** — `docker/Dockerfile.consumer` has not been tested in a full `docker-compose up` run.
- **`instrument_loader` tokens all zero** — placeholder values; does not affect live inference since the inference consumer looks up stocks by symbol string, not token.

---

## File Directory

### Real-Time (market hours)

| File | Purpose |
|------|---------|
| `scripts/kite_to_kafka.py` | Authenticates with KiteConnect, subscribes to 47 stocks + NIFTY50, streams live ticks to Kafka |
| `src/consumers/viz_consumer.py` | Builds 1-minute OHLCV bars from ticks (BarAccumulator), writes bar history and live bar state to Redis |
| `src/consumers/inference_consumer.py` | Detects minute boundaries from Kafka ticks, reads Redis bars, runs TFT inference, writes quantile predictions to Redis |
| `src/consumers/persistence_consumer.py` | Batches ticks from Kafka and writes them to InfluxDB for historical storage |
| `src/api/app.py` | FastAPI server exposing `/predict`, `/health`, `/model/version` endpoints over REST |
| `scripts/dashboard.py` | Streamlit UI: market overview table/heatmap, stock detail candlestick chart, TFT predictions tab |
| `scripts/live_terminal.py` | Terminal-based live tick display for quick monitoring without the full dashboard |
| `scripts/verify_consumers.py` | Health check script: validates consumer group lag, Redis key presence, and model file |
| `src/models/model_manager.py` | Singleton that lazy-loads the TFT checkpoint, runs inference, caches predictions in Redis, and handles circuit-breaker fallback |
| `src/core/redis_client.py` | Thread-safe Redis connection pool singleton with all required key operations |
| `src/core/kafka_producer.py` | Kafka producer wrapper with reconnect backoff (used in tests; not the live producer) |
| `src/data/tick_validator.py` | Validates incoming tick dicts for required fields, data types, value ranges, and price-change outliers |
| `config/settings.py` | Pydantic BaseSettings loaded from `.env`; single source of truth for all service addresses, credentials, and tuning knobs |

### Overnight / Training

| File | Purpose |
|------|---------|
| `src/models/tft_model.py` | Full TFT architecture: GRN, VSN, LSTM encoder/decoder, multi-head attention, quantile output head |
| `src/models/training.py` | TFTTrainer: QuantileLoss, AdamW optimizer, CosineAnnealingLR, early stopping, collapse detection, checkpoint save |
| `src/data/dataset.py` | TFTDataset: sliding 60-bar windows, 11-feature engineering, z-scoring, gap detection, walk-forward split support |
| `src/data/features.py` | Standalone functions for RSI, MACD histogram, and ATR — shared by both training (dataset.py) and live inference |
| `src/data/instrument_loader.py` | Stock metadata cache (sector, market cap embeddings); seeded from hardcoded STOCK_META dict, cached in Redis |
| `scripts/fetch_historical_data.py` | Downloads 3 years of 1-minute OHLCV from the Zerodha API for all 47 stocks + NIFTY50 index; saves as parquet |
| `scripts/evaluate_model.py` | Walk-forward validation: directional accuracy, MAE, RMSE, quantile calibration across 6 non-overlapping 14-day windows |
| `scripts/overnight_eval.py` | Post-market evaluation comparing the day's live predictions against actual closing returns |
| `scripts/backfill_redis_bars.py` | One-time utility: seeds `STOCK:{sym}:bars` in Redis from InfluxDB history (used after Redis wipe or first setup) |
| `scripts/generate_model.py` | Generates a random (untrained) TFT checkpoint for pipeline testing without a full training run |
| `scripts/test_kafka.py` | Kafka connectivity smoke test |
| `tests/test_tft_model.py` | Unit tests for TFT forward pass: output shape (B, 5, 5), no NaNs, device portability |
| `tests/test_validators.py` | Unit tests for tick validation: required fields, type coercion, outlier rejection, batch processing |
