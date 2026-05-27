# AlgoTrading — Status

_Last updated: 2026-05-28 (session 4 — architecture overhaul)_

**Project:** Real-time stock price prediction system for Nifty 50. Live market ticks stream via KiteConnect WebSocket → Kafka → three consumers: a Temporal Fusion Transformer (TFT) inference engine that predicts 5-minute-ahead log-return quantiles, a Redis state aggregator for the UI, and an InfluxDB persistence layer. Deployed on a single machine via Docker Compose.

---

## Sub-task Status

| # | Component | Goal | Status |
|---|-----------|------|--------|
| 1 | KiteConnect WebSocket | Stream 50 Nifty stocks live, <100ms RTT | ✅ Done — avg 27.6ms, tested live |
| 2 | Kafka Pipeline | Reliable tick delivery, per-stock ordering | ✅ Done — 1576 ticks confirmed, p50 15.3ms RTT |
| 3 | Redis Consumer | Rolling bar cache + live bar per symbol | ✅ Overhauled — `STOCK:*:current_bar` (tick-level) + `STOCK:*:bars` (300-bar rolling list, AOF-persistent) |
| 4 | InfluxDB Consumer | Persist all ticks to time-series DB | ✅ Done — 97% survival rate after dedup fix |
| 5 | Unit Tests | TFT forward pass + validator tests green | ✅ Done — 5/5 passing |
| 6 | Historical Data | 3yr 1-min OHLCV for all Nifty 50 + index | ✅ Done — 47/50 stocks + NIFTY50 (277K candles) |
| 7 | TFT Architecture | Full paper impl, correct I/O shapes | ✅ Done — 462K params, output `(B,5,5)` verified |
| 8 | Dataset Loader | Sliding window with all 11 features, train/val split | ✅ Done — gap detection, z-scoring, walk-forward support |
| 9 | TFT Training | 47-stock model with improving val loss | ✅ Done — full run complete, best val=0.213697, p50_std peaked 0.095, checkpoint safe for inference |
| 10 | Inference Consumer | Reads Redis bars, detects minute boundary, runs TFT | ✅ Overhauled — reads `STOCK:*:bars` from Redis; `_warm_start_from_influxdb` deleted; ~370 lines removed |
| 11 | Viz Consumer | Single bar-builder for entire system | ✅ Overhauled — owns BarAccumulator; writes `current_bar` every tick + `bars` list every minute |
| 12 | Persistence Consumer | Kafka → InfluxDB batch writes | ✅ Fixed — offset commit added; ready to test live |
| 13 | Model Manager | Load TFT weights, serve predictions, Redis cache | ✅ Fixed — lazy init applied; ready to test live |
| 14 | FastAPI | `/predict`, `/health`, `/history` endpoints | ✅ Written — untested (`/history` stub incomplete) |
| 15 | Docker Compose | Full pipeline orchestrated, all services healthy | ⚠️ Partial — Kafka partitions fixed (25); Dockerfiles for producer/consumer still unverified |
| 16 | Walk-forward Eval | Consistent directional accuracy >50% across regimes | ✅ Done — 52.65% mean dir. accuracy, std 0.37%, range 52.18–53.18% across 6 windows |
| 17 | Live Inference Test | `PRED:*:quantiles` in Redis within 60min of market open | ✅ Done — 143K+ ticks processed, all 47 stocks predicting live |
| 18 | Streamlit Dashboard | Hybrid heatmap+table UI with candlestick drill-down | ✅ Updated — Stock Detail now shows 30 completed 1-min bars + live bar; reads new Redis keys |

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

Three confirmed bugs were in `src/models/training.py` across all previous runs:

1. **Adam + weight_decay = silent prediction collapse.** `Adam(weight_decay=1e-4)` applies L2 regularisation inside the adaptive update. Once the model's output predictions start converging, the output head gradients shrink, Adam's internal momentum amplifies the L2 force, and weights get pushed to zero — making predictions even more constant. A feedback loop. Fix: switched to `AdamW` (decoupled weight decay), which breaks the loop. Validated against the pytorch-forecasting reference library, which defaults to `weight_decay=0` precisely because of this fragility.

2. **Gradient clipping at 1.0 instead of 0.1.** The reference library recommends `gradient_clip_val=0.1`. Clipping at 1.0 allows 10x larger gradient steps through the attention layers, which can cause attention entropy collapse (the model fixates on a single position in its context window). Fix: `max_norm=1.0` → `max_norm=0.1`.

3. **No collapse early-warning.** The existing alarm only fired at `p50_std < 1e-4` — far too late (Run 3 collapse started at epoch 1 and the alarm would have fired ~20 hours later). Fix: added monotonic decline detection (3 consecutive epoch drops triggers immediate warning).

Additional safeguards added: soft quantile spread diversity loss (penalises P90−P10 < 0.05), checkpoint p50_std guard (auto-skips warm-start if saved checkpoint has collapsed predictions), `--no-warmstart` CLI flag.

### Why early stopping at epoch 19 is expected, not a bug

Full Run 4 best val was epoch 4 (0.213697). Val loss then oscillated in a 0.0003 band for 15 epochs without improving — patience=15 fired correctly. The tight band means the model reached its genuine generalisation ceiling early; the remaining epochs improved training fit but not held-out performance. This is normal for stock log-return prediction. The checkpoint is at the best generalisation point.

---

## Session 4 — 2026-05-28 (Architecture Overhaul)

### Changes implemented

**Architectural overhaul — Redis as shared bar cache (Option A from session 3):**

- **VizConsumer** is now the single bar-builder for the entire system. It owns `BarAccumulator` (moved from InferenceConsumer). Every tick writes `STOCK:{sym}:current_bar` (running OHLCV + day-level KiteConnect data). Every completed bar does `LPUSH STOCK:{sym}:bars` + `LTRIM` to 300. Handles all 47 stocks + NIFTY50 identically.

- **InferenceConsumer** now reads bars from Redis instead of maintaining private in-memory accumulators. `_warm_start_from_influxdb` (~160 lines) deleted entirely — Redis AOF persistence means 300 bars from the previous session survive overnight automatically. Startup is instant. Minute-boundary detection via Kafka tick timestamps triggers `LRANGE STOCK:{sym}:bars 0 99` + inference. ~370 lines net removed.

- **Redis key schema changed:**

  | Key | Type | Written by | Read by |
  |-----|------|-----------|---------|
  | `STOCK:{sym}:current_bar` | Hash | VizConsumer (every tick) | Dashboard, InferenceConsumer |
  | `STOCK:{sym}:bars` | List max 300 | VizConsumer (every bar) | InferenceConsumer, Dashboard |
  | `PRED:{sym}:quantiles` | Hash | InferenceConsumer | Dashboard |
  | `PRED:{sym}:steps` | String JSON | InferenceConsumer | Dashboard |

  Old `STOCK:{sym}` tick hash removed entirely.

- **`redis_client.py`** — added `lpush`, `ltrim`, `lrange`, `lindex` methods to the wrapper.

- **Dashboard** — `fetch_stocks()` now reads `STOCK:*:current_bar`; symbol discovery updated; NIFTY50 filtered from table. Stock Detail tab replaced single-candle day chart with a 30-bar 1-min candlestick chart (completed bars in green/red + live current bar in blue/orange, updated every tick).

**Nifty50 subscription added (2026-05-28):**
- `kite_to_kafka.py` now subscribes token 256265 (`NSE:NIFTY 50`) in `MODE_QUOTE`. VizConsumer writes `STOCK:NIFTY50:bars` — col 10 (nifty_log_ret) in the TFT input tensor is now real data instead of zeros.

**Mock infrastructure removed:**
- Deleted `scripts/producer.py`, `scripts/kafka_live_view.py`, `docker/Dockerfile.producer`, producer service from `docker-compose.yml`.

### Overnight eval — 2026-05-27 session
- 47.6% directional accuracy (below 52.65% walk-forward benchmark)
- Likely causes: col 10 (Nifty50 log returns) was zeros throughout the session since NIFTY50 was not subscribed; mock producer contamination on 2026-05-26 (two prior days of training data were partly synthetic)
- Quantile calibration remains excellent (P10 empirical 10.7%, P90 89.6%)

---

## Session 3 — 2026-05-27 (First Full Live Session)

### What ran today
- Full pipeline live for the first time end-to-end: KiteConnect → Kafka → inference consumer + persistence consumer + viz consumer → Redis → Streamlit dashboard
- All 47 Nifty 50 stocks streaming live, inference consumer processed 143,000+ ticks
- InfluxDB persistence confirmed working — data stored for overnight evaluation
- `jinja2` version mismatch found and fixed (`pandas 3.0.3` requires `>=3.1.5`, had `3.1.4`)

### Dashboard redesign (session 3)

**Market Overview table:**
- Fixed blank heatmap — was using `values=volume` but all volumes reported as 0 at session start; switched to equal-area sizing (`values=[1]*n_stocks`), all 47 boxes now always visible
- Fixed color coding — `Symbol`, `LTP`, `Dir` columns now colored green/red by gain/loss direction (was plain white); required `jinja2>=3.1.5`

**TFT Predictions tab — full redesign:**
- Inference consumer now stores all 5 prediction steps (`PRED:{symbol}:steps` as JSON) in addition to existing single-step hash — previously only t+1 was stored, t+2 through t+5 were computed but discarded
- Dedicated `@st.fragment(run_every=60)` for TFT tab — previously inside the 2-second market data fragment, making the chart reload every 2 seconds and resetting zoom/pan; now refreshes every 60 seconds (aligned with 1-minute bar prediction cadence)
- Stock picker inside the TFT tab (all 47 symbols), syncs with sidebar
- Forecast chart redesigned to ΔPrice space: Y-axis = ₹ change from current LTP, zero line = current price, green bar extends up (bullish), red bar extends down (bearish) — eliminates the previous confusion where a "green" bar appeared to straddle the zero line
- Per-step quantile table below chart with compounded implied prices
- Market-wide P50 direction strip at bottom
- Sidebar prediction mini-chart replaced with a compact colored pill (↑ Bullish / ↓ Bearish + %)

### Architectural gap identified
The viz consumer only writes the **latest tick state** to Redis (`STOCK:{SYMBOL}` — one hash, overwritten every tick). It does not accumulate bar history. The inference consumer does **not** read bars from Redis — it maintains its own private in-memory `BarAccumulator` per stock, built live from Kafka ticks, and warm-starts from InfluxDB on startup.

This means **Redis is not being used as the shared bar cache** as originally intended. The dashboard has no access to historical bar data without querying InfluxDB. The intended architecture (viz consumer writes rolling bar history to Redis → inference consumer reads from Redis → dashboard reads same bars for chart context) was never built.

**Pending decision:** (a) refactor viz consumer to maintain rolling bar lists in Redis and have inference consumer read from Redis, or (b) keep inference consumer's private in-memory accumulators and just add a tick history list to Redis purely for dashboard use.

---

## UI / Dashboard

**File:** `scripts/dashboard.py` — run with `streamlit run scripts/dashboard.py`

**Design basis:** Researched Bloomberg Terminal, TradingView, Zerodha Kite, Finviz, and UX literature on real-time financial dashboards. Conclusion: hybrid table + heatmap + drill-down is optimal for 50 stocks.

| Tab | What it shows |
|-----|---------------|
| Market Overview → Table View | All/Gainers/Losers filter; sorted sector → symbol; row click selects stock |
| Market Overview → Heatmap View | Plotly Treemap — cell size = volume, color = Chg% (red↔green); click to select stock |
| Stock Detail | 30-bar 1-min candlestick from Redis (completed bars + live bar in blue) + session sparkline + TFT quantile scatter |
| TFT Predictions | Full P10–P90 table for all stocks; populates once inference consumer is running |

**Sidebar:** LTP/OHLCV detail for selected stock, sector badge, stock switcher dropdown, TFT horizontal bar chart (when predictions available).

**Auto-refresh:** 1–30s slider (default 2s), pauseable. Session sparklines accumulate LTP history in `st.session_state` (up to ~6.7 min at 2s refresh).

**Dependencies:** Requires `jinja2>=3.1.5` for pandas `.style` color coding (pandas 3.0.3 enforces this minimum). Previously had 3.1.4 which silently broke all table styling.

---

## Known Issues / Open Questions

### Active

- **47.6% live dir. accuracy vs 52.65% benchmark** — first live session underperformed. Root causes: col 10 was zeros (Nifty50 not subscribed), and two prior training-data days had mock-producer contamination. Fixed in session 4. Next live session should produce a cleaner result.
- **Fan chart not yet implemented** — TFT Predictions tab uses bar/whisker chart in ΔPrice space. Proper P10–P90 filled cone with historical bar context not yet built (Redis bar history is now available).
- **3 missing stocks** — LTIM, TATAMOTORS, ZOMATO excluded from checkpoint `target_stats`. Do not add until data is fetched and model is retrained.
- **`/history` endpoint stub** — `src/api/app.py` has the endpoint wired but InfluxDB query is incomplete; returns placeholder data.
- **Docker Compose Dockerfiles** — `docker/Dockerfile.consumer` untested in a full `docker-compose up` end-to-end run.
- **`instrument_loader` tokens** — all instrument tokens set to `0` (placeholder). Does not block live inference since inference consumer now reads from Redis by symbol string, not token.

### Fixed (2026-05-28)

- ~~Redis not used as shared bar cache~~ — VizConsumer now builds bars and writes `STOCK:*:bars` (300-bar rolling list) + `STOCK:*:current_bar`. InferenceConsumer reads from Redis.
- ~~InferenceConsumer warm-start requires InfluxDB~~ — `_warm_start_from_influxdb` deleted; Redis AOF persistence handles overnight bar survival automatically.
- ~~col 10 always zeros in live inference~~ — NIFTY50 token 256265 now subscribed in `kite_to_kafka.py`, bars stored in `STOCK:NIFTY50:bars`.
- ~~Dashboard shows only latest-tick data, no bar history~~ — Stock Detail tab now shows 30-bar 1-min candlestick chart from Redis.
- ~~Mock producer in docker-compose~~ — `producer` service, `scripts/producer.py`, `scripts/kafka_live_view.py`, `docker/Dockerfile.producer` all deleted.

### Fixed (2026-05-23)

- ~~Module-load singletons~~ — `model_manager` and `instrument_loader` now use lazy init; importing either file no longer connects to Redis or loads PyTorch.
- ~~Missing offset commit~~ — `persistence_consumer.py` now calls `consumer.commit()` after each successful InfluxDB write.
- ~~Kafka partition count~~ — `docker-compose.yml` now sets `KAFKA_CREATE_TOPICS: "stock-quotes:25:1"`; topic is created with 25 partitions at broker startup. ⚠️ Run `docker-compose down -v` if you have an existing volume with 1 partition.
- ~~`instrument_loader.py` hardcodes 6 stocks~~ — now derives all 47 stocks from `STOCK_META` via `settings.stock_list`.
- ~~`settings.py` hardcodes 6 stocks~~ — default expanded to all 47 trained stocks.
- ~~Walk-forward eval bug~~ — `_build_window_dataset` in `evaluate_model.py` had `val_days`/`val_start_days` swapped, returning 0 windows. Fixed.
- ~~`.env` stock list wrong~~ — had 50 stocks including 4 not in the trained model (BPCL, SHREECEM, TATAMOTORS, UPL) and missing BEL. Corrected to exactly the 47 trained stocks.
- ~~`kite_to_kafka.py` fallback~~ — default stock list was 3 stocks hardcoded. Updated fallback to all 47 trained stocks.
- ~~`.env.example` stock list~~ — updated from old 6-stock placeholder to all 47 trained stocks.

---

## Pre-flight Checklist (complete before next market session)

| # | Item | Status |
|---|------|--------|
| ✅ | Walk-forward eval passed (52.65% mean dir. accuracy, std 0.37%) | Done |
| ✅ | `.env` STOCKS corrected to 47 trained stocks (removed BPCL, SHREECEM, TATAMOTORS, UPL; added BEL) | Done |
| ✅ | `kite_to_kafka.py` fallback updated to 47 stocks | Done |
| ✅ | All code bugs fixed (singletons, offset commit, Kafka partitions, eval script) | Done |
| ⬜ | KiteConnect credentials confirmed in `.env` (API key, secret, user ID, password, **TOTP base32 secret**) | Manual — must verify before live run |
| ⬜ | `docker-compose down -v` run once to clear old Kafka volume | One-time — do before first `docker-compose up` |
| ⬜ | `pip install -r requirements.txt` confirmed up to date | Run once to confirm |

> **TOTP note:** `KITE_TWO_FA` must be the base32 *secret* from your authenticator app setup (long string like `JBSWY3DPEHPK3PXP`), not the rotating 6-digit code. `kite_to_kafka.py` uses `pyotp` to generate the code automatically.

## Next Steps

1. **Live session validation** _(next market day — session 5)_
   - Run with new architecture: NIFTY50 subscribed, col 10 real data, Redis bar cache live
   - Check inference triggers at every minute boundary (logs should show `_trigger_inference` calls)
   - Confirm `STOCK:*:bars` populating in Redis (`redis-cli LLEN STOCK:RELIANCE:bars` should reach 60+ within first hour)
   - Run overnight eval after close: expect dir. accuracy closer to 52.65% benchmark now that col 10 is real

2. **TFT fan chart** _(dashboard — Redis bar data now available)_
   - Proper P10–P90 filled cone, P30–P70 inner band, P50 line
   - Show last 5 completed bars as context to the left of the forecast horizon
   - All in absolute price (₹), Y-axis auto-fit

3. **overnight_eval.py cleanup** _(deferred)_
   - Update default date from 2026-05-26 → today's date (dynamic)
   - Fetch real NIFTY50 bars from InfluxDB for col 10 in feature matrix

4. **Docker Compose end-to-end** _(Phase 4)_
   ```
   docker-compose down -v    # clears old Kafka volume — one-time
   docker-compose up -d
   docker-compose ps         # all services should show healthy
   ```
   _`docker/Dockerfile.consumer` untested end-to-end — may need minor path/dependency fixes._
