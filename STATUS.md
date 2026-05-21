# AlgoTrading — Status

_Last updated: 2026-05-21_

**Project:** Real-time stock price prediction system for Nifty 50. Live market ticks stream via KiteConnect WebSocket → Kafka → three consumers: a Temporal Fusion Transformer (TFT) inference engine that predicts 5-minute-ahead log-return quantiles, a Redis state aggregator for the UI, and an InfluxDB persistence layer. Deployed on a single machine via Docker Compose.

---

## Sub-task Status

| # | Component | Goal | Status |
|---|-----------|------|--------|
| 1 | KiteConnect WebSocket | Stream 50 Nifty stocks live, <100ms RTT | ✅ Done — avg 27.6ms, tested live |
| 2 | Kafka Pipeline | Reliable tick delivery, per-stock ordering | ✅ Done — 1576 ticks confirmed, p50 15.3ms RTT |
| 3 | Redis Consumer | Live `STOCK:*` state updated per tick | ✅ Done — 49 keys verified |
| 4 | InfluxDB Consumer | Persist all ticks to time-series DB | ✅ Done — 97% survival rate after dedup fix |
| 5 | Unit Tests | TFT forward pass + validator tests green | ✅ Done — 5/5 passing |
| 6 | Historical Data | 3yr 1-min OHLCV for all Nifty 50 + index | ✅ Done — 47/50 stocks + NIFTY50 (277K candles) |
| 7 | TFT Architecture | Full paper impl, correct I/O shapes | ✅ Done — 462K params, output `(B,5,5)` verified |
| 8 | Dataset Loader | Sliding window with all 11 features, train/val split | ✅ Done — gap detection, z-scoring, walk-forward support |
| 9 | TFT Training | 47-stock model with improving val loss | ✅ Done — full run complete, best val=0.213697, p50_std peaked 0.095, checkpoint safe for inference |
| 10 | Inference Consumer | 1-min bar accumulator matching training features exactly | ✅ Written — untested live (needs trained model) |
| 11 | Viz Consumer | Kafka → Redis pub/sub state | ✅ Fixed — clean shutdown loop, Windows-safe signals; ready to test live |
| 12 | Persistence Consumer | Kafka → InfluxDB batch writes | ✅ Written — untested (missing offset commit) |
| 13 | Model Manager | Load TFT weights, serve predictions, Redis cache | ✅ Written — untested (singleton blocker) |
| 14 | FastAPI | `/predict`, `/health`, `/history` endpoints | ✅ Written — untested (`/history` stub incomplete) |
| 15 | Docker Compose | Full pipeline orchestrated, all services healthy | ⚠️ Partial — Dockerfiles for producer/consumer missing |
| 16 | Walk-forward Eval | Consistent directional accuracy >50% across regimes | ⏳ Pending — needs trained model |
| 17 | Live Inference Test | `PRED:*:quantiles` in Redis within 60min of market open | ⏳ Pending — needs trained model + market hours |
| 18 | Streamlit Dashboard | Hybrid heatmap+table UI with candlestick drill-down | ✅ Done — research-backed redesign, 3-tab layout |

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

## UI / Dashboard

**File:** `scripts/dashboard.py` — run with `streamlit run scripts/dashboard.py`

**Design basis:** Researched Bloomberg Terminal, TradingView, Zerodha Kite, Finviz, and UX literature on real-time financial dashboards. Conclusion: hybrid table + heatmap + drill-down is optimal for 50 stocks.

| Tab | What it shows |
|-----|---------------|
| Market Overview → Table View | All/Gainers/Losers filter; sorted sector → symbol; row click selects stock |
| Market Overview → Heatmap View | Plotly Treemap — cell size = volume, color = Chg% (red↔green); click to select stock |
| Stock Detail | Today's OHLC candlestick (close = live LTP) + session sparkline (activates after 5 refreshes) + TFT quantile scatter |
| TFT Predictions | Full P10–P90 table for all stocks; populates once inference consumer is running |

**Sidebar:** LTP/OHLCV detail for selected stock, sector badge, stock switcher dropdown, TFT horizontal bar chart (when predictions available).

**Auto-refresh:** 1–30s slider (default 2s), pauseable. Session sparklines accumulate LTP history in `st.session_state` (up to ~6.7 min at 2s refresh).

**Dependencies:** No new packages — uses Plotly 6.7.0 (already installed) for all charts. Requires `jinja2>=3.1.5` for table color coding (added to `requirements.txt`).

---

## Known Issues

- **Module-load singletons** — `redis_client` and `kafka_producer` are now lazy (fixed). `model_manager` and `instrument_loader` still eagerly connect on import — needs lazy init before running inference consumer standalone.
- **Missing offset commit** — `persistence_consumer.py` never calls `consumer.commit()`, messages re-process on restart.
- **Kafka partition count** — fresh Docker start auto-creates `stock-quotes` with 1 partition; must be manually set to 25.
- **3 missing stocks** — LTIM, TATAMOTORS, ZOMATO not fetched (47/50 present).
- **`instrument_loader.py`** — hardcodes 6 stocks; needs full KiteConnect instrument API.

---

## Next Steps

1. **Walk-forward eval** — `py scripts/evaluate_model.py --walk_forward 6`
   _Done when: directional accuracy >50% consistently across all 6 windows (not just one lucky period)._

2. **Fix inference consumer singletons** — `model_manager` and `instrument_loader` eagerly connect on import; needs lazy init before running inference consumer standalone.
   _Done when: `py -m src.consumers.inference_consumer` starts without errors outside Docker._

3. **Live inference test** — run KiteConnect producer + inference consumer together during market hours
   _Done when: `PRED:<SYMBOL>:quantiles` keys appear in Redis for all 47 stocks within 60 minutes of market open (60-bar warmup required)._

4. **Test viz consumer live** — run producer + viz_consumer together, then `streamlit run scripts/dashboard.py`
   _Done when: dashboard shows live LTP/OHLCV for all 47+ stocks updating every tick._

5. **Fix missing offset commit** in `persistence_consumer.py` — messages re-process on restart.

6. **Docker Compose** — add missing Dockerfiles for producer and consumers.
