# Architectural Changes

## 1. Eliminate Private BarAccumulator in InferenceConsumer

- BarAccumulator is no longer a private tool inside InferenceConsumer.
- The BarAccumulator **class itself is not deleted** — its bar-building logic is moved into VizConsumer, which becomes the single place that converts ticks into bars.
- InferenceConsumer no longer touches raw ticks for bar construction. It only reads completed bars from Redis.
- PersistenceConsumer (InfluxDB) remains untouched — raw tick archive, only used for overnight eval and historical analysis. Zero involvement in the live inference path, zero involvement in startup.

---

## 2. VizConsumer becomes the Bar Builder

- VizConsumer receives every tick from Kafka (unchanged).
- On every tick it does two things:
  1. **Overwrites** `STOCK:{sym}:current_bar` hash with the running OHLCV of the bar currently being built — open (first tick of this minute), running high/low, current close (= latest LTP), volume so far. Updated on every single tick.
  2. **Builds bars** using the BarAccumulator logic internally. When a bar completes (minute boundary crossed), pushes it to `STOCK:{sym}:bars` and starts a fresh `current_bar`.

- `STOCK:{sym}` (the old plain LTP hash) is **replaced entirely** by `STOCK:{sym}:current_bar`. The current bar already contains close (= LTP), high, low, open, and volume — everything the dashboard previously read from the tick hash, and more. One less key to maintain.

- Bar storage key: `STOCK:{sym}:bars` — a Redis List, newest bar at index 0.
  - On bar complete: `LPUSH STOCK:{sym}:bars <bar_json>` then `LTRIM STOCK:{sym}:bars 0 299` (caps at 300 bars).
  - Each bar JSON: `{o, h, l, c, v, ts}` — ~120 bytes. Total memory: 300 bars × 48 symbols × 120 bytes ≈ **1.7 MB**.

- **Covers 47 stocks + Nifty50** (token 256265, now subscribed in kite_to_kafka.py). Nifty50 gets its own list: `STOCK:NIFTY50:bars`. This is what feeds col 10 (nifty_log_ret) into the TFT input tensor — no more hardcoded zeros.

- **Volume delta handling:** KiteConnect sends cumulative daily volume (`volume_traded`), not per-bar volume. VizConsumer must track `bar_start_vol` per stock and compute `bar_vol = cumulative_vol_now - bar_start_vol` — exactly as BarAccumulator currently does. This is the one non-trivial piece of the migration and must be ported carefully.

- Bars persist overnight via Redis AOF (`appendonly yes` already set in docker-compose). 300 bars of yesterday's session are available the moment the system starts — no InfluxDB query needed on startup.

---

## 3. Kafka as Trigger for InferenceConsumer

- InferenceConsumer still reads ticks from Kafka, but **only to detect minute-boundary crossings** — it no longer builds bars from them.
- When the first tick of a new minute arrives for stock X, InferenceConsumer knows bar (minute - 1) has completed and should be in Redis.

### Race condition: is the bar actually written yet?

VizConsumer and InferenceConsumer are separate processes reading the same Kafka topic independently. InferenceConsumer might see the "new minute" tick slightly before VizConsumer has finished writing the completed bar to Redis.

**Fix:** After detecting a minute change for stock X, InferenceConsumer reads `LINDEX STOCK:X:bars 0` (the newest bar) and checks its timestamp. If it matches the just-completed minute, proceed. If not, wait 100ms and retry once. In practice the retry will almost never fire — VizConsumer writes synchronously to Redis before moving to the next tick, so by the time the two processes are separated by a full tick-processing cycle the bar is already there.

---

## 4. InferenceConsumer simplified flow

On each minute-change detection for stock X:
1. `LRANGE STOCK:X:bars 0 99` — read last 100 bars from Redis (100 for full MACD warmup, inference window uses last 60).
2. `LRANGE STOCK:NIFTY50:bars 0 99` — read Nifty50 bars for col 10.
3. Compute all 11 features from the bar arrays (same logic as before, now with real Nifty50 data).
4. Run TFT inference.
5. Write predictions to `PRED:X:quantiles` and `PRED:X:steps` (unchanged).

`_warm_start_from_influxdb` is **deleted entirely**. Redis already has 300 bars from yesterday.

---

## 5. What stays the same

- `PRED:{sym}:quantiles` and `PRED:{sym}:steps` — written by InferenceConsumer, read by dashboard. Unchanged.
- PersistenceConsumer — completely unchanged. Still writes every raw tick to InfluxDB.
- Tick validation — still happens in VizConsumer before bar building.
- TFT model, feature computation, prediction logic — unchanged.

---

## 6. Redis key summary (post-change)

| Key | Type | Written by | Read by | TTL |
|-----|------|-----------|---------|-----|
| `STOCK:{sym}:current_bar` | Hash | VizConsumer (every tick) | Dashboard | 1 hour |
| `STOCK:{sym}:bars` | List (max 300) | VizConsumer (every bar) | InferenceConsumer, Dashboard | Persistent (AOF) |
| `PRED:{sym}:quantiles` | Hash | InferenceConsumer (every bar) | Dashboard | 120s |
| `PRED:{sym}:steps` | String (JSON) | InferenceConsumer (every bar) | Dashboard | 120s |

48 symbols (47 stocks + NIFTY50). Dashboard now has the current live candle plus full bar history — the live candle updates every tick, completed bars update every minute. This directly unblocks the TFT fan chart's historical context view and replaces the old single-LTP display with a proper candlestick.
