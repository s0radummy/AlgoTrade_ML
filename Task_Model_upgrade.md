# On your inputs and target

## Predict returns, not prices.

This is the single most important change. If your target is LTP itself, you're asking a deep network to learn that "tomorrow's price ≈ today's price + small noise," and what it will dutifully learn is the naive persistence forecast that just outputs the last value. You'll get pretty-looking plots where the prediction line tracks the actual line with one-step lag, and your RMSE will look amazing, but the model has learned nothing tradeable. In investing we use returns not price because of non stationarity and auto-correlation of prices. Neural Networks like RNN/LSTM/GRU do not explicitly require the condition of stationarity in fact it doesn't require you to make any assumptions about the underlying distribution. In theory you do not need to make any of these transformations if you have enough data and the right architecture but in practice it is still very hard to predict non stationary time series. Use log-returns (log(p*t / p*{t-1})) as the target. They're approximately stationary, roughly mean-zero, and the model is forced to actually predict something rather than copy the last value. medium

## Think hard about what a "tick" means here.

Kite Connect streams quotes at roughly 200–400ms intervals during liquid hours, and at the true tick level you're mostly modeling market microstructure noise — bid-ask bounce, order-book reshuffles, latency artifacts — which has almost no predictive structure beyond a few hundred milliseconds. High-frequency data reflect rapid, noise-driven fluctuations, while lower-frequency data capture slower, macro-driven trends. These dynamics are often incompatible, and models designed for a single resolution typically fail to generalize across different temporal resolutions. I'd strongly suggest you resample to fixed-interval bars (1-second, 5-second, or 1-minute depending on what horizon you actually care about) and predict on those bars. Your last-60-ticks Redis window then becomes "last 60 bars," which is a much cleaner signal. If you want a tick-level model, you'll need a fundamentally different feature set (order book imbalance, signed volume, trade intensity) and you'll be working in HFT territory where TFT is not necessarily the best tool.

## Add a few features your current list is missing.

Realised volatility over a short rolling window (this lets the model condition on regime), log-volume (raw volume is heavy-tailed and will hurt training), spread or high-low range as a microstructure proxy, and an index return (Nifty50 itself) as a cross-sectional anchor — if every stock in your panel knows what the index is doing, it can separate idiosyncratic from systematic moves. Sector is fine as a static categorical, but consider also market-cap bucket and average-daily-volume bucket as static covariates; the cross-stock variation in tick behavior between Reliance and a mid-cap like, say, UPL is enormous, and giving TFT a static handle on this lets the variable selection network pick it up.
Normalize per-stock, not globally. Each stock has its own price scale, volume scale, and volatility regime. The standard approach with TFT (and what pytorch-forecasting's TimeSeriesDataSet does for you with GroupNormalizer) is to compute z-score normalization within each group_ids=[stock_id]. If you're rolling your own, do this — don't just StandardScaler the whole panel.

# On the output and forecasting horizon

## Use more quantiles.

P10/P50/P90 gives you a median and an 80% band, but P05/P25/P50/P75/P95 (the pytorch-forecasting default) costs almost nothing extra in parameters and gives you a much better view of tail risk, which is what you actually care about in trading. Add P50 implicitly — never train just on 3 quantiles when 7 is essentially free.

## Predict log-returns at each horizon

Only after that, then reconstruct prices. If you predict log-returns r*{t+1}, ..., r*{t+5}, your reconstructed price quantile at horizon h is p_t \* exp(sum of r's). This compounds the uncertainty correctly and avoids the model collapsing to "predict the current price."

## Be explicit about "5 ticks."

If those are tick-by-tick (~200ms apart), you're forecasting about a second ahead, and you're competing with the limit order book — TFT will not help you there. If those are 5 one-minute bars, that's a 5-minute-ahead forecast, which is a defensible horizon for a statistical model. If they're 5 five-minute bars (25 minutes), even better. Pick a horizon based on what you're actually trying to do with the prediction, not on what was a default in some tutorial.

## Always benchmark against:

- Naive forecast (r\_{t+h} = 0, i.e., expect no change)
- Persistence (r\_{t+h} = r_t)
- A simple LSTM on the same inputs
- A gradient-boosted tree (LightGBM) on hand-crafted features

If your TFT doesn't beat all four on a held-out walk-forward backtest, the TFT isn't earning its complexity yet. And do walk-forward validation, not random splits — random splits leak future information through the rolling features and will give you fantasy results.
