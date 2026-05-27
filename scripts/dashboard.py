"""
dashboard.py — Real-time AlgoTrading dashboard backed by Redis.

Architecture:
  - Main script   : renders sidebar + header ONCE (static, no auto-refresh)
  - @st.fragment  : refreshes only the data panels every N seconds
  - Result        : sidebar selectbox is never interrupted by the data refresh

Run:
    py -m streamlit run scripts/dashboard.py --server.headless true --browser.gatherUsageStats false

Prerequisites:
    - Redis   : docker-compose up -d redis
    - Ticks   : py -m src.consumers.viz_consumer
    - Kafka   : py scripts/kite_to_kafka.py
"""

import json
import math
import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import redis
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
MAX_HISTORY = 200

# ── Nifty 50 sector mapping ───────────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT",
    "HDFCBANK": "Finance", "ICICIBANK": "Finance", "KOTAKBANK": "Finance",
    "AXISBANK": "Finance", "SBIN": "Finance", "BAJFINANCE": "Finance",
    "BAJAJFINSV": "Finance", "HDFCLIFE": "Finance", "SBILIFE": "Finance",
    "INDUSINDBK": "Finance",
    "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "COALINDIA": "Energy",
    "LT": "Industrials", "ADANIPORTS": "Industrials", "BEL": "Industrials",
    "HINDUNILVR": "Consumer", "ITC": "Consumer", "NESTLEIND": "Consumer",
    "BRITANNIA": "Consumer", "TATACONSUM": "Consumer", "ASIANPAINT": "Consumer",
    "TITAN": "Consumer", "MARUTI": "Consumer", "HEROMOTOCO": "Consumer",
    "M&M": "Consumer", "BAJAJ-AUTO": "Consumer", "EICHERMOT": "Consumer",
    "ZOMATO": "Consumer", "TATAMOTORS": "Consumer",
    "TATASTEEL": "Materials", "JSWSTEEL": "Materials", "HINDALCO": "Materials",
    "GRASIM": "Materials", "ULTRACEMCO": "Materials",
    "SUNPHARMA": "Healthcare", "DRREDDY": "Healthcare", "CIPLA": "Healthcare",
    "DIVISLAB": "Healthcare", "APOLLOHOSP": "Healthcare",
    "BHARTIARTL": "Telecom",
    "ADANIENT": "Conglomerate",
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AlgoTrading Dashboard", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("<style>[data-testid='stMetricValue']{font-size:1.25rem}</style>",
            unsafe_allow_html=True)

st.session_state.setdefault("selected_stock", None)
st.session_state.setdefault("price_history",  {})
st.session_state.setdefault("refresh_count",  0)
st.session_state.setdefault("main_view",      "Market Overview")


# ── Redis helpers ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def redis_ok(r: redis.Redis) -> bool:
    try:
        r.ping(); return True
    except Exception:
        return False

def fetch_stocks(r: redis.Redis) -> pd.DataFrame:
    try:
        keys = sorted(r.keys("STOCK:*:current_bar"))
    except Exception:
        return pd.DataFrame()
    if not keys:
        return pd.DataFrame()
    pipe = r.pipeline()
    for k in keys: pipe.hgetall(k)
    rows = []
    for key, data in zip(keys, pipe.execute()):
        if not data: continue
        sym = key[6:-12]   # strip "STOCK:" prefix and ":current_bar" suffix
        if not sym or sym == "NIFTY50":
            continue
        ltp   = float(data.get("LTP",   0) or 0)
        close = float(data.get("Close", 0) or 0)
        chg   = ((ltp - close) / close * 100) if close else float(data.get("Change", 0) or 0)
        ts    = data.get("Last_Updated", "")
        if "T" in ts: ts = ts.split("T")[1][:8]
        rows.append({"Symbol": sym, "LTP": ltp,
                     "Open":   float(data.get("Open",   0) or 0),
                     "High":   float(data.get("High",   0) or 0),
                     "Low":    float(data.get("Low",    0) or 0),
                     "Close":  close,
                     "Volume": int(float(data.get("Volume", 0) or 0)),
                     "Chg%":   round(chg, 2), "Updated": ts})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def fetch_bars(r: redis.Redis, symbol: str, n: int = 30) -> list:
    """Read last n completed 1-min bars from STOCK:{sym}:bars (chronological order)."""
    try:
        raw = r.lrange(f"STOCK:{symbol}:bars", 0, n - 1)
        bars = []
        for item in reversed(raw):
            try:
                bars.append(json.loads(item))
            except Exception:
                pass
        return bars
    except Exception:
        return []

def fetch_current_bar(r: redis.Redis, symbol: str) -> dict:
    """Read STOCK:{sym}:current_bar hash from Redis."""
    try:
        return r.hgetall(f"STOCK:{symbol}:current_bar") or {}
    except Exception:
        return {}

def fetch_forecast_steps(r: redis.Redis, symbol: str) -> dict | None:
    """Read multi-step forecast (all 5 steps) from PRED:{symbol}:steps."""
    try:
        raw = r.get(f"PRED:{symbol}:steps")
        return json.loads(raw) if raw else None
    except Exception:
        return None

def fetch_predictions(r: redis.Redis, symbols: list) -> pd.DataFrame:
    if not symbols: return pd.DataFrame()
    pipe = r.pipeline()
    for s in symbols: pipe.hgetall(f"PRED:{s}:quantiles")
    rows = []
    for sym, data in zip(symbols, pipe.execute()):
        if not data: continue
        rows.append({"Symbol": sym,
                     "P10": float(data.get("P10", 0) or 0),
                     "P30": float(data.get("P30", 0) or 0),
                     "P50": float(data.get("P50", 0) or 0),
                     "P70": float(data.get("P70", 0) or 0),
                     "P90": float(data.get("P90", 0) or 0)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def market_status() -> tuple[str, str]:
    now = datetime.now()
    t, wd = now.hour * 60 + now.minute, now.weekday() < 5
    if wd and (9*60+15) <= t <= (15*60+30): return "open",   "🟢 Market Open"
    if wd and (9*60)    <= t <  (9*60+15):  return "pre",    "🟡 Pre-market"
    return "closed", "🔴 Market Closed"


# ── Chart builders ────────────────────────────────────────────────────────────
def _build_heatmap(df: pd.DataFrame) -> go.Figure:
    syms     = df["Symbol"].tolist()
    vols     = df["Volume"].tolist()
    chgs     = df["Chg%"].tolist()
    ltps     = df["LTP"].tolist()
    sectors  = [SECTOR_MAP.get(s, "Other") for s in syms]
    u_sec    = sorted(set(sectors))
    n        = len(u_sec)
    labels   = syms + u_sec
    parents  = sectors + [""] * n
    values   = [1] * len(syms) + [0] * n   # equal-area: volume shown only in hover
    colors   = chgs + [0.0] * n
    custom   = [[ltps[i], chgs[i], vols[i]] for i in range(len(syms))] + [[0,0,0]]*n
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values, customdata=custom,
        marker=dict(colors=colors,
                    colorscale=[[0,"#DC3545"],[0.5,"#1a1a2e"],[1,"#28A745"]],
                    cmid=0, cmin=-3.0, cmax=3.0, showscale=True,
                    colorbar=dict(title="Chg%", thickness=14, len=0.8)),
        texttemplate="<b>%{label}</b><br>%{customdata[1]:+.2f}%",
        hovertemplate=("<b>%{label}</b><br>LTP: ₹%{customdata[0]:.2f}<br>"
                       "Chg: %{customdata[1]:+.2f}%<br>Volume: %{customdata[2]:,.0f}"
                       "<extra></extra>"),
        tiling=dict(packing="squarify"),
    ))
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=10,b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def _build_multi_bar_chart(bars: list, current_bar_data: dict, symbol: str) -> go.Figure:
    """
    Multi-bar 1-min candlestick: completed bars (green/red) + live current bar (blue/orange).
    bars: chronological list of {open, high, low, close, volume, ts} dicts.
    current_bar_data: STOCK:{sym}:current_bar Redis hash.
    """
    fig = go.Figure()

    if bars:
        times  = [b.get("ts", "") for b in bars]
        opens  = [b["open"]  for b in bars]
        highs  = [b["high"]  for b in bars]
        lows   = [b["low"]   for b in bars]
        closes = [b["close"] for b in bars]
        fig.add_trace(go.Candlestick(
            x=times, open=opens, high=highs, low=lows, close=closes,
            name="Completed",
            increasing_line_color="#28A745", decreasing_line_color="#DC3545",
            increasing_fillcolor="#28A745",  decreasing_fillcolor="#DC3545",
        ))

    # Live current bar (in-progress, shown in blue/orange)
    bar_open  = float(current_bar_data.get("bar_open",  0) or 0)
    bar_high  = float(current_bar_data.get("bar_high",  0) or 0)
    bar_low   = float(current_bar_data.get("bar_low",   0) or 0)
    bar_close = float(current_bar_data.get("bar_close", 0) or 0)
    bar_ts    = current_bar_data.get("bar_ts", "")

    if bar_close > 0:
        live_x = bar_ts if bar_ts else "live"
        fig.add_trace(go.Candlestick(
            x=[live_x],
            open=[bar_open], high=[bar_high], low=[bar_low], close=[bar_close],
            name="Live bar",
            increasing_line_color="#17a2b8", decreasing_line_color="#fd7e14",
            increasing_fillcolor="#17a2b8",  decreasing_fillcolor="#fd7e14",
            opacity=0.85,
        ))

    n = len(bars)
    note = f"{n} completed bars" if n > 0 else "waiting for first bar…"
    fig.update_layout(
        title=f"{symbol} — 1-min bars  ·  {note}  ·  live bar in blue",
        height=420, xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=44, b=0),
        legend=dict(orientation="h", y=1.06),
        xaxis=dict(type="category", tickangle=-45, nticks=10),
    )
    return fig

def _build_sparkline(history: list, symbol: str) -> go.Figure | None:
    if len(history) < 5: return None
    color = "#28A745" if history[-1] >= history[0] else "#DC3545"
    fill  = "rgba(40,167,69,0.1)" if history[-1] >= history[0] else "rgba(220,53,69,0.1)"
    y_min, y_max = min(history), max(history)
    pad = max((y_max - y_min) * 0.1, y_min * 0.001, 0.01)
    fig = go.Figure(go.Scatter(x=list(range(len(history))), y=history,
        mode="lines", line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=fill,
        hovertemplate="₹%{y:.2f}<extra></extra>"))
    fig.update_layout(title=f"{symbol} — LTP this session ({len(history)} ticks)",
                      height=170, margin=dict(l=0,r=0,t=30,b=0),
                      yaxis=dict(range=[y_min-pad, y_max+pad]),
                      xaxis=dict(showticklabels=False, title="← older     newer →"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── TFT Forecast chart ────────────────────────────────────────────────────────
def _build_forecast_chart(steps: list, ltp: float, symbol: str, price_history: list) -> go.Figure:
    """
    Forecast chart in ΔPrice space (₹ change from current LTP).
    Zero line = current price = "no change".
    Green bar extends UP from zero  → bullish (P50 > 0).
    Red   bar extends DOWN from zero → bearish (P50 < 0).
    Bar body = 0 → P50 (direction + magnitude).
    Whisker  = P10 → P90 (full uncertainty range).
    Opacity fades t+1 → t+5.
    """
    opacities = [1.0, 0.60, 0.35, 0.18, 0.07]
    ltp_ref   = ltp if ltp > 0 else (price_history[-1] if price_history else 1.0)

    # Convert log returns to ΔPrice (₹ change from current LTP)
    def to_delta(log_ret: float) -> float:
        return ltp_ref * (math.exp(log_ret) - 1)

    fig = go.Figure()

    # Zero reference line — "current price, no change"
    fig.add_hline(y=0, line_color="#555555", line_width=1.5)
    fig.add_annotation(
        x=0.01, y=0, xref="paper", yref="y",
        text=f" ₹{ltp_ref:.2f} (now)",
        showarrow=False, xanchor="left",
        font=dict(size=11, color="#777777"),
    )

    for i, step in enumerate(steps):
        d10 = to_delta(step.get("P10", 0))
        d50 = to_delta(step.get("P50", 0))
        d90 = to_delta(step.get("P90", 0))

        color = "#28A745" if step.get("P50", 0) >= 0 else "#DC3545"
        op    = opacities[i]
        x_pos = i + 1

        implied_px = ltp_ref + d50
        pct_move   = (math.exp(step.get("P50", 0)) - 1) * 100

        # Whisker: P10 to P90 uncertainty range
        fig.add_trace(go.Scatter(
            x=[x_pos, x_pos], y=[d10, d90],
            mode="lines", line=dict(color=color, width=1),
            opacity=op * 0.5, showlegend=False, hoverinfo="skip",
        ))

        # Body bar: 0 → P50  (always starts at the reference line)
        fig.add_trace(go.Bar(
            x=[x_pos], y=[d50], base=[0],
            marker_color=color, opacity=op, width=0.40,
            name=f"t+{i+1}", showlegend=False,
            hovertemplate=(
                f"<b>t+{i+1}</b><br>"
                f"Direction: {'↑' if d50 >= 0 else '↓'} {pct_move:+.3f}%<br>"
                f"<b>P50: ₹{implied_px:.2f}</b>  ({d50:+.2f})<br>"
                f"Range P10→P90: {d10:+.2f} → {d90:+.2f}<extra></extra>"
            ),
        ))

        # P50 endpoint marker
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[d50], mode="markers",
            marker=dict(size=8, symbol="diamond", color="white",
                        line=dict(color=color, width=2)),
            opacity=op, showlegend=False, hoverinfo="skip",
        ))

    step_labels = {i + 1: f"t+{i + 1}" for i in range(len(steps))}
    fig.update_layout(
        title=f"{symbol} — TFT Forecast  ·  ΔPrice from ₹{ltp_ref:.2f}  ·  fading opacity = lower confidence",
        height=400, barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=44, b=0),
        xaxis=dict(
            tickmode="array",
            tickvals=list(step_labels.keys()),
            ticktext=list(step_labels.values()),
            gridcolor="#1e1e1e", zeroline=False,
        ),
        yaxis=dict(
            title="ΔPrice from now (₹)",
            gridcolor="#1e1e1e", tickformat=".2f",
            zeroline=True, zerolinecolor="#555555", zerolinewidth=1.5,
        ),
    )
    return fig


# ── Table helpers ─────────────────────────────────────────────────────────────
def _sort_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Sector"] = out["Symbol"].map(lambda s: SECTOR_MAP.get(s, "Other"))
    out["Dir"]    = out["Chg%"].apply(lambda v: "↑" if v > 0 else ("↓" if v < 0 else "—"))
    cols = ["Dir", "Symbol", "Sector", "LTP", "Chg%", "Volume", "High", "Low", "Updated"]
    return out[cols].sort_values(["Sector", "Symbol"]).reset_index(drop=True)

def _render_table_view(df: pd.DataFrame, filter_fn=None, key: str = "tbl"):
    frame = filter_fn(df) if filter_fn else df
    if frame.empty:
        st.caption("No stocks match this filter.")
        return
    display = _sort_by_sector(frame)
    def _chg_color(val: float) -> str:
        if val > 0: return "color: #28A745; font-weight: bold"
        if val < 0: return "color: #DC3545; font-weight: bold"
        return "color: #6c757d"
    def _row_color(row):
        if row["Chg%"] > 0:   c = "#28A745"
        elif row["Chg%"] < 0: c = "#DC3545"
        else:                  c = "#6c757d"
        return [f"color:{c};font-weight:bold" if col in ("Symbol", "LTP", "Dir") else "" for col in row.index]
    fmt = {"LTP":"{:.2f}","High":"{:.2f}","Low":"{:.2f}",
           "Volume":"{:,.0f}","Chg%":"{:+.2f}%"}
    try:
        styled = display.style.format(fmt).apply(_row_color, axis=1).map(_chg_color, subset=["Chg%"])
    except Exception:
        styled = display
    event = st.dataframe(styled, use_container_width=True,
                         height=min(620, 42 + len(display)*35),
                         on_select="rerun", selection_mode="single-row", key=key)
    try:
        rows = event.selection["rows"]
        if rows:
            clicked = display.iloc[rows[0]]["Symbol"]
            if clicked != st.session_state.get("selected_stock"):
                st.session_state["selected_stock"] = clicked
                st.rerun(scope="app")   # full rerun so sidebar selectbox syncs
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  STATIC SHELL  (renders once — sidebar + header never disturbed by refresh)
# ═════════════════════════════════════════════════════════════════════════════

r = get_redis_client()
connected = redis_ok(r)
_, mkt_label = market_status()

# Lightweight symbol list for sidebar selectbox (just key names, no hgetall)
try:
    _sym_keys = sorted(r.keys("STOCK:*:current_bar")) if connected else []
    _all_syms = [k[6:-12] for k in _sym_keys if k[6:-12] and k[6:-12] != "NIFTY50"]
except Exception:
    _all_syms = []

if _all_syms and st.session_state["selected_stock"] not in _all_syms:
    st.session_state["selected_stock"] = _all_syms[0]

# ── Sidebar (STATIC — only reruns on explicit user interaction) ───────────────
with st.sidebar:
    st.title("⚡ AlgoTrading")
    st.caption("Real-time Nifty 50 predictor")
    st.divider()

    st.subheader("Controls")
    refresh_s    = st.slider("Refresh interval (s)", 1, 30, 2)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    st.divider()

    st.subheader("Stock Detail")
    if _all_syms:
        _curr = st.session_state.get("selected_stock")
        _idx  = _all_syms.index(_curr) if _curr in _all_syms else 0
        _new  = st.selectbox("Select stock", options=_all_syms, index=_idx,
                             key="sidebar_sel")
        if _new != st.session_state["selected_stock"]:
            st.session_state["selected_stock"] = _new
            st.rerun()
    else:
        st.caption("No stocks yet — start the viz consumer.")

    _sidebar_detail = st.empty()   # ← filled by the fragment below
    st.divider()
    st.caption(f"Redis  `{REDIS_HOST}:{REDIS_PORT}`")
    st.caption(f"v0.2 · {datetime.now().strftime('%Y-%m-%d')}")

# ── Header (STATIC) ───────────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
with h1: st.title("AlgoTrading Dashboard")
with h2:
    if connected:
        st.success("Redis ✓")
    else:
        st.error("Redis ✗")
with h3: st.caption(mkt_label)
with h4: st.caption(f"⟳ {datetime.now().strftime('%H:%M:%S')}")
st.divider()

# ── View selector (STATIC — tab switching never triggers a fragment rerun) ──
st.segmented_control(
    "View",
    ["Market Overview", "Stock Detail", "TFT Predictions"],
    key="main_view",
    label_visibility="collapsed",
)

if not connected:
    st.warning("**Cannot reach Redis.**\n```\ndocker-compose up -d redis\n```")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  TFT PREDICTIONS FRAGMENT  (60-second refresh — stable for chart interaction)
# ═════════════════════════════════════════════════════════════════════════════

@st.fragment(run_every=60)
def tft_panel():
    r_tft   = get_redis_client()
    df_tft  = fetch_stocks(r_tft)
    syms    = df_tft["Symbol"].tolist() if not df_tft.empty else []
    pred_df = fetch_predictions(r_tft, syms)
    selected = st.session_state.get("selected_stock")

    c_pick, c_ltp, c_dir, c_band = st.columns([3, 2, 2, 2])
    with c_pick:
        tft_idx = syms.index(selected) if selected in syms else 0
        tft_sel = st.selectbox(
            "Stock", options=syms, index=tft_idx,
            key="tft_stock_sel", label_visibility="collapsed",
        )
        st.session_state["selected_stock"] = tft_sel

    tft_sym       = tft_sel or selected
    forecast      = fetch_forecast_steps(r_tft, tft_sym)
    row_data      = df_tft[df_tft["Symbol"] == tft_sym].iloc[0] if (not df_tft.empty and tft_sym in syms) else None
    ltp_now       = float(row_data["LTP"]) if row_data is not None else 0.0

    with c_ltp:
        if row_data is not None:
            st.metric("LTP", f"₹{ltp_now:.2f}", f"{row_data['Chg%']:+.2f}%")

    steps         = (forecast or {}).get("steps", [])
    ltp_for_chart = (forecast or {}).get("ltp") or ltp_now

    if steps:
        p50_t1      = steps[0]["P50"]
        p50_pct     = (math.exp(p50_t1) - 1) * 100
        uncertainty = steps[0]["P90"] - steps[0]["P10"]
        with c_dir:
            arrow = "↑" if p50_t1 >= 0 else "↓"
            st.metric(f"{arrow} t+1 P50", f"{p50_pct:+.3f}%",
                      delta_color="normal" if p50_t1 >= 0 else "inverse")
        with c_band:
            st.metric("t+1 P10–P90 band", f"{uncertainty:.5f}")

        hist_for_chart = st.session_state["price_history"].get(tft_sym, [])
        st.plotly_chart(
            _build_forecast_chart(steps, ltp_for_chart, tft_sym, hist_for_chart),
            use_container_width=True,
        )

        st.caption("Per-step quantile log returns  ·  Implied price compounds P50 from current LTP")
        cumulative_log = 0.0
        tbl_rows = []
        for i, s in enumerate(steps):
            cumulative_log += s["P50"]
            tbl_rows.append({
                "Step":          f"t+{i+1}",
                "P10":           s["P10"],
                "P30":           s["P30"],
                "P50":           s["P50"],
                "P70":           s["P70"],
                "P90":           s["P90"],
                "Implied Price": ltp_for_chart * math.exp(cumulative_log) if ltp_for_chart else 0,
            })
        steps_df = pd.DataFrame(tbl_rows).set_index("Step")
        def _qc(val):
            if not isinstance(val, float): return ""
            return "color:#28A745" if val > 0 else ("color:#DC3545" if val < 0 else "")
        try:
            steps_styled = steps_df.style.format(
                {c: "{:+.5f}" for c in ["P10","P30","P50","P70","P90"]}
            ).format({"Implied Price": "₹{:.2f}"}).map(_qc, subset=["P10","P30","P50","P70","P90"])
        except Exception:
            steps_styled = steps_df
        st.dataframe(steps_styled, use_container_width=True, height=215)

    else:
        st.info(
            "No multi-step forecast yet.  \n"
            "Start the inference consumer with a trained model checkpoint.  \n"
            "Single-step fallback (t+1 only):"
        )
        if not pred_df.empty and tft_sym in pred_df["Symbol"].values:
            pr     = pred_df[pred_df["Symbol"] == tft_sym].iloc[0]
            cols_q = ["P10","P30","P50","P70","P90"]
            df_fb  = pd.DataFrame([pr[cols_q].to_dict()])
            try:
                st.dataframe(df_fb.style.format({c: "{:+.5f}" for c in cols_q}),
                             use_container_width=True)
            except Exception:
                st.dataframe(df_fb, use_container_width=True)

    if not pred_df.empty:
        st.divider()
        st.caption("All stocks — t+1 P50 direction")
        wide = pred_df[["Symbol","P50"]].copy()
        wide["Dir"] = wide["P50"].apply(lambda v: "↑" if v > 0 else "↓")
        wide = wide.sort_values("P50", ascending=False).set_index("Symbol")
        def _dc(val):
            if not isinstance(val, str): return ""
            return "color:#28A745;font-weight:bold" if val == "↑" else "color:#DC3545;font-weight:bold"
        try:
            wide_styled = wide.style.format({"P50": "{:+.5f}"}).map(_dc, subset=["Dir"])
        except Exception:
            wide_styled = wide
        st.dataframe(wide_styled, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE DATA FRAGMENT  (auto-refreshes every N seconds, sidebar stays still)
# ═════════════════════════════════════════════════════════════════════════════

@st.fragment(run_every=refresh_s if auto_refresh else None)
def live_panel():
    df      = fetch_stocks(r)
    symbols = df["Symbol"].tolist() if not df.empty else []
    pred_df = fetch_predictions(r, symbols)

    # Accumulate LTP history
    if not df.empty:
        for _, row in df.iterrows():
            hist = st.session_state["price_history"].setdefault(row["Symbol"], [])
            hist.append(row["LTP"])
            if len(hist) > MAX_HISTORY:
                st.session_state["price_history"][row["Symbol"]] = hist[-MAX_HISTORY:]
        st.session_state["refresh_count"] += 1
        if st.session_state["selected_stock"] not in symbols:
            st.session_state["selected_stock"] = sorted(symbols)[0]

    if df.empty:
        st.info("**No live stock data yet.**\n"
                "```\npy -m src.consumers.viz_consumer\n```")
        return

    selected = st.session_state.get("selected_stock")

    # ── Summary metrics ────────────────────────────────────────────────────
    gainers  = int((df["Chg%"] > 0).sum())
    losers   = int((df["Chg%"] < 0).sum())
    flat     = len(df) - gainers - losers
    avg_chg  = df["Chg%"].mean()
    top_vol  = df.loc[df["Volume"].idxmax(), "Symbol"]
    best_row = df.loc[df["Chg%"].idxmax()]
    best_str = f"{best_row['Symbol']} {best_row['Chg%']:+.2f}%"

    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric("Stocks Live", len(df))
    m2.metric("Gainers 📈",  gainers)
    m3.metric("Losers 📉",   losers)
    m4.metric("Flat",        flat)
    m5.metric("Avg Chg%",   f"{avg_chg:+.2f}%")
    m6.metric("Most Active", top_vol)
    m7.metric("Best Gainer", best_str)
    st.divider()

    # ── View content (driven by segmented_control in static shell) ────────
    active = st.session_state.get("main_view", "Market Overview")

    # ── Market Overview ────────────────────────────────────────────────────
    if active == "Market Overview":
        subtab_tbl, subtab_heat = st.tabs(["Table View", "Heatmap View"])

        with subtab_tbl:
            cl, cr = st.columns([3, 1])
            with cl:
                view_mode = st.radio("Filter", ["All","Gainers","Losers"],
                                     horizontal=True, label_visibility="collapsed")
            with cr:
                st.caption("Sorted: sector → symbol")
            filters = {
                "All":     None,
                "Gainers": lambda d: d[d["Chg%"] > 0].copy(),
                "Losers":  lambda d: d[d["Chg%"] < 0].copy(),
            }
            _render_table_view(df, filter_fn=filters[view_mode],
                               key=f"tbl_{view_mode}")

        with subtab_heat:
            st.markdown(
                "<div style='font-size:0.8rem;color:#888;margin-bottom:6px'>"
                "📦 <b>Size</b> = Volume &nbsp;·&nbsp; "
                "<span style='color:#28A745;font-size:1rem'>■</span> <b>Green</b> = Gain &nbsp;·&nbsp; "
                "<span style='color:#DC3545;font-size:1rem'>■</span> <b>Red</b> = Loss &nbsp;·&nbsp; "
                "<span style='color:#444;font-size:1rem'>■</span> <b>Dark</b> = Flat &nbsp;·&nbsp; "
                "👆 Click any stock cell to select it"
                "</div>",
                unsafe_allow_html=True,
            )
            fig_heat = _build_heatmap(df)
            heat_evt = st.plotly_chart(fig_heat, on_select="rerun",
                                       key="heatmap_treemap", use_container_width=True)
            try:
                pts = heat_evt.selection["points"]
                if pts:
                    lbl = pts[0].get("label", "")
                    if lbl in symbols and lbl != selected:
                        st.session_state["selected_stock"] = lbl
                        st.rerun(scope="app")
            except Exception:
                pass

    # ── Stock Detail ───────────────────────────────────────────────────────
    elif active == "Stock Detail":
        if not selected or selected not in symbols:
            st.info("Select a stock from the Table View or Heatmap.")
        else:
            row          = df[df["Symbol"] == selected].iloc[0]
            history      = st.session_state["price_history"].get(selected, [])
            bars         = fetch_bars(r, selected, n=30)
            cur_bar_data = fetch_current_bar(r, selected)
            st.plotly_chart(_build_multi_bar_chart(bars, cur_bar_data, selected), use_container_width=True)
            fig_sp = _build_sparkline(history, selected)
            if fig_sp:
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                needed = 5 - len(history)
                st.caption(f"Building sparkline — {needed} more "
                           f"refresh{'es' if needed!=1 else ''} needed.")
            if not pred_df.empty and selected in pred_df["Symbol"].values:
                pred_row = pred_df[pred_df["Symbol"] == selected].iloc[0]
                qs = [pred_row[q] for q in ("P10","P30","P50","P70","P90")]
                lc = "#28A745" if qs[2] >= 0 else "#DC3545"
                fc = "rgba(40,167,69,0.12)" if qs[2] >= 0 else "rgba(220,53,69,0.12)"
                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(
                    x=["P10","P30","P50","P70","P90"], y=qs,
                    mode="lines+markers", fill="tozeroy", fillcolor=fc,
                    line=dict(color=lc, width=2), marker=dict(size=8)))
                fig_q.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
                fig_q.update_layout(
                    title=f"{selected} — t+1 Quantile Forecast (log returns)",
                    yaxis_title="Log return", height=230,
                    margin=dict(l=0,r=0,t=36,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_q, use_container_width=True)
            else:
                st.info("No TFT forecast for this stock yet.")

    # ── TFT Predictions ────────────────────────────────────────────────────
    else:
        tft_panel()   # own 60-second fragment — chart stays stable for interaction

    # ── Sidebar detail slot (written from inside fragment via closure) ──────
    if selected and selected in symbols:
        row    = df[df["Symbol"] == selected].iloc[0]
        sector = SECTOR_MAP.get(selected, "Other")
        chg    = row["Chg%"]
        with _sidebar_detail.container():
            st.markdown(f"**{selected}** · *{sector}*")
            st.metric("LTP", f"₹{row['LTP']:.2f}", delta=f"{chg:+.2f}%",
                      delta_color="normal" if chg >= 0 else "inverse")
            c1, c2 = st.columns(2)
            c1.metric("Open",  f"₹{row['Open']:.2f}")
            c2.metric("Close", f"₹{row['Close']:.2f}")
            c1.metric("High",  f"₹{row['High']:.2f}")
            c2.metric("Low",   f"₹{row['Low']:.2f}")
            st.metric("Volume", f"{int(row['Volume']):,.0f}")
            st.caption(f"Last tick: {row['Updated']}")
            fc_data = fetch_forecast_steps(r, selected)
            fc_steps = (fc_data or {}).get("steps", [])
            if fc_steps:
                p50    = fc_steps[0]["P50"]
                p50pct = (math.exp(p50) - 1) * 100
                color  = "#28A745" if p50 >= 0 else "#DC3545"
                label  = "↑ Bullish" if p50 >= 0 else "↓ Bearish"
                st.markdown(
                    f"<div style='margin-top:6px;padding:8px 10px;border-radius:6px;"
                    f"background:rgba(0,0,0,0.25);border-left:3px solid {color}'>"
                    f"<div style='font-size:0.7rem;color:#888;margin-bottom:2px'>TFT t+1 Forecast</div>"
                    f"<div style='font-size:1rem;color:{color};font-weight:bold'>{label}</div>"
                    f"<div style='font-size:0.85rem;color:{color}'>{p50pct:+.3f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif not pred_df.empty and selected in pred_df["Symbol"].values:
                pred_row = pred_df[pred_df["Symbol"] == selected].iloc[0]
                p50    = pred_row["P50"]
                p50pct = (math.exp(p50) - 1) * 100
                color  = "#28A745" if p50 >= 0 else "#DC3545"
                label  = "↑ Bullish" if p50 >= 0 else "↓ Bearish"
                st.markdown(
                    f"<div style='margin-top:6px;padding:8px 10px;border-radius:6px;"
                    f"background:rgba(0,0,0,0.25);border-left:3px solid {color}'>"
                    f"<div style='font-size:0.7rem;color:#888;margin-bottom:2px'>TFT t+1 Forecast</div>"
                    f"<div style='font-size:1rem;color:{color};font-weight:bold'>{label}</div>"
                    f"<div style='font-size:0.85rem;color:{color}'>{p50pct:+.3f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


live_panel()
