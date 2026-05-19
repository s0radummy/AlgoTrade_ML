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
    - Kafka   : py scripts/kite_to_kafka.py  (or kafka_live_view.py for mock)
"""

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
        keys = sorted(r.keys("STOCK:*"))
    except Exception:
        return pd.DataFrame()
    if not keys:
        return pd.DataFrame()
    pipe = r.pipeline()
    for k in keys: pipe.hgetall(k)
    rows = []
    for key, data in zip(keys, pipe.execute()):
        if not data: continue
        sym   = key[6:]
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
    values   = vols + [0] * n
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

def _build_candlestick(row: pd.Series, symbol: str) -> go.Figure:
    fig = go.Figure(go.Candlestick(
        x=[symbol], open=[row["Open"]], high=[row["High"]],
        low=[row["Low"]], close=[row["LTP"]],
        increasing_line_color="#28A745", decreasing_line_color="#DC3545",
    ))
    for label, val, color, side in [
        ("Open", row["Open"], "#aaa",     "bottom left"),
        ("High", row["High"], "#28A745",  "top right"),
        ("Low",  row["Low"],  "#DC3545",  "bottom right"),
    ]:
        fig.add_hline(y=val, line_dash="dot", line_color=color,
                      annotation_text=f"{label}: ₹{val:.2f}",
                      annotation_position=side)
    fig.update_layout(title=f"{symbol} — Today's Bar  (close = live LTP)",
                      height=320, xaxis_rangeslider_visible=False,
                      xaxis=dict(type="category"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0,r=0,t=42,b=0))
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
    fmt = {"LTP":"{:.2f}","High":"{:.2f}","Low":"{:.2f}",
           "Volume":"{:,.0f}","Chg%":"{:+.2f}%"}
    try:
        styled = display.style.format(fmt).map(_chg_color, subset=["Chg%"])
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
    _sym_keys  = sorted(r.keys("STOCK:*")) if connected else []
    _all_syms  = [k[6:] for k in _sym_keys]
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
            row     = df[df["Symbol"] == selected].iloc[0]
            history = st.session_state["price_history"].get(selected, [])
            st.plotly_chart(_build_candlestick(row, selected), use_container_width=True)
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
        st.subheader("TFT Predictions — All Stocks")
        if pred_df.empty:
            st.info("No predictions yet. Inference consumer needs `models/tft_v1.pth`.")
        else:
            st.caption("5-quantile t+1 forecast. Values are log returns — "
                       "positive = expected gain, negative = expected loss.")
            pred_disp = pred_df.copy()
            pred_disp["Dir"] = pred_disp["P50"].apply(lambda v: "↑" if v > 0 else "↓")
            pred_disp = pred_disp[["Symbol","Dir","P10","P30","P50","P70","P90"]].set_index("Symbol")
            def _pc(val):
                if not isinstance(val, float): return ""
                return "color:#28A745" if val>0 else ("color:#DC3545" if val<0 else "")
            try:
                pred_styled = pred_disp.style.format(
                    {c:"{:+.5f}" for c in ["P10","P30","P50","P70","P90"]}
                ).map(_pc, subset=["P10","P30","P50","P70","P90"])
            except Exception:
                pred_styled = pred_disp
            st.dataframe(pred_styled, use_container_width=True)

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
            if not pred_df.empty and selected in pred_df["Symbol"].values:
                pred_row = pred_df[pred_df["Symbol"] == selected].iloc[0]
                qs_vals  = [pred_row[q] for q in ("P10","P30","P50","P70","P90")]
                bar_clrs = ["#28A745" if v >= 0 else "#DC3545" for v in qs_vals]
                fig_mini = go.Figure(go.Bar(
                    x=qs_vals, y=["P10","P30","P50","P70","P90"], orientation="h",
                    marker_color=bar_clrs,
                    hovertemplate="%{y}: %{x:+.5f}<extra></extra>"))
                fig_mini.add_vline(x=0, line_color="#666", line_width=1)
                fig_mini.update_layout(
                    title="t+1 Forecast", height=170, margin=dict(l=0,r=0,t=30,b=0),
                    xaxis_title="log return",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_mini, use_container_width=True,
                                key="sidebar_pred_bar")


live_panel()
