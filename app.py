import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Asset Management & Quantitative Risk Engine", 
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.5rem; font-weight: 600; color: #374151; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; }
    .metric-card { background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 1.2rem; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton button { background: #1E3A8A; color: white; border-radius: 5px; width: 100%; }
    .info-box { background-color: #F3F4F6; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Asset Management & Quantitative Risk Engine</div>', unsafe_allow_html=True)
st.caption("Advanced Portfolio Optimization with Geopolitical Risk Integration")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Strategic Parameters")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Tickers (comma-separated)", default_tickers)
    ticker_list = [t.strip().upper() for t in assets.split(",") if t.strip()]

    default_start = datetime(2020, 1, 2)
    start_date = st.date_input("Analysis Start", value=default_start)
    default_end = datetime.now()
    end_date = st.date_input("Analysis End", value=default_end, max_value=default_end)

    st.divider()
    st.markdown("### Geopolitical Risk Overlay")
    geo_events = st.multiselect(
        "Active Events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability",
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes"],
        default=["US-China Tech Tensions"]
    )
    geo_intensity = st.slider("Risk Intensity", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.markdown("### Black-Litterman View")
    view_ticker = st.selectbox("Asset for View", ticker_list if ticker_list else ["AAPL"])
    view_return = st.slider("Expected Return (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.markdown("### Compliance & Risk")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification (L2) Penalty", 0.0, 2.0, 0.5)

    st.divider()
    st.markdown("### Developer Settings")
    debug_mode = st.checkbox("Show Debug Information", value=False)


# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start, end, debug=False):
    today_str = datetime.now().strftime('%Y-%m-%d')

    start_str = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else str(start)
    end_str   = end.strftime('%Y-%m-%d')   if hasattr(end,   'strftime') else str(end)

    if start_str > today_str: start_str = '2020-01-06'
    if end_str   > today_str: end_str   = today_str
    if start_str >= end_str:
        start_str = '2020-01-06'
        end_str   = today_str

    all_tickers = list(dict.fromkeys(tickers + ["^GSPC"]))   # preserve order, no dupes

    # ── individual downloads (most reliable) ──────────────────────────────
    close_prices = {}
    for t in all_tickers:
        for attempt in range(3):
            try:
                raw = yf.download(t, start=start_str, end=end_str,
                                  progress=False, auto_adjust=True)
                time.sleep(0.3)
                if raw.empty:
                    continue
                col = 'Close' if 'Close' in raw.columns else (
                      'Adj Close' if 'Adj Close' in raw.columns else None)
                if col:
                    series = pd.to_numeric(raw[col].squeeze(), errors='coerce').dropna()
                    if len(series) > 5:
                        close_prices[t] = series
                        break
            except Exception as e:
                if debug:
                    st.warning(f"Attempt {attempt+1} failed for {t}: {e}")
                time.sleep(0.5)

    if not close_prices:
        return pd.DataFrame(), pd.Series(), {}

    df = pd.DataFrame(close_prices)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.ffill().bfill().dropna(how='all')

    benchmark = df.pop("^GSPC") if "^GSPC" in df.columns else pd.Series()
    assets_df = df.dropna(axis=1, how='all')

    if assets_df.empty or len(assets_df) < 10:
        return pd.DataFrame(), pd.Series(), {}

    fixed_caps = {
        'AAPL': 3000, 'MSFT': 2800, 'JPM': 500, 'MC.PA': 400,
        'ASML': 350,  'NESN.SW': 300, 'GOOGL': 1800, 'AMZN': 1600,
        'TSLA': 600,  'NVDA': 2200,  'V': 500,  'JNJ': 380,
        'XOM': 400,   'WMT': 450,    'PG': 350,  'MA': 400
    }
    mcaps = {t: fixed_caps.get(t, 100) * 1e9 for t in assets_df.columns}

    return assets_df, benchmark, mcaps


# --- GEOPOLITICAL OVERLAY ---
def apply_geopolitical_overlay(weights, events, intensity):
    if not events or intensity <= 0.5:
        return weights
    sector_risk = {
        'Technology':    {'US-China Tech Tensions': 0.8, 'Supply Chain Disruption': 0.7, 'Trade Policy Changes': 0.6},
        'Financials':    {'Currency Volatility': 0.6, 'Middle East Instability': 0.3, 'Trade Policy Changes': 0.4},
        'Semiconductors':{'US-China Tech Tensions': 0.9, 'Supply Chain Disruption': 0.8, 'Trade Policy Changes': 0.7},
        'Healthcare':    {'EU Regulation Shift': 0.5, 'Trade Policy Changes': 0.3},
        'Automotive':    {'Supply Chain Disruption': 0.9, 'Trade Policy Changes': 0.7},
        'Consumer':      {'Supply Chain Disruption': 0.5, 'Currency Volatility': 0.3},
        'Energy':        {'Middle East Instability': 0.8, 'Trade Policy Changes': 0.6},
    }
    ticker_sectors = {
        'AAPL':'Technology','MSFT':'Technology','JPM':'Financials',
        'MC.PA':'Consumer','ASML':'Semiconductors','NESN.SW':'Healthcare',
        'GOOGL':'Technology','AMZN':'Technology','TSLA':'Automotive',
        'NVDA':'Semiconductors','V':'Financials','JNJ':'Healthcare',
        'XOM':'Energy','WMT':'Consumer','PG':'Consumer','MA':'Financials',
    }
    adj = {}
    for ticker, w in weights.items():
        if w == 0:
            adj[ticker] = 0; continue
        sector = ticker_sectors.get(ticker, 'Technology')
        risk_score = sum(sector_risk.get(sector, {}).get(e, 0.1) for e in events)
        adj[ticker] = max(0.01, w * (1 - risk_score * intensity * 0.15))
    total = sum(adj.values())
    return {k: v / total for k, v in adj.items()} if total > 0 else weights


# --- EFFICIENT FRONTIER ---
def plot_efficient_frontier(mu, S, rf=0.02):
    try:
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        min_vol = ef.portfolio_performance(risk_free_rate=rf)

        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()
        max_sharpe = ef.portfolio_performance(risk_free_rate=rf)

        target_returns = np.linspace(min_vol[0], float(mu.max()), 20)
        vols = []
        for tr in target_returns:
            try:
                ef2 = EfficientFrontier(mu, S)
                ef2.efficient_return(target_return=tr)
                _, v, _ = ef2.portfolio_performance(risk_free_rate=rf)
                vols.append(v)
            except:
                vols.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols, y=target_returns, mode='lines',
                                  name='Efficient Frontier',
                                  line=dict(color='#3B82F6', width=3)))
        fig.add_trace(go.Scatter(x=np.sqrt(np.diag(S.values) * 252), y=mu,
                                  mode='markers+text', name='Assets',
                                  marker=dict(size=12, color='#EF4444'),
                                  text=list(S.columns), textposition="top center"))
        fig.add_trace(go.Scatter(x=[min_vol[1]], y=[min_vol[0]], mode='markers',
                                  name='Min Volatility',
                                  marker=dict(size=15, color='#10B981', symbol='diamond')))
        fig.add_trace(go.Scatter(x=[max_sharpe[1]], y=[max_sharpe[0]], mode='markers',
                                  name='Max Sharpe',
                                  marker=dict(size=15, color='#F59E0B', symbol='star')))
        fig.update_layout(title="Efficient Frontier", xaxis_title="Annual Volatility",
                          yaxis_title="Expected Annual Return", template="plotly_white",
                          height=500, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="right", x=1))
        return fig
    except Exception as e:
        if debug_mode:
            st.warning(f"Efficient frontier error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
try:
    if not ticker_list:
        st.info("👈 Please enter ticker symbols in the sidebar to begin analysis.")
        st.stop()

    with st.expander("📋 Current Parameters", expanded=False):
        st.write(f"**Tickers:** {', '.join(ticker_list)}")
        st.write(f"**Start Date:** {start_date}  |  **End Date:** {end_date}")
        st.write(f"**View:** {view_ticker} @ {view_return:.1%} (conf {view_conf:.0%})")
        st.write(f"**Geo events:** {', '.join(geo_events) or 'None'}  |  Intensity {geo_intensity:.1f}x")

    with st.spinner("📊 Fetching market data…"):
        prices, bench_prices, market_caps = get_clean_data(
            ticker_list, start_date, end_date, debug=debug_mode)

    if prices.empty:
        st.error("❌ No data available. Check ticker symbols and date range.")
        st.stop()

    # reconcile ticker_list to what actually loaded
    available = [t for t in ticker_list if t in prices.columns]
    missing   = set(ticker_list) - set(available)
    if missing:
        st.warning(f"⚠️ Tickers not found: {', '.join(sorted(missing))}")
    if not available:
        st.error("❌ None of the entered tickers returned valid data.")
        st.stop()

    ticker_list = available                      # authoritative list from here on
    prices      = prices[ticker_list]
    market_caps = {t: market_caps[t] for t in ticker_list if t in market_caps}

    st.success(f"✅ Data loaded: {len(ticker_list)} tickers, {len(prices)} trading days")
    st.info(f"📅 {prices.index[0].date()} → {prices.index[-1].date()}")

    # ── adjust view_ticker if it was dropped ──────────────────────────────
    if view_ticker not in ticker_list:
        view_ticker = ticker_list[0]
        st.info(f"View asset adjusted to: {view_ticker}")

    # ── covariance matrix ─────────────────────────────────────────────────
    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(prices)

    # S.columns is now the ground truth — keep everything aligned to it
    tickers_final = list(S.columns)

    # ── Black-Litterman ───────────────────────────────────────────────────
    try:
        # market caps aligned EXACTLY to S.columns order
        mcap_series = pd.Series(
            {t: market_caps.get(t, 1e11) for t in tickers_final},
            index=tickers_final
        )

        prior_rets = black_litterman.market_implied_prior_returns(
            mcap_series, 2.5, S)

        # sanity-check alignment
        assert list(prior_rets.index) == tickers_final, "prior_rets misaligned"

        bl = black_litterman.BlackLittermanModel(
            S,
            pi=prior_rets,
            absolute_views={view_ticker: view_return},
            omega="idzorek",
            view_confidences=[min(view_conf, 0.99)]
        )
        bl_mu = bl.bl_returns()
        # ensure bl_mu is also aligned
        bl_mu = bl_mu.reindex(tickers_final)

    except Exception as e:
        st.warning(f"⚠️ Black-Litterman failed ({e}). Falling back to historical returns.")
        returns_tmp = prices.pct_change().dropna()
        bl_mu = (returns_tmp.mean() * 252).reindex(tickers_final)
        if bl_mu.isna().any():
            bl_mu = bl_mu.fillna(0.10)

    # ── Mean-Variance optimisation ────────────────────────────────────────
    try:
        ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
        ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
        ef.max_sharpe()
        optimized_weights = ef.clean_weights()
        if abs(sum(optimized_weights.values()) - 1.0) > 0.01:
            total = sum(optimized_weights.values())
            optimized_weights = {k: v / total for k, v in optimized_weights.items()}
    except Exception as e:
        st.error(f"❌ Optimisation failed: {e}")
        optimized_weights = {t: 1 / len(tickers_final) for t in tickers_final}
        st.info("Using equal weights as fallback.")

    # ── Geopolitical overlay ──────────────────────────────────────────────
    final_weights = (apply_geopolitical_overlay(optimized_weights, geo_events, geo_intensity)
                     if geo_events and geo_intensity > 0.5 else optimized_weights)

    # ── Performance metrics ───────────────────────────────────────────────
    weights_arr = np.array([final_weights.get(t, 0) for t in tickers_final])
    returns     = prices.pct_change().dropna().astype(float)

    if len(returns) < 5:
        st.error("❌ Not enough return data to compute metrics.")
        st.stop()

    p_rets = (returns * weights_arr).sum(axis=1)
    p_cum  = (1 + p_rets).cumprod()

    ann_ret = p_rets.mean() * 252
    ann_vol = p_rets.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    rolling_max = p_cum.expanding().max()
    max_dd      = ((p_cum - rolling_max) / rolling_max).min()

    down_vol = p_rets[p_rets < 0].std() * np.sqrt(252)
    sortino  = ann_ret / down_vol if down_vol > 0 else 0

    # ── Display ───────────────────────────────────────────────────────────
    st.markdown('<div class="sub-header">Portfolio Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Sharpe Ratio", "Annual Return", "Annual Volatility", "Max Drawdown"],
        [f"{sharpe:.2f}", f"{ann_ret:.1%}", f"{ann_vol:.1%}", f"{max_dd:.1%}"]
    ):
        col.markdown(f'<div class="metric-card">{label}<br><h3>{val}</h3></div>',
                     unsafe_allow_html=True)

    st.markdown('<div class="sub-header">Efficient Frontier</div>', unsafe_allow_html=True)
    fig_ef = plot_efficient_frontier(bl_mu, S)
    if fig_ef:
        st.plotly_chart(fig_ef, use_container_width=True)

    st.markdown('<div class="sub-header">Portfolio Allocation</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        w_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        w_df = w_df[w_df['Weight'] > 0.001]
        if not w_df.empty:
            fig_pie = px.pie(names=w_df.index, values=w_df['Weight'], hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Prism,
                             title="Portfolio Weights")
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.markdown("**Detailed Weights**")
        wt = pd.DataFrame({'Ticker': list(final_weights.keys()),
                           'Weight': [f"{v:.2%}" for v in final_weights.values()]}
                          ).sort_values('Weight', ascending=False)
        st.dataframe(wt, use_container_width=True, hide_index=True)
        if geo_events and geo_intensity > 0.5:
            st.info(f"Geo overlay: {len(geo_events)} event(s) @ {geo_intensity:.1f}x")

    st.markdown('<div class="sub-header">Performance Comparison</div>', unsafe_allow_html=True)
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=p_cum.index, y=p_cum, name="Optimized Portfolio",
                                   line=dict(color='#1E3A8A', width=3),
                                   fill='tozeroy', fillcolor='rgba(30,58,138,0.1)'))
    if not bench_prices.empty:
        b_ret = bench_prices.pct_change().dropna()
        common = p_cum.index.intersection(b_ret.index)
        if len(common) > 0:
            b_cum = (1 + b_ret.loc[common]).cumprod()
            fig_perf.add_trace(go.Scatter(x=b_cum.index, y=b_cum, name="S&P 500",
                                           line=dict(color='#94A3B8', dash='dot', width=2)))
    fig_perf.update_layout(template="plotly_white", height=500,
                            xaxis_title="Date", yaxis_title="Cumulative Return",
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="right", x=1))
    st.plotly_chart(fig_perf, use_container_width=True)

    # ── Exports ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sub-header">Export Results</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        csv_w = pd.DataFrame.from_dict(final_weights, orient='index',
                                        columns=['Weight']).to_csv().encode()
        st.download_button("📥 Portfolio Weights", csv_w, "portfolio_weights.csv", "text/csv")
    with e2:
        csv_p = pd.DataFrame({'Date': p_cum.index, 'Return': p_rets.values,
                               'Cumulative': p_cum.values}).to_csv(index=False).encode()
        st.download_button("📥 Performance Data", csv_p, "portfolio_performance.csv", "text/csv")
    with e3:
        params_df = pd.DataFrame([{
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d'),
            'Tickers': ', '.join(tickers_final), 'View_Asset': view_ticker,
            'View_Return': f"{view_return:.2%}", 'View_Confidence': f"{view_conf:.0%}",
            'Sharpe': f"{sharpe:.2f}", 'Annual_Return': f"{ann_ret:.2%}",
            'Annual_Vol': f"{ann_vol:.2%}", 'Max_DD': f"{max_dd:.2%}",
        }])
        st.download_button("📥 Strategy Parameters", params_df.to_csv(index=False).encode(),
                           "strategy_parameters.csv", "text/csv")

    st.divider()
    with st.expander("⚠️ Risk Disclaimer"):
        st.markdown("""
        - Educational and research purposes only  
        - Past performance is not indicative of future results  
        - Consult a qualified financial adviser before making investment decisions
        """)

    if debug_mode:
        with st.expander("🔍 Debug Info"):
            st.write(f"tickers_final: {tickers_final}")
            st.write(f"S.columns: {list(S.columns)}")
            st.write(f"bl_mu.index: {list(bl_mu.index)}")
            st.write(f"prices shape: {prices.shape}")
            st.dataframe(prices.tail(3))

except Exception as e:
    st.error(f"🚨 Engine Error: {e}")
    with st.expander("🔧 Traceback", expanded=debug_mode):
        import traceback
        st.code(traceback.format_exc())
