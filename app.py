import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Asset Management & Quantitative Risk Engine", 
    layout="wide",
    page_icon="📊"
)

# Custom CSS for professional styling
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

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.markdown("### Strategic Parameters")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Tickers (comma-separated)", default_tickers)
    ticker_list = [t.strip().upper() for t in assets.split(",") if t.strip()]
    
    # FIXED: Use a safe default date that's definitely in the past
    default_start = datetime(2020, 1, 2)  # First trading day of 2020
    start_date = st.date_input("Analysis Start", value=default_start)
    
    # ADDED: End date control to prevent future date issues
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
    view_return = st.slider(f"Expected Return (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.markdown("### Compliance & Risk")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification (L2) Penalty", 0.0, 2.0, 0.5)

# --- RE-ENGINEERED DATA FETCHING WITH FIXED DATE HANDLING ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start, end):
    """Robust data fetching with fallback mechanisms and proper date handling"""
    if not tickers:
        return pd.DataFrame(), pd.Series(), {}
    
    # FIX: Validate and convert dates properly
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # Handle start date
    if hasattr(start, 'strftime'):
        start_str = start.strftime('%Y-%m-%d')
    elif isinstance(start, str):
        start_str = start
    else:
        start_str = '2020-01-02'
        st.warning(f"Date format issue. Using default start date: {start_str}")
    
    # Handle end date
    if hasattr(end, 'strftime'):
        end_str = end.strftime('%Y-%m-%d')
    elif isinstance(end, str):
        end_str = end
    else:
        end_str = today_str
    
    # Ensure we're not using future dates
    if start_str > today_str:
        start_str = '2020-01-02'
        st.warning(f"Start date is in the future. Using default: {start_str}")
    
    if end_str > today_str:
        end_str = today_str
        st.warning(f"End date is in the future. Using today: {end_str}")
    
    # Ensure start is before end
    if start_str >= end_str:
        start_str = '2020-01-02'
        end_str = today_str
        st.warning(f"Start date must be before end date. Using default range.")
    
    # Ensure unique tickers
    all_tickers = list(set(tickers + ["^GSPC"]))
    
    close_prices = pd.DataFrame()
    failed_tickers = []
    
    try:
        # Try bulk download first with proper date range
        raw_data = yf.download(
            all_tickers, 
            start=start_str, 
            end=end_str,
            progress=False, 
            group_by='ticker',
            auto_adjust=True
        )
        time.sleep(1)  # Rate limit protection
        
        # Extract close prices from the downloaded data
        for t in all_tickers:
            try:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    # MultiIndex case
                    if (t, 'Close') in raw_data.columns:
                        close_prices[t] = raw_data[(t, 'Close')]
                    elif (t, 'Adj Close') in raw_data.columns:
                        close_prices[t] = raw_data[(t, 'Adj Close')]
                else:
                    # Single ticker case
                    if 'Close' in raw_data.columns:
                        close_prices[t] = raw_data['Close']
                    elif 'Adj Close' in raw_data.columns:
                        close_prices[t] = raw_data['Adj Close']
            except Exception as e:
                failed_tickers.append((t, str(e)))
        
        # Fallback for failed tickers - download individually
        for t, error in failed_tickers:
            try:
                ticker_data = yf.download(
                    t, 
                    start=start_str, 
                    end=end_str,
                    progress=False,
                    auto_adjust=True
                )
                if not ticker_data.empty:
                    if 'Close' in ticker_data.columns:
                        close_prices[t] = ticker_data['Close']
                    elif 'Adj Close' in ticker_data.columns:
                        close_prices[t] = ticker_data['Adj Close']
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"Failed to download {t}: {str(e)}")
                continue
        
        if close_prices.empty:
            st.error("❌ No data could be fetched for any ticker. Please check internet connection and try again.")
            return pd.DataFrame(), pd.Series(), {}
        
        # Check if we have any data
        if close_prices.shape[0] == 0:
            st.error("❌ Data fetched but contains no rows. Try a different start date.")
            return pd.DataFrame(), pd.Series(), {}
            
        # Clean data - forward fill and drop any remaining NaNs
        clean_df = close_prices.ffill().bfill().dropna(how='all')
        
        # Check if cleaning removed all data
        if clean_df.empty or clean_df.shape[0] < 5:
            st.error(f"❌ Insufficient data after cleaning. Only {clean_df.shape[0]} rows available.")
            return pd.DataFrame(), pd.Series(), {}
            
        # Separate benchmark and assets
        benchmark = pd.Series()
        if "^GSPC" in clean_df.columns:
            benchmark = clean_df["^GSPC"]
            assets_data = clean_df.drop(columns=["^GSPC"])
        else:
            assets_data = clean_df
        
        # Static Market Cap Estimates (in USD billions)
        fixed_caps = {
            'AAPL': 3000, 'MSFT': 2800, 'JPM': 500, 'MC.PA': 400, 
            'ASML': 350, 'NESN.SW': 300, 'GOOGL': 1800, 'AMZN': 1600,
            'TSLA': 600, 'NVDA': 2200, 'V': 500, 'JNJ': 380,
            'XOM': 400, 'WMT': 450, 'PG': 350, 'MA': 400
        }
        
        mcaps = {}
        for t in tickers:
            if t in assets_data.columns:
                # Convert billions to actual value for Black-Litterman
                mcaps[t] = fixed_caps.get(t, 100) * 1e9
            else:
                mcaps[t] = 1e11  # Default 100B market cap
        
        # Debug info
        st.success(f"✅ Successfully fetched data for {len(assets_data.columns)} tickers with {len(assets_data)} days of data")
        st.info(f"📅 Date range: {assets_data.index[0].strftime('%Y-%m-%d')} to {assets_data.index[-1].strftime('%Y-%m-%d')}")
        
        return assets_data, benchmark, mcaps
        
    except Exception as e:
        st.error(f"❌ Data fetch error: {str(e)}")
        return pd.DataFrame(), pd.Series(), {}

# --- OPTIMIZATION LOGIC ---
def apply_geopolitical_overlay(weights, tickers, events, intensity):
    """Apply geopolitical risk adjustments to portfolio weights"""
    if not events or intensity <= 0.5:
        return weights
    
    # Sector risk exposure mapping
    sector_risk = {
        'Technology': {'US-China Tech Tensions': 0.8, 'Supply Chain Disruption': 0.7, 'Trade Policy Changes': 0.6},
        'Financials': {'Currency Volatility': 0.6, 'Middle East Instability': 0.3, 'Trade Policy Changes': 0.4},
        'Semiconductors': {'US-China Tech Tensions': 0.9, 'Supply Chain Disruption': 0.8, 'Trade Policy Changes': 0.7},
        'Healthcare': {'EU Regulation Shift': 0.5, 'Trade Policy Changes': 0.3},
        'Automotive': {'Supply Chain Disruption': 0.9, 'Trade Policy Changes': 0.7},
        'Consumer': {'Supply Chain Disruption': 0.5, 'Currency Volatility': 0.3},
        'Energy': {'Middle East Instability': 0.8, 'Trade Policy Changes': 0.6}
    }
    
    # Ticker to sector mapping
    ticker_sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financials',
        'MC.PA': 'Consumer', 'ASML': 'Semiconductors', 'NESN.SW': 'Healthcare',
        'GOOGL': 'Technology', 'AMZN': 'Technology', 'TSLA': 'Automotive',
        'NVDA': 'Semiconductors', 'V': 'Financials', 'JNJ': 'Healthcare',
        'XOM': 'Energy', 'WMT': 'Consumer', 'PG': 'Consumer', 'MA': 'Financials'
    }
    
    adjustments = {}
    for ticker, weight in weights.items():
        if weight == 0:
            adjustments[ticker] = 0
            continue
            
        sector = ticker_sectors.get(ticker, 'Technology')
        # Calculate total risk exposure for this ticker
        risk_score = 0
        for event in events:
            risk_score += sector_risk.get(sector, {}).get(event, 0.1)
        
        # Adjust weight based on risk (higher risk = lower weight)
        reduction_factor = 1 - (risk_score * intensity * 0.15)
        adjustments[ticker] = max(0.01, weight * reduction_factor)
    
    # Re-normalize to ensure weights sum to 1
    total = sum(adjustments.values())
    if total > 0:
        return {k: v/total for k, v in adjustments.items()}
    return weights

# --- EFFICIENT FRONTIER PLOT FUNCTION ---
def plot_efficient_frontier(mu, S, risk_free_rate=0.02):
    """Generate efficient frontier plot"""
    try:
        # Generate portfolio statistics
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        min_vol = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()
        max_sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        # Generate frontier points
        target_returns = np.linspace(min_vol[0], max(mu), 20)
        volatilities = []
        for target in target_returns:
            ef = EfficientFrontier(mu, S)
            try:
                ef.efficient_return(target_return=target)
                _, vol, _ = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                volatilities.append(vol)
            except:
                volatilities.append(np.nan)
        
        # Create plot
        fig = go.Figure()
        
        # Efficient frontier line
        fig.add_trace(go.Scatter(
            x=volatilities, y=target_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#3B82F6', width=3)
        ))
        
        # Individual assets
        fig.add_trace(go.Scatter(
            x=np.sqrt(np.diag(S * 252)), y=mu,
            mode='markers+text',
            name='Assets',
            marker=dict(size=12, color='#EF4444'),
            text=list(S.columns),
            textposition="top center"
        ))
        
        # Min volatility portfolio
        fig.add_trace(go.Scatter(
            x=[min_vol[1]], y=[min_vol[0]],
            mode='markers',
            name='Min Volatility',
            marker=dict(size=15, color='#10B981', symbol='diamond')
        ))
        
        # Max Sharpe portfolio
        fig.add_trace(go.Scatter(
            x=[max_sharpe[1]], y=[max_sharpe[0]],
            mode='markers',
            name='Max Sharpe',
            marker=dict(size=15, color='#F59E0B', symbol='star')
        ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Annual Volatility",
            yaxis_title="Expected Annual Return",
            template="plotly_white",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate efficient frontier: {str(e)}")
        return None

# --- MAIN EXECUTION ---
try:
    if not ticker_list:
        st.info("👈 Please enter ticker symbols in the sidebar to begin analysis.")
        st.stop()
    
    # Display current parameters
    with st.expander("📋 Current Parameters", expanded=False):
        st.write(f"**Tickers:** {', '.join(ticker_list)}")
        st.write(f"**Start Date:** {start_date}")
        st.write(f"**End Date:** {end_date}")
        st.write(f"**View Asset:** {view_ticker} with {view_return:.1%} expected return")
        st.write(f"**View Confidence:** {view_conf:.0%}")
        st.write(f"**Geopolitical Events:** {', '.join(geo_events) if geo_events else 'None'}")
        st.write(f"**Risk Intensity:** {geo_intensity:.1f}x")
    
    with st.spinner("📊 Fetching market data and optimizing portfolio..."):
        prices, bench_prices, market_caps = get_clean_data(ticker_list, start_date, end_date)
    
    if prices.empty:
        st.error("""
        ❌ **No data available for the selected tickers.** 
        
        **Possible reasons:**
        1. **Incorrect ticker format** - Make sure international stocks have exchange codes (e.g., `MC.PA` for LVMH Paris)
        2. **Start date is too recent** - Try an earlier date like 2020-01-01
        3. **End date is in the future** - The system now auto-fixes this
        4. **Yahoo Finance rate limits** - Wait 60 seconds and refresh
        5. **Network issues** - Check your internet connection
        
        **Current default tickers should work:** AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW
        """)
        st.stop()
    
    # Ensure we only work with tickers that have data
    available_tickers = [t for t in ticker_list if t in prices.columns]
    if not available_tickers:
        st.error("❌ None of the entered tickers returned valid data.")
        st.stop()
    
    if len(available_tickers) != len(ticker_list):
        missing = set(ticker_list) - set(available_tickers)
        st.warning(f"⚠️ Some tickers were not found: {', '.join(missing)}")
        st.info(f"✅ Proceeding with available tickers: {', '.join(available_tickers)}")
    
    ticker_list = available_tickers
    
    # --- OPTIMIZATION PROCESS ---
    
    # 1. Calculate covariance matrix with shrinkage
    try:
        S = risk_models.CovarianceShrinkage(prices[ticker_list]).ledoit_wolf()
    except Exception as e:
        st.warning(f"Using sample covariance matrix: {str(e)}")
        S = risk_models.sample_cov(prices[ticker_list])
    
    delta = 2.5  # Risk aversion coefficient
    
    # 2. Black-Litterman Optimization
    try:
        # Market-implied prior returns
        prior_rets = black_litterman.market_implied_prior_returns(market_caps, delta, S)
        
        # Ensure view ticker is available
        if view_ticker not in ticker_list:
            view_ticker = ticker_list[0]
            st.info(f"View ticker adjusted to: {view_ticker}")
        
        # Create Black-Litterman model
        bl = black_litterman.BlackLittermanModel(
            S, 
            pi=prior_rets, 
            absolute_views={view_ticker: view_return},
            omega="idzorek",
            view_confidences=[min(view_conf, 0.99)]
        )
        bl_mu = bl.bl_returns()
        
    except Exception as e:
        st.warning(f"⚠️ Black-Litterman optimization failed. Using historical returns: {str(e)}")
        # Fallback to historical mean returns
        historical_returns = prices[ticker_list].pct_change().mean() * 252
        bl_mu = historical_returns
    
    # 3. Mean-Variance Optimization
    try:
        ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
        ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
        raw_weights = ef.max_sharpe()
        optimized_weights = ef.clean_weights()
        
        # Check if optimization produced valid weights
        total_weight = sum(optimized_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_weight:.2%}, normalizing to 100%")
            optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
            
    except Exception as e:
        st.error(f"❌ Portfolio optimization failed: {str(e)}")
        # Equal weight fallback
        optimized_weights = {t: 1/len(ticker_list) for t in ticker_list}
        st.info("Using equal weights as fallback")
    
    # 4. Apply Geopolitical Overlay
    if geo_events and geo_intensity > 0.5:
        final_weights = apply_geopolitical_overlay(optimized_weights, ticker_list, geo_events, geo_intensity)
    else:
        final_weights = optimized_weights
    
    # 5. Calculate Portfolio Performance - FIXED WITH PROPER VALIDATION
    weights_arr = np.array([final_weights.get(t, 0) for t in ticker_list])
    
    # FIX: Check if we have enough price data
    if len(prices) < 2:
        st.error(f"❌ Insufficient data: only {len(prices)} day(s) of data. Need at least 2 days to calculate returns.")
        st.info("Try an earlier start date or later end date.")
        st.stop()
    
    # Calculate returns
    returns = prices[ticker_list].pct_change().dropna()
    
    # FIX: Check if we have any returns data
    if returns.empty or len(returns) < 5:
        st.error(f"❌ No valid returns data. Got {len(prices)} price days but only {len(returns)} return days.")
        st.info(f"""
        **Debug Information:**
        - Price data shape: {prices.shape}
        - Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}
        - Number of trading days: {len(prices)}
        - Valid return days: {len(returns)}
        
        **Suggestions:**
        1. Extend your date range to at least 1 month of data
        2. Use start date: 2020-01-01 or earlier
        3. Ensure end date is not today (use yesterday's date)
        """)
        st.stop()
    
    # Calculate portfolio returns
    p_rets = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    
    # Performance metrics
    ann_ret = p_rets.mean() * 252
    ann_vol = p_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Calculate maximum drawdown
    rolling_max = p_cum.expanding().max()
    daily_drawdown = (p_cum - rolling_max) / rolling_max
    max_dd = daily_drawdown.min()
    
    # Sortino ratio (only downside volatility)
    downside_returns = p_rets[p_rets < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    
    # --- DISPLAY RESULTS ---
    
    # Performance Metrics
    st.markdown('<div class="sub-header">Portfolio Performance</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card">Sharpe Ratio<br><h3>{sharpe:.2f}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card">Annual Return<br><h3>{ann_ret:.1%}</h3></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card">Annual Volatility<br><h3>{ann_vol:.1%}</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card">Max Drawdown<br><h3>{max_dd:.1%}</h3></div>', unsafe_allow_html=True)
    
    # Efficient Frontier Plot
    st.markdown('<div class="sub-header">Efficient Frontier</div>', unsafe_allow_html=True)
    frontier_fig = plot_efficient_frontier(bl_mu, S)
    if frontier_fig:
        st.plotly_chart(frontier_fig, use_container_width=True)
    else:
        st.info("Efficient frontier plot could not be generated with current data.")
    
    # Portfolio Allocation
    st.markdown('<div class="sub-header">Portfolio Allocation</div>', unsafe_allow_html=True)
    
    col_alloc1, col_alloc2 = st.columns([2, 1])
    
    with col_alloc1:
        # Pie chart
        w_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        w_df = w_df[w_df['Weight'] > 0.001]
        
        if not w_df.empty:
            fig_alloc = px.pie(
                names=w_df.index, 
                values=w_df['Weight'],
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Prism,
                title="Portfolio Weights"
            )
            fig_alloc.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("No significant allocations found.")
    
    with col_alloc2:
        # Weight table
        st.markdown("**Detailed Weights**")
        weight_table = pd.DataFrame({
            'Ticker': list(final_weights.keys()),
            'Weight': [f"{w:.2%}" for w in final_weights.values()]
        }).sort_values('Weight', ascending=False)
        st.dataframe(weight_table, use_container_width=True, hide_index=True)
        
        # Geopolitical impact summary
        if geo_events and geo_intensity > 0.5:
            st.markdown("**Geopolitical Impact**")
            st.info(f"Applied risk overlay for {len(geo_events)} event(s) at intensity {geo_intensity:.1f}x")
    
    # Performance Chart
    st.markdown('<div class="sub-header">Performance Comparison</div>', unsafe_allow_html=True)
    
    fig_perf = go.Figure()
    
    # Portfolio performance
    fig_perf.add_trace(go.Scatter(
        x=p_cum.index, 
        y=p_cum, 
        name="Optimized Portfolio", 
        line=dict(color='#1E3A8A', width=3),
        fill='tozeroy',
        fillcolor='rgba(30, 58, 138, 0.1)'
    ))
    
    # Benchmark performance if available
    if not bench_prices.empty:
        bench_returns = bench_prices.pct_change().dropna()
        # Align dates
        common_idx = p_cum.index.intersection(bench_returns.index)
        if len(common_idx) > 0:
            bench_aligned = bench_returns.loc[common_idx]
            b_cum = (1 + bench_aligned).cumprod()
            fig_perf.add_trace(go.Scatter(
                x=b_cum.index, 
                y=b_cum, 
                name="S&P 500 Index", 
                line=dict(color='#94A3B8', dash='dot', width=2)
            ))
    
    fig_perf.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # --- EXPORT SECTION ---
    st.divider()
    st.markdown('<div class="sub-header">Export Results</div>', unsafe_allow_html=True)
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Export portfolio weights
        export_weights = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        csv_weights = export_weights.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Portfolio Weights",
            data=csv_weights,
            file_name='portfolio_weights.csv',
            mime='text/csv'
        )
    
    with col_exp2:
        # Export performance data
        perf_data = pd.DataFrame({
            'Date': p_cum.index,
            'Portfolio_Return': p_rets.values,
            'Portfolio_Cumulative': p_cum.values
        })
        csv_perf = perf_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Performance Data",
            data=csv_perf,
            file_name='portfolio_performance.csv',
            mime='text/csv'
        )
    
    with col_exp3:
        # Export optimization parameters
        params = {
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d'),
            'Tickers': ', '.join(ticker_list),
            'Start_Date': start_date.strftime('%Y-%m-%d'),
            'End_Date': end_date.strftime('%Y-%m-%d'),
            'View_Asset': view_ticker,
            'View_Return': f"{view_return:.2%}",
            'View_Confidence': f"{view_conf:.0%}",
            'Max_Weight': f"{max_cap:.0%}",
            'Geopolitical_Events': ', '.join(geo_events) if geo_events else 'None',
            'Risk_Intensity': geo_intensity,
            'Sharpe_Ratio': f"{sharpe:.2f}",
            'Annual_Return': f"{ann_ret:.2%}",
            'Annual_Volatility': f"{ann_vol:.2%}"
        }
        params_df = pd.DataFrame([params])
        csv_params = params_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Strategy Parameters",
            data=csv_params,
            file_name='strategy_parameters.csv',
            mime='text/csv'
        )
    
    # --- RISK DISCLAIMER ---
    st.divider()
    with st.expander("⚠️ Risk Disclaimer"):
        st.markdown("""
        **Important Information:**
        
        - This tool is for educational and research purposes only
        - Past performance is not indicative of future results
        - All investments carry risk, including potential loss of principal
        - The models used make assumptions that may not hold in real markets
        - Geopolitical risk adjustments are qualitative estimates
        - Consult with a qualified financial advisor before making investment decisions
        
        *The outputs of this engine should not be considered investment advice.*
        """)

except Exception as e:
    st.error(f"🚨 Engine Execution Error: {str(e)}")
    
    # Provide helpful debugging information
    with st.expander("🔧 Technical Details"):
        st.code(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
    
    st.markdown("""
    <div class="info-box">
    <strong>Common issues and solutions:</strong><br>
    1. <strong>Date format issue:</strong> Fixed - now using proper date handling with end date<br>
    2. <strong>Invalid ticker symbols:</strong> Check if all tickers are correct and include exchange codes for international stocks (e.g., MC.PA for LVMH)<br>
    3. <strong>Yahoo Finance rate limits:</strong> Wait 1-2 minutes and try again with fewer tickers<br>
    4. <strong>Insufficient data:</strong> Using default start date 2020-01-02 which has plenty of history<br>
    5. <strong>Future dates:</strong> End date is now capped at today's date automatically<br>
    6. <strong>Returns calculation:</strong> Now properly validates minimum data requirements
    </div>
    """, unsafe_allow_html=True)
