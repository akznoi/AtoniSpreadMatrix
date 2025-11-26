"""
MyOptionsPick - Options Strategy Analyzer
Main Streamlit Application
"""

import streamlit as st
from datetime import date

from config import (
    APP_TITLE,
    APP_ICON,
    DEFAULT_WIN_PROBABILITY,
    MIN_WIN_PROBABILITY,
    MAX_WIN_PROBABILITY,
    RISK_FREE_RATE,
    MAX_SPREADS_TO_DISPLAY,
)
from data.market_data import (
    get_stock_info, 
    get_historical_volatility, 
    get_risk_free_rate,
    get_price_history,
    get_trend_analysis,
    get_stock_news,
)
from data.options_chain import (
    get_expiration_dates,
    get_options_chain,
    calculate_time_to_expiry,
    get_atm_iv,
    filter_liquid_options,
)
from core.strategies.spreads import find_vertical_spreads
from ui.components import (
    display_stock_info,
    display_spread_details,
    display_spreads_table,
    display_error,
    display_iv_analysis,
    display_earnings_dividend_info,
    display_technical_indicators,
    display_expected_move,
    display_liquidity_analysis,
    display_sentiment_analysis,
    display_historical_earnings,
    display_sector_performance,
    display_trade_management,
)
from core.analytics import (
    calculate_iv_rank,
    get_earnings_info,
    get_dividend_info,
    calculate_liquidity_score,
    calculate_put_call_ratio,
    calculate_rsi,
    calculate_support_resistance,
    calculate_expected_move,
    get_beta,
    calculate_prob_50_profit,
    get_historical_earnings_moves,
    get_sector_performance,
    get_trade_management_suggestions,
)
from ui.charts import create_pl_chart, create_probability_chart, create_risk_reward_chart, create_price_chart
from datetime import datetime
from utils.validation import validate_ticker


# Strategy definitions
STRATEGIES = {
    "Put Credit Spread": {"option_type": "put", "strategy_type": "credit", "direction": "Bullish"},
    "Put Debit Spread": {"option_type": "put", "strategy_type": "debit", "direction": "Bearish"},
    "Call Credit Spread": {"option_type": "call", "strategy_type": "credit", "direction": "Bearish"},
    "Call Debit Spread": {"option_type": "call", "strategy_type": "debit", "direction": "Bullish"},
}


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple CSS styling (uses Streamlit's default light theme)
st.markdown("""
<style>
    .main-header {
        font-size: 7.5rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-bottom: 0;
        letter-spacing: -2px;
    }
    .sub-header {
        color: #555;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .strategy-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .bullish {
        background-color: rgba(0, 200, 83, 0.2);
        color: #00c853;
    }
    .bearish {
        background-color: rgba(255, 82, 82, 0.2);
        color: #ff5252;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">AtoniSpreadMatrix</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Options Strategy Analyzer - Vertical Spreads</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.header("Strategy Parameters")
    
    # Ticker input
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        max_chars=5,
        help="Enter a stock symbol (e.g., AAPL, MSFT, SPY)",
    ).upper().strip()
    
    # Strategy type
    strategy = st.selectbox(
        "Strategy",
        options=list(STRATEGIES.keys()),
        help="Select options strategy to analyze",
    )
    
    # Show strategy direction
    strategy_info = STRATEGIES[strategy]
    direction_class = "bullish" if strategy_info["direction"] == "Bullish" else "bearish"
    st.markdown(f'<span class="strategy-badge {direction_class}">{strategy_info["direction"]} Strategy</span>', unsafe_allow_html=True)
    
    # Strategy description
    if strategy == "Put Credit Spread":
        st.caption("Sell higher strike put, buy lower strike put. Profit if stock stays above short strike.")
    elif strategy == "Put Debit Spread":
        st.caption("Buy higher strike put, sell lower strike put. Profit if stock falls below long strike.")
    elif strategy == "Call Credit Spread":
        st.caption("Sell lower strike call, buy higher strike call. Profit if stock stays below short strike.")
    elif strategy == "Call Debit Spread":
        st.caption("Buy lower strike call, sell higher strike call. Profit if stock rises above long strike.")
    
    st.markdown("")
    
    # Win probability slider
    win_probability = st.slider(
        "Target Win Probability",
        min_value=int(MIN_WIN_PROBABILITY * 100),
        max_value=int(MAX_WIN_PROBABILITY * 100),
        value=int(DEFAULT_WIN_PROBABILITY * 100),
        step=5,
        format="%d%%",
        help="Minimum probability of profit (based on delta)",
    ) / 100
    
    # Spread width options
    spread_widths = st.multiselect(
        "Spread Widths ($)",
        options=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0],
        default=[2.5, 5.0, 10.0],
        help="Strike width between short and long options",
    )
    
    # Liquidity filter
    min_open_interest = st.number_input(
        "Min Open Interest",
        min_value=0,
        max_value=1000,
        value=10,
        step=10,
        help="Minimum open interest for options to consider",
    )
    
    st.markdown("---")
    
    # Analyze button
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Main content
if ticker:
    # Validate ticker
    is_valid, error_msg = validate_ticker(ticker)
    
    if not is_valid:
        display_error(error_msg)
    else:
        try:
            # Fetch stock info
            with st.spinner(f"Fetching data for {ticker}..."):
                stock_info = get_stock_info(ticker)
                current_price = stock_info["price"]
                hist_vol = get_historical_volatility(ticker)
                risk_free = get_risk_free_rate()
            
            # Display stock info
            display_stock_info(stock_info, hist_vol)
            
            st.markdown("---")
            
            # Advanced Analytics Section
            with st.expander("📊 **Advanced Analytics** (IV, Earnings, Technical, Sentiment)", expanded=True):
                # Fetch all analytics data
                with st.spinner("Loading analytics..."):
                    # Row 1: IV Analysis + Earnings/Dividends
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        iv_data = calculate_iv_rank(ticker, hist_vol)
                        display_iv_analysis(iv_data)
                    
                    with col_right:
                        earnings_data = get_earnings_info(ticker)
                        dividend_data = get_dividend_info(ticker)
                        display_earnings_dividend_info(earnings_data, dividend_data)
                    
                    st.markdown("---")
                    
                    # Row 2: Technical Indicators
                    rsi_data = calculate_rsi(ticker)
                    support_resistance = calculate_support_resistance(ticker)
                    beta_data = get_beta(ticker)
                    display_technical_indicators(rsi_data, support_resistance, beta_data)
                    
                    st.markdown("---")
                    
                    # Row 3: Sentiment + Historical Earnings
                    col_left2, col_right2 = st.columns(2)
                    
                    with col_left2:
                        put_call_data = calculate_put_call_ratio(ticker)
                        display_sentiment_analysis(put_call_data)
                    
                    with col_right2:
                        earnings_moves = get_historical_earnings_moves(ticker)
                        display_historical_earnings(earnings_moves)
                    
                    st.markdown("---")
                    
                    # Row 4: Sector Performance
                    sector_data = get_sector_performance(ticker)
                    display_sector_performance(sector_data)
            
            st.markdown("---")
            
            # Fetch expiration dates
            try:
                expirations = get_expiration_dates(ticker)
                
                if not expirations:
                    display_error(f"No options available for {ticker}")
                else:
                    # Expiration date selector
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Format dates for display
                        exp_options = {
                            exp: f"{exp.strftime('%b %d, %Y')} ({(exp - date.today()).days} days)"
                            for exp in expirations[:12]  # Limit to next 12 expirations
                        }
                        
                        selected_exp = st.selectbox(
                            "Expiration Date",
                            options=list(exp_options.keys()),
                            format_func=lambda x: exp_options[x],
                        )
                    
                    with col2:
                        days_to_exp = (selected_exp - date.today()).days
                        st.metric("Days to Expiration", days_to_exp)
                    
                    # Get strategy configuration
                    strat_config = STRATEGIES[strategy]
                    option_type = strat_config["option_type"]
                    strategy_type = strat_config["strategy_type"]
                    
                    # Analyze spreads when button clicked or expiration changes
                    if analyze_clicked or "last_analysis" not in st.session_state or st.session_state.get("last_strategy") != strategy:
                        with st.spinner(f"Analyzing {strategy} opportunities..."):
                            # Fetch options chain
                            calls, puts = get_options_chain(ticker, selected_exp)
                            
                            # Select the appropriate options chain
                            options_df = calls if option_type == "call" else puts
                            
                            # Filter for liquidity
                            options_filtered = filter_liquid_options(
                                options_df,
                                min_open_interest=min_open_interest,
                            )
                            
                            if options_filtered.empty:
                                st.warning(f"No liquid {option_type} options found. Try lowering the minimum open interest.")
                            else:
                                # Calculate time to expiry
                                tte = calculate_time_to_expiry(selected_exp)
                                
                                # Get ATM implied volatility
                                atm_iv = get_atm_iv(options_filtered, current_price)
                                
                                # Find spreads
                                spreads = find_vertical_spreads(
                                    options_df=options_filtered,
                                    underlying_price=current_price,
                                    expiration_date=selected_exp,
                                    time_to_expiry=tte,
                                    risk_free_rate=risk_free,
                                    option_type=option_type,
                                    strategy_type=strategy_type,
                                    min_probability=win_probability,
                                    spread_widths=spread_widths if spread_widths else [5.0],
                                    max_results=MAX_SPREADS_TO_DISPLAY,
                                )
                                
                                # Store in session state
                                st.session_state["last_analysis"] = {
                                    "spreads": spreads,
                                    "current_price": current_price,
                                    "strategy": strategy,
                                    "options_df": options_filtered,
                                    "days_to_exp": days_to_exp,
                                    "hist_vol": hist_vol,
                                }
                                st.session_state["last_strategy"] = strategy
                    
                    # Display results
                    if "last_analysis" in st.session_state:
                        spreads = st.session_state["last_analysis"]["spreads"]
                        analysis_price = st.session_state["last_analysis"]["current_price"]
                        
                        if spreads:
                            # Risk/reward comparison chart
                            st.plotly_chart(
                                create_risk_reward_chart(spreads),
                                use_container_width=True,
                            )
                            
                            # Spreads table with selection
                            selected_idx = display_spreads_table(spreads)
                            
                            if selected_idx is not None:
                                selected_spread = spreads[selected_idx]
                                
                                st.markdown("---")
                                
                                # Display detailed view
                                display_spread_details(selected_spread)
                                
                                st.markdown("---")
                                
                                # Expected Move and Liquidity Analysis
                                with st.expander("🎯 **Expected Move & Liquidity**", expanded=True):
                                    col_em, col_liq = st.columns(2)
                                    
                                    with col_em:
                                        # Calculate expected move based on ATM IV
                                        try:
                                            analysis_data = st.session_state.get("last_analysis", {})
                                            stored_vol = analysis_data.get("hist_vol", hist_vol)
                                            stored_days = analysis_data.get("days_to_exp", days_to_exp)
                                            exp_move_data = calculate_expected_move(
                                                current_price=analysis_price,
                                                iv=stored_vol,
                                                days_to_expiry=stored_days,
                                            )
                                            display_expected_move(exp_move_data, analysis_price)
                                        except:
                                            st.info("Expected move data unavailable")
                                    
                                    with col_liq:
                                        # Calculate liquidity for selected options
                                        try:
                                            analysis_data = st.session_state.get("last_analysis", {})
                                            stored_options = analysis_data.get("options_df")
                                            if stored_options is not None:
                                                liq_data = calculate_liquidity_score(stored_options)
                                                display_liquidity_analysis(liq_data)
                                            else:
                                                st.info("Liquidity data unavailable")
                                        except:
                                            st.info("Liquidity data unavailable")
                                
                                # Trade Management Section
                                with st.expander("🎯 **Trade Management Suggestions**", expanded=True):
                                    try:
                                        trade_mgmt = get_trade_management_suggestions(
                                            selected_spread,
                                            analysis_price
                                        )
                                        prob_50 = calculate_prob_50_profit(selected_spread)
                                        display_trade_management(trade_mgmt, prob_50)
                                    except Exception as e:
                                        st.info(f"Trade management suggestions unavailable")
                                
                                st.markdown("---")
                                
                                # Charts
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.plotly_chart(
                                        create_pl_chart(selected_spread, analysis_price),
                                        use_container_width=True,
                                    )
                                
                                with col2:
                                    st.plotly_chart(
                                        create_probability_chart(selected_spread, analysis_price),
                                        use_container_width=True,
                                    )
                        else:
                            st.info(
                                f"No {strategy} opportunities found with ≥{win_probability:.0%} win probability. "
                                "Try lowering the probability threshold or selecting a different expiration."
                            )
            
            except ValueError as e:
                display_error(str(e))
        
        except ValueError as e:
            display_error(str(e))
        except Exception as e:
            display_error(f"An unexpected error occurred: {str(e)}")
else:
    st.info("👈 Enter a stock ticker in the sidebar to get started")

# Trend Analysis and News Section
if ticker and is_valid:
    st.markdown("---")
    st.markdown("## 📊 Market Analysis")
    
    try:
        # Fetch trend data
        trend_info = get_trend_analysis(ticker)
        price_history = get_price_history(ticker, period="3mo")
        
        # Display trend summary
        col1, col2, col3, col4 = st.columns(4)
        
        trend = trend_info.get("trend", "Unknown")
        if "Bullish" in trend:
            trend_color = "green"
            trend_icon = "📈"
        elif "Bearish" in trend:
            trend_color = "red"
            trend_icon = "📉"
        else:
            trend_color = "orange"
            trend_icon = "➡️"
        
        with col1:
            st.metric(
                "Trend",
                f"{trend_icon} {trend}",
            )
        
        with col2:
            change_1w = trend_info.get("change_1w", 0)
            st.metric(
                "1-Week Change",
                f"{change_1w:+.2f}%",
                delta=f"{change_1w:.2f}%",
            )
        
        with col3:
            change_1m = trend_info.get("change_1m", 0)
            st.metric(
                "1-Month Change",
                f"{change_1m:+.2f}%",
                delta=f"{change_1m:.2f}%",
            )
        
        with col4:
            sma_20 = trend_info.get("sma_20", 0)
            st.metric(
                "20-Day SMA",
                f"${sma_20:.2f}",
            )
        
        # Display trend description
        st.caption(trend_info.get("description", ""))
        
        # Price chart
        if not price_history.empty:
            st.plotly_chart(
                create_price_chart(price_history, ticker, trend_info),
                use_container_width=True,
            )
        
        # News section
        st.markdown("### 📰 Latest News")
        
        news_items = get_stock_news(ticker, max_items=10)
        
        if news_items:
            for i, news in enumerate(news_items):
                # Format date
                date_val = news.get("date", "")
                news_date = ""
                if date_val:
                    try:
                        if isinstance(date_val, str):
                            # ISO format string like "2025-11-26T19:24:37Z"
                            dt = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                            news_date = dt.strftime("%b %d, %Y")
                        elif isinstance(date_val, (int, float)):
                            # Unix timestamp
                            news_date = datetime.fromtimestamp(date_val).strftime("%b %d, %Y")
                    except:
                        news_date = ""
                
                publisher = news.get("publisher", "")
                title = news.get("title", "")
                link = news.get("link", "#")
                
                # Display news item
                st.markdown(
                    f"**{i+1}.** [{title}]({link})  \n"
                    f"<span style='color: #888; font-size: 0.85rem;'>{publisher} • {news_date}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent news available for this ticker.")
    
    except Exception as e:
        st.warning(f"Unable to load market analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>Data provided by Yahoo Finance. Options trading involves risk. 
        This tool is for educational purposes only.</p>
        <p>Probability calculations are based on delta approximation and may not reflect actual outcomes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
