"""Streamlit UI components."""

from typing import Dict, Any, List, Optional, Union
import streamlit as st
import pandas as pd

from core.strategies.bull_put_spread import BullPutSpread
from core.strategies.spreads import VerticalSpread

# Type alias for any spread type
SpreadType = Union[BullPutSpread, VerticalSpread]


def display_stock_info(info: Dict[str, Any], volatility: float) -> None:
    """
    Display stock information in a formatted card.
    
    Parameters:
        info: Stock info dictionary
        volatility: Historical volatility
    """
    st.markdown(f"### {info['name']} ({info['ticker']})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${info['price']:.2f}")
    
    with col2:
        st.metric("30-Day Volatility", f"{volatility:.1%}")
    
    with col3:
        if info.get("fifty_two_week_high"):
            st.metric("52W High", f"${info['fifty_two_week_high']:.2f}")
    
    with col4:
        if info.get("fifty_two_week_low"):
            st.metric("52W Low", f"${info['fifty_two_week_low']:.2f}")


def display_spread_details(spread: SpreadType) -> None:
    """
    Display detailed information about a selected spread.
    
    Parameters:
        spread: Spread object (VerticalSpread, IronCondor, or IronButterfly)
    """
    st.markdown(f"### Selected Spread Details")
    
    # Check if this is a multi-leg strategy
    is_iron_condor = hasattr(spread, 'short_put_strike')
    is_iron_butterfly = hasattr(spread, 'center_strike')
    
    if is_iron_condor:
        # Iron Condor display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Position (4 Legs)")
            st.markdown(f"""
            **Put Credit Spread:**
            - Sell Put @ ${spread.short_put_strike:.2f}
            - Buy Put @ ${spread.long_put_strike:.2f}
            
            **Call Credit Spread:**
            - Sell Call @ ${spread.short_call_strike:.2f}
            - Buy Call @ ${spread.long_call_strike:.2f}
            
            **Expiration**: {spread.expiration_date.strftime('%B %d, %Y')}
            """)
        
        with col2:
            st.markdown("#### Key Metrics")
            st.markdown(f"""
            - **Total Credit**: ${spread.net_premium:.2f} per share (${spread.net_premium * 100:.2f} per contract)
            - **Profit Range**: ${spread.breakeven_lower:.2f} - ${spread.breakeven_upper:.2f}
            - **Width**: ${spread.profit_range:.2f} 
            """)
    
    elif is_iron_butterfly:
        # Iron Butterfly display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Position (4 Legs)")
            st.markdown(f"""
            **ATM Short Straddle:**
            - Sell Put @ ${spread.center_strike:.2f}
            - Sell Call @ ${spread.center_strike:.2f}
            
            **Wings (Protection):**
            - Buy Put @ ${spread.long_put_strike:.2f}
            - Buy Call @ ${spread.long_call_strike:.2f}
            
            **Expiration**: {spread.expiration_date.strftime('%B %d, %Y')}
            """)
        
        with col2:
            st.markdown("#### Key Metrics")
            st.markdown(f"""
            - **Total Credit**: ${spread.net_premium:.2f} per share (${spread.net_premium * 100:.2f} per contract)
            - **Profit Range**: ${spread.breakeven_lower:.2f} - ${spread.breakeven_upper:.2f}
            - **Wing Width**: ${spread.wing_width:.2f}
            """)
    
    else:
        # Regular vertical spread display
        if hasattr(spread, 'option_type'):
            option_type = spread.option_type.capitalize()
            short_leg = spread.short_leg
            long_leg = spread.long_leg
        else:
            option_type = "Put"
            short_leg = spread.short_put
            long_leg = spread.long_put
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Position")
            st.markdown(f"""
            - **Sell {option_type}** @ ${spread.short_strike:.2f} for ${short_leg.premium:.2f}
            - **Buy {option_type}** @ ${spread.long_strike:.2f} for ${long_leg.premium:.2f}
            - **Expiration**: {spread.expiration_date.strftime('%B %d, %Y')}
            """)
        
        with col2:
            st.markdown("#### Key Metrics")
            if hasattr(spread, 'net_premium'):
                net_prem = spread.net_premium
                prem_label = "Net Credit" if spread.is_credit else "Net Debit"
            else:
                net_prem = spread.net_credit
                prem_label = "Net Credit"
            st.markdown(f"""
            - **{prem_label}**: ${abs(net_prem):.2f} per share (${abs(net_prem) * 100:.2f} per contract)
            - **Spread Width**: ${spread.spread_width:.2f}
            - **Breakeven**: ${spread.breakeven:.2f}
            """)
    
    # P/L and Risk metrics
    st.markdown("---")
    
    # Adjust help text based on strategy type
    is_iron_condor = hasattr(spread, 'short_put_strike')
    is_iron_butterfly = hasattr(spread, 'center_strike')
    
    if is_iron_condor or is_iron_butterfly:
        profit_help = "Maximum profit if stock stays within breakeven range"
        loss_help = "Maximum loss if stock moves beyond wings"
    else:
        profit_help = "Maximum profit if stock stays above short strike"
        loss_help = "Maximum loss if stock falls below long strike"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Max Profit", 
            f"${spread.max_profit:.2f}",
            help=profit_help
        )
    
    with col2:
        st.metric(
            "Max Loss", 
            f"${spread.max_loss:.2f}",
            help=loss_help
        )
    
    with col3:
        st.metric(
            "Win Probability", 
            f"{spread.probability_of_profit:.1%}",
            help="Probability of profit based on delta"
        )
    
    with col4:
        st.metric(
            "Return on Capital", 
            f"{spread.return_on_capital:.1%}",
            help="Max profit / capital required"
        )
    
    # Trade Summary (Robinhood-style)
    st.markdown("---")
    st.markdown("#### Trade Summary")
    
    if is_iron_condor:
        # Iron Condor contracts
        st.caption("Put Credit Spread")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown(f"**${spread.short_put_strike:.2f} Put** - Sell to open")
        with pcol2:
            st.markdown(f"**${spread.long_put_strike:.2f} Put** - Buy to open")
        
        st.caption("Call Credit Spread")
        ccol1, ccol2 = st.columns(2)
        with ccol1:
            st.markdown(f"**${spread.short_call_strike:.2f} Call** - Sell to open")
        with ccol2:
            st.markdown(f"**${spread.long_call_strike:.2f} Call** - Buy to open")
        
        net_prem = spread.net_premium
        prem_label = "Total Credit"
        
    elif is_iron_butterfly:
        # Iron Butterfly contracts
        st.caption("Short ATM Straddle")
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown(f"**${spread.center_strike:.2f} Put** - Sell to open")
        with scol2:
            st.markdown(f"**${spread.center_strike:.2f} Call** - Sell to open")
        
        st.caption("Long Wings (Protection)")
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            st.markdown(f"**${spread.long_put_strike:.2f} Put** - Buy to open")
        with wcol2:
            st.markdown(f"**${spread.long_call_strike:.2f} Call** - Buy to open")
        
        net_prem = spread.net_premium
        prem_label = "Total Credit"
        
    else:
        # Regular vertical spread
        if hasattr(spread, 'option_type'):
            option_type = spread.option_type.capitalize()
            short_leg = spread.short_leg
            long_leg = spread.long_leg
            net_prem = abs(spread.net_premium)
            prem_label = "Total Credit" if spread.is_credit else "Total Debit"
        else:
            option_type = "Put"
            short_leg = spread.short_put
            long_leg = spread.long_put
            net_prem = spread.net_credit
            prem_label = "Total Credit"
        
        st.caption("Contracts")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown(f"**${spread.short_strike:.2f} {option_type}** - Sell to open")
            st.markdown(f"Premium: **${short_leg.premium:.2f}**")
        with tcol2:
            st.markdown(f"**${spread.long_strike:.2f} {option_type}** - Buy to open")
            st.markdown(f"Premium: **${long_leg.premium:.2f}**")
    
    st.markdown("")
    
    # Summary metrics in a highlighted box
    if is_iron_condor or is_iron_butterfly:
        # 5 columns for multi-leg strategies (2 breakevens)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.caption(prem_label)
            st.subheader(f"${net_prem:.2f}")
        
        with col2:
            st.caption("Max Profit")
            st.subheader(f":green[${spread.max_profit:.2f}]")
        
        with col3:
            st.caption("Lower BE")
            st.subheader(f"${spread.breakeven_lower:.2f}")
        
        with col4:
            st.caption("Upper BE")
            st.subheader(f"${spread.breakeven_upper:.2f}")
        
        with col5:
            st.caption("Max Loss")
            st.subheader(f":red[-${spread.max_loss:.2f}]")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.caption(prem_label)
            st.subheader(f"${net_prem:.2f}")
        
        with col2:
            st.caption("Max Profit")
            st.subheader(f":green[${spread.max_profit:.2f}]")
        
        with col3:
            st.caption("Breakeven")
            st.subheader(f"${spread.breakeven:.2f}")
        
        with col4:
            st.caption("Max Loss")
            st.subheader(f":red[-${spread.max_loss:.2f}]")
    
    # Greeks
    st.markdown("---")
    st.markdown("#### Greeks (per contract)")
    greeks = spread.get_greeks()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Delta", f"{greeks['delta']:.2f}")
    
    with col2:
        st.metric("Gamma", f"{greeks['gamma']:.4f}")
    
    with col3:
        st.metric("Theta", f"${greeks['theta']:.2f}/day")
    
    with col4:
        st.metric("Vega", f"${greeks['vega']:.2f}")
    
    with col5:
        st.metric("Capital Required", f"${spread.capital_requirement:.2f}")


def display_spreads_table(spreads: List[SpreadType]) -> Optional[int]:
    """
    Display a table of spread opportunities with selection.
    
    Parameters:
        spreads: List of spread objects (VerticalSpread, IronCondor, or IronButterfly)
    
    Returns:
        Index of selected spread, or None
    """
    if not spreads:
        st.warning("No spreads found matching your criteria. Try adjusting the probability or expiration date.")
        return None
    
    st.markdown("### Available Spreads")
    st.caption("Sorted by risk/reward ratio (lower is better)")
    
    # Check if this is a multi-leg strategy
    first_spread = spreads[0]
    is_iron_condor = hasattr(first_spread, 'short_put_strike')
    is_iron_butterfly = hasattr(first_spread, 'center_strike')
    
    # Create DataFrame for display
    data = []
    for i, spread in enumerate(spreads):
        # Handle different spread types
        if hasattr(spread, 'net_premium'):
            net_prem = abs(spread.net_premium)
            prem_label = "Credit" if getattr(spread, 'is_credit', True) or spread.strategy_type == "credit" else "Debit"
        else:
            net_prem = getattr(spread, 'net_credit', 0)
            prem_label = "Credit"
        
        # Calculate risk/reward ratio
        risk_reward = abs(spread.max_loss) / spread.max_profit if spread.max_profit > 0 else 0
        
        if is_iron_condor:
            data.append({
                "#": i + 1,
                "Expiration": spread.expiration_date.strftime("%b %d"),
                "Put Spread": f"${spread.short_put_strike:.0f}/${spread.long_put_strike:.0f}",
                "Call Spread": f"${spread.short_call_strike:.0f}/${spread.long_call_strike:.0f}",
                "Credit": f"${net_prem:.2f}",
                "Max Profit": f"${spread.max_profit:.2f}",
                "Max Loss": f"${spread.max_loss:.2f}",
                "Lower BE": f"${spread.breakeven_lower:.2f}",
                "Upper BE": f"${spread.breakeven_upper:.2f}",
                "Win Prob": f"{spread.probability_of_profit:.1%}",
                "Risk/Reward": f"{risk_reward:.2f}",
                "ROC": f"{spread.return_on_capital:.1%}",
            })
        elif is_iron_butterfly:
            data.append({
                "#": i + 1,
                "Expiration": spread.expiration_date.strftime("%b %d"),
                "Center": f"${spread.center_strike:.0f}",
                "Wings": f"${spread.long_put_strike:.0f} / ${spread.long_call_strike:.0f}",
                "Credit": f"${net_prem:.2f}",
                "Max Profit": f"${spread.max_profit:.2f}",
                "Max Loss": f"${spread.max_loss:.2f}",
                "Lower BE": f"${spread.breakeven_lower:.2f}",
                "Upper BE": f"${spread.breakeven_upper:.2f}",
                "Win Prob": f"{spread.probability_of_profit:.1%}",
                "Risk/Reward": f"{risk_reward:.2f}",
                "ROC": f"{spread.return_on_capital:.1%}",
            })
        else:
            # Regular vertical spread
            data.append({
                "#": i + 1,
                "Expiration": spread.expiration_date.strftime("%b %d"),
                "Short Strike": f"${spread.short_strike:.2f}",
                "Long Strike": f"${spread.long_strike:.2f}",
                "Width": f"${spread.spread_width:.2f}",
                f"{prem_label}/Share": f"${net_prem:.2f}",
                "Risk/Reward": f"{risk_reward:.2f}",
                "Max Profit": f"${spread.max_profit:.2f}",
                "Max Loss": f"${spread.max_loss:.2f}",
                "Breakeven": f"${spread.breakeven:.2f}",
                "Win Prob": f"{spread.probability_of_profit:.1%}",
                "ROC": f"{spread.return_on_capital:.1%}",
            })
    
    df = pd.DataFrame(data)
    
    # Remove columns with no data (but keep important columns like Expiration)
    cols_to_keep = ['#', 'Expiration', 'Win Prob', 'ROC', 'Risk/Reward']
    cols_to_drop = []
    for col in df.columns:
        if col in cols_to_keep:
            continue
        if df[col].isna().all() or (df[col] == '').all() or (df[col] == 'N/A').all():
            cols_to_drop.append(col)
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
    
    # Selection format based on strategy type
    if len(spreads) > 0:
        if is_iron_condor:
            format_func = lambda i: f"#{i+1}: Put ${spreads[i].short_put_strike:.0f}/${spreads[i].long_put_strike:.0f} | Call ${spreads[i].short_call_strike:.0f}/${spreads[i].long_call_strike:.0f} (Win: {spreads[i].probability_of_profit:.1%})"
        elif is_iron_butterfly:
            format_func = lambda i: f"#{i+1}: Center ${spreads[i].center_strike:.0f} | Wings ${spreads[i].long_put_strike:.0f}-${spreads[i].long_call_strike:.0f} (Win: {spreads[i].probability_of_profit:.1%})"
        else:
            format_func = lambda i: f"#{i+1}: ${spreads[i].short_strike:.2f}/${spreads[i].long_strike:.2f} spread (Win: {spreads[i].probability_of_profit:.1%})"
        
        selected = st.selectbox(
            "Select a spread to view details:",
            options=range(len(spreads)),
            format_func=format_func,
        )
        return selected
    
    return None


def display_error(message: str) -> None:
    """Display an error message."""
    st.error(f"⚠️ {message}")


def display_loading(message: str = "Loading...") -> None:
    """Display a loading spinner with message."""
    with st.spinner(message):
        pass


def display_iv_analysis(iv_data: Dict[str, Any]) -> None:
    """Display IV Rank and IV Percentile analysis."""
    st.markdown("### 📊 Implied Volatility Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    iv_rank = iv_data.get("iv_rank")
    iv_pctl = iv_data.get("iv_percentile")
    
    with col1:
        if iv_rank is not None:
            # Color code based on IV rank
            if iv_rank > 50:
                color = "green"
                label = "High (Good for Selling)"
            elif iv_rank > 25:
                color = "orange"
                label = "Moderate"
            else:
                color = "red"
                label = "Low (Good for Buying)"
            st.metric("IV Rank", f"{iv_rank:.1f}%", help="Where current IV stands in the 52-week range")
            st.caption(f":{color}[{label}]")
        else:
            st.metric("IV Rank", "N/A")
    
    with col2:
        if iv_pctl is not None:
            st.metric("IV Percentile", f"{iv_pctl:.1f}%", help="% of days IV was lower than today")
        else:
            st.metric("IV Percentile", "N/A")
    
    with col3:
        if iv_data.get("current_iv"):
            st.metric("Current IV", f"{iv_data['current_iv']:.1f}%")
        else:
            st.metric("Current IV", "N/A")
    
    with col4:
        iv_high = iv_data.get("iv_high")
        iv_low = iv_data.get("iv_low")
        if iv_high and iv_low:
            st.metric("52W IV Range", f"{iv_low:.1f}% - {iv_high:.1f}%")
        else:
            st.metric("52W IV Range", "N/A")


def display_earnings_dividend_info(earnings_data: Dict[str, Any], dividend_data: Dict[str, Any]) -> None:
    """Display earnings and dividend information."""
    st.markdown("### 📅 Key Dates")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        earnings_date = earnings_data.get("next_earnings_date")
        if earnings_date:
            days = earnings_data.get("days_until_earnings", 0)
            if days <= 7:
                st.metric("Next Earnings", earnings_date, delta=f"{days} days ⚠️", delta_color="inverse")
            else:
                st.metric("Next Earnings", earnings_date, delta=f"{days} days")
        else:
            st.metric("Next Earnings", "Unknown")
    
    with col2:
        if earnings_data.get("earnings_confirmed"):
            st.caption("✅ Confirmed")
        else:
            st.caption("⏳ Estimated")
    
    with col3:
        ex_div = dividend_data.get("ex_dividend_date")
        if ex_div:
            st.metric("Ex-Dividend Date", ex_div)
        else:
            st.metric("Ex-Dividend Date", "N/A")
    
    with col4:
        div_yield = dividend_data.get("dividend_yield")
        div_amt = dividend_data.get("dividend_amount")
        if div_yield and div_amt:
            st.metric("Dividend", f"${div_amt:.2f}", delta=f"{div_yield:.2f}% yield")
        elif div_amt:
            st.metric("Dividend", f"${div_amt:.2f}")
        else:
            st.metric("Dividend", "None")


def display_technical_indicators(rsi_data: Dict[str, Any], support_resistance: Dict[str, Any], beta_data: Dict[str, Any]) -> None:
    """Display technical indicators."""
    st.markdown("### 📈 Technical Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi = rsi_data.get("rsi")
        signal = rsi_data.get("signal", "")
        if rsi is not None:
            if signal == "Overbought":
                color = "red"
            elif signal == "Oversold":
                color = "green"
            else:
                color = "gray"
            st.metric("RSI (14)", f"{rsi:.1f}", help="Relative Strength Index")
            st.caption(f":{color}[{signal}]")
        else:
            st.metric("RSI (14)", "N/A")
    
    with col2:
        beta = beta_data.get("beta")
        interp = beta_data.get("interpretation", "")
        if beta is not None:
            st.metric("Beta", f"{beta:.2f}", help="Volatility relative to S&P 500")
            st.caption(interp)
        else:
            st.metric("Beta", "N/A")
    
    with col3:
        current = support_resistance.get("current_price", 0)
        supports = support_resistance.get("support_levels", [])
        resistances = support_resistance.get("resistance_levels", [])
        
        if supports:
            st.caption("Support Levels")
            for s in supports[:2]:
                pct = ((current - s) / current * 100)
                st.markdown(f"${s:.2f} ({pct:.1f}% below)")
        
        if resistances:
            st.caption("Resistance Levels")
            for r in resistances[:2]:
                pct = ((r - current) / current * 100)
                st.markdown(f"${r:.2f} ({pct:.1f}% above)")


def display_expected_move(expected_move_data: Dict[str, Any], current_price: float) -> None:
    """Display expected move based on IV."""
    st.markdown("### 🎯 Expected Move")
    
    col1, col2, col3 = st.columns(3)
    
    move = expected_move_data.get("expected_move")
    move_pct = expected_move_data.get("expected_move_pct")
    range_1sd = expected_move_data.get("range_1sd")
    range_2sd = expected_move_data.get("range_2sd")
    
    with col1:
        if move is not None:
            st.metric("Expected Move", f"±${move:.2f}", delta=f"±{move_pct:.1f}%")
        else:
            st.metric("Expected Move", "N/A")
    
    with col2:
        if range_1sd:
            st.metric("1σ Range (68%)", f"${range_1sd[0]:.2f} - ${range_1sd[1]:.2f}")
        else:
            st.metric("1σ Range (68%)", "N/A")
    
    with col3:
        if range_2sd:
            st.metric("2σ Range (95%)", f"${range_2sd[0]:.2f} - ${range_2sd[1]:.2f}")
        else:
            st.metric("2σ Range (95%)", "N/A")


def display_liquidity_analysis(liquidity_data: Dict[str, Any]) -> None:
    """Display options liquidity metrics."""
    st.markdown("### 💧 Liquidity Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = liquidity_data.get("liquidity_score")
        if score is not None:
            if score >= 70:
                color = "green"
                label = "Good"
            elif score >= 40:
                color = "orange"
                label = "Fair"
            else:
                color = "red"
                label = "Poor"
            st.metric("Liquidity Score", f"{score:.0f}/100")
            st.caption(f":{color}[{label}]")
        else:
            st.metric("Liquidity Score", "N/A")
    
    with col2:
        spread_pct = liquidity_data.get("avg_bid_ask_spread_pct")
        if spread_pct is not None:
            if spread_pct < 5:
                st.metric("Avg Bid-Ask Spread", f"{spread_pct:.1f}%", delta="Tight ✓", delta_color="off")
            else:
                st.metric("Avg Bid-Ask Spread", f"{spread_pct:.1f}%", delta="Wide ⚠", delta_color="off")
        else:
            st.metric("Avg Bid-Ask Spread", "N/A")
    
    with col3:
        vol_oi = liquidity_data.get("avg_volume_oi_ratio")
        if vol_oi is not None:
            st.metric("Volume/OI Ratio", f"{vol_oi:.2f}")
        else:
            st.metric("Volume/OI Ratio", "N/A")


def display_sentiment_analysis(put_call_data: Dict[str, Any]) -> None:
    """Display put/call ratio and sentiment."""
    st.markdown("### 🎭 Market Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ratio = put_call_data.get("put_call_volume_ratio")
        if ratio is not None:
            st.metric("P/C Volume Ratio", f"{ratio:.2f}", help="Put volume / Call volume")
        else:
            st.metric("P/C Volume Ratio", "N/A")
    
    with col2:
        oi_ratio = put_call_data.get("put_call_oi_ratio")
        if oi_ratio is not None:
            st.metric("P/C Open Interest", f"{oi_ratio:.2f}")
        else:
            st.metric("P/C Open Interest", "N/A")
    
    with col3:
        sentiment = put_call_data.get("sentiment")
        if sentiment:
            if sentiment == "Bullish":
                st.markdown("### 📈 :green[Bullish]")
            elif sentiment == "Bearish":
                st.markdown("### 📉 :red[Bearish]")
            else:
                st.markdown("### ➡️ :gray[Neutral]")
        else:
            st.markdown("### ❓ Unknown")


def display_historical_earnings(earnings_moves: Dict[str, Any]) -> None:
    """Display historical earnings move analysis."""
    st.markdown("### 📊 Historical Earnings Moves")
    
    avg_move = earnings_moves.get("avg_move")
    max_move = earnings_moves.get("max_move")
    min_move = earnings_moves.get("min_move")
    moves = earnings_moves.get("moves", [])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if avg_move is not None:
            st.metric("Avg Move", f"±{avg_move:.1f}%", help="Average absolute move on earnings")
        else:
            st.metric("Avg Move", "N/A")
    
    with col2:
        if max_move is not None:
            st.metric("Max Move", f"±{max_move:.1f}%")
        else:
            st.metric("Max Move", "N/A")
    
    with col3:
        if min_move is not None:
            st.metric("Min Move", f"±{min_move:.1f}%")
        else:
            st.metric("Min Move", "N/A")
    
    with col4:
        if moves:
            st.caption("Last 4 Earnings")
            st.markdown(" | ".join([f"±{m:.1f}%" for m in moves[:4]]))


def display_sector_performance(sector_data: Dict[str, Any]) -> None:
    """Display sector performance comparison."""
    st.markdown("### 🏢 Sector Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sector = sector_data.get("sector", "Unknown")
        industry = sector_data.get("industry", "")
        st.markdown(f"**{sector}**")
        if industry:
            st.caption(industry)
    
    with col2:
        stock_ret = sector_data.get("stock_1m_return")
        if stock_ret is not None:
            delta_color = "normal" if stock_ret >= 0 else "inverse"
            st.metric("Stock (1M)", f"{stock_ret:+.1f}%")
    
    with col3:
        sector_ret = sector_data.get("sector_1m_return")
        rs_sector = sector_data.get("rs_vs_sector")
        if sector_ret is not None:
            st.metric(f"Sector (1M)", f"{sector_ret:+.1f}%")
            if rs_sector is not None:
                if rs_sector > 0:
                    st.caption(f":green[+{rs_sector:.1f}% vs sector]")
                else:
                    st.caption(f":red[{rs_sector:.1f}% vs sector]")
    
    with col4:
        market_ret = sector_data.get("market_1m_return")
        rs_market = sector_data.get("rs_vs_market")
        if market_ret is not None:
            st.metric("S&P 500 (1M)", f"{market_ret:+.1f}%")
            if rs_market is not None:
                if rs_market > 0:
                    st.caption(f":green[+{rs_market:.1f}% vs market]")
                else:
                    st.caption(f":red[{rs_market:.1f}% vs market]")


def display_trade_management(trade_mgmt: Dict[str, Any], prob_50: float = None, max_profit: float = None) -> None:
    """Display trade management suggestions."""
    st.markdown("### 🎯 Trade Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Profit Targets (Close When)**")
        targets = trade_mgmt.get("profit_targets", {})
        if targets and max_profit:
            # Calculate debit to close at each profit level (per share)
            # Debit to close = (max_profit - profit_target) / 100
            close_50 = (max_profit - targets.get('50_pct', 0)) / 100
            close_75 = (max_profit - targets.get('75_pct', 0)) / 100
            close_90 = (max_profit - targets.get('90_pct', 0)) / 100
            st.markdown(f"- 50% Profit: **${targets.get('50_pct', 0):.2f}** | Close @ **${close_50:.2f}** ✓")
            st.markdown(f"- 75% Profit: ${targets.get('75_pct', 0):.2f} | Close @ ${close_75:.2f}")
            st.markdown(f"- 90% Profit: ${targets.get('90_pct', 0):.2f} | Close @ ${close_90:.2f}")
        elif targets:
            st.markdown(f"- 50% Profit: **${targets.get('50_pct', 0):.2f}** (Recommended)")
            st.markdown(f"- 75% Profit: ${targets.get('75_pct', 0):.2f}")
            st.markdown(f"- 90% Profit: ${targets.get('90_pct', 0):.2f}")
        
        if prob_50:
            st.markdown(f"📊 Prob. of 50% Profit: **{prob_50:.1f}%**")
    
    with col2:
        st.markdown("**Stop Loss Targets (Close When)**")
        if max_profit:
            # Calculate debit to close at each stop loss percentage
            # Stop loss = when you've lost X% of max profit potential
            loss_50 = max_profit * 0.50  # Lost 50% of credit
            loss_75 = max_profit * 0.75  # Lost 75% of credit
            loss_90 = max_profit * 0.90  # Lost 90% of credit
            
            # Debit to close = original credit + loss
            close_50 = (max_profit + loss_50) / 100  # Per share
            close_75 = (max_profit + loss_75) / 100
            close_90 = (max_profit + loss_90) / 100
            
            st.markdown(f"- 50% Loss: -${loss_50:.2f} | Close @ **${close_50:.2f}**")
            st.markdown(f"- 75% Loss: -${loss_75:.2f} | Close @ ${close_75:.2f}")
            st.markdown(f"- 90% Loss: -${loss_90:.2f} | Close @ ${close_90:.2f}")
    
    st.markdown("---")
    
    # Price alerts
    warning = trade_mgmt.get("warning_price")
    danger = trade_mgmt.get("danger_price")
    
    if warning and danger:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"⚠️ **Warning Price**: ${warning:.2f}")
        with col2:
            st.markdown(f"🚨 **Danger Price**: ${danger:.2f}")
        with col3:
            st.caption(trade_mgmt.get("roll_suggestion", ""))
    
    # Recommended action
    action = trade_mgmt.get("recommended_action")
    if action:
        st.info(f"💡 **Recommendation**: {action}")


def display_risk_alerts(
    dte_check: Dict[str, Any],
    iv_hv_comparison: Dict[str, Any],
    earnings_warning: Dict[str, Any],
    assignment_risk: Dict[str, Any] = None,
    prob_touch: Dict[str, Any] = None,
) -> None:
    """
    Display risk alerts and important indicators for options trading.
    
    Parameters:
        dte_check: Optimal DTE check result
        iv_hv_comparison: IV vs HV comparison result
        earnings_warning: Earnings before expiration check
        assignment_risk: Assignment risk check (optional, for selected spread)
        prob_touch: Probability of touch (optional, for selected spread)
    """
    st.markdown("### ⚠️ Risk Alerts & Indicators")
    
    # Row 1: DTE and IV vs HV
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Days to Expiration (DTE)**")
        dte = dte_check.get("days_to_expiry", 0)
        status = dte_check.get("status", "UNKNOWN")
        message = dte_check.get("message", "")
        
        # Color code based on status
        if status == "OPTIMAL":
            st.success(f"**{dte} DTE** - {status}")
        elif status in ["GOOD", "ACCEPTABLE"]:
            st.info(f"**{dte} DTE** - {status}")
        elif status in ["CAUTION", "FAR OUT"]:
            st.warning(f"**{dte} DTE** - {status}")
        else:
            st.error(f"**{dte} DTE** - {status}")
        
        st.caption(message)
    
    with col2:
        st.markdown("**IV vs Historical Volatility**")
        iv_pct = iv_hv_comparison.get("iv_pct")
        hv_pct = iv_hv_comparison.get("hv_pct")
        ratio = iv_hv_comparison.get("iv_hv_ratio")
        status = iv_hv_comparison.get("status")
        emoji = iv_hv_comparison.get("emoji", "")
        recommendation = iv_hv_comparison.get("recommendation", "")
        
        if iv_pct and hv_pct:
            st.metric(
                "IV / HV Ratio",
                f"{ratio:.2f}x" if ratio else "N/A",
                f"IV: {iv_pct:.1f}% | HV: {hv_pct:.1f}%"
            )
            st.caption(f"{emoji} {recommendation}")
        else:
            st.info("IV/HV data unavailable")
    
    # Row 2: Earnings Warning
    if earnings_warning.get("earnings_before_exp"):
        st.error(earnings_warning.get("warning", ""))
    
    # Row 3: Assignment Risk and Probability of Touch (only if spread selected)
    if assignment_risk or prob_touch:
        col1, col2 = st.columns(2)
        
        with col1:
            if assignment_risk:
                st.markdown("**Assignment Risk**")
                risk_level = assignment_risk.get("risk_level", "UNKNOWN")
                distance_pct = assignment_risk.get("distance_pct", 0)
                
                if risk_level == "HIGH":
                    st.error(f"🚨 **{risk_level}** - Short strike is ITM!")
                elif risk_level == "ELEVATED":
                    st.warning(f"⚠️ **{risk_level}** - {distance_pct:.1f}% from ITM")
                elif risk_level == "MODERATE":
                    st.info(f"**{risk_level}** - {distance_pct:.1f}% from ITM")
                else:
                    st.success(f"✅ **{risk_level}** - {distance_pct:.1f}% from ITM")
                
                if assignment_risk.get("warning"):
                    st.caption(assignment_risk.get("warning"))
        
        with col2:
            if prob_touch:
                st.markdown("**Probability of Touch**")
                p_touch = prob_touch.get("probability_of_touch", 0)
                p_itm = prob_touch.get("probability_itm", 0)
                
                if p_touch >= 70:
                    st.error(f"🎯 **{p_touch:.0f}%** chance price touches short strike")
                elif p_touch >= 50:
                    st.warning(f"🎯 **{p_touch:.0f}%** chance price touches short strike")
                else:
                    st.info(f"🎯 **{p_touch:.0f}%** chance price touches short strike")
                
                st.caption(f"(Prob of expiring ITM: {p_itm:.0f}%)")

