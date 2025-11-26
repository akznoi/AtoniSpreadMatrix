"""Plotly chart builders for options visualization."""

from typing import Union
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.strategies.bull_put_spread import BullPutSpread
from core.strategies.spreads import VerticalSpread

# Type alias for any spread type
SpreadType = Union[BullPutSpread, VerticalSpread]


def create_pl_chart(spread: SpreadType, current_price: float) -> go.Figure:
    """
    Create a P/L diagram for a bull put spread.
    
    Parameters:
        spread: BullPutSpread object
        current_price: Current stock price
    
    Returns:
        Plotly Figure object
    """
    # Calculate price range based on strikes
    min_strike = min(spread.short_strike, spread.long_strike)
    max_strike = max(spread.short_strike, spread.long_strike)
    price_min = min_strike * 0.85
    price_max = max_strike * 1.15
    
    # Generate price points
    prices = np.linspace(price_min, price_max, 200)
    
    # Calculate P/L at each price
    pnl = [spread.profit_at_price(p) for p in prices]
    
    # Create figure
    fig = go.Figure()
    
    # P/L line - split into profit (green) and loss (red) regions
    pnl_array = np.array(pnl)
    
    # Add filled area for profit
    fig.add_trace(go.Scatter(
        x=prices,
        y=np.maximum(pnl_array, 0),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.3)',
        line=dict(color='rgba(0, 200, 83, 0.5)', width=0),
        name='Profit Zone',
        showlegend=True,
    ))
    
    # Add filled area for loss
    fig.add_trace(go.Scatter(
        x=prices,
        y=np.minimum(pnl_array, 0),
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.3)',
        line=dict(color='rgba(255, 82, 82, 0.5)', width=0),
        name='Loss Zone',
        showlegend=True,
    ))
    
    # Main P/L line
    fig.add_trace(go.Scatter(
        x=prices,
        y=pnl,
        mode='lines',
        name='P/L at Expiration',
        line=dict(color='#1f77b4', width=3),
    ))
    
    # Add vertical lines for key prices
    # Current price
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="top",
    )
    
    # Breakeven
    fig.add_vline(
        x=spread.breakeven,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Breakeven: ${spread.breakeven:.2f}",
        annotation_position="bottom",
    )
    
    # Short strike
    fig.add_vline(
        x=spread.short_strike,
        line_dash="solid",
        line_color="red",
        opacity=0.5,
        annotation_text=f"Short: ${spread.short_strike:.2f}",
        annotation_position="top right",
    )
    
    # Long strike
    fig.add_vline(
        x=spread.long_strike,
        line_dash="solid",
        line_color="green",
        opacity=0.5,
        annotation_text=f"Long: ${spread.long_strike:.2f}",
        annotation_position="bottom left",
    )
    
    # Add horizontal line at zero
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    # Add annotations for max profit and max loss
    fig.add_annotation(
        x=price_max * 0.98,
        y=spread.max_profit,
        text=f"Max Profit: ${spread.max_profit:.2f}",
        showarrow=False,
        bgcolor="rgba(0, 200, 83, 0.8)",
        font=dict(color="white"),
    )
    
    fig.add_annotation(
        x=price_min * 1.02,
        y=-spread.max_loss,
        text=f"Max Loss: ${spread.max_loss:.2f}",
        showarrow=False,
        bgcolor="rgba(255, 82, 82, 0.8)",
        font=dict(color="white"),
    )
    
    # Get strategy name
    if hasattr(spread, 'name'):
        strategy_name = spread.name
    else:
        strategy_name = "Bull Put Spread"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{strategy_name} P/L Diagram: ${spread.short_strike:.2f}/${spread.long_strike:.2f}",
            font=dict(size=16),
        ),
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        hovermode="x unified",
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )
    
    # Update axes
    fig.update_xaxes(
        tickprefix="$",
        tickformat=",.0f",
        gridcolor='rgba(128, 128, 128, 0.2)',
    )
    
    fig.update_yaxes(
        tickprefix="$",
        tickformat="+,.0f",
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2,
    )
    
    return fig


def create_probability_chart(spread: SpreadType, current_price: float) -> go.Figure:
    """
    Create a probability distribution visualization showing win/loss zones.
    
    Parameters:
        spread: BullPutSpread object
        current_price: Current stock price
    
    Returns:
        Plotly Figure object
    """
    # Create a simple visualization of probability zones
    win_prob = spread.probability_of_profit
    loss_prob = 1 - win_prob
    
    fig = go.Figure()
    
    # Add bars for win/loss probability
    fig.add_trace(go.Bar(
        x=['Win (Above Breakeven)', 'Loss (Below Breakeven)'],
        y=[win_prob * 100, loss_prob * 100],
        marker_color=['#00c853', '#ff5252'],
        text=[f'{win_prob:.1%}', f'{loss_prob:.1%}'],
        textposition='auto',
        textfont=dict(size=16, color='white'),
    ))
    
    fig.update_layout(
        title=dict(
            text="Probability Distribution",
            font=dict(size=16),
        ),
        yaxis_title="Probability (%)",
        template="plotly_dark",
        height=300,
        showlegend=False,
    )
    
    fig.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
    )
    
    return fig


def create_price_chart(price_history, ticker: str, trend_info: dict) -> go.Figure:
    """
    Create a price chart with moving averages and trend indication.
    
    Parameters:
        price_history: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        trend_info: Dictionary with trend analysis
    
    Returns:
        Plotly Figure object
    """
    import pandas as pd
    
    fig = go.Figure()
    
    # Calculate moving averages
    price_history = price_history.copy()
    price_history["SMA_20"] = price_history["Close"].rolling(window=20).mean()
    if len(price_history) >= 50:
        price_history["SMA_50"] = price_history["Close"].rolling(window=50).mean()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=price_history.index,
        open=price_history["Open"],
        high=price_history["High"],
        low=price_history["Low"],
        close=price_history["Close"],
        name="Price",
        increasing_line_color='#00c853',
        decreasing_line_color='#ff5252',
    ))
    
    # Add 20-day SMA
    fig.add_trace(go.Scatter(
        x=price_history.index,
        y=price_history["SMA_20"],
        mode='lines',
        name='20-Day SMA',
        line=dict(color='#ffd700', width=2),
    ))
    
    # Add 50-day SMA if available
    if "SMA_50" in price_history.columns:
        fig.add_trace(go.Scatter(
            x=price_history.index,
            y=price_history["SMA_50"],
            mode='lines',
            name='50-Day SMA',
            line=dict(color='#00bcd4', width=2),
        ))
    
    # Determine trend color
    trend = trend_info.get("trend", "Neutral")
    if "Bullish" in trend:
        trend_color = "#00c853"
    elif "Bearish" in trend:
        trend_color = "#ff5252"
    else:
        trend_color = "#ffd700"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} Price Chart - <span style='color:{trend_color}'>{trend}</span>",
            font=dict(size=18),
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=450,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        xaxis_rangeslider_visible=False,
    )
    
    fig.update_xaxes(gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(gridcolor='rgba(128, 128, 128, 0.2)', tickprefix="$")
    
    return fig


def create_risk_reward_chart(spreads: list) -> go.Figure:
    """
    Create a scatter plot comparing spreads by risk/reward.
    
    Parameters:
        spreads: List of BullPutSpread objects
    
    Returns:
        Plotly Figure object
    """
    if not spreads:
        return go.Figure()
    
    fig = go.Figure()
    
    x_vals = [s.probability_of_profit * 100 for s in spreads]
    y_vals = [s.return_on_capital * 100 for s in spreads]
    labels = [f"${s.short_strike:.0f}/${s.long_strike:.0f}" for s in spreads]
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        marker=dict(
            size=15,
            color=y_vals,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="ROC %"),
        ),
        text=labels,
        textposition="top center",
        hovertemplate=(
            "Spread: %{text}<br>"
            "Win Probability: %{x:.1f}%<br>"
            "Return on Capital: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title=dict(
            text="Spread Comparison: Win Probability vs Return on Capital",
            font=dict(size=16),
        ),
        xaxis_title="Win Probability (%)",
        yaxis_title="Return on Capital (%)",
        template="plotly_dark",
        height=400,
    )
    
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")
    
    return fig

