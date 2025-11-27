# AtoniSpreadMatrix - Options Strategy Analyzer

A comprehensive Python web application for analyzing options spread strategies with advanced analytics and risk management tools.

**Live Demo**: [atonispreadmatrix.streamlit.app](https://atonispreadmatrix.streamlit.app/)

## Features

### 📊 Strategy Analysis

| Strategy | Outlook | Description |
|----------|---------|-------------|
| **Put Credit Spread** | Bullish | Sell higher strike put, buy lower strike put |
| **Put Debit Spread** | Bearish | Buy higher strike put, sell lower strike put |
| **Call Credit Spread** | Bearish | Sell lower strike call, buy higher strike call |
| **Call Debit Spread** | Bullish | Buy lower strike call, sell higher strike call |
| **Iron Condor** | Neutral | Put credit spread + Call credit spread. Profit when stock stays in range. |
| **Iron Butterfly** | Neutral | Short ATM straddle + long OTM wings. Higher premium, narrower profit zone. |

### ⚠️ Risk Alerts & Indicators
- **Optimal DTE Check**: Highlights if expiration is in the 30-45 day sweet spot
- **IV vs HV Comparison**: Shows if options are expensive (sell) or cheap (buy)
- **Earnings Warning**: Alerts if earnings fall before expiration (IV crush risk)
- **Assignment Risk**: Monitors how close short strike is to ITM
- **Probability of Touch**: ~2x probability of ITM - likelihood price touches short strike

### 📈 Advanced Analytics
- **IV Rank & IV Percentile**: Determine if options are expensive or cheap
- **Earnings & Dividend Dates**: Avoid surprise events
- **RSI Indicator**: Overbought/Oversold signals
- **Beta**: Stock volatility vs market
- **Support/Resistance Levels**: Key price zones
- **Put/Call Ratio**: Market sentiment analysis
- **Sector Performance**: Compare stock vs sector vs market
- **Expected Move**: IV-based price range predictions
- **Liquidity Score**: Bid-ask spread analysis

### 🎯 Trade Management
- **Probability of 50% Profit**: More realistic profit targets
- **Profit Targets**: 50%, 75%, 90% levels with close prices
- **Stop Loss Levels**: 50%, 75%, 90% loss with close prices
- **Warning/Danger Price Alerts**: Know when to act
- **Roll Recommendations**: When to adjust positions

### 📰 Market Intelligence
- **Trend Analysis**: Bullish/Bearish identification with moving averages
- **Price Charts**: Interactive candlestick with 20/50-day SMA
- **Latest News**: Top 10 headlines for the ticker

### 💹 Core Features
- **Delta-Based Probability**: Win probability using option delta
- **Full Greeks Display**: Delta, Gamma, Theta, Vega
- **Interactive P/L Charts**: Visualize profit/loss at expiration
- **Risk/Reward Comparison**: Compare spreads side-by-side
- **Real-Time Data**: Live market data via Yahoo Finance
- **All Strike Combinations**: Shows all available spreads, not just preset widths

## Quick Start

### 1. Setup Environment

```bash
cd AtoniSpreadMatrix
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

Opens your browser to `http://localhost:8501`

## How to Use

1. **Enter a Ticker**: Type a stock symbol (e.g., AAPL, SPY, TSLA)
2. **Select Strategy**: Choose from vertical spreads or Iron Condor/Butterfly
3. **Set Win Probability**: Use the slider (50-90%)
4. **Set Min Open Interest**: Filter for liquidity (default: 1)
5. **Click Analyze**: Load the data
6. **Choose Expiration**: Select from available dates (defaults to ~30 days)
7. **Review Risk Alerts**: Check DTE, IV vs HV, earnings warnings
8. **Select a Spread**: See detailed P/L chart, Greeks, and trade management tips

## Understanding the Output

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Net Credit/Debit** | Premium received or paid when opening |
| **Max Profit** | Best case scenario at expiration |
| **Max Loss** | Worst case scenario at expiration |
| **Breakeven** | Stock price where P/L = $0 |
| **Win Probability** | Probability of profit (delta-based) |
| **ROC** | Return on Capital = Max profit ÷ Capital required |
| **Risk/Reward** | Max Loss ÷ Max Profit (lower is better) |

### Risk Alert Indicators

| Indicator | What It Tells You |
|-----------|-------------------|
| **DTE: OPTIMAL** | 30-45 days - ideal theta decay |
| **DTE: CAUTION** | <21 days - high gamma risk |
| **IV > HV** | Options expensive - good for selling |
| **IV < HV** | Options cheap - good for buying |
| **Earnings Before Exp** | IV will crush after earnings! |

### IV Analysis

| Metric | Meaning |
|--------|---------|
| **IV Rank > 50%** | Options are expensive → Good for selling |
| **IV Rank < 50%** | Options are cheap → Good for buying |
| **IV Percentile** | % of days IV was lower than today |

## Technology Stack

- **Streamlit**: Web framework
- **yfinance**: Yahoo Finance data
- **NumPy/SciPy**: Numerical calculations (Black-Scholes)
- **Plotly**: Interactive charts
- **Pandas**: Data manipulation

## Deployment

### Streamlit Community Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Deploy!

### Requirements
- Python 3.9+
- See `requirements.txt` for dependencies

## Disclaimer

⚠️ This tool is for **educational purposes only**. Options trading involves significant risk of loss. The probability calculations are approximations based on delta and may not reflect actual market outcomes. Always do your own research and consider paper trading and consulting a financial advisor before trading options.

---

**Created by JPatino**

## License

MIT License - See LICENSE file for details.
