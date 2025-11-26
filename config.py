"""Application configuration constants."""

# Risk-free rate (10-year Treasury approximate)
RISK_FREE_RATE = 0.045  # 4.5%

# Default probability thresholds
DEFAULT_WIN_PROBABILITY = 0.60  # 60%
MIN_WIN_PROBABILITY = 0.50
MAX_WIN_PROBABILITY = 0.90

# Volatility calculation
HISTORICAL_VOL_DAYS = 30
TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365  # Calendar days per year

# UI Configuration
APP_TITLE = "AtoniSpreadMatrix - Options Strategy Analyzer"
APP_ICON = "📊"

# Spread configuration
DEFAULT_SPREAD_WIDTH = 5  # $5 between strikes
MAX_SPREADS_TO_DISPLAY = 10

