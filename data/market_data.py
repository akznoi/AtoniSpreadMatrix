"""Market data fetching using yfinance."""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import yfinance as yf

from config import HISTORICAL_VOL_DAYS, TRADING_DAYS_PER_YEAR


def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch current stock information.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock info including price, name, etc.
    
    Raises:
        ValueError: If ticker is invalid or data unavailable
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        # Get current price - try multiple fields
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        
        if price is None:
            # Try to get from history as fallback
            hist = stock.history(period="1d")
            if hist.empty:
                raise ValueError(f"No data available for ticker: {ticker}")
            price = hist["Close"].iloc[-1]
        
        return {
            "ticker": ticker.upper(),
            "price": price,
            "name": info.get("shortName", info.get("longName", ticker.upper())),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
        }
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")


def get_stock_price(ticker: str) -> float:
    """
    Get current stock price.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Current stock price
    """
    info = get_stock_info(ticker)
    return info["price"]


def get_historical_volatility(
    ticker: str, 
    days: int = HISTORICAL_VOL_DAYS,
    annualize: bool = True
) -> float:
    """
    Calculate historical volatility from price data.
    
    Parameters:
        ticker: Stock ticker symbol
        days: Number of trading days to use for calculation
        annualize: Whether to annualize the volatility
    
    Returns:
        Historical volatility (annualized if specified)
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Fetch extra days to ensure we have enough data after filtering
        hist = stock.history(period=f"{days + 10}d")
        
        if len(hist) < 5:
            raise ValueError(f"Insufficient historical data for {ticker}")
        
        # Calculate daily log returns
        close_prices = hist["Close"].dropna()
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Use only the requested number of days
        log_returns = log_returns.tail(days)
        
        # Calculate standard deviation of returns
        daily_vol = log_returns.std()
        
        if annualize:
            return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        return daily_vol
    
    except Exception as e:
        raise ValueError(f"Error calculating volatility for {ticker}: {str(e)}")


def get_risk_free_rate() -> float:
    """
    Fetch current risk-free rate (10-year Treasury yield).
    Falls back to default if unavailable.
    
    Returns:
        Risk-free rate as decimal (e.g., 0.045 for 4.5%)
    """
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            # TNX is quoted in percentage points
            return hist["Close"].iloc[-1] / 100
    except:
        pass
    
    # Fallback to config default
    from config import RISK_FREE_RATE
    return RISK_FREE_RATE


def get_price_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch historical price data for charting.
    
    Parameters:
        ticker: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y, etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        raise ValueError(f"Error fetching price history for {ticker}: {str(e)}")


def get_trend_analysis(ticker: str) -> Dict[str, Any]:
    """
    Analyze price trend using moving averages.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with trend analysis
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="3mo")
        
        if len(hist) < 20:
            return {"trend": "Unknown", "strength": 0, "description": "Insufficient data"}
        
        # Calculate moving averages
        hist["SMA_20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA_50"] = hist["Close"].rolling(window=50).mean() if len(hist) >= 50 else hist["SMA_20"]
        
        current_price = hist["Close"].iloc[-1]
        sma_20 = hist["SMA_20"].iloc[-1]
        sma_50 = hist["SMA_50"].iloc[-1] if len(hist) >= 50 else sma_20
        
        # Calculate price change
        price_1w_ago = hist["Close"].iloc[-5] if len(hist) >= 5 else hist["Close"].iloc[0]
        price_1m_ago = hist["Close"].iloc[-20] if len(hist) >= 20 else hist["Close"].iloc[0]
        
        change_1w = ((current_price - price_1w_ago) / price_1w_ago) * 100
        change_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
        
        # Determine trend
        if current_price > sma_20 and sma_20 > sma_50:
            trend = "Bullish"
            strength = min(100, int(((current_price - sma_20) / sma_20) * 1000))
            description = "Price is above both moving averages, indicating upward momentum"
        elif current_price < sma_20 and sma_20 < sma_50:
            trend = "Bearish"
            strength = min(100, int(((sma_20 - current_price) / current_price) * 1000))
            description = "Price is below both moving averages, indicating downward momentum"
        elif current_price > sma_20:
            trend = "Slightly Bullish"
            strength = 50
            description = "Price is above short-term average but trend is mixed"
        elif current_price < sma_20:
            trend = "Slightly Bearish"
            strength = 50
            description = "Price is below short-term average but trend is mixed"
        else:
            trend = "Neutral"
            strength = 25
            description = "Price is consolidating around moving averages"
        
        return {
            "trend": trend,
            "strength": strength,
            "description": description,
            "current_price": current_price,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "change_1w": change_1w,
            "change_1m": change_1m,
        }
    except Exception as e:
        return {"trend": "Unknown", "strength": 0, "description": f"Error: {str(e)}"}


def get_stock_news(ticker: str, max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent news for a stock.
    
    Parameters:
        ticker: Stock ticker symbol
        max_items: Maximum number of news items to return
    
    Returns:
        List of news items with title, link, publisher, date
    """
    try:
        stock = yf.Ticker(ticker.upper())
        news = stock.news
        
        if not news:
            return []
        
        result = []
        for item in news[:max_items]:
            if not isinstance(item, dict):
                continue
                
            # yfinance news format has nested 'content' object
            content = item.get("content", {})
            if isinstance(content, dict):
                # Get title from content
                title = content.get("title", "")
                
                # Get link from clickThroughUrl or canonicalUrl
                click_url = content.get("clickThroughUrl", {})
                canon_url = content.get("canonicalUrl", {})
                link = (
                    (click_url.get("url") if isinstance(click_url, dict) else None) or
                    (canon_url.get("url") if isinstance(canon_url, dict) else None) or
                    "#"
                )
                
                # Get publisher from provider
                provider = content.get("provider", {})
                publisher = provider.get("displayName", "News") if isinstance(provider, dict) else "News"
                
                # Get date from pubDate (ISO format string)
                pub_date = content.get("pubDate", "")
                
                if title:
                    result.append({
                        "title": title,
                        "link": link,
                        "publisher": publisher,
                        "date": pub_date,  # ISO format string
                    })
            else:
                # Fallback for old format
                title = item.get("title", "")
                if title:
                    result.append({
                        "title": title,
                        "link": item.get("link", "#"),
                        "publisher": item.get("publisher", "News"),
                        "date": item.get("providerPublishTime", 0),
                    })
        
        return result
    except Exception as e:
        return []

