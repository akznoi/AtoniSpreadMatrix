"""Options chain data fetching and processing."""

from datetime import date, datetime
from typing import List, Optional, Tuple, Union
import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def get_expiration_dates(ticker: str) -> List[date]:
    """
    Get available options expiration dates for a ticker.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        List of expiration dates sorted chronologically
    
    Raises:
        ValueError: If no options available for ticker
    """
    try:
        stock = yf.Ticker(ticker.upper())
        expirations = stock.options
        
        if not expirations:
            raise ValueError(f"No options available for {ticker}")
        
        # Convert to date objects
        dates = []
        for exp in expirations:
            try:
                dt = datetime.strptime(exp, "%Y-%m-%d").date()
                dates.append(dt)
            except ValueError:
                continue
        
        return sorted(dates)
    
    except Exception as e:
        if "No options available" in str(e):
            raise
        raise ValueError(f"Error fetching expiration dates for {ticker}: {str(e)}")


@st.cache_data(ttl=120, show_spinner=False)  # Cache for 2 minutes (options prices change frequently)
def get_options_chain(
    ticker: str, 
    expiration: Union[str, date]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get options chain (calls and puts) for a specific expiration.
    
    Parameters:
        ticker: Stock ticker symbol
        expiration: Expiration date (string 'YYYY-MM-DD' or date object)
    
    Returns:
        Tuple of (calls_df, puts_df) DataFrames
    
    Raises:
        ValueError: If unable to fetch options chain
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Convert date to string if needed
        if isinstance(expiration, date):
            expiration = expiration.strftime("%Y-%m-%d")
        
        chain = stock.option_chain(expiration)
        
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        
        # Clean up the data
        for df in [calls, puts]:
            # Ensure required columns exist
            if "impliedVolatility" not in df.columns:
                df["impliedVolatility"] = 0.3  # Default IV
            
            # Handle missing bid/ask
            if "bid" not in df.columns:
                df["bid"] = df.get("lastPrice", 0)
            if "ask" not in df.columns:
                df["ask"] = df.get("lastPrice", 0)
            
            # Fill NaN values
            df["bid"] = df["bid"].fillna(0)
            df["ask"] = df["ask"].fillna(0)
            df["lastPrice"] = df["lastPrice"].fillna(0)
            df["impliedVolatility"] = df["impliedVolatility"].fillna(0.3)
            df["volume"] = df["volume"].fillna(0)
            df["openInterest"] = df["openInterest"].fillna(0)
        
        return calls, puts
    
    except Exception as e:
        raise ValueError(f"Error fetching options chain for {ticker}: {str(e)}")


def calculate_time_to_expiry(expiration: date) -> float:
    """
    Calculate time to expiration in years.
    
    Parameters:
        expiration: Expiration date
    
    Returns:
        Time to expiration in years (fraction)
    """
    today = date.today()
    days_to_expiry = (expiration - today).days
    
    # Use calendar days / 365 for time calculation
    return max(days_to_expiry / 365.0, 0.001)  # Minimum to avoid division by zero


def get_atm_iv(puts_df: pd.DataFrame, current_price: float) -> float:
    """
    Get implied volatility of the at-the-money put option.
    
    Parameters:
        puts_df: DataFrame with put options
        current_price: Current stock price
    
    Returns:
        Implied volatility of nearest ATM option
    """
    if puts_df.empty:
        return 0.3  # Default
    
    # Find strike closest to current price
    puts_df = puts_df.copy()
    puts_df["distance"] = abs(puts_df["strike"] - current_price)
    atm_row = puts_df.loc[puts_df["distance"].idxmin()]
    
    iv = atm_row.get("impliedVolatility", 0.3)
    
    # Sanity check
    if iv <= 0 or iv > 5:
        return 0.3
    
    return iv


def filter_liquid_options(
    df: pd.DataFrame, 
    min_volume: int = 0,
    min_open_interest: int = 10,
    max_spread_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Filter options chain to only include liquid options.
    
    Parameters:
        df: Options chain DataFrame
        min_volume: Minimum daily volume
        min_open_interest: Minimum open interest
        max_spread_pct: Maximum bid-ask spread as percentage of mid price
    
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    # Filter by open interest
    filtered = filtered[filtered["openInterest"] >= min_open_interest]
    
    # Filter by volume (if specified)
    if min_volume > 0:
        filtered = filtered[filtered["volume"] >= min_volume]
    
    # Filter by bid-ask spread
    if max_spread_pct > 0:
        mid_price = (filtered["bid"] + filtered["ask"]) / 2
        spread = filtered["ask"] - filtered["bid"]
        spread_pct = spread / mid_price.replace(0, 1)
        filtered = filtered[spread_pct <= max_spread_pct]
    
    return filtered

