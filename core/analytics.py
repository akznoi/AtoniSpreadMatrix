"""Advanced analytics for options trading decisions."""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from config import DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR


@st.cache_data(ttl=300, show_spinner=False)
def calculate_iv_rank(ticker: str, current_iv: float, lookback_days: int = 252) -> Dict[str, Any]:
    """
    Calculate IV Rank and IV Percentile.
    
    IV Rank: Where current IV stands relative to the high-low range over the past year
    IV Percentile: Percentage of days IV was lower than today
    
    Parameters:
        ticker: Stock ticker symbol
        current_iv: Current implied volatility (as decimal, e.g., 0.25 for 25%)
        lookback_days: Number of trading days to look back (default 252 = 1 year)
    
    Returns:
        Dictionary with iv_rank, iv_percentile, iv_high, iv_low
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="1y")
        
        if len(hist) < 20:
            return {
                "iv_rank": None,
                "iv_percentile": None,
                "iv_high": None,
                "iv_low": None,
                "error": "Insufficient historical data"
            }
        
        # Calculate historical volatility as proxy for IV history
        # (True IV history requires options data which isn't readily available)
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        
        # Calculate rolling 20-day volatility annualized
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) < 10:
            return {
                "iv_rank": None,
                "iv_percentile": None,
                "iv_high": None,
                "iv_low": None,
                "error": "Insufficient volatility data"
            }
        
        iv_high = rolling_vol.max()
        iv_low = rolling_vol.min()
        
        # IV Rank = (Current IV - 52 Week Low) / (52 Week High - 52 Week Low)
        if iv_high - iv_low > 0:
            iv_rank = (current_iv - iv_low) / (iv_high - iv_low) * 100
        else:
            iv_rank = 50.0
        
        # IV Percentile = % of days where IV was lower than current
        iv_percentile = (rolling_vol < current_iv).sum() / len(rolling_vol) * 100
        
        return {
            "iv_rank": round(min(max(iv_rank, 0), 100), 1),
            "iv_percentile": round(min(max(iv_percentile, 0), 100), 1),
            "iv_high": round(iv_high * 100, 1),  # Convert to percentage
            "iv_low": round(iv_low * 100, 1),
            "current_iv": round(current_iv * 100, 1),
        }
    except Exception as e:
        return {
            "iv_rank": None,
            "iv_percentile": None,
            "iv_high": None,
            "iv_low": None,
            "error": str(e)
        }


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_earnings_info(ticker: str) -> Dict[str, Any]:
    """
    Get earnings date information.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with next_earnings_date, days_until_earnings, earnings_confirmed
    """
    try:
        stock = yf.Ticker(ticker.upper())
        calendar = stock.calendar
        
        if calendar is None or calendar.empty:
            return {
                "next_earnings_date": None,
                "days_until_earnings": None,
                "earnings_confirmed": False,
                "error": "No earnings data available"
            }
        
        # Handle different calendar formats
        if isinstance(calendar, pd.DataFrame):
            if "Earnings Date" in calendar.columns:
                earnings_dates = calendar["Earnings Date"].tolist()
            elif "Earnings Date" in calendar.index:
                earnings_dates = [calendar.loc["Earnings Date", 0]]
            else:
                # Try to get from the first column
                earnings_dates = calendar.iloc[0].tolist() if len(calendar) > 0 else []
        else:
            earnings_dates = []
        
        if not earnings_dates:
            return {
                "next_earnings_date": None,
                "days_until_earnings": None,
                "earnings_confirmed": False,
            }
        
        # Get the next earnings date
        today = datetime.now().date()
        next_earnings = None
        
        for ed in earnings_dates:
            if ed is not None:
                if isinstance(ed, datetime):
                    ed_date = ed.date()
                elif isinstance(ed, date):
                    ed_date = ed
                elif isinstance(ed, str):
                    try:
                        ed_date = datetime.strptime(ed, "%Y-%m-%d").date()
                    except:
                        continue
                else:
                    continue
                
                if ed_date >= today:
                    next_earnings = ed_date
                    break
        
        if next_earnings:
            days_until = (next_earnings - today).days
            return {
                "next_earnings_date": next_earnings.strftime("%b %d, %Y"),
                "days_until_earnings": days_until,
                "earnings_confirmed": True,
            }
        
        return {
            "next_earnings_date": None,
            "days_until_earnings": None,
            "earnings_confirmed": False,
        }
    except Exception as e:
        return {
            "next_earnings_date": None,
            "days_until_earnings": None,
            "earnings_confirmed": False,
            "error": str(e)
        }


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_dividend_info(ticker: str) -> Dict[str, Any]:
    """
    Get dividend information including ex-dividend date.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with ex_dividend_date, dividend_amount, dividend_yield
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        ex_div_date = info.get("exDividendDate")
        div_rate = info.get("dividendRate", 0)
        div_yield = info.get("dividendYield", 0)
        
        # Convert timestamp to date string
        if ex_div_date:
            if isinstance(ex_div_date, (int, float)):
                ex_div_date = datetime.fromtimestamp(ex_div_date).strftime("%b %d, %Y")
            elif isinstance(ex_div_date, datetime):
                ex_div_date = ex_div_date.strftime("%b %d, %Y")
        
        return {
            "ex_dividend_date": ex_div_date,
            "dividend_amount": round(div_rate, 2) if div_rate else None,
            "dividend_yield": round(div_yield * 100, 2) if div_yield else None,
        }
    except Exception as e:
        return {
            "ex_dividend_date": None,
            "dividend_amount": None,
            "dividend_yield": None,
            "error": str(e)
        }


def calculate_liquidity_score(options_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate liquidity metrics for options.
    
    Parameters:
        options_df: DataFrame with options chain data
    
    Returns:
        Dictionary with avg_bid_ask_spread, liquidity_score, volume_oi_ratio
    """
    try:
        if options_df.empty:
            return {
                "avg_bid_ask_spread_pct": None,
                "liquidity_score": None,
                "avg_volume_oi_ratio": None,
            }
        
        # Calculate bid-ask spread percentage
        mid_price = (options_df["bid"] + options_df["ask"]) / 2
        spread = options_df["ask"] - options_df["bid"]
        spread_pct = (spread / mid_price * 100).replace([np.inf, -np.inf], np.nan)
        avg_spread_pct = spread_pct.mean()
        
        # Calculate volume/OI ratio
        vol_oi = (options_df["volume"] / options_df["openInterest"]).replace([np.inf, -np.inf], np.nan)
        avg_vol_oi = vol_oi.mean()
        
        # Liquidity score (0-100, higher is better)
        # Based on spread and volume
        spread_score = max(0, 100 - avg_spread_pct * 10) if not np.isnan(avg_spread_pct) else 50
        volume_score = min(100, options_df["volume"].mean() / 10) if "volume" in options_df else 50
        oi_score = min(100, options_df["openInterest"].mean() / 100) if "openInterest" in options_df else 50
        
        liquidity_score = (spread_score * 0.4 + volume_score * 0.3 + oi_score * 0.3)
        
        return {
            "avg_bid_ask_spread_pct": round(avg_spread_pct, 2) if not np.isnan(avg_spread_pct) else None,
            "liquidity_score": round(liquidity_score, 1),
            "avg_volume_oi_ratio": round(avg_vol_oi, 2) if not np.isnan(avg_vol_oi) else None,
        }
    except Exception as e:
        return {
            "avg_bid_ask_spread_pct": None,
            "liquidity_score": None,
            "avg_volume_oi_ratio": None,
            "error": str(e)
        }


@st.cache_data(ttl=300, show_spinner=False)
def calculate_put_call_ratio(ticker: str) -> Dict[str, Any]:
    """
    Calculate put/call ratio for sentiment analysis.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with put_call_ratio, sentiment
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Get all expiration dates
        expirations = stock.options
        if not expirations:
            return {"put_call_ratio": None, "sentiment": None}
        
        total_put_volume = 0
        total_call_volume = 0
        total_put_oi = 0
        total_call_oi = 0
        
        # Sample first 3 expirations for efficiency
        for exp in expirations[:3]:
            try:
                chain = stock.option_chain(exp)
                total_put_volume += chain.puts["volume"].sum()
                total_call_volume += chain.calls["volume"].sum()
                total_put_oi += chain.puts["openInterest"].sum()
                total_call_oi += chain.calls["openInterest"].sum()
            except:
                continue
        
        # Calculate ratios
        volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else None
        oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None
        
        # Determine sentiment
        if volume_ratio is not None:
            if volume_ratio > 1.2:
                sentiment = "Bearish"
            elif volume_ratio < 0.8:
                sentiment = "Bullish"
            else:
                sentiment = "Neutral"
        else:
            sentiment = None
        
        return {
            "put_call_volume_ratio": round(volume_ratio, 2) if volume_ratio else None,
            "put_call_oi_ratio": round(oi_ratio, 2) if oi_ratio else None,
            "sentiment": sentiment,
        }
    except Exception as e:
        return {
            "put_call_volume_ratio": None,
            "put_call_oi_ratio": None,
            "sentiment": None,
            "error": str(e)
        }


@st.cache_data(ttl=300, show_spinner=False)
def calculate_rsi(ticker: str, period: int = 14) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
        ticker: Stock ticker symbol
        period: RSI period (default 14)
    
    Returns:
        Dictionary with rsi, signal (overbought/oversold/neutral)
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="3mo")
        
        if len(hist) < period + 1:
            return {"rsi": None, "signal": None}
        
        # Calculate price changes
        delta = hist["Close"].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Determine signal
        if current_rsi >= 70:
            signal = "Overbought"
        elif current_rsi <= 30:
            signal = "Oversold"
        else:
            signal = "Neutral"
        
        return {
            "rsi": round(current_rsi, 1),
            "signal": signal,
        }
    except Exception as e:
        return {"rsi": None, "signal": None, "error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def calculate_support_resistance(ticker: str) -> Dict[str, Any]:
    """
    Calculate support and resistance levels.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with support levels and resistance levels
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="6mo")
        
        if len(hist) < 20:
            return {"support_levels": [], "resistance_levels": []}
        
        current_price = hist["Close"].iloc[-1]
        
        # Find local minima (support) and maxima (resistance)
        window = 10
        
        # Rolling min/max
        rolling_min = hist["Low"].rolling(window=window, center=True).min()
        rolling_max = hist["High"].rolling(window=window, center=True).max()
        
        # Find pivot points where price equals rolling min/max
        support_points = hist["Low"][hist["Low"] == rolling_min].dropna()
        resistance_points = hist["High"][hist["High"] == rolling_max].dropna()
        
        # Get unique levels (cluster nearby values)
        def cluster_levels(levels, threshold=0.02):
            if len(levels) == 0:
                return []
            sorted_levels = sorted(levels.values)
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            clusters.append(np.mean(current_cluster))
            return clusters
        
        support_levels = cluster_levels(support_points)
        resistance_levels = cluster_levels(resistance_points)
        
        # Filter to levels near current price (within 15%)
        support_levels = [s for s in support_levels if s < current_price and s > current_price * 0.85]
        resistance_levels = [r for r in resistance_levels if r > current_price and r < current_price * 1.15]
        
        # Sort and take top 3
        support_levels = sorted(support_levels, reverse=True)[:3]
        resistance_levels = sorted(resistance_levels)[:3]
        
        return {
            "support_levels": [round(s, 2) for s in support_levels],
            "resistance_levels": [round(r, 2) for r in resistance_levels],
            "current_price": round(current_price, 2),
        }
    except Exception as e:
        return {"support_levels": [], "resistance_levels": [], "error": str(e)}


def calculate_expected_move(current_price: float, iv: float, days_to_expiry: int) -> Dict[str, Any]:
    """
    Calculate expected move based on implied volatility.
    
    Parameters:
        current_price: Current stock price
        iv: Implied volatility (as decimal)
        days_to_expiry: Days until expiration
    
    Returns:
        Dictionary with expected_move, price_range
    """
    try:
        # Expected move = Stock Price × IV × √(DTE/365)
        time_factor = np.sqrt(days_to_expiry / DAYS_PER_YEAR)
        expected_move = current_price * iv * time_factor
        
        # 1 standard deviation range
        lower_1sd = current_price - expected_move
        upper_1sd = current_price + expected_move
        
        # 2 standard deviation range (covers ~95% of outcomes)
        lower_2sd = current_price - (expected_move * 2)
        upper_2sd = current_price + (expected_move * 2)
        
        return {
            "expected_move": round(expected_move, 2),
            "expected_move_pct": round(expected_move / current_price * 100, 2),
            "range_1sd": (round(lower_1sd, 2), round(upper_1sd, 2)),
            "range_2sd": (round(lower_2sd, 2), round(upper_2sd, 2)),
        }
    except Exception as e:
        return {"expected_move": None, "error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_beta(ticker: str) -> Dict[str, Any]:
    """
    Get stock beta (volatility relative to market).
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with beta value and interpretation
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        beta = info.get("beta")
        
        if beta is None:
            # Calculate beta manually
            stock_hist = stock.history(period="1y")["Close"]
            spy = yf.Ticker("SPY").history(period="1y")["Close"]
            
            # Align dates
            combined = pd.DataFrame({"stock": stock_hist, "spy": spy}).dropna()
            
            if len(combined) < 20:
                return {"beta": None, "interpretation": None}
            
            stock_returns = combined["stock"].pct_change().dropna()
            spy_returns = combined["spy"].pct_change().dropna()
            
            covariance = stock_returns.cov(spy_returns)
            variance = spy_returns.var()
            
            beta = covariance / variance if variance > 0 else 1.0
        
        # Interpretation
        if beta > 1.5:
            interpretation = "High Volatility"
        elif beta > 1.0:
            interpretation = "Above Market"
        elif beta > 0.5:
            interpretation = "Below Market"
        else:
            interpretation = "Low Volatility"
        
        return {
            "beta": round(beta, 2),
            "interpretation": interpretation,
        }
    except Exception as e:
        return {"beta": None, "interpretation": None, "error": str(e)}


def calculate_prob_50_profit(spread) -> float:
    """
    Calculate probability of reaching 50% of max profit.
    
    For credit spreads, this is typically higher than probability of max profit.
    
    Parameters:
        spread: VerticalSpread object
    
    Returns:
        Probability as percentage
    """
    try:
        # For credit spreads, 50% profit is achieved when the spread value
        # decreases to half the credit received
        # This happens at a price further from the short strike
        
        # Approximate: probability of 50% profit ≈ win_probability + (1 - win_probability) * 0.3
        base_prob = spread.probability_of_profit
        prob_50 = base_prob + (1 - base_prob) * 0.25
        
        return round(min(prob_50 * 100, 99), 1)
    except:
        return None


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_historical_earnings_moves(ticker: str, num_quarters: int = 4) -> Dict[str, Any]:
    """
    Get historical stock moves around earnings.
    
    Parameters:
        ticker: Stock ticker symbol
        num_quarters: Number of past earnings to analyze
    
    Returns:
        Dictionary with average move, max move, moves list
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Get earnings history
        earnings = stock.earnings_dates
        
        if earnings is None or earnings.empty:
            return {"avg_move": None, "moves": []}
        
        # Get historical prices
        hist = stock.history(period="2y")
        
        if hist.empty:
            return {"avg_move": None, "moves": []}
        
        moves = []
        
        # Get past earnings dates
        past_earnings = earnings[earnings.index < datetime.now()].head(num_quarters)
        
        for earn_date in past_earnings.index:
            try:
                earn_date = earn_date.date() if hasattr(earn_date, 'date') else earn_date
                
                # Find the closest trading day
                close_prices = hist["Close"]
                
                # Get price before and after earnings
                before_idx = close_prices.index.get_indexer([earn_date], method="ffill")[0]
                after_idx = min(before_idx + 1, len(close_prices) - 1)
                
                if before_idx >= 0 and after_idx < len(close_prices):
                    price_before = close_prices.iloc[before_idx]
                    price_after = close_prices.iloc[after_idx]
                    
                    move_pct = abs((price_after - price_before) / price_before * 100)
                    moves.append(round(move_pct, 2))
            except:
                continue
        
        if moves:
            return {
                "avg_move": round(np.mean(moves), 2),
                "max_move": round(max(moves), 2),
                "min_move": round(min(moves), 2),
                "moves": moves,
            }
        
        return {"avg_move": None, "moves": []}
    except Exception as e:
        return {"avg_move": None, "moves": [], "error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def get_sector_performance(ticker: str) -> Dict[str, Any]:
    """
    Get sector performance relative to market.
    
    Parameters:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with sector name, sector performance, relative strength
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        
        # Sector ETF mapping
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financial Services": "XLF",
            "Financials": "XLF",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Basic Materials": "XLB",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Communication Services": "XLC",
        }
        
        sector_etf = sector_etfs.get(sector, "SPY")
        
        # Get performance
        spy = yf.Ticker("SPY")
        sector_ticker = yf.Ticker(sector_etf)
        
        spy_hist = spy.history(period="1mo")
        sector_hist = sector_ticker.history(period="1mo")
        stock_hist = stock.history(period="1mo")
        
        if len(spy_hist) > 1 and len(sector_hist) > 1 and len(stock_hist) > 1:
            spy_return = (spy_hist["Close"].iloc[-1] / spy_hist["Close"].iloc[0] - 1) * 100
            sector_return = (sector_hist["Close"].iloc[-1] / sector_hist["Close"].iloc[0] - 1) * 100
            stock_return = (stock_hist["Close"].iloc[-1] / stock_hist["Close"].iloc[0] - 1) * 100
            
            # Relative strength
            rs_vs_market = stock_return - spy_return
            rs_vs_sector = stock_return - sector_return
            
            return {
                "sector": sector,
                "industry": industry,
                "sector_etf": sector_etf,
                "stock_1m_return": round(stock_return, 2),
                "sector_1m_return": round(sector_return, 2),
                "market_1m_return": round(spy_return, 2),
                "rs_vs_market": round(rs_vs_market, 2),
                "rs_vs_sector": round(rs_vs_sector, 2),
            }
        
        return {"sector": sector, "industry": industry}
    except Exception as e:
        return {"sector": "Unknown", "industry": "Unknown", "error": str(e)}


def get_trade_management_suggestions(spread, current_price: float) -> Dict[str, Any]:
    """
    Generate trade management suggestions.
    
    Parameters:
        spread: VerticalSpread object
        current_price: Current stock price
    
    Returns:
        Dictionary with profit targets, stop loss levels, rolling suggestions
    """
    try:
        max_profit = spread.max_profit
        max_loss = abs(spread.max_loss)
        breakeven = spread.breakeven
        
        # Profit targets
        profit_targets = {
            "50_pct": round(max_profit * 0.5, 2),
            "75_pct": round(max_profit * 0.75, 2),
            "90_pct": round(max_profit * 0.90, 2),
        }
        
        # Stop loss suggestions (based on multiple of credit received)
        stop_loss_levels = {
            "1x_credit": round(max_profit, 2),  # Close if loss = credit received
            "2x_credit": round(max_profit * 2, 2),
            "50_pct_max_loss": round(max_loss * 0.5, 2),
        }
        
        # Price-based alerts
        if spread.option_type == "put":
            # For put credit spreads
            warning_price = breakeven + (current_price - breakeven) * 0.5
            danger_price = breakeven
        else:
            # For call credit spreads
            warning_price = breakeven - (breakeven - current_price) * 0.5
            danger_price = breakeven
        
        # Rolling suggestion
        days_to_exp = spread.days_to_expiry if hasattr(spread, 'days_to_expiry') else 30
        if days_to_exp <= 21:
            roll_suggestion = "Consider rolling if at 50% profit or if tested"
        elif days_to_exp <= 7:
            roll_suggestion = "Close or roll soon - gamma risk increases"
        else:
            roll_suggestion = "Monitor position"
        
        return {
            "profit_targets": profit_targets,
            "stop_loss_levels": stop_loss_levels,
            "warning_price": round(warning_price, 2),
            "danger_price": round(danger_price, 2),
            "roll_suggestion": roll_suggestion,
            "recommended_action": "Take profits at 50% to maximize win rate" if days_to_exp > 7 else "Consider closing position",
        }
    except Exception as e:
        return {"error": str(e)}

