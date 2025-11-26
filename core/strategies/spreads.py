"""Vertical spread strategy implementations."""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Any, List, Optional
import pandas as pd

from ..black_scholes import delta, gamma, theta, vega, rho
from ..probability import probability_of_profit
from .base import OptionsStrategy, StrategyLeg


@dataclass
class VerticalSpread(OptionsStrategy):
    """
    Base class for vertical spreads (same expiration, different strikes).
    """
    
    short_leg: StrategyLeg
    long_leg: StrategyLeg
    underlying_price: float
    expiration_date: date
    time_to_expiry: float  # in years
    risk_free_rate: float
    implied_volatility: float
    strategy_type: str  # 'credit' or 'debit'
    option_type: str  # 'call' or 'put'
    
    @property
    def name(self) -> str:
        option_name = "Call" if self.option_type == "call" else "Put"
        spread_name = "Credit" if self.strategy_type == "credit" else "Debit"
        return f"{option_name} {spread_name} Spread"
    
    @property
    def short_strike(self) -> float:
        return self.short_leg.strike
    
    @property
    def long_strike(self) -> float:
        return self.long_leg.strike
    
    @property
    def spread_width(self) -> float:
        return abs(self.short_strike - self.long_strike)
    
    @property
    def net_premium(self) -> float:
        """Net premium (positive for credit, negative for debit)."""
        return self.short_leg.premium - self.long_leg.premium
    
    @property
    def is_credit(self) -> bool:
        return self.strategy_type == "credit"
    
    @property
    def max_profit(self) -> float:
        """Maximum profit (per contract, 100 shares)."""
        if self.is_credit:
            # Credit spread: max profit is premium received
            return self.net_premium * 100
        else:
            # Debit spread: max profit is spread width - premium paid
            return (self.spread_width - abs(self.net_premium)) * 100
    
    @property
    def max_loss(self) -> float:
        """Maximum loss (per contract, 100 shares)."""
        if self.is_credit:
            # Credit spread: max loss is spread width - premium received
            return (self.spread_width - self.net_premium) * 100
        else:
            # Debit spread: max loss is premium paid
            return abs(self.net_premium) * 100
    
    @property
    def breakeven(self) -> float:
        """Breakeven price."""
        if self.option_type == "put":
            if self.is_credit:
                # Put credit spread: short strike - net credit
                return self.short_strike - self.net_premium
            else:
                # Put debit spread: long strike - net debit
                return self.long_strike - abs(self.net_premium)
        else:
            if self.is_credit:
                # Call credit spread: short strike + net credit
                return self.short_strike + self.net_premium
            else:
                # Call debit spread: long strike + net debit
                return self.long_strike + abs(self.net_premium)
    
    @property
    def capital_requirement(self) -> float:
        """Capital/margin required (per contract)."""
        if self.is_credit:
            return self.spread_width * 100
        else:
            return abs(self.net_premium) * 100
    
    @property
    def risk_reward_ratio(self) -> float:
        """Risk to reward ratio (lower is better)."""
        if self.max_profit == 0:
            return float("inf")
        return self.max_loss / self.max_profit
    
    @property
    def return_on_capital(self) -> float:
        """Potential return on capital."""
        if self.capital_requirement == 0:
            return 0
        return self.max_profit / self.capital_requirement
    
    @property
    def probability_of_profit(self) -> float:
        """Probability of profit."""
        if self.option_type == "put":
            if self.is_credit:
                # Put credit: profit if stock stays above breakeven
                return probability_of_profit(
                    S=self.underlying_price,
                    K=self.breakeven,
                    T=self.time_to_expiry,
                    r=self.risk_free_rate,
                    sigma=self.implied_volatility,
                    option_type="put",
                    position="short",
                )
            else:
                # Put debit: profit if stock falls below breakeven
                return probability_of_profit(
                    S=self.underlying_price,
                    K=self.breakeven,
                    T=self.time_to_expiry,
                    r=self.risk_free_rate,
                    sigma=self.implied_volatility,
                    option_type="put",
                    position="long",
                )
        else:
            if self.is_credit:
                # Call credit: profit if stock stays below breakeven
                return probability_of_profit(
                    S=self.underlying_price,
                    K=self.breakeven,
                    T=self.time_to_expiry,
                    r=self.risk_free_rate,
                    sigma=self.implied_volatility,
                    option_type="call",
                    position="short",
                )
            else:
                # Call debit: profit if stock rises above breakeven
                return probability_of_profit(
                    S=self.underlying_price,
                    K=self.breakeven,
                    T=self.time_to_expiry,
                    r=self.risk_free_rate,
                    sigma=self.implied_volatility,
                    option_type="call",
                    position="long",
                )
    
    def profit_at_price(self, price: float) -> float:
        """Calculate P/L at a given stock price at expiration (per contract)."""
        if self.option_type == "put":
            # Put values at expiration
            short_value = max(self.short_strike - price, 0)
            long_value = max(self.long_strike - price, 0)
        else:
            # Call values at expiration
            short_value = max(price - self.short_strike, 0)
            long_value = max(price - self.long_strike, 0)
        
        # Short leg P/L: premium received - value at expiration
        short_pnl = (self.short_leg.premium - short_value) * 100
        # Long leg P/L: value at expiration - premium paid
        long_pnl = (long_value - self.long_leg.premium) * 100
        
        return short_pnl + long_pnl
    
    def get_greeks(self) -> Dict[str, float]:
        """Calculate combined Greeks for the spread."""
        params = {
            "S": self.underlying_price,
            "T": self.time_to_expiry,
            "r": self.risk_free_rate,
            "sigma": self.implied_volatility,
        }
        
        # Short leg Greeks (negative because we're short)
        short_delta = -delta(K=self.short_strike, option_type=self.option_type, **params)
        short_gamma = -gamma(K=self.short_strike, **params)
        short_theta = -theta(K=self.short_strike, option_type=self.option_type, **params)
        short_vega = -vega(K=self.short_strike, **params)
        short_rho = -rho(K=self.short_strike, option_type=self.option_type, **params)
        
        # Long leg Greeks
        long_delta = delta(K=self.long_strike, option_type=self.option_type, **params)
        long_gamma = gamma(K=self.long_strike, **params)
        long_theta = theta(K=self.long_strike, option_type=self.option_type, **params)
        long_vega = vega(K=self.long_strike, **params)
        long_rho = rho(K=self.long_strike, option_type=self.option_type, **params)
        
        return {
            "delta": (short_delta + long_delta) * 100,
            "gamma": (short_gamma + long_gamma) * 100,
            "theta": (short_theta + long_theta) * 100,
            "vega": (short_vega + long_vega) * 100,
            "rho": (short_rho + long_rho) * 100,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for display."""
        greeks = self.get_greeks()
        return {
            "Strategy": self.name,
            "Short Strike": self.short_strike,
            "Long Strike": self.long_strike,
            "Spread Width": self.spread_width,
            "Net Premium": f"${abs(self.net_premium):.2f}",
            "Max Profit": f"${self.max_profit:.2f}",
            "Max Loss": f"${self.max_loss:.2f}",
            "Breakeven": f"${self.breakeven:.2f}",
            "Win Probability": f"{self.probability_of_profit:.1%}",
            "Risk/Reward": f"{self.risk_reward_ratio:.2f}",
            "Return on Capital": f"{self.return_on_capital:.1%}",
            "Capital Required": f"${self.capital_requirement:.2f}",
            "Delta": f"{greeks['delta']:.2f}",
            "Theta": f"${greeks['theta']:.2f}/day",
            "Vega": f"${greeks['vega']:.2f}",
        }


def find_vertical_spreads(
    options_df: pd.DataFrame,
    underlying_price: float,
    expiration_date: date,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str,  # 'call' or 'put'
    strategy_type: str,  # 'credit' or 'debit'
    min_probability: float = 0.50,
    spread_widths: Optional[List[float]] = None,
    max_results: int = 10,
) -> List[VerticalSpread]:
    """
    Find suitable vertical spread opportunities.
    
    Parameters:
        options_df: DataFrame with options (columns: strike, lastPrice/bid/ask, impliedVolatility)
        underlying_price: Current stock price
        expiration_date: Option expiration date
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate
        option_type: 'call' or 'put'
        strategy_type: 'credit' or 'debit'
        min_probability: Minimum probability of profit
        spread_widths: List of spread widths to consider
        max_results: Maximum number of spreads to return
    
    Returns:
        List of VerticalSpread objects sorted by risk/reward ratio
    """
    if spread_widths is None:
        spread_widths = [2.5, 5.0, 10.0]
    
    def get_mid_price(row):
        bid = row.get("bid", 0)
        ask = row.get("ask", 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return row.get("lastPrice", 0)
    
    spreads = []
    strikes = sorted(options_df["strike"].unique())
    
    for i, strike1 in enumerate(strikes):
        row1 = options_df[options_df["strike"] == strike1].iloc[0]
        premium1 = get_mid_price(row1)
        iv1 = row1.get("impliedVolatility", 0.3)
        
        for width in spread_widths:
            if option_type == "put":
                if strategy_type == "credit":
                    # Put Credit Spread: sell higher strike, buy lower strike
                    short_strike = strike1
                    long_strike = strike1 - width
                else:
                    # Put Debit Spread: buy higher strike, sell lower strike
                    long_strike = strike1
                    short_strike = strike1 - width
            else:
                if strategy_type == "credit":
                    # Call Credit Spread: sell lower strike, buy higher strike
                    short_strike = strike1
                    long_strike = strike1 + width
                else:
                    # Call Debit Spread: buy lower strike, sell higher strike
                    long_strike = strike1
                    short_strike = strike1 + width
            
            # Check if the other strike exists
            other_strike = long_strike if strike1 == short_strike else short_strike
            if other_strike not in strikes:
                continue
            
            row2 = options_df[options_df["strike"] == other_strike].iloc[0]
            premium2 = get_mid_price(row2)
            
            # Determine which is short and which is long
            if strike1 == short_strike:
                short_premium = premium1
                long_premium = premium2
                short_iv = iv1
            else:
                short_premium = premium2
                long_premium = premium1
                short_iv = row2.get("impliedVolatility", 0.3)
            
            # Skip invalid spreads
            if short_premium <= 0 or long_premium <= 0:
                continue
            
            # For credit spreads, we need net credit
            if strategy_type == "credit" and short_premium <= long_premium:
                continue
            
            # For debit spreads, we need net debit
            if strategy_type == "debit" and long_premium <= short_premium:
                continue
            
            # Additional filters based on position relative to current price
            if option_type == "put" and strategy_type == "credit":
                # Put credit: short strike should be below current price
                if short_strike > underlying_price:
                    continue
            elif option_type == "call" and strategy_type == "credit":
                # Call credit: short strike should be above current price
                if short_strike < underlying_price:
                    continue
            
            spread = VerticalSpread(
                short_leg=StrategyLeg(
                    option_type=option_type,
                    strike=short_strike,
                    premium=short_premium,
                    position="short",
                ),
                long_leg=StrategyLeg(
                    option_type=option_type,
                    strike=long_strike,
                    premium=long_premium,
                    position="long",
                ),
                underlying_price=underlying_price,
                expiration_date=expiration_date,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                implied_volatility=short_iv if short_iv > 0 else 0.3,
                strategy_type=strategy_type,
                option_type=option_type,
            )
            
            # Filter by probability
            if spread.probability_of_profit >= min_probability:
                spreads.append(spread)
    
    # Sort by risk/reward ratio (lower is better)
    spreads.sort(key=lambda s: s.risk_reward_ratio)
    
    return spreads[:max_results]

