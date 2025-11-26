"""Bull Put Spread strategy implementation."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Any, List, Optional
import pandas as pd

from ..black_scholes import delta, gamma, theta, vega, rho
from ..probability import probability_of_profit
from .base import OptionsStrategy, StrategyLeg


@dataclass
class BullPutSpread(OptionsStrategy):
    """
    Bull Put Spread (Credit Put Spread) Strategy.
    
    - Sell a put at higher strike (receive premium)
    - Buy a put at lower strike (pay premium)
    - Net credit received
    - Bullish to neutral outlook
    """
    
    short_put: StrategyLeg
    long_put: StrategyLeg
    underlying_price: float
    expiration_date: date
    time_to_expiry: float  # in years
    risk_free_rate: float
    implied_volatility: float
    
    @property
    def name(self) -> str:
        return "Bull Put Spread"
    
    @property
    def short_strike(self) -> float:
        return self.short_put.strike
    
    @property
    def long_strike(self) -> float:
        return self.long_put.strike
    
    @property
    def spread_width(self) -> float:
        return self.short_strike - self.long_strike
    
    @property
    def net_credit(self) -> float:
        """Net premium received (per share)."""
        return self.short_put.premium - self.long_put.premium
    
    @property
    def max_profit(self) -> float:
        """Maximum profit = net credit received (per contract, 100 shares)."""
        return self.net_credit * 100
    
    @property
    def max_loss(self) -> float:
        """Maximum loss = spread width - net credit (per contract, 100 shares)."""
        return (self.spread_width - self.net_credit) * 100
    
    @property
    def breakeven(self) -> float:
        """Breakeven price = short strike - net credit."""
        return self.short_strike - self.net_credit
    
    @property
    def capital_requirement(self) -> float:
        """
        Margin requirement = spread width * 100 (per contract).
        This is the maximum loss, which is held as collateral.
        """
        return self.spread_width * 100
    
    @property
    def risk_reward_ratio(self) -> float:
        """Risk to reward ratio (lower is better)."""
        if self.max_profit == 0:
            return float("inf")
        return self.max_loss / self.max_profit
    
    @property
    def return_on_capital(self) -> float:
        """Potential return on capital (max profit / capital required)."""
        if self.capital_requirement == 0:
            return 0
        return self.max_profit / self.capital_requirement
    
    @property
    def probability_of_profit(self) -> float:
        """Probability that the spread expires profitable."""
        return probability_of_profit(
            S=self.underlying_price,
            K=self.short_strike,
            T=self.time_to_expiry,
            r=self.risk_free_rate,
            sigma=self.implied_volatility,
            option_type="put",
            position="short",
        )
    
    def profit_at_price(self, price: float) -> float:
        """Calculate P/L at a given stock price at expiration (per contract)."""
        # Short put payoff
        short_put_value = max(self.short_strike - price, 0)
        short_put_pnl = (self.short_put.premium - short_put_value) * 100
        
        # Long put payoff
        long_put_value = max(self.long_strike - price, 0)
        long_put_pnl = (long_put_value - self.long_put.premium) * 100
        
        return short_put_pnl + long_put_pnl
    
    def get_greeks(self) -> Dict[str, float]:
        """Calculate combined Greeks for the spread."""
        params = {
            "S": self.underlying_price,
            "T": self.time_to_expiry,
            "r": self.risk_free_rate,
            "sigma": self.implied_volatility,
        }
        
        # Short put Greeks (negative because we're short)
        short_delta = -delta(K=self.short_strike, option_type="put", **params)
        short_gamma = -gamma(K=self.short_strike, **params)
        short_theta = -theta(K=self.short_strike, option_type="put", **params)
        short_vega = -vega(K=self.short_strike, **params)
        short_rho = -rho(K=self.short_strike, option_type="put", **params)
        
        # Long put Greeks
        long_delta = delta(K=self.long_strike, option_type="put", **params)
        long_gamma = gamma(K=self.long_strike, **params)
        long_theta = theta(K=self.long_strike, option_type="put", **params)
        long_vega = vega(K=self.long_strike, **params)
        long_rho = rho(K=self.long_strike, option_type="put", **params)
        
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
            "Short Strike": self.short_strike,
            "Long Strike": self.long_strike,
            "Spread Width": self.spread_width,
            "Short Premium": f"${self.short_put.premium:.2f}",
            "Long Premium": f"${self.long_put.premium:.2f}",
            "Net Credit": f"${self.net_credit:.2f}",
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


def find_bull_put_spreads(
    puts_df: pd.DataFrame,
    underlying_price: float,
    expiration_date: date,
    time_to_expiry: float,
    risk_free_rate: float,
    min_probability: float = 0.70,
    spread_widths: Optional[List[float]] = None,
    max_results: int = 10,
) -> List[BullPutSpread]:
    """
    Find suitable bull put spread opportunities.
    
    Parameters:
        puts_df: DataFrame with put options (columns: strike, lastPrice/bid/ask, impliedVolatility)
        underlying_price: Current stock price
        expiration_date: Option expiration date
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate
        min_probability: Minimum probability of profit
        spread_widths: List of spread widths to consider (e.g., [2.5, 5, 10])
        max_results: Maximum number of spreads to return
    
    Returns:
        List of BullPutSpread objects sorted by risk/reward ratio
    """
    if spread_widths is None:
        spread_widths = [2.5, 5.0, 10.0]
    
    # Use mid price (average of bid/ask) for more accurate pricing like Robinhood
    def get_mid_price(row):
        bid = row.get("bid", 0)
        ask = row.get("ask", 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return row.get("lastPrice", 0)
    
    def get_sell_price(row):
        return get_mid_price(row)
    
    def get_buy_price(row):
        return get_mid_price(row)
    
    spreads = []
    strikes = sorted(puts_df["strike"].unique(), reverse=True)
    
    for short_strike in strikes:
        # Skip if short strike is above current price (too aggressive)
        if short_strike > underlying_price:
            continue
        
        short_row = puts_df[puts_df["strike"] == short_strike].iloc[0]
        short_iv = short_row.get("impliedVolatility", 0.3)
        
        # Check probability for short strike
        prob = probability_of_profit(
            S=underlying_price,
            K=short_strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=short_iv,
            option_type="put",
            position="short",
        )
        
        if prob < min_probability:
            continue
        
        for width in spread_widths:
            long_strike = short_strike - width
            
            # Check if long strike exists
            if long_strike not in strikes:
                continue
            
            long_row = puts_df[puts_df["strike"] == long_strike].iloc[0]
            
            short_premium = get_sell_price(short_row)
            long_premium = get_buy_price(long_row)
            
            # Skip if no credit or invalid prices
            if short_premium <= long_premium or short_premium <= 0:
                continue
            
            # Get IV (use short strike IV as representative)
            iv = short_iv if short_iv > 0 else 0.3
            
            spread = BullPutSpread(
                short_put=StrategyLeg(
                    option_type="put",
                    strike=short_strike,
                    premium=short_premium,
                    position="short",
                ),
                long_put=StrategyLeg(
                    option_type="put",
                    strike=long_strike,
                    premium=long_premium,
                    position="long",
                ),
                underlying_price=underlying_price,
                expiration_date=expiration_date,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                implied_volatility=iv,
            )
            
            # Only include spreads with positive credit
            if spread.net_credit > 0:
                spreads.append(spread)
    
    # Sort by risk/reward ratio (lower is better)
    spreads.sort(key=lambda s: s.risk_reward_ratio)
    
    return spreads[:max_results]

