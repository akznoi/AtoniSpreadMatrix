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
    max_results: int = 50,
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
        spread_widths: List of spread widths to consider (None = all available)
        max_results: Maximum number of spreads to return
    
    Returns:
        List of VerticalSpread objects sorted by risk/reward ratio
    """
    def get_mid_price(row):
        bid = row.get("bid", 0)
        ask = row.get("ask", 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return row.get("lastPrice", 0)
    
    spreads = []
    strikes = sorted(options_df["strike"].unique())
    
    # Generate all possible strike pairs
    for i, strike1 in enumerate(strikes):
        row1 = options_df[options_df["strike"] == strike1].iloc[0]
        premium1 = get_mid_price(row1)
        iv1 = row1.get("impliedVolatility", 0.3)
        
        for j, strike2 in enumerate(strikes):
            if i == j:
                continue
            
            # Determine short and long based on strategy
            if option_type == "put":
                if strategy_type == "credit":
                    # Put Credit Spread: sell higher strike, buy lower strike
                    if strike1 <= strike2:
                        continue  # Need strike1 > strike2
                    short_strike = strike1
                    long_strike = strike2
                else:
                    # Put Debit Spread: buy higher strike, sell lower strike
                    if strike1 <= strike2:
                        continue  # Need strike1 > strike2
                    long_strike = strike1
                    short_strike = strike2
            else:
                if strategy_type == "credit":
                    # Call Credit Spread: sell lower strike, buy higher strike
                    if strike1 >= strike2:
                        continue  # Need strike1 < strike2
                    short_strike = strike1
                    long_strike = strike2
                else:
                    # Call Debit Spread: buy lower strike, sell higher strike
                    if strike1 >= strike2:
                        continue  # Need strike1 < strike2
                    long_strike = strike1
                    short_strike = strike2
            
            row2 = options_df[options_df["strike"] == strike2].iloc[0]
            premium2 = get_mid_price(row2)
            iv2 = row2.get("impliedVolatility", 0.3)
            
            # Determine premiums based on which strike is short/long
            if short_strike == strike1:
                short_premium = premium1
                long_premium = premium2
                short_iv = iv1
            else:
                short_premium = premium2
                long_premium = premium1
                short_iv = iv2
            
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


@dataclass
class IronCondor:
    """
    Iron Condor: Put Credit Spread + Call Credit Spread.
    Profits when stock stays within a range.
    """
    
    put_spread: VerticalSpread  # Put credit spread (lower)
    call_spread: VerticalSpread  # Call credit spread (upper)
    underlying_price: float
    expiration_date: date
    
    @property
    def name(self) -> str:
        return "Iron Condor"
    
    @property
    def strategy_type(self) -> str:
        return "credit"
    
    @property
    def option_type(self) -> str:
        return "both"
    
    @property
    def short_put_strike(self) -> float:
        return self.put_spread.short_strike
    
    @property
    def long_put_strike(self) -> float:
        return self.put_spread.long_strike
    
    @property
    def short_call_strike(self) -> float:
        return self.call_spread.short_strike
    
    @property
    def long_call_strike(self) -> float:
        return self.call_spread.long_strike
    
    @property
    def put_spread_width(self) -> float:
        return self.put_spread.spread_width
    
    @property
    def call_spread_width(self) -> float:
        return self.call_spread.spread_width
    
    @property
    def net_premium(self) -> float:
        """Total credit received."""
        return self.put_spread.net_premium + self.call_spread.net_premium
    
    @property
    def max_profit(self) -> float:
        """Maximum profit = total credit received × 100."""
        return self.net_premium * 100
    
    @property
    def max_loss(self) -> float:
        """Maximum loss = wider spread width - credit received."""
        wider_spread = max(self.put_spread_width, self.call_spread_width)
        return (wider_spread - self.net_premium) * 100
    
    @property
    def breakeven_lower(self) -> float:
        """Lower breakeven = short put strike - net credit."""
        return self.short_put_strike - self.net_premium
    
    @property
    def breakeven_upper(self) -> float:
        """Upper breakeven = short call strike + net credit."""
        return self.short_call_strike + self.net_premium
    
    @property
    def breakeven(self) -> float:
        """Return lower breakeven for display compatibility."""
        return self.breakeven_lower
    
    @property
    def profit_range(self) -> float:
        """Price range where trade is profitable."""
        return self.breakeven_upper - self.breakeven_lower
    
    @property
    def probability_of_profit(self) -> float:
        """Approximate probability both spreads expire worthless."""
        # Simplified: use the lower of the two probabilities
        # In reality, these are correlated so combined prob is higher
        return min(self.put_spread.probability_of_profit, 
                   self.call_spread.probability_of_profit)
    
    @property
    def capital_requirement(self) -> float:
        """Margin required = wider spread width × 100."""
        return max(self.put_spread_width, self.call_spread_width) * 100
    
    @property
    def return_on_capital(self) -> float:
        """Potential return on capital."""
        if self.capital_requirement == 0:
            return 0
        return self.max_profit / self.capital_requirement
    
    @property
    def risk_reward_ratio(self) -> float:
        """Risk to reward ratio."""
        if self.max_profit == 0:
            return float("inf")
        return self.max_loss / self.max_profit
    
    @property
    def short_strike(self) -> float:
        """For compatibility - return short put strike."""
        return self.short_put_strike
    
    @property
    def long_strike(self) -> float:
        """For compatibility - return long put strike."""
        return self.long_put_strike
    
    @property
    def spread_width(self) -> float:
        """For compatibility - return wider spread."""
        return max(self.put_spread_width, self.call_spread_width)
    
    @property
    def implied_volatility(self) -> float:
        """Average IV from both spreads."""
        return (self.put_spread.implied_volatility + self.call_spread.implied_volatility) / 2
    
    @property
    def time_to_expiry(self) -> float:
        return self.put_spread.time_to_expiry
    
    def get_greeks(self) -> Dict[str, float]:
        """Combined Greeks from both spreads."""
        put_greeks = self.put_spread.get_greeks()
        call_greeks = self.call_spread.get_greeks()
        return {
            "delta": put_greeks["delta"] + call_greeks["delta"],
            "gamma": put_greeks["gamma"] + call_greeks["gamma"],
            "theta": put_greeks["theta"] + call_greeks["theta"],
            "vega": put_greeks["vega"] + call_greeks["vega"],
            "rho": put_greeks["rho"] + call_greeks["rho"],
        }
    
    def to_dict(self) -> Dict[str, Any]:
        greeks = self.get_greeks()
        return {
            "Strategy": self.name,
            "Short Put": self.short_put_strike,
            "Long Put": self.long_put_strike,
            "Short Call": self.short_call_strike,
            "Long Call": self.long_call_strike,
            "Net Credit": f"${self.net_premium:.2f}",
            "Max Profit": f"${self.max_profit:.2f}",
            "Max Loss": f"${self.max_loss:.2f}",
            "Lower BE": f"${self.breakeven_lower:.2f}",
            "Upper BE": f"${self.breakeven_upper:.2f}",
            "Win Probability": f"{self.probability_of_profit:.1%}",
            "Risk/Reward": f"{self.risk_reward_ratio:.2f}",
            "Return on Capital": f"{self.return_on_capital:.1%}",
            "Delta": f"{greeks['delta']:.2f}",
            "Theta": f"${greeks['theta']:.2f}/day",
        }


@dataclass
class IronButterfly:
    """
    Iron Butterfly: Sell ATM put + Sell ATM call + Buy OTM put + Buy OTM call.
    All short strikes are at the same price (ATM).
    Higher premium but narrower profit zone than Iron Condor.
    """
    
    put_spread: VerticalSpread  # Put credit spread
    call_spread: VerticalSpread  # Call credit spread
    underlying_price: float
    expiration_date: date
    center_strike: float  # The ATM strike where both shorts are placed
    
    @property
    def name(self) -> str:
        return "Iron Butterfly"
    
    @property
    def strategy_type(self) -> str:
        return "credit"
    
    @property
    def option_type(self) -> str:
        return "both"
    
    @property
    def short_strike(self) -> float:
        """Center strike (same for put and call)."""
        return self.center_strike
    
    @property
    def long_put_strike(self) -> float:
        return self.put_spread.long_strike
    
    @property
    def long_call_strike(self) -> float:
        return self.call_spread.long_strike
    
    @property
    def wing_width(self) -> float:
        """Width from center to wings."""
        return max(
            abs(self.center_strike - self.long_put_strike),
            abs(self.long_call_strike - self.center_strike)
        )
    
    @property
    def net_premium(self) -> float:
        """Total credit received."""
        return self.put_spread.net_premium + self.call_spread.net_premium
    
    @property
    def max_profit(self) -> float:
        """Maximum profit = total credit received × 100."""
        return self.net_premium * 100
    
    @property
    def max_loss(self) -> float:
        """Maximum loss = wing width - credit received."""
        return (self.wing_width - self.net_premium) * 100
    
    @property
    def breakeven_lower(self) -> float:
        """Lower breakeven = center strike - net credit."""
        return self.center_strike - self.net_premium
    
    @property
    def breakeven_upper(self) -> float:
        """Upper breakeven = center strike + net credit."""
        return self.center_strike + self.net_premium
    
    @property
    def breakeven(self) -> float:
        """Return lower breakeven for display compatibility."""
        return self.breakeven_lower
    
    @property
    def probability_of_profit(self) -> float:
        """Probability of any profit (stock between breakevens)."""
        # Iron butterflies have lower probability but higher premium
        return min(self.put_spread.probability_of_profit,
                   self.call_spread.probability_of_profit) * 0.9
    
    @property
    def capital_requirement(self) -> float:
        """Margin required = wing width × 100."""
        return self.wing_width * 100
    
    @property
    def return_on_capital(self) -> float:
        if self.capital_requirement == 0:
            return 0
        return self.max_profit / self.capital_requirement
    
    @property
    def risk_reward_ratio(self) -> float:
        if self.max_profit == 0:
            return float("inf")
        return self.max_loss / self.max_profit
    
    @property
    def long_strike(self) -> float:
        """For compatibility - return long put strike."""
        return self.long_put_strike
    
    @property
    def spread_width(self) -> float:
        """For compatibility."""
        return self.wing_width
    
    @property
    def implied_volatility(self) -> float:
        return (self.put_spread.implied_volatility + self.call_spread.implied_volatility) / 2
    
    @property
    def time_to_expiry(self) -> float:
        return self.put_spread.time_to_expiry
    
    def get_greeks(self) -> Dict[str, float]:
        put_greeks = self.put_spread.get_greeks()
        call_greeks = self.call_spread.get_greeks()
        return {
            "delta": put_greeks["delta"] + call_greeks["delta"],
            "gamma": put_greeks["gamma"] + call_greeks["gamma"],
            "theta": put_greeks["theta"] + call_greeks["theta"],
            "vega": put_greeks["vega"] + call_greeks["vega"],
            "rho": put_greeks["rho"] + call_greeks["rho"],
        }
    
    def to_dict(self) -> Dict[str, Any]:
        greeks = self.get_greeks()
        return {
            "Strategy": self.name,
            "Center Strike": self.center_strike,
            "Long Put": self.long_put_strike,
            "Long Call": self.long_call_strike,
            "Net Credit": f"${self.net_premium:.2f}",
            "Max Profit": f"${self.max_profit:.2f}",
            "Max Loss": f"${self.max_loss:.2f}",
            "Lower BE": f"${self.breakeven_lower:.2f}",
            "Upper BE": f"${self.breakeven_upper:.2f}",
            "Win Probability": f"{self.probability_of_profit:.1%}",
            "Risk/Reward": f"{self.risk_reward_ratio:.2f}",
            "Return on Capital": f"{self.return_on_capital:.1%}",
            "Delta": f"{greeks['delta']:.2f}",
            "Theta": f"${greeks['theta']:.2f}/day",
        }


def find_iron_condors(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    underlying_price: float,
    expiration_date: date,
    time_to_expiry: float,
    risk_free_rate: float,
    min_probability: float = 0.50,
    max_results: int = 20,
) -> List[IronCondor]:
    """
    Find Iron Condor opportunities.
    
    An Iron Condor combines:
    - Put Credit Spread (below current price)
    - Call Credit Spread (above current price)
    """
    def get_mid_price(row):
        bid = row.get("bid", 0)
        ask = row.get("ask", 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return row.get("lastPrice", 0)
    
    condors = []
    
    put_strikes = sorted(puts_df["strike"].unique())
    call_strikes = sorted(calls_df["strike"].unique())
    
    # Find put credit spreads (below current price)
    put_spreads = []
    for i, short_strike in enumerate(put_strikes):
        if short_strike >= underlying_price:
            continue  # Skip ITM puts
        
        row_short = puts_df[puts_df["strike"] == short_strike].iloc[0]
        short_premium = get_mid_price(row_short)
        short_iv = row_short.get("impliedVolatility", 0.3)
        
        for long_strike in put_strikes:
            if long_strike >= short_strike:
                continue  # Long must be lower
            
            row_long = puts_df[puts_df["strike"] == long_strike].iloc[0]
            long_premium = get_mid_price(row_long)
            
            if short_premium <= long_premium or short_premium <= 0 or long_premium <= 0:
                continue
            
            spread = VerticalSpread(
                short_leg=StrategyLeg(option_type="put", strike=short_strike, premium=short_premium, position="short"),
                long_leg=StrategyLeg(option_type="put", strike=long_strike, premium=long_premium, position="long"),
                underlying_price=underlying_price,
                expiration_date=expiration_date,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                implied_volatility=short_iv if short_iv > 0 else 0.3,
                strategy_type="credit",
                option_type="put",
            )
            
            if spread.probability_of_profit >= min_probability:
                put_spreads.append(spread)
    
    # Find call credit spreads (above current price)
    call_spreads = []
    for short_strike in call_strikes:
        if short_strike <= underlying_price:
            continue  # Skip ITM calls
        
        row_short = calls_df[calls_df["strike"] == short_strike].iloc[0]
        short_premium = get_mid_price(row_short)
        short_iv = row_short.get("impliedVolatility", 0.3)
        
        for long_strike in call_strikes:
            if long_strike <= short_strike:
                continue  # Long must be higher
            
            row_long = calls_df[calls_df["strike"] == long_strike].iloc[0]
            long_premium = get_mid_price(row_long)
            
            if short_premium <= long_premium or short_premium <= 0 or long_premium <= 0:
                continue
            
            spread = VerticalSpread(
                short_leg=StrategyLeg(option_type="call", strike=short_strike, premium=short_premium, position="short"),
                long_leg=StrategyLeg(option_type="call", strike=long_strike, premium=long_premium, position="long"),
                underlying_price=underlying_price,
                expiration_date=expiration_date,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                implied_volatility=short_iv if short_iv > 0 else 0.3,
                strategy_type="credit",
                option_type="call",
            )
            
            if spread.probability_of_profit >= min_probability:
                call_spreads.append(spread)
    
    # Combine into Iron Condors
    for put_spread in put_spreads[:15]:  # Limit combinations
        for call_spread in call_spreads[:15]:
            condor = IronCondor(
                put_spread=put_spread,
                call_spread=call_spread,
                underlying_price=underlying_price,
                expiration_date=expiration_date,
            )
            
            # Filter by combined probability
            if condor.probability_of_profit >= min_probability:
                condors.append(condor)
    
    # Sort by risk/reward ratio
    condors.sort(key=lambda c: c.risk_reward_ratio)
    
    return condors[:max_results]


def find_iron_butterflies(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    underlying_price: float,
    expiration_date: date,
    time_to_expiry: float,
    risk_free_rate: float,
    min_probability: float = 0.30,  # Lower threshold for butterflies
    max_results: int = 20,
) -> List[IronButterfly]:
    """
    Find Iron Butterfly opportunities.
    
    An Iron Butterfly has:
    - Short ATM put + Short ATM call (same strike)
    - Long OTM put (lower) + Long OTM call (higher)
    """
    def get_mid_price(row):
        bid = row.get("bid", 0)
        ask = row.get("ask", 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return row.get("lastPrice", 0)
    
    butterflies = []
    
    # Find common strikes between puts and calls
    put_strikes = set(puts_df["strike"].unique())
    call_strikes = set(calls_df["strike"].unique())
    common_strikes = sorted(put_strikes & call_strikes)
    
    if not common_strikes:
        return []
    
    # Find ATM strike (closest to current price)
    atm_candidates = sorted(common_strikes, key=lambda s: abs(s - underlying_price))[:5]
    
    for center_strike in atm_candidates:
        # Get short options at center strike
        put_row = puts_df[puts_df["strike"] == center_strike]
        call_row = calls_df[calls_df["strike"] == center_strike]
        
        if put_row.empty or call_row.empty:
            continue
        
        short_put_premium = get_mid_price(put_row.iloc[0])
        short_call_premium = get_mid_price(call_row.iloc[0])
        put_iv = put_row.iloc[0].get("impliedVolatility", 0.3)
        call_iv = call_row.iloc[0].get("impliedVolatility", 0.3)
        
        if short_put_premium <= 0 or short_call_premium <= 0:
            continue
        
        # Find wing strikes
        lower_strikes = [s for s in common_strikes if s < center_strike]
        upper_strikes = [s for s in common_strikes if s > center_strike]
        
        for long_put_strike in lower_strikes[-5:]:  # Closest lower strikes
            long_put_row = puts_df[puts_df["strike"] == long_put_strike]
            if long_put_row.empty:
                continue
            long_put_premium = get_mid_price(long_put_row.iloc[0])
            
            for long_call_strike in upper_strikes[:5]:  # Closest upper strikes
                long_call_row = calls_df[calls_df["strike"] == long_call_strike]
                if long_call_row.empty:
                    continue
                long_call_premium = get_mid_price(long_call_row.iloc[0])
                
                if long_put_premium <= 0 or long_call_premium <= 0:
                    continue
                
                # Create the component spreads
                put_spread = VerticalSpread(
                    short_leg=StrategyLeg(option_type="put", strike=center_strike, premium=short_put_premium, position="short"),
                    long_leg=StrategyLeg(option_type="put", strike=long_put_strike, premium=long_put_premium, position="long"),
                    underlying_price=underlying_price,
                    expiration_date=expiration_date,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    implied_volatility=put_iv if put_iv > 0 else 0.3,
                    strategy_type="credit",
                    option_type="put",
                )
                
                call_spread = VerticalSpread(
                    short_leg=StrategyLeg(option_type="call", strike=center_strike, premium=short_call_premium, position="short"),
                    long_leg=StrategyLeg(option_type="call", strike=long_call_strike, premium=long_call_premium, position="long"),
                    underlying_price=underlying_price,
                    expiration_date=expiration_date,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    implied_volatility=call_iv if call_iv > 0 else 0.3,
                    strategy_type="credit",
                    option_type="call",
                )
                
                butterfly = IronButterfly(
                    put_spread=put_spread,
                    call_spread=call_spread,
                    underlying_price=underlying_price,
                    expiration_date=expiration_date,
                    center_strike=center_strike,
                )
                
                if butterfly.probability_of_profit >= min_probability:
                    butterflies.append(butterfly)
    
    # Sort by risk/reward ratio
    butterflies.sort(key=lambda b: b.risk_reward_ratio)
    
    return butterflies[:max_results]

