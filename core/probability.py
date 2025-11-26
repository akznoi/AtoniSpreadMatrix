"""Probability calculations for options strategies."""

import pandas as pd
from .black_scholes import delta


def probability_of_profit(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    position: str = "short",
) -> float:
    """
    Calculate probability of profit for an option position using delta approximation.
    
    For a short put: P(profit) ≈ 1 - |put delta| = probability stock stays above strike
    For a short call: P(profit) ≈ 1 - call delta = probability stock stays below strike
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        position: 'long' or 'short'
    
    Returns:
        Probability of profit (0 to 1)
    """
    option_delta = delta(S, K, T, r, sigma, option_type)
    
    if option_type.lower() == "put":
        # Put delta is negative, |delta| approximates P(ITM)
        prob_itm = abs(option_delta)
        if position.lower() == "short":
            # Short put profits when stock stays above strike
            return 1 - prob_itm
        else:
            # Long put profits when stock goes below strike
            return prob_itm
    else:
        # Call delta is positive, approximates P(ITM)
        prob_itm = option_delta
        if position.lower() == "short":
            # Short call profits when stock stays below strike
            return 1 - prob_itm
        else:
            # Long call profits when stock goes above strike
            return prob_itm


def filter_strikes_by_probability(
    options_df: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    min_probability: float,
    option_type: str = "put",
    position: str = "short",
) -> pd.DataFrame:
    """
    Filter options chain to only include strikes meeting probability threshold.
    
    Parameters:
        options_df: DataFrame with options chain data (must have 'strike' and 'impliedVolatility' columns)
        S: Current stock price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        min_probability: Minimum probability of profit (0 to 1)
        option_type: 'call' or 'put'
        position: 'long' or 'short'
    
    Returns:
        Filtered DataFrame with additional 'probability' column
    """
    df = options_df.copy()
    
    # Calculate probability for each strike
    df["probability"] = df.apply(
        lambda row: probability_of_profit(
            S=S,
            K=row["strike"],
            T=T,
            r=r,
            sigma=row.get("impliedVolatility", 0.3),
            option_type=option_type,
            position=position,
        ),
        axis=1,
    )
    
    # Filter by minimum probability
    filtered = df[df["probability"] >= min_probability].copy()
    
    return filtered.sort_values("strike", ascending=False)

