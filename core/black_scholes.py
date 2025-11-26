"""Black-Scholes option pricing model and Greeks calculations."""

from typing import Tuple
import numpy as np
from scipy.stats import norm


def calculate_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float]:
    """
    Calculate d1 and d2 parameters for Black-Scholes formula.
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
    
    Returns:
        Tuple of (d1, d2)
    """
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate European call option price using Black-Scholes.
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
    
    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)
    
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate European put option price using Black-Scholes.
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
    
    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - S, 0)
    
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Calculate option delta (rate of change of option price with respect to stock price).
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Delta value (-1 to 1)
    """
    if T <= 0 or sigma <= 0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    
    if option_type.lower() == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option gamma (rate of change of delta with respect to stock price).
    Gamma is the same for both calls and puts.
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
    
    Returns:
        Gamma value
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Calculate option theta (rate of change of option price with respect to time).
    Returns daily theta (divide annual by 365).
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Theta value (daily, typically negative)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    
    # Common term
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if option_type.lower() == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        annual_theta = term1 + term2
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        annual_theta = term1 + term2
    
    # Return daily theta
    return annual_theta / 365


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option vega (rate of change of option price with respect to volatility).
    Vega is the same for both calls and puts.
    Returns vega per 1% change in volatility.
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
    
    Returns:
        Vega value (per 1% vol change)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    # Divide by 100 to get vega per 1% change
    return (S * norm.pdf(d1) * np.sqrt(T)) / 100


def rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Calculate option rho (rate of change of option price with respect to interest rate).
    Returns rho per 1% change in interest rate.
    
    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Rho value (per 1% rate change)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    
    if option_type.lower() == "call":
        return (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    else:
        return (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
        market_price: Current market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        option_type: 'call' or 'put'
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
    
    Returns:
        Implied volatility (annualized)
    """
    if T <= 0:
        return 0.0
    
    # Initial guess
    sigma = 0.3
    
    for _ in range(max_iterations):
        if option_type.lower() == "call":
            price = call_price(S, K, T, r, sigma)
        else:
            price = put_price(S, K, T, r, sigma)
        
        diff = price - market_price
        
        if abs(diff) < tolerance:
            return sigma
        
        # Vega for Newton-Raphson (need raw vega, not per 1%)
        d1, _ = calculate_d1_d2(S, K, T, r, sigma)
        vega_raw = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega_raw < 1e-10:
            break
        
        sigma = sigma - diff / vega_raw
        
        # Keep sigma in reasonable bounds
        sigma = max(0.01, min(sigma, 5.0))
    
    return sigma

