"""Input validation utilities."""

import re
from typing import Tuple


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate stock ticker symbol.
    
    Parameters:
        ticker: Ticker symbol to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ticker:
        return False, "Ticker symbol is required"
    
    ticker = ticker.strip().upper()
    
    # Check length (1-5 characters for US stocks)
    if len(ticker) > 5:
        return False, "Ticker symbol too long (max 5 characters)"
    
    # Check format (letters only for basic US stocks)
    if not re.match(r"^[A-Z]+$", ticker):
        return False, "Ticker should contain only letters"
    
    return True, ""


def validate_probability(probability: float) -> Tuple[bool, str]:
    """
    Validate probability input.
    
    Parameters:
        probability: Probability value (0-1 or 0-100)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Handle percentage input
    if probability > 1:
        probability = probability / 100
    
    if probability < 0.5:
        return False, "Probability should be at least 50%"
    
    if probability > 0.99:
        return False, "Probability should be less than 99%"
    
    return True, ""


def format_currency(value: float, include_sign: bool = False) -> str:
    """
    Format a number as currency.
    
    Parameters:
        value: Numeric value
        include_sign: Whether to include + sign for positive values
    
    Returns:
        Formatted currency string
    """
    if include_sign and value > 0:
        return f"+${value:,.2f}"
    elif value < 0:
        return f"-${abs(value):,.2f}"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format a decimal as percentage.
    
    Parameters:
        value: Decimal value (e.g., 0.75 for 75%)
        decimal_places: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"

