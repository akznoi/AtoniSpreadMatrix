"""Base class for options strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, Any, List, Union


@dataclass
class StrategyLeg:
    """Represents a single leg of an options strategy."""
    
    option_type: str  # 'call' or 'put'
    strike: float
    premium: float
    position: str  # 'long' or 'short'
    quantity: int = 1
    
    @property
    def is_long(self) -> bool:
        return self.position.lower() == "long"
    
    @property
    def is_short(self) -> bool:
        return self.position.lower() == "short"
    
    @property
    def net_premium(self) -> float:
        """Positive for credit (short), negative for debit (long)."""
        if self.is_short:
            return self.premium * self.quantity
        else:
            return -self.premium * self.quantity


class OptionsStrategy(ABC):
    """Abstract base class for options strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @property
    @abstractmethod
    def max_profit(self) -> float:
        """Maximum potential profit."""
        pass
    
    @property
    @abstractmethod
    def max_loss(self) -> float:
        """Maximum potential loss."""
        pass
    
    @property
    @abstractmethod
    def breakeven(self) -> Union[float, List[float]]:
        """Breakeven price(s)."""
        pass
    
    @property
    @abstractmethod
    def capital_requirement(self) -> float:
        """Capital/margin required to enter the position."""
        pass
    
    @abstractmethod
    def profit_at_price(self, price: float) -> float:
        """Calculate profit/loss at a given stock price at expiration."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for display."""
        pass

