"""Base classes for Trading Decision Engine."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import TradingSignal


class TradingDecisionEngine(ABC):
    """Abstract base class for trading decision engines."""
    
    @abstractmethod
    async def process_sentiment(self, sentiment_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Process sentiment data into trading signal.
        
        Args:
            sentiment_data: Dictionary containing sentiment analysis results
            
        Returns:
            TradingSignal if actionable, None otherwise
        """
        pass
    
    @abstractmethod
    async def evaluate_portfolio(self, current_positions: Dict[str, Any]) -> List[TradingSignal]:
        """
        Evaluate current portfolio and generate rebalancing signals.
        
        Args:
            current_positions: Dictionary of current positions
            
        Returns:
            List of trading signals for portfolio adjustments
        """
        pass
    
    @abstractmethod
    async def process_market_data(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Process market data for technical signals.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            List of trading signals based on technical analysis
        """
        pass
    
    @abstractmethod
    def set_risk_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update risk management parameters.
        
        Args:
            params: Dictionary of risk parameters
        """
        pass
    
    @abstractmethod
    def get_active_signals(self) -> List[TradingSignal]:
        """
        Get currently active trading signals.
        
        Returns:
            List of active trading signals
        """
        pass