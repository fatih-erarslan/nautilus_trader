"""
Trading Decision Engine Base Classes
Defines abstract interfaces for trading decision systems
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .models import TradingSignal


class TradingDecisionEngine(ABC):
    """
    Main trading decision engine interface
    
    Processes various data sources (sentiment, market data, portfolio state)
    and generates trading signals following defined strategies and risk parameters.
    """
    
    @abstractmethod
    async def process_sentiment(self, sentiment_data: Dict[str, Any]) -> TradingSignal:
        """
        Convert news sentiment data to trading signal
        
        Args:
            sentiment_data: Dictionary containing:
                - asset: str - Asset symbol
                - sentiment_score: float - Sentiment score (-1 to 1)
                - confidence: float - Confidence level (0 to 1)
                - market_impact: dict - Expected market impact
                - source_events: list - Source news/event IDs
                
        Returns:
            TradingSignal object or None if no signal generated
        """
        pass
    
    @abstractmethod
    async def evaluate_portfolio(self, current_positions: Dict[str, Any]) -> List[TradingSignal]:
        """
        Evaluate current portfolio and generate rebalancing signals
        
        Args:
            current_positions: Dictionary mapping asset symbols to position info:
                - size: float - Position size as fraction of portfolio
                - entry_price: float - Original entry price
                - current_price: float - Current market price
                - unrealized_pnl: float - Unrealized profit/loss percentage
                
        Returns:
            List of TradingSignal objects for rebalancing
        """
        pass
    
    @abstractmethod
    async def process_market_data(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Process market data for technical analysis signals
        
        Args:
            market_data: Dictionary containing:
                - asset: str - Asset symbol
                - price: float - Current price
                - volume: float - Trading volume
                - technical_indicators: dict - RSI, MACD, etc.
                
        Returns:
            List of TradingSignal objects based on technical analysis
        """
        pass
    
    @abstractmethod
    def set_risk_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update risk management parameters
        
        Args:
            params: Dictionary of risk parameters:
                - max_position_size: float - Maximum size per position
                - max_portfolio_risk: float - Maximum total portfolio risk
                - max_correlation: float - Maximum correlation between positions
                - stop_loss_multiplier: float - ATR multiplier for stops
        """
        pass
    
    @abstractmethod
    def get_active_signals(self) -> List[TradingSignal]:
        """
        Get currently active trading signals
        
        Returns:
            List of active TradingSignal objects
        """
        pass


class SignalFilter(ABC):
    """
    Base class for signal filters
    
    Filters allow modular processing of trading signals based on
    various criteria (risk, strategy, asset type, etc.)
    """
    
    @abstractmethod
    def filter(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Filter trading signals based on criteria
        
        Args:
            signals: List of TradingSignal objects to filter
            
        Returns:
            Filtered list of TradingSignal objects
        """
        pass


class StrategyBase(ABC):
    """
    Base class for trading strategies
    
    Each strategy implements specific logic for generating
    trading signals based on its methodology
    """
    
    @abstractmethod
    async def generate_signal(self, data: Dict[str, Any]) -> TradingSignal:
        """
        Generate trading signal based on strategy logic
        
        Args:
            data: Strategy-specific data dictionary
            
        Returns:
            TradingSignal object or None if no signal
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate if signal meets strategy requirements
        
        Args:
            signal: TradingSignal to validate
            
        Returns:
            True if signal is valid for this strategy
        """
        pass