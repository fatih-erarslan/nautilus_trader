"""
Trading API Interfaces

This module defines the base interfaces that all trading API implementations
must follow to ensure consistency across different platforms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class TradingAPIInterface(ABC):
    """
    Abstract base class for trading API implementations.
    
    All trading APIs (crypto exchanges, DeFi protocols, etc.) should
    implement this interface to ensure compatibility with the trading system.
    """
    
    @abstractmethod
    async def get_balance(self, asset: str) -> Dict[str, Any]:
        """
        Get balance for a specific asset.
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with balance information
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price ticker for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with ticker information
        """
        pass
    
    @abstractmethod
    async def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            order_params: Dictionary with order parameters
            
        Returns:
            Dictionary with order result
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dictionary with cancellation result
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of an order.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dictionary with order status
        """
        pass


class MarketDataInterface(ABC):
    """
    Abstract base class for market data providers.
    """
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book data."""
        pass
    
    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        pass
    
    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[List[Any]]:
        """Get candlestick/kline data."""
        pass


class AccountInterface(ABC):
    """
    Abstract base class for account management.
    """
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trade history."""
        pass
    
    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        pass