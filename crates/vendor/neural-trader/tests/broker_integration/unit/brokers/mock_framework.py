"""
Mock Framework for Broker APIs

This module provides comprehensive mocking capabilities for all supported broker APIs,
enabling reliable unit testing without external dependencies.
"""

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from unittest.mock import Mock, MagicMock, patch
import logging

from ..fixtures.broker_responses import BrokerResponseFixtures


class BaseBrokerMock(ABC):
    """Base class for all broker API mocks"""
    
    def __init__(self, name: str):
        self.name = name
        self.call_history: List[Dict[str, Any]] = []
        self.response_queue: List[Dict[str, Any]] = []
        self.error_mode: Optional[str] = None
        self.latency_ms: int = 0
        self.rate_limit_enabled: bool = False
        self.rate_limit_calls: int = 0
        self.rate_limit_window: int = 60  # seconds
        self.rate_limit_max_calls: int = 200
        self.connection_status: str = "connected"
        self.logger = logging.getLogger(f"mock.{name}")
        
        # Mock state
        self.accounts: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}
        self.market_data: Dict[str, Dict] = {}
        
        # Callbacks
        self.on_order_callback: Optional[Callable] = None
        self.on_fill_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
    
    def _log_call(self, method: str, args: tuple, kwargs: dict) -> None:
        """Log API call for testing verification"""
        call_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "broker": self.name
        }
        self.call_history.append(call_info)
        self.logger.debug(f"API call: {method} with args={args}, kwargs={kwargs}")
    
    def _simulate_latency(self) -> None:
        """Simulate network latency"""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
    
    def _check_rate_limit(self) -> None:
        """Check rate limit constraints"""
        if not self.rate_limit_enabled:
            return
            
        current_time = time.time()
        # Clean old calls
        self.rate_limit_calls = len([
            call for call in self.call_history 
            if current_time - call.get("timestamp", 0) < self.rate_limit_window
        ])
        
        if self.rate_limit_calls >= self.rate_limit_max_calls:
            raise Exception("Rate limit exceeded")
    
    def _handle_error_mode(self) -> None:
        """Handle error simulation modes"""
        if self.error_mode == "connection_error":
            raise ConnectionError("Connection failed")
        elif self.error_mode == "timeout":
            raise TimeoutError("Request timed out")
        elif self.error_mode == "unauthorized":
            raise Exception("Unauthorized access")
        elif self.error_mode == "rate_limited":
            raise Exception("Rate limit exceeded")
        elif self.error_mode == "server_error":
            raise Exception("Internal server error")
        elif self.error_mode == "random":
            if random.random() < 0.1:  # 10% chance of error
                errors = ["connection_error", "timeout", "server_error"]
                self._handle_error_mode_type(random.choice(errors))
    
    def _handle_error_mode_type(self, error_type: str) -> None:
        """Handle specific error type"""
        original_error_mode = self.error_mode
        self.error_mode = error_type
        try:
            self._handle_error_mode()
        finally:
            self.error_mode = original_error_mode
    
    def set_response(self, response: Dict[str, Any]) -> None:
        """Queue a response for the next API call"""
        self.response_queue.append(response)
    
    def enable_error_mode(self, error_type: str) -> None:
        """Enable specific error simulation"""
        self.error_mode = error_type
        self.logger.info(f"Error mode enabled: {error_type}")
    
    def disable_error_mode(self) -> None:
        """Disable error simulation"""
        self.error_mode = None
        self.logger.info("Error mode disabled")
    
    def set_latency(self, ms: int) -> None:
        """Set network latency simulation"""
        self.latency_ms = ms
        self.logger.info(f"Latency set to {ms}ms")
    
    def enable_rate_limiting(self, max_calls: int = 200, window: int = 60) -> None:
        """Enable rate limiting simulation"""
        self.rate_limit_enabled = True
        self.rate_limit_max_calls = max_calls
        self.rate_limit_window = window
        self.logger.info(f"Rate limiting enabled: {max_calls} calls per {window} seconds")
    
    def disable_rate_limiting(self) -> None:
        """Disable rate limiting simulation"""
        self.rate_limit_enabled = False
        self.logger.info("Rate limiting disabled")
    
    def set_connection_status(self, status: str) -> None:
        """Set connection status"""
        self.connection_status = status
        self.logger.info(f"Connection status: {status}")
    
    def clear_history(self) -> None:
        """Clear call history"""
        self.call_history.clear()
        self.logger.debug("Call history cleared")
    
    def get_call_count(self) -> int:
        """Get total number of API calls made"""
        return len(self.call_history)
    
    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get last API call made"""
        return self.call_history[-1] if self.call_history else None
    
    def verify_call(self, method: str, args: tuple = None, kwargs: dict = None) -> bool:
        """Verify if a specific call was made"""
        for call in self.call_history:
            if call["method"] == method:
                if args is not None and call["args"] != args:
                    continue
                if kwargs is not None and call["kwargs"] != kwargs:
                    continue
                return True
        return False
    
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    @abstractmethod
    def submit_order(self, symbol: str, qty: int, side: str, order_type: str, **kwargs) -> Dict[str, Any]:
        """Submit order"""
        pass
    
    @abstractmethod
    def get_orders(self, status: str = None) -> List[Dict[str, Any]]:
        """Get orders"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions"""
        pass


class AlpacaMock(BaseBrokerMock):
    """Alpaca-specific API mock"""
    
    def __init__(self):
        super().__init__("alpaca")
        self.accounts["default"] = BrokerResponseFixtures.ALPACA_ACCOUNT_RESPONSE.copy()
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        self._log_call("get_account", (), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        if self.response_queue:
            return self.response_queue.pop(0)
        return self.accounts["default"]
    
    def submit_order(self, symbol: str, qty: int, side: str, order_type: str, **kwargs) -> Dict[str, Any]:
        """Submit order"""
        self._log_call("submit_order", (symbol, qty, side, order_type), kwargs)
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        if self.response_queue:
            order = self.response_queue.pop(0)
        else:
            order = BrokerResponseFixtures.ALPACA_ORDER_RESPONSE.copy()
            order.update({
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": order_type,
                "order_type": order_type,
                **kwargs
            })
        
        order_id = order["id"]
        self.orders[order_id] = order
        
        # Trigger callback if set
        if self.on_order_callback:
            self.on_order_callback(order)
        
        return order
    
    def get_orders(self, status: str = None) -> List[Dict[str, Any]]:
        """Get orders"""
        self._log_call("get_orders", (), {"status": status})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        orders = list(self.orders.values())
        if status:
            orders = [order for order in orders if order.get("status") == status]
        
        return orders
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions"""
        self._log_call("get_positions", (), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        return list(self.positions.values())
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        self._log_call("cancel_order", (order_id,), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        if order_id in self.orders:
            self.orders[order_id]["status"] = "canceled"
            self.orders[order_id]["canceled_at"] = datetime.now(timezone.utc).isoformat()
            return self.orders[order_id]
        else:
            raise Exception(f"Order {order_id} not found")
    
    def get_portfolio_history(self, period: str = "1M") -> Dict[str, Any]:
        """Get portfolio history"""
        self._log_call("get_portfolio_history", (), {"period": period})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        return BrokerResponseFixtures.ALPACA_PORTFOLIO_HISTORY
    
    def simulate_order_fill(self, order_id: str, fill_price: float, fill_qty: int = None) -> None:
        """Simulate order fill for testing"""
        if order_id not in self.orders:
            raise Exception(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        original_qty = int(order["qty"])
        filled_qty = int(order.get("filled_qty", 0))
        
        if fill_qty is None:
            fill_qty = original_qty - filled_qty
        
        new_filled_qty = filled_qty + fill_qty
        
        order.update({
            "filled_qty": str(new_filled_qty),
            "filled_avg_price": str(fill_price),
            "status": "filled" if new_filled_qty >= original_qty else "partially_filled",
            "filled_at": datetime.now(timezone.utc).isoformat() if new_filled_qty >= original_qty else None
        })
        
        # Update position
        symbol = order["symbol"]
        side = order["side"]
        
        if symbol not in self.positions:
            self.positions[symbol] = BrokerResponseFixtures.ALPACA_POSITION_RESPONSE.copy()
            self.positions[symbol]["symbol"] = symbol
            self.positions[symbol]["qty"] = "0"
        
        position = self.positions[symbol]
        current_qty = int(position["qty"])
        
        if side == "buy":
            new_qty = current_qty + fill_qty
        else:
            new_qty = current_qty - fill_qty
        
        position["qty"] = str(new_qty)
        position["side"] = "long" if new_qty > 0 else "short" if new_qty < 0 else "flat"
        
        # Trigger callback if set
        if self.on_fill_callback:
            self.on_fill_callback(order, fill_price, fill_qty)


class InteractiveBrokersMock(BaseBrokerMock):
    """Interactive Brokers API mock"""
    
    def __init__(self):
        super().__init__("ibkr")
        self.accounts["default"] = BrokerResponseFixtures.IBKR_ACCOUNT_RESPONSE.copy()
        self.portfolio_data = BrokerResponseFixtures.IBKR_PORTFOLIO_RESPONSE.copy()
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        self._log_call("get_account", (), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        return self.accounts["default"]
    
    def submit_order(self, symbol: str, qty: int, side: str, order_type: str, **kwargs) -> Dict[str, Any]:
        """Submit order"""
        self._log_call("submit_order", (symbol, qty, side, order_type), kwargs)
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        order = BrokerResponseFixtures.IBKR_ORDER_RESPONSE.copy()
        order.update({
            "symbol": symbol,
            "totalSize": qty,
            "side": side.upper(),
            "orderType": order_type.upper(),
            **kwargs
        })
        
        order_id = str(order["orderId"])
        self.orders[order_id] = order
        
        return order
    
    def get_orders(self, status: str = None) -> List[Dict[str, Any]]:
        """Get orders"""
        self._log_call("get_orders", (), {"status": status})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        orders = list(self.orders.values())
        if status:
            orders = [order for order in orders if order.get("status") == status]
        
        return orders
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions"""
        self._log_call("get_positions", (), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        return list(self.positions.values())
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio information"""
        self._log_call("get_portfolio", (), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        return self.portfolio_data


class TDAmeritradeMock(BaseBrokerMock):
    """TD Ameritrade API mock"""
    
    def __init__(self):
        super().__init__("td_ameritrade")
        self.accounts["default"] = BrokerResponseFixtures.TD_ACCOUNT_RESPONSE.copy()
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        self._log_call("get_account", (), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        return self.accounts["default"]
    
    def submit_order(self, symbol: str, qty: int, side: str, order_type: str, **kwargs) -> Dict[str, Any]:
        """Submit order"""
        self._log_call("submit_order", (symbol, qty, side, order_type), kwargs)
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        order = BrokerResponseFixtures.TD_ORDER_RESPONSE.copy()
        order.update({
            "orderLegCollection": [{
                "orderLegType": "EQUITY",
                "legId": 1,
                "instrument": {
                    "assetType": "EQUITY",
                    "symbol": symbol
                },
                "instruction": side.upper(),
                "quantity": qty
            }],
            "quantity": qty,
            "orderType": order_type.upper(),
            **kwargs
        })
        
        order_id = str(order["orderId"])
        self.orders[order_id] = order
        
        return order
    
    def get_orders(self, status: str = None) -> List[Dict[str, Any]]:
        """Get orders"""
        self._log_call("get_orders", (), {"status": status})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        orders = list(self.orders.values())
        if status:
            orders = [order for order in orders if order.get("status") == status]
        
        return orders
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions"""
        self._log_call("get_positions", (), {})
        self._simulate_latency()
        self._check_rate_limit()
        self._handle_error_mode()
        
        return list(self.positions.values())


class NewsAPIMock:
    """Mock for news aggregation APIs"""
    
    def __init__(self):
        self.articles_db: List[Dict[str, Any]] = []
        self.sentiment_scores: Dict[str, float] = {}
        self.call_history: List[Dict[str, Any]] = []
        self.error_mode: Optional[str] = None
        self.latency_ms: int = 0
        self.logger = logging.getLogger("mock.news_api")
    
    def _log_call(self, method: str, args: tuple, kwargs: dict) -> None:
        """Log API call for testing verification"""
        call_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "args": args,
            "kwargs": kwargs
        }
        self.call_history.append(call_info)
        self.logger.debug(f"API call: {method} with args={args}, kwargs={kwargs}")
    
    def _simulate_latency(self) -> None:
        """Simulate network latency"""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
    
    def _handle_error_mode(self) -> None:
        """Handle error simulation modes"""
        if self.error_mode == "connection_error":
            raise ConnectionError("Connection failed")
        elif self.error_mode == "timeout":
            raise TimeoutError("Request timed out")
        elif self.error_mode == "api_error":
            raise Exception("API error")
    
    def add_article(self, article: Dict[str, Any]) -> None:
        """Add test article to mock database"""
        self.articles_db.append(article)
    
    def get_articles(self, symbol: str = None, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Return mock articles for symbol"""
        self._log_call("get_articles", (symbol, limit), kwargs)
        self._simulate_latency()
        self._handle_error_mode()
        
        articles = self.articles_db
        
        if symbol:
            articles = [a for a in articles if symbol in a.get("symbols", [])]
        
        return articles[:limit]
    
    def get_sentiment(self, article_id: str) -> Dict[str, Any]:
        """Get sentiment for specific article"""
        self._log_call("get_sentiment", (article_id,), {})
        self._simulate_latency()
        self._handle_error_mode()
        
        if article_id in self.sentiment_scores:
            return {"sentiment": self.sentiment_scores[article_id]}
        
        # Find article in database
        for article in self.articles_db:
            if article["id"] == article_id:
                return article.get("sentiment", {"polarity": 0.0, "confidence": 0.5})
        
        raise Exception(f"Article {article_id} not found")
    
    def set_sentiment(self, article_id: str, sentiment: float) -> None:
        """Set sentiment score for testing"""
        self.sentiment_scores[article_id] = sentiment
    
    def enable_error_mode(self, error_type: str) -> None:
        """Enable error simulation"""
        self.error_mode = error_type
    
    def disable_error_mode(self) -> None:
        """Disable error simulation"""
        self.error_mode = None
    
    def set_latency(self, ms: int) -> None:
        """Set network latency simulation"""
        self.latency_ms = ms
    
    def clear_articles(self) -> None:
        """Clear all articles from mock database"""
        self.articles_db.clear()
    
    def clear_history(self) -> None:
        """Clear call history"""
        self.call_history.clear()


class MockBrokerFactory:
    """Factory for creating broker mocks"""
    
    @staticmethod
    def create_mock(broker_type: str) -> BaseBrokerMock:
        """Create appropriate broker mock"""
        if broker_type.lower() == "alpaca":
            return AlpacaMock()
        elif broker_type.lower() == "ibkr":
            return InteractiveBrokersMock()
        elif broker_type.lower() == "td_ameritrade":
            return TDAmeritradeMock()
        else:
            raise ValueError(f"Unknown broker type: {broker_type}")
    
    @staticmethod
    def create_all_mocks() -> Dict[str, BaseBrokerMock]:
        """Create all broker mocks"""
        return {
            "alpaca": AlpacaMock(),
            "ibkr": InteractiveBrokersMock(),
            "td_ameritrade": TDAmeritradeMock()
        }


class MockManager:
    """Centralized mock management"""
    
    def __init__(self):
        self.broker_mocks: Dict[str, BaseBrokerMock] = {}
        self.news_mock: NewsAPIMock = NewsAPIMock()
        self.patches: List[Any] = []
    
    def setup_broker_mocks(self, brokers: List[str]) -> None:
        """Setup broker mocks"""
        for broker in brokers:
            self.broker_mocks[broker] = MockBrokerFactory.create_mock(broker)
    
    def setup_news_mock(self) -> None:
        """Setup news API mock"""
        self.news_mock = NewsAPIMock()
    
    def apply_patches(self) -> None:
        """Apply all patches"""
        # This would typically patch the actual broker clients
        # with the mock implementations
        pass
    
    def remove_patches(self) -> None:
        """Remove all patches"""
        for patch_obj in self.patches:
            patch_obj.stop()
        self.patches.clear()
    
    def reset_all_mocks(self) -> None:
        """Reset all mocks to initial state"""
        for mock in self.broker_mocks.values():
            mock.clear_history()
            mock.disable_error_mode()
            mock.set_latency(0)
            mock.disable_rate_limiting()
        
        self.news_mock.clear_history()
        self.news_mock.clear_articles()
        self.news_mock.disable_error_mode()
        self.news_mock.set_latency(0)
    
    def get_broker_mock(self, broker: str) -> BaseBrokerMock:
        """Get specific broker mock"""
        return self.broker_mocks.get(broker)
    
    def get_news_mock(self) -> NewsAPIMock:
        """Get news API mock"""
        return self.news_mock