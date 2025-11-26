"""
Utility functions for IBKR integration

Provides helper functions for common tasks like contract creation,
data conversion, error handling, and performance monitoring.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class ContractType(Enum):
    """Contract types supported by IB"""
    STOCK = "STK"
    OPTION = "OPT"
    FUTURE = "FUT"
    FOREX = "CASH"
    INDEX = "IND"
    CRYPTO = "CRYPTO"


class OrderType(Enum):
    """Order types supported by IB"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    MARKET_ON_CLOSE = "MOC"
    LIMIT_ON_CLOSE = "LOC"
    PEGGED_TO_MARKET = "PEG MKT"
    RELATIVE = "REL"
    BRACKET = "BRK"
    TRAILING_STOP = "TRAIL"


class TimeInForce(Enum):
    """Time in force options"""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date
    OPG = "OPG"  # At the Open
    CLS = "CLS"  # At the Close


class Exchange(Enum):
    """Common exchanges"""
    SMART = "SMART"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    CBOE = "CBOE"
    CME = "CME"
    NYMEX = "NYMEX"
    IDEALPRO = "IDEALPRO"  # Forex


@dataclass
class ContractDetails:
    """Contract details for creating IB contracts"""
    symbol: str
    contract_type: ContractType
    exchange: Exchange = Exchange.SMART
    currency: str = "USD"
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None  # 'C' for call, 'P' for put
    multiplier: Optional[str] = None
    trading_class: Optional[str] = None
    local_symbol: Optional[str] = None


@dataclass
class OrderDetails:
    """Order details for creating IB orders"""
    symbol: str
    quantity: int
    side: str  # 'BUY' or 'SELL'
    order_type: OrderType
    price: Optional[float] = None
    aux_price: Optional[float] = None  # Stop price for stop orders
    time_in_force: TimeInForce = TimeInForce.DAY
    parent_id: Optional[int] = None
    oca_group: Optional[str] = None
    transmit: bool = True
    hidden: bool = False
    min_qty: Optional[int] = None
    percent_offset: Optional[float] = None
    trail_stop_price: Optional[float] = None
    good_after_time: Optional[str] = None
    good_till_date: Optional[str] = None


class ContractHelper:
    """Helper class for creating IB contracts"""
    
    @staticmethod
    def create_stock_contract(symbol: str, 
                             exchange: Exchange = Exchange.SMART,
                             currency: str = "USD"):
        """Create a stock contract"""
        try:
            from ib_insync import Stock
            return Stock(symbol, exchange.value, currency)
        except ImportError:
            # Mock for development
            return {
                'symbol': symbol,
                'secType': 'STK',
                'exchange': exchange.value,
                'currency': currency
            }
    
    @staticmethod
    def create_option_contract(symbol: str,
                              expiry: str,
                              strike: float,
                              right: str,
                              exchange: Exchange = Exchange.SMART,
                              currency: str = "USD"):
        """Create an option contract"""
        try:
            from ib_insync import Option
            return Option(symbol, expiry, strike, right, exchange.value, currency)
        except ImportError:
            # Mock for development
            return {
                'symbol': symbol,
                'secType': 'OPT',
                'expiry': expiry,
                'strike': strike,
                'right': right,
                'exchange': exchange.value,
                'currency': currency
            }
    
    @staticmethod
    def create_future_contract(symbol: str,
                              expiry: str,
                              exchange: Exchange,
                              currency: str = "USD"):
        """Create a futures contract"""
        try:
            from ib_insync import Future
            return Future(symbol, expiry, exchange.value, currency)
        except ImportError:
            # Mock for development
            return {
                'symbol': symbol,
                'secType': 'FUT',
                'expiry': expiry,
                'exchange': exchange.value,
                'currency': currency
            }
    
    @staticmethod
    def create_forex_contract(symbol: str,
                             base_currency: str = "USD",
                             quote_currency: str = "EUR"):
        """Create a forex contract"""
        try:
            from ib_insync import Forex
            return Forex(f"{base_currency}{quote_currency}")
        except ImportError:
            # Mock for development
            return {
                'symbol': f"{base_currency}{quote_currency}",
                'secType': 'CASH',
                'exchange': 'IDEALPRO',
                'currency': quote_currency
            }
    
    @staticmethod
    def create_contract_from_details(details: ContractDetails):
        """Create contract from ContractDetails"""
        if details.contract_type == ContractType.STOCK:
            return ContractHelper.create_stock_contract(
                details.symbol, details.exchange, details.currency
            )
        elif details.contract_type == ContractType.OPTION:
            return ContractHelper.create_option_contract(
                details.symbol, details.expiry, details.strike,
                details.right, details.exchange, details.currency
            )
        elif details.contract_type == ContractType.FUTURE:
            return ContractHelper.create_future_contract(
                details.symbol, details.expiry, details.exchange, details.currency
            )
        elif details.contract_type == ContractType.FOREX:
            return ContractHelper.create_forex_contract(details.symbol)
        else:
            raise ValueError(f"Unsupported contract type: {details.contract_type}")


class OrderHelper:
    """Helper class for creating IB orders"""
    
    @staticmethod
    def create_market_order(side: str, quantity: int, **kwargs):
        """Create a market order"""
        try:
            from ib_insync import MarketOrder
            order = MarketOrder(side, quantity)
            OrderHelper._apply_order_parameters(order, kwargs)
            return order
        except ImportError:
            # Mock for development
            return {
                'action': side,
                'totalQuantity': quantity,
                'orderType': 'MKT',
                **kwargs
            }
    
    @staticmethod
    def create_limit_order(side: str, quantity: int, price: float, **kwargs):
        """Create a limit order"""
        try:
            from ib_insync import LimitOrder
            order = LimitOrder(side, quantity, price)
            OrderHelper._apply_order_parameters(order, kwargs)
            return order
        except ImportError:
            # Mock for development
            return {
                'action': side,
                'totalQuantity': quantity,
                'orderType': 'LMT',
                'lmtPrice': price,
                **kwargs
            }
    
    @staticmethod
    def create_stop_order(side: str, quantity: int, stop_price: float, **kwargs):
        """Create a stop order"""
        try:
            from ib_insync import StopOrder
            order = StopOrder(side, quantity, stop_price)
            OrderHelper._apply_order_parameters(order, kwargs)
            return order
        except ImportError:
            # Mock for development
            return {
                'action': side,
                'totalQuantity': quantity,
                'orderType': 'STP',
                'auxPrice': stop_price,
                **kwargs
            }
    
    @staticmethod
    def create_stop_limit_order(side: str, quantity: int, 
                               limit_price: float, stop_price: float, **kwargs):
        """Create a stop-limit order"""
        try:
            from ib_insync import StopLimitOrder
            order = StopLimitOrder(side, quantity, limit_price, stop_price)
            OrderHelper._apply_order_parameters(order, kwargs)
            return order
        except ImportError:
            # Mock for development
            return {
                'action': side,
                'totalQuantity': quantity,
                'orderType': 'STP LMT',
                'lmtPrice': limit_price,
                'auxPrice': stop_price,
                **kwargs
            }
    
    @staticmethod
    def create_bracket_order(parent_order_id: int,
                            side: str,
                            quantity: int,
                            limit_price: float,
                            take_profit: float,
                            stop_loss: float):
        """Create a bracket order (parent + profit taker + stop loss)"""
        # Main order
        main_order = OrderHelper.create_limit_order(side, quantity, limit_price)
        
        # Profit taker
        profit_side = "SELL" if side == "BUY" else "BUY"
        profit_order = OrderHelper.create_limit_order(profit_side, quantity, take_profit)
        
        # Stop loss
        stop_order = OrderHelper.create_stop_order(profit_side, quantity, stop_loss)
        
        # Set parent relationships
        try:
            # IB-specific bracket order setup
            main_order.transmit = False
            profit_order.parentId = parent_order_id
            profit_order.transmit = False
            stop_order.parentId = parent_order_id
            stop_order.transmit = True
            
            # Create OCA group
            oca_group = f"OCA_{parent_order_id}_{int(time.time())}"
            profit_order.ocaGroup = oca_group
            stop_order.ocaGroup = oca_group
            
            return main_order, profit_order, stop_order
        except:
            # Mock for development
            return [main_order, profit_order, stop_order]
    
    @staticmethod
    def _apply_order_parameters(order, parameters: Dict[str, Any]):
        """Apply additional parameters to order"""
        for key, value in parameters.items():
            if hasattr(order, key):
                setattr(order, key, value)
    
    @staticmethod
    def create_order_from_details(details: OrderDetails):
        """Create order from OrderDetails"""
        if details.order_type == OrderType.MARKET:
            return OrderHelper.create_market_order(
                details.side, details.quantity,
                tif=details.time_in_force.value,
                transmit=details.transmit,
                hidden=details.hidden
            )
        elif details.order_type == OrderType.LIMIT:
            return OrderHelper.create_limit_order(
                details.side, details.quantity, details.price,
                tif=details.time_in_force.value,
                transmit=details.transmit,
                hidden=details.hidden
            )
        elif details.order_type == OrderType.STOP:
            return OrderHelper.create_stop_order(
                details.side, details.quantity, details.aux_price,
                tif=details.time_in_force.value,
                transmit=details.transmit,
                hidden=details.hidden
            )
        elif details.order_type == OrderType.STOP_LIMIT:
            return OrderHelper.create_stop_limit_order(
                details.side, details.quantity, details.price, details.aux_price,
                tif=details.time_in_force.value,
                transmit=details.transmit,
                hidden=details.hidden
            )
        else:
            raise ValueError(f"Unsupported order type: {details.order_type}")


class DataConverter:
    """Helper class for data conversion and formatting"""
    
    @staticmethod
    def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
        """Convert timestamp to datetime"""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> float:
        """Convert datetime to timestamp"""
        return dt.timestamp()
    
    @staticmethod
    def format_price(price: float, decimals: int = 2) -> str:
        """Format price for display"""
        return f"{price:.{decimals}f}"
    
    @staticmethod
    def format_quantity(quantity: int) -> str:
        """Format quantity for display"""
        return f"{quantity:,}"
    
    @staticmethod
    def format_currency(amount: float, currency: str = "USD") -> str:
        """Format currency amount"""
        if currency == "USD":
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """Calculate percentage"""
        if total == 0:
            return 0.0
        return (value / total) * 100
    
    @staticmethod
    def calculate_spread_bps(bid: float, ask: float) -> float:
        """Calculate spread in basis points"""
        if bid == 0 or ask == 0:
            return 0.0
        mid = (bid + ask) / 2
        spread = ask - bid
        return (spread / mid) * 10000  # Convert to basis points
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize symbol format"""
        return symbol.upper().strip()
    
    @staticmethod
    def parse_contract_string(contract_str: str) -> Dict[str, str]:
        """Parse contract string (e.g., 'AAPL-STK-SMART-USD')"""
        parts = contract_str.split('-')
        if len(parts) >= 4:
            return {
                'symbol': parts[0],
                'secType': parts[1],
                'exchange': parts[2],
                'currency': parts[3]
            }
        return {}


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_timer(self, name: str) -> str:
        """Start a timer"""
        timer_id = f"{name}_{int(time.time()*1000000)}"
        self.metrics[timer_id] = {
            'name': name,
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End a timer and return duration"""
        if timer_id in self.metrics:
            metric = self.metrics[timer_id]
            metric['end_time'] = time.time()
            metric['duration'] = metric['end_time'] - metric['start_time']
            return metric['duration']
        return 0.0
    
    def record_metric(self, name: str, value: float, unit: str = "ms"):
        """Record a metric value"""
        if name not in self.metrics:
            self.metrics[name] = {'values': [], 'unit': unit}
        self.metrics[name]['values'].append(value)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]['values']
        if not values:
            return {}
        
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p50': sorted(values)[len(values)//2],
            'p95': sorted(values)[int(len(values)*0.95)] if len(values) > 20 else max(values),
            'p99': sorted(values)[int(len(values)*0.99)] if len(values) > 100 else max(values)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'uptime': time.time() - self.start_time,
            'metrics': {}
        }
        
        for name, data in self.metrics.items():
            if 'values' in data:
                summary['metrics'][name] = {
                    'unit': data['unit'],
                    'stats': self.get_stats(name)
                }
        
        return summary


class ErrorHandler:
    """Error handling utilities"""
    
    @staticmethod
    def is_connection_error(error: Exception) -> bool:
        """Check if error is connection-related"""
        connection_errors = [
            'Connection refused',
            'Connection reset',
            'Connection timeout',
            'Socket error',
            'Network unreachable'
        ]
        
        error_str = str(error).lower()
        return any(err in error_str for err in connection_errors)
    
    @staticmethod
    def is_order_error(error: Exception) -> bool:
        """Check if error is order-related"""
        order_errors = [
            'Order rejected',
            'Invalid order',
            'Insufficient funds',
            'Position limit exceeded',
            'Invalid price'
        ]
        
        error_str = str(error).lower()
        return any(err in error_str for err in order_errors)
    
    @staticmethod
    def is_data_error(error: Exception) -> bool:
        """Check if error is data-related"""
        data_errors = [
            'No market data',
            'Invalid symbol',
            'Data not available',
            'Subscription failed'
        ]
        
        error_str = str(error).lower()
        return any(err in error_str for err in data_errors)
    
    @staticmethod
    def get_error_category(error: Exception) -> str:
        """Get error category"""
        if ErrorHandler.is_connection_error(error):
            return "connection"
        elif ErrorHandler.is_order_error(error):
            return "order"
        elif ErrorHandler.is_data_error(error):
            return "data"
        else:
            return "unknown"
    
    @staticmethod
    def should_retry(error: Exception) -> bool:
        """Check if error should trigger retry"""
        # Retry connection errors and some data errors
        return ErrorHandler.is_connection_error(error) or \
               "temporarily unavailable" in str(error).lower()


class RateLimiter:
    """Rate limiting utility"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self) -> bool:
        """Acquire rate limit permission"""
        now = time.time()
        
        # Remove old requests
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # Check if we can make a request
        if len(self.requests) >= self.max_requests:
            return False
        
        # Add current request
        self.requests.append(now)
        return True
    
    async def wait_for_slot(self) -> None:
        """Wait for next available slot"""
        while not await self.acquire():
            await asyncio.sleep(0.01)  # 10ms wait


class IdGenerator:
    """Generate unique IDs"""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.counter = 0
    
    def next_id(self) -> str:
        """Generate next ID"""
        self.counter += 1
        timestamp = int(time.time() * 1000000)  # microseconds
        return f"{self.prefix}{timestamp}_{self.counter}"
    
    def hash_id(self, data: str) -> str:
        """Generate hash-based ID"""
        hash_obj = hashlib.md5(data.encode())
        return f"{self.prefix}{hash_obj.hexdigest()[:8]}"


# Global utilities
performance_monitor = PerformanceMonitor()
order_id_generator = IdGenerator("ORD_")
request_id_generator = IdGenerator("REQ_")

# Rate limiters for different operations
order_rate_limiter = RateLimiter(max_requests=100, time_window=60.0)  # 100 orders per minute
data_rate_limiter = RateLimiter(max_requests=1000, time_window=60.0)  # 1000 data requests per minute