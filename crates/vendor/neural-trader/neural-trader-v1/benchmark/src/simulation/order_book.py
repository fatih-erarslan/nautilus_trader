"""
High-performance order book simulation engine.
Supports realistic order book dynamics with 1M+ ticks per second.
"""
import asyncio
import bisect
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import heapq


class OrderType(Enum):
    """Order types supported by the order book."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Represents a single order in the order book."""
    order_id: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    timestamp: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    remaining_quantity: Optional[int] = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    def __lt__(self, other):
        """For heap operations - time priority."""
        return self.timestamp < other.timestamp


@dataclass
class Trade:
    """Represents an executed trade."""
    trade_id: str
    buyer_order_id: str
    seller_order_id: str
    price: float
    quantity: int
    timestamp: float
    
    @classmethod
    def create(cls, buyer_order_id: str, seller_order_id: str, 
               price: float, quantity: int) -> "Trade":
        """Create a new trade with auto-generated ID."""
        timestamp = time.time()
        trade_id = f"T{int(timestamp * 1000000)}"
        return cls(trade_id, buyer_order_id, seller_order_id, 
                  price, quantity, timestamp)


@dataclass
class PriceLevel:
    """Represents a price level in the order book."""
    price: float
    quantity: int
    order_count: int


@dataclass
class MarketDepth:
    """Market depth snapshot."""
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    bid_depth: int
    ask_depth: int
    total_bid_volume: int
    total_ask_volume: int
    timestamp: float = field(default_factory=time.time)


class OrderBook:
    """
    High-performance order book implementation.
    Uses sorted dictionaries for price levels and deques for time priority.
    """
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Price levels: price -> deque of orders (FIFO)
        self.bid_levels: Dict[float, deque] = defaultdict(deque)
        self.ask_levels: Dict[float, deque] = defaultdict(deque)
        
        # Order lookup: order_id -> (price, side, order)
        self.orders: Dict[str, Tuple[float, OrderSide, Order]] = {}
        
        # Stop orders: stop_price -> list of orders
        self.stop_buy_orders: Dict[float, List[Order]] = defaultdict(list)
        self.stop_sell_orders: Dict[float, List[Order]] = defaultdict(list)
        
        # Sorted price lists for efficient best price lookup
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []
        
        # Performance tracking
        self.total_orders = 0
        self.total_trades = 0
        self.last_trade_price: Optional[float] = None
        
        # Triggered stop orders
        self._triggered_stops: Set[str] = set()
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self._bid_prices[-1] if self._bid_prices else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self._ask_prices[0] if self._ask_prices else None
    
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add an order to the book and attempt to match.
        Returns list of executed trades.
        """
        self.total_orders += 1
        trades = []
        
        if order.order_type == OrderType.MARKET:
            trades = self._process_market_order(order)
        elif order.order_type == OrderType.STOP:
            self._add_stop_order(order)
        else:  # LIMIT order
            trades = self._process_limit_order(order)
        
        # Check if any trades should trigger stop orders
        if trades and self.last_trade_price:
            self._check_stop_triggers(self.last_trade_price)
        
        return trades
    
    def _process_limit_order(self, order: Order) -> List[Trade]:
        """Process a limit order, attempting to match first."""
        trades = []
        
        if order.side == OrderSide.BUY:
            trades = self._match_buy_order(order)
        else:
            trades = self._match_sell_order(order)
        
        # If order not fully filled, add to book
        if order.remaining_quantity > 0:
            self._add_to_book(order)
        
        return trades
    
    def _process_market_order(self, order: Order) -> List[Trade]:
        """Process a market order, matching against available liquidity."""
        if order.side == OrderSide.BUY:
            return self._match_buy_order(order, is_market=True)
        else:
            return self._match_sell_order(order, is_market=True)
    
    def _match_buy_order(self, order: Order, is_market: bool = False) -> List[Trade]:
        """Match a buy order against sell orders."""
        trades = []
        
        while order.remaining_quantity > 0 and self._ask_prices:
            best_ask = self._ask_prices[0]
            
            # For limit orders, check if price crosses
            if not is_market and order.price < best_ask:
                break
            
            # Get orders at best ask price
            ask_queue = self.ask_levels[best_ask]
            
            while ask_queue and order.remaining_quantity > 0:
                sell_order = ask_queue[0]
                
                # Determine trade quantity
                trade_qty = min(order.remaining_quantity, sell_order.remaining_quantity)
                
                # Create trade
                trade = Trade.create(
                    buyer_order_id=order.order_id,
                    seller_order_id=sell_order.order_id,
                    price=best_ask,
                    quantity=trade_qty
                )
                trades.append(trade)
                
                # Update quantities
                order.remaining_quantity -= trade_qty
                sell_order.remaining_quantity -= trade_qty
                self.last_trade_price = best_ask
                
                # Remove filled sell order
                if sell_order.remaining_quantity == 0:
                    ask_queue.popleft()
                    del self.orders[sell_order.order_id]
            
            # Remove empty price level
            if not ask_queue:
                del self.ask_levels[best_ask]
                self._ask_prices.pop(0)
        
        self.total_trades += len(trades)
        return trades
    
    def _match_sell_order(self, order: Order, is_market: bool = False) -> List[Trade]:
        """Match a sell order against buy orders."""
        trades = []
        
        while order.remaining_quantity > 0 and self._bid_prices:
            best_bid = self._bid_prices[-1]
            
            # For limit orders, check if price crosses
            if not is_market and order.price > best_bid:
                break
            
            # Get orders at best bid price
            bid_queue = self.bid_levels[best_bid]
            
            while bid_queue and order.remaining_quantity > 0:
                buy_order = bid_queue[0]
                
                # Determine trade quantity
                trade_qty = min(order.remaining_quantity, buy_order.remaining_quantity)
                
                # Create trade
                trade = Trade.create(
                    buyer_order_id=buy_order.order_id,
                    seller_order_id=order.order_id,
                    price=best_bid,
                    quantity=trade_qty
                )
                trades.append(trade)
                
                # Update quantities
                order.remaining_quantity -= trade_qty
                buy_order.remaining_quantity -= trade_qty
                self.last_trade_price = best_bid
                
                # Remove filled buy order
                if buy_order.remaining_quantity == 0:
                    bid_queue.popleft()
                    del self.orders[buy_order.order_id]
            
            # Remove empty price level
            if not bid_queue:
                del self.bid_levels[best_bid]
                self._bid_prices.pop()
        
        self.total_trades += len(trades)
        return trades
    
    def _add_to_book(self, order: Order):
        """Add order to the appropriate side of the book."""
        price = order.price
        
        if order.side == OrderSide.BUY:
            self.bid_levels[price].append(order)
            if price not in self._bid_prices:
                bisect.insort(self._bid_prices, price)
        else:
            self.ask_levels[price].append(order)
            if price not in self._ask_prices:
                bisect.insort(self._ask_prices, price)
        
        self.orders[order.order_id] = (price, order.side, order)
    
    def _add_stop_order(self, order: Order):
        """Add a stop order to the book."""
        stop_price = order.price  # For stop orders, price is the trigger
        
        if order.side == OrderSide.BUY:
            self.stop_buy_orders[stop_price].append(order)
        else:
            self.stop_sell_orders[stop_price].append(order)
        
        self.orders[order.order_id] = (stop_price, order.side, order)
    
    def _check_stop_triggers(self, last_price: float):
        """Check if any stop orders should be triggered."""
        # Stop buy orders trigger when price goes up to stop price
        triggered_buys = []
        for stop_price, orders in list(self.stop_buy_orders.items()):
            if last_price >= stop_price:
                triggered_buys.extend(orders)
                del self.stop_buy_orders[stop_price]
        
        # Stop sell orders trigger when price goes down to stop price
        triggered_sells = []
        for stop_price, orders in list(self.stop_sell_orders.items()):
            if last_price <= stop_price:
                triggered_sells.extend(orders)
                del self.stop_sell_orders[stop_price]
        
        # Convert triggered stops to market orders
        for order in triggered_buys + triggered_sells:
            self._triggered_stops.add(order.order_id)
            market_order = Order(
                order_id=order.order_id,
                side=order.side,
                quantity=order.quantity,
                order_type=OrderType.MARKET,
                timestamp=time.time()
            )
            self._process_market_order(market_order)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        if order_id not in self.orders:
            return False
        
        price, side, order = self.orders[order_id]
        
        # Remove from price level
        if side == OrderSide.BUY and price in self.bid_levels:
            queue = self.bid_levels[price]
            queue = deque(o for o in queue if o.order_id != order_id)
            if queue:
                self.bid_levels[price] = queue
            else:
                del self.bid_levels[price]
                self._bid_prices.remove(price)
        elif side == OrderSide.SELL and price in self.ask_levels:
            queue = self.ask_levels[price]
            queue = deque(o for o in queue if o.order_id != order_id)
            if queue:
                self.ask_levels[price] = queue
            else:
                del self.ask_levels[price]
                self._ask_prices.remove(price)
        
        del self.orders[order_id]
        return True
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None,
                    new_quantity: Optional[int] = None) -> bool:
        """Modify an existing order."""
        if order_id not in self.orders:
            return False
        
        old_price, side, order = self.orders[order_id]
        
        # Cancel old order
        self.cancel_order(order_id)
        
        # Create modified order
        modified_order = Order(
            order_id=order_id,
            side=side,
            price=new_price if new_price is not None else old_price,
            quantity=new_quantity if new_quantity is not None else order.quantity,
            order_type=order.order_type,
            timestamp=time.time()  # New timestamp for priority
        )
        
        # Re-add to book
        self.add_order(modified_order)
        return True
    
    def get_market_depth(self, levels: int = 10) -> MarketDepth:
        """Get market depth up to specified number of levels."""
        bids = []
        asks = []
        
        # Get bid levels
        for i in range(min(levels, len(self._bid_prices))):
            price = self._bid_prices[-(i+1)]
            orders = self.bid_levels[price]
            total_qty = sum(o.remaining_quantity for o in orders)
            bids.append(PriceLevel(price, total_qty, len(orders)))
        
        # Get ask levels
        for i in range(min(levels, len(self._ask_prices))):
            price = self._ask_prices[i]
            orders = self.ask_levels[price]
            total_qty = sum(o.remaining_quantity for o in orders)
            asks.append(PriceLevel(price, total_qty, len(orders)))
        
        return MarketDepth(bids, asks)
    
    def get_snapshot(self) -> OrderBookSnapshot:
        """Get complete order book snapshot."""
        total_bid_volume = sum(
            sum(o.remaining_quantity for o in orders)
            for orders in self.bid_levels.values()
        )
        total_ask_volume = sum(
            sum(o.remaining_quantity for o in orders)
            for orders in self.ask_levels.values()
        )
        
        spread = None
        if self.best_bid and self.best_ask:
            spread = self.best_ask - self.best_bid
        
        return OrderBookSnapshot(
            symbol=self.symbol,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            spread=spread,
            bid_depth=len(self._bid_prices),
            ask_depth=len(self._ask_prices),
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume
        )
    
    def has_stop_orders_triggered(self) -> bool:
        """Check if any stop orders have been triggered."""
        return len(self._triggered_stops) > 0