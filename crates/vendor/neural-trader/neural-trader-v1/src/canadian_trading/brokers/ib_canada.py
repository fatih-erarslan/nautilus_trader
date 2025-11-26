"""
Interactive Brokers Canada Integration Module

Production-ready implementation for Interactive Brokers Canada with:
- Async/await pattern for high performance
- Comprehensive error handling and automatic reconnection
- Support for CAD/USD currencies and TSX/US markets
- Real-time market data streaming
- Portfolio tracking and management
- Order execution with advanced order types
- Risk management integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict
import socket
import time

try:
    from ib_insync import (
        IB, Stock, Order, MarketOrder, LimitOrder, StopOrder, 
        Contract, Trade, Position, AccountValue, PortfolioItem,
        BarData, Ticker, OrderStatus, util
    )
except ImportError:
    raise ImportError("Please install ib_insync: pip install ib_insync>=0.9.86")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class OrderType(Enum):
    """Supported order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MARKET_ON_CLOSE = "market_on_close"


@dataclass
class ConnectionConfig:
    """IB Gateway/TWS connection configuration"""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for TWS paper, 7496 for TWS live, 4002 for Gateway paper, 4001 for Gateway live
    client_id: int = 1
    account: Optional[str] = None
    timeout: int = 30
    max_retries: int = 5
    retry_delay: int = 5
    is_paper: bool = True  # Set to False for live trading


@dataclass
class MarketDataConfig:
    """Market data streaming configuration"""
    snapshot: bool = False
    regulatory_snapshot: bool = False
    generic_tick_list: str = ""  # Comma-separated list of generic tick types
    update_interval: float = 0.5  # Seconds between updates


@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    contract: Contract
    position: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    account: str
    currency: str
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class OrderInfo:
    """Order information with tracking"""
    order_id: int
    contract: Contract
    order: Order
    status: OrderStatus
    filled: float = 0.0
    remaining: float = 0.0
    avg_fill_price: float = 0.0
    last_fill_time: Optional[datetime] = None
    commission: float = 0.0
    realized_pnl: float = 0.0
    parent_id: Optional[int] = None
    client_id: int = 0


class IBCanadaClient:
    """
    Production-ready Interactive Brokers Canada client
    
    Features:
    - Automatic connection management with retry logic
    - Real-time market data streaming
    - Order execution with multiple order types
    - Portfolio and position tracking
    - Error handling and recovery
    - Support for Canadian (TSX) and US markets
    """
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        """
        Initialize IB Canada client
        
        Args:
            config: Connection configuration (uses defaults if None)
        """
        self.config = config or ConnectionConfig()
        self.ib = IB()
        self.state = ConnectionState.DISCONNECTED
        self._positions: Dict[str, PositionInfo] = {}
        self._orders: Dict[int, OrderInfo] = {}
        self._market_data_subscriptions: Dict[Contract, Ticker] = {}
        self._account_values: Dict[str, AccountValue] = {}
        self._portfolio_items: Dict[str, PortfolioItem] = {}
        self._connection_lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._last_connection_time: Optional[datetime] = None
        self._connection_attempts = 0
        
        # Set up IB event handlers
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Set up IB event handlers"""
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        self.ib.positionEvent += self._on_position
        self.ib.accountValueEvent += self._on_account_value
        self.ib.updatePortfolioEvent += self._on_portfolio_update
        
    async def connect(self) -> bool:
        """
        Establish connection to IB Gateway/TWS
        
        Returns:
            bool: True if connected successfully
        """
        async with self._connection_lock:
            if self.state == ConnectionState.CONNECTED:
                logger.info("Already connected to IB")
                return True
                
            self.state = ConnectionState.CONNECTING
            self._connection_attempts = 0
            
            while self._connection_attempts < self.config.max_retries:
                try:
                    self._connection_attempts += 1
                    logger.info(f"Connecting to IB at {self.config.host}:{self.config.port} "
                              f"(attempt {self._connection_attempts}/{self.config.max_retries})")
                    
                    await self.ib.connectAsync(
                        host=self.config.host,
                        port=self.config.port,
                        clientId=self.config.client_id,
                        timeout=self.config.timeout
                    )
                    
                    # Request account info
                    if self.config.account:
                        self.ib.reqAccountUpdates(True, self.config.account)
                    else:
                        # Get all accounts
                        accounts = self.ib.managedAccounts()
                        if accounts:
                            self.config.account = accounts[0]
                            self.ib.reqAccountUpdates(True, self.config.account)
                    
                    # Request positions
                    self.ib.reqPositions()
                    
                    self.state = ConnectionState.CONNECTED
                    self._last_connection_time = datetime.now()
                    logger.info(f"Successfully connected to IB. Account: {self.config.account}")
                    
                    # Emit connected event
                    await self._emit_event('connected', {'account': self.config.account})
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Connection attempt {self._connection_attempts} failed: {e}")
                    
                    if self._connection_attempts < self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        self.state = ConnectionState.ERROR
                        raise ConnectionError(f"Failed to connect after {self.config.max_retries} attempts: {e}")
            
            return False
    
    async def disconnect(self):
        """Disconnect from IB Gateway/TWS"""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
            
        if self.state == ConnectionState.CONNECTED:
            logger.info("Disconnecting from IB")
            
            # Cancel all market data subscriptions
            for contract, ticker in self._market_data_subscriptions.items():
                self.ib.cancelMktData(contract)
                
            self._market_data_subscriptions.clear()
            
            # Disconnect
            self.ib.disconnect()
            self.state = ConnectionState.DISCONNECTED
            
            await self._emit_event('disconnected', {})
    
    async def ensure_connected(self) -> bool:
        """Ensure client is connected, attempt reconnection if needed"""
        if self.state == ConnectionState.CONNECTED and self.ib.isConnected():
            return True
            
        logger.warning("Not connected, attempting to reconnect")
        return await self.connect()
    
    def create_canadian_stock(self, symbol: str, currency: str = "CAD") -> Stock:
        """
        Create a Canadian stock contract
        
        Args:
            symbol: Stock symbol (without exchange suffix)
            currency: Currency (CAD or USD)
            
        Returns:
            Stock contract
        """
        # Remove exchange suffix if present
        if '.' in symbol:
            symbol = symbol.split('.')[0]
            
        # Determine exchange based on currency
        if currency == "CAD":
            exchange = "TSE"  # Toronto Stock Exchange
        else:
            exchange = "SMART"  # Smart routing for US stocks
            
        return Stock(symbol=symbol, exchange=exchange, currency=currency)
    
    def create_us_stock(self, symbol: str) -> Stock:
        """Create a US stock contract"""
        return Stock(symbol=symbol, exchange="SMART", currency="USD")
    
    async def get_contract_details(self, contract: Contract) -> List[Contract]:
        """
        Get detailed contract information
        
        Args:
            contract: Contract to query
            
        Returns:
            List of qualified contracts
        """
        await self.ensure_connected()
        
        try:
            contracts = await self.ib.qualifyContractsAsync(contract)
            return contracts
        except Exception as e:
            logger.error(f"Failed to get contract details: {e}")
            return []
    
    async def get_market_data(self, 
                            contract: Contract,
                            snapshot: bool = False,
                            streaming: bool = True) -> Optional[Ticker]:
        """
        Get market data for a contract
        
        Args:
            contract: Contract to get data for
            snapshot: Get snapshot instead of streaming
            streaming: Enable streaming updates
            
        Returns:
            Ticker object with market data
        """
        await self.ensure_connected()
        
        try:
            # Check if already subscribed
            if contract in self._market_data_subscriptions and not snapshot:
                return self._market_data_subscriptions[contract]
            
            # Request market data
            ticker = self.ib.reqMktData(
                contract,
                genericTickList="",
                snapshot=snapshot,
                regulatorySnapshot=False,
                mktDataOptions=[]
            )
            
            if streaming and not snapshot:
                self._market_data_subscriptions[contract] = ticker
                
            # Wait for data
            await asyncio.sleep(0.5)
            
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to get market data for {contract.symbol}: {e}")
            return None
    
    async def get_historical_data(self,
                                contract: Contract,
                                duration: str = "1 D",
                                bar_size: str = "1 min",
                                what_to_show: str = "TRADES",
                                use_rth: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical data for a contract
        
        Args:
            contract: Contract to get data for
            duration: Time duration (e.g., "1 D", "1 W", "1 M")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)
            use_rth: Use regular trading hours only
            
        Returns:
            DataFrame with historical data
        """
        await self.ensure_connected()
        
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1
            )
            
            if bars:
                df = util.df(bars)
                df['symbol'] = contract.symbol
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {contract.symbol}: {e}")
            return None
    
    async def place_order(self,
                        contract: Contract,
                        order_type: OrderType,
                        action: str,
                        quantity: float,
                        price: Optional[float] = None,
                        stop_price: Optional[float] = None,
                        trailing_amount: Optional[float] = None,
                        tif: str = "DAY",
                        outside_rth: bool = False,
                        **kwargs) -> Optional[Trade]:
        """
        Place an order
        
        Args:
            contract: Contract to trade
            order_type: Type of order
            action: BUY or SELL
            quantity: Number of shares/contracts
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            trailing_amount: Trailing amount (for trailing stop)
            tif: Time in force (DAY, GTC, IOC, FOK)
            outside_rth: Allow trading outside regular hours
            **kwargs: Additional order attributes
            
        Returns:
            Trade object if successful
        """
        await self.ensure_connected()
        
        try:
            # Create order based on type
            if order_type == OrderType.MARKET:
                order = MarketOrder(action=action, totalQuantity=quantity)
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Limit price required for limit orders")
                order = LimitOrder(action=action, totalQuantity=quantity, lmtPrice=price)
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    raise ValueError("Stop price required for stop orders")
                order = StopOrder(action=action, totalQuantity=quantity, stopPrice=stop_price)
            elif order_type == OrderType.STOP_LIMIT:
                if price is None or stop_price is None:
                    raise ValueError("Both limit and stop prices required for stop-limit orders")
                order = Order()
                order.action = action
                order.orderType = "STP LMT"
                order.totalQuantity = quantity
                order.lmtPrice = price
                order.auxPrice = stop_price
            elif order_type == OrderType.TRAILING_STOP:
                if trailing_amount is None:
                    raise ValueError("Trailing amount required for trailing stop orders")
                order = Order()
                order.action = action
                order.orderType = "TRAIL"
                order.totalQuantity = quantity
                order.trailingPercent = trailing_amount
            elif order_type == OrderType.MARKET_ON_CLOSE:
                order = Order()
                order.action = action
                order.orderType = "MOC"
                order.totalQuantity = quantity
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Set common order attributes
            order.tif = tif
            order.outsideRth = outside_rth
            
            # Apply any additional attributes
            for key, value in kwargs.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            # Place the order
            trade = self.ib.placeOrder(contract, order)
            
            # Store order info
            order_info = OrderInfo(
                order_id=trade.order.orderId,
                contract=contract,
                order=order,
                status=trade.orderStatus,
                client_id=self.config.client_id
            )
            self._orders[trade.order.orderId] = order_info
            
            # Wait for order to be acknowledged
            await asyncio.sleep(0.1)
            
            # Emit order placed event
            await self._emit_event('order_placed', {
                'order_id': trade.order.orderId,
                'symbol': contract.symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type.value
            })
            
            logger.info(f"Order placed: {trade.order.orderId} - {action} {quantity} {contract.symbol}")
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            await self._emit_event('order_error', {'error': str(e)})
            return None
    
    async def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancel request was sent
        """
        await self.ensure_connected()
        
        try:
            if order_id in self._orders:
                order_info = self._orders[order_id]
                self.ib.cancelOrder(order_info.order)
                
                logger.info(f"Cancel request sent for order {order_id}")
                
                await self._emit_event('order_cancel_requested', {'order_id': order_id})
                
                return True
            else:
                logger.warning(f"Order {order_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def modify_order(self,
                         order_id: int,
                         quantity: Optional[float] = None,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity (optional)
            price: New limit price (optional)
            stop_price: New stop price (optional)
            **kwargs: Additional order attributes to modify
            
        Returns:
            bool: True if modification was successful
        """
        await self.ensure_connected()
        
        try:
            if order_id not in self._orders:
                logger.warning(f"Order {order_id} not found")
                return False
                
            order_info = self._orders[order_id]
            order = order_info.order
            
            # Modify order attributes
            if quantity is not None:
                order.totalQuantity = quantity
            if price is not None and hasattr(order, 'lmtPrice'):
                order.lmtPrice = price
            if stop_price is not None and hasattr(order, 'auxPrice'):
                order.auxPrice = stop_price
                
            # Apply additional modifications
            for key, value in kwargs.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            # Submit modification
            trade = self.ib.placeOrder(order_info.contract, order)
            
            logger.info(f"Order {order_id} modified")
            
            await self._emit_event('order_modified', {
                'order_id': order_id,
                'modifications': {
                    'quantity': quantity,
                    'price': price,
                    'stop_price': stop_price,
                    **kwargs
                }
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False
    
    def get_positions(self) -> Dict[str, PositionInfo]:
        """Get all current positions"""
        return self._positions.copy()
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for a specific symbol"""
        return self._positions.get(symbol)
    
    def get_orders(self, active_only: bool = True) -> Dict[int, OrderInfo]:
        """
        Get orders
        
        Args:
            active_only: Only return active orders
            
        Returns:
            Dictionary of orders
        """
        if active_only:
            return {
                order_id: order_info
                for order_id, order_info in self._orders.items()
                if order_info.status.status not in ['Filled', 'Cancelled', 'Inactive']
            }
        return self._orders.copy()
    
    def get_order(self, order_id: int) -> Optional[OrderInfo]:
        """Get specific order info"""
        return self._orders.get(order_id)
    
    def get_account_values(self) -> Dict[str, AccountValue]:
        """Get account values"""
        return self._account_values.copy()
    
    def get_account_value(self, key: str, currency: str = "BASE") -> Optional[float]:
        """
        Get specific account value
        
        Args:
            key: Value key (e.g., 'NetLiquidation', 'TotalCashValue')
            currency: Currency (BASE, CAD, USD)
            
        Returns:
            Value if found
        """
        lookup_key = f"{key}-{currency}"
        if lookup_key in self._account_values:
            return float(self._account_values[lookup_key].value)
        return None
    
    def get_portfolio_items(self) -> Dict[str, PortfolioItem]:
        """Get portfolio items"""
        return self._portfolio_items.copy()
    
    async def get_pnl(self, account: Optional[str] = None) -> Dict[str, float]:
        """
        Get P&L information
        
        Args:
            account: Account to get P&L for (uses default if None)
            
        Returns:
            Dictionary with P&L values
        """
        await self.ensure_connected()
        
        account = account or self.config.account
        
        try:
            # Request P&L
            pnl = self.ib.reqPnL(account)
            await asyncio.sleep(0.5)  # Wait for data
            
            return {
                'daily_pnl': pnl.dailyPnL or 0.0,
                'unrealized_pnl': pnl.unrealizedPnL or 0.0,
                'realized_pnl': pnl.realizedPnL or 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get P&L: {e}")
            return {
                'daily_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
    
    # Event handlers
    def _on_connected(self):
        """Handle connected event"""
        logger.info("IB connected event received")
        self.state = ConnectionState.CONNECTED
        
    def _on_disconnected(self):
        """Handle disconnected event"""
        logger.warning("IB disconnected event received")
        self.state = ConnectionState.DISCONNECTED
        
        # Start reconnection task if not already running
        if not self._reconnect_task or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._auto_reconnect())
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Optional[Contract]):
        """Handle error events"""
        if errorCode in [1100, 1101, 1102]:  # Connection-related errors
            logger.warning(f"Connection error {errorCode}: {errorString}")
        elif errorCode == 2104:  # Market data farm connected
            logger.info(f"Market data farm connected: {errorString}")
        elif errorCode == 2106:  # HMDS data farm connected
            logger.info(f"Historical data farm connected: {errorString}")
        else:
            logger.error(f"IB Error - ReqId: {reqId}, Code: {errorCode}, Message: {errorString}")
            
        # Emit error event
        asyncio.create_task(self._emit_event('error', {
            'req_id': reqId,
            'error_code': errorCode,
            'error_string': errorString,
            'contract': contract
        }))
    
    def _on_order_status(self, trade: Trade):
        """Handle order status updates"""
        order_id = trade.order.orderId
        
        if order_id in self._orders:
            order_info = self._orders[order_id]
            order_info.status = trade.orderStatus
            order_info.filled = trade.orderStatus.filled
            order_info.remaining = trade.orderStatus.remaining
            order_info.avg_fill_price = trade.orderStatus.avgFillPrice
            
            if trade.fills:
                last_fill = trade.fills[-1]
                order_info.last_fill_time = last_fill.time
                order_info.commission = sum(fill.commissionReport.commission for fill in trade.fills if fill.commissionReport)
                order_info.realized_pnl = sum(fill.commissionReport.realizedPNL for fill in trade.fills if fill.commissionReport)
            
            logger.info(f"Order {order_id} status: {trade.orderStatus.status} - "
                       f"Filled: {trade.orderStatus.filled}/{trade.order.totalQuantity}")
            
            # Emit order status event
            asyncio.create_task(self._emit_event('order_status', {
                'order_id': order_id,
                'status': trade.orderStatus.status,
                'filled': trade.orderStatus.filled,
                'remaining': trade.orderStatus.remaining,
                'avg_fill_price': trade.orderStatus.avgFillPrice
            }))
    
    def _on_exec_details(self, trade: Trade, fill):
        """Handle execution details"""
        logger.info(f"Execution: {fill.contract.symbol} - {fill.execution.side} "
                   f"{fill.execution.shares} @ {fill.execution.price}")
        
        # Emit execution event
        asyncio.create_task(self._emit_event('execution', {
            'symbol': fill.contract.symbol,
            'side': fill.execution.side,
            'shares': fill.execution.shares,
            'price': fill.execution.price,
            'order_id': fill.execution.orderId,
            'exec_id': fill.execution.execId
        }))
    
    def _on_position(self, position: Position):
        """Handle position updates"""
        if position.position != 0:
            pos_info = PositionInfo(
                symbol=position.contract.symbol,
                contract=position.contract,
                position=position.position,
                avg_cost=position.avgCost,
                market_value=0.0,  # Will be updated from portfolio
                unrealized_pnl=0.0,  # Will be updated from portfolio
                realized_pnl=0.0,  # Will be updated from portfolio
                account=position.account,
                currency=position.contract.currency
            )
            self._positions[position.contract.symbol] = pos_info
        else:
            # Position closed
            if position.contract.symbol in self._positions:
                del self._positions[position.contract.symbol]
                
        # Emit position event
        asyncio.create_task(self._emit_event('position', {
            'symbol': position.contract.symbol,
            'position': position.position,
            'avg_cost': position.avgCost,
            'account': position.account
        }))
    
    def _on_account_value(self, account_value: AccountValue):
        """Handle account value updates"""
        key = f"{account_value.tag}-{account_value.currency}"
        self._account_values[key] = account_value
        
        # Log important values
        if account_value.tag in ['NetLiquidation', 'TotalCashValue', 'UnrealizedPnL']:
            logger.debug(f"Account {account_value.account} - {account_value.tag}: "
                        f"{account_value.value} {account_value.currency}")
    
    def _on_portfolio_update(self, portfolio_item: PortfolioItem):
        """Handle portfolio updates"""
        self._portfolio_items[portfolio_item.contract.symbol] = portfolio_item
        
        # Update position info with portfolio data
        if portfolio_item.contract.symbol in self._positions:
            pos_info = self._positions[portfolio_item.contract.symbol]
            pos_info.market_value = portfolio_item.marketValue
            pos_info.unrealized_pnl = portfolio_item.unrealizedPNL
            pos_info.realized_pnl = portfolio_item.realizedPNL
            pos_info.last_update = datetime.now()
    
    async def _auto_reconnect(self):
        """Automatic reconnection logic"""
        logger.info("Starting auto-reconnection...")
        
        while self.state != ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.retry_delay)
                
                if self.state != ConnectionState.CONNECTED:
                    logger.info("Attempting to reconnect...")
                    connected = await self.connect()
                    
                    if connected:
                        logger.info("Successfully reconnected")
                        
                        # Re-subscribe to market data
                        for contract in list(self._market_data_subscriptions.keys()):
                            await self.get_market_data(contract, streaming=True)
                            
                        break
                        
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
                
        logger.info("Auto-reconnection task completed")
    
    async def _emit_event(self, event_name: str, data: Dict[str, Any]):
        """Emit event to registered handlers"""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")
    
    def on(self, event_name: str, handler: Callable):
        """Register event handler"""
        self._event_handlers[event_name].append(handler)
        
    def off(self, event_name: str, handler: Callable):
        """Unregister event handler"""
        if event_name in self._event_handlers and handler in self._event_handlers[event_name]:
            self._event_handlers[event_name].remove(handler)
    
    # Risk management integration
    async def validate_order_risk(self,
                                contract: Contract,
                                action: str,
                                quantity: float,
                                order_type: OrderType,
                                price: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate order against risk parameters
        
        Args:
            contract: Contract to trade
            action: BUY or SELL
            quantity: Order quantity
            order_type: Type of order
            price: Order price (for limit orders)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Get current position
            position = self.get_position(contract.symbol)
            current_position = position.position if position else 0
            
            # Calculate new position
            new_position = current_position + (quantity if action == "BUY" else -quantity)
            
            # Get account value
            net_liquidation = self.get_account_value("NetLiquidation", "BASE")
            if not net_liquidation:
                return False, "Unable to get account value"
            
            # Get market data for position value calculation
            ticker = await self.get_market_data(contract, snapshot=True)
            if not ticker or not ticker.marketPrice():
                return False, "Unable to get market price"
            
            market_price = ticker.marketPrice()
            position_value = abs(new_position * market_price)
            
            # Check position size limits (example: 25% of portfolio)
            max_position_value = net_liquidation * 0.25
            if position_value > max_position_value:
                return False, f"Position value ${position_value:.2f} exceeds limit of ${max_position_value:.2f}"
            
            # Check daily loss limit (example: 2% of portfolio)
            daily_pnl = (await self.get_pnl()).get('daily_pnl', 0)
            daily_loss_limit = net_liquidation * 0.02
            if daily_pnl < -daily_loss_limit:
                return False, f"Daily loss ${-daily_pnl:.2f} exceeds limit of ${daily_loss_limit:.2f}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return False, f"Risk validation error: {str(e)}"
    
    # Utility methods
    async def wait_for_order_fill(self,
                                order_id: int,
                                timeout: float = 30.0) -> Optional[OrderInfo]:
        """
        Wait for an order to be filled
        
        Args:
            order_id: Order ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            OrderInfo if filled, None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            order_info = self.get_order(order_id)
            
            if order_info and order_info.status.status == "Filled":
                return order_info
            elif order_info and order_info.status.status in ["Cancelled", "Inactive"]:
                return None
                
            await asyncio.sleep(0.1)
            
        return None
    
    async def close_position(self,
                           symbol: str,
                           order_type: OrderType = OrderType.MARKET) -> Optional[Trade]:
        """
        Close a position
        
        Args:
            symbol: Symbol to close
            order_type: Order type to use
            
        Returns:
            Trade object if successful
        """
        position = self.get_position(symbol)
        if not position or position.position == 0:
            logger.warning(f"No position to close for {symbol}")
            return None
            
        action = "SELL" if position.position > 0 else "BUY"
        quantity = abs(position.position)
        
        logger.info(f"Closing position: {action} {quantity} {symbol}")
        
        return await self.place_order(
            contract=position.contract,
            order_type=order_type,
            action=action,
            quantity=quantity
        )
    
    async def close_all_positions(self,
                                order_type: OrderType = OrderType.MARKET) -> List[Trade]:
        """
        Close all positions
        
        Args:
            order_type: Order type to use
            
        Returns:
            List of trades
        """
        trades = []
        positions = self.get_positions()
        
        for symbol, position in positions.items():
            if position.position != 0:
                trade = await self.close_position(symbol, order_type)
                if trade:
                    trades.append(trade)
                    
        return trades
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'state': self.state.value,
            'host': self.config.host,
            'port': self.config.port,
            'client_id': self.config.client_id,
            'account': self.config.account,
            'is_paper': self.config.is_paper,
            'connected_duration': (
                (datetime.now() - self._last_connection_time).total_seconds()
                if self._last_connection_time and self.state == ConnectionState.CONNECTED
                else 0
            ),
            'connection_attempts': self._connection_attempts
        }


# Example usage and testing
async def example_usage():
    """Example usage of IBCanadaClient"""
    
    # Create client with paper trading configuration
    config = ConnectionConfig(
        host="127.0.0.1",
        port=7497,  # TWS paper trading port
        client_id=1,
        is_paper=True
    )
    
    client = IBCanadaClient(config)
    
    # Register event handlers
    def on_order_status(data):
        print(f"Order status update: {data}")
        
    def on_position(data):
        print(f"Position update: {data}")
        
    client.on('order_status', on_order_status)
    client.on('position', on_position)
    
    try:
        # Connect
        connected = await client.connect()
        if not connected:
            print("Failed to connect")
            return
            
        print(f"Connected to IB. Account: {client.config.account}")
        
        # Get account info
        net_liq = client.get_account_value("NetLiquidation", "BASE")
        print(f"Net Liquidation Value: ${net_liq:,.2f}")
        
        # Create Canadian stock contract
        shop = client.create_canadian_stock("SHOP", "CAD")
        
        # Get market data
        ticker = await client.get_market_data(shop, snapshot=True)
        if ticker:
            print(f"SHOP.TO - Bid: {ticker.bid}, Ask: {ticker.ask}, Last: {ticker.last}")
        
        # Get historical data
        hist_data = await client.get_historical_data(shop, duration="5 D", bar_size="1 hour")
        if hist_data is not None:
            print(f"Historical data: {len(hist_data)} bars")
            print(hist_data.tail())
        
        # Example order (commented out for safety)
        # trade = await client.place_order(
        #     contract=shop,
        #     order_type=OrderType.LIMIT,
        #     action="BUY",
        #     quantity=10,
        #     price=100.00
        # )
        
        # Get positions
        positions = client.get_positions()
        print(f"Positions: {len(positions)}")
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos.position} @ {pos.avg_cost:.2f} "
                  f"(P&L: ${pos.unrealized_pnl:.2f})")
        
        # Get orders
        orders = client.get_orders(active_only=True)
        print(f"Active orders: {len(orders)}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Disconnect
        await client.disconnect()
        print("Disconnected from IB")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())