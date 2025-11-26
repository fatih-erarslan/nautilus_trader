"""Execution Engine for ultra-low latency order execution.

Converts signals to orders with pre-trade risk checks and
latency tracking. Target < 50ms end-to-end execution.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import aiohttp
from decimal import Decimal, ROUND_DOWN

from .order_manager import OrderManager, Order, OrderStatus, OrderType, TimeInForce
from .smart_router import SmartRouter, RouteDecision
from .slippage_controller import SlippageController

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal from strategy."""
    signal_id: str
    symbol: str
    action: str  # 'buy', 'sell', 'close'
    quantity: float
    urgency: str  # 'high', 'medium', 'low'
    strategy_id: str
    confidence: float  # 0-1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RiskCheck:
    """Pre-trade risk check result."""
    passed: bool
    checks: Dict[str, bool]
    reasons: List[str]
    adjusted_quantity: Optional[float] = None
    latency_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order: Optional[Order] = None
    error: Optional[str] = None
    latency_breakdown: Dict[str, float] = None
    total_latency_ms: float = 0.0


class ExecutionEngine:
    """High-performance execution engine with sub-50ms latency.
    
    Features:
    - Signal to order conversion
    - Pre-trade risk checks
    - Smart order routing
    - Slippage control
    - Comprehensive latency tracking
    - Async Alpaca REST API integration
    """
    
    def __init__(self, api_key: str, api_secret: str, 
                 base_url: str = "https://paper-api.alpaca.markets",
                 max_position_size: float = 10000,
                 max_order_value: float = 50000,
                 max_daily_trades: int = 100):
        """Initialize execution engine.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: API base URL
            max_position_size: Maximum position size per symbol
            max_order_value: Maximum order value
            max_daily_trades: Maximum trades per day
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
        # Risk limits
        self.max_position_size = max_position_size
        self.max_order_value = max_order_value
        self.max_daily_trades = max_daily_trades
        
        # Components
        self.order_manager = OrderManager(api_key, api_secret, base_url)
        self.smart_router = SmartRouter()
        self.slippage_controller = SlippageController()
        
        # HTTP session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Position tracking
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._daily_trades: Dict[str, int] = {}  # date -> count
        
        # Performance metrics
        self._metrics = {
            'signals_processed': 0,
            'orders_submitted': 0,
            'orders_rejected': 0,
            'avg_total_latency_ms': 0,
            'avg_risk_check_ms': 0,
            'avg_routing_ms': 0,
            'avg_submission_ms': 0
        }
    
    async def initialize(self):
        """Initialize HTTP session and components."""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=5, connect=1)
            connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.api_secret
                }
            )
        
        # Load current positions
        await self._load_positions()
    
    async def close(self):
        """Clean up resources."""
        if self._session:
            await self._session.close()
    
    async def execute_signal(self, signal: Signal) -> ExecutionResult:
        """Execute a trading signal with ultra-low latency.
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Execution result with latency breakdown
        """
        start_time = time.perf_counter()
        latency_breakdown = {}
        
        try:
            # Update metrics
            self._metrics['signals_processed'] += 1
            
            # Step 1: Pre-trade risk checks
            risk_start = time.perf_counter()
            risk_check = await self._perform_risk_checks(signal)
            latency_breakdown['risk_check_ms'] = (time.perf_counter() - risk_start) * 1000
            
            if not risk_check.passed:
                self._metrics['orders_rejected'] += 1
                return ExecutionResult(
                    success=False,
                    error=f"Risk check failed: {', '.join(risk_check.reasons)}",
                    latency_breakdown=latency_breakdown,
                    total_latency_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Step 2: Get market data and routing decision
            route_start = time.perf_counter()
            market_data = await self._get_market_data(signal.symbol)
            route_decision = await self.smart_router.route_order(
                signal=signal,
                market_data=market_data,
                quantity=risk_check.adjusted_quantity or signal.quantity
            )
            latency_breakdown['routing_ms'] = (time.perf_counter() - route_start) * 1000
            
            # Step 3: Apply slippage control
            slippage_start = time.perf_counter()
            adjusted_price = await self.slippage_controller.adjust_price(
                symbol=signal.symbol,
                side=signal.action,
                order_type=route_decision.order_type,
                base_price=route_decision.limit_price,
                urgency=signal.urgency,
                market_data=market_data
            )
            if adjusted_price:
                route_decision.limit_price = adjusted_price
            latency_breakdown['slippage_ms'] = (time.perf_counter() - slippage_start) * 1000
            
            # Step 4: Create order
            order = await self._create_order_from_signal(signal, route_decision, risk_check)
            
            # Step 5: Submit order to Alpaca
            submit_start = time.perf_counter()
            submitted_order = await self._submit_order(order)
            latency_breakdown['submission_ms'] = (time.perf_counter() - submit_start) * 1000
            
            if submitted_order:
                # Update metrics
                self._metrics['orders_submitted'] += 1
                self._update_latency_metrics(latency_breakdown)
                
                # Update position tracking
                await self._update_position(signal.symbol, signal.action, 
                                          risk_check.adjusted_quantity or signal.quantity)
                
                # Record submission latency
                submitted_order.submission_latency_ms = latency_breakdown['submission_ms']
                
                total_latency = (time.perf_counter() - start_time) * 1000
                logger.info(f"Order executed in {total_latency:.1f}ms - "
                          f"Risk: {latency_breakdown['risk_check_ms']:.1f}ms, "
                          f"Route: {latency_breakdown['routing_ms']:.1f}ms, "
                          f"Submit: {latency_breakdown['submission_ms']:.1f}ms")
                
                return ExecutionResult(
                    success=True,
                    order=submitted_order,
                    latency_breakdown=latency_breakdown,
                    total_latency_ms=total_latency
                )
            else:
                return ExecutionResult(
                    success=False,
                    error="Failed to submit order to Alpaca",
                    latency_breakdown=latency_breakdown,
                    total_latency_ms=(time.perf_counter() - start_time) * 1000
                )
            
        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=str(e),
                latency_breakdown=latency_breakdown,
                total_latency_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def _perform_risk_checks(self, signal: Signal) -> RiskCheck:
        """Perform pre-trade risk checks.
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk check result
        """
        start_time = time.perf_counter()
        checks = {}
        reasons = []
        adjusted_quantity = signal.quantity
        
        # Check 1: Daily trade limit
        today = datetime.utcnow().date().isoformat()
        daily_count = self._daily_trades.get(today, 0)
        checks['daily_limit'] = daily_count < self.max_daily_trades
        if not checks['daily_limit']:
            reasons.append(f"Daily trade limit reached ({self.max_daily_trades})")
        
        # Check 2: Position size limit
        current_position = self._positions.get(signal.symbol, 0)
        if signal.action == 'buy':
            new_position = current_position + signal.quantity
        else:
            new_position = current_position - signal.quantity
        
        checks['position_size'] = abs(new_position) <= self.max_position_size
        if not checks['position_size']:
            # Adjust quantity to fit within limit
            if signal.action == 'buy':
                max_buy = self.max_position_size - current_position
                adjusted_quantity = max(0, min(signal.quantity, max_buy))
            else:
                max_sell = current_position + self.max_position_size
                adjusted_quantity = max(0, min(signal.quantity, max_sell))
            
            if adjusted_quantity == 0:
                reasons.append(f"Position size limit exceeded")
            else:
                logger.info(f"Adjusted quantity from {signal.quantity} to {adjusted_quantity}")
        
        # Check 3: Order value limit
        if signal.target_price:
            order_value = adjusted_quantity * signal.target_price
        else:
            # Use a conservative estimate if no target price
            order_value = adjusted_quantity * 1000  # Placeholder
        
        checks['order_value'] = order_value <= self.max_order_value
        if not checks['order_value']:
            reasons.append(f"Order value ${order_value:.2f} exceeds limit ${self.max_order_value}")
        
        # Check 4: Symbol validity (basic check)
        checks['symbol_valid'] = len(signal.symbol) > 0 and signal.symbol.isalnum()
        if not checks['symbol_valid']:
            reasons.append(f"Invalid symbol: {signal.symbol}")
        
        # Check 5: Quantity validity
        checks['quantity_valid'] = adjusted_quantity > 0
        if not checks['quantity_valid']:
            reasons.append(f"Invalid quantity: {adjusted_quantity}")
        
        # Overall pass/fail
        passed = all(checks.values()) and len(reasons) == 0
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RiskCheck(
            passed=passed,
            checks=checks,
            reasons=reasons,
            adjusted_quantity=adjusted_quantity if adjusted_quantity != signal.quantity else None,
            latency_ms=latency_ms
        )
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dictionary
        """
        try:
            url = f"{self.base_url}/v2/stocks/{symbol}/quotes/latest"
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'bid': float(data['quote']['bp']),
                        'ask': float(data['quote']['ap']),
                        'bid_size': int(data['quote']['bs']),
                        'ask_size': int(data['quote']['as']),
                        'last': float(data['quote']['ap']),  # Use ask as last
                        'spread': float(data['quote']['ap']) - float(data['quote']['bp']),
                        'timestamp': data['quote']['t']
                    }
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
        
        # Return empty data on error
        return {
            'bid': 0,
            'ask': 0,
            'bid_size': 0,
            'ask_size': 0,
            'last': 0,
            'spread': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _create_order_from_signal(self, signal: Signal, route: RouteDecision, 
                                       risk_check: RiskCheck) -> Order:
        """Create order object from signal and routing decision.
        
        Args:
            signal: Trading signal
            route: Routing decision
            risk_check: Risk check result
            
        Returns:
            Order object
        """
        # Generate client order ID
        client_order_id = f"{signal.strategy_id}_{signal.signal_id}_{int(time.time() * 1000)}"
        
        # Use adjusted quantity if available
        quantity = risk_check.adjusted_quantity or signal.quantity
        
        # Create order
        order = Order(
            order_id="",  # Will be assigned by Alpaca
            client_order_id=client_order_id,
            symbol=signal.symbol,
            side=signal.action,
            qty=quantity,
            order_type=route.order_type,
            time_in_force=route.time_in_force,
            limit_price=route.limit_price,
            stop_price=route.stop_price,
            extended_hours=route.extended_hours,
            signal_id=signal.signal_id,
            strategy_id=signal.strategy_id,
            notes=f"Signal confidence: {signal.confidence:.2f}"
        )
        
        # Track order
        await self.order_manager.create_order(order)
        
        return order
    
    async def _submit_order(self, order: Order) -> Optional[Order]:
        """Submit order to Alpaca.
        
        Args:
            order: Order to submit
            
        Returns:
            Updated order or None on failure
        """
        try:
            # Build order request
            order_data = {
                'symbol': order.symbol,
                'qty': str(order.qty),
                'side': order.side,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force.value,
                'client_order_id': order.client_order_id
            }
            
            # Add price fields
            if order.limit_price is not None:
                order_data['limit_price'] = str(order.limit_price)
            if order.stop_price is not None:
                order_data['stop_price'] = str(order.stop_price)
            
            # Extended hours
            if order.extended_hours:
                order_data['extended_hours'] = True
            
            # Submit order
            url = f"{self.base_url}/v2/orders"
            async with self._session.post(url, json=order_data) as response:
                if response.status in (200, 201):
                    alpaca_order = await response.json()
                    
                    # Update order with Alpaca response
                    order.submitted_at = datetime.utcnow()
                    updated_order = await self.order_manager.update_order_status(
                        order.client_order_id, alpaca_order
                    )
                    
                    logger.info(f"Order submitted: {order.client_order_id} -> {alpaca_order['id']}")
                    return updated_order
                else:
                    error_text = await response.text()
                    logger.error(f"Order submission failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Order submission error: {e}", exc_info=True)
            return None
    
    async def _load_positions(self):
        """Load current positions from Alpaca."""
        try:
            url = f"{self.base_url}/v2/positions"
            async with self._session.get(url) as response:
                if response.status == 200:
                    positions = await response.json()
                    self._positions.clear()
                    for pos in positions:
                        self._positions[pos['symbol']] = float(pos['qty'])
                    logger.info(f"Loaded {len(self._positions)} positions")
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
    
    async def _update_position(self, symbol: str, side: str, quantity: float):
        """Update position tracking.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
        """
        current = self._positions.get(symbol, 0)
        if side == 'buy':
            self._positions[symbol] = current + quantity
        else:
            self._positions[symbol] = current - quantity
        
        # Update daily trade count
        today = datetime.utcnow().date().isoformat()
        self._daily_trades[today] = self._daily_trades.get(today, 0) + 1
    
    def _update_latency_metrics(self, latency_breakdown: Dict[str, float]):
        """Update latency metrics.
        
        Args:
            latency_breakdown: Latency breakdown by component
        """
        count = self._metrics['orders_submitted']
        
        # Update averages
        for key in ['risk_check_ms', 'routing_ms', 'submission_ms']:
            if key in latency_breakdown:
                metric_key = f'avg_{key}'
                current_avg = self._metrics.get(metric_key, 0)
                new_value = latency_breakdown[key]
                
                if count == 1:
                    self._metrics[metric_key] = new_value
                else:
                    self._metrics[metric_key] = (
                        (current_avg * (count - 1) + new_value) / count
                    )
        
        # Update total latency
        total = sum(latency_breakdown.values())
        current_avg = self._metrics['avg_total_latency_ms']
        if count == 1:
            self._metrics['avg_total_latency_ms'] = total
        else:
            self._metrics['avg_total_latency_ms'] = (
                (current_avg * (count - 1) + total) / count
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()
        metrics['order_manager_metrics'] = self.order_manager.get_metrics()
        metrics['slippage_metrics'] = self.slippage_controller.get_metrics()
        return metrics
    
    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            True if cancellation submitted
        """
        order = await self.order_manager.get_order(client_order_id)
        if not order or not order.alpaca_order_id:
            return False
        
        try:
            url = f"{self.base_url}/v2/orders/{order.alpaca_order_id}"
            async with self._session.delete(url) as response:
                if response.status in (200, 204):
                    logger.info(f"Order cancelled: {client_order_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Cancel failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False
    
    async def replace_order(self, client_order_id: str, new_qty: Optional[float] = None,
                           new_limit_price: Optional[float] = None) -> bool:
        """Replace an order.
        
        Args:
            client_order_id: Client order ID
            new_qty: New quantity (optional)
            new_limit_price: New limit price (optional)
            
        Returns:
            True if replacement submitted
        """
        order = await self.order_manager.get_order(client_order_id)
        if not order or not order.alpaca_order_id:
            return False
        
        try:
            replace_data = {}
            if new_qty is not None:
                replace_data['qty'] = str(new_qty)
            if new_limit_price is not None:
                replace_data['limit_price'] = str(new_limit_price)
            
            url = f"{self.base_url}/v2/orders/{order.alpaca_order_id}"
            async with self._session.patch(url, json=replace_data) as response:
                if response.status == 200:
                    alpaca_order = await response.json()
                    await self.order_manager.update_order_status(
                        client_order_id, alpaca_order
                    )
                    logger.info(f"Order replaced: {client_order_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Replace failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Replace error: {e}")
            return False
