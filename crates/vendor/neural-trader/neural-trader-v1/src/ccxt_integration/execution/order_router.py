"""
Smart Order Router for CCXT

Implements intelligent order routing across multiple exchanges with
best execution, slippage minimization, and order splitting capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

from ..interfaces.ccxt_interface import CCXTInterface
from ..core.client_manager import ClientManager

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Order routing strategies"""
    BEST_PRICE = "best_price"
    LOWEST_FEE = "lowest_fee"
    HIGHEST_LIQUIDITY = "highest_liquidity"
    FASTEST_EXECUTION = "fastest_execution"
    SMART_ROUTING = "smart_routing"  # Combines all factors


@dataclass
class OrderRequest:
    """Represents an order routing request"""
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    order_type: str = 'market'
    price: Optional[float] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.SMART_ROUTING
    max_slippage: float = 0.01  # 1%
    time_limit_ms: Optional[int] = None
    preferred_exchanges: Optional[List[str]] = None
    excluded_exchanges: Optional[List[str]] = None
    split_order: bool = True
    min_split_size: Optional[float] = None


@dataclass
class ExecutionPlan:
    """Execution plan for an order"""
    exchange: str
    amount: float
    expected_price: float
    expected_fee: float
    liquidity_score: float
    execution_time_estimate_ms: int


@dataclass
class OrderResult:
    """Result of order execution"""
    order_id: str
    exchange: str
    symbol: str
    side: str
    amount: float
    filled: float
    price: float
    fee: float
    status: str
    timestamp: datetime
    raw_order: Dict[str, Any]


class OrderRouter:
    """
    Smart order router that finds the best execution venue and
    intelligently routes orders across multiple exchanges.
    """
    
    def __init__(self, client_manager: ClientManager):
        """
        Initialize the order router.
        
        Args:
            client_manager: ClientManager instance for exchange access
        """
        self.client_manager = client_manager
        self._orderbook_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_ms = 1000  # 1 second cache TTL
        self._last_cache_update: Dict[str, datetime] = {}
        
    async def route_order(self, request: OrderRequest) -> List[OrderResult]:
        """
        Route an order using the specified strategy.
        
        Args:
            request: OrderRequest with routing parameters
            
        Returns:
            List of OrderResult for executed orders
        """
        # Get execution plan
        execution_plans = await self._create_execution_plan(request)
        
        if not execution_plans:
            raise ValueError(f"No suitable exchange found for {request.symbol}")
            
        # Execute orders based on plan
        results = []
        
        for plan in execution_plans:
            try:
                result = await self._execute_on_exchange(plan, request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute on {plan.exchange}: {str(e)}")
                
                # If partial execution, continue with remaining
                if results:
                    continue
                else:
                    raise
                    
        return results
        
    async def _create_execution_plan(self, request: OrderRequest) -> List[ExecutionPlan]:
        """
        Create an optimal execution plan for the order.
        
        Args:
            request: OrderRequest
            
        Returns:
            List of ExecutionPlan
        """
        # Get candidate exchanges
        candidates = await self._get_candidate_exchanges(request)
        
        if not candidates:
            return []
            
        # Analyze each exchange
        exchange_analysis = []
        
        for exchange_name in candidates:
            try:
                analysis = await self._analyze_exchange(exchange_name, request)
                if analysis:
                    exchange_analysis.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze {exchange_name}: {str(e)}")
                continue
                
        if not exchange_analysis:
            return []
            
        # Apply routing strategy
        if request.routing_strategy == RoutingStrategy.BEST_PRICE:
            return self._route_by_best_price(exchange_analysis, request)
        elif request.routing_strategy == RoutingStrategy.LOWEST_FEE:
            return self._route_by_lowest_fee(exchange_analysis, request)
        elif request.routing_strategy == RoutingStrategy.HIGHEST_LIQUIDITY:
            return self._route_by_liquidity(exchange_analysis, request)
        elif request.routing_strategy == RoutingStrategy.FASTEST_EXECUTION:
            return self._route_by_speed(exchange_analysis, request)
        else:  # SMART_ROUTING
            return self._smart_route(exchange_analysis, request)
            
    async def _get_candidate_exchanges(self, request: OrderRequest) -> List[str]:
        """Get list of candidate exchanges for the order."""
        all_exchanges = list(self.client_manager.active_exchanges)
        
        # Apply filters
        if request.preferred_exchanges:
            all_exchanges = [e for e in all_exchanges if e in request.preferred_exchanges]
            
        if request.excluded_exchanges:
            all_exchanges = [e for e in all_exchanges if e not in request.excluded_exchanges]
            
        # Check symbol availability
        candidates = []
        
        for exchange_name in all_exchanges:
            client = await self.client_manager.get_client(exchange_name)
            if client:
                try:
                    markets = await client.get_markets()
                    if request.symbol in markets:
                        candidates.append(exchange_name)
                except:
                    continue
                    
        return candidates
        
    async def _analyze_exchange(
        self,
        exchange_name: str,
        request: OrderRequest
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze an exchange for order execution.
        
        Returns dictionary with:
        - exchange: str
        - price: float
        - liquidity: float
        - fee: float
        - response_time_ms: float
        """
        client = await self.client_manager.get_client(exchange_name)
        if not client:
            return None
            
        try:
            start_time = datetime.now()
            
            # Get orderbook
            orderbook = await self._get_cached_orderbook(client, request.symbol)
            
            # Calculate execution price and liquidity
            if request.side == 'buy':
                price, liquidity = self._calculate_buy_price(
                    orderbook['asks'],
                    request.amount
                )
            else:
                price, liquidity = self._calculate_sell_price(
                    orderbook['bids'],
                    request.amount
                )
                
            # Get trading fee
            fee = await self._get_trading_fee(client, request.symbol)
            
            # Calculate response time
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'exchange': exchange_name,
                'price': price,
                'liquidity': liquidity,
                'fee': fee,
                'response_time_ms': response_time_ms
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {exchange_name}: {str(e)}")
            return None
            
    async def _get_cached_orderbook(
        self,
        client: CCXTInterface,
        symbol: str
    ) -> Dict[str, Any]:
        """Get orderbook with caching."""
        cache_key = f"{client.config.name}_{symbol}"
        
        # Check cache
        if cache_key in self._orderbook_cache:
            last_update = self._last_cache_update.get(cache_key)
            if last_update:
                age_ms = (datetime.now() - last_update).total_seconds() * 1000
                if age_ms < self._cache_ttl_ms:
                    return self._orderbook_cache[cache_key]
                    
        # Fetch fresh orderbook
        orderbook = await client.get_orderbook(symbol, limit=100)
        
        # Update cache
        self._orderbook_cache[cache_key] = orderbook
        self._last_cache_update[cache_key] = datetime.now()
        
        return orderbook
        
    def _calculate_buy_price(
        self,
        asks: List[List[float]],
        amount: float
    ) -> Tuple[float, float]:
        """Calculate average buy price and available liquidity."""
        total_cost = 0
        total_amount = 0
        
        for ask_price, ask_amount in asks:
            if total_amount >= amount:
                break
                
            fill_amount = min(ask_amount, amount - total_amount)
            total_cost += fill_amount * ask_price
            total_amount += fill_amount
            
        if total_amount == 0:
            return float('inf'), 0
            
        avg_price = total_cost / total_amount
        liquidity = total_amount / amount if amount > 0 else 0
        
        return avg_price, liquidity
        
    def _calculate_sell_price(
        self,
        bids: List[List[float]],
        amount: float
    ) -> Tuple[float, float]:
        """Calculate average sell price and available liquidity."""
        total_proceeds = 0
        total_amount = 0
        
        for bid_price, bid_amount in bids:
            if total_amount >= amount:
                break
                
            fill_amount = min(bid_amount, amount - total_amount)
            total_proceeds += fill_amount * bid_price
            total_amount += fill_amount
            
        if total_amount == 0:
            return 0, 0
            
        avg_price = total_proceeds / total_amount
        liquidity = total_amount / amount if amount > 0 else 0
        
        return avg_price, liquidity
        
    async def _get_trading_fee(self, client: CCXTInterface, symbol: str) -> float:
        """Get trading fee for a symbol."""
        try:
            account_info = await client.get_account_info()
            fees = account_info.get('fees', {})
            
            # Try to get symbol-specific fee
            if symbol in fees:
                return fees[symbol].get('taker', 0.001)
                
            # Get default taker fee
            if 'trading' in fees:
                return fees['trading'].get('taker', 0.001)
                
            # Default fee
            return 0.001
            
        except:
            return 0.001  # Default 0.1%
            
    def _route_by_best_price(
        self,
        analysis: List[Dict[str, Any]],
        request: OrderRequest
    ) -> List[ExecutionPlan]:
        """Route by best price."""
        # Sort by price (ascending for buy, descending for sell)
        if request.side == 'buy':
            analysis.sort(key=lambda x: x['price'])
        else:
            analysis.sort(key=lambda x: x['price'], reverse=True)
            
        best = analysis[0]
        
        return [ExecutionPlan(
            exchange=best['exchange'],
            amount=request.amount,
            expected_price=best['price'],
            expected_fee=best['fee'] * request.amount * best['price'],
            liquidity_score=best['liquidity'],
            execution_time_estimate_ms=int(best['response_time_ms'])
        )]
        
    def _route_by_lowest_fee(
        self,
        analysis: List[Dict[str, Any]],
        request: OrderRequest
    ) -> List[ExecutionPlan]:
        """Route by lowest fee."""
        analysis.sort(key=lambda x: x['fee'])
        
        best = analysis[0]
        
        return [ExecutionPlan(
            exchange=best['exchange'],
            amount=request.amount,
            expected_price=best['price'],
            expected_fee=best['fee'] * request.amount * best['price'],
            liquidity_score=best['liquidity'],
            execution_time_estimate_ms=int(best['response_time_ms'])
        )]
        
    def _route_by_liquidity(
        self,
        analysis: List[Dict[str, Any]],
        request: OrderRequest
    ) -> List[ExecutionPlan]:
        """Route by highest liquidity."""
        analysis.sort(key=lambda x: x['liquidity'], reverse=True)
        
        best = analysis[0]
        
        return [ExecutionPlan(
            exchange=best['exchange'],
            amount=request.amount,
            expected_price=best['price'],
            expected_fee=best['fee'] * request.amount * best['price'],
            liquidity_score=best['liquidity'],
            execution_time_estimate_ms=int(best['response_time_ms'])
        )]
        
    def _route_by_speed(
        self,
        analysis: List[Dict[str, Any]],
        request: OrderRequest
    ) -> List[ExecutionPlan]:
        """Route by fastest execution."""
        analysis.sort(key=lambda x: x['response_time_ms'])
        
        best = analysis[0]
        
        return [ExecutionPlan(
            exchange=best['exchange'],
            amount=request.amount,
            expected_price=best['price'],
            expected_fee=best['fee'] * request.amount * best['price'],
            liquidity_score=best['liquidity'],
            execution_time_estimate_ms=int(best['response_time_ms'])
        )]
        
    def _smart_route(
        self,
        analysis: List[Dict[str, Any]],
        request: OrderRequest
    ) -> List[ExecutionPlan]:
        """
        Smart routing that combines multiple factors.
        
        Scoring weights:
        - Price impact: 40%
        - Fees: 20%
        - Liquidity: 25%
        - Speed: 15%
        """
        # Normalize metrics
        min_price = min(a['price'] for a in analysis)
        max_price = max(a['price'] for a in analysis)
        min_fee = min(a['fee'] for a in analysis)
        max_fee = max(a['fee'] for a in analysis)
        max_liquidity = max(a['liquidity'] for a in analysis)
        min_time = min(a['response_time_ms'] for a in analysis)
        max_time = max(a['response_time_ms'] for a in analysis)
        
        # Calculate scores
        for a in analysis:
            # Price score (lower is better for buy, higher for sell)
            if request.side == 'buy':
                price_score = 1 - (a['price'] - min_price) / (max_price - min_price + 0.0001)
            else:
                price_score = (a['price'] - min_price) / (max_price - min_price + 0.0001)
                
            # Fee score (lower is better)
            fee_score = 1 - (a['fee'] - min_fee) / (max_fee - min_fee + 0.0001)
            
            # Liquidity score (higher is better)
            liquidity_score = a['liquidity'] / (max_liquidity + 0.0001)
            
            # Speed score (lower is better)
            speed_score = 1 - (a['response_time_ms'] - min_time) / (max_time - min_time + 0.0001)
            
            # Combined score
            a['score'] = (
                price_score * 0.4 +
                fee_score * 0.2 +
                liquidity_score * 0.25 +
                speed_score * 0.15
            )
            
        # Sort by score
        analysis.sort(key=lambda x: x['score'], reverse=True)
        
        # Check if order splitting is beneficial
        if request.split_order and request.amount > (request.min_split_size or 0):
            return self._create_split_order_plan(analysis, request)
            
        # Single exchange execution
        best = analysis[0]
        
        return [ExecutionPlan(
            exchange=best['exchange'],
            amount=request.amount,
            expected_price=best['price'],
            expected_fee=best['fee'] * request.amount * best['price'],
            liquidity_score=best['liquidity'],
            execution_time_estimate_ms=int(best['response_time_ms'])
        )]
        
    def _create_split_order_plan(
        self,
        analysis: List[Dict[str, Any]],
        request: OrderRequest
    ) -> List[ExecutionPlan]:
        """Create a plan to split order across multiple exchanges."""
        plans = []
        remaining_amount = request.amount
        
        for a in analysis:
            if remaining_amount <= 0:
                break
                
            # Determine amount to execute on this exchange
            available_liquidity = a['liquidity'] * request.amount
            exec_amount = min(remaining_amount, available_liquidity)
            
            if request.min_split_size and exec_amount < request.min_split_size:
                continue
                
            plans.append(ExecutionPlan(
                exchange=a['exchange'],
                amount=exec_amount,
                expected_price=a['price'],
                expected_fee=a['fee'] * exec_amount * a['price'],
                liquidity_score=a['liquidity'],
                execution_time_estimate_ms=int(a['response_time_ms'])
            ))
            
            remaining_amount -= exec_amount
            
        return plans
        
    async def _execute_on_exchange(
        self,
        plan: ExecutionPlan,
        request: OrderRequest
    ) -> OrderResult:
        """Execute order on a specific exchange."""
        client = await self.client_manager.get_client(plan.exchange)
        if not client:
            raise ValueError(f"Exchange {plan.exchange} not available")
            
        # Prepare order parameters
        order_params = {
            'symbol': request.symbol,
            'type': request.order_type,
            'side': request.side,
            'amount': plan.amount
        }
        
        if request.order_type == 'limit' and request.price:
            order_params['price'] = request.price
            
        # Execute order
        order = await client.place_order(order_params)
        
        return OrderResult(
            order_id=order['id'],
            exchange=plan.exchange,
            symbol=request.symbol,
            side=request.side,
            amount=plan.amount,
            filled=order.get('filled', 0),
            price=order.get('price', plan.expected_price),
            fee=order.get('fee', {}).get('cost', plan.expected_fee),
            status=order['status'],
            timestamp=datetime.now(),
            raw_order=order
        )