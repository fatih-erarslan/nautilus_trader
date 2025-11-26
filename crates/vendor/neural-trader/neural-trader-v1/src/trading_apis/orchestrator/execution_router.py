"""
Smart Order Routing Engine for Low Latency Trading

Features:
- Dynamic order splitting and routing
- Liquidity aggregation across APIs
- Cost optimization
- Latency-aware execution
- Real-time market impact analysis
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
import heapq
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from .api_selector import APISelector

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class ExecutionStrategy(Enum):
    AGGRESSIVE = "aggressive"  # Speed over price
    PASSIVE = "passive"       # Price over speed
    BALANCED = "balanced"     # Balance speed and price
    STEALTH = "stealth"      # Minimize market impact


@dataclass
class OrderSlice:
    """Represents a portion of an order routed to specific API"""
    api: str
    symbol: str
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    time_in_force: str = "IOC"  # Immediate or cancel for low latency
    slice_id: str = ""
    parent_order_id: str = ""
    
    
@dataclass
class ExecutionResult:
    """Result of order execution"""
    order_id: str
    api: str
    status: str
    filled_quantity: float
    avg_fill_price: float
    latency_us: float
    timestamp: datetime
    fees: float = 0.0
    slippage: float = 0.0
    

@dataclass
class MarketDepth:
    """Market depth information from an API"""
    api: str
    symbol: str
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    timestamp: datetime
    latency_us: float


class ExecutionRouter:
    """
    Intelligent order routing with multi-API execution
    """
    
    def __init__(self, 
                 api_selector: APISelector,
                 apis: Dict[str, Any],  # API instances
                 max_slice_ratio: float = 0.3,
                 min_slice_size: float = 100.0):
        """
        Initialize execution router
        
        Args:
            api_selector: API selector instance
            apis: Dictionary of API instances
            max_slice_ratio: Maximum % of order per API
            min_slice_size: Minimum slice size in base currency
        """
        self.api_selector = api_selector
        self.apis = apis
        self.max_slice_ratio = max_slice_ratio
        self.min_slice_size = min_slice_size
        
        # Performance tracking
        self.execution_history: List[ExecutionResult] = []
        self.market_depth_cache: Dict[str, Dict[str, MarketDepth]] = defaultdict(dict)
        self.cache_ttl_ms = 100  # 100ms cache for market data
        
        # Execution optimization
        self.executor = ThreadPoolExecutor(max_workers=len(apis))
        self._liquidity_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    async def execute_order(self,
                          symbol: str,
                          quantity: float,
                          order_type: OrderType,
                          strategy: ExecutionStrategy = ExecutionStrategy.BALANCED,
                          price: Optional[float] = None,
                          urgency: float = 0.5) -> List[ExecutionResult]:
        """
        Execute order with smart routing
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            order_type: Type of order
            strategy: Execution strategy
            price: Limit price (if applicable)
            urgency: Urgency factor (0-1, affects aggressiveness)
            
        Returns:
            List of execution results from different APIs
        """
        start_time = time.perf_counter()
        
        # Get market depth from all available APIs
        market_depths = await self._fetch_market_depths(symbol)
        
        # Calculate optimal routing
        routing_plan = self._calculate_routing(
            symbol, quantity, order_type, strategy, price, market_depths, urgency
        )
        
        # Execute slices in parallel
        execution_tasks = []
        for slice_order in routing_plan:
            task = self._execute_slice(slice_order)
            execution_tasks.append(task)
        
        # Wait for all executions
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Execution failed: {result}")
                failed_results.append(result)
            else:
                successful_results.append(result)
                self.execution_history.append(result)
        
        # Handle partial fills and failures
        if failed_results:
            await self._handle_failed_executions(
                failed_results, successful_results, symbol, quantity
            )
        
        total_latency = (time.perf_counter() - start_time) * 1_000_000
        logger.info(f"Order execution completed in {total_latency:.0f}μs")
        
        return successful_results
    
    def _calculate_routing(self,
                          symbol: str,
                          quantity: float,
                          order_type: OrderType,
                          strategy: ExecutionStrategy,
                          price: Optional[float],
                          market_depths: Dict[str, MarketDepth],
                          urgency: float) -> List[OrderSlice]:
        """Calculate optimal order routing across APIs"""
        
        routing_plan = []
        remaining_quantity = quantity
        
        # Score each API for this order
        api_scores = self._score_apis_for_order(
            symbol, quantity, order_type, strategy, market_depths, urgency
        )
        
        # Sort APIs by score (highest first)
        sorted_apis = sorted(api_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate order slices
        for api, score in sorted_apis:
            if remaining_quantity <= 0:
                break
            
            # Calculate slice size based on score and constraints
            slice_ratio = min(score, self.max_slice_ratio)
            slice_size = min(
                quantity * slice_ratio,
                remaining_quantity
            )
            
            # Check minimum slice size
            if slice_size < self.min_slice_size and remaining_quantity > self.min_slice_size:
                continue
            
            # Adjust for available liquidity
            if api in market_depths:
                available_liquidity = self._calculate_available_liquidity(
                    market_depths[api], order_type, price
                )
                slice_size = min(slice_size, available_liquidity * 0.8)  # 80% of shown liquidity
            
            if slice_size > 0:
                slice_order = OrderSlice(
                    api=api,
                    symbol=symbol,
                    quantity=slice_size,
                    order_type=order_type,
                    price=price,
                    slice_id=f"{symbol}_{api}_{int(time.time()*1000)}",
                    parent_order_id=f"{symbol}_parent_{int(time.time()*1000)}"
                )
                routing_plan.append(slice_order)
                remaining_quantity -= slice_size
        
        # If any quantity remains, route to best API
        if remaining_quantity > 0 and sorted_apis:
            best_api = sorted_apis[0][0]
            slice_order = OrderSlice(
                api=best_api,
                symbol=symbol,
                quantity=remaining_quantity,
                order_type=order_type,
                price=price,
                slice_id=f"{symbol}_{best_api}_final_{int(time.time()*1000)}",
                parent_order_id=f"{symbol}_parent_{int(time.time()*1000)}"
            )
            routing_plan.append(slice_order)
        
        return routing_plan
    
    def _score_apis_for_order(self,
                             symbol: str,
                             quantity: float,
                             order_type: OrderType,
                             strategy: ExecutionStrategy,
                             market_depths: Dict[str, MarketDepth],
                             urgency: float) -> Dict[str, float]:
        """Score each API for order execution"""
        
        scores = {}
        
        for api_name in self.api_selector.apis:
            # Base score from API selector
            base_score = 0.5
            try:
                selected = self.api_selector.select_api(
                    operation_type="order",
                    order_size=quantity,
                    priority=self._strategy_to_priority(strategy)
                )
                if selected == api_name:
                    base_score = 0.8
            except:
                base_score = 0.3
            
            # Liquidity score
            liquidity_score = 0.5
            if api_name in market_depths:
                liquidity_score = self._calculate_liquidity_score(
                    market_depths[api_name], quantity, order_type
                )
            
            # Latency score (from market depth fetch)
            latency_score = 0.5
            if api_name in market_depths:
                latency_us = market_depths[api_name].latency_us
                latency_score = 1.0 - min(latency_us / 10000, 1.0)  # 10ms max
            
            # Calculate composite score based on strategy
            if strategy == ExecutionStrategy.AGGRESSIVE:
                # Prioritize speed
                score = (0.6 * latency_score + 0.3 * base_score + 0.1 * liquidity_score)
            elif strategy == ExecutionStrategy.PASSIVE:
                # Prioritize liquidity and price
                score = (0.1 * latency_score + 0.2 * base_score + 0.7 * liquidity_score)
            elif strategy == ExecutionStrategy.STEALTH:
                # Distribute evenly to minimize impact
                score = 0.5 + (0.2 * liquidity_score)
            else:  # BALANCED
                score = (0.3 * latency_score + 0.4 * base_score + 0.3 * liquidity_score)
            
            # Apply urgency factor
            score = score * (0.5 + 0.5 * urgency)
            
            scores[api_name] = score
        
        return scores
    
    def _calculate_liquidity_score(self,
                                  market_depth: MarketDepth,
                                  quantity: float,
                                  order_type: OrderType) -> float:
        """Calculate liquidity score based on market depth"""
        
        if order_type in [OrderType.MARKET, OrderType.LIMIT]:
            # For buy orders, check asks; for sell orders, check bids
            # Assuming buy order for this example
            book_side = market_depth.asks
            
            total_available = sum(level[1] for level in book_side[:5])  # Top 5 levels
            
            if total_available == 0:
                return 0.0
            
            # Score based on how much of order can be filled
            fill_ratio = min(total_available / quantity, 1.0)
            
            # Penalize if need to go deep in book
            levels_needed = 0
            remaining = quantity
            for price, size in book_side:
                remaining -= size
                levels_needed += 1
                if remaining <= 0:
                    break
            
            depth_penalty = 1.0 - (levels_needed - 1) * 0.1  # 10% penalty per level
            
            return fill_ratio * max(depth_penalty, 0.5)
        
        return 0.5  # Default for other order types
    
    def _calculate_available_liquidity(self,
                                     market_depth: MarketDepth,
                                     order_type: OrderType,
                                     price: Optional[float]) -> float:
        """Calculate available liquidity at price level"""
        
        if order_type == OrderType.MARKET:
            # Sum top 3 levels
            return sum(level[1] for level in market_depth.asks[:3])
        elif order_type == OrderType.LIMIT and price:
            # Sum liquidity at or better than limit price
            available = 0.0
            for ask_price, ask_size in market_depth.asks:
                if ask_price <= price:
                    available += ask_size
                else:
                    break
            return available
        
        return float('inf')  # No constraint
    
    def _strategy_to_priority(self, strategy: ExecutionStrategy) -> str:
        """Convert execution strategy to API selector priority"""
        if strategy == ExecutionStrategy.AGGRESSIVE:
            return "latency"
        elif strategy == ExecutionStrategy.PASSIVE:
            return "cost"
        else:
            return "balanced"
    
    async def _fetch_market_depths(self, symbol: str) -> Dict[str, MarketDepth]:
        """Fetch market depth from all available APIs"""
        
        depths = {}
        
        # Check cache first
        now = datetime.now()
        cached_depths = self.market_depth_cache.get(symbol, {})
        
        tasks = []
        for api_name, api_instance in self.apis.items():
            # Check if cached data is fresh
            if api_name in cached_depths:
                cached = cached_depths[api_name]
                age_ms = (now - cached.timestamp).total_seconds() * 1000
                if age_ms < self.cache_ttl_ms:
                    depths[api_name] = cached
                    continue
            
            # Fetch fresh data
            task = self._fetch_single_depth(api_name, api_instance, symbol)
            tasks.append(task)
        
        # Fetch all depths in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, MarketDepth):
                    depths[result.api] = result
                    # Update cache
                    self.market_depth_cache[symbol][result.api] = result
        
        return depths
    
    async def _fetch_single_depth(self, 
                                 api_name: str, 
                                 api_instance: Any,
                                 symbol: str) -> MarketDepth:
        """Fetch market depth from single API"""
        
        start_time = time.perf_counter()
        
        try:
            # Call API-specific method to get order book
            if hasattr(api_instance, 'get_order_book'):
                book = await api_instance.get_order_book(symbol)
            else:
                # Fallback to generic market data
                book = await api_instance.get_market_data(symbol)
            
            latency_us = (time.perf_counter() - start_time) * 1_000_000
            
            # Convert to standard format
            depth = MarketDepth(
                api=api_name,
                symbol=symbol,
                bids=[(float(bid['price']), float(bid['quantity'])) 
                      for bid in book.get('bids', [])[:10]],
                asks=[(float(ask['price']), float(ask['quantity'])) 
                      for ask in book.get('asks', [])[:10]],
                timestamp=datetime.now(),
                latency_us=latency_us
            )
            
            # Update metrics
            self.api_selector.update_metrics(api_name, True, latency_us)
            
            return depth
            
        except Exception as e:
            logger.error(f"Failed to fetch depth from {api_name}: {e}")
            self.api_selector.update_metrics(api_name, False)
            
            # Return empty depth
            return MarketDepth(
                api=api_name,
                symbol=symbol,
                bids=[],
                asks=[],
                timestamp=datetime.now(),
                latency_us=float('inf')
            )
    
    async def _execute_slice(self, slice_order: OrderSlice) -> ExecutionResult:
        """Execute a single order slice"""
        
        start_time = time.perf_counter()
        api_instance = self.apis.get(slice_order.api)
        
        if not api_instance:
            raise ValueError(f"API {slice_order.api} not found")
        
        try:
            # Build order parameters
            order_params = {
                'symbol': slice_order.symbol,
                'quantity': slice_order.quantity,
                'order_type': slice_order.order_type.value,
                'time_in_force': slice_order.time_in_force
            }
            
            if slice_order.price:
                order_params['price'] = slice_order.price
            
            # Execute order
            if hasattr(api_instance, 'place_order'):
                response = await api_instance.place_order(**order_params)
            else:
                # Fallback for different API interfaces
                response = await api_instance.create_order(**order_params)
            
            latency_us = (time.perf_counter() - start_time) * 1_000_000
            
            # Parse response
            result = ExecutionResult(
                order_id=response.get('order_id', slice_order.slice_id),
                api=slice_order.api,
                status=response.get('status', 'submitted'),
                filled_quantity=float(response.get('filled_quantity', 0)),
                avg_fill_price=float(response.get('avg_fill_price', 0)),
                latency_us=latency_us,
                timestamp=datetime.now(),
                fees=float(response.get('fees', 0))
            )
            
            # Calculate slippage if filled
            if result.filled_quantity > 0 and slice_order.price:
                result.slippage = abs(result.avg_fill_price - slice_order.price) / slice_order.price
            
            # Update metrics
            self.api_selector.update_metrics(
                slice_order.api, 
                True, 
                latency_us,
                response.get('rate_limit_info')
            )
            
            logger.info(f"Executed slice {slice_order.slice_id} on {slice_order.api}: "
                       f"{result.filled_quantity}/{slice_order.quantity} @ {result.avg_fill_price}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute slice on {slice_order.api}: {e}")
            self.api_selector.update_metrics(slice_order.api, False)
            
            # Return failed result
            return ExecutionResult(
                order_id=slice_order.slice_id,
                api=slice_order.api,
                status='failed',
                filled_quantity=0,
                avg_fill_price=0,
                latency_us=(time.perf_counter() - start_time) * 1_000_000,
                timestamp=datetime.now()
            )
    
    async def _handle_failed_executions(self,
                                      failed_results: List[Exception],
                                      successful_results: List[ExecutionResult],
                                      symbol: str,
                                      total_quantity: float):
        """Handle failed order slices with reallocation"""
        
        # Calculate unfilled quantity
        filled_quantity = sum(r.filled_quantity for r in successful_results)
        unfilled_quantity = total_quantity - filled_quantity
        
        if unfilled_quantity <= 0:
            return
        
        logger.warning(f"Reallocating {unfilled_quantity} unfilled quantity for {symbol}")
        
        # Get APIs that succeeded
        successful_apis = {r.api for r in successful_results}
        
        # Try to reallocate to successful APIs
        if successful_apis:
            # Use most successful API
            best_api = max(successful_results, key=lambda r: r.filled_quantity).api
            
            try:
                # Create new slice for unfilled quantity
                reallocation_slice = OrderSlice(
                    api=best_api,
                    symbol=symbol,
                    quantity=unfilled_quantity,
                    order_type=OrderType.MARKET,  # Use market order for urgency
                    slice_id=f"{symbol}_{best_api}_realloc_{int(time.time()*1000)}",
                    parent_order_id=f"{symbol}_realloc_{int(time.time()*1000)}"
                )
                
                # Execute reallocation
                result = await self._execute_slice(reallocation_slice)
                successful_results.append(result)
                
            except Exception as e:
                logger.error(f"Reallocation failed: {e}")
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get execution performance analytics"""
        
        if not self.execution_history:
            return {}
        
        # Group by API
        api_stats = defaultdict(lambda: {
            'total_orders': 0,
            'filled_orders': 0,
            'total_volume': 0.0,
            'total_fees': 0.0,
            'avg_latency_us': 0.0,
            'avg_slippage': 0.0
        })
        
        for result in self.execution_history:
            stats = api_stats[result.api]
            stats['total_orders'] += 1
            if result.filled_quantity > 0:
                stats['filled_orders'] += 1
            stats['total_volume'] += result.filled_quantity * result.avg_fill_price
            stats['total_fees'] += result.fees
            stats['avg_latency_us'] = (
                (stats['avg_latency_us'] * (stats['total_orders'] - 1) + result.latency_us) 
                / stats['total_orders']
            )
            if result.slippage > 0:
                stats['avg_slippage'] = (
                    (stats['avg_slippage'] * (stats['filled_orders'] - 1) + result.slippage)
                    / stats['filled_orders']
                )
        
        return {
            'api_statistics': dict(api_stats),
            'total_executions': len(self.execution_history),
            'avg_latency_us': np.mean([r.latency_us for r in self.execution_history]),
            'fill_rate': sum(1 for r in self.execution_history if r.filled_quantity > 0) / len(self.execution_history)
        }