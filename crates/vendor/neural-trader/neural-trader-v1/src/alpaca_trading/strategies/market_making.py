"""
Market Making Strategy for Alpaca WebSocket Trading

Bid-ask spread capture with inventory management,
quote adjustment logic, and risk limits.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import deque
from .base_strategy import BaseStreamStrategy, TradingSignal, SignalType


class MarketMakingStrategy(BaseStreamStrategy):
    """
    Market making strategy for capturing bid-ask spreads
    
    Features:
    - Dynamic spread pricing based on volatility
    - Inventory risk management
    - Quote skewing based on position
    - Adverse selection protection
    - Multiple order management
    """
    
    def __init__(self,
                 symbols: List[str],
                 spread_basis_points: float = 10.0,  # 0.1%
                 max_inventory: int = 1000,          # Max shares per symbol
                 inventory_skew_factor: float = 0.5,  # How much to skew quotes
                 min_spread_bps: float = 5.0,        # Minimum spread
                 max_spread_bps: float = 50.0,       # Maximum spread
                 volatility_window: int = 100,
                 order_lifetime_seconds: int = 30,
                 **kwargs):
        """
        Initialize market making strategy
        
        Args:
            spread_basis_points: Target spread in basis points
            max_inventory: Maximum inventory per symbol
            inventory_skew_factor: How much to adjust quotes based on inventory
            min_spread_bps: Minimum allowed spread
            max_spread_bps: Maximum allowed spread
            volatility_window: Window for volatility calculation
            order_lifetime_seconds: How long to keep orders active
        """
        super().__init__(symbols, **kwargs)
        
        self.spread_basis_points = spread_basis_points
        self.max_inventory = max_inventory
        self.inventory_skew_factor = inventory_skew_factor
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps
        self.volatility_window = volatility_window
        self.order_lifetime_seconds = order_lifetime_seconds
        
        # Market data tracking
        self.bid_ask_history = {s: deque(maxlen=100) for s in symbols}
        self.trade_flow = {s: deque(maxlen=200) for s in symbols}
        self.volatility_estimates = {s: 0.0 for s in symbols}
        
        # Inventory tracking
        self.inventory = {s: 0 for s in symbols}
        self.inventory_value = {s: 0.0 for s in symbols}
        
        # Order management
        self.active_quotes = {s: {'bid': None, 'ask': None} for s in symbols}
        self.order_timestamps = {s: {'bid': None, 'ask': None} for s in symbols}
        
        # Performance tracking
        self.filled_orders = {s: [] for s in symbols}
        self.spread_captures = {s: [] for s in symbols}
        self.adverse_fills = {s: 0 for s in symbols}
        
    def _on_trade(self, trade: Dict[str, Any]):
        """Process trade data for market making decisions"""
        symbol = trade['symbol']
        price = trade['price']
        size = trade['size']
        
        # Track trade flow for adverse selection detection
        self.trade_flow[symbol].append({
            'price': price,
            'size': size,
            'timestamp': trade['timestamp'],
            'side': self._infer_trade_side(symbol, price)
        })
        
        # Update volatility estimate
        self._update_volatility(symbol)
        
        # Check if our quotes were hit
        self._check_fill(symbol, price, size)
    
    def _on_quote(self, quote: Dict[str, Any]):
        """Process quote data for spread analysis"""
        symbol = quote['symbol']
        
        # Store bid-ask data
        self.bid_ask_history[symbol].append({
            'bid': quote['bid_price'],
            'ask': quote['ask_price'],
            'bid_size': quote['bid_size'],
            'ask_size': quote['ask_size'],
            'spread': quote['ask_price'] - quote['bid_price'],
            'timestamp': quote['timestamp']
        })
        
        # Update quotes if market moved significantly
        self._update_quotes_if_needed(symbol)
    
    def _on_bar(self, bar: Dict[str, Any]):
        """Process bar data for longer-term analysis"""
        # Can use for additional volatility estimates
        pass
    
    def _infer_trade_side(self, symbol: str, price: float) -> str:
        """Infer if trade was buyer or seller initiated"""
        if symbol in self.latest_quotes:
            quote = self.latest_quotes[symbol]
            mid = (quote['bid_price'] + quote['ask_price']) / 2
            return 'buy' if price >= mid else 'sell'
        return 'unknown'
    
    def _update_volatility(self, symbol: str):
        """Update volatility estimate for dynamic spread adjustment"""
        trades = list(self.trade_flow[symbol])
        if len(trades) < 10:
            return
            
        # Calculate realized volatility from recent trades
        prices = [t['price'] for t in trades[-self.volatility_window:]]
        if len(prices) >= 2:
            returns = np.diff(np.log(prices))
            # Annualized volatility
            self.volatility_estimates[symbol] = np.std(returns) * np.sqrt(252 * 6.5 * 60)  # Per minute
    
    def _check_fill(self, symbol: str, price: float, size: int):
        """Check if our quotes were filled"""
        quotes = self.active_quotes[symbol]
        
        # Check bid fill
        if quotes['bid'] and price <= quotes['bid']['price']:
            # We bought
            self.inventory[symbol] += size
            self.inventory_value[symbol] += size * price
            
            self.filled_orders[symbol].append({
                'side': 'buy',
                'price': price,
                'size': size,
                'timestamp': datetime.now()
            })
            
            # Reset bid
            self.active_quotes[symbol]['bid'] = None
            self.order_timestamps[symbol]['bid'] = None
            
            self.logger.info(f"BID FILLED: {symbol} @ ${price:.2f} x {size}")
            
        # Check ask fill
        elif quotes['ask'] and price >= quotes['ask']['price']:
            # We sold
            self.inventory[symbol] -= size
            self.inventory_value[symbol] -= size * self._get_avg_inventory_cost(symbol)
            
            # Calculate spread capture
            avg_cost = self._get_avg_inventory_cost(symbol)
            spread_capture = (price - avg_cost) * size
            self.spread_captures[symbol].append(spread_capture)
            
            self.filled_orders[symbol].append({
                'side': 'sell',
                'price': price,
                'size': size,
                'timestamp': datetime.now()
            })
            
            # Reset ask
            self.active_quotes[symbol]['ask'] = None
            self.order_timestamps[symbol]['ask'] = None
            
            self.logger.info(f"ASK FILLED: {symbol} @ ${price:.2f} x {size}, Capture: ${spread_capture:.2f}")
    
    def _get_avg_inventory_cost(self, symbol: str) -> float:
        """Get average cost of inventory"""
        if self.inventory[symbol] > 0:
            return self.inventory_value[symbol] / self.inventory[symbol]
        return self.latest_trades.get(symbol, {}).get('price', 0)
    
    def _update_quotes_if_needed(self, symbol: str):
        """Update quotes if market moved or orders are stale"""
        now = datetime.now()
        
        # Check if orders are stale
        for side in ['bid', 'ask']:
            if self.order_timestamps[symbol][side]:
                age = (now - self.order_timestamps[symbol][side]).total_seconds()
                if age > self.order_lifetime_seconds:
                    self.active_quotes[symbol][side] = None
                    self.order_timestamps[symbol][side] = None
    
    def calculate_optimal_spread(self, symbol: str) -> float:
        """
        Calculate optimal spread based on volatility and adverse selection
        """
        base_spread = self.spread_basis_points / 10000  # Convert to decimal
        
        # Adjust for volatility
        volatility = self.volatility_estimates[symbol]
        if volatility > 0:
            # Higher volatility = wider spread
            vol_adjustment = max(1.0, volatility / 0.20)  # 20% annual vol baseline
        else:
            vol_adjustment = 1.0
        
        # Adjust for adverse selection
        recent_fills = self.filled_orders[symbol][-10:] if symbol in self.filled_orders else []
        if recent_fills:
            # Check if we're consistently being picked off
            adverse_ratio = self._calculate_adverse_selection_ratio(symbol, recent_fills)
            if adverse_ratio > 0.3:  # 30% adverse fills
                vol_adjustment *= 1.5
        
        # Calculate final spread
        optimal_spread = base_spread * vol_adjustment
        
        # Apply limits
        min_spread = self.min_spread_bps / 10000
        max_spread = self.max_spread_bps / 10000
        
        return max(min_spread, min(max_spread, optimal_spread))
    
    def _calculate_adverse_selection_ratio(self, symbol: str, recent_fills: List[Dict]) -> float:
        """Calculate ratio of adverse fills"""
        if not recent_fills:
            return 0.0
            
        adverse_count = 0
        for fill in recent_fills:
            # Check if price moved against us after fill
            fill_price = fill['price']
            current_price = self.latest_trades.get(symbol, {}).get('price', fill_price)
            
            if fill['side'] == 'buy' and current_price < fill_price * 0.999:
                adverse_count += 1
            elif fill['side'] == 'sell' and current_price > fill_price * 1.001:
                adverse_count += 1
                
        return adverse_count / len(recent_fills)
    
    def calculate_quote_prices(self, symbol: str) -> Tuple[float, float]:
        """
        Calculate bid and ask prices with inventory skew
        """
        if symbol not in self.latest_trades or 'price' not in self.latest_trades[symbol]:
            return 0.0, 0.0
            
        mid_price = self.latest_trades[symbol]['price']
        optimal_spread = self.calculate_optimal_spread(symbol)
        
        # Base bid/ask
        half_spread = optimal_spread * mid_price / 2
        base_bid = mid_price - half_spread
        base_ask = mid_price + half_spread
        
        # Inventory skew
        inventory_ratio = self.inventory[symbol] / self.max_inventory
        skew = self.inventory_skew_factor * inventory_ratio * half_spread
        
        # Adjust quotes based on inventory
        # Long inventory: lower bid, lower ask (want to sell)
        # Short inventory: higher bid, higher ask (want to buy)
        bid_price = base_bid - skew
        ask_price = base_ask - skew
        
        return bid_price, ask_price
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate market making signals (quote updates)"""
        # Check inventory limits
        if abs(self.inventory[symbol]) >= self.max_inventory:
            # Need to reduce inventory
            if self.inventory[symbol] > 0:
                # Only sell
                return self._generate_sell_signal(symbol)
            else:
                # Only buy
                return self._generate_buy_signal(symbol)
        
        # Calculate optimal quotes
        bid_price, ask_price = self.calculate_quote_prices(symbol)
        
        if bid_price <= 0 or ask_price <= 0:
            return None
        
        # Determine which side to update
        current_quotes = self.active_quotes[symbol]
        now = datetime.now()
        
        # Update bid if needed
        if (not current_quotes['bid'] or 
            abs(current_quotes['bid']['price'] - bid_price) > bid_price * 0.0001):
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                timestamp=now,
                price=bid_price,
                quantity=self._calculate_quote_size(symbol, 'bid'),
                confidence=0.7,
                reason=f"Market making bid: spread={optimal_spread*10000:.1f}bps",
                metadata={
                    'quote_type': 'bid',
                    'spread_bps': optimal_spread * 10000,
                    'inventory': self.inventory[symbol],
                    'skew': (bid_price - (self.latest_trades[symbol]['price'] - optimal_spread * self.latest_trades[symbol]['price'] / 2)) / bid_price
                }
            )
            
            self.active_quotes[symbol]['bid'] = {'price': bid_price, 'size': signal.quantity}
            self.order_timestamps[symbol]['bid'] = now
            
            return signal
        
        # Update ask if needed
        elif (not current_quotes['ask'] or 
              abs(current_quotes['ask']['price'] - ask_price) > ask_price * 0.0001):
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                timestamp=now,
                price=ask_price,
                quantity=self._calculate_quote_size(symbol, 'ask'),
                confidence=0.7,
                reason=f"Market making ask: spread={optimal_spread*10000:.1f}bps",
                metadata={
                    'quote_type': 'ask',
                    'spread_bps': optimal_spread * 10000,
                    'inventory': self.inventory[symbol]
                }
            )
            
            self.active_quotes[symbol]['ask'] = {'price': ask_price, 'size': signal.quantity}
            self.order_timestamps[symbol]['ask'] = now
            
            return signal
        
        return None
    
    def _generate_buy_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate buy signal when need to increase inventory"""
        bid_price, _ = self.calculate_quote_prices(symbol)
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            timestamp=datetime.now(),
            price=bid_price,
            quantity=self._calculate_quote_size(symbol, 'bid'),
            confidence=0.6,
            reason="Inventory management: need to buy",
            metadata={'inventory': self.inventory[symbol]}
        )
    
    def _generate_sell_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate sell signal when need to reduce inventory"""
        _, ask_price = self.calculate_quote_prices(symbol)
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            timestamp=datetime.now(),
            price=ask_price,
            quantity=min(abs(self.inventory[symbol]), self._calculate_quote_size(symbol, 'ask')),
            confidence=0.6,
            reason="Inventory management: need to sell",
            metadata={'inventory': self.inventory[symbol]}
        )
    
    def _calculate_quote_size(self, symbol: str, side: str) -> int:
        """Calculate appropriate quote size"""
        # Base size from position sizing
        if self.latest_trades.get(symbol, {}).get('price', 0) > 0:
            base_size = int(self.position_size * 0.1 / self.latest_trades[symbol]['price'])
        else:
            base_size = 100
            
        # Adjust based on inventory
        inventory_ratio = abs(self.inventory[symbol]) / self.max_inventory
        
        # Reduce size as inventory grows
        size_adjustment = max(0.2, 1.0 - inventory_ratio * 0.8)
        
        # Different sizing for bid vs ask based on inventory
        if side == 'bid' and self.inventory[symbol] > 0:
            # Already long, reduce bid size
            size_adjustment *= 0.5
        elif side == 'ask' and self.inventory[symbol] < 0:
            # Already short, reduce ask size  
            size_adjustment *= 0.5
            
        return max(1, int(base_size * size_adjustment))
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get market making specific metrics"""
        base_metrics = self.get_performance_summary()
        
        # Calculate MM-specific metrics
        mm_metrics = {
            'inventory': self.inventory,
            'inventory_value': self.inventory_value,
            'active_quotes': {},
            'spread_captures': {},
            'fill_rates': {},
            'average_spreads': {}
        }
        
        for symbol in self.symbols:
            # Active quotes
            mm_metrics['active_quotes'][symbol] = {
                'bid': self.active_quotes[symbol]['bid']['price'] if self.active_quotes[symbol]['bid'] else None,
                'ask': self.active_quotes[symbol]['ask']['price'] if self.active_quotes[symbol]['ask'] else None
            }
            
            # Spread captures
            if self.spread_captures[symbol]:
                mm_metrics['spread_captures'][symbol] = {
                    'total': sum(self.spread_captures[symbol]),
                    'average': np.mean(self.spread_captures[symbol]),
                    'count': len(self.spread_captures[symbol])
                }
            
            # Fill rates
            if self.filled_orders[symbol]:
                mm_metrics['fill_rates'][symbol] = len(self.filled_orders[symbol])
                
            # Average spreads
            if self.bid_ask_history[symbol]:
                recent_spreads = [ba['spread'] for ba in list(self.bid_ask_history[symbol])[-20:]]
                mm_metrics['average_spreads'][symbol] = np.mean(recent_spreads)
        
        base_metrics['market_making_metrics'] = mm_metrics
        
        return base_metrics