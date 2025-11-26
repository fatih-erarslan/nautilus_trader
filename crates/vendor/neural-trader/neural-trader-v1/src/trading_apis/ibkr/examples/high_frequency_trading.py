"""
High-frequency trading example using IBKR integration

This example demonstrates ultra-low latency trading with
optimized order placement and market data processing.
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from src.trading_apis.ibkr import IBKRClient, IBKRDataStream, ConnectionConfig, StreamConfig, DataType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal with timing information"""
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: Optional[float] = None
    signal_time: float = 0
    confidence: float = 0.0


class HighFrequencyTrader:
    """High-frequency trading strategy implementation"""
    
    def __init__(self, client: IBKRClient, stream: IBKRDataStream):
        self.client = client
        self.stream = stream
        self.positions: Dict[str, int] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.trade_count = 0
        self.pnl = 0.0
        self.last_prices: Dict[str, float] = {}
        self.signal_threshold = 0.01  # 1% price move threshold
        self.max_position_size = 100
        self.latency_stats = []
    
    async def on_market_data(self, data):
        """Process market data and generate trading signals"""
        symbol = data['symbol']
        
        try:
            # Only process snapshots for speed
            if data.get('type') == 'snapshot':
                snapshot = data['snapshot']
                
                # Generate trading signal
                signal = self._generate_signal(snapshot)
                
                if signal:
                    # Execute trade with latency tracking
                    await self._execute_trade(signal)
                    
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _generate_signal(self, snapshot) -> Optional[TradingSignal]:
        """Generate trading signal from market snapshot"""
        symbol = snapshot.symbol
        
        # Skip if no valid bid/ask
        if snapshot.bid <= 0 or snapshot.ask <= 0:
            return None
        
        # Check spread - skip if too wide
        spread_pct = (snapshot.spread / snapshot.mid) * 100
        if spread_pct > 0.1:  # 0.1% max spread
            return None
        
        # Simple mean reversion strategy
        mid_price = snapshot.mid
        last_price = self.last_prices.get(symbol, mid_price)
        
        if last_price == 0:
            last_price = mid_price
        
        # Calculate price change
        price_change = (mid_price - last_price) / last_price
        
        # Update last price
        self.last_prices[symbol] = mid_price
        
        # Check position limits
        current_position = self.positions.get(symbol, 0)
        
        # Generate signal based on mean reversion
        if price_change > self.signal_threshold:
            # Price moved up significantly, sell signal
            if current_position > -self.max_position_size:
                return TradingSignal(
                    symbol=symbol,
                    side="SELL",
                    quantity=min(10, self.max_position_size + current_position),
                    price=snapshot.bid,  # Sell at bid for immediate fill
                    signal_time=time.time(),
                    confidence=min(abs(price_change) * 10, 1.0)
                )
        
        elif price_change < -self.signal_threshold:
            # Price moved down significantly, buy signal
            if current_position < self.max_position_size:
                return TradingSignal(
                    symbol=symbol,
                    side="BUY",
                    quantity=min(10, self.max_position_size - current_position),
                    price=snapshot.ask,  # Buy at ask for immediate fill
                    signal_time=time.time(),
                    confidence=min(abs(price_change) * 10, 1.0)
                )
        
        return None
    
    async def _execute_trade(self, signal: TradingSignal):
        """Execute trade with latency tracking"""
        start_time = time.time()
        
        try:
            # Place order
            order_id = await self.client.place_order(
                symbol=signal.symbol,
                quantity=signal.quantity,
                order_type="LMT",  # Use limit orders for better control
                side=signal.side,
                price=signal.price,
                tif="IOC"  # Immediate or Cancel
            )
            
            if order_id:
                # Track order
                self.pending_orders[order_id] = {
                    'signal': signal,
                    'order_time': start_time,
                    'order_id': order_id
                }
                
                # Update position estimate
                quantity_signed = signal.quantity if signal.side == "BUY" else -signal.quantity
                self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + quantity_signed
                
                # Track latency
                execution_latency = (time.time() - start_time) * 1000
                self.latency_stats.append(execution_latency)
                
                self.trade_count += 1
                
                logger.info(f"Trade {self.trade_count}: {signal.side} {signal.quantity} {signal.symbol} "
                           f"@ {signal.price:.2f} (Latency: {execution_latency:.1f}ms)")
                
                # Keep only last 1000 latency measurements
                if len(self.latency_stats) > 1000:
                    self.latency_stats = self.latency_stats[-1000:]
                
            else:
                logger.error(f"Failed to place order for {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def on_order_status(self, order_id: str, status):
        """Handle order status updates"""
        if order_id in self.pending_orders:
            order_info = self.pending_orders[order_id]
            signal = order_info['signal']
            
            if status.status == 'Filled':
                # Calculate P&L
                fill_price = float(status.avgFillPrice)
                quantity_signed = signal.quantity if signal.side == "BUY" else -signal.quantity
                
                # Simple P&L calculation (assuming we close positions)
                if signal.side == "SELL":
                    pnl_contribution = (fill_price - signal.price) * signal.quantity
                else:
                    pnl_contribution = (signal.price - fill_price) * signal.quantity
                
                self.pnl += pnl_contribution
                
                # Calculate fill latency
                fill_latency = (time.time() - order_info['order_time']) * 1000
                
                logger.info(f"Order {order_id} filled at {fill_price:.2f} "
                           f"(Fill latency: {fill_latency:.1f}ms, P&L: ${pnl_contribution:.2f})")
                
                # Remove from pending
                del self.pending_orders[order_id]
                
            elif status.status in ['Cancelled', 'Rejected']:
                logger.warning(f"Order {order_id} {status.status}")
                
                # Revert position estimate
                signal = order_info['signal']
                quantity_signed = signal.quantity if signal.side == "BUY" else -signal.quantity
                self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) - quantity_signed
                
                del self.pending_orders[order_id]
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        avg_latency = sum(self.latency_stats) / len(self.latency_stats) if self.latency_stats else 0
        
        return {
            'trade_count': self.trade_count,
            'total_pnl': self.pnl,
            'positions': self.positions.copy(),
            'pending_orders': len(self.pending_orders),
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max(self.latency_stats) if self.latency_stats else 0,
            'min_latency_ms': min(self.latency_stats) if self.latency_stats else 0
        }


async def main():
    """Main high-frequency trading example"""
    
    # Configure connection for lowest latency
    config = ConnectionConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=3,
        auto_reconnect=True,
        readonly=False,
        timeout=2.0  # Shorter timeout for faster detection
    )
    
    # Configure streaming for ultra-low latency
    stream_config = StreamConfig(
        buffer_size=100000,
        batch_size=500,
        batch_timeout_ms=1.0,  # 1ms batching
        conflation_ms=0,  # No conflation
        use_native_parsing=True,
        compression=False,  # Disable compression for speed
        snapshot_interval_ms=100.0  # Frequent snapshots
    )
    
    # Create client and stream
    client = IBKRClient(config)
    
    try:
        # Connect to TWS
        logger.info("Connecting to TWS for high-frequency trading...")
        if not await client.connect():
            logger.error("Failed to connect to TWS")
            return
        
        logger.info("Connected successfully!")
        
        # Create data stream
        stream = IBKRDataStream(client, stream_config)
        
        # Create trader
        trader = HighFrequencyTrader(client, stream)
        
        # Register order status callback
        client.register_callback('order_status', trader.on_order_status)
        
        # Subscribe to high-volume symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
        
        for symbol in symbols:
            success = await stream.subscribe(
                symbol=symbol,
                data_types=[DataType.TRADES, DataType.QUOTES],
                callback=trader.on_market_data
            )
            
            if success:
                logger.info(f"Subscribed to {symbol}")
            else:
                logger.error(f"Failed to subscribe to {symbol}")
        
        # Run trading for 60 seconds
        logger.info("Starting high-frequency trading for 60 seconds...")
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < 60:
            await asyncio.sleep(5)
            
            # Print statistics every 5 seconds
            trader_stats = trader.get_statistics()
            stream_stats = stream.get_statistics()
            client_latency = client.get_latency_report()
            
            logger.info(f"Trader stats: {trader_stats}")
            logger.info(f"Stream processed {stream_stats.get('ticks_processed', 0)} ticks")
            logger.info(f"Client latency: {client_latency}")
        
        # Final statistics
        logger.info("Final trading statistics:")
        final_stats = trader.get_statistics()
        logger.info(f"Final stats: {final_stats}")
        
        # Stop streaming
        await stream.stop()
        
    except Exception as e:
        logger.error(f"Error in high-frequency trading: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        await client.disconnect()
        logger.info("Disconnected from TWS")


if __name__ == "__main__":
    asyncio.run(main())