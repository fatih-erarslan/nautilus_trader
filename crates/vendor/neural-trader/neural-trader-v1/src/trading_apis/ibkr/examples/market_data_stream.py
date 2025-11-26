"""
Market data streaming example using IBKR integration

This example demonstrates real-time market data streaming with
low-latency processing and callback handling.
"""

import asyncio
import logging
from src.trading_apis.ibkr import IBKRClient, IBKRDataStream, ConnectionConfig, StreamConfig, DataType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataHandler:
    """Handle market data updates"""
    
    def __init__(self):
        self.tick_count = 0
        self.last_snapshot = {}
    
    async def on_market_data(self, data):
        """Process market data updates"""
        symbol = data['symbol']
        
        if data.get('type') == 'snapshot':
            # Handle snapshot update
            snapshot = data['snapshot']
            self.last_snapshot[symbol] = snapshot
            
            logger.info(f"Snapshot {symbol}: "
                       f"Bid: {snapshot.bid}@{snapshot.bid_size}, "
                       f"Ask: {snapshot.ask}@{snapshot.ask_size}, "
                       f"Last: {snapshot.last}@{snapshot.last_size}, "
                       f"Spread: {snapshot.spread:.4f}")
        
        else:
            # Handle tick updates
            updates = data.get('updates', [])
            for update in updates:
                if update.get('type') == 'tick':
                    self.tick_count += 1
                    
                    # Log every 100th tick
                    if self.tick_count % 100 == 0:
                        logger.info(f"Processed {self.tick_count} ticks for {symbol}")
    
    def get_stats(self):
        """Get handler statistics"""
        return {
            'tick_count': self.tick_count,
            'symbols_tracked': len(self.last_snapshot)
        }


async def main():
    """Main market data streaming example"""
    
    # Configure connection
    config = ConnectionConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=2,
        auto_reconnect=True,
        readonly=True  # Read-only for market data
    )
    
    # Configure streaming
    stream_config = StreamConfig(
        buffer_size=50000,
        batch_size=200,
        batch_timeout_ms=5.0,  # 5ms batching
        conflation_ms=0,  # No conflation for lowest latency
        use_native_parsing=True,
        compression=True
    )
    
    # Create client and stream
    client = IBKRClient(config)
    data_handler = MarketDataHandler()
    
    try:
        # Connect to TWS
        logger.info("Connecting to TWS for market data...")
        if not await client.connect():
            logger.error("Failed to connect to TWS")
            return
        
        logger.info("Connected successfully!")
        
        # Create data stream
        stream = IBKRDataStream(client, stream_config)
        
        # Subscribe to multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        
        for symbol in symbols:
            success = await stream.subscribe(
                symbol=symbol,
                data_types=[DataType.TRADES, DataType.QUOTES, DataType.DEPTH],
                callback=data_handler.on_market_data
            )
            
            if success:
                logger.info(f"Subscribed to {symbol}")
            else:
                logger.error(f"Failed to subscribe to {symbol}")
        
        # Run for 30 seconds
        logger.info("Streaming market data for 30 seconds...")
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < 30:
            await asyncio.sleep(1)
            
            # Print statistics every 5 seconds
            if int(asyncio.get_event_loop().time() - start_time) % 5 == 0:
                handler_stats = data_handler.get_stats()
                stream_stats = stream.get_statistics()
                
                logger.info(f"Handler stats: {handler_stats}")
                logger.info(f"Stream stats: {stream_stats}")
                
                # Print current snapshots
                for symbol in symbols:
                    snapshot = stream.get_snapshot(symbol)
                    if snapshot:
                        logger.info(f"{symbol}: "
                                   f"Bid: {snapshot.bid:.2f}@{snapshot.bid_size}, "
                                   f"Ask: {snapshot.ask:.2f}@{snapshot.ask_size}, "
                                   f"Last: {snapshot.last:.2f}")
        
        # Final statistics
        logger.info("Final statistics:")
        logger.info(f"Handler: {data_handler.get_stats()}")
        logger.info(f"Stream: {stream.get_statistics()}")
        
        # Stop streaming
        await stream.stop()
        
    except Exception as e:
        logger.error(f"Error in market data example: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        await client.disconnect()
        logger.info("Disconnected from TWS")


if __name__ == "__main__":
    asyncio.run(main())