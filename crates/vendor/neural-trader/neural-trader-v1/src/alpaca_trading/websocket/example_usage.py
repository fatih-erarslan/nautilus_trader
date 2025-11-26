"""Example usage of Alpaca WebSocket client."""

import asyncio
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

from .alpaca_client import AlpacaWebSocketClient
from .stream_manager import StreamManager
from .message_handler import MessageHandler, TradeMessage, QuoteMessage, BarMessage
from .connection_pool import ConnectionPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def handle_trades(messages: List[TradeMessage]) -> None:
    """Handle trade messages."""
    for trade in messages:
        logger.info(
            f"Trade: {trade.symbol} @ ${trade.price:.2f} "
            f"x {trade.size} on {trade.exchange}"
        )


async def handle_quotes(messages: List[QuoteMessage]) -> None:
    """Handle quote messages."""
    for quote in messages:
        spread = quote.ask_price - quote.bid_price
        logger.info(
            f"Quote: {quote.symbol} "
            f"Bid: ${quote.bid_price:.2f} x {quote.bid_size} "
            f"Ask: ${quote.ask_price:.2f} x {quote.ask_size} "
            f"Spread: ${spread:.4f}"
        )


async def handle_bars(messages: List[BarMessage]) -> None:
    """Handle bar messages."""
    for bar in messages:
        logger.info(
            f"Bar: {bar.symbol} "
            f"O: ${bar.open:.2f} H: ${bar.high:.2f} "
            f"L: ${bar.low:.2f} C: ${bar.close:.2f} "
            f"V: {bar.volume:,}"
        )


async def example_basic_client():
    """Example using basic WebSocket client."""
    logger.info("=== Basic Client Example ===")
    
    # Get credentials from environment
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return
    
    # Create client
    client = AlpacaWebSocketClient(
        api_key=api_key,
        api_secret=api_secret,
        stream_type="data",
        feed="sip"  # Use SIP feed for best data
    )
    
    # Register message handlers
    async def handle_trade_message(message: Dict[str, Any]):
        logger.info(f"Trade: {message}")
    
    async def handle_quote_message(message: Dict[str, Any]):
        logger.info(f"Quote: {message}")
    
    client.register_handler("t", handle_trade_message)
    client.register_handler("q", handle_quote_message)
    
    try:
        # Connect
        await client.connect()
        
        # Subscribe to data
        await client.subscribe(
            trades=["AAPL", "MSFT", "GOOGL"],
            quotes=["AAPL", "MSFT"]
        )
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Get metrics
        metrics = client.get_metrics()
        logger.info(f"Client metrics: {metrics}")
        
    finally:
        # Disconnect
        await client.disconnect()


async def example_with_stream_manager():
    """Example using stream manager for organized subscriptions."""
    logger.info("=== Stream Manager Example ===")
    
    # Get credentials
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return
    
    # Create client and stream manager
    client = AlpacaWebSocketClient(api_key, api_secret)
    stream_manager = StreamManager(client)
    
    try:
        # Connect and start stream manager
        await client.connect()
        stream_manager.start()
        
        # Subscribe to tech stocks
        tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
        await stream_manager.subscribe(
            symbols=tech_stocks,
            data_types=["trades", "quotes"],
            handler=lambda msg: logger.info(f"Tech stock update: {msg}")
        )
        
        # Subscribe to SPY with different handler
        await stream_manager.subscribe(
            symbols=["SPY"],
            data_types=["trades", "bars"],
            handler=lambda msg: logger.info(f"SPY update: {msg}")
        )
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Get subscription stats
        stats = stream_manager.get_subscription_stats()
        logger.info(f"Subscription stats: {stats}")
        
        # Unsubscribe from some symbols
        await stream_manager.unsubscribe(["AMZN", "META"], ["trades"])
        
        # Run for another 10 seconds
        await asyncio.sleep(10)
        
    finally:
        stream_manager.stop()
        await client.disconnect()


async def example_with_message_handler():
    """Example using message handler for processing pipeline."""
    logger.info("=== Message Handler Example ===")
    
    # Get credentials
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return
    
    # Create components
    client = AlpacaWebSocketClient(api_key, api_secret)
    message_handler = MessageHandler(
        buffer_size=10000,
        batch_size=100,
        batch_timeout=0.1,
        worker_count=4
    )
    
    # Register processors
    message_handler.register_processor("trades", handle_trades)
    message_handler.register_processor("quotes", handle_quotes)
    message_handler.register_processor("bars", handle_bars)
    
    # Wire up client to message handler
    client.register_handler("t", message_handler.handle_message)
    client.register_handler("q", message_handler.handle_message)
    client.register_handler("b", message_handler.handle_message)
    
    try:
        # Start components
        await client.connect()
        message_handler.start()
        
        # Subscribe to data
        await client.subscribe(
            trades=["AAPL", "MSFT", "GOOGL", "TSLA"],
            quotes=["AAPL", "MSFT"],
            bars=["SPY", "QQQ"]
        )
        
        # Run for 60 seconds
        await asyncio.sleep(60)
        
        # Get processing metrics
        metrics = message_handler.get_metrics()
        logger.info(f"Processing metrics: {metrics}")
        
    finally:
        message_handler.stop()
        await client.disconnect()


async def example_with_connection_pool():
    """Example using connection pool for high-throughput scenarios."""
    logger.info("=== Connection Pool Example ===")
    
    # Get credentials
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return
    
    # Create connection pool
    pool = ConnectionPool(
        api_key=api_key,
        api_secret=api_secret,
        pool_size=3,
        max_subscriptions_per_connection=200
    )
    
    try:
        # Start pool
        await pool.start()
        
        # Get connections for different symbol groups
        # Connection 1: Large caps
        large_caps = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
        conn1 = await pool.get_connection(large_caps)
        await conn1.subscribe(trades=large_caps, quotes=large_caps)
        
        # Connection 2: ETFs
        etfs = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"]
        conn2 = await pool.get_connection(etfs)
        await conn2.subscribe(trades=etfs, bars=etfs)
        
        # Connection 3: Financial sector
        financials = ["JPM", "BAC", "WFC", "C", "GS", "MS"]
        conn3 = await pool.get_connection(financials)
        await conn3.subscribe(trades=financials, quotes=financials)
        
        # Run for 60 seconds
        await asyncio.sleep(60)
        
        # Get pool metrics
        metrics = pool.get_pool_metrics()
        logger.info(f"Pool metrics: {metrics}")
        
        # Release some symbols
        await pool.release_symbols(["TSLA", "META"])
        
        # Run for another 30 seconds
        await asyncio.sleep(30)
        
    finally:
        await pool.stop()


async def example_full_integration():
    """Example showing full integration of all components."""
    logger.info("=== Full Integration Example ===")
    
    # Get credentials
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return
    
    # Create all components
    pool = ConnectionPool(api_key, api_secret, pool_size=2)
    message_handler = MessageHandler()
    
    # Track some metrics
    trade_count = 0
    quote_count = 0
    
    async def count_trades(messages: List[TradeMessage]) -> None:
        nonlocal trade_count
        trade_count += len(messages)
        for trade in messages:
            if trade.price > 1000:  # High-priced stocks
                logger.info(f"High-value trade: {trade.symbol} @ ${trade.price:.2f}")
    
    async def count_quotes(messages: List[QuoteMessage]) -> None:
        nonlocal quote_count
        quote_count += len(messages)
        # Log wide spreads
        for quote in messages:
            spread_pct = ((quote.ask_price - quote.bid_price) / quote.bid_price) * 100
            if spread_pct > 0.1:  # Spread > 0.1%
                logger.info(f"Wide spread: {quote.symbol} {spread_pct:.3f}%")
    
    # Register processors
    message_handler.register_processor("trades", count_trades)
    message_handler.register_processor("quotes", count_quotes)
    
    try:
        # Start components
        await pool.start()
        message_handler.start()
        
        # Get a connection and set up stream manager
        conn = await pool.get_connection()
        stream_manager = StreamManager(conn)
        stream_manager.start()
        
        # Wire up message flow
        conn.register_handler("t", message_handler.handle_message)
        conn.register_handler("q", message_handler.handle_message)
        
        # Subscribe to watchlist
        watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]
        await stream_manager.subscribe(
            symbols=watchlist,
            data_types=["trades", "quotes"]
        )
        
        # Run for 2 minutes
        logger.info("Streaming market data for 2 minutes...")
        await asyncio.sleep(120)
        
        # Print final statistics
        logger.info(f"\n=== Final Statistics ===")
        logger.info(f"Total trades processed: {trade_count:,}")
        logger.info(f"Total quotes processed: {quote_count:,}")
        
        pool_metrics = pool.get_pool_metrics()
        logger.info(f"Pool metrics: {pool_metrics}")
        
        handler_metrics = message_handler.get_metrics()
        logger.info(f"Handler metrics: {handler_metrics}")
        
        sub_stats = stream_manager.get_subscription_stats()
        logger.info(f"Subscription stats: {sub_stats}")
        
    finally:
        stream_manager.stop()
        message_handler.stop()
        await pool.stop()


async def main():
    """Run examples."""
    # Uncomment the example you want to run:
    
    # await example_basic_client()
    # await example_with_stream_manager()
    # await example_with_message_handler()
    # await example_with_connection_pool()
    await example_full_integration()


if __name__ == "__main__":
    asyncio.run(main())