"""
CCXT Integration - Basic Usage Example

This script demonstrates how to use the CCXT integration module for:
- Connecting to exchanges
- Fetching market data
- Placing orders
- Streaming real-time data
"""

import asyncio
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccxt_integration.interfaces.ccxt_interface import CCXTInterface, ExchangeConfig
from ccxt_integration.core.client_manager import ClientManager
from ccxt_integration.core.exchange_registry import ExchangeRegistry
from ccxt_integration.streaming.websocket_manager import WebSocketManager
from ccxt_integration.execution.order_router import OrderRouter, OrderRequest, RoutingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_interface():
    """Example: Basic interface usage"""
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Interface Usage")
    print("="*50)
    
    # Create exchange configuration
    config = ExchangeConfig(
        name='binance',
        api_key='your_api_key_here',  # Replace with actual key
        secret='your_secret_here',     # Replace with actual secret
        sandbox=True,  # Use sandbox for testing
        enable_rate_limit=True
    )
    
    # Create and initialize interface
    interface = CCXTInterface(config)
    
    try:
        await interface.initialize()
        print(f"✅ Connected to {config.name} (sandbox: {config.sandbox})")
        
        # Fetch ticker
        ticker = await interface.get_ticker('BTC/USDT')
        print(f"\n📊 BTC/USDT Ticker:")
        print(f"  Last Price: ${ticker['last']:,.2f}")
        print(f"  Bid: ${ticker['bid']:,.2f}")
        print(f"  Ask: ${ticker['ask']:,.2f}")
        print(f"  24h Volume: {ticker['volume']:,.2f} BTC")
        
        # Fetch orderbook
        orderbook = await interface.get_orderbook('BTC/USDT', limit=5)
        print(f"\n📖 Order Book (Top 5):")
        print("  Bids:")
        for bid in orderbook['bids'][:5]:
            print(f"    ${bid[0]:,.2f} - {bid[1]:.4f} BTC")
        print("  Asks:")
        for ask in orderbook['asks'][:5]:
            print(f"    ${ask[0]:,.2f} - {ask[1]:.4f} BTC")
            
        # Fetch recent trades
        trades = await interface.get_trades('BTC/USDT', limit=5)
        print(f"\n🔄 Recent Trades:")
        for trade in trades[:5]:
            print(f"  {trade['side']:4s} {trade['amount']:.4f} BTC @ ${trade['price']:,.2f}")
            
        # Fetch OHLCV data
        ohlcv = await interface.get_klines('BTC/USDT', '1h', limit=5)
        print(f"\n📈 Hourly Candles:")
        for candle in ohlcv:
            print(f"  O: ${candle[1]:,.2f} H: ${candle[2]:,.2f} L: ${candle[3]:,.2f} C: ${candle[4]:,.2f} V: {candle[5]:.2f}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        await interface.close()
        print("\n✅ Connection closed")


async def example_client_manager():
    """Example: Using ClientManager for multi-exchange operations"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Multi-Exchange Client Manager")
    print("="*50)
    
    # Create client manager
    manager = ClientManager(max_clients_per_exchange=2)
    
    try:
        # Add multiple exchanges
        exchanges = [
            ExchangeConfig(name='binance', sandbox=True),
            ExchangeConfig(name='kraken', sandbox=False),
            # Add more exchanges as needed
        ]
        
        for config in exchanges:
            try:
                await manager.add_exchange(config)
                print(f"✅ Added {config.name} to manager")
            except Exception as e:
                print(f"⚠️ Failed to add {config.name}: {str(e)}")
                
        # Start health monitoring
        await manager.start_health_monitoring(interval_seconds=30)
        print("\n🏥 Started health monitoring")
        
        # Get best exchange for a symbol
        best_exchange = await manager.get_best_exchange('BTC/USDT')
        if best_exchange:
            print(f"\n🎯 Best exchange for BTC/USDT: {best_exchange}")
            
            # Get client and fetch ticker
            client = await manager.get_client(best_exchange)
            if client:
                ticker = await client.get_ticker('BTC/USDT')
                print(f"  Price on {best_exchange}: ${ticker['last']:,.2f}")
                
        # Execute with failover
        print("\n🔄 Executing with failover...")
        try:
            result = await manager.execute_with_failover(
                'get_ticker',
                'ETH/USDT'
            )
            print(f"  ETH/USDT Price: ${result['last']:,.2f}")
        except Exception as e:
            print(f"  Failed: {str(e)}")
            
        # Get health report
        health_report = await manager.get_health_report()
        print("\n📊 Health Report:")
        print(f"  Total Exchanges: {health_report['total_exchanges']}")
        print(f"  Healthy: {health_report['healthy_exchanges']}")
        print(f"  Unhealthy: {health_report['unhealthy_exchanges']}")
        
    finally:
        await manager.close_all()
        print("\n✅ All connections closed")


async def example_websocket_streaming():
    """Example: Real-time data streaming with WebSocket"""
    print("\n" + "="*50)
    print("EXAMPLE 3: WebSocket Streaming")
    print("="*50)
    
    # Note: Requires ccxt.pro (paid version)
    try:
        # Create WebSocket manager
        ws_manager = WebSocketManager(
            exchange_name='binance',
            config={'sandbox': True}
        )
        
        await ws_manager.initialize()
        print("✅ WebSocket manager initialized")
        
        # Define callbacks
        def on_ticker(ticker):
            print(f"  📊 Ticker Update: {ticker['symbol']} - ${ticker['last']:,.2f}")
            
        def on_trade(trade):
            print(f"  🔄 New Trade: {trade['side']} {trade['amount']} @ ${trade['price']:,.2f}")
            
        def on_orderbook(orderbook):
            best_bid = orderbook['bids'][0] if orderbook['bids'] else [0, 0]
            best_ask = orderbook['asks'][0] if orderbook['asks'] else [0, 0]
            print(f"  📖 Orderbook: Bid ${best_bid[0]:,.2f} / Ask ${best_ask[0]:,.2f}")
            
        # Subscribe to streams
        ticker_sub = await ws_manager.subscribe_ticker('BTC/USDT', on_ticker)
        trades_sub = await ws_manager.subscribe_trades('BTC/USDT', on_trade)
        orderbook_sub = await ws_manager.subscribe_orderbook('BTC/USDT', on_orderbook, limit=10)
        
        print("\n📡 Streaming real-time data for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get subscription status
        status = ws_manager.get_subscription_status()
        print(f"\n📊 Stream Status:")
        print(f"  Active Streams: {status['active_streams']}")
        print(f"  Subscriptions: {len(status['subscriptions'])}")
        
        # Unsubscribe
        await ws_manager.unsubscribe(ticker_sub)
        await ws_manager.unsubscribe(trades_sub)
        await ws_manager.unsubscribe(orderbook_sub)
        
    except ImportError:
        print("⚠️ CCXT Pro not installed. WebSocket features require the paid version.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'ws_manager' in locals():
            await ws_manager.close()
            print("✅ WebSocket manager closed")


async def example_smart_order_routing():
    """Example: Smart order routing across exchanges"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Smart Order Routing")
    print("="*50)
    
    # Create client manager
    manager = ClientManager()
    
    try:
        # Add exchanges
        await manager.add_exchange(ExchangeConfig(name='binance', sandbox=True))
        await manager.add_exchange(ExchangeConfig(name='kraken', sandbox=True))
        print("✅ Added exchanges to manager")
        
        # Create order router
        router = OrderRouter(manager)
        
        # Create order request
        order_request = OrderRequest(
            symbol='BTC/USDT',
            side='buy',
            amount=0.01,
            order_type='market',
            routing_strategy=RoutingStrategy.SMART_ROUTING,
            max_slippage=0.01,  # 1% max slippage
            split_order=True,
            min_split_size=0.005
        )
        
        print(f"\n📦 Order Request:")
        print(f"  Symbol: {order_request.symbol}")
        print(f"  Side: {order_request.side}")
        print(f"  Amount: {order_request.amount} BTC")
        print(f"  Strategy: {order_request.routing_strategy.value}")
        
        # Route order (simulation - won't execute without real API keys)
        print("\n🚀 Routing order...")
        print("  ⚠️ Note: This is a simulation. Real execution requires valid API keys.")
        
        # In real usage:
        # results = await router.route_order(order_request)
        # for result in results:
        #     print(f"  ✅ Executed on {result.exchange}: {result.filled}/{result.amount} @ ${result.price:,.2f}")
        
    finally:
        await manager.close_all()
        print("\n✅ Order routing example completed")


async def example_exchange_registry():
    """Example: Using the exchange registry"""
    print("\n" + "="*50)
    print("EXAMPLE 5: Exchange Registry")
    print("="*50)
    
    # Create registry
    registry = ExchangeRegistry()
    
    # List all exchanges
    all_exchanges = registry.list_exchanges()
    print(f"📋 Available Exchanges: {', '.join(all_exchanges)}")
    
    # Get exchanges with specific capabilities
    websocket_exchanges = registry.get_exchanges_by_capability('websocket_support')
    print(f"\n🔌 Exchanges with WebSocket: {', '.join(websocket_exchanges)}")
    
    futures_exchanges = registry.get_exchanges_by_capability('futures_trading')
    print(f"📈 Exchanges with Futures: {', '.join(futures_exchanges)}")
    
    sandbox_exchanges = registry.get_exchanges_by_capability('sandbox_available')
    print(f"🧪 Exchanges with Sandbox: {', '.join(sandbox_exchanges)}")
    
    # Get exchanges by quote currency
    usdt_exchanges = registry.get_exchanges_by_quote_currency('USDT')
    print(f"\n💵 Exchanges supporting USDT: {', '.join(usdt_exchanges)}")
    
    # Get best exchange for a pair
    best_exchange = registry.get_best_exchange_for_pair('BTC', 'USDT')
    if best_exchange:
        metadata = registry.get_exchange(best_exchange)
        print(f"\n🎯 Best exchange for BTC/USDT: {best_exchange}")
        print(f"  Fees: Maker {metadata.trading_fees.get('maker', 0)*100:.2f}% / Taker {metadata.trading_fees.get('taker', 0)*100:.2f}%")
        print(f"  Country: {metadata.country}")
        print(f"  Max Request Rate: {metadata.capabilities.max_request_rate} req/s")
        
    # Get registry summary
    summary = registry.get_summary()
    print(f"\n📊 Registry Summary:")
    print(f"  Total Exchanges: {summary['total_exchanges']}")
    print(f"  Active Exchanges: {summary['active_exchanges']}")
    print(f"  Capabilities:")
    for capability, count in summary['capabilities'].items():
        print(f"    {capability}: {count} exchanges")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print(" CCXT INTEGRATION MODULE - USAGE EXAMPLES")
    print("="*60)
    print("\n⚠️ Note: These examples use sandbox mode for safety.")
    print("Replace API keys with real ones for actual trading.\n")
    
    # Run examples
    await example_basic_interface()
    await example_client_manager()
    await example_websocket_streaming()
    await example_smart_order_routing()
    await example_exchange_registry()
    
    print("\n" + "="*60)
    print(" ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())