"""
Example usage of Alpha Vantage German Stocks API integration
Demonstrates how to fetch German stock data using Alpha Vantage
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.trading_apis.alpha_vantage.alpha_vantage_trading_api import AlphaVantageTradingAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage example"""
    print("=== Alpha Vantage German Stocks - Basic Usage ===")
    
    # Configuration
    config = {
        'credentials': {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        },
        'settings': {
            'tier': 'free',
            'timezone': 'Europe/Berlin',
            'enable_german_stocks': True
        },
        'german_exchanges': [
            {'exchange': 'XETRA', 'suffix': '.DE', 'currency': 'EUR'},
            {'exchange': 'STUTTGART', 'suffix': '.STU', 'currency': 'EUR'}
        ]
    }
    
    # Initialize API
    api = AlphaVantageTradingAPI(config)
    
    try:
        # Connect
        print("Connecting to Alpha Vantage...")
        connected = await api.connect()
        if not connected:
            print("Failed to connect!")
            return
        
        print("✅ Connected successfully!")
        
        # Health check
        health = await api.health_check()
        print(f"Health Status: {health['status']}")
        if health['status'] == 'healthy':
            print(f"Latency: {health['latency_ms']:.2f}ms")
        
        # Get rate limit info
        rate_info = api.get_rate_limit_info()
        print(f"API Tier: {rate_info['tier']}")
        print(f"Daily Remaining: {rate_info['daily_remaining']}/{rate_info['daily_limit']}")
        
        # Get DAX components
        dax_symbols = api.get_dax_components()
        print(f"DAX Components Available: {len(dax_symbols)}")
        
        # Test with a few major German stocks
        test_symbols = ['SAP.DE', 'BMW.DE', 'SIE.DE']
        print(f"\nTesting with symbols: {test_symbols}")
        
        # Get market data
        print("\n--- Market Data ---")
        market_data = await api.get_market_data(test_symbols)
        
        for data in market_data:
            print(f"{data.symbol}: €{data.last:.2f} "
                  f"(Volume: {data.volume:,})")
        
        # Get trading session info
        print("\n--- Trading Session Info ---")
        session_info = api.get_trading_session_info()
        print(f"Timezone: {session_info['timezone']}")
        print(f"Base Currency: {session_info['base_currency']}")
        
        # Example with fundamentals (if available)
        if test_symbols:
            print(f"\n--- Fundamentals for {test_symbols[0]} ---")
            fundamentals = await api.get_german_stock_fundamentals(test_symbols[0])
            if fundamentals:
                print(f"Company: {fundamentals.get('Name', 'N/A')}")
                print(f"Sector: {fundamentals.get('Sector', 'N/A')}")
                print(f"Market Cap: {fundamentals.get('MarketCapitalization', 'N/A')}")
            else:
                print("No fundamentals data available")
        
        # EUR/USD rate
        print("\n--- EUR/USD Exchange Rate ---")
        eur_usd = await api.get_eur_usd_rate()
        if eur_usd:
            print(f"EUR/USD: {eur_usd:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Example error: {e}")
    
    finally:
        await api.disconnect()
        print("\n✅ Disconnected from Alpha Vantage")


async def example_market_data_subscription():
    """Market data subscription example"""
    print("\n=== Alpha Vantage German Stocks - Market Data Subscription ===")
    
    config = {
        'credentials': {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        },
        'settings': {
            'tier': 'free',
            'timezone': 'Europe/Berlin'
        }
    }
    
    api = AlphaVantageTradingAPI(config)
    
    # Market data callback
    def market_data_callback(market_data):
        print(f"📊 {market_data.symbol}: €{market_data.last:.2f} "
              f"({market_data.timestamp.strftime('%H:%M:%S')})")
    
    try:
        await api.connect()
        
        # Subscribe to market data
        symbols = ['SAP.DE', 'BMW.DE']
        print(f"Subscribing to market data for: {symbols}")
        
        await api.subscribe_market_data(symbols, market_data_callback)
        
        # Let it run for a bit (in real usage, this would run longer)
        print("Receiving market data updates for 30 seconds...")
        await asyncio.sleep(30)
        
        # Unsubscribe
        await api.unsubscribe_market_data(symbols)
        print("Unsubscribed from market data")
        
    except Exception as e:
        print(f"Subscription error: {e}")
    
    finally:
        await api.disconnect()


async def example_batch_analysis():
    """Batch analysis of German stocks"""
    print("\n=== Alpha Vantage German Stocks - Batch Analysis ===")
    
    config = {
        'credentials': {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        },
        'settings': {
            'tier': 'free',
            'timezone': 'Europe/Berlin'
        }
    }
    
    api = AlphaVantageTradingAPI(config)
    
    try:
        await api.connect()
        
        # Get DAX components
        dax_symbols = api.get_dax_components()
        
        # Analyze top 5 DAX stocks
        top_symbols = dax_symbols[:5]
        print(f"Analyzing top 5 DAX stocks: {top_symbols}")
        
        # Get market data
        market_data = await api.get_market_data(top_symbols)
        
        # Calculate metrics
        total_volume = sum(data.volume for data in market_data)
        avg_price = sum(data.last for data in market_data) / len(market_data)
        
        positive_movers = [data for data in market_data if data.last > 0]  # This is a simplified check
        
        print(f"\n📈 Analysis Results:")
        print(f"Total Volume: {total_volume:,}")
        print(f"Average Price: €{avg_price:.2f}")
        print(f"Stocks Analyzed: {len(market_data)}")
        
        # Top performers
        if market_data:
            sorted_by_volume = sorted(market_data, key=lambda x: x.volume, reverse=True)
            print(f"\nTop by Volume:")
            for i, data in enumerate(sorted_by_volume[:3], 1):
                print(f"{i}. {data.symbol}: {data.volume:,} shares")
        
        # Get news sentiment for top stock
        if market_data:
            top_stock = market_data[0].symbol
            print(f"\n📰 News Sentiment for {top_stock}:")
            news_data = await api.get_german_stock_news([top_stock])
            if news_data and 'feed' in news_data:
                feed = news_data['feed']
                print(f"News Articles: {len(feed)}")
                if feed:
                    latest_article = feed[0]
                    print(f"Latest: {latest_article.get('title', 'No title')[:60]}...")
        
    except Exception as e:
        print(f"Analysis error: {e}")
    
    finally:
        await api.disconnect()


async def example_intraday_data():
    """Intraday data example"""
    print("\n=== Alpha Vantage German Stocks - Intraday Data ===")
    
    config = {
        'credentials': {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        },
        'settings': {
            'tier': 'free',
            'timezone': 'Europe/Berlin'
        }
    }
    
    api = AlphaVantageTradingAPI(config)
    
    try:
        await api.connect()
        
        symbol = 'SAP.DE'
        print(f"Getting intraday data for {symbol}...")
        
        # Get 1-minute intraday data
        intraday_df = await api.get_intraday_data(symbol, '1min')
        
        if intraday_df is not None:
            print(f"📊 Intraday Data Shape: {intraday_df.shape}")
            print(f"Latest 5 data points:")
            print(intraday_df.tail())
            
            # Calculate some basic metrics
            if len(intraday_df) > 0:
                latest_price = intraday_df['close'].iloc[-1]
                day_high = intraday_df['high'].max()
                day_low = intraday_df['low'].min()
                total_volume = intraday_df['volume'].sum()
                
                print(f"\n📈 Day Summary for {symbol}:")
                print(f"Latest Price: €{latest_price:.2f}")
                print(f"Day High: €{day_high:.2f}")
                print(f"Day Low: €{day_low:.2f}")
                print(f"Total Volume: {total_volume:,}")
        else:
            print("No intraday data available")
        
    except Exception as e:
        print(f"Intraday data error: {e}")
    
    finally:
        await api.disconnect()


async def main():
    """Run all examples"""
    print("🚀 Alpha Vantage German Stocks API Examples")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key or api_key == 'demo':
        print("⚠️  No Alpha Vantage API key found!")
        print("Set ALPHA_VANTAGE_API_KEY environment variable to run with real data")
        print("Using demo mode for testing...")
        print()
    
    # Run examples
    try:
        await example_basic_usage()
        await asyncio.sleep(2)  # Brief pause between examples
        
        await example_market_data_subscription()
        await asyncio.sleep(2)
        
        await example_batch_analysis()
        await asyncio.sleep(2)
        
        await example_intraday_data()
        
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.error(f"Main error: {e}")
    
    print("\n✅ All examples completed!")


if __name__ == '__main__':
    asyncio.run(main())