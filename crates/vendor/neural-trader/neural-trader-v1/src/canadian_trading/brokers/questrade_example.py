"""
Questrade API Usage Examples

This module demonstrates how to use the Questrade API integration
for various trading operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from questrade import (
    QuestradeAPI,
    QuestradeDataFeed,
    QuestradeOrderManager,
    QuestradeAPIError
)
from ..utils import setup_broker_authentication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def authentication_example():
    """Example: Authenticate with Questrade"""
    print("\n=== Authentication Example ===")
    
    # Method 1: Initial authentication with manual refresh token
    # Get this from Questrade App Hub
    manual_refresh_token = "your_refresh_token_here"
    
    try:
        auth_result = await setup_broker_authentication(
            "questrade",
            refresh_token=manual_refresh_token
        )
        print(f"Authentication successful: {auth_result}")
    except Exception as e:
        print(f"Authentication failed: {e}")
    
    # Method 2: Use stored tokens (after initial authentication)
    try:
        auth_result = await setup_broker_authentication("questrade")
        if auth_result["status"] == "authenticated":
            print("Successfully authenticated using stored tokens")
        else:
            print("Stored tokens not found or expired")
    except Exception as e:
        print(f"Authentication error: {e}")


async def account_info_example(api: QuestradeAPI):
    """Example: Get account information"""
    print("\n=== Account Information Example ===")
    
    try:
        # Get all accounts
        accounts = await api.get_accounts()
        print(f"Found {len(accounts)} accounts:")
        
        for account in accounts:
            account_id = account["number"]
            account_type = account["type"]
            print(f"\nAccount: {account_id} ({account_type})")
            
            # Get account balances
            balances = await api.get_account_balances(account_id)
            for balance in balances:
                currency = balance["currency"]
                cash = balance["cash"]
                market_value = balance["marketValue"]
                total_equity = balance["totalEquity"]
                print(f"  {currency}: Cash=${cash:,.2f}, Market=${market_value:,.2f}, Total=${total_equity:,.2f}")
            
            # Get positions
            positions = await api.get_account_positions(account_id)
            if positions:
                print(f"  Positions:")
                for position in positions[:5]:  # Show first 5
                    symbol = position["symbol"]
                    quantity = position["openQuantity"]
                    avg_price = position["averageEntryPrice"]
                    current_price = position["currentPrice"]
                    pnl = position["openPnl"]
                    print(f"    {symbol}: {quantity} shares @ ${avg_price:.2f}, Current: ${current_price:.2f}, P&L: ${pnl:,.2f}")
            
    except QuestradeAPIError as e:
        print(f"API Error: {e}")


async def market_data_example(api: QuestradeAPI):
    """Example: Get market data"""
    print("\n=== Market Data Example ===")
    
    data_feed = QuestradeDataFeed(api)
    
    try:
        # Get single quote
        symbol = "SHOP.TO"
        quote = await data_feed.get_quote(symbol)
        if quote:
            print(f"\n{symbol} Quote:")
            print(f"  Last: ${quote['lastTradePrice']:.2f}")
            print(f"  Bid: ${quote['bidPrice']:.2f} x {quote['bidSize']}")
            print(f"  Ask: ${quote['askPrice']:.2f} x {quote['askSize']}")
            print(f"  Volume: {quote['volume']:,}")
            print(f"  Day Range: ${quote['lowPrice']:.2f} - ${quote['highPrice']:.2f}")
        
        # Get multiple quotes
        symbols = ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO"]
        quotes = await data_feed.get_quotes_batch(symbols)
        
        print(f"\nBanking Sector Quotes:")
        for symbol, quote in quotes.items():
            last = quote['lastTradePrice']
            change = quote['lastTradePrice'] - quote['lastTradePriceTrHrs']
            change_pct = (change / quote['lastTradePriceTrHrs']) * 100 if quote['lastTradePriceTrHrs'] > 0 else 0
            print(f"  {symbol}: ${last:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        candles = await data_feed.get_historical_data(
            "SHOP.TO",
            start_date,
            end_date,
            interval="OneDay"
        )
        
        if candles:
            print(f"\nHistorical Data for SHOP.TO (last 5 days):")
            for candle in candles[-5:]:
                date = candle['start'][:10]
                open_price = candle['open']
                high = candle['high']
                low = candle['low']
                close = candle['close']
                volume = candle['volume']
                print(f"  {date}: O=${open_price:.2f} H=${high:.2f} L=${low:.2f} C=${close:.2f} V={volume:,}")
        
    except QuestradeAPIError as e:
        print(f"API Error: {e}")


async def order_management_example(api: QuestradeAPI):
    """Example: Place and manage orders (DEMO - validation only)"""
    print("\n=== Order Management Example ===")
    
    order_manager = QuestradeOrderManager(api)
    
    try:
        # Get accounts
        accounts = await api.get_accounts()
        if not accounts:
            print("No accounts found")
            return
        
        account_id = accounts[0]["number"]
        
        # Example 1: Validate a market order
        symbol = "SHOP.TO"
        symbol_id = await order_manager.data_feed.get_symbol_id(symbol)
        
        if symbol_id:
            print(f"\nValidating market order for {symbol}:")
            validation_result = await api.validate_order(
                account_id=account_id,
                order_data={
                    "symbolId": symbol_id,
                    "orderType": "Market",
                    "action": "Buy",
                    "quantity": 10,
                    "timeInForce": "Day"
                },
                impact=True
            )
            print(f"  Validation result: {validation_result}")
        
        # Example 2: Get current orders
        open_orders = await api.get_account_orders(account_id, state_filter="Open")
        print(f"\nOpen Orders: {len(open_orders)}")
        for order in open_orders[:5]:  # Show first 5
            symbol = order.get("symbol", "Unknown")
            side = order["side"]
            quantity = order["totalQuantity"]
            order_type = order["orderType"]
            status = order["state"]
            print(f"  {symbol}: {side} {quantity} {order_type} - Status: {status}")
        
        # Example 3: Place a limit order (DEMO MODE)
        print("\nDemo: Placing limit order...")
        # In production, remove the validate_only parameter
        """
        result = await order_manager.place_limit_order(
            account_id=account_id,
            symbol="RY.TO",
            quantity=100,
            limit_price=150.00,
            action="Buy",
            time_in_force="Day"
        )
        print(f"Order placed: {result}")
        """
        
    except QuestradeAPIError as e:
        print(f"API Error: {e}")


async def streaming_example(api: QuestradeAPI):
    """Example: Stream real-time quotes"""
    print("\n=== Streaming Example ===")
    
    data_feed = QuestradeDataFeed(api)
    
    # Define callback for quote updates
    async def quote_callback(quote: Dict):
        symbol_id = quote.get("symbolId")
        last_price = quote.get("lastTradePrice")
        bid = quote.get("bidPrice")
        ask = quote.get("askPrice")
        volume = quote.get("volume")
        
        print(f"Quote Update - Symbol ID: {symbol_id}, Last: ${last_price:.2f}, "
              f"Bid: ${bid:.2f}, Ask: ${ask:.2f}, Volume: {volume:,}")
    
    try:
        # Stream quotes for multiple symbols
        symbols = ["SHOP.TO", "RY.TO", "CNR.TO"]
        print(f"Starting stream for: {symbols}")
        
        # Start streaming
        streaming_task = asyncio.create_task(
            data_feed.stream_quotes(symbols, quote_callback)
        )
        
        # Stream for 30 seconds
        await asyncio.sleep(30)
        
        # Stop streaming
        await api.stop_streaming()
        streaming_task.cancel()
        
        print("Streaming stopped")
        
    except QuestradeAPIError as e:
        print(f"Streaming Error: {e}")


async def advanced_trading_example(api: QuestradeAPI):
    """Example: Advanced trading strategies"""
    print("\n=== Advanced Trading Example ===")
    
    order_manager = QuestradeOrderManager(api)
    data_feed = QuestradeDataFeed(api)
    
    try:
        accounts = await api.get_accounts()
        if not accounts:
            return
        
        account_id = accounts[0]["number"]
        symbol = "SHOP.TO"
        
        # Get current quote
        quote = await data_feed.get_quote(symbol)
        if not quote:
            print(f"Could not get quote for {symbol}")
            return
        
        current_price = quote["lastTradePrice"]
        print(f"\nCurrent {symbol} price: ${current_price:.2f}")
        
        # Example: Bracket order calculation
        entry_price = current_price - 1.00  # Buy $1 below current
        stop_loss = entry_price * 0.98  # 2% stop loss
        take_profit = entry_price * 1.05  # 5% take profit
        
        print(f"\nBracket Order Strategy:")
        print(f"  Entry (Limit Buy): ${entry_price:.2f}")
        print(f"  Stop Loss: ${stop_loss:.2f} (-2%)")
        print(f"  Take Profit: ${take_profit:.2f} (+5%)")
        
        # In production, you would execute:
        """
        result = await order_manager.place_bracket_order(
            account_id=account_id,
            symbol=symbol,
            quantity=100,
            limit_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit
        )
        print(f"Bracket order result: {result}")
        """
        
        # Example: Options chain
        symbol_id = await data_feed.get_symbol_id(symbol)
        if symbol_id:
            options = await api.get_option_chain(symbol_id)
            if options:
                print(f"\nOptions Available: {len(options)} expiration dates")
        
    except QuestradeAPIError as e:
        print(f"API Error: {e}")


async def risk_management_example(api: QuestradeAPI):
    """Example: Risk management and position monitoring"""
    print("\n=== Risk Management Example ===")
    
    try:
        accounts = await api.get_accounts()
        if not accounts:
            return
        
        for account in accounts:
            account_id = account["number"]
            print(f"\nRisk Analysis for Account: {account_id}")
            
            # Get positions
            positions = await api.get_account_positions(account_id)
            
            if positions:
                total_market_value = 0
                position_risks = []
                
                for position in positions:
                    symbol = position["symbol"]
                    quantity = position["openQuantity"]
                    current_price = position["currentPrice"]
                    market_value = quantity * current_price
                    total_market_value += market_value
                    
                    position_risks.append({
                        "symbol": symbol,
                        "market_value": market_value,
                        "quantity": quantity,
                        "current_price": current_price
                    })
                
                # Calculate position concentrations
                print(f"  Total Market Value: ${total_market_value:,.2f}")
                print(f"  Position Concentration:")
                
                for risk in sorted(position_risks, key=lambda x: x["market_value"], reverse=True)[:5]:
                    concentration = (risk["market_value"] / total_market_value) * 100 if total_market_value > 0 else 0
                    print(f"    {risk['symbol']}: ${risk['market_value']:,.2f} ({concentration:.1f}%)")
                
                # Check for concentrated positions (>25%)
                concentrated = [r for r in position_risks if (r["market_value"] / total_market_value) > 0.25]
                if concentrated:
                    print(f"\n  ⚠️  Warning: Concentrated positions detected:")
                    for risk in concentrated:
                        print(f"    {risk['symbol']}: {(risk['market_value'] / total_market_value) * 100:.1f}%")
            
            # Get recent activities
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            activities = await api.get_account_activities(account_id, start_date, end_date)
            
            if activities:
                print(f"\n  Recent Activity (last 7 days): {len(activities)} transactions")
                
                # Calculate daily trading volume
                daily_volumes = {}
                for activity in activities:
                    date = activity["tradeDate"][:10]
                    amount = abs(activity.get("netAmount", 0))
                    daily_volumes[date] = daily_volumes.get(date, 0) + amount
                
                for date, volume in sorted(daily_volumes.items()):
                    print(f"    {date}: ${volume:,.2f}")
    
    except QuestradeAPIError as e:
        print(f"API Error: {e}")


async def main():
    """Run all examples"""
    # NOTE: You need to set your refresh token first
    # Get it from https://www.questrade.com/api/documentation/getting-started
    
    refresh_token = "your_refresh_token_here"  # Replace with your token
    
    # Create API instance
    api = QuestradeAPI(refresh_token=refresh_token)
    
    try:
        # Initialize connection
        await api.initialize()
        
        # Run examples
        await authentication_example()
        await account_info_example(api)
        await market_data_example(api)
        await order_management_example(api)
        # await streaming_example(api)  # Uncomment to test streaming
        await advanced_trading_example(api)
        await risk_management_example(api)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Clean up
        await api.close()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())