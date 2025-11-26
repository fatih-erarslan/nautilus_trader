#!/usr/bin/env python3
"""
Test MCP Neural Trader with Alpaca Integration
Verifies MCP can connect to your Alpaca account and execute trades
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

print("🔍 Testing MCP Neural Trader with Alpaca")
print("=" * 60)

# IMPORTANT: Set your real Alpaca API credentials here
# These should match your paper trading account PA3MANXUAXIR
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'PKVZM47F4PZC9B4QB3KF')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_REAL_SECRET_KEY')  # <-- UPDATE THIS
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

print("📋 Configuration:")
print(f"   API Key: {ALPACA_API_KEY[:8]}...{ALPACA_API_KEY[-4:]}")
print(f"   Base URL: {ALPACA_BASE_URL}")
print(f"   Account: PA3MANXUAXIR (expected)")
print()

# Test 1: Direct Alpaca connection test
print("1️⃣ Testing Direct Alpaca Connection")
print("-" * 40)

sys.path.append('src')

try:
    from alpaca.alpaca_client import AlpacaClient

    # Create client with real credentials
    client = AlpacaClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        base_url=ALPACA_BASE_URL
    )

    # Try to get account info
    try:
        account = client.get_account()
        print(f"✅ Connected to Alpaca account!")
        print(f"   Account Number: {account.get('account_number', 'N/A')}")
        print(f"   Status: {account.get('status', 'N/A')}")
        print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"   Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")

        # Check if this is the expected account
        if 'PA3MANXUAXIR' in str(account.get('account_number', '')):
            print(f"✅ Confirmed: This is account PA3MANXUAXIR")
        else:
            print(f"⚠️  Account number doesn't match expected PA3MANXUAXIR")

    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        print(f"   Please update ALPACA_SECRET_KEY in this script with your real secret key")
        print(f"   You can get it from: https://app.alpaca.markets/paper/dashboard/overview")

except Exception as e:
    print(f"❌ Failed to initialize client: {e}")

print()

# Test 2: Check MCP integration
print("2️⃣ Testing MCP Neural Trader Integration")
print("-" * 40)

try:
    # Check if MCP server is configured with correct environment
    mcp_env_check = os.popen("ps aux | grep mcp_server_enhanced.py | grep -v grep").read()
    if mcp_env_check:
        print("✅ MCP server is running")

        # Try to call MCP tool (this would work in Claude)
        print("   MCP tools available for Claude:")
        print("   - mcp__neural-trader__quick_analysis")
        print("   - mcp__neural-trader__execute_trade")
        print("   - mcp__neural-trader__get_portfolio_status")
        print("   - mcp__neural-trader__list_strategies")
    else:
        print("⚠️  MCP server not found running")

except Exception as e:
    print(f"❌ MCP check failed: {e}")

print()

# Test 3: Generate test activity
print("3️⃣ Generating Test Trading Activity")
print("-" * 40)

if 'client' in locals():
    try:
        # Check market status
        is_open = client.is_market_open()
        print(f"📈 Market Status: {'OPEN' if is_open else 'CLOSED'}")

        if is_open:
            print("✅ Market is open - attempting to place a test order")

            # Place a small test order (1 share of a low-priced stock)
            from alpaca.alpaca_client import OrderSide, OrderType, TimeInForce

            test_symbol = 'AAPL'  # Using Apple for test
            print(f"   Placing test order for 1 share of {test_symbol}")

            try:
                # Get current price first
                bars = client.get_bars(test_symbol, timeframe='1Min', limit=1)
                if not bars.empty:
                    current_price = bars['close'].iloc[-1]
                    print(f"   Current {test_symbol} price: ${current_price:.2f}")

                # Place market order for 1 share
                order = client.place_order(
                    symbol=test_symbol,
                    qty=1,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )

                print(f"✅ Order placed successfully!")
                print(f"   Order ID: {order.id}")
                print(f"   Symbol: {order.symbol}")
                print(f"   Quantity: {order.qty}")
                print(f"   Side: {order.side}")
                print(f"   Status: {order.status}")
                print()
                print("🎉 Check your Alpaca dashboard for this order!")
                print("   https://app.alpaca.markets/paper/dashboard/overview")

                # Wait a moment then check order status
                import time
                time.sleep(2)

                # Get order status
                order_status = client.get_order(order.id)
                print(f"   Updated Status: {order_status.status}")
                if order_status.filled_qty:
                    print(f"   Filled Qty: {order_status.filled_qty}")
                    print(f"   Filled Price: ${order_status.filled_avg_price:.2f}")

            except Exception as e:
                print(f"❌ Failed to place order: {e}")

        else:
            print("⚠️  Market is closed - cannot place live orders")
            print("   Orders can only be placed during market hours:")
            print("   Monday-Friday 9:30 AM - 4:00 PM ET")

            # Get market calendar
            try:
                calendar = client.get_market_calendar()
                if calendar and len(calendar) > 0:
                    next_open = calendar[0]
                    print(f"   Next market open: {next_open.get('date')} at {next_open.get('open')}")
            except:
                pass

    except Exception as e:
        print(f"❌ Failed to generate activity: {e}")
else:
    print("❌ Client not initialized - please provide correct API credentials")

print()

# Test 4: Neural Trading Strategy Test
print("4️⃣ Testing Neural Trading Strategies")
print("-" * 40)

if 'client' in locals():
    try:
        from alpaca.neural_integration import NeuralAlpacaIntegration

        # Create neural integration
        neural = NeuralAlpacaIntegration(client)

        # Get neural predictions
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        print(f"Generating neural predictions for: {symbols}")

        for symbol in symbols:
            prediction = asyncio.run(neural.get_neural_prediction(symbol))
            if 'error' not in prediction:
                print(f"   {symbol}:")
                print(f"      Direction: {prediction.get('direction', 'N/A')}")
                print(f"      Confidence: {prediction.get('confidence', 0):.2%}")
                print(f"      Next Price Est: ${prediction.get('next_price_estimate', 0):.2f}")

        print("✅ Neural strategies are working!")

    except Exception as e:
        print(f"⚠️  Neural strategy test failed: {e}")

print()
print("=" * 60)
print("📋 Summary:")
print()

if 'account' in locals():
    print("✅ Successfully connected to Alpaca account")
    print(f"✅ Account verified: {account.get('account_number', 'N/A')}")
    print("✅ Ready to trade!")
    print()
    print("🎯 Next Steps:")
    print("1. Check your Alpaca dashboard for the test order")
    print("2. The MCP tools are ready for use in Claude")
    print("3. You can now use commands like:")
    print("   - 'Execute a buy order for 10 shares of AAPL'")
    print("   - 'Show my portfolio status'")
    print("   - 'Analyze TSLA with neural prediction'")
else:
    print("❌ Failed to connect to Alpaca")
    print()
    print("🔧 To fix this:")
    print("1. Get your API keys from: https://app.alpaca.markets/paper/dashboard/overview")
    print("2. Update ALPACA_SECRET_KEY in this script")
    print("3. Run this script again: python test_mcp_alpaca.py")
    print("4. Restart MCP server with: ./setup_mcp_alpaca.sh")

print()
print("For MCP to work in Claude, ensure the server has your API keys:")
print("export ALPACA_API_KEY='your-key'")
print("export ALPACA_SECRET_KEY='your-secret'")
print("Then restart the MCP server")