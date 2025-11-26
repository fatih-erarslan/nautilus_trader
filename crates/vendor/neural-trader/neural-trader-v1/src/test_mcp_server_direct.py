#!/usr/bin/env python3
"""Test MCP server integration directly"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root .env
env_path = Path(__file__).parent.parent / '.env'
print(f"Loading .env from: {env_path}")
load_dotenv(env_path, override=True)

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"ALPACA_API_KEY: {os.getenv('ALPACA_API_KEY')}")
print(f"ALPACA_SECRET_KEY: {os.getenv('ALPACA_SECRET_KEY')[:20]}..." if os.getenv('ALPACA_SECRET_KEY') else "NOT SET")
print(f"ALPACA_BASE_URL: {os.getenv('ALPACA_BASE_URL')}")

# Now import the MCP server modules
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "=" * 60)
print("TESTING ALPACA INTEGRATION")
print("=" * 60)

try:
    from alpaca.mcp_integration import get_mcp_bridge, ALPACA_INTEGRATION_AVAILABLE

    print(f"Alpaca Integration Available: {ALPACA_INTEGRATION_AVAILABLE}")

    if ALPACA_INTEGRATION_AVAILABLE:
        bridge = get_mcp_bridge()
        print(f"Bridge Demo Mode: {bridge.demo_mode}")
        print(f"Bridge Has Credentials: {bridge.has_credentials}")

        # Test portfolio
        portfolio = bridge.get_portfolio_status()
        print(f"\nPortfolio Status:")
        print(f"  Demo Mode: {portfolio.get('demo_mode')}")
        print(f"  Account: {portfolio.get('account_number')}")
        print(f"  Buying Power: ${portfolio.get('buying_power')}")

        # Test trade execution
        print(f"\nTesting Trade Execution:")
        result = bridge.execute_trade(
            symbol='AMD',
            action='buy',
            quantity=1,
            strategy='momentum'
        )
        print(f"  Demo Mode: {result.get('demo_mode')}")
        print(f"  Status: {result.get('status')}")
        if not result.get('demo_mode'):
            print(f"  Order ID: {result.get('order_id')}")
            print(f"  Message: {result.get('message')}")

except Exception as e:
    print(f"Error loading Alpaca integration: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TESTING MCP SERVER MODULE")
print("=" * 60)

# Test what the MCP server sees
try:
    # Import the MCP server module
    from mcp import mcp_server_enhanced

    # Check if the module has the integration flag
    print(f"MCP Server ALPACA_INTEGRATION_AVAILABLE: {getattr(mcp_server_enhanced, 'ALPACA_INTEGRATION_AVAILABLE', 'NOT SET')}")

    # Try calling get_portfolio_status directly
    if hasattr(mcp_server_enhanced, 'get_portfolio_status'):
        print("\nCalling get_portfolio_status directly:")
        result = mcp_server_enhanced.get_portfolio_status(include_analytics=True)
        print(f"  Demo Mode: {result.get('demo_mode')}")
        print(f"  Account: {result.get('account_number', 'N/A')}")

    # Try calling execute_trade directly
    if hasattr(mcp_server_enhanced, 'execute_trade'):
        print("\nCalling execute_trade directly:")
        result = mcp_server_enhanced.execute_trade(
            strategy='momentum_trading_optimized',
            symbol='AAPL',
            action='buy',
            quantity=1
        )
        print(f"  Demo Mode: {result.get('demo_mode')}")
        print(f"  Trade ID: {result.get('trade_id')}")

except Exception as e:
    print(f"Error testing MCP server: {e}")
    import traceback
    traceback.print_exc()