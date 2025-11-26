#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print('Environment Check:')
print(f'ALPACA_API_KEY: {os.getenv("ALPACA_API_KEY")}')
print(f'ALPACA_SECRET_KEY: {os.getenv("ALPACA_SECRET_KEY")[:10]}...' if os.getenv('ALPACA_SECRET_KEY') else 'No secret')
print(f'ALPACA_BASE_URL: {os.getenv("ALPACA_BASE_URL")}')

# Add src to path
sys.path.insert(0, 'src')

from alpaca.mcp_integration import get_mcp_bridge

bridge = get_mcp_bridge()
print(f'\nBridge Status:')
print(f'Demo Mode: {bridge.demo_mode}')
print(f'Has Credentials: {bridge.has_credentials}')

portfolio = bridge.get_portfolio_status()
print(f'\nPortfolio:')
print(f'Demo Mode: {portfolio.get("demo_mode")}')
print(f'Account: {portfolio.get("account_number")}')
print(f'Buying Power: ${portfolio.get("buying_power")}')
print(f'Cash: ${portfolio.get("cash")}')

# Test placing an order
print('\n🚀 Testing Trade Execution:')
result = bridge.execute_trade(
    symbol='TSLA',
    action='buy',
    quantity=1,
    strategy='momentum'
)
print(f'Trade Result:')
print(f'  Demo Mode: {result.get("demo_mode")}')
print(f'  Status: {result.get("status")}')
print(f'  Order ID: {result.get("order_id")}')
print(f'  Message: {result.get("message")}')