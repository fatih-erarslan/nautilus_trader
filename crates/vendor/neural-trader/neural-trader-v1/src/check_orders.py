#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, 'src')

from alpaca.alpaca_client import AlpacaClient

def check_orders():
    """Check recent orders and positions"""

    client = AlpacaClient(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL')
    )

    # Check recent orders
    orders = client.get_orders(status='all', limit=10)
    print('📋 Recent Orders:')
    print('=' * 60)

    if isinstance(orders, list):
        for order in orders[:10]:
            if isinstance(order, dict):
                print(f"  {order.get('symbol')} - {order.get('qty')} shares - {order.get('side')} - Status: {order.get('status')}")
                print(f"    Order ID: {order.get('id')}")
                print(f"    Submitted: {order.get('submitted_at')}")
                if order.get('filled_at'):
                    print(f"    Filled at: ${order.get('filled_avg_price', 'N/A')}")
                print()

    # Check positions
    positions = client.get_positions()
    print('\n📊 Current Positions:')
    print('=' * 60)
    if positions:
        for pos in positions:
            print(f"  {pos.get('symbol')}: {pos.get('qty')} shares @ ${pos.get('avg_entry_price')}")
            print(f"    Current Price: ${pos.get('current_price', 'N/A')}")
            print(f"    P&L: ${pos.get('unrealized_pl', 'N/A')}")
            print()
    else:
        print('  No open positions yet')

    # Check account
    account = client.get_account()
    print('\n💰 Account Summary:')
    print('=' * 60)
    print(f"  Buying Power: ${account.get('buying_power')}")
    print(f"  Cash: ${account.get('cash')}")
    print(f"  Portfolio Value: ${account.get('portfolio_value')}")

if __name__ == "__main__":
    check_orders()