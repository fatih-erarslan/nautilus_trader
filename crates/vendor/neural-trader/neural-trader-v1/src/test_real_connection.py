#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, 'src')

from alpaca.alpaca_client import AlpacaClient

def test_real_connection():
    """Test real connection to Alpaca with new credentials"""

    print("🔍 Testing Real Alpaca Connection")
    print("=" * 60)

    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL')

    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"Secret: {secret_key[:10]}..." if secret_key else "No secret key")
    print(f"Base URL: {base_url}")

    # Initialize client
    client = AlpacaClient(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url
    )

    # Check account
    print("\n📊 Account Status:")
    account = client.get_account()
    if account:
        print(f"✅ Successfully connected to Alpaca!")
        print(f"   Account Number: {account.get('account_number')}")
        print(f"   Status: {account.get('status')}")
        print(f"   Buying Power: ${account.get('buying_power')}")
        print(f"   Cash: ${account.get('cash')}")
        print(f"   Portfolio Value: ${account.get('portfolio_value')}")

        # Check positions
        print("\n📈 Current Positions:")
        positions = client.get_positions()
        if positions:
            for pos in positions:
                print(f"   {pos.get('symbol')}: {pos.get('qty')} shares @ ${pos.get('avg_entry_price')}")
        else:
            print("   No open positions")

        # Place a real order
        print("\n🚀 Placing a real market order for AAPL...")
        order = client.place_order(
            symbol='AAPL',
            qty=1,
            side='buy',
            order_type='market',
            time_in_force='day'
        )

        if order:
            print(f"✅ Order placed successfully!")
            print(f"   Order ID: {order.get('id')}")
            print(f"   Symbol: {order.get('symbol')}")
            print(f"   Quantity: {order.get('qty')}")
            print(f"   Status: {order.get('status')}")
            print(f"\n✨ Check your Alpaca dashboard to see the order!")
        else:
            print("❌ Failed to place order")
    else:
        print("❌ Failed to connect to Alpaca")
        print("Please check your credentials in .env file")

if __name__ == "__main__":
    test_real_connection()