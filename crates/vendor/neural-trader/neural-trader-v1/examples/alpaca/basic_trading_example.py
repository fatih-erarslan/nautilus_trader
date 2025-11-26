"""
Basic Alpaca Trading Example
A simple example showing how to use the Alpaca client and trading strategies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpaca.alpaca_client import AlpacaClient, OrderSide, OrderType
from alpaca.trading_strategies import MomentumStrategy, MeanReversionStrategy, TradingBot
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main example function"""

    print("🚀 Alpaca Trading Example")
    print("=" * 50)

    try:
        # Initialize Alpaca client
        client = AlpacaClient()
        print("✅ Alpaca client initialized successfully")

        # Test connection and get account info
        account = client.get_account()
        print(f"\n📊 Account Information:")
        print(f"   Status: {account.get('status')}")
        print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"   Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")

        # Check if market is open
        is_open = client.is_market_open()
        print(f"   Market Open: {'Yes' if is_open else 'No'}")

        # Get current positions
        positions = client.get_positions()
        print(f"\n📈 Current Positions ({len(positions)}):")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.qty} shares, ${float(pos.market_value):,.2f} value, "
                  f"P&L: ${float(pos.unrealized_pl):,.2f} ({float(pos.unrealized_plpc)*100:.2f}%)")

        # Test market data
        print(f"\n📊 Market Data Example:")
        symbols = ['AAPL', 'GOOGL', 'MSFT']

        for symbol in symbols:
            try:
                # Get latest quote
                quote = client.get_latest_quote(symbol)
                if quote:
                    print(f"   {symbol}: Bid ${quote.get('bp', 0):.2f} / Ask ${quote.get('ap', 0):.2f}")

                # Get recent bars
                bars = client.get_bars(symbol, timeframe='1Day', limit=5)
                if not bars.empty:
                    latest_close = bars['close'].iloc[-1]
                    prev_close = bars['close'].iloc[-2]
                    change = (latest_close - prev_close) / prev_close * 100
                    print(f"   {symbol}: ${latest_close:.2f} ({change:+.2f}%)")

            except Exception as e:
                print(f"   Error getting data for {symbol}: {e}")

        # Example: Trading Strategy Demonstration
        print(f"\n🤖 Trading Strategy Demo:")

        # Initialize trading bot
        bot = TradingBot(client)

        # Add momentum strategy
        momentum_strategy = MomentumStrategy(client, lookback_days=20)
        bot.add_strategy(momentum_strategy)

        # Add mean reversion strategy
        mean_reversion_strategy = MeanReversionStrategy(client, lookback_days=20)
        bot.add_strategy(mean_reversion_strategy)

        # Run strategies once (demo mode)
        print("   Running strategies in demo mode...")
        trading_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

        # Get market data for analysis
        data = bot.get_market_data(trading_symbols, days=30)

        for strategy in bot.strategies:
            print(f"\n   📈 {strategy.name} Strategy Signals:")
            signals = strategy.generate_signals(data)

            if signals:
                for signal in signals:
                    print(f"      {signal.symbol}: {signal.action.upper()} "
                          f"(strength: {signal.strength:.2f}) - {signal.reason}")
            else:
                print("      No signals generated")

        # Portfolio summary
        summary = bot.get_portfolio_summary()
        print(f"\n💼 Portfolio Summary:")
        print(f"   Total Value: ${summary['total_value']:,.2f}")
        print(f"   Cash: ${summary['cash']:,.2f}")
        print(f"   Number of Positions: {summary['num_positions']}")

        if summary['positions']:
            print(f"   Position Breakdown:")
            for pos in summary['positions']:
                print(f"      {pos['symbol']}: ${float(pos['market_value']):,.2f} "
                      f"({pos['weight']*100:.1f}%) - P&L: ${float(pos['unrealized_pl']):,.2f}")

        # Example: Place a small test order (commented out for safety)
        """
        if is_open and float(account.get('buying_power', 0)) > 100:
            print(f"\n🛒 Example Order (Paper Trading):")
            try:
                # Place a small market order for 1 share of AAPL
                order = client.place_order(
                    symbol='AAPL',
                    qty=1,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET
                )
                print(f"   Order placed: {order.id} - {order.side} {order.qty} {order.symbol}")

                # Cancel the order immediately (if it hasn't filled)
                time.sleep(1)
                try:
                    client.cancel_order(order.id)
                    print(f"   Order cancelled: {order.id}")
                except:
                    print(f"   Order may have already filled")

            except Exception as e:
                print(f"   Error placing test order: {e}")
        """

        print(f"\n✅ Example completed successfully!")
        print(f"\n💡 Next Steps:")
        print(f"   1. Configure your .env file with real API keys")
        print(f"   2. Test with paper trading first")
        print(f"   3. Implement risk management rules")
        print(f"   4. Backtest strategies before going live")
        print(f"   5. Monitor performance and adjust parameters")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Example failed: {e}")

if __name__ == "__main__":
    main()