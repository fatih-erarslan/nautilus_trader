"""Example usage of the Trading Decision Engine and Strategies."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.news_trading.decision_engine import NewsDecisionEngine


async def main():
    """Demonstrate the Trading Decision Engine."""
    
    # Initialize the decision engine
    engine = NewsDecisionEngine(account_size=100000, max_portfolio_risk=0.2)
    
    print("=" * 60)
    print("AI News Trading Decision Engine Demo")
    print("=" * 60)
    
    # Example 1: Process bullish news sentiment
    print("\n1. Processing Bullish News Sentiment for Apple")
    bullish_sentiment = {
        "asset": "AAPL",
        "sentiment_score": 0.85,
        "confidence": 0.9,
        "market_impact": {
            "direction": "bullish",
            "magnitude": 0.8,
            "timeframe": "short-term"
        },
        "source_events": ["earnings-beat-q4-2024"],
        "entities": ["Apple", "iPhone", "Record Sales"]
    }
    
    signal = await engine.process_sentiment(bullish_sentiment)
    if signal:
        print(f"  Signal Generated: {signal.signal_type.value}")
        print(f"  Strategy: {signal.strategy.value}")
        print(f"  Entry Price: ${signal.entry_price:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Position Size: {signal.position_size:.1%} of portfolio")
        print(f"  Risk Level: {signal.risk_level.value}")
        print(f"  Holding Period: {signal.holding_period}")
    
    # Example 2: Process bearish crypto sentiment
    print("\n2. Processing Bearish Crypto Sentiment for Bitcoin")
    bearish_crypto = {
        "asset": "BTC",
        "asset_type": "crypto",
        "sentiment_score": -0.7,
        "confidence": 0.8,
        "market_impact": {
            "direction": "bearish",
            "magnitude": 0.6
        }
    }
    
    crypto_signal = await engine.process_sentiment(bearish_crypto)
    if crypto_signal:
        print(f"  Signal Generated: {crypto_signal.signal_type.value}")
        print(f"  Asset Type: {crypto_signal.asset_type.value}")
        print(f"  Risk Level: {crypto_signal.risk_level.value}")
        print(f"  Position Size: {crypto_signal.position_size:.1%}")
    
    # Example 3: Portfolio rebalancing
    print("\n3. Evaluating Portfolio for Rebalancing")
    current_positions = {
        "NVDA": {
            "size": 0.3,
            "entry_price": 400,
            "current_price": 520,
            "unrealized_pnl": 0.30  # 30% gain
        },
        "TSLA": {
            "size": 0.2,
            "entry_price": 200,
            "current_price": 170,
            "unrealized_pnl": -0.15  # 15% loss
        }
    }
    
    rebalance_signals = await engine.evaluate_portfolio(current_positions)
    for signal in rebalance_signals:
        print(f"  {signal.asset}: {signal.signal_type.value} - {signal.reasoning}")
    
    # Example 4: Technical analysis signals
    print("\n4. Processing Market Data for Technical Signals")
    market_data = {
        "MSFT": {
            "price": 400,
            "price_change_5d": 0.08,
            "price_change_20d": 0.15,
            "volume_ratio": 1.8,
            "ma_50": 385,
            "ma_200": 370,
            "rsi": 62,
            "atr": 8
        }
    }
    
    tech_signals = await engine.process_market_data(market_data)
    for signal in tech_signals:
        print(f"  {signal.asset}: {signal.strategy.value} strategy detected")
        print(f"    Entry: ${signal.entry_price:.2f}, Target: ${signal.take_profit:.2f}")
    
    # Example 5: Risk parameter adjustment
    print("\n5. Adjusting Risk Parameters")
    engine.set_risk_parameters({
        "max_position_size": 0.05,  # Reduce to 5% max per position
        "max_portfolio_risk": 0.15  # Reduce to 15% total risk
    })
    print("  Risk parameters updated: Max position 5%, Max portfolio risk 15%")
    
    # Example 6: Get active signals
    print(f"\n6. Active Signals: {len(engine.get_active_signals())} signals")
    for signal in engine.get_active_signals():
        print(f"  {signal.asset}: {signal.signal_type.value} ({signal.strategy.value})")


if __name__ == "__main__":
    asyncio.run(main())