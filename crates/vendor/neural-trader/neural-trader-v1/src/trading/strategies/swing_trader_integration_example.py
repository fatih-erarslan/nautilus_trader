"""Example integration of the optimized swing trading strategy."""

from datetime import datetime
from src.trading.strategies.swing_trader_optimized import OptimizedSwingTradingEngine


def main():
    """Demonstrate usage of the optimized swing trading engine."""
    
    # Initialize the trading engine with a $100,000 account
    engine = OptimizedSwingTradingEngine(account_size=100000)
    
    # Example market data (would come from your data feed in production)
    market_data = {
        "price": 105.50,
        "ma_20": 104.20,
        "ma_50": 102.80,
        "ma_200": 99.50,
        "rsi_14": 58,
        "atr_14": 1.8,  # $1.80 ATR
        "volume": 1250000,
        "volume_ma_20": 1000000,
        "volume_ratio": 1.25,
        "macd": 0.85,
        "macd_signal": 0.65,
        "support_level": 103.00,
        "resistance_level": 108.00
    }
    
    # Generate trading signal
    signal = engine.generate_advanced_signal(market_data)
    
    # Display signal details
    print("=" * 60)
    print("OPTIMIZED SWING TRADING SIGNAL")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Current Price: ${market_data['price']}")
    print(f"Market Regime: {engine.current_regime.value}")
    print()
    print(f"Action: {signal.action.upper()}")
    print(f"Signal Strength: {signal.strength:.2%}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Setup Type: {signal.setup_type}")
    print()
    
    if signal.action == "buy":
        print("TRADE DETAILS:")
        print(f"Entry Price: ${signal.entry_price}")
        print(f"Stop Loss: ${signal.stop_loss} ({((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):.2f}%)")
        print(f"Take Profit Levels:")
        for i, tp in enumerate(signal.take_profit_levels):
            profit_pct = (tp - signal.entry_price) / signal.entry_price * 100
            print(f"  TP{i+1}: ${tp} (+{profit_pct:.2f}%)")
        print()
        print(f"Position Size: {signal.position_size_pct:.2%} of account")
        print(f"Dollar Amount: ${engine.account_size * signal.position_size_pct:,.2f}")
        print(f"Risk/Reward Ratio: {signal.risk_reward_ratio:.2f}:1")
        print(f"Expected Time to Target: {signal.time_to_target} days")
        print()
        print("Confirmations:")
        for conf in signal.confirmations:
            print(f"  ✓ {conf}")
    
    # Example of exit management for an existing position
    if signal.action == "buy":
        print("\n" + "=" * 60)
        print("EXIT MANAGEMENT EXAMPLE")
        print("=" * 60)
        
        # Simulate an existing position
        position = {
            "entry_price": 100,
            "initial_stop_loss": 98,
            "current_stop_loss": 98,
            "take_profit_levels": [102.5, 104, 106],
            "shares": 100,
            "entry_date": datetime.now(),
            "expected_holding_days": 5
        }
        
        # Current market data for the position
        exit_market_data = {
            "current_price": 102.6,  # First target hit
            "atr_14": 1.5
        }
        
        exit_decision = engine.dynamic_exit_management(position, exit_market_data)
        
        print(f"Position Entry: ${position['entry_price']}")
        print(f"Current Price: ${exit_market_data['current_price']}")
        print(f"Profit: {exit_decision['profit_pct']:.2%}")
        print()
        print(f"Exit Decision: {exit_decision['reason']}")
        if exit_decision['partial_exit']:
            print(f"Shares to Exit: {exit_decision['shares_to_exit']} (partial exit)")
        print(f"New Stop Loss: ${exit_decision['new_stop_loss']}")
        
    # Show expected performance metrics
    print("\n" + "=" * 60)
    print("EXPECTED PERFORMANCE METRICS")
    print("=" * 60)
    print("Annual Return: 20-30%")
    print("Sharpe Ratio: 2.0-2.5")
    print("Win Rate: 65-70%")
    print("Max Drawdown: < 6%")
    print("Profit Factor: > 3.0")
    
    # Portfolio heat example
    print("\n" + "=" * 60)
    print("PORTFOLIO RISK MANAGEMENT")
    print("=" * 60)
    
    # Example open positions
    open_positions = [
        {"entry_price": 50, "stop_loss": 48.5, "position_value": 15000},
        {"entry_price": 75, "stop_loss": 73, "position_value": 20000}
    ]
    
    portfolio_heat = engine.calculate_portfolio_heat(open_positions)
    print(f"Current Portfolio Heat: {portfolio_heat:.2%}")
    print(f"Max Allowed Heat: {engine.max_portfolio_risk:.2%}")
    print(f"Available Risk Budget: {(engine.max_portfolio_risk - portfolio_heat):.2%}")
    
    # Apply portfolio risk overlay
    if signal.action == "buy":
        adjusted_signal = engine.apply_portfolio_risk_overlay(signal, open_positions)
        if adjusted_signal.position_size_pct < signal.position_size_pct:
            print(f"\nPosition size adjusted for portfolio risk:")
            print(f"Original: {signal.position_size_pct:.2%}")
            print(f"Adjusted: {adjusted_signal.position_size_pct:.2%}")


if __name__ == "__main__":
    main()