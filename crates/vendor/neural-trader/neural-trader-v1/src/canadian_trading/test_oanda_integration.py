"""
Test and example usage of OANDA Canada integration
Demonstrates key features and functionality
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from canadian_trading import (
    OANDACanada, ForexSignal, ForexUtils,
    CurrencyStrength, OptimalTradingTimes
)


async def test_oanda_integration():
    """Test OANDA integration with demo account"""
    
    # Initialize OANDA client (use your demo credentials)
    print("=== Initializing OANDA Canada Client ===")
    oanda = OANDACanada(
        api_token="YOUR_DEMO_API_TOKEN",  # Replace with demo token
        account_id="YOUR_DEMO_ACCOUNT_ID",  # Replace with demo account
        environment="practice"
    )
    
    # Test 1: Account Information
    print("\n=== Account Information ===")
    if oanda.account_info:
        print(f"Account Currency: {oanda.account_info['currency']}")
        print(f"Balance: {oanda.account_info['balance']}")
        print(f"Margin Available: {oanda.margin_info.available_margin}")
        print(f"Margin Used: {oanda.margin_info.required_margin}")
    
    # Test 2: Get Current Prices
    print("\n=== Current Forex Prices ===")
    instruments = ['USD_CAD', 'EUR_CAD', 'GBP_CAD']
    for instrument in instruments:
        try:
            price = oanda._get_current_price(instrument)
            print(f"{instrument}: {price:.5f}")
        except Exception as e:
            print(f"Error getting price for {instrument}: {e}")
    
    # Test 3: Spread Analysis
    print("\n=== Spread Analysis ===")
    spread_analysis = oanda.get_spread_analysis('USD_CAD')
    print(f"USD/CAD Current Spread: {spread_analysis.current_spread:.1f} pips")
    print(f"Is Favorable for Trading: {spread_analysis.is_favorable}")
    
    # Test 4: Optimal Execution Time
    print("\n=== Optimal Execution Time ===")
    optimal_time = oanda.get_optimal_execution_time('USD_CAD', 'buy')
    print(f"Best time to trade USD/CAD: {optimal_time}")
    
    # Test 5: Create Trading Signal
    print("\n=== Creating Trading Signal ===")
    signal = ForexSignal(
        instrument="USD_CAD",
        direction="buy",
        confidence=0.75,
        entry_price=1.3500,
        stop_loss=1.3450,
        take_profit=1.3600,
        volatility=0.008,
        spread_impact=0.0002,
        optimal_execution_time=optimal_time,
        risk_reward_ratio=2.0,
        kelly_position_size=0.05,
        market_session="american"
    )
    
    # Test 6: Calculate Position Size
    print("\n=== Position Sizing (Kelly Criterion) ===")
    position_size = oanda.calculate_position_size(
        instrument="USD_CAD",
        signal=signal,
        risk_percentage=0.02
    )
    print(f"Recommended Position Size: {position_size:,} units")
    
    # Test 7: Get Open Positions
    print("\n=== Current Positions ===")
    positions = oanda.get_positions()
    if positions:
        for instrument, pos in positions.items():
            print(f"{instrument}:")
            print(f"  Units: {pos['units']:,}")
            print(f"  Avg Price: {pos['average_price']:.5f}")
            print(f"  Current Price: {pos['current_price']:.5f}")
            print(f"  P&L: ${pos['unrealized_pl']:.2f} ({pos['pl_percentage']:.2f}%)")
    else:
        print("No open positions")
    
    # Test 8: Risk Metrics
    print("\n=== Risk Analysis ===")
    if positions:
        risk_metrics = oanda.calculate_forex_risk_metrics(positions)
        print(f"Total Exposure: ${risk_metrics['total_exposure']:,.2f}")
        print(f"Margin Utilization: {risk_metrics['margin_utilization']:.1f}%")
        print(f"Largest Position: {risk_metrics['largest_position']['instrument']}")
        print(f"CAD Correlation Risk: {risk_metrics['correlation_risk']}")
        print(f"CAD Exposure: {risk_metrics['cad_exposure_percentage']:.1f}%")
        
        # Recommendations
        if risk_metrics['recommendations']:
            print("\nRisk Recommendations:")
            for rec in risk_metrics['recommendations']:
                print(f"  - {rec}")
    
    # Test 9: Margin Closeout Calculator
    print("\n=== Margin Closeout Analysis ===")
    if positions:
        closeout = oanda.get_margin_closeout_calculator(positions)
        print(f"Current Margin Level: {closeout['current_margin_level']:.1f}%")
        print(f"Closeout Level: {closeout['closeout_level']:.1f}%")
        print(f"Safety Buffer: {closeout['buffer_percentage']:.1f}%")
        
        if closeout['worst_case_scenarios']:
            print("\nWorst Case Scenarios:")
            for instrument, scenario in closeout['worst_case_scenarios'].items():
                print(f"  {instrument}:")
                print(f"    Closeout at: {scenario['closeout_price']:.5f}")
                print(f"    Move Required: {scenario['price_movement_pips']:.1f} pips")
    
    # Test 10: Execution Analytics
    print("\n=== Execution Analytics ===")
    analytics = oanda.get_execution_analytics(lookback_days=30)
    if analytics['average_slippage']:
        print(f"Average Slippage: {analytics['average_slippage']:.2f} pips")
        print(f"Positive Slippage Rate: {analytics['positive_slippage_rate']:.1%}")
        if analytics['best_execution_session']:
            print(f"Best Session: {analytics['best_execution_session'][0]} "
                  f"({analytics['best_execution_session'][1]:.2f} pips)")
    
    # Test 11: Stream Prices (Run for 10 seconds)
    print("\n=== Starting Price Streaming (10 seconds) ===")
    try:
        await oanda.start_streaming(['USD_CAD', 'EUR_CAD'])
        await asyncio.sleep(10)
        await oanda.stop_streaming()
        
        # Check cached prices
        if 'USD_CAD' in oanda.price_cache and oanda.price_cache['USD_CAD']:
            latest = oanda.price_cache['USD_CAD'][-1]
            print(f"Latest USD/CAD - Bid: {latest['bid']:.5f}, Ask: {latest['ask']:.5f}")
            print(f"Spread: {latest['spread_pips']:.1f} pips")
    except Exception as e:
        print(f"Streaming error: {e}")


def test_forex_utils():
    """Test forex utilities"""
    print("\n\n=== Testing Forex Utilities ===")
    
    forex_utils = ForexUtils()
    
    # Test 1: Currency Strength
    print("\n=== Currency Strength Analysis ===")
    
    # Create mock price data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    price_data = {
        'USD_CAD': list(1.35 + np.random.randn(100).cumsum() * 0.001),
        'EUR_CAD': list(1.45 + np.random.randn(100).cumsum() * 0.001),
        'GBP_CAD': list(1.70 + np.random.randn(100).cumsum() * 0.001),
        'AUD_CAD': list(0.90 + np.random.randn(100).cumsum() * 0.001),
        'CAD_JPY': list(110 + np.random.randn(100).cumsum() * 0.1)
    }
    
    strength = forex_utils.calculate_currency_strength(price_data, lookback_period=20)
    
    print("Currency Strength Scores:")
    for currency, analysis in sorted(strength.items(), 
                                   key=lambda x: x[1].strength_score, 
                                   reverse=True):
        print(f"  {currency}: {analysis.strength_score:+.2f} ({analysis.trend})")
    
    # Test 2: Pair Correlations
    print("\n=== Correlation Analysis ===")
    
    # Convert to pandas Series for correlation analysis
    price_series = {
        pair: pd.Series(prices, index=dates) 
        for pair, prices in price_data.items()
    }
    
    correlation = forex_utils.calculate_pair_correlations(price_series, rolling_window=30)
    
    print("Significant Correlations:")
    for pair1, pair2, corr in correlation.significant_correlations:
        print(f"  {pair1} <-> {pair2}: {corr:.3f}")
    
    if correlation.risk_warnings:
        print("\nRisk Warnings:")
        for warning in correlation.risk_warnings:
            print(f"  ⚠️ {warning}")
    
    # Test 3: Optimal Trading Times
    print("\n=== Optimal Trading Times ===")
    
    # Create mock OHLC data
    ohlc_data = pd.DataFrame({
        'timestamp': dates,
        'open': 1.35 + np.random.randn(100).cumsum() * 0.001,
        'high': 1.36 + np.random.randn(100).cumsum() * 0.001,
        'low': 1.34 + np.random.randn(100).cumsum() * 0.001,
        'close': 1.35 + np.random.randn(100).cumsum() * 0.001,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    optimal_times = forex_utils.analyze_optimal_trading_times('USD_CAD', ohlc_data)
    
    print(f"Best Trading Hours (UTC): {optimal_times.best_hours}")
    print(f"Best Sessions: {optimal_times.best_sessions}")
    
    # Test 4: Pattern Detection
    print("\n=== Pattern Detection ===")
    
    patterns = forex_utils.detect_forex_patterns(ohlc_data, min_confidence=0.7)
    
    if patterns:
        print(f"Detected {len(patterns)} patterns:")
        for pattern in patterns[:3]:  # Show first 3
            print(f"  Pattern: {pattern.pattern_name}")
            print(f"  Confidence: {pattern.confidence:.1%}")
            print(f"  Entry: {pattern.entry_price:.5f}")
            print(f"  Stop Loss: {pattern.stop_loss:.5f}")
            print(f"  Take Profit: {pattern.take_profit:.5f}")
            print(f"  Risk/Reward: {pattern.risk_reward_ratio:.2f}")
            print()
    else:
        print("No patterns detected in sample data")
    
    # Test 5: Economic Calendar Impact
    print("\n=== Economic Calendar Impact ===")
    
    # Mock economic events
    economic_events = [
        {
            'currency': 'CAD',
            'name': 'Bank of Canada Interest Rate Decision',
            'impact': 'high',
            'time': datetime.now() + timedelta(hours=2)
        },
        {
            'currency': 'USD',
            'name': 'Non-Farm Payrolls',
            'impact': 'high',
            'time': datetime.now() + timedelta(hours=6)
        }
    ]
    
    impact = forex_utils.analyze_economic_calendar_impact('USD_CAD', economic_events)
    
    print(f"Risk Level: {impact['risk_level'].upper()}")
    print(f"High Impact Events: {len(impact['high_impact_events'])}")
    print(f"Suggested Position Adjustment: {impact['suggested_position_adjustment']:.0%}")
    
    if impact['recommendations']:
        print("\nRecommendations:")
        for rec in impact['recommendations']:
            print(f"  [{rec['type'].upper()}] {rec['message']}")
            print(f"    Action: {rec['action']}")
    
    # Test 6: Carry Trade Analysis
    print("\n=== Carry Trade Opportunities ===")
    
    interest_rates = {
        'USD': 5.50,
        'CAD': 5.00,
        'EUR': 4.50,
        'GBP': 5.25,
        'JPY': -0.10,
        'AUD': 4.35,
        'NZD': 5.50,
        'CHF': 1.75
    }
    
    pairs = ['USD_CAD', 'EUR_CAD', 'CAD_JPY', 'AUD_CAD', 'GBP_CAD']
    carry_trades = forex_utils.calculate_carry_trade_opportunity(interest_rates, pairs)
    
    print("Top Carry Trade Opportunities:")
    for trade in carry_trades[:3]:
        print(f"  {trade['pair']} ({trade['direction'].upper()})")
        print(f"    Rate Differential: {trade['rate_differential']:.2f}%")
        print(f"    Annual Carry: {trade['annual_carry_percent']:.2f}%")
        print(f"    Monthly Carry: {trade['monthly_carry_percent']:.3f}%")
        print(f"    Risk Level: {trade['risk_warning']}")
        print()


async def main():
    """Run all tests"""
    print("="*60)
    print("OANDA Canada Forex Integration Test Suite")
    print("="*60)
    
    # Test OANDA integration
    await test_oanda_integration()
    
    # Test forex utilities
    test_forex_utils()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    # Run the async tests
    asyncio.run(main())