# OANDA Canada Forex Trading Integration

This module provides a complete production-ready integration with OANDA Canada for forex trading, including advanced features for the AI News Trading Platform.

## Features

### OANDA v20 REST API Integration
- Full v20 API implementation with async support
- Real-time WebSocket streaming for live rates
- Comprehensive order management (market, limit, stop orders)
- Position management with trailing stops
- Advanced risk management tools

### CAD Pairs Specialization
- Optimized for Canadian Dollar pairs (USD/CAD, EUR/CAD, GBP/CAD, etc.)
- Session-based trading analysis for optimal execution
- CAD-specific margin calculations
- Correlation analysis for CAD exposure management

### Advanced Forex Features
- **Spread Analysis**: Real-time spread monitoring and historical analysis
- **Optimal Execution Timing**: AI-driven execution time recommendations
- **Currency Strength Analysis**: Multi-pair strength calculations
- **Pattern Recognition**: Automated detection of forex patterns
- **Risk Metrics**: VaR, margin utilization, correlation risk
- **Carry Trade Analysis**: Interest rate differential opportunities

## Installation

```bash
pip install oandapyV20 pandas numpy scipy scikit-learn websockets
```

## Quick Start

### 1. Initialize OANDA Client

```python
from canadian_trading import OANDACanada, ForexSignal

# Initialize client
oanda = OANDACanada(
    api_token="your_api_token",
    account_id="your_account_id",
    environment="practice"  # or "live"
)

# Start streaming real-time rates
await oanda.start_streaming(['USD_CAD', 'EUR_CAD', 'GBP_CAD'])
```

### 2. Execute Trading Signals

```python
# Create a trading signal
signal = ForexSignal(
    instrument="USD_CAD",
    direction="buy",
    confidence=0.85,
    entry_price=1.3500,
    stop_loss=1.3450,
    take_profit=1.3600,
    volatility=0.008,
    spread_impact=0.0002,
    optimal_execution_time=datetime.now(),
    risk_reward_ratio=2.0,
    kelly_position_size=0.05,
    market_session="american"
)

# Execute the signal
result = await oanda.execute_forex_signal(signal)
```

### 3. Position Management

```python
# Get all open positions
positions = oanda.get_positions()

# Modify position (adjust stops)
oanda.modify_position(
    instrument="USD_CAD",
    stop_loss=1.3480,
    trailing_stop=0.0020  # 20 pip trailing stop
)

# Close position
oanda.close_position("USD_CAD", units=5000)  # Partial close
```

### 4. Risk Management

```python
# Calculate position size using Kelly Criterion
position_size = oanda.calculate_position_size(
    instrument="USD_CAD",
    signal=signal,
    risk_percentage=0.02  # 2% risk per trade
)

# Get risk metrics
risk_metrics = oanda.calculate_forex_risk_metrics(positions)
print(f"Margin utilization: {risk_metrics['margin_utilization']}%")
print(f"CAD exposure: {risk_metrics['cad_exposure_percentage']}%")

# Margin closeout calculator
closeout_info = oanda.get_margin_closeout_calculator(positions)
```

### 5. Spread Analysis

```python
# Get current spread analysis
spread_analysis = oanda.get_spread_analysis("USD_CAD")
print(f"Current spread: {spread_analysis.current_spread} pips")
print(f"Is favorable: {spread_analysis.is_favorable}")

# Session-based analysis
print(f"Best session spreads: {spread_analysis.session_analysis}")
```

## Forex Utilities

The module includes comprehensive forex utilities for advanced analysis:

### Currency Strength Analysis

```python
from canadian_trading import ForexUtils

forex_utils = ForexUtils()

# Calculate currency strength
strength = forex_utils.calculate_currency_strength(
    price_data={'USD_CAD': prices, 'EUR_CAD': prices2},
    lookback_period=20
)

for currency, analysis in strength.items():
    print(f"{currency}: Strength={analysis.strength_score:.2f}")
```

### Correlation Analysis

```python
# Analyze pair correlations
correlation = forex_utils.calculate_pair_correlations(
    price_data={'USD_CAD': prices, 'EUR_CAD': prices2},
    rolling_window=30
)

print(f"Significant correlations: {correlation.significant_correlations}")
print(f"Risk warnings: {correlation.risk_warnings}")
```

### Pattern Detection

```python
# Detect forex patterns
patterns = forex_utils.detect_forex_patterns(
    price_data=ohlc_dataframe,
    min_confidence=0.7
)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_name}")
    print(f"Entry: {pattern.entry_price}, SL: {pattern.stop_loss}")
```

### Optimal Trading Times

```python
# Find best trading times
optimal_times = forex_utils.analyze_optimal_trading_times(
    pair="USD_CAD",
    historical_data=price_df,
    spread_data=spread_df
)

print(f"Best hours: {optimal_times.best_hours}")
print(f"Best sessions: {optimal_times.best_sessions}")
```

## Advanced Features

### Economic Calendar Integration

```python
# Analyze economic impact
impact = forex_utils.analyze_economic_calendar_impact(
    pair="USD_CAD",
    economic_events=upcoming_events
)

print(f"Risk level: {impact['risk_level']}")
print(f"Position adjustment: {impact['suggested_position_adjustment']}")
```

### Carry Trade Analysis

```python
# Find carry trade opportunities
interest_rates = {
    'USD': 5.25, 'CAD': 5.00, 'EUR': 4.00,
    'GBP': 5.25, 'JPY': -0.10, 'AUD': 4.35
}

carry_trades = forex_utils.calculate_carry_trade_opportunity(
    interest_rates=interest_rates,
    pairs=['USD_CAD', 'EUR_CAD', 'CAD_JPY']
)
```

### Execution Analytics

```python
# Get execution quality metrics
analytics = oanda.get_execution_analytics(lookback_days=30)
print(f"Average slippage: {analytics['average_slippage']} pips")
print(f"Best execution session: {analytics['best_execution_session']}")
```

## WebSocket Streaming

The integration includes real-time WebSocket streaming for live forex rates:

```python
# Start streaming
await oanda.start_streaming(['USD_CAD', 'EUR_CAD'])

# Price updates are automatically processed and stored
# Access latest prices from cache
latest_prices = oanda.price_cache['USD_CAD'][-1]
print(f"Bid: {latest_prices['bid']}, Ask: {latest_prices['ask']}")

# Stop streaming
await oanda.stop_streaming()
```

## Risk Management Best Practices

1. **Position Sizing**: Use Kelly Criterion with conservative fraction (0.25)
2. **Margin Management**: Keep margin utilization below 50%
3. **Correlation Risk**: Monitor CAD exposure across all pairs
4. **Stop Loss**: Always use stops, consider trailing stops for trends
5. **Economic Events**: Reduce position size during high-impact events

## Error Handling

The module includes comprehensive error handling:

```python
try:
    result = await oanda.execute_forex_signal(signal)
    if result['status'] == 'success':
        print(f"Order executed: {result['order_id']}")
    else:
        print(f"Execution failed: {result['reason']}")
except Exception as e:
    logger.error(f"Trading error: {e}")
```

## Performance Optimization

- Streaming connections are persistent and auto-reconnect
- Price data is cached for fast access
- Spread analysis runs asynchronously
- Position calculations use vectorized operations

## Compliance

- Supports CIRO compliance requirements
- Includes position limit checks
- Automated margin monitoring
- Trade audit logging

## Support

For issues or questions:
- Check OANDA v20 API documentation
- Review error logs for detailed messages
- Ensure API credentials have proper permissions
- Verify account has sufficient margin

## License

This module is part of the AI News Trading Platform and follows the main project license.