# 🛡️ Advanced Stop Loss Strategies for Neural Trading

## Overview
Stop losses are critical for protecting capital and managing risk. This guide provides multiple stop loss strategies optimized for your mirror trading system.

## 1. 📊 Percentage-Based Stop Loss

### Fixed Percentage Stop
```python
def fixed_percentage_stop(entry_price, stop_percent=0.06):
    """
    Simple fixed percentage stop loss
    Default: 6% below entry (your current setting)
    """
    return entry_price * (1 - stop_percent)

# Example for your positions
positions = {
    'SPY': {'entry': 667.00, 'stop': 627.38},  # -6%
    'NVDA': {'entry': 180.00, 'stop': 169.20}, # -6%
    'TSLA': {'entry': 437.00, 'stop': 410.78}  # -6%
}
```

### Volatility-Adjusted Percentage Stop
```python
def volatility_adjusted_stop(entry_price, atr, multiplier=2):
    """
    Adjusts stop based on Average True Range (ATR)
    More volatile stocks get wider stops
    """
    stop_distance = atr * multiplier
    return entry_price - stop_distance

# Recommended multipliers by volatility
volatility_stops = {
    'low_vol': 1.5,    # SPY, AAPL
    'medium_vol': 2.0, # MSFT, GOOGL
    'high_vol': 2.5    # NVDA, TSLA
}
```

## 2. 🎯 Trailing Stop Loss

### Classic Trailing Stop
```python
class TrailingStop:
    def __init__(self, trail_percent=0.05):
        self.trail_percent = trail_percent
        self.highest_price = 0
        self.stop_price = 0

    def update(self, current_price):
        """Updates stop as price moves up"""
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.stop_price = current_price * (1 - self.trail_percent)
        return self.stop_price

# Example: 5% trailing stop
# Buy at $100 → Stop at $95
# Price rises to $110 → Stop moves to $104.50
# Price drops to $105 → Stop stays at $104.50
```

### Chandelier Exit (ATR-Based Trailing)
```python
def chandelier_exit(high_price, atr, multiplier=3):
    """
    Trails based on highest high minus ATR multiple
    Better for trending markets
    """
    return high_price - (atr * multiplier)

# Recommended settings
chandelier_settings = {
    'conservative': 3.0,  # 3x ATR
    'moderate': 2.5,      # 2.5x ATR
    'aggressive': 2.0     # 2x ATR
}
```

## 3. 🔄 Dynamic Stop Loss Strategies

### Parabolic SAR Stop
```python
class ParabolicSAR:
    """
    Accelerating stop that follows trend
    Tightens as trend strengthens
    """
    def __init__(self, af_start=0.02, af_increment=0.02, af_max=0.20):
        self.af = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        self.ep = 0  # Extreme point
        self.sar = 0

    def calculate(self, high, low, is_long):
        if is_long:
            if high > self.ep:
                self.ep = high
                self.af = min(self.af + self.af_increment, self.af_max)
            self.sar = self.sar + self.af * (self.ep - self.sar)
        return self.sar
```

### Support Level Stop
```python
def support_level_stop(price_data, lookback=20):
    """
    Places stop below recent support level
    More intelligent than fixed percentage
    """
    recent_lows = price_data[-lookback:].min()
    support = recent_lows * 0.99  # 1% below support
    return support

# Key support levels for your positions
support_stops = {
    'SPY': 650,   # 20-day support
    'NVDA': 175,  # Recent consolidation
    'TSLA': 430   # Psychological level
}
```

## 4. ⚡ Regime-Based Stop Loss

### Market Regime Adaptive Stops
```python
def regime_based_stop(entry_price, market_regime, vix_level):
    """
    Adjusts stop based on market conditions
    Tighter in bear markets, wider in bull
    """
    base_stop = 0.06  # Your default

    regime_multipliers = {
        'BULLISH': 1.2,    # Wider stop (7.2%)
        'BEARISH': 0.67,   # Tighter stop (4%)
        'SIDEWAYS': 0.83   # Medium stop (5%)
    }

    # VIX adjustment
    if vix_level > 20:
        vix_adj = 1.3  # Wider during high volatility
    elif vix_level < 15:
        vix_adj = 0.9  # Tighter during low volatility
    else:
        vix_adj = 1.0

    stop_percent = base_stop * regime_multipliers[market_regime] * vix_adj
    return entry_price * (1 - stop_percent)
```

## 5. 🎪 Multi-Layer Stop System

### Three-Tier Stop Loss
```python
class MultiLayerStop:
    """
    Multiple stop levels with position scaling
    Reduces risk while maintaining upside
    """
    def __init__(self, entry_price, position_size):
        self.entry = entry_price
        self.size = position_size

        # Three stop levels
        self.stops = {
            'tight': {
                'price': entry_price * 0.97,   # -3%
                'exit_percent': 0.33            # Exit 1/3
            },
            'medium': {
                'price': entry_price * 0.94,   # -6%
                'exit_percent': 0.50            # Exit 1/2 of remaining
            },
            'wide': {
                'price': entry_price * 0.91,   # -9%
                'exit_percent': 1.0             # Exit all
            }
        }

    def check_stops(self, current_price):
        """Returns how much to sell at current price"""
        for level in ['tight', 'medium', 'wide']:
            if current_price <= self.stops[level]['price']:
                return self.stops[level]['exit_percent']
        return 0
```

## 6. 🤖 AI-Enhanced Stop Loss

### Neural Network Predicted Stop
```python
def neural_stop_predictor(symbol, entry_price, market_data):
    """
    Uses ML to predict optimal stop placement
    Based on historical patterns and current conditions
    """
    features = {
        'volatility': calculate_atr(market_data),
        'trend_strength': calculate_adx(market_data),
        'support_distance': find_nearest_support(market_data),
        'volume_profile': analyze_volume(market_data),
        'correlation': calculate_market_correlation(symbol)
    }

    # Neural network prediction (simplified)
    optimal_stop_percent = predict_stop_level(features)
    return entry_price * (1 - optimal_stop_percent)
```

## 7. ⏰ Time-Based Stop Loss

### Time Decay Stop
```python
def time_decay_stop(entry_price, days_held, initial_stop=0.06):
    """
    Tightens stop over time if trade doesn't work
    Forces exit from non-performing positions
    """
    decay_rate = 0.005  # 0.5% per day
    current_stop = initial_stop - (days_held * decay_rate)
    min_stop = 0.02  # Never tighter than 2%

    final_stop = max(current_stop, min_stop)
    return entry_price * (1 - final_stop)

# Example timeline
# Day 1: -6% stop
# Day 5: -4% stop
# Day 10: -2% stop (minimum reached)
```

## 8. 📈 Options-Based Stop Protection

### Protective Put Strategy
```python
def protective_put_stop(stock_price, put_strike, put_premium):
    """
    Uses options for guaranteed stop price
    Costs premium but provides absolute protection
    """
    max_loss = stock_price - put_strike + put_premium
    protection_cost = (put_premium / stock_price) * 100

    return {
        'guaranteed_stop': put_strike,
        'max_loss_per_share': max_loss,
        'cost_percent': protection_cost
    }

# Example for your SPY position
spy_protection = {
    'stock_price': 667,
    'put_strike': 650,    # -2.5% protection
    'put_premium': 5.50,
    'max_loss': 22.50,     # $17 + $5.50 premium
    'cost': 0.82           # 0.82% insurance cost
}
```

## 🎯 Recommended Stop Loss Strategy for Your Portfolio

Based on your current positions and market conditions:

### Position-Specific Recommendations

| Symbol | Current | Strategy | Stop Price | Reason |
|--------|---------|----------|------------|--------|
| **SPY** | $667.00 | Support Stop | $650 | Major support level |
| **NVDA** | $180.00 | Volatility ATR | $171 | High volatility stock |
| **TSLA** | $437.00 | Trailing 5% | $415 | Trending position |
| **AAPL** | $256.00 | Fixed 4% | $246 | Low volatility |
| **META** | $762.00 | Chandelier | $735 | Strong trend |
| **AMD** | $161.00 | Multi-Layer | $156/$152/$147 | Small position |
| **AMZN** | $223.00 | Regime-Based | $213 | Bear market tight |
| **GOOGL** | $252.00 | Time Decay | $244 | Underperforming |

### Composite Strategy Implementation

```python
class OptimalStopLossManager:
    def __init__(self):
        self.strategies = {
            'low_volatility': self.fixed_percentage_stop,
            'high_volatility': self.atr_based_stop,
            'trending': self.trailing_stop,
            'ranging': self.support_level_stop,
            'small_position': self.multi_layer_stop
        }

    def select_strategy(self, symbol, volatility, trend, position_size):
        """Automatically selects best stop strategy"""
        if position_size < 10000:
            return 'small_position'
        elif trend > 30:  # ADX > 30
            return 'trending'
        elif volatility > 0.02:  # 2% daily volatility
            return 'high_volatility'
        elif volatility < 0.01:
            return 'low_volatility'
        else:
            return 'ranging'

    def calculate_stop(self, symbol, entry, current, strategy_type):
        """Returns optimal stop price"""
        return self.strategies[strategy_type](entry, current)
```

## 💡 Pro Tips for Stop Loss Management

1. **Never Move Stops Down** - Only tighten, never loosen
2. **Use Alerts, Not Market Orders** - Avoid stop hunting
3. **Consider Time of Day** - Wider stops during first/last 30 minutes
4. **Account for Dividends** - Adjust stops for ex-dividend drops
5. **Review Weekly** - Adjust based on changing conditions
6. **Mental Stops for Options** - Market orders can be costly
7. **Combine Strategies** - Use multiple approaches together

## 🚀 Quick Implementation

For immediate use with your GOAP strategy:

```python
# Add to your trading config
stop_loss_config = {
    'default_strategy': 'volatility_adjusted',
    'base_stop': 0.06,
    'use_trailing': True,
    'trail_activation': 0.03,  # Activate after 3% profit
    'trail_distance': 0.05,    # 5% trailing
    'regime_adaptive': True,
    'multi_layer': False,       # Enable for positions > $20k
    'time_decay': True,
    'max_hold_days': 30
}
```

This comprehensive stop loss framework will protect your capital while allowing winners to run!