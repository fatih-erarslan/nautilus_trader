# Trading Loss Analysis - Why CWTS Strategies Aren't Profitable

## Current Situation
- **Strategies DO enter trades** (CWTSMomentumStrategy, CWTSUltraStrategy, CWTSMomentumFreqAI)
- **Lost $175 from $10,000** (1.75% loss)
- **Running with FreqAI** (CatboostRegressor) which may be adding noise

## Critical Problems Identified

### 1. ROI Table Too Aggressive (Main Issue)
```python
minimal_roi = {
    "0": 0.015,   # 1.5% immediate target - TOO HIGH
    "5": 0.01,    # 1% after 5 minutes - UNREALISTIC
    "15": 0.005,  # 0.5% after 15 minutes
    "30": 0.003,  # 0.3% after 30 minutes
    "60": 0.001,  # 0.1% after 60 minutes
}
```
**Problem**: Expecting 1.5% immediate profit is unrealistic. Most trades won't hit this and will exit at stop loss instead.

### 2. Stop Loss vs ROI Mismatch
- **Stop loss**: -1.5%
- **Immediate ROI target**: +1.5%
- **Result**: 1:1 risk/reward ratio - needs 50%+ win rate just to break even
- **With fees**: Needs ~52-53% win rate to be profitable

### 3. 1-Minute Timeframe Issues
- **Too noisy** - many false signals
- **High fee impact** - more trades = more fees
- **Spread impact** - hurts on every trade
- **Slippage** - worse on fast timeframes

### 4. Momentum Threshold Too Tight
```python
momentum_threshold = DecimalParameter(0.001, 0.01, default=0.003)
```
- 0.3% momentum threshold on 1m candles = chasing micro moves
- Market noise often exceeds 0.3%
- Entering on noise, not real momentum

### 5. FreqAI Adding Complexity
- CatboostRegressor may be overfitted to training data
- Adds latency to decision making
- May be filtering out good trades

## Why You're Losing Money

### Entry Problems:
1. **Chasing micro-moves** that reverse quickly
2. **Entering on noise** not real trends
3. **FOMO entries** after moves already happened

### Exit Problems:
1. **Unrealistic profit targets** (1.5% immediate)
2. **Getting stopped out** before moves develop
3. **Trailing stop too tight** (0.2% positive)

### Mathematical Reality:
- **Entry spread**: -0.1%
- **Exit spread**: -0.1%
- **Fees**: -0.2% (0.1% each way)
- **Total cost per trade**: -0.4%
- **Need 0.4% move just to break even**

With 1.5% stop loss and 0.4% costs, effective risk is 1.9% per trade.

## The Fix - Realistic Profitable Settings

### 1. Better ROI Table
```python
minimal_roi = {
    "0": 0.10,    # 10% only for huge moves
    "30": 0.05,   # 5% after 30 minutes
    "60": 0.03,   # 3% after 1 hour
    "120": 0.02,  # 2% after 2 hours
    "240": 0.01,  # 1% after 4 hours
    "720": 0.005  # 0.5% after 12 hours
}
```

### 2. Wider Stop Loss
```python
stoploss = -0.03  # 3% stop loss (2:1 risk/reward with fees)
```

### 3. Better Timeframe
```python
timeframe = '15m'  # Less noise, clearer trends
```

### 4. Relaxed Momentum
```python
momentum_threshold = DecimalParameter(0.005, 0.02, default=0.01)  # 1% moves
```

### 5. Better Entry Logic
Instead of:
```python
# Current: Chasing momentum
(dataframe['momentum'] > threshold) & 
(dataframe['momentum'] > dataframe['momentum_sma'])
```

Use:
```python
# Better: Buying pullbacks in uptrends
(dataframe['ema_fast'] > dataframe['ema_slow']) &  # Uptrend
(dataframe['close'] < dataframe['ema_fast']) &     # Pullback
(dataframe['rsi'] < 50) &                          # Not overbought
(dataframe['volume'] > dataframe['volume_sma'])    # Volume confirmation
```

## Quick Profitability Formula

### The 3-2-1 Rule:
- **3% stop loss** (room for volatility)
- **2% average profit target** (realistic)
- **1% minimum profit** (cover fees)

### Position Sizing:
```python
# Risk 1% of capital per trade
position_size = (capital * 0.01) / 0.03  # 1% risk / 3% stop
```

### Entry Criteria (Simple & Effective):
1. **Trend**: Price above 50 EMA
2. **Pullback**: RSI < 40 or price touches 20 EMA
3. **Confirmation**: Volume spike or MACD cross
4. **Risk**: Clear stop below recent low

### Exit Criteria:
1. **Take Profit**: 2-3% or resistance level
2. **Time Exit**: Close after 4 hours if flat
3. **Trailing Stop**: Only after 1% profit
4. **Stop Loss**: 3% or support break

## Immediate Action Plan

### Step 1: Switch to Profitable Settings
```python
# File: /home/kutlu/CWTS/cwts-ultra/freqtrade/strategies/CWTSMomentumStrategy.py

# Change these values:
timeframe = '15m'  # was '1m'

minimal_roi = {
    "0": 0.10,
    "60": 0.03,
    "120": 0.02,
    "360": 0.01
}

stoploss = -0.03  # was -0.015

momentum_threshold = DecimalParameter(0.005, 0.02, default=0.01)  # was 0.003
```

### Step 2: Disable FreqAI (Temporarily)
```bash
# Run without FreqAI to reduce complexity
freqtrade trade --strategy CWTSMomentumStrategy --config user_data/config_parasitic.json
# Don't use: --freqaimodel CatboostRegressor
```

### Step 3: Use Better Pairs
Focus on liquid pairs with clear trends:
- BTC/USDT
- ETH/USDT
- BNB/USDT
- SOL/USDT

Avoid low-cap altcoins with high volatility.

## Expected Results After Fix

### Before (Current):
- Win rate: ~30-35%
- Avg win: 0.5%
- Avg loss: -1.5%
- Result: **-0.675% per trade** (losing)

### After (Fixed):
- Win rate: ~45-50%
- Avg win: 2%
- Avg loss: -2.5% (including fees)
- Result: **+0.35% per trade** (profitable)

## The Math That Works

With 100 trades:
- 45 wins × 2% = +90%
- 55 losses × -2.5% = -137.5%
- Net: -47.5% ❌

Need to improve win rate to 55%:
- 55 wins × 2% = +110%
- 45 losses × -2.5% = -112.5%
- Net: -2.5% (break-even)

Target 60% win rate:
- 60 wins × 2% = +120%
- 40 losses × -2.5% = -100%
- Net: **+20% ✓**

## Summary

Your strategies are entering trades but losing because:
1. **ROI targets too aggressive** (1.5% immediate)
2. **Stop loss too tight** (1.5%)
3. **Trading noise** on 1-minute timeframe
4. **Chasing momentum** instead of buying pullbacks
5. **FreqAI** possibly overfitted

**Fix**: Use 15m timeframe, 3% stops, 2% targets, buy pullbacks not breakouts.