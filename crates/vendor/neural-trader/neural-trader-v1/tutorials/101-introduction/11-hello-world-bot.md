# Part 11: Hello World Trading Bot
**Duration**: 15 minutes | **Difficulty**: Beginner-Intermediate

## 🤖 Your First Trading Bot

Let's build a complete trading bot from scratch that monitors markets, generates signals, and executes trades (in paper mode).

## 📝 Bot Requirements

Our "Hello World" bot will:
1. Monitor a single stock (AAPL)
2. Use simple moving average crossover
3. Generate buy/sell signals
4. Execute paper trades
5. Track performance

## 🚀 Step 1: Initialize the Bot (2 min)

```bash
# Create project structure
claude "Create a new trading bot project:
- Name: hello-trader
- Strategy: SMA crossover
- Symbol: AAPL
- Mode: paper trading"
```

This creates:
```
hello-trader/
├── config.json
├── strategy.py
├── main.py
└── requirements.txt
```

## 📊 Step 2: Configure Strategy (3 min)

```python
# config.json
{
    "symbol": "AAPL",
    "strategy": "sma_crossover",
    "parameters": {
        "fast_period": 10,
        "slow_period": 30,
        "position_size": 100,  # shares
        "stop_loss": 0.02,     # 2%
        "take_profit": 0.05    # 5%
    },
    "mode": "paper",
    "capital": 10000
}
```

Deploy configuration:
```bash
claude "Configure hello-trader bot with:
- Fast SMA: 10 days
- Slow SMA: 30 days
- Position: 100 shares
- Stop loss: 2%"
```

## 💻 Step 3: Implement Core Logic (5 min)

### Strategy Implementation
```python
# strategy.py
import pandas as pd
import numpy as np

class SMAStrategy:
    def __init__(self, fast_period=10, slow_period=30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position = 0
        
    def calculate_signals(self, data):
        """Generate trading signals from price data"""
        # Calculate moving averages
        data['SMA_fast'] = data['close'].rolling(self.fast_period).mean()
        data['SMA_slow'] = data['close'].rolling(self.slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['SMA_fast'] > data['SMA_slow'], 'signal'] = 1
        data.loc[data['SMA_fast'] < data['SMA_slow'], 'signal'] = -1
        
        return data
```

### Main Bot Loop
```python
# main.py
import time
from datetime import datetime

class TradingBot:
    def __init__(self, config):
        self.config = config
        self.strategy = SMAStrategy(
            config['parameters']['fast_period'],
            config['parameters']['slow_period']
        )
        self.running = False
        
    def run(self):
        """Main bot loop"""
        self.running = True
        print(f"🤖 Trading bot started at {datetime.now()}")
        
        while self.running:
            try:
                # Fetch latest data
                data = self.fetch_data()
                
                # Generate signals
                signals = self.strategy.calculate_signals(data)
                
                # Execute trades
                self.execute_trades(signals)
                
                # Wait for next iteration
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                print("Bot stopped by user")
                self.running = False
```

Deploy the bot:
```bash
claude "Deploy hello-trader bot with SMA crossover strategy"
```

## 📈 Step 4: Add Data Feed (2 min)

```bash
# Connect to market data
claude "Connect hello-trader to real-time data:
- Source: Yahoo Finance
- Interval: 1 minute
- History: 30 days for SMA calculation"
```

Data fetching function:
```python
def fetch_data(self):
    """Fetch latest market data"""
    import yfinance as yf
    
    ticker = yf.Ticker(self.config['symbol'])
    data = ticker.history(period='30d', interval='1d')
    
    # Format for strategy
    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    return data
```

## 🔄 Step 5: Trade Execution (2 min)

```python
def execute_trades(self, signals):
    """Execute trades based on signals"""
    latest_signal = signals['signal'].iloc[-1]
    current_price = signals['close'].iloc[-1]
    
    # Buy signal
    if latest_signal == 1 and self.position == 0:
        self.buy(current_price)
    
    # Sell signal
    elif latest_signal == -1 and self.position > 0:
        self.sell(current_price)
    
    # Check stop loss / take profit
    self.check_risk_limits(current_price)

def buy(self, price):
    """Execute buy order"""
    shares = self.config['parameters']['position_size']
    print(f"📈 BUY {shares} shares at ${price:.2f}")
    self.position = shares
    self.entry_price = price

def sell(self, price):
    """Execute sell order"""
    profit = (price - self.entry_price) * self.position
    print(f"📉 SELL {self.position} shares at ${price:.2f}")
    print(f"💰 Profit: ${profit:.2f}")
    self.position = 0
```

## 🛡 Step 6: Risk Management (1 min)

```python
def check_risk_limits(self, current_price):
    """Check stop loss and take profit"""
    if self.position > 0:
        # Calculate P&L
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        # Stop loss
        if pnl_pct <= -self.config['parameters']['stop_loss']:
            print(f"🛑 Stop loss triggered at ${current_price:.2f}")
            self.sell(current_price)
        
        # Take profit
        elif pnl_pct >= self.config['parameters']['take_profit']:
            print(f"🎯 Take profit triggered at ${current_price:.2f}")
            self.sell(current_price)
```

## 🚦 Step 7: Run the Bot (1 min)

### Start Trading
```bash
# Launch the bot
claude "Start hello-trader bot in paper mode"
```

Expected output:
```
🤖 Trading bot started at 2024-12-19 09:30:00
📊 Fetching market data for AAPL...
📈 SMA(10): 234.56, SMA(30): 232.10
✅ Buy signal detected!
📈 BUY 100 shares at $235.20
⏰ Next check in 60 seconds...
```

### Monitor Performance
```bash
# Check bot status
claude "Show hello-trader performance:
- Current position
- P&L
- Trade history
- Signal strength"
```

## 📊 Step 8: Performance Tracking (1 min)

```python
class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.balance = 10000
        
    def record_trade(self, trade):
        """Record trade details"""
        self.trades.append({
            'timestamp': datetime.now(),
            'action': trade['action'],
            'price': trade['price'],
            'shares': trade['shares'],
            'pnl': trade.get('pnl', 0)
        })
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades)
        
        return {
            'total_trades': len(self.trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'current_balance': self.balance + total_pnl
        }
```

## 🔧 Step 9: Enhancements (2 min)

### Add Neural Predictions
```bash
claude "Enhance hello-trader with neural predictions:
- Add LSTM price predictor
- Combine with SMA signals
- Weight: 60% SMA, 40% neural"
```

### Multi-Symbol Support
```bash
claude "Expand hello-trader to trade multiple symbols:
- Add MSFT, GOOGL
- Allocate capital equally
- Independent signals per symbol"
```

### Advanced Risk Management
```bash
claude "Add advanced risk features:
- Trailing stop loss
- Position sizing by volatility
- Maximum daily loss limit
- Correlation checks"
```

## 🧪 Step 10: Testing & Optimization (2 min)

### Backtest Strategy
```bash
claude "Backtest hello-trader on 2 years of data:
- Show total return
- Maximum drawdown
- Sharpe ratio
- Win/loss ratio"
```

### Optimize Parameters
```bash
claude "Optimize SMA periods:
- Fast: test 5-20
- Slow: test 20-50
- Find best combination
- Use walk-forward analysis"
```

## 📈 Complete Bot Code

Here's the full working bot:

```bash
# Create and run complete bot
claude "Create complete hello-trader bot with:
1. SMA crossover strategy
2. Real-time data feed
3. Paper trading execution
4. Risk management
5. Performance tracking
Then start it on AAPL"
```

## 🎯 Exercises

### Exercise 1: Modify Strategy
```bash
claude "Change hello-trader to use RSI instead of SMA:
- Buy when RSI < 30
- Sell when RSI > 70
- Test on last month"
```

### Exercise 2: Add Features
```bash
claude "Add these features to hello-trader:
- Email alerts on trades
- Daily performance report
- Automatic parameter tuning"
```

### Exercise 3: Production Ready
```bash
claude "Make hello-trader production ready:
- Add error handling
- Implement logging
- Create Docker container
- Set up monitoring"
```

## ✅ Checklist

- [ ] Bot structure created
- [ ] Strategy configured
- [ ] Data feed connected
- [ ] Trade execution working
- [ ] Risk management active
- [ ] Performance tracking enabled
- [ ] Paper trading tested
- [ ] Ready for live trading

## 🚀 Next Steps

Your bot is running! To go further:
1. Add more indicators
2. Implement machine learning
3. Connect to broker API
4. Deploy to cloud
5. Scale to multiple strategies

## ⏭ What's Next?

Learn about supported APIs in [Supported APIs](12-supported-apis.md)

---

**Progress**: 105 min / 2 hours | [← Previous: Neural Networks](10-neural-network-training.md) | [Back to Contents](README.md) | [Next: APIs →](12-supported-apis.md)