# Alpaca Trading API Tutorial

This comprehensive tutorial will guide you through building a complete algorithmic trading system using the Alpaca API. From basic setup to advanced strategies, you'll learn everything needed to create a professional trading application.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic API Usage](#basic-api-usage)
3. [Trading Strategies](#trading-strategies)
4. [Risk Management](#risk-management)
5. [Portfolio Management](#portfolio-management)
6. [Real-time Data Streaming](#real-time-data-streaming)
7. [Backtesting](#backtesting)
8. [Production Deployment](#production-deployment)

## Getting Started

### Prerequisites

- Python 3.8+
- Alpaca brokerage account (paper trading or live)
- Basic understanding of financial markets
- Familiarity with Python and pandas

### Environment Setup

1. **Create Alpaca Account**
   - Sign up at [alpaca.markets](https://alpaca.markets)
   - Generate API keys for paper trading
   - Note your API key and secret

2. **Configure Environment Variables**

   Your `.env` file should contain:
   ```bash
   # Alpaca Trading API
   ALPACA_API_KEY=your-alpaca-api-key
   ALPACA_SECRET_KEY=your-alpaca-secret-key
   ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2  # Paper trading
   # ALPACA_BASE_URL=https://api.alpaca.markets/v2      # Live trading
   ```

3. **Install Dependencies**
   ```bash
   pip install pandas numpy requests websocket-client python-dotenv
   ```

### API Key Security

⚠️ **Important Security Notes:**
- Never commit API keys to version control
- Use environment variables for all credentials
- Start with paper trading (free virtual money)
- Implement proper error handling and logging

## Basic API Usage

### 1. Initialize the Client

```python
from src.alpaca.alpaca_client import AlpacaClient

# Initialize client (uses environment variables)
client = AlpacaClient()

# Test connection
account = client.get_account()
print(f"Account Status: {account['status']}")
print(f"Buying Power: ${account['buying_power']}")
```

### 2. Account Information

```python
# Get account details
account = client.get_account()
print(f"Portfolio Value: ${account['portfolio_value']}")
print(f"Cash: ${account['cash']}")
print(f"Day Trading Buying Power: ${account['daytrading_buying_power']}")

# Check market status
is_open = client.is_market_open()
print(f"Market Open: {is_open}")
```

### 3. Market Data

```python
# Get historical data
bars = client.get_bars('AAPL', timeframe='1Day', limit=30)
print(f"Latest AAPL close: ${bars['close'].iloc[-1]:.2f}")

# Get latest quote
quote = client.get_latest_quote('AAPL')
print(f"AAPL Bid: ${quote['bp']:.2f}, Ask: ${quote['ap']:.2f}")
```

### 4. Order Management

```python
from src.alpaca.alpaca_client import OrderSide, OrderType, TimeInForce

# Place market order
order = client.place_order(
    symbol='AAPL',
    qty=10,
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    time_in_force=TimeInForce.DAY
)

# Place limit order
limit_order = client.place_order(
    symbol='AAPL',
    qty=10,
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    limit_price=150.00,
    time_in_force=TimeInForce.GTC
)

# Check order status
orders = client.get_orders(status='open')
for order in orders:
    print(f"Order {order.id}: {order.side} {order.qty} {order.symbol}")

# Cancel order
client.cancel_order(order.id)
```

### 5. Position Management

```python
# Get all positions
positions = client.get_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.qty} shares, P&L: ${pos.unrealized_pl}")

# Get specific position
aapl_position = client.get_position('AAPL')
if aapl_position:
    print(f"AAPL Position: {aapl_position.qty} shares")
```

## Trading Strategies

### 1. Momentum Strategy

The momentum strategy buys stocks showing strong upward price movement with high volume.

```python
from src.alpaca.trading_strategies import MomentumStrategy, TradingBot

# Initialize strategy
momentum = MomentumStrategy(client, lookback_days=20, volume_threshold=1.5)

# Add to trading bot
bot = TradingBot(client)
bot.add_strategy(momentum)

# Define trading universe
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Run strategies
bot.run_strategies(symbols)
```

**Momentum Strategy Logic:**
- Calculate price momentum over lookback period
- Check for above-average volume
- Buy when momentum > 5% and volume > 1.5x average
- Sell when momentum < -3%

### 2. Mean Reversion Strategy

This strategy uses Bollinger Bands to identify overbought/oversold conditions.

```python
from src.alpaca.trading_strategies import MeanReversionStrategy

# Initialize mean reversion strategy
mean_reversion = MeanReversionStrategy(client, lookback_days=20, std_dev=2.0)

bot.add_strategy(mean_reversion)
```

**Mean Reversion Logic:**
- Calculate Bollinger Bands (20-day MA ± 2 standard deviations)
- Buy when price touches lower band (oversold)
- Sell when price touches upper band (overbought)
- Position size based on how far from mean

### 3. Buy and Hold with Rebalancing

A passive strategy that maintains target portfolio weights.

```python
from src.alpaca.trading_strategies import BuyAndHoldStrategy

# Define target allocation
target_weights = {
    'AAPL': 0.25,   # 25% Apple
    'GOOGL': 0.25,  # 25% Google
    'MSFT': 0.25,   # 25% Microsoft
    'AMZN': 0.25    # 25% Amazon
}

# Initialize buy and hold strategy
buy_hold = BuyAndHoldStrategy(client, target_weights, rebalance_threshold=0.05)
bot.add_strategy(buy_hold)
```

### 4. Custom Strategy Development

Create your own strategy by inheriting from the base class:

```python
from src.alpaca.trading_strategies import TradingStrategy, Signal
import pandas as pd

class RSIStrategy(TradingStrategy):
    def __init__(self, client, rsi_period=14, oversold=30, overbought=70):
        super().__init__(client, "RSI")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data):
        signals = []

        for symbol, df in data.items():
            if len(df) < self.rsi_period:
                continue

            rsi = self.calculate_rsi(df['close'])
            latest_rsi = rsi.iloc[-1]

            if latest_rsi < self.oversold:
                # Oversold - buy signal
                signal = Signal(
                    symbol=symbol,
                    action='buy',
                    strength=(self.oversold - latest_rsi) / self.oversold,
                    price=df['close'].iloc[-1],
                    timestamp=datetime.now(),
                    reason=f"RSI oversold: {latest_rsi:.1f}"
                )
                signals.append(signal)

            elif latest_rsi > self.overbought:
                # Overbought - sell signal
                signal = Signal(
                    symbol=symbol,
                    action='sell',
                    strength=(latest_rsi - self.overbought) / (100 - self.overbought),
                    price=df['close'].iloc[-1],
                    timestamp=datetime.now(),
                    reason=f"RSI overbought: {latest_rsi:.1f}"
                )
                signals.append(signal)

        return signals

    def calculate_position_size(self, signal, account_value):
        # Your position sizing logic here
        max_position_pct = 0.1
        target_value = account_value * max_position_pct * signal.strength
        target_qty = target_value / signal.price

        current_pos = self.client.get_position(signal.symbol)
        current_qty = float(current_pos.qty) if current_pos else 0
        action_qty = target_qty - current_qty

        return PositionSize(
            symbol=signal.symbol,
            target_qty=target_qty,
            current_qty=current_qty,
            action_qty=action_qty,
            action=signal.action
        )
```

## Risk Management

### 1. Position Sizing

```python
class RiskManager:
    def __init__(self, max_position_pct=0.1, max_portfolio_risk=0.02):
        self.max_position_pct = max_position_pct      # Max 10% per position
        self.max_portfolio_risk = max_portfolio_risk   # Max 2% portfolio risk

    def calculate_position_size(self, account_value, entry_price, stop_loss_price):
        """Calculate position size based on risk parameters"""
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)

        # Maximum dollar risk
        max_dollar_risk = account_value * self.max_portfolio_risk

        # Position size based on risk
        risk_based_qty = max_dollar_risk / risk_per_share

        # Position size based on portfolio weight
        max_position_value = account_value * self.max_position_pct
        weight_based_qty = max_position_value / entry_price

        # Use the smaller of the two
        position_size = min(risk_based_qty, weight_based_qty)

        return max(0, position_size)
```

### 2. Stop Losses

```python
def place_stop_loss_order(client, position, stop_loss_pct=0.05):
    """Place stop loss order for existing position"""
    if position.qty > 0:  # Long position
        stop_price = position.avg_entry_price * (1 - stop_loss_pct)
        side = OrderSide.SELL
    else:  # Short position
        stop_price = position.avg_entry_price * (1 + stop_loss_pct)
        side = OrderSide.BUY

    order = client.place_order(
        symbol=position.symbol,
        qty=abs(position.qty),
        side=side,
        order_type=OrderType.STOP,
        stop_price=stop_price,
        time_in_force=TimeInForce.GTC
    )
    return order
```

### 3. Portfolio Risk Monitoring

```python
def calculate_portfolio_risk(positions, correlation_matrix):
    """Calculate portfolio risk using correlation matrix"""
    weights = []
    symbols = []

    total_value = sum(abs(float(pos.market_value)) for pos in positions)

    for pos in positions:
        symbols.append(pos.symbol)
        weights.append(float(pos.market_value) / total_value)

    # Calculate portfolio standard deviation
    portfolio_variance = 0
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            correlation = correlation_matrix.get((symbol1, symbol2), 0)
            portfolio_variance += weights[i] * weights[j] * correlation

    portfolio_risk = portfolio_variance ** 0.5
    return portfolio_risk
```

## Real-time Data Streaming

### WebSocket Implementation

```python
import json
import threading

def start_real_time_trading(client, symbols, strategies):
    """Start real-time trading with WebSocket data"""

    def on_message(data):
        """Handle incoming market data"""
        try:
            for message in data:
                if message.get('T') == 't':  # Trade message
                    symbol = message['S']
                    price = message['p']

                    # Update strategy with new price
                    for strategy in strategies:
                        # Process real-time signal
                        # This would be expanded for full implementation
                        pass

                elif message.get('T') == 'q':  # Quote message
                    symbol = message['S']
                    bid = message['bp']
                    ask = message['ap']

                    # Update order management with new quotes
                    pass

        except Exception as e:
            logger.error(f"Error processing real-time data: {e}")

    # Start WebSocket streaming
    client.start_streaming(symbols, on_message)
```

### Real-time Strategy Execution

```python
class RealTimeBot:
    def __init__(self, client, strategies):
        self.client = client
        self.strategies = strategies
        self.last_prices = {}
        self.running = False

    def process_trade(self, symbol, price, volume):
        """Process incoming trade data"""
        self.last_prices[symbol] = price

        # Check if any strategy needs to act on this price update
        for strategy in self.strategies:
            if hasattr(strategy, 'on_price_update'):
                strategy.on_price_update(symbol, price, volume)

    def start(self, symbols):
        """Start real-time trading"""
        self.running = True

        def on_data(messages):
            for msg in messages:
                if msg.get('T') == 't':  # Trade
                    self.process_trade(msg['S'], msg['p'], msg['s'])

        self.client.start_streaming(symbols, on_data)
```

## Backtesting

### Simple Backtesting Framework

```python
import pandas as pd
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def backtest_strategy(self, strategy, data, start_date, end_date):
        """Run backtest for a strategy"""

        # Filter data by date range
        test_data = {}
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            test_data[symbol] = df.loc[mask]

        # Simulate trading day by day
        all_dates = sorted(set().union(*[df.index for df in test_data.values()]))

        for date in all_dates:
            # Get data up to current date
            historical_data = {}
            for symbol, df in test_data.items():
                historical_data[symbol] = df.loc[df.index <= date]

            # Generate signals
            signals = strategy.generate_signals(historical_data)

            # Execute signals
            for signal in signals:
                self.execute_backtest_trade(signal, date)

            # Record equity
            self.record_equity(date)

        return self.calculate_performance_metrics()

    def execute_backtest_trade(self, signal, date):
        """Execute trade in backtest"""
        symbol = signal.symbol
        price = signal.price

        # Calculate position size (simplified)
        if signal.action == 'buy':
            max_investment = self.capital * 0.1  # Max 10% per position
            qty = max_investment / price
            cost = qty * price

            if cost <= self.capital:
                self.capital -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + qty

                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'buy',
                    'qty': qty,
                    'price': price,
                    'value': cost
                })

        elif signal.action == 'sell' and symbol in self.positions:
            qty = self.positions[symbol]
            if qty > 0:
                proceeds = qty * price
                self.capital += proceeds
                self.positions[symbol] = 0

                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'qty': qty,
                    'price': price,
                    'value': proceeds
                })

    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()

        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        volatility = equity_df['returns'].std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades)
        }

    def record_equity(self, date):
        """Record current equity value"""
        # Calculate current portfolio value (simplified)
        portfolio_value = self.capital
        # Add position values (would need current prices in real implementation)

        self.equity_curve.append({
            'date': date,
            'equity': portfolio_value
        })
```

### Running Backtest

```python
# Example backtest
backtester = Backtester(initial_capital=100000)

# Create strategy
strategy = MomentumStrategy(client, lookback_days=20)

# Get historical data
symbols = ['AAPL', 'GOOGL', 'MSFT']
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

data = {}
for symbol in symbols:
    data[symbol] = client.get_bars(symbol, '1Day', start_date, end_date)

# Run backtest
results = backtester.backtest_strategy(
    strategy,
    data,
    start_date,
    end_date
)

print(f"Backtest Results:")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Annualized Return: {results['annualized_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Production Deployment

### 1. Environment Configuration

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class TradingConfig:
    # API Configuration
    alpaca_api_key: str = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key: str = os.getenv('ALPACA_SECRET_KEY')
    alpaca_base_url: str = os.getenv('ALPACA_BASE_URL')

    # Trading Parameters
    max_position_pct: float = 0.1
    max_portfolio_risk: float = 0.02
    update_interval: int = 300  # 5 minutes

    # Risk Management
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_daily_loss_pct: float = 0.03

    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_file: str = 'trading.log'

    def validate(self):
        """Validate configuration"""
        required_fields = [
            'alpaca_api_key',
            'alpaca_secret_key',
            'alpaca_base_url'
        ]

        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(f"Missing required configuration: {field}")
```

### 2. Logging and Monitoring

```python
import logging
import sys
from datetime import datetime

def setup_logging(config):
    """Setup logging configuration"""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(config.log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, config.log_level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

class PerformanceMonitor:
    def __init__(self):
        self.daily_pnl = 0
        self.start_time = datetime.now()
        self.trade_count = 0
        self.error_count = 0

    def record_trade(self, pnl):
        """Record trade performance"""
        self.daily_pnl += pnl
        self.trade_count += 1

    def record_error(self):
        """Record error occurrence"""
        self.error_count += 1

    def get_statistics(self):
        """Get performance statistics"""
        runtime = datetime.now() - self.start_time
        return {
            'daily_pnl': self.daily_pnl,
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'runtime_hours': runtime.total_seconds() / 3600,
            'trades_per_hour': self.trade_count / max(runtime.total_seconds() / 3600, 1)
        }
```

### 3. Production Trading Application

```python
# main.py
import signal
import sys
import time
from src.alpaca.alpaca_client import AlpacaClient
from src.alpaca.trading_strategies import MomentumStrategy, TradingBot

class ProductionTradingApp:
    def __init__(self, config):
        self.config = config
        self.client = AlpacaClient()
        self.bot = TradingBot(self.client)
        self.monitor = PerformanceMonitor()
        self.running = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()

    def setup_strategies(self):
        """Setup trading strategies"""
        # Add momentum strategy
        momentum = MomentumStrategy(self.client, lookback_days=20)
        self.bot.add_strategy(momentum)

        logger.info("Strategies configured")

    def check_risk_limits(self):
        """Check if risk limits are exceeded"""
        account = self.client.get_account()
        portfolio_value = float(account['portfolio_value'])

        # Check daily loss limit
        daily_pnl_pct = self.monitor.daily_pnl / portfolio_value
        if daily_pnl_pct < -self.config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit exceeded: {daily_pnl_pct:.2%}")
            return False

        return True

    def run(self):
        """Main trading loop"""
        logger.info("Starting production trading application")

        self.setup_strategies()
        self.running = True

        # Trading universe
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

        while self.running:
            try:
                # Check if market is open
                if not self.client.is_market_open():
                    logger.info("Market closed, waiting...")
                    time.sleep(300)  # Check every 5 minutes
                    continue

                # Check risk limits
                if not self.check_risk_limits():
                    logger.error("Risk limits exceeded, stopping trading")
                    break

                # Run strategies
                logger.info("Running trading strategies")
                self.bot.run_strategies(symbols)

                # Log performance
                stats = self.monitor.get_statistics()
                logger.info(f"Performance: P&L: ${stats['daily_pnl']:.2f}, "
                           f"Trades: {stats['trade_count']}, "
                           f"Errors: {stats['error_count']}")

                # Wait for next update
                time.sleep(self.config.update_interval)

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.monitor.record_error()
                time.sleep(60)  # Wait before retrying

        self.shutdown()

    def shutdown(self):
        """Shutdown trading application"""
        logger.info("Shutting down trading application")

        # Cancel all open orders
        try:
            self.client.cancel_all_orders()
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

        # Stop WebSocket streaming
        self.client.stop_streaming()

        # Log final statistics
        stats = self.monitor.get_statistics()
        logger.info(f"Final Statistics: {stats}")

        self.running = False
        sys.exit(0)

if __name__ == "__main__":
    # Load configuration
    config = TradingConfig()
    config.validate()

    # Setup logging
    logger = setup_logging(config)

    # Create and run application
    app = ProductionTradingApp(config)
    app.run()
```

### 4. Deployment Script

```bash
#!/bin/bash
# deploy.sh

echo "Deploying Alpaca Trading Bot..."

# Create virtual environment
python3 -m venv trading_env
source trading_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env with your API keys"
    exit 1
fi

# Create systemd service
sudo tee /etc/systemd/system/alpaca-trading.service > /dev/null <<EOF
[Unit]
Description=Alpaca Trading Bot
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/alpaca-trading
Environment=PATH=/opt/alpaca-trading/trading_env/bin
ExecStart=/opt/alpaca-trading/trading_env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable alpaca-trading
sudo systemctl start alpaca-trading

echo "Deployment complete!"
echo "Check status: sudo systemctl status alpaca-trading"
echo "View logs: sudo journalctl -u alpaca-trading -f"
```

## Best Practices

### 1. Testing Strategy

- **Paper Trading First**: Always test with paper trading before live deployment
- **Backtesting**: Validate strategies on historical data
- **Unit Testing**: Write tests for all strategy components
- **Integration Testing**: Test full workflow end-to-end

### 2. Risk Management

- **Position Sizing**: Never risk more than 1-2% per trade
- **Stop Losses**: Always use stop losses to limit downside
- **Diversification**: Don't put all capital in one strategy or asset
- **Daily Limits**: Set maximum daily loss limits

### 3. Monitoring

- **Real-time Alerts**: Setup alerts for system errors and large losses
- **Performance Tracking**: Monitor strategy performance continuously
- **Error Logging**: Log all errors and API failures
- **Backup Systems**: Have fallback procedures for system failures

### 4. Security

- **API Key Management**: Use environment variables, never hardcode
- **Access Control**: Limit API permissions to minimum required
- **Audit Trail**: Log all trading decisions and executions
- **Regular Updates**: Keep dependencies and systems updated

## Conclusion

This tutorial provides a comprehensive foundation for building algorithmic trading systems with Alpaca. Start with paper trading, implement proper risk management, and gradually scale to more sophisticated strategies.

### Next Steps

1. **Implement Additional Strategies**: RSI, MACD, pairs trading
2. **Add Machine Learning**: Use ML models for signal generation
3. **Optimize Performance**: Implement caching and parallel processing
4. **Scale Infrastructure**: Deploy on cloud platforms for reliability
5. **Regulatory Compliance**: Ensure compliance with trading regulations

### Resources

- [Alpaca Documentation](https://alpaca.markets/docs/)
- [Alpaca Python SDK](https://github.com/alpacahq/alpaca-trade-api-python)
- [Financial Data APIs](https://polygon.io/, https://finnhub.io/)
- [Backtesting Libraries](https://github.com/pmorissette/backtrader)

Remember: **Past performance does not guarantee future results. Always trade responsibly and within your risk tolerance.**