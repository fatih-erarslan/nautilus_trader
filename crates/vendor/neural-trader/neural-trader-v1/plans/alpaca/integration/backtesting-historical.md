# Alpaca API Backtesting & Historical Data Integration

## Overview
Alpaca provides comprehensive historical market data access and seamless integration with popular backtesting frameworks. The platform supports historical data for stocks, crypto, and options, enabling thorough strategy validation before live deployment.

## Historical Data API

### Data Access Capabilities
- **Asset Classes**: Stocks, ETFs, Crypto, Options
- **Data Types**: Trades, Quotes, Bars (1Min, 5Min, 15Min, 1Hour, 1Day)
- **Historical Depth**: Multiple years of historical data
- **Real-time**: Live streaming during market hours
- **Adjustments**: Corporate actions, splits, dividends

### Historical Data Client Implementation
```python
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd

class AlpacaHistoricalDataManager:
    def __init__(self, api_key, secret_key):
        self.stock_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )

        self.crypto_client = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )

    def get_stock_bars(self, symbols, start_date, end_date, timeframe=TimeFrame.Day):
        """Get historical stock bar data"""
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            adjustment='raw'  # 'raw', 'split', or 'dividend'
        )

        bars = self.stock_client.get_stock_bars(request_params)
        return self.bars_to_dataframe(bars)

    def get_stock_trades(self, symbols, start_date, end_date):
        """Get historical trade data"""
        request_params = StockTradesRequest(
            symbol_or_symbols=symbols,
            start=start_date,
            end=end_date
        )

        trades = self.stock_client.get_stock_trades(request_params)
        return self.trades_to_dataframe(trades)

    def get_stock_quotes(self, symbols, start_date, end_date):
        """Get historical quote data"""
        request_params = StockQuotesRequest(
            symbol_or_symbols=symbols,
            start=start_date,
            end=end_date
        )

        quotes = self.stock_client.get_stock_quotes(request_params)
        return self.quotes_to_dataframe(quotes)

    def bars_to_dataframe(self, bars_response):
        """Convert bars response to pandas DataFrame"""
        data = []

        for symbol, bars in bars_response.items():
            for bar in bars:
                data.append({
                    'symbol': symbol,
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'trade_count': bar.trade_count,
                    'vwap': float(bar.vwap) if bar.vwap else None
                })

        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index(['symbol', 'timestamp'], inplace=True)

        return df

    def trades_to_dataframe(self, trades_response):
        """Convert trades response to pandas DataFrame"""
        data = []

        for symbol, trades in trades_response.items():
            for trade in trades:
                data.append({
                    'symbol': symbol,
                    'timestamp': trade.timestamp,
                    'price': float(trade.price),
                    'size': int(trade.size),
                    'exchange': trade.exchange,
                    'conditions': trade.conditions
                })

        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index(['symbol', 'timestamp'], inplace=True)

        return df

    def quotes_to_dataframe(self, quotes_response):
        """Convert quotes response to pandas DataFrame"""
        data = []

        for symbol, quotes in quotes_response.items():
            for quote in quotes:
                data.append({
                    'symbol': symbol,
                    'timestamp': quote.timestamp,
                    'bid_price': float(quote.bid_price),
                    'ask_price': float(quote.ask_price),
                    'bid_size': int(quote.bid_size),
                    'ask_size': int(quote.ask_size),
                    'bid_exchange': quote.bid_exchange,
                    'ask_exchange': quote.ask_exchange
                })

        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index(['symbol', 'timestamp'], inplace=True)

        return df

    def get_crypto_bars(self, symbols, start_date, end_date, timeframe=TimeFrame.Day):
        """Get historical crypto bar data"""
        from alpaca.data.requests import CryptoBarsRequest

        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        bars = self.crypto_client.get_crypto_bars(request_params)
        return self.bars_to_dataframe(bars)

# Example usage
data_manager = AlpacaHistoricalDataManager("api_key", "secret_key")

# Get 1-year daily data for multiple stocks
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

historical_data = data_manager.get_stock_bars(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date=start_date,
    end_date=end_date,
    timeframe=TimeFrame.Day
)
```

## Backtrader Integration

### Alpaca-Backtrader Setup
```python
import backtrader as bt
from alpaca_backtrader_api import AlpacaStore
import pandas as pd

class AlpacaBacktestEnvironment:
    def __init__(self, api_key, secret_key, paper=True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Initialize Alpaca store
        self.store = AlpacaStore(
            key_id=api_key,
            secret_key=secret_key,
            paper=paper,
            usePolygon=False  # Use Alpaca data instead of Polygon
        )

    def create_data_feed(self, symbol, start_date, end_date, timeframe='1D'):
        """Create Alpaca data feed for backtesting"""
        data = self.store.getdata(
            dataname=symbol,
            timeframe=bt.TimeFrame.Days if timeframe == '1D' else bt.TimeFrame.Minutes,
            compression=1,
            fromdate=start_date,
            todate=end_date,
            historical=True  # Important: Use historical data for backtesting
        )
        return data

    def run_backtest(self, strategy_class, symbols, start_date, end_date, initial_cash=100000):
        """Run backtest with Alpaca data"""
        cerebro = bt.Cerebro()

        # Add strategy
        cerebro.addstrategy(strategy_class)

        # Add data feeds
        for symbol in symbols:
            data = self.create_data_feed(symbol, start_date, end_date)
            cerebro.adddata(data, name=symbol)

        # Set initial cash
        cerebro.broker.setcash(initial_cash)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run backtest
        print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
        results = cerebro.run()
        print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

        return results

# Example trading strategy
class MeanReversionStrategy(bt.Strategy):
    params = (
        ('lookback', 20),
        ('buy_threshold', -2),
        ('sell_threshold', 2),
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.lookback)
        self.stddev = bt.indicators.StandardDeviation(self.data.close, period=self.params.lookback)
        self.zscore = (self.data.close - self.sma) / self.stddev

    def next(self):
        if not self.position:
            if self.zscore[0] < self.params.buy_threshold:
                self.buy()
        else:
            if self.zscore[0] > self.params.sell_threshold:
                self.sell()

# Usage example
backtest_env = AlpacaBacktestEnvironment("api_key", "secret_key", paper=True)

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

results = backtest_env.run_backtest(
    strategy_class=MeanReversionStrategy,
    symbols=["AAPL"],
    start_date=start_date,
    end_date=end_date,
    initial_cash=100000
)

# Analyze results
strategy_result = results[0]
print(f"Sharpe Ratio: {strategy_result.analyzers.sharpe.get_analysis()['sharperatio']:.3f}")
print(f"Total Return: {strategy_result.analyzers.returns.get_analysis()['rtot']:.2%}")
print(f"Max Drawdown: {strategy_result.analyzers.drawdown.get_analysis()['max']['drawdown']:.2%}")
```

### Vectorbt Integration
```python
import vectorbt as vbt
import numpy as np

class VectorbtAlpacaBacktest:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def prepare_data(self, symbols, start_date, end_date, timeframe=TimeFrame.Day):
        """Prepare data for vectorbt backtesting"""
        historical_data = self.data_manager.get_stock_bars(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

        # Pivot data for vectorbt
        close_prices = historical_data.reset_index().pivot(
            index='timestamp',
            columns='symbol',
            values='close'
        )

        volume_data = historical_data.reset_index().pivot(
            index='timestamp',
            columns='symbol',
            values='volume'
        )

        return close_prices, volume_data

    def run_sma_crossover_backtest(self, symbols, start_date, end_date, fast_window=10, slow_window=30):
        """Run SMA crossover strategy backtest"""
        close_prices, volume_data = self.prepare_data(symbols, start_date, end_date)

        # Calculate moving averages
        fast_sma = close_prices.rolling(window=fast_window).mean()
        slow_sma = close_prices.rolling(window=slow_window).mean()

        # Generate signals
        entries = fast_sma > slow_sma
        exits = fast_sma < slow_sma

        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            close_prices,
            entries,
            exits,
            init_cash=100000,
            fees=0.001,  # 0.1% fees
            freq='1D'
        )

        return portfolio

    def run_mean_reversion_backtest(self, symbols, start_date, end_date, lookback=20, threshold=2):
        """Run mean reversion strategy backtest"""
        close_prices, volume_data = self.prepare_data(symbols, start_date, end_date)

        # Calculate z-score
        sma = close_prices.rolling(window=lookback).mean()
        std = close_prices.rolling(window=lookback).std()
        zscore = (close_prices - sma) / std

        # Generate signals
        entries = zscore < -threshold  # Buy when oversold
        exits = zscore > threshold     # Sell when overbought

        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            close_prices,
            entries,
            exits,
            init_cash=100000,
            fees=0.001,
            freq='1D'
        )

        return portfolio

    def analyze_results(self, portfolio):
        """Analyze backtest results"""
        stats = portfolio.stats()

        analysis = {
            'total_return': stats['Total Return [%]'],
            'sharpe_ratio': stats['Sharpe Ratio'],
            'max_drawdown': stats['Max Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'profit_factor': stats['Profit Factor'],
            'total_trades': stats['Total Trades']
        }

        return analysis

# Example usage
data_manager = AlpacaHistoricalDataManager("api_key", "secret_key")
backtest = VectorbtAlpacaBacktest(data_manager)

# Run SMA crossover backtest
portfolio = backtest.run_sma_crossover_backtest(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    fast_window=10,
    slow_window=30
)

# Analyze results
results = backtest.analyze_results(portfolio)
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")

# Plot results
portfolio.plot().show()
```

## Custom Backtesting Framework

### Event-Driven Backtesting Engine
```python
from abc import ABC, abstractmethod
from enum import Enum
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

class EventType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"

@dataclass
class Event:
    timestamp: datetime
    event_type: EventType
    data: Dict

class EventDrivenBacktest:
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission

        self.event_queue = []
        self.current_time = None

        # Portfolio tracking
        self.portfolio = Portfolio(initial_capital)
        self.performance_tracker = PerformanceTracker()

    def add_data_source(self, symbol, data):
        """Add market data to event queue"""
        for timestamp, row in data.iterrows():
            event = Event(
                timestamp=timestamp,
                event_type=EventType.MARKET_DATA,
                data={
                    'symbol': symbol,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
            )
            heapq.heappush(self.event_queue, (timestamp, event))

    def run_backtest(self, strategy):
        """Run event-driven backtest"""
        while self.event_queue:
            timestamp, event = heapq.heappop(self.event_queue)
            self.current_time = timestamp

            if event.event_type == EventType.MARKET_DATA:
                # Update portfolio with new prices
                self.portfolio.update_market_data(event.data)

                # Generate trading signals
                signals = strategy.generate_signals(event.data, self.portfolio)

                for signal in signals:
                    order_event = Event(
                        timestamp=timestamp,
                        event_type=EventType.ORDER,
                        data=signal
                    )
                    # Process order immediately (simulated execution)
                    self.process_order(order_event)

            # Track performance
            self.performance_tracker.update(timestamp, self.portfolio.total_value())

    def process_order(self, order_event):
        """Process order and generate fill"""
        order_data = order_event.data
        symbol = order_data['symbol']
        quantity = order_data['quantity']
        order_type = order_data['type']

        # Simulate order execution
        current_price = self.portfolio.current_prices.get(symbol, 0)

        if order_type == 'BUY' and self.portfolio.cash >= quantity * current_price:
            cost = quantity * current_price * (1 + self.commission)
            self.portfolio.cash -= cost
            self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + quantity

        elif order_type == 'SELL' and self.portfolio.positions.get(symbol, 0) >= quantity:
            proceeds = quantity * current_price * (1 - self.commission)
            self.portfolio.cash += proceeds
            self.portfolio.positions[symbol] -= quantity

class Portfolio:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.current_prices = {}

    def update_market_data(self, market_data):
        """Update current market prices"""
        symbol = market_data['symbol']
        self.current_prices[symbol] = market_data['close']

    def total_value(self):
        """Calculate total portfolio value"""
        stock_value = sum(
            quantity * self.current_prices.get(symbol, 0)
            for symbol, quantity in self.positions.items()
        )
        return self.cash + stock_value

    def get_position(self, symbol):
        """Get current position for symbol"""
        return self.positions.get(symbol, 0)

class PerformanceTracker:
    def __init__(self):
        self.equity_curve = []
        self.returns = []

    def update(self, timestamp, portfolio_value):
        """Update performance metrics"""
        self.equity_curve.append((timestamp, portfolio_value))

        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2][1]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.returns.append(daily_return)

    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.returns:
            return {}

        returns_array = np.array(self.returns)

        total_return = (self.equity_curve[-1][1] / self.equity_curve[0][1]) - 1
        volatility = returns_array.std() * np.sqrt(252)
        sharpe_ratio = returns_array.mean() / returns_array.std() * np.sqrt(252)

        # Calculate max drawdown
        equity_values = [value for _, value in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0

        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            'total_return': total_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.returns)
        }

# Strategy interface
class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, market_data, portfolio):
        pass

# Example strategy implementation
class SimpleMomentumStrategy(Strategy):
    def __init__(self, lookback_period=20):
        self.lookback_period = lookback_period
        self.price_history = {}

    def generate_signals(self, market_data, portfolio):
        symbol = market_data['symbol']
        price = market_data['close']

        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(price)

        # Keep only required history
        if len(self.price_history[symbol]) > self.lookback_period:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_period:]

        signals = []

        if len(self.price_history[symbol]) >= self.lookback_period:
            # Calculate momentum
            current_price = self.price_history[symbol][-1]
            avg_price = sum(self.price_history[symbol]) / len(self.price_history[symbol])
            momentum = (current_price - avg_price) / avg_price

            current_position = portfolio.get_position(symbol)

            # Generate signals based on momentum
            if momentum > 0.02 and current_position == 0:  # Buy signal
                signals.append({
                    'symbol': symbol,
                    'type': 'BUY',
                    'quantity': 100
                })
            elif momentum < -0.02 and current_position > 0:  # Sell signal
                signals.append({
                    'symbol': symbol,
                    'type': 'SELL',
                    'quantity': current_position
                })

        return signals

# Usage example
def run_custom_backtest():
    # Get historical data
    data_manager = AlpacaHistoricalDataManager("api_key", "secret_key")

    historical_data = data_manager.get_stock_bars(
        symbols=["AAPL"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        timeframe=TimeFrame.Day
    )

    # Initialize backtest
    backtest = EventDrivenBacktest(initial_capital=100000, commission=0.001)

    # Add data for each symbol
    for symbol in historical_data.index.get_level_values('symbol').unique():
        symbol_data = historical_data.loc[symbol]
        backtest.add_data_source(symbol, symbol_data)

    # Run backtest with strategy
    strategy = SimpleMomentumStrategy(lookback_period=20)
    backtest.run_backtest(strategy)

    # Analyze results
    metrics = backtest.performance_tracker.calculate_metrics()
    print(f"Backtest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Annual Volatility: {metrics['annual_volatility']:.2%}")

    return backtest

# Run the backtest
backtest_results = run_custom_backtest()
```

## Paper Trading Integration

### Seamless Paper-to-Live Transition
```python
class AlpacaBacktestToLive:
    def __init__(self, strategy_class, config):
        self.strategy_class = strategy_class
        self.config = config

        # Start with paper trading
        self.paper_client = TradingClient(
            api_key=config['api_key'],
            secret_key=config['secret_key'],
            paper=True
        )

        self.live_client = None  # Initialize when ready for live trading

    def run_historical_backtest(self, symbols, start_date, end_date):
        """Run historical backtest"""
        data_manager = AlpacaHistoricalDataManager(
            self.config['api_key'],
            self.config['secret_key']
        )

        # Your backtesting logic here
        results = self.run_backtest_with_data(data_manager, symbols, start_date, end_date)
        return results

    def run_paper_trading(self, symbols, duration_days=30):
        """Run strategy in paper trading environment"""
        from alpaca.data.live.stock import StockDataStream

        # Set up live data stream for paper trading
        stream = StockDataStream(
            api_key=self.config['api_key'],
            secret_key=self.config['secret_key']
        )

        # Initialize strategy with paper trading client
        strategy = self.strategy_class(self.paper_client)

        # Set up real-time event handlers
        @stream.on_trade(*symbols)
        async def trade_handler(trade):
            await strategy.process_trade(trade)

        # Run for specified duration
        stream.subscribe_trades(*symbols)
        stream.run()

    def validate_for_live_trading(self):
        """Validate strategy performance before live trading"""
        # Get paper trading results
        paper_account = self.paper_client.get_account()
        paper_positions = self.paper_client.get_all_positions()

        # Calculate paper trading metrics
        paper_metrics = self.calculate_paper_metrics(paper_account, paper_positions)

        # Validation criteria
        validation_passed = (
            paper_metrics['sharpe_ratio'] > 1.0 and
            paper_metrics['max_drawdown'] < 0.10 and
            paper_metrics['win_rate'] > 0.50
        )

        return validation_passed, paper_metrics

    def transition_to_live(self):
        """Transition from paper to live trading"""
        validation_passed, metrics = self.validate_for_live_trading()

        if not validation_passed:
            raise ValueError(f"Strategy validation failed: {metrics}")

        # Initialize live trading client
        self.live_client = TradingClient(
            api_key=self.config['api_key'],
            secret_key=self.config['secret_key'],
            paper=False
        )

        print("🔴 TRANSITIONING TO LIVE TRADING WITH REAL MONEY!")
        return self.live_client

    def calculate_paper_metrics(self, account, positions):
        """Calculate metrics from paper trading"""
        # Implementation for calculating actual performance metrics
        # from paper trading results
        pass
```

## Performance Optimization

### Efficient Data Handling
```python
class OptimizedDataManager:
    def __init__(self, api_key, secret_key, cache_enabled=True):
        self.data_manager = AlpacaHistoricalDataManager(api_key, secret_key)
        self.cache_enabled = cache_enabled
        self.cache = {}

    def get_optimized_data(self, symbols, start_date, end_date, timeframe=TimeFrame.Day):
        """Get data with caching and optimization"""
        cache_key = f"{'-'.join(symbols)}_{start_date}_{end_date}_{timeframe}"

        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]

        # Batch request for multiple symbols
        data = self.data_manager.get_stock_bars(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

        if self.cache_enabled:
            self.cache[cache_key] = data

        return data

    def preload_data(self, symbols, start_date, end_date, timeframes):
        """Preload data for multiple timeframes"""
        for timeframe in timeframes:
            self.get_optimized_data(symbols, start_date, end_date, timeframe)

# Usage for large-scale backtesting
optimized_data = OptimizedDataManager("api_key", "secret_key", cache_enabled=True)

# Preload commonly used data
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
timeframes = [TimeFrame.Day, TimeFrame.Hour, TimeFrame.Minute]

optimized_data.preload_data(
    symbols=symbols,
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframes=timeframes
)
```

This comprehensive guide provides all the tools needed for robust backtesting and historical data analysis using the Alpaca API, enabling thorough strategy validation before live deployment.