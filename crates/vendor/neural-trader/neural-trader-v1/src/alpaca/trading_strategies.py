"""
Trading Strategies for Alpaca
Implements various algorithmic trading strategies
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

from .alpaca_client import AlpacaClient, OrderSide, OrderType, TimeInForce

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1, confidence in signal
    price: float
    timestamp: datetime
    reason: str

@dataclass
class PositionSize:
    """Position sizing information"""
    symbol: str
    target_qty: float
    current_qty: float
    action_qty: float  # quantity to buy/sell
    action: str  # 'buy', 'sell', 'hold'

class TradingStrategy(ABC):
    """Base class for trading strategies"""

    def __init__(self, client: AlpacaClient, name: str):
        self.client = client
        self.name = name
        self.positions = {}
        self.last_update = None

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trading signals based on market data"""
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Signal, account_value: float) -> PositionSize:
        """Calculate position size for a signal"""
        pass

    def execute_signal(self, signal: Signal) -> bool:
        """Execute a trading signal"""
        try:
            # Get current position
            current_pos = self.client.get_position(signal.symbol)
            current_qty = float(current_pos.qty) if current_pos else 0

            # Get account info
            account = self.client.get_account()
            account_value = float(account['portfolio_value'])

            # Calculate position size
            pos_size = self.calculate_position_size(signal, account_value)

            if abs(pos_size.action_qty) < 0.01:  # Skip very small trades
                return True

            # Execute trade
            side = OrderSide.BUY if pos_size.action_qty > 0 else OrderSide.SELL
            qty = abs(pos_size.action_qty)

            order = self.client.place_order(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )

            logger.info(f"Executed {self.name} signal: {side.value} {qty} {signal.symbol} at ${signal.price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to execute signal for {signal.symbol}: {e}")
            return False

class MomentumStrategy(TradingStrategy):
    """Momentum trading strategy based on price and volume"""

    def __init__(self, client: AlpacaClient, lookback_days: int = 20, volume_threshold: float = 1.5):
        super().__init__(client, "Momentum")
        self.lookback_days = lookback_days
        self.volume_threshold = volume_threshold

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate momentum signals"""
        signals = []

        for symbol, df in data.items():
            if len(df) < self.lookback_days:
                continue

            # Calculate momentum indicators
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(self.lookback_days).mean()
            df['price_ma'] = df['close'].rolling(self.lookback_days).mean()
            df['momentum'] = df['close'] / df['close'].shift(self.lookback_days) - 1

            latest = df.iloc[-1]

            # Signal conditions
            strong_momentum = latest['momentum'] > 0.05  # 5% gain over lookback period
            high_volume = latest['volume'] > latest['volume_ma'] * self.volume_threshold
            above_ma = latest['close'] > latest['price_ma']

            # Generate signal
            if strong_momentum and high_volume and above_ma:
                signal = Signal(
                    symbol=symbol,
                    action='buy',
                    strength=min(latest['momentum'] * 10, 1.0),  # Scale momentum to 0-1
                    price=latest['close'],
                    timestamp=datetime.now(),
                    reason=f"Strong momentum: {latest['momentum']:.2%}, High volume"
                )
                signals.append(signal)

            elif latest['momentum'] < -0.03:  # 3% loss
                signal = Signal(
                    symbol=symbol,
                    action='sell',
                    strength=min(abs(latest['momentum']) * 10, 1.0),
                    price=latest['close'],
                    timestamp=datetime.now(),
                    reason=f"Negative momentum: {latest['momentum']:.2%}"
                )
                signals.append(signal)

        return signals

    def calculate_position_size(self, signal: Signal, account_value: float) -> PositionSize:
        """Calculate position size based on signal strength and risk management"""
        max_position_pct = 0.1  # Max 10% of portfolio per position
        risk_pct = 0.02  # Risk 2% per trade

        # Get current position
        current_pos = self.client.get_position(signal.symbol)
        current_qty = float(current_pos.qty) if current_pos else 0

        # Calculate target position size
        if signal.action == 'buy':
            target_value = account_value * max_position_pct * signal.strength
            target_qty = target_value / signal.price
        elif signal.action == 'sell':
            target_qty = 0  # Close position
        else:
            target_qty = current_qty  # Hold

        action_qty = target_qty - current_qty

        return PositionSize(
            symbol=signal.symbol,
            target_qty=target_qty,
            current_qty=current_qty,
            action_qty=action_qty,
            action=signal.action
        )

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy using Bollinger Bands"""

    def __init__(self, client: AlpacaClient, lookback_days: int = 20, std_dev: float = 2.0):
        super().__init__(client, "MeanReversion")
        self.lookback_days = lookback_days
        self.std_dev = std_dev

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate mean reversion signals"""
        signals = []

        for symbol, df in data.items():
            if len(df) < self.lookback_days:
                continue

            # Calculate Bollinger Bands
            df = df.copy()
            df['ma'] = df['close'].rolling(self.lookback_days).mean()
            df['std'] = df['close'].rolling(self.lookback_days).std()
            df['upper_band'] = df['ma'] + (df['std'] * self.std_dev)
            df['lower_band'] = df['ma'] - (df['std'] * self.std_dev)
            df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

            latest = df.iloc[-1]

            # Signal conditions
            oversold = latest['bb_position'] < 0.1  # Below lower band
            overbought = latest['bb_position'] > 0.9  # Above upper band

            # Generate signals
            if oversold:
                signal = Signal(
                    symbol=symbol,
                    action='buy',
                    strength=1.0 - latest['bb_position'],  # Stronger signal when more oversold
                    price=latest['close'],
                    timestamp=datetime.now(),
                    reason=f"Oversold: BB position {latest['bb_position']:.2f}"
                )
                signals.append(signal)

            elif overbought:
                signal = Signal(
                    symbol=symbol,
                    action='sell',
                    strength=latest['bb_position'],  # Stronger signal when more overbought
                    price=latest['close'],
                    timestamp=datetime.now(),
                    reason=f"Overbought: BB position {latest['bb_position']:.2f}"
                )
                signals.append(signal)

        return signals

    def calculate_position_size(self, signal: Signal, account_value: float) -> PositionSize:
        """Calculate position size for mean reversion strategy"""
        max_position_pct = 0.05  # Max 5% per position for mean reversion

        # Get current position
        current_pos = self.client.get_position(signal.symbol)
        current_qty = float(current_pos.qty) if current_pos else 0

        # Calculate target position
        if signal.action == 'buy':
            target_value = account_value * max_position_pct * signal.strength
            target_qty = target_value / signal.price
        elif signal.action == 'sell':
            # For mean reversion, we might want to reduce position gradually
            target_qty = current_qty * 0.5  # Reduce by half
        else:
            target_qty = current_qty

        action_qty = target_qty - current_qty

        return PositionSize(
            symbol=signal.symbol,
            target_qty=target_qty,
            current_qty=current_qty,
            action_qty=action_qty,
            action=signal.action
        )

class BuyAndHoldStrategy(TradingStrategy):
    """Simple buy and hold strategy with rebalancing"""

    def __init__(self, client: AlpacaClient, target_weights: Dict[str, float], rebalance_threshold: float = 0.05):
        super().__init__(client, "BuyAndHold")
        self.target_weights = target_weights  # {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}
        self.rebalance_threshold = rebalance_threshold

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate rebalancing signals"""
        signals = []

        # Get current portfolio
        account = self.client.get_account()
        portfolio_value = float(account['portfolio_value'])

        positions = self.client.get_positions()
        current_weights = {}

        for pos in positions:
            if pos.symbol in self.target_weights:
                current_weights[pos.symbol] = float(pos.market_value) / portfolio_value

        # Check each target symbol
        for symbol, target_weight in self.target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight

            # Only rebalance if difference is significant
            if abs(weight_diff) > self.rebalance_threshold:
                if symbol in data and len(data[symbol]) > 0:
                    price = data[symbol]['close'].iloc[-1]
                    action = 'buy' if weight_diff > 0 else 'sell'

                    signal = Signal(
                        symbol=symbol,
                        action=action,
                        strength=abs(weight_diff) / target_weight,
                        price=price,
                        timestamp=datetime.now(),
                        reason=f"Rebalance: current {current_weight:.1%} vs target {target_weight:.1%}"
                    )
                    signals.append(signal)

        return signals

    def calculate_position_size(self, signal: Signal, account_value: float) -> PositionSize:
        """Calculate position size to achieve target weight"""
        target_weight = self.target_weights[signal.symbol]
        target_value = account_value * target_weight
        target_qty = target_value / signal.price

        # Get current position
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

class TradingBot:
    """Main trading bot that manages multiple strategies"""

    def __init__(self, client: AlpacaClient):
        self.client = client
        self.strategies = []
        self.running = False
        self.update_interval = 300  # 5 minutes

    def add_strategy(self, strategy: TradingStrategy) -> None:
        """Add a trading strategy"""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a trading strategy"""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        logger.info(f"Removed strategy: {strategy_name}")

    def get_market_data(self, symbols: List[str], timeframe: str = '1Day', days: int = 50) -> Dict[str, pd.DataFrame]:
        """Fetch market data for all symbols"""
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for symbol in symbols:
            try:
                df = self.client.get_bars(symbol, timeframe, start_date, end_date)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        return data

    def run_strategies(self, symbols: List[str]) -> None:
        """Run all strategies once"""
        if not self.client.is_market_open():
            logger.info("Market is closed, skipping strategy execution")
            return

        # Get market data
        data = self.get_market_data(symbols)

        if not data:
            logger.warning("No market data available")
            return

        # Run each strategy
        for strategy in self.strategies:
            try:
                logger.info(f"Running strategy: {strategy.name}")

                # Generate signals
                signals = strategy.generate_signals(data)

                # Execute signals
                for signal in signals:
                    logger.info(f"Signal: {signal.action} {signal.symbol} (strength: {signal.strength:.2f})")
                    success = strategy.execute_signal(signal)

                    if success:
                        logger.info(f"Successfully executed signal for {signal.symbol}")
                    else:
                        logger.error(f"Failed to execute signal for {signal.symbol}")

                    # Small delay between orders
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error running strategy {strategy.name}: {e}")

    def start(self, symbols: List[str]) -> None:
        """Start the trading bot"""
        self.running = True
        logger.info(f"Starting trading bot with {len(self.strategies)} strategies")

        while self.running:
            try:
                self.run_strategies(symbols)
                logger.info(f"Sleeping for {self.update_interval} seconds...")
                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping bot...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait before retrying

        self.stop()

    def stop(self) -> None:
        """Stop the trading bot"""
        self.running = False
        logger.info("Trading bot stopped")

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        account = self.client.get_account()
        positions = self.client.get_positions()

        total_value = float(account['portfolio_value'])
        cash = float(account['cash'])

        position_summary = []
        for pos in positions:
            position_summary.append({
                'symbol': pos.symbol,
                'qty': pos.qty,
                'market_value': pos.market_value,
                'unrealized_pl': pos.unrealized_pl,
                'weight': float(pos.market_value) / total_value if total_value > 0 else 0
            })

        return {
            'total_value': total_value,
            'cash': cash,
            'positions': position_summary,
            'num_positions': len(positions)
        }

# Example usage
if __name__ == "__main__":
    # Initialize client and bot
    client = AlpacaClient()
    bot = TradingBot(client)

    # Add strategies
    momentum = MomentumStrategy(client, lookback_days=20)
    mean_reversion = MeanReversionStrategy(client, lookback_days=20)
    buy_hold = BuyAndHoldStrategy(client, {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4})

    bot.add_strategy(momentum)
    bot.add_strategy(mean_reversion)
    bot.add_strategy(buy_hold)

    # Define universe
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

    # Run once
    bot.run_strategies(symbols)

    # Print portfolio summary
    summary = bot.get_portfolio_summary()
    print(f"Portfolio Value: ${summary['total_value']:,.2f}")
    print(f"Cash: ${summary['cash']:,.2f}")
    print(f"Positions: {summary['num_positions']}")