#!/usr/bin/env python3
"""
Perpetual Neural Trading System
Automated trading with continuous execution and risk management
"""

import asyncio
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.append('/workspaces/neural-trader/src')

# Import Alpaca client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import CryptoDataStream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingStrategy(Enum):
    """Available trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    NEURAL_PREDICTION = "neural_prediction"

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    # Account settings
    api_key: str = "PKAJQDPYIZ1S8BHWU7GD"
    secret_key: str = "zJvREGAi3qQi6zdjhMuemKeUlWhDid78mPIGLkTw"
    paper_trading: bool = True

    # Strategy settings
    strategy: TradingStrategy = TradingStrategy.MOMENTUM
    symbol: str = "BTC/USD"

    # Position sizing
    max_position_size: float = 10000  # USD
    position_size_pct: float = 0.02  # 2% of account per trade
    max_positions: int = 3

    # Risk management
    stop_loss_pct: float = 0.001  # 0.1%
    take_profit_pct: float = 0.005  # 0.5%
    max_drawdown_pct: float = 0.001  # 0.1% max drawdown

    # Trading parameters
    min_win_rate: float = 0.60  # 60% minimum win rate
    risk_reward_ratio: float = 5.0  # 1:5 risk/reward

    # Timing
    trading_hours_only: bool = False
    min_time_between_trades: int = 60  # seconds

    # Technical indicators
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    volume_threshold: float = 1.5  # 1.5x average volume

    # Neural model settings
    use_neural_predictions: bool = True
    min_confidence: float = 0.70

class PerpetualNeuralTrader:
    """Perpetual automated trading system"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.trading_client = None
        self.crypto_client = None
        self.stream = None

        # Trading state
        self.is_running = False
        self.current_positions = {}
        self.trade_history = []
        self.last_trade_time = None

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.current_drawdown = 0

        # Initialize clients
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize Alpaca API clients"""
        try:
            # Trading client for orders
            self.trading_client = TradingClient(
                self.config.api_key,
                self.config.secret_key,
                paper=self.config.paper_trading
            )

            # Crypto data client
            self.crypto_client = CryptoHistoricalDataClient()

            # WebSocket stream for real-time data
            self.stream = CryptoDataStream(
                self.config.api_key,
                self.config.secret_key
            )

            logger.info("✅ Initialized Alpaca clients successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            raise

    async def get_market_data(self) -> Dict:
        """Get current market data and indicators"""
        try:
            # Get latest quote
            quote_request = CryptoLatestQuoteRequest(
                symbol_or_symbols=self.config.symbol
            )
            quote = self.crypto_client.get_crypto_latest_quote(quote_request)

            # Get recent bars for indicators
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)

            bars_request = CryptoBarsRequest(
                symbol_or_symbols=self.config.symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time
            )
            bars = self.crypto_client.get_crypto_bars(bars_request)
            df = bars.df

            if df.empty:
                return None

            # Calculate indicators
            df = self._calculate_indicators(df)
            latest = df.iloc[-1]

            # Get quote data
            symbol_quote = quote[self.config.symbol]

            return {
                'timestamp': datetime.now(),
                'bid': symbol_quote.bid_price,
                'ask': symbol_quote.ask_price,
                'mid': (symbol_quote.bid_price + symbol_quote.ask_price) / 2,
                'spread': symbol_quote.ask_price - symbol_quote.bid_price,
                'close': latest['close'],
                'volume': latest['volume'],
                'rsi': latest.get('rsi', 50),
                'macd': latest.get('macd', 0),
                'signal': latest.get('signal', 0),
                'bb_upper': latest.get('bb_upper', 0),
                'bb_lower': latest.get('bb_lower', 0),
                'sma20': latest.get('sma20', latest['close']),
                'volume_ratio': latest.get('volume_ratio', 1.0)
            }

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['sma20'] - (df['bb_std'] * 2)

        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        return df

    def should_enter_trade(self, data: Dict) -> Tuple[bool, str, float]:
        """Determine if we should enter a trade"""

        # Check timing constraints
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).seconds
            if time_since_last < self.config.min_time_between_trades:
                return False, "none", 0

        # Check position limits
        if len(self.current_positions) >= self.config.max_positions:
            return False, "none", 0

        signal = "none"
        confidence = 0

        # Strategy-specific signals
        if self.config.strategy == TradingStrategy.MOMENTUM:
            signal, confidence = self._momentum_signal(data)
        elif self.config.strategy == TradingStrategy.MEAN_REVERSION:
            signal, confidence = self._mean_reversion_signal(data)
        elif self.config.strategy == TradingStrategy.BREAKOUT:
            signal, confidence = self._breakout_signal(data)
        elif self.config.strategy == TradingStrategy.SCALPING:
            signal, confidence = self._scalping_signal(data)

        # Apply filters
        if signal != "none":
            # Volume filter
            if data['volume_ratio'] < self.config.volume_threshold:
                return False, "none", 0

            # Win rate filter (check historical performance)
            if self.total_trades > 10:
                current_win_rate = self.winning_trades / self.total_trades
                if current_win_rate < self.config.min_win_rate:
                    logger.warning(f"Win rate {current_win_rate:.1%} below minimum")
                    return False, "none", 0

        return signal != "none", signal, confidence

    def _momentum_signal(self, data: Dict) -> Tuple[str, float]:
        """Momentum trading signal"""
        rsi = data['rsi']
        macd = data['macd']
        signal = data['signal']
        price = data['mid']
        sma = data['sma20']

        # Bullish momentum
        if (rsi > 50 and rsi < 65 and
            macd > signal and
            price > sma and
            data['volume_ratio'] > 1.2):
            confidence = min(0.8, (rsi - 50) / 15 * 0.3 + 0.5)
            return "long", confidence

        # Bearish momentum
        elif (rsi < 50 and rsi > 35 and
              macd < signal and
              price < sma and
              data['volume_ratio'] > 1.2):
            confidence = min(0.8, (50 - rsi) / 15 * 0.3 + 0.5)
            return "short", confidence

        return "none", 0

    def _mean_reversion_signal(self, data: Dict) -> Tuple[str, float]:
        """Mean reversion trading signal"""
        rsi = data['rsi']
        price = data['mid']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']

        # Oversold - buy signal
        if rsi < self.config.rsi_oversold and price < bb_lower:
            confidence = min(0.85, (30 - rsi) / 30 * 0.5 + 0.35)
            return "long", confidence

        # Overbought - sell signal
        elif rsi > self.config.rsi_overbought and price > bb_upper:
            confidence = min(0.85, (rsi - 70) / 30 * 0.5 + 0.35)
            return "short", confidence

        return "none", 0

    def _breakout_signal(self, data: Dict) -> Tuple[str, float]:
        """Breakout trading signal"""
        price = data['mid']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']
        volume_ratio = data['volume_ratio']

        # Upside breakout
        if price > bb_upper and volume_ratio > 2.0:
            confidence = min(0.75, volume_ratio / 3 * 0.5 + 0.25)
            return "long", confidence

        # Downside breakout
        elif price < bb_lower and volume_ratio > 2.0:
            confidence = min(0.75, volume_ratio / 3 * 0.5 + 0.25)
            return "short", confidence

        return "none", 0

    def _scalping_signal(self, data: Dict) -> Tuple[str, float]:
        """High-frequency scalping signal"""
        spread_pct = (data['spread'] / data['mid']) * 100

        # Only scalp when spread is tight
        if spread_pct > 0.05:  # Skip if spread > 5 bps
            return "none", 0

        # Quick mean reversion on micro moves
        rsi = data['rsi']

        if rsi < 45:
            return "long", 0.65
        elif rsi > 55:
            return "short", 0.65

        return "none", 0

    async def execute_trade(self, signal: str, data: Dict, confidence: float):
        """Execute a trade based on signal"""
        try:
            # Calculate position size
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)
            position_value = min(
                self.config.max_position_size,
                buying_power * self.config.position_size_pct
            )

            price = data['mid']
            quantity = position_value / price

            # Round to appropriate decimals for crypto
            quantity = round(quantity, 8)

            # Calculate stop loss and take profit
            if signal == "long":
                side = OrderSide.BUY
                stop_price = price * (1 - self.config.stop_loss_pct)
                take_profit_price = price * (1 + self.config.take_profit_pct)
            else:
                side = OrderSide.SELL
                stop_price = price * (1 + self.config.stop_loss_pct)
                take_profit_price = price * (1 - self.config.take_profit_pct)

            # Create market order
            order_data = MarketOrderRequest(
                symbol=self.config.symbol.replace('/', ''),
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC
            )

            # Submit order
            order = self.trading_client.submit_order(order_data)

            # Track position
            position_info = {
                'order_id': order.id,
                'signal': signal,
                'entry_price': price,
                'quantity': quantity,
                'stop_loss': stop_price,
                'take_profit': take_profit_price,
                'entry_time': datetime.now(),
                'confidence': confidence
            }

            self.current_positions[order.id] = position_info
            self.last_trade_time = datetime.now()
            self.total_trades += 1

            logger.info(f"""
            ✅ TRADE EXECUTED:
            Signal: {signal.upper()}
            Price: ${price:,.2f}
            Quantity: {quantity:.8f}
            Stop Loss: ${stop_price:,.2f} ({self.config.stop_loss_pct:.1%})
            Take Profit: ${take_profit_price:,.2f} ({self.config.take_profit_pct:.1%})
            Confidence: {confidence:.1%}
            """)

            # Set OCO (One-Cancels-Other) orders for stop loss and take profit
            await self._set_exit_orders(order.id, stop_price, take_profit_price, quantity, side)

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")

    async def _set_exit_orders(self, position_id: str, stop_price: float,
                              take_profit_price: float, quantity: float,
                              entry_side: OrderSide):
        """Set stop loss and take profit orders"""
        try:
            # Opposite side for exit
            exit_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY

            # Stop loss order
            stop_loss_data = LimitOrderRequest(
                symbol=self.config.symbol.replace('/', ''),
                qty=quantity,
                side=exit_side,
                time_in_force=TimeInForce.GTC,
                limit_price=stop_price
            )

            # Take profit order
            take_profit_data = LimitOrderRequest(
                symbol=self.config.symbol.replace('/', ''),
                qty=quantity,
                side=exit_side,
                time_in_force=TimeInForce.GTC,
                limit_price=take_profit_price
            )

            # Submit orders
            self.trading_client.submit_order(stop_loss_data)
            self.trading_client.submit_order(take_profit_data)

            logger.info("✅ Exit orders placed successfully")

        except Exception as e:
            logger.error(f"❌ Failed to set exit orders: {e}")

    async def monitor_positions(self):
        """Monitor and manage open positions"""
        while self.is_running:
            try:
                if self.current_positions:
                    # Get current market data
                    data = await self.get_market_data()
                    if not data:
                        await asyncio.sleep(5)
                        continue

                    current_price = data['mid']

                    # Check each position
                    for position_id, position in list(self.current_positions.items()):
                        entry_price = position['entry_price']

                        # Calculate PnL
                        if position['signal'] == "long":
                            pnl_pct = (current_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price

                        # Check if we hit stop or target
                        if position['signal'] == "long":
                            if current_price <= position['stop_loss']:
                                await self._close_position(position_id, "stop_loss", current_price)
                            elif current_price >= position['take_profit']:
                                await self._close_position(position_id, "take_profit", current_price)
                        else:
                            if current_price >= position['stop_loss']:
                                await self._close_position(position_id, "stop_loss", current_price)
                            elif current_price <= position['take_profit']:
                                await self._close_position(position_id, "take_profit", current_price)

                        # Update drawdown
                        if pnl_pct < 0:
                            self.current_drawdown = max(abs(pnl_pct), self.current_drawdown)
                            self.max_drawdown = max(self.current_drawdown, self.max_drawdown)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)

    async def _close_position(self, position_id: str, reason: str, exit_price: float):
        """Close a position"""
        try:
            position = self.current_positions[position_id]

            # Calculate PnL
            if position['signal'] == "long":
                pnl = (exit_price - position['entry_price']) * position['quantity']
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

            # Update statistics
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            # Log trade result
            logger.info(f"""
            📊 POSITION CLOSED:
            Reason: {reason}
            Entry: ${position['entry_price']:,.2f}
            Exit: ${exit_price:,.2f}
            PnL: ${pnl:,.2f} ({pnl_pct:.2%})
            Duration: {(datetime.now() - position['entry_time']).seconds}s
            """)

            # Remove from active positions
            del self.current_positions[position_id]

            # Reset drawdown on winning trade
            if pnl > 0:
                self.current_drawdown = 0

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting perpetual trading system...")
        logger.info(f"Strategy: {self.config.strategy.value}")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Risk per trade: {self.config.stop_loss_pct:.1%}")
        logger.info(f"Target per trade: {self.config.take_profit_pct:.1%}")

        self.is_running = True

        # Start position monitoring task
        monitor_task = asyncio.create_task(self.monitor_positions())

        while self.is_running:
            try:
                # Get market data
                data = await self.get_market_data()
                if not data:
                    await asyncio.sleep(5)
                    continue

                # Check for trade signal
                should_trade, signal, confidence = self.should_enter_trade(data)

                if should_trade and confidence >= self.config.min_confidence:
                    await self.execute_trade(signal, data, confidence)

                # Performance check
                if self.total_trades > 0 and self.total_trades % 10 == 0:
                    win_rate = self.winning_trades / self.total_trades
                    logger.info(f"""
                    📈 PERFORMANCE UPDATE:
                    Total Trades: {self.total_trades}
                    Win Rate: {win_rate:.1%}
                    Total PnL: ${self.total_pnl:,.2f}
                    Max Drawdown: {self.max_drawdown:.2%}
                    """)

                # Rate limiting
                await asyncio.sleep(5)  # Check every 5 seconds

            except KeyboardInterrupt:
                logger.info("⏹️ Stopping trading system...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

        # Cancel monitoring task
        monitor_task.cancel()

    def stop(self):
        """Stop the trading system"""
        self.is_running = False
        logger.info("Trading system stopped")

async def main():
    """Main entry point"""
    # Create configuration
    config = TradingConfig(
        strategy=TradingStrategy.MOMENTUM,
        symbol="BTC/USD",
        max_position_size=1000,  # Start small
        position_size_pct=0.02,
        stop_loss_pct=0.001,  # 0.1%
        take_profit_pct=0.005,  # 0.5%
        min_win_rate=0.60,
        paper_trading=True  # Always start with paper trading
    )

    # Create and run trader
    trader = PerpetualNeuralTrader(config)

    try:
        await trader.run_trading_loop()
    except KeyboardInterrupt:
        trader.stop()

if __name__ == "__main__":
    asyncio.run(main())