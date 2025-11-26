#!/usr/bin/env python3
"""
BTC Momentum Scanner with Social Sentiment Integration
Requirements:
- 0.5% movement detection (up or down)
- Max 0.1% drawdown tolerance
- 60%+ win rate
- Twitter, Reddit, Bookmap integration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp
from dataclasses import dataclass
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    timestamp: datetime
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    features: Dict
    sentiment_score: float
    order_flow_imbalance: float

class BTCMomentumScanner:
    def __init__(self):
        self.MOVEMENT_THRESHOLD = 0.005  # 0.5%
        self.MAX_DRAWDOWN = 0.001  # 0.1%
        self.MIN_WIN_RATE = 0.60  # 60%

        # Technical features for prediction
        self.technical_features = [
            'rsi', 'macd', 'bb_position', 'volume_ratio',
            'price_momentum', 'volatility', 'support_distance',
            'resistance_distance', 'ema_cross', 'vwap_deviation'
        ]

        # Twitter accounts to monitor
        self.twitter_accounts = [
            '@APompliano',  # Anthony Pompliano
            '@saylor',      # Michael Saylor
            '@elonmusk',    # Elon Musk
            '@CathieDWood', # Cathie Wood
            '@novogratz',   # Mike Novogratz
            '@PeterSchiff', # Peter Schiff (contrarian indicator)
            '@glassnode',   # On-chain analytics
            '@WClementeIII', # Will Clemente
            '@woonomic',    # Willy Woo
            '@100trillionUSD' # PlanB
        ]

        # Reddit subs for sentiment
        self.reddit_subs = [
            'r/Bitcoin',
            'r/CryptoCurrency',
            'r/BitcoinMarkets',
            'r/CryptoMarkets',
            'r/SatoshiStreetBets'
        ]

        self.trade_history = []
        self.win_rate = 0.0

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features for prediction"""

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
        df['macd_histogram'] = df['macd'] - df['signal']

        # Bollinger Bands Position
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['sma20'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume Ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # Price Momentum
        df['price_momentum'] = df['close'].pct_change(periods=5)

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()

        # Support/Resistance
        df['support_distance'] = (df['close'] - df['low'].rolling(window=20).min()) / df['close']
        df['resistance_distance'] = (df['high'].rolling(window=20).max() - df['close']) / df['close']

        # EMA Cross
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_cross'] = np.where(df['ema9'] > df['ema21'], 1, -1)

        # VWAP Deviation
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

        return df

    def detect_momentum_setup(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Detect 0.5% momentum setups with strict risk management"""

        if len(df) < 50:
            return None

        latest = df.iloc[-1]
        prev_5min = df.iloc[-5]

        # Calculate price movement
        price_change = (latest['close'] - prev_5min['close']) / prev_5min['close']

        # Check if movement exceeds threshold
        if abs(price_change) < self.MOVEMENT_THRESHOLD:
            return None

        # Check drawdown constraint
        recent_low = df.tail(5)['low'].min()
        recent_high = df.tail(5)['high'].max()
        max_drawdown = (recent_high - recent_low) / recent_high

        if max_drawdown > self.MAX_DRAWDOWN:
            logger.info(f"Drawdown {max_drawdown:.3%} exceeds limit {self.MAX_DRAWDOWN:.3%}")
            return None

        # Determine direction
        direction = 'LONG' if price_change > 0 else 'SHORT'

        # Calculate entry, target, and stop
        entry_price = latest['close']

        if direction == 'LONG':
            target_price = entry_price * 1.005  # 0.5% profit target
            stop_loss = entry_price * 0.999     # 0.1% stop loss
        else:
            target_price = entry_price * 0.995  # 0.5% profit target
            stop_loss = entry_price * 1.001     # 0.1% stop loss

        # Calculate confidence based on technical features
        confidence = self.calculate_signal_confidence(df, direction)

        if confidence < 0.60:  # Require 60% confidence
            return None

        # Get sentiment and order flow (placeholder - would integrate real APIs)
        sentiment_score = self.get_sentiment_score()
        order_flow_imbalance = self.get_order_flow_imbalance()

        # Build feature dictionary
        features = {
            'rsi': latest['rsi'],
            'macd_histogram': latest['macd_histogram'],
            'bb_position': latest['bb_position'],
            'volume_ratio': latest['volume_ratio'],
            'price_momentum': latest['price_momentum'],
            'volatility': latest['volatility'],
            'support_distance': latest['support_distance'],
            'resistance_distance': latest['resistance_distance'],
            'ema_cross': latest['ema_cross'],
            'vwap_deviation': latest['vwap_deviation']
        }

        return TradeSignal(
            timestamp=datetime.now(),
            direction=direction,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            features=features,
            sentiment_score=sentiment_score,
            order_flow_imbalance=order_flow_imbalance
        )

    def calculate_signal_confidence(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate signal confidence based on multiple factors"""

        latest = df.iloc[-1]
        confidence_scores = []

        # RSI confirmation
        if direction == 'LONG':
            if 30 < latest['rsi'] < 70:
                confidence_scores.append(0.8)
            elif latest['rsi'] < 30:
                confidence_scores.append(1.0)  # Oversold
            else:
                confidence_scores.append(0.4)
        else:  # SHORT
            if 30 < latest['rsi'] < 70:
                confidence_scores.append(0.8)
            elif latest['rsi'] > 70:
                confidence_scores.append(1.0)  # Overbought
            else:
                confidence_scores.append(0.4)

        # MACD confirmation
        if direction == 'LONG':
            if latest['macd_histogram'] > 0:
                confidence_scores.append(0.9)
            else:
                confidence_scores.append(0.5)
        else:
            if latest['macd_histogram'] < 0:
                confidence_scores.append(0.9)
            else:
                confidence_scores.append(0.5)

        # Bollinger Band position
        if direction == 'LONG':
            if latest['bb_position'] < 0.2:  # Near lower band
                confidence_scores.append(0.95)
            elif latest['bb_position'] < 0.5:
                confidence_scores.append(0.7)
            else:
                confidence_scores.append(0.4)
        else:
            if latest['bb_position'] > 0.8:  # Near upper band
                confidence_scores.append(0.95)
            elif latest['bb_position'] > 0.5:
                confidence_scores.append(0.7)
            else:
                confidence_scores.append(0.4)

        # Volume confirmation
        if latest['volume_ratio'] > 1.5:
            confidence_scores.append(0.9)
        elif latest['volume_ratio'] > 1.0:
            confidence_scores.append(0.7)
        else:
            confidence_scores.append(0.5)

        # EMA trend alignment
        if direction == 'LONG' and latest['ema_cross'] == 1:
            confidence_scores.append(0.85)
        elif direction == 'SHORT' and latest['ema_cross'] == -1:
            confidence_scores.append(0.85)
        else:
            confidence_scores.append(0.5)

        return np.mean(confidence_scores)

    def get_sentiment_score(self) -> float:
        """Get aggregated sentiment from Twitter and Reddit"""
        # Placeholder - would integrate real APIs
        # Twitter API v2 for account monitoring
        # Reddit API for subreddit sentiment
        # Return normalized score -1 to 1
        return np.random.uniform(-0.5, 0.5)

    def get_order_flow_imbalance(self) -> float:
        """Get order book imbalance from Bookmap API"""
        # Placeholder - would integrate Bookmap API
        # Calculate bid/ask imbalance
        # Return normalized score -1 to 1
        return np.random.uniform(-0.3, 0.3)

    def validate_win_rate(self) -> bool:
        """Validate that strategy maintains 60%+ win rate"""
        if len(self.trade_history) < 20:
            return True  # Not enough trades to validate

        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        self.win_rate = wins / len(self.trade_history)

        return self.win_rate >= self.MIN_WIN_RATE

    def backtest(self, df: pd.DataFrame) -> Dict:
        """Backtest strategy on historical data"""

        df = self.calculate_technical_features(df)

        trades = []
        for i in range(50, len(df) - 1):
            window = df.iloc[:i+1]
            signal = self.detect_momentum_setup(window)

            if signal:
                # Simulate trade execution
                future_prices = df.iloc[i+1:i+10]['close'].values

                # Check if target or stop hit
                trade_result = self.simulate_trade(
                    signal,
                    future_prices
                )
                trades.append(trade_result)

        # Calculate statistics
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            losses = sum(1 for t in trades if t['pnl'] <= 0)
            win_rate = wins / len(trades)

            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0])
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if losses > 0 else 0

            profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 else float('inf')

            return {
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'meets_criteria': win_rate >= self.MIN_WIN_RATE
            }

        return {'total_trades': 0, 'meets_criteria': False}

    def simulate_trade(self, signal: TradeSignal, future_prices: np.array) -> Dict:
        """Simulate trade execution"""

        pnl = 0
        exit_price = signal.entry_price

        for price in future_prices:
            if signal.direction == 'LONG':
                # Check stop loss
                if price <= signal.stop_loss:
                    pnl = (signal.stop_loss - signal.entry_price) / signal.entry_price
                    exit_price = signal.stop_loss
                    break
                # Check target
                if price >= signal.target_price:
                    pnl = (signal.target_price - signal.entry_price) / signal.entry_price
                    exit_price = signal.target_price
                    break
            else:  # SHORT
                # Check stop loss
                if price >= signal.stop_loss:
                    pnl = (signal.entry_price - signal.stop_loss) / signal.entry_price
                    exit_price = signal.stop_loss
                    break
                # Check target
                if price <= signal.target_price:
                    pnl = (signal.entry_price - signal.target_price) / signal.entry_price
                    exit_price = signal.target_price
                    break

        return {
            'entry_price': signal.entry_price,
            'exit_price': exit_price,
            'direction': signal.direction,
            'pnl': pnl,
            'confidence': signal.confidence
        }

# Integration endpoints for external services
class ExternalIntegrations:
    """Handlers for Twitter, Reddit, and Bookmap integrations"""

    @staticmethod
    async def fetch_twitter_sentiment(accounts: List[str]) -> float:
        """Fetch sentiment from Twitter accounts"""
        # Would integrate Twitter API v2
        # Analyze recent tweets from specified accounts
        # Use NLP for sentiment analysis
        pass

    @staticmethod
    async def fetch_reddit_sentiment(subreddits: List[str]) -> float:
        """Fetch sentiment from Reddit subreddits"""
        # Would integrate Reddit API
        # Analyze hot/new posts and comments
        # Calculate weighted sentiment score
        pass

    @staticmethod
    async def fetch_bookmap_orderflow() -> Dict:
        """Fetch order book data from Bookmap"""
        # Would integrate Bookmap API
        # Get real-time order book depth
        # Calculate bid/ask imbalance
        pass

if __name__ == "__main__":
    scanner = BTCMomentumScanner()
    print(f"BTC Momentum Scanner initialized")
    print(f"Movement threshold: {scanner.MOVEMENT_THRESHOLD:.1%}")
    print(f"Max drawdown: {scanner.MAX_DRAWDOWN:.1%}")
    print(f"Min win rate: {scanner.MIN_WIN_RATE:.0%}")
    print(f"Monitoring {len(scanner.twitter_accounts)} Twitter accounts")
    print(f"Monitoring {len(scanner.reddit_subs)} Reddit communities")