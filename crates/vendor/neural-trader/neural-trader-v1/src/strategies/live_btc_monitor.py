#!/usr/bin/env python3
"""
Live BTC Monitoring System with Real-time Alerts
Integrates Alpaca data with social sentiment and order flow
"""

import sys
sys.path.append('/workspaces/neural-trader/src')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional
import aiohttp
import websocket
import threading
import time

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from btc_momentum_scanner import BTCMomentumScanner, TradeSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveBTCMonitor:
    def __init__(self):
        # Initialize scanner
        self.scanner = BTCMomentumScanner()

        # Alpaca clients
        self.crypto_client = CryptoHistoricalDataClient()

        # Trade management
        self.active_position = None
        self.position_entry_time = None
        self.trade_results = []

        # Monitoring state
        self.is_running = False
        self.monitoring_interval = 5  # seconds

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_streak = 0
        self.max_drawdown = 0

    def get_latest_bars(self, minutes: int = 60) -> pd.DataFrame:
        """Fetch latest minute bars from Alpaca"""

        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes)

            request = CryptoBarsRequest(
                symbol_or_symbols="BTC/USD",
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time
            )

            bars = self.crypto_client.get_crypto_bars(request)
            df = bars.df

            if not df.empty:
                df = df.reset_index()
                df = self.scanner.calculate_technical_features(df)
                return df

        except Exception as e:
            logger.error(f"Error fetching bars: {e}")

        return pd.DataFrame()

    def check_position_status(self, current_price: float) -> Optional[Dict]:
        """Check if current position hit target or stop"""

        if not self.active_position:
            return None

        signal = self.active_position

        # Check position age (close after 5 minutes if not hit)
        position_age = (datetime.now() - self.position_entry_time).seconds
        if position_age > 300:  # 5 minutes
            pnl = (current_price - signal.entry_price) / signal.entry_price
            if signal.direction == 'SHORT':
                pnl = -pnl

            return {
                'action': 'TIMEOUT',
                'pnl': pnl,
                'exit_price': current_price
            }

        # Check stop loss and target
        if signal.direction == 'LONG':
            if current_price <= signal.stop_loss:
                pnl = (signal.stop_loss - signal.entry_price) / signal.entry_price
                return {
                    'action': 'STOP_LOSS',
                    'pnl': pnl,
                    'exit_price': signal.stop_loss
                }
            elif current_price >= signal.target_price:
                pnl = (signal.target_price - signal.entry_price) / signal.entry_price
                return {
                    'action': 'TARGET',
                    'pnl': pnl,
                    'exit_price': signal.target_price
                }
        else:  # SHORT
            if current_price >= signal.stop_loss:
                pnl = (signal.entry_price - signal.stop_loss) / signal.entry_price
                return {
                    'action': 'STOP_LOSS',
                    'pnl': pnl,
                    'exit_price': signal.stop_loss
                }
            elif current_price <= signal.target_price:
                pnl = (signal.entry_price - signal.target_price) / signal.entry_price
                return {
                    'action': 'TARGET',
                    'pnl': pnl,
                    'exit_price': signal.target_price
                }

        return None

    def close_position(self, result: Dict):
        """Close active position and record results"""

        if not self.active_position:
            return

        # Update statistics
        self.total_trades += 1
        if result['pnl'] > 0:
            self.winning_trades += 1
            self.current_streak = max(0, self.current_streak) + 1
        else:
            self.losing_trades += 1
            self.current_streak = min(0, self.current_streak) - 1

        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'direction': self.active_position.direction,
            'entry_price': self.active_position.entry_price,
            'exit_price': result['exit_price'],
            'pnl': result['pnl'],
            'exit_reason': result['action'],
            'confidence': self.active_position.confidence,
            'sentiment': self.active_position.sentiment_score,
            'order_flow': self.active_position.order_flow_imbalance
        }

        self.trade_results.append(trade_record)

        # Calculate win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # Log results
        logger.info(f"📊 Position Closed: {result['action']}")
        logger.info(f"   PnL: {result['pnl']:.2%}")
        logger.info(f"   Win Rate: {win_rate:.1%} ({self.winning_trades}/{self.total_trades})")
        logger.info(f"   Streak: {self.current_streak}")

        # Clear position
        self.active_position = None
        self.position_entry_time = None

    def display_monitoring_status(self):
        """Display current monitoring status"""

        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        print("\n" + "="*60)
        print("🎯 BTC MOMENTUM SCANNER - LIVE MONITORING")
        print("="*60)

        print(f"\n📊 STRATEGY PARAMETERS:")
        print(f"  Movement Threshold: {self.scanner.MOVEMENT_THRESHOLD:.1%}")
        print(f"  Max Drawdown: {self.scanner.MAX_DRAWDOWN:.1%}")
        print(f"  Required Win Rate: {self.scanner.MIN_WIN_RATE:.0%}")

        print(f"\n📈 PERFORMANCE:")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Winning: {self.winning_trades}")
        print(f"  Losing: {self.losing_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Current Streak: {self.current_streak}")

        if self.active_position:
            print(f"\n🔴 ACTIVE POSITION:")
            print(f"  Direction: {self.active_position.direction}")
            print(f"  Entry: ${self.active_position.entry_price:,.2f}")
            print(f"  Target: ${self.active_position.target_price:,.2f}")
            print(f"  Stop: ${self.active_position.stop_loss:,.2f}")
            print(f"  Confidence: {self.active_position.confidence:.1%}")
        else:
            print(f"\n⚪ No active position")

        print(f"\n📱 SOCIAL MONITORING:")
        print(f"  Twitter: {len(self.scanner.twitter_accounts)} accounts")
        print(f"  Reddit: {len(self.scanner.reddit_subs)} subreddits")
        print(f"  Order Flow: Bookmap integration pending")

        print("="*60)

    async def monitoring_loop(self):
        """Main monitoring loop"""

        logger.info("Starting BTC momentum monitoring...")

        while self.is_running:
            try:
                # Get latest data
                df = self.get_latest_bars(minutes=60)

                if df.empty:
                    logger.warning("No data available")
                    await asyncio.sleep(self.monitoring_interval)
                    continue

                # Get current price
                latest_quote = self.crypto_client.get_crypto_latest_quote(
                    CryptoLatestQuoteRequest(symbol_or_symbols='BTC/USD')
                )
                current_price = list(latest_quote.values())[0].ask_price

                # Check existing position
                if self.active_position:
                    result = self.check_position_status(current_price)
                    if result:
                        self.close_position(result)
                else:
                    # Look for new signal
                    signal = self.scanner.detect_momentum_setup(df)
                    if signal:
                        # Validate win rate requirement
                        if self.scanner.validate_win_rate():
                            logger.info(f"🚀 NEW SIGNAL DETECTED!")
                            logger.info(f"   Direction: {signal.direction}")
                            logger.info(f"   Entry: ${signal.entry_price:,.2f}")
                            logger.info(f"   Target: ${signal.target_price:,.2f}")
                            logger.info(f"   Stop: ${signal.stop_loss:,.2f}")
                            logger.info(f"   Confidence: {signal.confidence:.1%}")

                            self.active_position = signal
                            self.position_entry_time = datetime.now()
                        else:
                            logger.warning(f"Win rate below threshold: {self.scanner.win_rate:.1%}")

                # Display status
                self.display_monitoring_status()

                # Add latest trade to scanner history for win rate calculation
                if self.trade_results:
                    self.scanner.trade_history = [
                        {'pnl': t['pnl']} for t in self.trade_results
                    ]

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            await asyncio.sleep(self.monitoring_interval)

    def start_monitoring(self):
        """Start live monitoring"""
        self.is_running = True
        asyncio.run(self.monitoring_loop())

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        logger.info("Monitoring stopped")

        # Save results
        if self.trade_results:
            df = pd.DataFrame(self.trade_results)
            df.to_csv('btc_momentum_trades.csv', index=False)
            logger.info(f"Saved {len(self.trade_results)} trades to btc_momentum_trades.csv")

def run_backtest():
    """Run historical backtest"""

    print("\n🔬 Running Backtest...")

    scanner = BTCMomentumScanner()
    client = CryptoHistoricalDataClient()

    # Get historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    request = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=end_time
    )

    bars = client.get_crypto_bars(request)
    df = bars.df.reset_index()

    # Run backtest
    results = scanner.backtest(df)

    print("\n📊 BACKTEST RESULTS:")
    print(f"  Total Trades: {results.get('total_trades', 0)}")
    print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
    print(f"  Avg Win: {results.get('avg_win', 0):.2%}")
    print(f"  Avg Loss: {results.get('avg_loss', 0):.2%}")
    print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"  ✅ Meets Criteria: {results.get('meets_criteria', False)}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        run_backtest()
    else:
        monitor = LiveBTCMonitor()
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nMonitoring stopped by user")