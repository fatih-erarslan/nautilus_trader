#!/usr/bin/env python3
"""
Real-time news-driven trading system with Alpaca WebSocket streaming
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alpaca_trading.websocket.alpaca_client import AlpacaWebSocketClient
from src.alpaca_trading.websocket.stream_manager import StreamManager
from src.alpaca_trading.rest_client import AlpacaRESTClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_time_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class NewsSignal:
    """News-based trading signal"""
    symbol: str
    sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    magnitude: float  # Expected price move
    source: str
    timestamp: datetime
    headline: str
    
    @property
    def signal_strength(self) -> float:
        """Combined signal strength"""
        return abs(self.sentiment) * self.confidence * self.magnitude


@dataclass
class MarketSignal:
    """Market data-based trading signal"""
    symbol: str
    signal_type: str  # momentum, breakout, reversal
    strength: float  # -1 to 1
    price: float
    volume: float
    timestamp: datetime
    
    
@dataclass
class TradingDecision:
    """Trading decision with risk parameters"""
    symbol: str
    action: str  # buy, sell, hold
    quantity: int
    price_limit: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    reasoning: str
    timestamp: datetime


class NewsAnalyzer:
    """Real-time news sentiment analyzer"""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['bullish', 'growth', 'beat', 'strong', 'up', 'gain', 'rise', 'surge', 'boom'],
            'negative': ['bearish', 'decline', 'miss', 'weak', 'down', 'loss', 'fall', 'crash', 'drop']
        }
        
    async def analyze_news(self, headlines: List[str], symbol: str) -> List[NewsSignal]:
        """Analyze news headlines for trading signals"""
        signals = []
        
        for headline in headlines:
            sentiment = self._calculate_sentiment(headline)
            confidence = self._calculate_confidence(headline, symbol)
            magnitude = self._estimate_magnitude(headline)
            
            signal = NewsSignal(
                symbol=symbol,
                sentiment=sentiment,
                confidence=confidence,
                magnitude=magnitude,
                source="aggregated_news",
                timestamp=datetime.now(),
                headline=headline
            )
            
            signals.append(signal)
            
        return signals
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text"""
        text_lower = text.lower()
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
            
        sentiment = (positive_count - negative_count) / max(total_words * 0.1, 1)
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_confidence(self, text: str, symbol: str) -> float:
        """Calculate confidence in sentiment analysis"""
        # Higher confidence if symbol is mentioned
        confidence = 0.5
        if symbol.lower() in text.lower():
            confidence += 0.3
            
        # Higher confidence for specific financial terms
        financial_terms = ['earnings', 'revenue', 'profit', 'guidance', 'forecast']
        if any(term in text.lower() for term in financial_terms):
            confidence += 0.2
            
        return min(1.0, confidence)
    
    def _estimate_magnitude(self, text: str) -> float:
        """Estimate expected price magnitude"""
        magnitude_words = {
            'surge': 0.05, 'boom': 0.04, 'soar': 0.04,
            'crash': 0.05, 'plunge': 0.04, 'tank': 0.04,
            'rise': 0.02, 'gain': 0.02, 'up': 0.015,
            'fall': 0.02, 'drop': 0.02, 'down': 0.015
        }
        
        text_lower = text.lower()
        max_magnitude = 0.01  # Default 1%
        
        for word, magnitude in magnitude_words.items():
            if word in text_lower:
                max_magnitude = max(max_magnitude, magnitude)
                
        return max_magnitude


class MarketAnalyzer:
    """Real-time market data analyzer"""
    
    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        
    async def analyze_trade(self, trade_data: Dict) -> Optional[MarketSignal]:
        """Analyze trade data for signals"""
        symbol = trade_data.get('S')
        price = trade_data.get('p', 0)
        size = trade_data.get('s', 0)
        
        if not symbol or not price:
            return None
            
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append({'price': price, 'time': datetime.now()})
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(minutes=5)
        self.price_history[symbol] = [
            p for p in self.price_history[symbol] 
            if p['time'] > cutoff
        ]
        
        # Generate momentum signal
        return self._generate_momentum_signal(symbol, price, size)
    
    def _generate_momentum_signal(self, symbol: str, current_price: float, volume: float) -> Optional[MarketSignal]:
        """Generate momentum-based signal"""
        if len(self.price_history.get(symbol, [])) < 5:
            return None
            
        prices = [p['price'] for p in self.price_history[symbol][-10:]]
        
        # Calculate momentum
        if len(prices) >= 2:
            momentum = (current_price - prices[0]) / prices[0]
            
            # Generate signal based on momentum
            if abs(momentum) > 0.002:  # 0.2% threshold
                signal_type = "momentum"
                strength = max(-1.0, min(1.0, momentum * 10))  # Scale momentum
                
                return MarketSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=strength,
                    price=current_price,
                    volume=volume,
                    timestamp=datetime.now()
                )
        
        return None


class TradingEngine:
    """Trading decision engine combining news and market signals"""
    
    def __init__(self, max_position_size: float = 1000.0, risk_per_trade: float = 0.02):
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.positions = {}
        self.pending_orders = {}
        
    async def make_decision(self, 
                          symbol: str,
                          news_signals: List[NewsSignal], 
                          market_signal: Optional[MarketSignal],
                          current_price: float) -> Optional[TradingDecision]:
        """Make trading decision based on combined signals"""
        
        # Calculate combined signal strength
        news_strength = self._combine_news_signals(news_signals)
        market_strength = market_signal.strength if market_signal else 0.0
        
        # Weight the signals
        combined_strength = (news_strength * 0.6) + (market_strength * 0.4)
        
        # Apply filters
        if abs(combined_strength) < 0.3:  # Minimum signal threshold
            return None
            
        # Check if we already have a position
        current_position = self.positions.get(symbol, 0)
        
        # Determine action
        if combined_strength > 0.5 and current_position <= 0:
            action = "buy"
            quantity = self._calculate_position_size(current_price, combined_strength)
        elif combined_strength < -0.5 and current_position >= 0:
            action = "sell"
            quantity = max(abs(current_position), 
                         self._calculate_position_size(current_price, abs(combined_strength)))
        else:
            return None
            
        # Calculate risk parameters
        stop_loss = self._calculate_stop_loss(current_price, action)
        take_profit = self._calculate_take_profit(current_price, action, combined_strength)
        
        reasoning = self._generate_reasoning(news_signals, market_signal, combined_strength)
        
        return TradingDecision(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price_limit=current_price * (1.001 if action == "buy" else 0.999),  # Small buffer
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=abs(combined_strength),
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _combine_news_signals(self, signals: List[NewsSignal]) -> float:
        """Combine multiple news signals"""
        if not signals:
            return 0.0
            
        # Weight by signal strength and recency
        total_weight = 0.0
        weighted_sentiment = 0.0
        
        for signal in signals:
            weight = signal.signal_strength
            total_weight += weight
            weighted_sentiment += signal.sentiment * weight
            
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _calculate_position_size(self, price: float, strength: float) -> int:
        """Calculate position size based on risk management"""
        risk_amount = self.max_position_size * self.risk_per_trade * strength
        shares = int(risk_amount / price)
        return max(1, min(shares, int(self.max_position_size / price)))
    
    def _calculate_stop_loss(self, price: float, action: str) -> float:
        """Calculate stop loss price"""
        stop_pct = 0.02  # 2% stop loss
        if action == "buy":
            return price * (1 - stop_pct)
        else:
            return price * (1 + stop_pct)
    
    def _calculate_take_profit(self, price: float, action: str, strength: float) -> float:
        """Calculate take profit price"""
        profit_pct = 0.03 * strength  # Scale profit target with signal strength
        if action == "buy":
            return price * (1 + profit_pct)
        else:
            return price * (1 - profit_pct)
    
    def _generate_reasoning(self, 
                          news_signals: List[NewsSignal], 
                          market_signal: Optional[MarketSignal],
                          combined_strength: float) -> str:
        """Generate human-readable reasoning"""
        reasons = []
        
        if news_signals:
            avg_sentiment = sum(s.sentiment for s in news_signals) / len(news_signals)
            sentiment_desc = "positive" if avg_sentiment > 0 else "negative"
            reasons.append(f"News sentiment: {sentiment_desc} ({avg_sentiment:.2f})")
            
        if market_signal:
            reasons.append(f"Market momentum: {market_signal.signal_type} ({market_signal.strength:.2f})")
            
        reasons.append(f"Combined signal strength: {combined_strength:.2f}")
        
        return " | ".join(reasons)


class RealTimeTrader:
    """Main real-time trading system"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        
        # Initialize components
        self.ws_client = AlpacaWebSocketClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            stream_type="data",
            feed="iex"
        )
        
        self.alpaca_client = AlpacaRESTClient()
        self.news_analyzer = NewsAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        self.trading_engine = TradingEngine()
        
        # Trading symbols
        self.symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
        
        # Statistics
        self.stats = {
            'trades_executed': 0,
            'trades_profitable': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now()
        }
        
        self.running = False
        
    async def start(self):
        """Start the real-time trading system"""
        logger.info("🚀 Starting Real-Time News-Driven Trading System")
        logger.info(f"📈 Monitoring symbols: {', '.join(self.symbols)}")
        
        self.running = True
        
        try:
            # Connect to WebSocket
            await self.ws_client.connect()
            logger.info("✅ Connected to Alpaca WebSocket")
            
            # Register handlers
            self.ws_client.register_handler("t", self._handle_trade)
            self.ws_client.register_handler("q", self._handle_quote)
            
            # Subscribe to symbols
            await self.ws_client.subscribe(
                trades=self.symbols,
                quotes=self.symbols
            )
            logger.info(f"✅ Subscribed to {len(self.symbols)} symbols")
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._news_monitor()),
                asyncio.create_task(self._trade_monitor()),
                asyncio.create_task(self._stats_reporter()),
                asyncio.create_task(self._keep_alive())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"❌ Trading system error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _handle_trade(self, messages):
        """Handle trade messages from WebSocket"""
        for trade in messages:
            symbol = trade.get('S')
            price = trade.get('p', 0)
            
            logger.debug(f"📈 Trade: {symbol} @ ${price:.2f}")
            
            # Generate market signal
            market_signal = await self.market_analyzer.analyze_trade(trade)
            
            if market_signal and abs(market_signal.strength) > 0.3:
                logger.info(f"🎯 Market signal: {symbol} {market_signal.signal_type} strength: {market_signal.strength:.2f}")
                
                # Get recent news (mock for now - would integrate with real news API)
                news_signals = await self._get_news_signals(symbol)
                
                # Make trading decision
                decision = await self.trading_engine.make_decision(
                    symbol, news_signals, market_signal, price
                )
                
                if decision:
                    await self._execute_trade(decision)
    
    async def _handle_quote(self, messages):
        """Handle quote messages from WebSocket"""
        for quote in messages:
            symbol = quote.get('S')
            bid = quote.get('bp', 0)
            ask = quote.get('ap', 0)
            
            # Log every 100th quote to avoid spam
            if hasattr(self, '_quote_count'):
                self._quote_count += 1
            else:
                self._quote_count = 1
                
            if self._quote_count % 100 == 0:
                logger.debug(f"📊 Quote: {symbol} ${bid:.2f}/${ask:.2f}")
    
    async def _get_news_signals(self, symbol: str) -> List[NewsSignal]:
        """Get news signals for symbol (mock implementation)"""
        # This would integrate with real news APIs like:
        # - Alpha Vantage News
        # - NewsAPI
        # - Finnhub
        # - Twitter/X API
        
        # Mock news headlines for demonstration
        mock_headlines = [
            f"{symbol} beats earnings expectations",
            f"Analysts upgrade {symbol} to buy",
            f"{symbol} announces new product launch"
        ]
        
        return await self.news_analyzer.analyze_news(mock_headlines, symbol)
    
    async def _execute_trade(self, decision: TradingDecision):
        """Execute trading decision"""
        try:
            logger.info(f"💰 TRADE DECISION: {decision.action.upper()} {decision.quantity} {decision.symbol}")
            logger.info(f"   Reasoning: {decision.reasoning}")
            logger.info(f"   Confidence: {decision.confidence:.2f}")
            
            # Execute the trade (using paper trading)
            if decision.action == "buy":
                order = await self._place_buy_order(decision)
            elif decision.action == "sell":
                order = await self._place_sell_order(decision)
            else:
                return
                
            if order:
                logger.info(f"✅ Order placed: {order}")
                self.stats['trades_executed'] += 1
                
                # Update position tracking
                if decision.action == "buy":
                    self.trading_engine.positions[decision.symbol] = \
                        self.trading_engine.positions.get(decision.symbol, 0) + decision.quantity
                elif decision.action == "sell":
                    self.trading_engine.positions[decision.symbol] = \
                        self.trading_engine.positions.get(decision.symbol, 0) - decision.quantity
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
    
    async def _place_buy_order(self, decision: TradingDecision) -> Optional[Dict]:
        """Place buy order"""
        try:
            # Use limit order with small buffer
            order_data = {
                'symbol': decision.symbol,
                'qty': str(decision.quantity),
                'side': 'buy',
                'type': 'limit',
                'time_in_force': 'day',
                'limit_price': str(decision.price_limit)
            }
            
            # Add stop loss and take profit as bracket order
            if decision.stop_loss and decision.take_profit:
                order_data.update({
                    'order_class': 'bracket',
                    'stop_loss': {'stop_price': str(decision.stop_loss)},
                    'take_profit': {'limit_price': str(decision.take_profit)}
                })
            
            async with self.alpaca_client as client:
                result = await client.create_order(**order_data)
            return result
            
        except Exception as e:
            logger.error(f"❌ Buy order failed: {e}")
            return None
    
    async def _place_sell_order(self, decision: TradingDecision) -> Optional[Dict]:
        """Place sell order"""
        try:
            order_data = {
                'symbol': decision.symbol,
                'qty': str(decision.quantity),
                'side': 'sell',
                'type': 'limit',
                'time_in_force': 'day',
                'limit_price': str(decision.price_limit)
            }
            
            async with self.alpaca_client as client:
                result = await client.create_order(**order_data)
            return result
            
        except Exception as e:
            logger.error(f"❌ Sell order failed: {e}")
            return None
    
    async def _news_monitor(self):
        """Background task to monitor news"""
        while self.running:
            try:
                # Monitor news sources every 30 seconds
                await asyncio.sleep(30)
                
                # This would check real news APIs
                logger.debug("📰 Checking news sources...")
                
            except Exception as e:
                logger.error(f"❌ News monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _trade_monitor(self):
        """Background task to monitor trade execution"""
        while self.running:
            try:
                # Check order status every 10 seconds
                await asyncio.sleep(10)
                
                # Get open orders
                async with self.alpaca_client as client:
                    orders = await client.list_orders(status='open')
                
                if orders:
                    logger.debug(f"📋 {len(orders)} open orders")
                
            except Exception as e:
                logger.error(f"❌ Trade monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _stats_reporter(self):
        """Background task to report statistics"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                runtime = datetime.now() - self.stats['start_time']
                
                logger.info("📊 TRADING STATISTICS")
                logger.info(f"   Runtime: {runtime}")
                logger.info(f"   Trades executed: {self.stats['trades_executed']}")
                logger.info(f"   Win rate: {self.stats['trades_profitable'] / max(1, self.stats['trades_executed']) * 100:.1f}%")
                logger.info(f"   Total P&L: ${self.stats['total_pnl']:.2f}")
                
                # Get current positions
                async with self.alpaca_client as client:
                    positions = await client.list_positions()
                if positions:
                    logger.info(f"   Open positions: {len(positions)}")
                    for pos in positions[:3]:  # Show first 3
                        pnl = float(pos.get('unrealized_pl', 0))
                        logger.info(f"     {pos['symbol']}: {pos['qty']} shares, P&L: ${pnl:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Stats reporter error: {e}")
                await asyncio.sleep(60)
    
    async def _keep_alive(self):
        """Keep the system running"""
        while self.running:
            await asyncio.sleep(1)
    
    async def shutdown(self):
        """Shutdown the trading system"""
        logger.info("🛑 Shutting down trading system...")
        self.running = False
        
        if self.ws_client:
            await self.ws_client.disconnect()
        
        logger.info("✅ Trading system shutdown complete")


async def main():
    """Main entry point"""
    trader = RealTimeTrader()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        logger.info("⏹️ Stopped by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        await trader.shutdown()


if __name__ == "__main__":
    asyncio.run(main())