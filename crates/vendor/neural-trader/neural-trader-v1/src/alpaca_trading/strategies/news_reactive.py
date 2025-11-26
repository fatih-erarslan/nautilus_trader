"""
News Reactive Strategy for Alpaca WebSocket Trading

Integration with news sentiment for event-driven signals,
reaction time optimization, and confidence-based position sizing.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
import re
from .base_strategy import BaseStreamStrategy, TradingSignal, SignalType


@dataclass
class NewsEvent:
    """News event with sentiment analysis"""
    timestamp: datetime
    symbol: str
    headline: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float      # 0.0 to 1.0
    category: str         # earnings, merger, regulatory, etc.
    source: str
    metadata: Dict[str, Any]


class NewsReactiveStrategy(BaseStreamStrategy):
    """
    News-driven trading strategy with sentiment analysis
    
    Features:
    - Real-time news sentiment integration
    - Event categorization and impact assessment
    - Reaction time optimization
    - Confidence-based position sizing
    - News decay modeling
    """
    
    def __init__(self,
                 symbols: List[str],
                 sentiment_threshold: float = 0.6,
                 min_confidence: float = 0.7,
                 reaction_window_seconds: int = 300,  # 5 minutes
                 news_decay_half_life: int = 1800,   # 30 minutes
                 use_category_weights: bool = True,
                 momentum_confirmation: bool = True,
                 **kwargs):
        """
        Initialize news reactive strategy
        
        Args:
            sentiment_threshold: Minimum absolute sentiment to trade
            min_confidence: Minimum confidence score to consider news
            reaction_window_seconds: Time window to react to news
            news_decay_half_life: Half-life for news impact decay
            use_category_weights: Weight signals by news category
            momentum_confirmation: Require price momentum confirmation
        """
        super().__init__(symbols, **kwargs)
        
        self.sentiment_threshold = sentiment_threshold
        self.min_confidence = min_confidence
        self.reaction_window_seconds = reaction_window_seconds
        self.news_decay_half_life = news_decay_half_life
        self.use_category_weights = use_category_weights
        self.momentum_confirmation = momentum_confirmation
        
        # News event storage
        self.news_events: Dict[str, deque] = {s: deque(maxlen=100) for s in symbols}
        self.active_news: Dict[str, List[NewsEvent]] = {s: [] for s in symbols}
        
        # Category impact weights
        self.category_weights = {
            'earnings': 1.5,
            'merger': 1.3,
            'regulatory': 1.2,
            'product': 1.0,
            'management': 0.8,
            'analyst': 0.7,
            'general': 0.5
        }
        
        # Price reaction tracking
        self.pre_news_prices: Dict[str, Dict] = {}
        self.news_reactions: Dict[str, List] = {s: [] for s in symbols}
        
        # Performance tracking
        self.news_trades = {s: [] for s in symbols}
        self.sentiment_accuracy = {s: {'correct': 0, 'total': 0} for s in symbols}
    
    def _on_trade(self, trade: Dict[str, Any]):
        """Process trade data for price reaction analysis"""
        symbol = trade['symbol']
        price = trade['price']
        
        # Check for news-driven price movements
        self._analyze_price_reaction(symbol, price)
        
        # Update momentum for confirmation
        if self.momentum_confirmation:
            self._update_price_momentum(symbol, price)
    
    def _on_quote(self, quote: Dict[str, Any]):
        """Process quote data"""
        # Can use for spread analysis during news events
        pass
    
    def _on_bar(self, bar: Dict[str, Any]):
        """Process bar data"""
        # Can use for longer-term news impact analysis
        pass
    
    def on_news(self, news: Dict[str, Any]):
        """
        Process incoming news event
        
        Args:
            news: News data with sentiment analysis
        """
        # Parse news event
        event = self._parse_news_event(news)
        
        if not event or event.confidence < self.min_confidence:
            return
            
        symbol = event.symbol
        
        # Store news event
        self.news_events[symbol].append(event)
        self.active_news[symbol].append(event)
        
        # Record pre-news price
        if symbol in self.latest_trades:
            self.pre_news_prices[symbol] = {
                'price': self.latest_trades[symbol]['price'],
                'timestamp': event.timestamp,
                'event': event
            }
        
        self.logger.info(
            f"NEWS: {symbol} - {event.headline[:50]}... "
            f"Sentiment: {event.sentiment_score:.2f}, "
            f"Category: {event.category}"
        )
    
    def _parse_news_event(self, news_data: Dict[str, Any]) -> Optional[NewsEvent]:
        """Parse raw news data into NewsEvent"""
        try:
            # Extract symbol - could be in various formats
            symbol = news_data.get('symbol')
            if not symbol:
                # Try to extract from headline
                symbols_mentioned = self._extract_symbols_from_text(
                    news_data.get('headline', '') + ' ' + news_data.get('summary', '')
                )
                if symbols_mentioned:
                    symbol = symbols_mentioned[0]
                else:
                    return None
                    
            # Only process if we're tracking this symbol
            if symbol not in self.symbols:
                return None
                
            return NewsEvent(
                timestamp=news_data.get('timestamp', datetime.now()),
                symbol=symbol,
                headline=news_data.get('headline', ''),
                sentiment_score=news_data.get('sentiment_score', 0.0),
                confidence=news_data.get('confidence', 0.5),
                category=self._categorize_news(news_data),
                source=news_data.get('source', 'unknown'),
                metadata=news_data.get('metadata', {})
            )
        except Exception as e:
            self.logger.error(f"Error parsing news: {e}")
            return None
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # Simple pattern for stock symbols (uppercase, 1-5 chars)
        pattern = r'\b[A-Z]{1,5}\b'
        potential_symbols = re.findall(pattern, text)
        
        # Filter to tracked symbols
        return [s for s in potential_symbols if s in self.symbols]
    
    def _categorize_news(self, news_data: Dict[str, Any]) -> str:
        """Categorize news event"""
        headline = news_data.get('headline', '').lower()
        summary = news_data.get('summary', '').lower()
        text = headline + ' ' + summary
        
        # Category keywords
        categories = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'quarterly'],
            'merger': ['merger', 'acquisition', 'acquire', 'buyout', 'deal'],
            'regulatory': ['sec', 'fda', 'regulatory', 'investigation', 'fine'],
            'product': ['product', 'launch', 'release', 'announce', 'unveil'],
            'management': ['ceo', 'cfo', 'executive', 'resign', 'appoint'],
            'analyst': ['upgrade', 'downgrade', 'rating', 'price target', 'analyst']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
                
        return 'general'
    
    def _analyze_price_reaction(self, symbol: str, current_price: float):
        """Analyze price reaction to news events"""
        if symbol not in self.pre_news_prices:
            return
            
        pre_news_data = self.pre_news_prices[symbol]
        time_since_news = (datetime.now() - pre_news_data['timestamp']).total_seconds()
        
        # Only analyze within reaction window
        if time_since_news > self.reaction_window_seconds:
            del self.pre_news_prices[symbol]
            return
            
        # Calculate price reaction
        price_change = (current_price - pre_news_data['price']) / pre_news_data['price']
        event = pre_news_data['event']
        
        # Check if reaction matches sentiment
        reaction_correct = (
            (event.sentiment_score > 0 and price_change > 0) or
            (event.sentiment_score < 0 and price_change < 0)
        )
        
        # Update accuracy tracking
        self.sentiment_accuracy[symbol]['total'] += 1
        if reaction_correct:
            self.sentiment_accuracy[symbol]['correct'] += 1
            
        # Store reaction data
        self.news_reactions[symbol].append({
            'event': event,
            'price_change': price_change,
            'reaction_time': time_since_news,
            'correct': reaction_correct
        })
    
    def _update_price_momentum(self, symbol: str, price: float):
        """Update price momentum for confirmation"""
        # Handled by price history in base class
        pass
    
    def calculate_news_impact(self, symbol: str) -> float:
        """
        Calculate current news impact with decay
        
        Returns:
            Weighted sentiment score with decay
        """
        current_time = datetime.now()
        total_impact = 0.0
        
        # Clean up old news
        self.active_news[symbol] = [
            event for event in self.active_news[symbol]
            if (current_time - event.timestamp).total_seconds() < self.reaction_window_seconds * 3
        ]
        
        for event in self.active_news[symbol]:
            # Time decay
            time_elapsed = (current_time - event.timestamp).total_seconds()
            decay_factor = 0.5 ** (time_elapsed / self.news_decay_half_life)
            
            # Category weight
            category_weight = self.category_weights.get(event.category, 0.5) if self.use_category_weights else 1.0
            
            # Calculate impact
            impact = event.sentiment_score * event.confidence * decay_factor * category_weight
            total_impact += impact
            
        return total_impact
    
    def check_momentum_confirmation(self, symbol: str, sentiment: float) -> bool:
        """Check if price momentum confirms news sentiment"""
        if not self.momentum_confirmation:
            return True
            
        # Need price history
        if len(self.price_history[symbol]) < 10:
            return False
            
        # Calculate recent price momentum
        recent_prices = list(self.price_history[symbol])[-10:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Check if momentum aligns with sentiment
        if sentiment > 0:
            return price_change > 0.001  # Positive momentum for positive news
        else:
            return price_change < -0.001  # Negative momentum for negative news
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate news-based trading signals"""
        # Calculate current news impact
        news_impact = self.calculate_news_impact(symbol)
        
        # Need significant news impact
        if abs(news_impact) < self.sentiment_threshold:
            return None
            
        # Get current price
        if symbol not in self.latest_trades:
            return None
            
        current_price = self.latest_trades[symbol]['price']
        has_position = symbol in self.positions
        
        # Entry signals
        if not has_position:
            # Check momentum confirmation
            if self.momentum_confirmation and not self.check_momentum_confirmation(symbol, news_impact):
                return None
                
            # Determine signal type based on sentiment
            if news_impact > self.sentiment_threshold:
                signal_type = SignalType.BUY
                reason = self._generate_buy_reason(symbol)
            elif news_impact < -self.sentiment_threshold:
                # For now, skip short positions
                return None
            else:
                return None
                
            # Calculate confidence based on news impact and accuracy
            accuracy_rate = self._get_sentiment_accuracy(symbol)
            confidence = min(0.95, abs(news_impact) * accuracy_rate)
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=datetime.now(),
                price=current_price,
                quantity=None,  # Will be calculated based on confidence
                confidence=confidence,
                reason=reason,
                metadata={
                    'news_impact': news_impact,
                    'active_news_count': len(self.active_news[symbol]),
                    'sentiment_accuracy': accuracy_rate
                }
            )
            
            # Track trade
            self.news_trades[symbol].append({
                'signal': signal,
                'news_events': list(self.active_news[symbol])
            })
            
            return signal
            
        else:
            # Exit signals for existing positions
            position = self.positions[symbol]
            
            # Exit if news sentiment reverses
            if position.metadata.get('entry_reason', '').startswith('Positive news'):
                if news_impact < -self.sentiment_threshold * 0.5:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE,
                        timestamp=datetime.now(),
                        price=current_price,
                        quantity=position.quantity,
                        confidence=0.8,
                        reason="News sentiment reversed",
                        metadata={'news_impact': news_impact}
                    )
                    
            # Exit if no more active news and price momentum fades
            if not self.active_news[symbol]:
                # Check if we should take profit
                if position.unrealized_pnl > self.position_size * 0.02:  # 2% profit
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE,
                        timestamp=datetime.now(),
                        price=current_price,
                        quantity=position.quantity,
                        confidence=0.7,
                        reason="News impact faded - taking profit",
                        metadata={'final_pnl': position.unrealized_pnl}
                    )
                    
        return None
    
    def _generate_buy_reason(self, symbol: str) -> str:
        """Generate detailed buy reason based on news"""
        if not self.active_news[symbol]:
            return "Positive news sentiment"
            
        # Get most impactful news
        latest_news = max(self.active_news[symbol], key=lambda e: abs(e.sentiment_score * e.confidence))
        
        return f"Positive news: {latest_news.headline[:50]}... ({latest_news.category})"
    
    def _get_sentiment_accuracy(self, symbol: str) -> float:
        """Get historical sentiment prediction accuracy"""
        stats = self.sentiment_accuracy[symbol]
        if stats['total'] == 0:
            return 0.7  # Default accuracy
            
        return stats['correct'] / stats['total']
    
    def calculate_position_size(self, signal: TradingSignal) -> int:
        """
        Calculate position size based on news confidence and impact
        """
        base_size = super().calculate_position_size(signal)
        
        # Adjust based on confidence
        confidence_adjustment = signal.confidence
        
        # Adjust based on sentiment accuracy
        symbol = signal.symbol
        accuracy = self._get_sentiment_accuracy(symbol)
        accuracy_adjustment = max(0.5, min(1.5, accuracy / 0.7))  # Scale around 70% accuracy
        
        # Adjust based on number of confirming news events
        news_count = len(self.active_news[symbol])
        news_adjustment = min(1.5, 1.0 + news_count * 0.1)
        
        # Calculate final size
        adjusted_size = int(base_size * confidence_adjustment * accuracy_adjustment * news_adjustment)
        
        return max(1, adjusted_size)
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get news strategy specific metrics"""
        base_metrics = self.get_performance_summary()
        
        # News-specific metrics
        news_metrics = {
            'active_news_events': {},
            'sentiment_accuracy': {},
            'news_trades': {},
            'category_performance': {}
        }
        
        # Track category performance
        category_pnl = {cat: 0.0 for cat in self.category_weights}
        category_counts = {cat: 0 for cat in self.category_weights}
        
        for symbol in self.symbols:
            # Active news count
            news_metrics['active_news_events'][symbol] = len(self.active_news[symbol])
            
            # Sentiment accuracy
            if self.sentiment_accuracy[symbol]['total'] > 0:
                news_metrics['sentiment_accuracy'][symbol] = self._get_sentiment_accuracy(symbol)
                
            # News trades count
            news_metrics['news_trades'][symbol] = len(self.news_trades[symbol])
            
            # Category performance
            for trade_data in self.news_trades[symbol]:
                for event in trade_data['news_events']:
                    category = event.category
                    # Would need to track PnL per trade for accurate category performance
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        news_metrics['category_performance'] = {
            cat: {
                'count': category_counts[cat],
                'weight': self.category_weights.get(cat, 0.5)
            }
            for cat in category_counts
        }
        
        base_metrics['news_metrics'] = news_metrics
        
        return base_metrics