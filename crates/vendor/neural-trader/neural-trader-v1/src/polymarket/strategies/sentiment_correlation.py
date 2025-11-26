"""
Sentiment Correlation Trading Strategy

Correlates news sentiment analysis with prediction market prices to identify
trading opportunities based on sentiment-price divergences.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentSignal:
    """Represents a sentiment-based trading signal"""
    market_id: str
    sentiment_score: float  # -1.0 to 1.0
    price: float
    divergence: float
    confidence: float
    news_volume: int
    sources: List[str]
    signal_strength: str  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    expected_move: float


class SentimentCorrelationStrategy:
    """
    Trading strategy that correlates news sentiment with market prices.
    
    Key principles:
    1. News sentiment often leads price movements
    2. Large sentiment-price divergences create opportunities
    3. Higher news volume increases signal reliability
    4. Multiple confirming sources increase confidence
    """
    
    def __init__(self,
                 sentiment_threshold: float = 0.3,
                 divergence_threshold: float = 0.2,
                 min_news_volume: int = 5,
                 price_impact_decay: float = 0.1,
                 lookback_hours: int = 24):
        """
        Initialize sentiment correlation strategy.
        
        Args:
            sentiment_threshold: Minimum absolute sentiment to trade
            divergence_threshold: Minimum sentiment-price divergence
            min_news_volume: Minimum news articles for signal
            price_impact_decay: Hourly decay rate for sentiment impact
            lookback_hours: Hours of history to analyze
        """
        self.sentiment_threshold = sentiment_threshold
        self.divergence_threshold = divergence_threshold
        self.min_news_volume = min_news_volume
        self.price_impact_decay = price_impact_decay
        self.lookback_hours = lookback_hours
        
    def analyze_sentiment(self,
                         news_data: List[Dict],
                         market_price: float,
                         market_id: str) -> Optional[SentimentSignal]:
        """
        Analyze news sentiment and generate trading signal.
        
        Args:
            news_data: List of news items with sentiment scores
            market_price: Current market price
            market_id: Market identifier
            
        Returns:
            SentimentSignal if actionable, None otherwise
        """
        if len(news_data) < self.min_news_volume:
            return None
        
        # Calculate weighted sentiment score
        sentiment_score = self._calculate_weighted_sentiment(news_data)
        
        # Calculate expected price based on sentiment
        expected_price = self._sentiment_to_price(sentiment_score)
        
        # Calculate divergence
        divergence = expected_price - market_price
        
        if abs(divergence) < self.divergence_threshold:
            return None
        
        # Determine signal strength
        signal_strength = self._determine_signal_strength(sentiment_score, divergence)
        
        # Calculate confidence based on news volume and source diversity
        confidence = self._calculate_confidence(news_data, divergence)
        
        # Expected price move
        expected_move = divergence * confidence * 0.7  # Conservative estimate
        
        return SentimentSignal(
            market_id=market_id,
            sentiment_score=sentiment_score,
            price=market_price,
            divergence=divergence,
            confidence=confidence,
            news_volume=len(news_data),
            sources=list(set(item.get('source', 'unknown') for item in news_data)),
            signal_strength=signal_strength,
            expected_move=expected_move
        )
    
    def _calculate_weighted_sentiment(self, news_data: List[Dict]) -> float:
        """
        Calculate time-weighted sentiment score.
        
        More recent news has higher weight.
        """
        current_time = datetime.now()
        total_weight = 0
        weighted_sentiment = 0
        
        for item in news_data:
            # Extract sentiment and timestamp
            sentiment = item.get('sentiment', 0)
            timestamp = item.get('timestamp', current_time)
            
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Calculate time decay weight
            hours_old = (current_time - timestamp).total_seconds() / 3600
            weight = np.exp(-self.price_impact_decay * hours_old)
            
            # Weight by credibility if available
            credibility = item.get('credibility', 1.0)
            weight *= credibility
            
            weighted_sentiment += sentiment * weight
            total_weight += weight
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0
    
    def _sentiment_to_price(self, sentiment: float) -> float:
        """
        Convert sentiment score to expected price.
        
        Uses sigmoid transformation to map sentiment to probability.
        """
        # Sigmoid transformation: p = 1 / (1 + exp(-k*sentiment))
        # k controls sensitivity
        k = 2.0
        return 1 / (1 + np.exp(-k * sentiment))
    
    def _determine_signal_strength(self, 
                                 sentiment: float,
                                 divergence: float) -> str:
        """
        Determine trading signal strength based on sentiment and divergence.
        """
        abs_sentiment = abs(sentiment)
        abs_divergence = abs(divergence)
        
        if abs_sentiment > 0.7 and abs_divergence > 0.3:
            return 'strong_buy' if divergence > 0 else 'strong_sell'
        elif abs_sentiment > 0.5 and abs_divergence > 0.2:
            return 'buy' if divergence > 0 else 'sell'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, 
                            news_data: List[Dict],
                            divergence: float) -> float:
        """
        Calculate confidence score based on news characteristics.
        """
        # Base confidence from news volume
        volume_confidence = min(len(news_data) / 20, 1.0)  # Max at 20 articles
        
        # Source diversity bonus
        unique_sources = len(set(item.get('source', 'unknown') for item in news_data))
        diversity_bonus = min(unique_sources / 5, 0.3)  # Max 0.3 bonus at 5+ sources
        
        # Sentiment consensus
        sentiments = [item.get('sentiment', 0) for item in news_data]
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 1.0
        consensus_factor = max(0, 1 - sentiment_std)  # Higher consensus = higher confidence
        
        # Divergence magnitude factor
        divergence_factor = min(abs(divergence) / 0.5, 1.0)  # Max at 0.5 divergence
        
        # Combine factors
        confidence = (volume_confidence * 0.4 + 
                     diversity_bonus + 
                     consensus_factor * 0.3 +
                     divergence_factor * 0.3)
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def correlate_historical(self,
                           sentiment_history: List[Tuple[datetime, float]],
                           price_history: List[Tuple[datetime, float]]) -> Dict:
        """
        Analyze historical correlation between sentiment and price.
        
        Returns correlation metrics and optimal lag time.
        """
        # Ensure same length and alignment
        min_length = min(len(sentiment_history), len(price_history))
        sentiment_values = np.array([s[1] for s in sentiment_history[-min_length:]])
        price_values = np.array([p[1] for p in price_history[-min_length:]])
        
        # Calculate correlations at different lags
        max_lag = min(24, min_length // 4)  # Max 24 hours lag
        correlations = []
        
        for lag in range(max_lag):
            if lag == 0:
                corr = np.corrcoef(sentiment_values, price_values)[0, 1]
            else:
                corr = np.corrcoef(sentiment_values[:-lag], price_values[lag:])[0, 1]
            correlations.append((lag, corr))
        
        # Find optimal lag
        optimal_lag = max(correlations, key=lambda x: abs(x[1]))
        
        # Calculate predictive power
        if optimal_lag[0] > 0:
            # Use lagged sentiment to predict price
            predictions = sentiment_values[:-optimal_lag[0]]
            actuals = price_values[optimal_lag[0]:]
            mse = np.mean((predictions - actuals) ** 2)
            r_squared = 1 - (mse / np.var(actuals))
        else:
            r_squared = optimal_lag[1] ** 2
        
        return {
            'correlation': optimal_lag[1],
            'optimal_lag_hours': optimal_lag[0],
            'r_squared': r_squared,
            'all_correlations': correlations,
            'predictive_power': 'high' if r_squared > 0.5 else 'medium' if r_squared > 0.3 else 'low'
        }
    
    def generate_signals(self,
                        markets: List[Dict],
                        news_data: Dict[str, List[Dict]]) -> List[SentimentSignal]:
        """
        Generate trading signals for multiple markets.
        
        Args:
            markets: List of market data
            news_data: Dict of market_id -> news items
            
        Returns:
            List of sentiment signals
        """
        signals = []
        
        for market in markets:
            market_id = market['id']
            market_news = news_data.get(market_id, [])
            
            if market_news:
                signal = self.analyze_sentiment(
                    market_news,
                    market['price'],
                    market_id
                )
                
                if signal and signal.signal_strength != 'neutral':
                    signals.append(signal)
        
        # Sort by expected absolute return
        signals.sort(key=lambda s: abs(s.expected_move), reverse=True)
        
        return signals
    
    def calculate_position_size(self, 
                              signal: SentimentSignal,
                              portfolio_value: float,
                              existing_position: float = 0) -> float:
        """
        Calculate optimal position size based on signal strength and confidence.
        
        Args:
            signal: Sentiment signal
            portfolio_value: Total portfolio value
            existing_position: Current position in this market
            
        Returns:
            Recommended position size (positive for buy, negative for sell)
        """
        # Base allocation based on signal strength
        strength_allocations = {
            'strong_buy': 0.10,
            'buy': 0.05,
            'neutral': 0.0,
            'sell': -0.05,
            'strong_sell': -0.10
        }
        
        base_allocation = strength_allocations.get(signal.signal_strength, 0)
        
        # Adjust by confidence
        adjusted_allocation = base_allocation * signal.confidence
        
        # Apply Kelly Criterion
        win_prob = signal.confidence
        avg_win = abs(signal.expected_move)
        avg_loss = 0.1  # Assume 10% stop loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Final allocation
        final_allocation = min(abs(adjusted_allocation), kelly_fraction) * np.sign(adjusted_allocation)
        
        # Calculate position size
        target_position = portfolio_value * final_allocation
        
        # Adjust for existing position
        position_delta = target_position - existing_position
        
        return position_delta
    
    def backtest_sentiment_strategy(self,
                                  historical_data: Dict,
                                  initial_capital: float = 10000) -> Dict:
        """
        Backtest the sentiment correlation strategy.
        
        Args:
            historical_data: Dict with 'prices', 'sentiments', 'news' keys
            initial_capital: Starting capital
            
        Returns:
            Backtest results with performance metrics
        """
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = []
        
        # Simulate trading over historical data
        timestamps = historical_data['timestamps']
        
        for i, timestamp in enumerate(timestamps):
            # Get current market data
            current_prices = historical_data['prices'][i]
            current_news = historical_data['news'].get(timestamp, {})
            
            # Generate signals
            market_list = [{'id': k, 'price': v} for k, v in current_prices.items()]
            signals = self.generate_signals(market_list, current_news)
            
            # Execute trades based on signals
            for signal in signals[:5]:  # Limit to top 5 signals
                position_size = self.calculate_position_size(
                    signal,
                    capital,
                    positions.get(signal.market_id, 0)
                )
                
                if abs(position_size) > 50:  # Minimum trade size
                    # Record trade
                    trades.append({
                        'timestamp': timestamp,
                        'market_id': signal.market_id,
                        'action': 'buy' if position_size > 0 else 'sell',
                        'size': abs(position_size),
                        'price': signal.price,
                        'signal_strength': signal.signal_strength,
                        'sentiment': signal.sentiment_score
                    })
                    
                    # Update position
                    positions[signal.market_id] = positions.get(signal.market_id, 0) + position_size
                    capital -= abs(position_size) * 0.002  # Transaction cost
            
            # Calculate portfolio value
            portfolio_value = capital
            for market_id, position in positions.items():
                current_price = current_prices.get(market_id, 0.5)
                portfolio_value += position * current_price
            
            equity_curve.append((timestamp, portfolio_value))
        
        # Calculate performance metrics
        returns = np.diff([e[1] for e in equity_curve]) / [e[1] for e in equity_curve[:-1]]
        
        return {
            'total_return': (equity_curve[-1][1] - initial_capital) / initial_capital,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'total_trades': len(trades),
            'win_rate': self._calculate_win_rate(trades, historical_data),
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[Tuple]) -> float:
        """Calculate maximum drawdown from equity curve."""
        values = [e[1] for e in equity_curve]
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict], historical_data: Dict) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0
        
        wins = 0
        for trade in trades:
            # Simplified: check if price moved in expected direction
            # In real implementation, would track actual P&L
            entry_price = trade['price']
            market_id = trade['market_id']
            
            # Find exit price (simplified: next day's price)
            # Real implementation would track actual exit
            if trade['action'] == 'buy' and entry_price < 0.5:
                wins += 1
            elif trade['action'] == 'sell' and entry_price > 0.5:
                wins += 1
        
        return wins / len(trades)