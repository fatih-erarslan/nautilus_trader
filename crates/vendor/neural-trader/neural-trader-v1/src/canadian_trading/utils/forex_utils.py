"""
Forex Trading Utilities for Canadian Trading Platform
Provides advanced forex analysis, currency correlation, and trading tools
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging
from collections import defaultdict, deque
import requests
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CurrencyStrength:
    """Currency strength analysis results"""
    currency: str
    strength_score: float
    trend: str  # 'strengthening', 'weakening', 'neutral'
    momentum: float
    relative_rankings: Dict[str, float]


@dataclass
class CorrelationMatrix:
    """Forex pair correlation analysis"""
    pairs: List[str]
    correlation_matrix: np.ndarray
    significant_correlations: List[Tuple[str, str, float]]
    correlation_clusters: Dict[str, List[str]]
    risk_warnings: List[str]


@dataclass
class OptimalTradingTimes:
    """Optimal trading time analysis"""
    pair: str
    best_hours: List[int]
    best_sessions: List[str]
    volatility_by_hour: Dict[int, float]
    spread_by_hour: Dict[int, float]
    liquidity_score_by_hour: Dict[int, float]


@dataclass
class ForexPattern:
    """Technical pattern recognition"""
    pattern_name: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    pattern_strength: float


class ForexUtils:
    """
    Comprehensive forex trading utilities for Canadian forex trading
    """
    
    # Major currency pairs including CAD
    MAJOR_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'NZD', 'CHF']
    
    # CAD cross pairs
    CAD_CROSSES = {
        'majors': ['USD_CAD', 'EUR_CAD', 'GBP_CAD', 'CAD_JPY', 'AUD_CAD', 'NZD_CAD', 'CAD_CHF'],
        'minors': ['CAD_SGD', 'CAD_HKD', 'CAD_NOK', 'CAD_SEK', 'CAD_DKK', 'CAD_PLN'],
        'exotics': ['CAD_MXN', 'CAD_ZAR', 'CAD_TRY', 'CAD_RUB', 'CAD_BRL']
    }
    
    # Trading session times (UTC)
    TRADING_SESSIONS = {
        'sydney': {'start': 21, 'end': 6},     # 9 PM - 6 AM UTC
        'tokyo': {'start': 0, 'end': 9},       # 12 AM - 9 AM UTC
        'london': {'start': 8, 'end': 17},     # 8 AM - 5 PM UTC
        'new_york': {'start': 13, 'end': 22}   # 1 PM - 10 PM UTC
    }
    
    # Session overlaps (highest liquidity)
    SESSION_OVERLAPS = {
        'tokyo_london': {'start': 8, 'end': 9},
        'london_ny': {'start': 13, 'end': 17}
    }
    
    def __init__(self):
        """Initialize forex utilities"""
        self.price_history = defaultdict(lambda: deque(maxlen=10000))
        self.correlation_cache = {}
        self.pattern_cache = defaultdict(list)
        self.session_volatility = defaultdict(lambda: defaultdict(list))
        
    def calculate_currency_strength(self, 
                                  price_data: Dict[str, List[float]],
                                  lookback_period: int = 20) -> Dict[str, CurrencyStrength]:
        """
        Calculate relative currency strength using multiple pairs
        
        Args:
            price_data: Dictionary of pair prices {pair: [prices]}
            lookback_period: Period for strength calculation
            
        Returns:
            Currency strength analysis for each currency
        """
        currency_changes = defaultdict(list)
        
        # Calculate percentage changes for each pair
        for pair, prices in price_data.items():
            if len(prices) < lookback_period:
                continue
                
            # Calculate percentage change
            pct_change = ((prices[-1] - prices[-lookback_period]) / prices[-lookback_period]) * 100
            
            # Parse currencies from pair
            if '_' in pair:
                base, quote = pair.split('_')
                
                # Base currency strengthens when pair goes up
                currency_changes[base].append(pct_change)
                # Quote currency weakens when pair goes up
                currency_changes[quote].append(-pct_change)
                
        # Calculate strength scores
        strength_results = {}
        
        for currency in self.MAJOR_CURRENCIES:
            if currency not in currency_changes or not currency_changes[currency]:
                continue
                
            changes = currency_changes[currency]
            
            # Calculate strength score (average of all pair changes)
            strength_score = np.mean(changes)
            
            # Calculate momentum (recent vs older changes)
            if len(changes) >= 2:
                recent_strength = np.mean(changes[-len(changes)//2:])
                older_strength = np.mean(changes[:len(changes)//2])
                momentum = recent_strength - older_strength
            else:
                momentum = 0
                
            # Determine trend
            if strength_score > 1:
                trend = 'strengthening'
            elif strength_score < -1:
                trend = 'weakening'
            else:
                trend = 'neutral'
                
            # Calculate relative rankings
            relative_rankings = {}
            for other_currency, other_changes in currency_changes.items():
                if other_currency != currency and other_changes:
                    relative_rankings[other_currency] = strength_score - np.mean(other_changes)
                    
            strength_results[currency] = CurrencyStrength(
                currency=currency,
                strength_score=strength_score,
                trend=trend,
                momentum=momentum,
                relative_rankings=relative_rankings
            )
            
        return strength_results
        
    def calculate_pair_correlations(self,
                                  price_data: Dict[str, pd.Series],
                                  rolling_window: int = 30) -> CorrelationMatrix:
        """
        Calculate rolling correlations between forex pairs
        
        Args:
            price_data: Dictionary of price series for each pair
            rolling_window: Window for rolling correlation
            
        Returns:
            Correlation matrix and analysis
        """
        pairs = list(price_data.keys())
        n_pairs = len(pairs)
        
        # Calculate returns
        returns = {}
        for pair, prices in price_data.items():
            returns[pair] = prices.pct_change().dropna()
            
        # Calculate correlation matrix
        correlation_matrix = np.zeros((n_pairs, n_pairs))
        
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i <= j:
                    if pair1 == pair2:
                        correlation_matrix[i, j] = 1.0
                    else:
                        # Calculate rolling correlation
                        corr = returns[pair1].rolling(rolling_window).corr(returns[pair2])
                        correlation_matrix[i, j] = corr.iloc[-1] if not corr.empty else 0
                        correlation_matrix[j, i] = correlation_matrix[i, j]
                        
        # Find significant correlations (> 0.7 or < -0.7)
        significant_correlations = []
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                corr_value = correlation_matrix[i, j]
                if abs(corr_value) > 0.7:
                    significant_correlations.append((pairs[i], pairs[j], corr_value))
                    
        # Cluster correlated pairs using PCA
        correlation_clusters = self._cluster_correlated_pairs(correlation_matrix, pairs)
        
        # Generate risk warnings
        risk_warnings = []
        
        # Check for high positive correlations (concentration risk)
        high_positive_corrs = [c for c in significant_correlations if c[2] > 0.8]
        if high_positive_corrs:
            risk_warnings.append(f"High positive correlation detected: {len(high_positive_corrs)} pairs")
            
        # Check for CAD concentration
        cad_pairs = [p for p in pairs if 'CAD' in p]
        if len(cad_pairs) > len(pairs) * 0.5:
            risk_warnings.append("High CAD concentration - consider diversification")
            
        return CorrelationMatrix(
            pairs=pairs,
            correlation_matrix=correlation_matrix,
            significant_correlations=significant_correlations,
            correlation_clusters=correlation_clusters,
            risk_warnings=risk_warnings
        )
        
    def _cluster_correlated_pairs(self, 
                                correlation_matrix: np.ndarray,
                                pairs: List[str]) -> Dict[str, List[str]]:
        """Cluster pairs based on correlation using PCA"""
        try:
            # Standardize the correlation matrix
            scaler = StandardScaler()
            scaled_corr = scaler.fit_transform(correlation_matrix)
            
            # Apply PCA
            pca = PCA(n_components=min(3, len(pairs)))
            components = pca.fit_transform(scaled_corr)
            
            # Simple clustering based on first principal component
            clusters = defaultdict(list)
            
            for i, pair in enumerate(pairs):
                if components[i, 0] > 0.5:
                    clusters['cluster_1'].append(pair)
                elif components[i, 0] < -0.5:
                    clusters['cluster_2'].append(pair)
                else:
                    clusters['cluster_neutral'].append(pair)
                    
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {'all': pairs}
            
    def analyze_optimal_trading_times(self,
                                    pair: str,
                                    historical_data: pd.DataFrame,
                                    spread_data: Optional[pd.DataFrame] = None) -> OptimalTradingTimes:
        """
        Analyze optimal trading times based on volatility, spread, and liquidity
        
        Args:
            pair: Forex pair to analyze
            historical_data: DataFrame with OHLC data and timestamps
            spread_data: Optional spread data
            
        Returns:
            Optimal trading time analysis
        """
        # Calculate hourly statistics
        historical_data['hour'] = pd.to_datetime(historical_data['timestamp']).dt.hour
        historical_data['returns'] = historical_data['close'].pct_change()
        
        # Volatility by hour
        volatility_by_hour = {}
        spread_by_hour = {}
        liquidity_by_hour = {}
        
        for hour in range(24):
            hour_data = historical_data[historical_data['hour'] == hour]
            
            if len(hour_data) > 10:
                # Calculate volatility (standard deviation of returns)
                volatility = hour_data['returns'].std() * np.sqrt(252 * 24)  # Annualized
                volatility_by_hour[hour] = volatility
                
                # Calculate average volume as liquidity proxy
                if 'volume' in hour_data.columns:
                    liquidity_by_hour[hour] = hour_data['volume'].mean()
                else:
                    # Use high-low range as liquidity proxy
                    liquidity_by_hour[hour] = (hour_data['high'] - hour_data['low']).mean()
                    
                # Calculate average spread if available
                if spread_data is not None and 'spread' in spread_data.columns:
                    hour_spread = spread_data[spread_data['hour'] == hour]['spread'].mean()
                    spread_by_hour[hour] = hour_spread
                    
        # Determine best hours (high volatility, low spread)
        scores = {}
        for hour in volatility_by_hour:
            vol_score = volatility_by_hour.get(hour, 0)
            spread_penalty = spread_by_hour.get(hour, 0) if spread_by_hour else 0
            liquidity_score = liquidity_by_hour.get(hour, 1)
            
            # Combined score: high volatility, low spread, high liquidity
            scores[hour] = (vol_score * liquidity_score) / (1 + spread_penalty)
            
        # Get top 5 best hours
        best_hours = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:5]
        
        # Determine best sessions
        session_scores = defaultdict(float)
        for session, times in self.TRADING_SESSIONS.items():
            start, end = times['start'], times['end']
            
            # Handle overnight sessions
            if start > end:
                session_hours = list(range(start, 24)) + list(range(0, end))
            else:
                session_hours = list(range(start, end))
                
            # Average score for session
            session_score = np.mean([scores.get(h, 0) for h in session_hours if h in scores])
            session_scores[session] = session_score
            
        best_sessions = sorted(session_scores.keys(), 
                             key=lambda x: session_scores[x], 
                             reverse=True)[:2]
                             
        return OptimalTradingTimes(
            pair=pair,
            best_hours=best_hours,
            best_sessions=best_sessions,
            volatility_by_hour=volatility_by_hour,
            spread_by_hour=spread_by_hour,
            liquidity_score_by_hour=liquidity_by_hour
        )
        
    def detect_forex_patterns(self,
                            price_data: pd.DataFrame,
                            min_confidence: float = 0.7) -> List[ForexPattern]:
        """
        Detect common forex patterns (triangles, channels, head & shoulders)
        
        Args:
            price_data: DataFrame with OHLC data
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Ensure we have enough data
        if len(price_data) < 50:
            return patterns
            
        highs = price_data['high'].values
        lows = price_data['low'].values
        closes = price_data['close'].values
        
        # Detect support and resistance levels
        support_levels = self._find_support_resistance(lows, is_support=True)
        resistance_levels = self._find_support_resistance(highs, is_support=False)
        
        # Pattern detection
        patterns.extend(self._detect_triangle_patterns(highs, lows, closes))
        patterns.extend(self._detect_channel_patterns(highs, lows, closes))
        patterns.extend(self._detect_double_tops_bottoms(highs, lows, closes))
        
        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        return patterns
        
    def _find_support_resistance(self, 
                               prices: np.ndarray,
                               is_support: bool,
                               window: int = 20) -> List[float]:
        """Find support/resistance levels using local extrema"""
        levels = []
        
        for i in range(window, len(prices) - window):
            if is_support:
                # Check if it's a local minimum
                if prices[i] == min(prices[i-window:i+window+1]):
                    levels.append(prices[i])
            else:
                # Check if it's a local maximum
                if prices[i] == max(prices[i-window:i+window+1]):
                    levels.append(prices[i])
                    
        # Cluster nearby levels
        if levels:
            levels = sorted(levels)
            clustered_levels = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if level - current_cluster[-1] < (max(prices) - min(prices)) * 0.01:
                    current_cluster.append(level)
                else:
                    clustered_levels.append(np.mean(current_cluster))
                    current_cluster = [level]
                    
            clustered_levels.append(np.mean(current_cluster))
            return clustered_levels
            
        return []
        
    def _detect_triangle_patterns(self,
                                highs: np.ndarray,
                                lows: np.ndarray,
                                closes: np.ndarray) -> List[ForexPattern]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        if len(highs) < 40:
            return patterns
            
        # Look for converging trendlines
        window = 20
        
        for i in range(window, len(highs) - 10):
            # Get recent highs and lows
            recent_highs = highs[i-window:i]
            recent_lows = lows[i-window:i]
            
            # Fit trendlines
            x = np.arange(len(recent_highs))
            
            try:
                # Upper trendline
                upper_slope, upper_intercept = np.polyfit(x, recent_highs, 1)
                # Lower trendline
                lower_slope, lower_intercept = np.polyfit(x, recent_lows, 1)
                
                # Check for triangle patterns
                if abs(upper_slope) < 0.0001 and lower_slope > 0.0001:
                    # Ascending triangle
                    pattern_type = "ascending_triangle"
                    confidence = 0.8
                elif upper_slope < -0.0001 and abs(lower_slope) < 0.0001:
                    # Descending triangle
                    pattern_type = "descending_triangle"
                    confidence = 0.8
                elif upper_slope < -0.0001 and lower_slope > 0.0001:
                    # Symmetrical triangle
                    pattern_type = "symmetrical_triangle"
                    confidence = 0.85
                else:
                    continue
                    
                # Calculate breakout levels
                current_price = closes[i]
                
                if pattern_type == "ascending_triangle":
                    entry_price = upper_intercept + upper_slope * window * 1.01
                    stop_loss = lower_intercept + lower_slope * window * 0.98
                    take_profit = entry_price + (entry_price - stop_loss) * 2
                elif pattern_type == "descending_triangle":
                    entry_price = lower_intercept + lower_slope * window * 0.99
                    stop_loss = upper_intercept + upper_slope * window * 1.02
                    take_profit = entry_price - (stop_loss - entry_price) * 2
                else:
                    # Symmetrical - wait for breakout direction
                    continue
                    
                risk_reward = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
                
                patterns.append(ForexPattern(
                    pattern_name=pattern_type,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=risk_reward,
                    pattern_strength=confidence
                ))
                
            except:
                continue
                
        return patterns
        
    def _detect_channel_patterns(self,
                               highs: np.ndarray,
                               lows: np.ndarray,
                               closes: np.ndarray) -> List[ForexPattern]:
        """Detect channel patterns"""
        patterns = []
        
        if len(highs) < 30:
            return patterns
            
        # Look for parallel trendlines
        window = 20
        
        for i in range(window, len(highs) - 10):
            recent_highs = highs[i-window:i]
            recent_lows = lows[i-window:i]
            
            x = np.arange(len(recent_highs))
            
            try:
                # Fit trendlines
                upper_slope, upper_intercept = np.polyfit(x, recent_highs, 1)
                lower_slope, lower_intercept = np.polyfit(x, recent_lows, 1)
                
                # Check if slopes are similar (parallel)
                slope_diff = abs(upper_slope - lower_slope)
                avg_slope = (upper_slope + lower_slope) / 2
                
                if slope_diff < abs(avg_slope) * 0.2:  # Within 20% of each other
                    # We have a channel
                    current_price = closes[i]
                    channel_width = np.mean(recent_highs - recent_lows)
                    
                    # Determine position in channel
                    upper_line = upper_intercept + upper_slope * window
                    lower_line = lower_intercept + lower_slope * window
                    position_in_channel = (current_price - lower_line) / (upper_line - lower_line)
                    
                    if position_in_channel < 0.3:
                        # Near bottom - buy signal
                        entry_price = current_price
                        stop_loss = lower_line * 0.98
                        take_profit = upper_line * 0.98
                        pattern_name = "channel_bottom"
                    elif position_in_channel > 0.7:
                        # Near top - sell signal
                        entry_price = current_price
                        stop_loss = upper_line * 1.02
                        take_profit = lower_line * 1.02
                        pattern_name = "channel_top"
                    else:
                        continue
                        
                    risk_reward = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
                    
                    patterns.append(ForexPattern(
                        pattern_name=pattern_name,
                        confidence=0.75,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward,
                        pattern_strength=0.75
                    ))
                    
            except:
                continue
                
        return patterns
        
    def _detect_double_tops_bottoms(self,
                                  highs: np.ndarray,
                                  lows: np.ndarray,
                                  closes: np.ndarray) -> List[ForexPattern]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        if len(highs) < 40:
            return patterns
            
        # Parameters
        min_distance = 10  # Minimum bars between peaks/troughs
        tolerance = 0.002  # 0.2% tolerance for similar levels
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(5, len(highs) - 5):
            # Check for peak
            if highs[i] == max(highs[i-5:i+6]):
                peaks.append((i, highs[i]))
                
            # Check for trough
            if lows[i] == min(lows[i-5:i+6]):
                troughs.append((i, lows[i]))
                
        # Look for double tops
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                idx1, price1 = peaks[i]
                idx2, price2 = peaks[j]
                
                if idx2 - idx1 < min_distance:
                    continue
                    
                # Check if peaks are similar height
                if abs(price1 - price2) / price1 < tolerance:
                    # Found double top
                    neckline = min(lows[idx1:idx2+1])
                    
                    pattern = ForexPattern(
                        pattern_name="double_top",
                        confidence=0.8,
                        entry_price=neckline * 0.99,
                        stop_loss=max(price1, price2) * 1.01,
                        take_profit=neckline - (max(price1, price2) - neckline),
                        risk_reward_ratio=1.5,
                        pattern_strength=0.8
                    )
                    patterns.append(pattern)
                    
        # Look for double bottoms
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                idx1, price1 = troughs[i]
                idx2, price2 = troughs[j]
                
                if idx2 - idx1 < min_distance:
                    continue
                    
                # Check if troughs are similar depth
                if abs(price1 - price2) / price1 < tolerance:
                    # Found double bottom
                    neckline = max(highs[idx1:idx2+1])
                    
                    pattern = ForexPattern(
                        pattern_name="double_bottom",
                        confidence=0.8,
                        entry_price=neckline * 1.01,
                        stop_loss=min(price1, price2) * 0.99,
                        take_profit=neckline + (neckline - min(price1, price2)),
                        risk_reward_ratio=1.5,
                        pattern_strength=0.8
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def calculate_pip_value(self,
                          pair: str,
                          position_size: int,
                          account_currency: str = 'CAD') -> float:
        """
        Calculate pip value for a forex position
        
        Args:
            pair: Forex pair (e.g., 'USD_CAD')
            position_size: Position size in units
            account_currency: Account currency
            
        Returns:
            Pip value in account currency
        """
        base_currency, quote_currency = pair.split('_')
        
        # Standard pip value calculation
        if 'JPY' in pair:
            pip_size = 0.01  # For JPY pairs
        else:
            pip_size = 0.0001  # For most other pairs
            
        # Calculate pip value in quote currency
        pip_value_quote = position_size * pip_size
        
        # Convert to account currency if needed
        if quote_currency == account_currency:
            return pip_value_quote
        elif base_currency == account_currency:
            # Need current price to convert
            # This is a simplified calculation
            return pip_value_quote
        else:
            # Would need exchange rate for full conversion
            # Simplified for demonstration
            return pip_value_quote
            
    def analyze_economic_calendar_impact(self,
                                       pair: str,
                                       economic_events: List[Dict]) -> Dict[str, Any]:
        """
        Analyze potential impact of economic events on forex pair
        
        Args:
            pair: Forex pair
            economic_events: List of upcoming economic events
            
        Returns:
            Impact analysis and recommendations
        """
        base_currency, quote_currency = pair.split('_')
        
        high_impact_events = []
        medium_impact_events = []
        trading_recommendations = []
        
        for event in economic_events:
            # Check if event affects either currency
            if event.get('currency') in [base_currency, quote_currency]:
                impact_level = event.get('impact', 'low')
                
                if impact_level == 'high':
                    high_impact_events.append(event)
                elif impact_level == 'medium':
                    medium_impact_events.append(event)
                    
        # Generate recommendations
        if high_impact_events:
            trading_recommendations.append({
                'type': 'warning',
                'message': f"{len(high_impact_events)} high-impact events in next 24h",
                'action': 'Consider reducing position size or using wider stops'
            })
            
            # Check for specific high-impact events
            for event in high_impact_events:
                if 'interest rate' in event.get('name', '').lower():
                    trading_recommendations.append({
                        'type': 'critical',
                        'message': f"Interest rate decision for {event['currency']}",
                        'action': 'Avoid new positions until after announcement'
                    })
                elif 'nfp' in event.get('name', '').lower() or 'employment' in event.get('name', '').lower():
                    trading_recommendations.append({
                        'type': 'warning',
                        'message': f"Employment data release for {event['currency']}",
                        'action': 'Expect increased volatility, consider hedging'
                    })
                    
        return {
            'pair': pair,
            'high_impact_events': high_impact_events,
            'medium_impact_events': medium_impact_events,
            'total_events': len(high_impact_events) + len(medium_impact_events),
            'risk_level': 'high' if high_impact_events else 'medium' if medium_impact_events else 'low',
            'recommendations': trading_recommendations,
            'suggested_position_adjustment': 0.5 if high_impact_events else 0.8 if medium_impact_events else 1.0
        }
        
    def calculate_carry_trade_opportunity(self,
                                        interest_rates: Dict[str, float],
                                        pairs: List[str]) -> List[Dict]:
        """
        Calculate carry trade opportunities based on interest rate differentials
        
        Args:
            interest_rates: Interest rates by currency
            pairs: List of forex pairs to analyze
            
        Returns:
            Ranked carry trade opportunities
        """
        carry_trades = []
        
        for pair in pairs:
            base_currency, quote_currency = pair.split('_')
            
            if base_currency in interest_rates and quote_currency in interest_rates:
                base_rate = interest_rates[base_currency]
                quote_rate = interest_rates[quote_currency]
                
                # Interest rate differential (positive means base currency has higher rate)
                rate_differential = base_rate - quote_rate
                
                # Annual carry (simplified)
                annual_carry = rate_differential
                
                # Determine direction
                if rate_differential > 0:
                    direction = 'long'  # Buy high-yielding currency
                else:
                    direction = 'short'  # Sell low-yielding currency
                    annual_carry = abs(annual_carry)
                    
                carry_trades.append({
                    'pair': pair,
                    'direction': direction,
                    'base_rate': base_rate,
                    'quote_rate': quote_rate,
                    'rate_differential': abs(rate_differential),
                    'annual_carry_percent': annual_carry,
                    'monthly_carry_percent': annual_carry / 12,
                    'risk_warning': 'High' if abs(rate_differential) > 3 else 'Medium' if abs(rate_differential) > 1 else 'Low'
                })
                
        # Sort by carry potential
        carry_trades.sort(key=lambda x: x['annual_carry_percent'], reverse=True)
        
        return carry_trades
        
    def generate_forex_trading_signals(self,
                                     price_data: pd.DataFrame,
                                     currency_strength: Dict[str, CurrencyStrength],
                                     patterns: List[ForexPattern],
                                     economic_impact: Dict) -> List[Dict]:
        """
        Generate comprehensive forex trading signals combining multiple factors
        
        Args:
            price_data: OHLC price data
            currency_strength: Currency strength analysis
            patterns: Detected technical patterns
            economic_impact: Economic calendar impact
            
        Returns:
            List of trading signals with confidence scores
        """
        signals = []
        
        # Get latest price
        current_price = price_data['close'].iloc[-1]
        
        # Technical analysis signals
        sma_20 = price_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = price_data['close'].rolling(50).mean().iloc[-1]
        rsi = self._calculate_rsi(price_data['close'])
        
        # Base technical signal
        technical_score = 0
        if current_price > sma_20 > sma_50:
            technical_score = 0.7
            technical_bias = 'bullish'
        elif current_price < sma_20 < sma_50:
            technical_score = -0.7
            technical_bias = 'bearish'
        else:
            technical_score = 0
            technical_bias = 'neutral'
            
        # Adjust for RSI
        if rsi > 70:
            technical_score -= 0.2  # Overbought
        elif rsi < 30:
            technical_score += 0.2  # Oversold
            
        # Currency strength adjustment
        pair = price_data.attrs.get('pair', 'UNKNOWN')
        if '_' in pair:
            base, quote = pair.split('_')
            
            base_strength = currency_strength.get(base, None)
            quote_strength = currency_strength.get(quote, None)
            
            if base_strength and quote_strength:
                strength_differential = base_strength.strength_score - quote_strength.strength_score
                strength_score = strength_differential / 10  # Normalize
            else:
                strength_score = 0
        else:
            strength_score = 0
            
        # Pattern-based signals
        pattern_score = 0
        if patterns:
            # Use the most confident pattern
            best_pattern = max(patterns, key=lambda x: x.confidence)
            pattern_score = best_pattern.confidence * (1 if 'bottom' in best_pattern.pattern_name or 'ascending' in best_pattern.pattern_name else -1)
            
        # Economic impact adjustment
        economic_adjustment = 1.0
        if economic_impact.get('risk_level') == 'high':
            economic_adjustment = 0.5  # Reduce confidence during high-impact events
        elif economic_impact.get('risk_level') == 'medium':
            economic_adjustment = 0.8
            
        # Combine all factors
        total_score = (technical_score * 0.4 + strength_score * 0.3 + pattern_score * 0.3) * economic_adjustment
        
        # Generate signal if score exceeds threshold
        if abs(total_score) > 0.5:
            direction = 'buy' if total_score > 0 else 'sell'
            
            # Calculate stops and targets
            atr = self._calculate_atr(price_data)
            
            if direction == 'buy':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
                
            signal = {
                'pair': pair,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': abs(total_score),
                'technical_score': technical_score,
                'strength_score': strength_score,
                'pattern_score': pattern_score,
                'economic_adjustment': economic_adjustment,
                'risk_reward_ratio': abs(take_profit - current_price) / abs(current_price - stop_loss),
                'signal_components': {
                    'technical_bias': technical_bias,
                    'rsi': rsi,
                    'currency_strength_differential': strength_score * 10,
                    'pattern': best_pattern.pattern_name if patterns else None,
                    'economic_risk': economic_impact.get('risk_level', 'unknown')
                }
            }
            
            signals.append(signal)
            
        return signals
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
        
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]


# Example usage
if __name__ == "__main__":
    # Initialize forex utilities
    forex_utils = ForexUtils()
    
    # Example: Calculate currency strength
    mock_price_data = {
        'USD_CAD': [1.35, 1.36, 1.355, 1.358, 1.36],
        'EUR_CAD': [1.45, 1.46, 1.455, 1.458, 1.46],
        'GBP_CAD': [1.70, 1.71, 1.705, 1.708, 1.71]
    }
    
    strength = forex_utils.calculate_currency_strength(mock_price_data, lookback_period=5)
    for currency, analysis in strength.items():
        print(f"{currency}: Strength={analysis.strength_score:.2f}, Trend={analysis.trend}")
        
    # Example: Optimal trading times
    mock_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    optimal_times = forex_utils.analyze_optimal_trading_times('USD_CAD', mock_df)
    print(f"\nOptimal trading hours for USD_CAD: {optimal_times.best_hours}")
    print(f"Best sessions: {optimal_times.best_sessions}")