"""
Data aggregator for combining data from multiple sources
"""
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from dataclasses import dataclass

from ..realtime_manager import DataPoint, AggregatedData

logger = logging.getLogger(__name__)


@dataclass
class AggregationRule:
    """Rules for aggregating data"""
    method: str  # 'weighted_average', 'simple_average', 'median', 'latest', 'best_quality'
    weight_by_source: bool = False
    weight_by_latency: bool = False
    weight_by_volume: bool = False
    max_age_seconds: float = 300.0  # 5 minutes
    min_sources: int = 1
    confidence_threshold: float = 0.5


class DataAggregator:
    """Aggregates data from multiple sources for better accuracy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Source weights (higher = more trusted)
        self.source_weights = self.config.get('source_weights', {
            'finnhub': 1.0,
            'coinbase': 0.9,
            'yahoo': 0.8,
            'alphavantage': 0.7
        })
        
        # Aggregation rules per symbol type
        self.aggregation_rules = self._setup_aggregation_rules()
        
        # Data storage
        self.raw_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_cache: Dict[str, AggregatedData] = {}
        
        # Metrics
        self.aggregations_performed = 0
        self.symbols_aggregated = set()
        self.source_contributions: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.enable_quality_scoring = self.config.get('enable_quality_scoring', True)
        self.enable_outlier_detection = self.config.get('enable_outlier_detection', True)
        self.outlier_threshold_std = self.config.get('outlier_threshold_std', 2.0)
    
    def _setup_aggregation_rules(self) -> Dict[str, AggregationRule]:
        """Setup aggregation rules for different symbol types"""
        rules = {}
        
        # Stock aggregation - weighted average by volume and source quality
        rules['stock'] = AggregationRule(
            method='weighted_average',
            weight_by_source=True,
            weight_by_volume=True,
            weight_by_latency=True,
            max_age_seconds=60.0,
            min_sources=1,
            confidence_threshold=0.6
        )
        
        # Crypto aggregation - simple average due to high volatility
        rules['crypto'] = AggregationRule(
            method='simple_average',
            weight_by_source=True,
            weight_by_latency=True,
            max_age_seconds=30.0,
            min_sources=1,
            confidence_threshold=0.7
        )
        
        # Bond aggregation - weighted by source quality
        rules['bond'] = AggregationRule(
            method='weighted_average',
            weight_by_source=True,
            max_age_seconds=300.0,  # Bonds update less frequently
            min_sources=1,
            confidence_threshold=0.5
        )
        
        # Treasury yield aggregation - median to avoid outliers
        rules['treasury'] = AggregationRule(
            method='median',
            weight_by_source=False,
            max_age_seconds=600.0,  # Yields update slowly
            min_sources=1,
            confidence_threshold=0.8
        )
        
        # Default rule
        rules['default'] = AggregationRule(
            method='simple_average',
            weight_by_source=True,
            max_age_seconds=120.0,
            min_sources=1,
            confidence_threshold=0.5
        )
        
        return rules
    
    def add_data_point(self, data_point: DataPoint) -> Optional[AggregatedData]:
        """Add data point and return aggregated data if updated"""
        symbol = data_point.symbol
        
        # Store raw data
        self.raw_data[symbol].append(data_point)
        self.source_contributions[data_point.source] += 1
        
        # Determine symbol type for aggregation rules
        symbol_type = self._classify_symbol(symbol)
        
        # Aggregate data
        aggregated = self._aggregate_symbol_data(symbol, symbol_type)
        
        if aggregated:
            self.aggregated_cache[symbol] = aggregated
            self.aggregations_performed += 1
            self.symbols_aggregated.add(symbol)
        
        return aggregated
    
    def get_aggregated_data(self, symbol: str) -> Optional[AggregatedData]:
        """Get latest aggregated data for symbol"""
        return self.aggregated_cache.get(symbol)
    
    def get_all_aggregated_data(self) -> Dict[str, AggregatedData]:
        """Get all aggregated data"""
        return self.aggregated_cache.copy()
    
    def _classify_symbol(self, symbol: str) -> str:
        """Classify symbol type for aggregation rules"""
        symbol_upper = symbol.upper()
        
        # Crypto symbols
        if symbol_upper in ['BTC', 'ETH', 'SOL', 'DOGE', 'MATIC', 'LINK', 'DOT', 'UNI', 'ADA', 'XRP']:
            return 'crypto'
        
        if '-USD' in symbol_upper or '/' in symbol_upper:
            return 'crypto'
        
        # Treasury symbols
        if symbol_upper in ['^TNX', '^FVX', '^TYX', '^IRX', '^TNS', '^TNF', '10Y', '5Y', '30Y', '3M', '2Y', '1Y']:
            return 'treasury'
        
        # Bond ETFs
        if symbol_upper in ['TLT', 'IEF', 'SHY', 'AGG', 'BND', 'HYG', 'LQD']:
            return 'bond'
        
        # Default to stock
        return 'stock'
    
    def _aggregate_symbol_data(self, symbol: str, symbol_type: str) -> Optional[AggregatedData]:
        """Aggregate data for a specific symbol"""
        if symbol not in self.raw_data:
            return None
        
        # Get aggregation rule
        rule = self.aggregation_rules.get(symbol_type, self.aggregation_rules['default'])
        
        # Get recent data points
        recent_data = self._get_recent_data(symbol, rule.max_age_seconds)
        
        if len(recent_data) < rule.min_sources:
            return None
        
        # Remove outliers if enabled
        if self.enable_outlier_detection:
            recent_data = self._remove_outliers(recent_data)
        
        if not recent_data:
            return None
        
        # Aggregate based on method
        if rule.method == 'weighted_average':
            return self._weighted_average_aggregation(symbol, recent_data, rule)
        elif rule.method == 'simple_average':
            return self._simple_average_aggregation(symbol, recent_data, rule)
        elif rule.method == 'median':
            return self._median_aggregation(symbol, recent_data, rule)
        elif rule.method == 'latest':
            return self._latest_aggregation(symbol, recent_data, rule)
        elif rule.method == 'best_quality':
            return self._best_quality_aggregation(symbol, recent_data, rule)
        else:
            return self._simple_average_aggregation(symbol, recent_data, rule)
    
    def _get_recent_data(self, symbol: str, max_age_seconds: float) -> List[DataPoint]:
        """Get recent data points within time window"""
        if symbol not in self.raw_data:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        recent_data = []
        
        # Get data from newest to oldest
        for data_point in reversed(self.raw_data[symbol]):
            if data_point.timestamp >= cutoff_time:
                recent_data.append(data_point)
            else:
                break  # Data is sorted by time, so we can break early
        
        return recent_data
    
    def _remove_outliers(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """Remove outlier data points using statistical methods"""
        if len(data_points) < 3:
            return data_points
        
        prices = [dp.price for dp in data_points]
        
        # Calculate z-scores
        mean_price = statistics.mean(prices)
        std_price = statistics.stdev(prices) if len(prices) > 1 else 0
        
        if std_price == 0:
            return data_points
        
        # Filter outliers
        filtered_data = []
        for data_point in data_points:
            z_score = abs((data_point.price - mean_price) / std_price)
            if z_score <= self.outlier_threshold_std:
                filtered_data.append(data_point)
        
        # Return original data if too many outliers removed
        if len(filtered_data) < len(data_points) * 0.5:
            return data_points
        
        return filtered_data
    
    def _weighted_average_aggregation(self, symbol: str, data_points: List[DataPoint], rule: AggregationRule) -> AggregatedData:
        """Aggregate using weighted average"""
        total_weight = 0
        weighted_price_sum = 0
        total_volume = 0
        weighted_bid_sum = 0
        weighted_ask_sum = 0
        bid_weight_sum = 0
        ask_weight_sum = 0
        
        sources = []
        latest_timestamp = None
        
        for data_point in data_points:
            weight = self._calculate_weight(data_point, rule)
            
            weighted_price_sum += data_point.price * weight
            total_weight += weight
            total_volume += data_point.volume or 0
            
            # Handle bid/ask
            if data_point.bid is not None:
                weighted_bid_sum += data_point.bid * weight
                bid_weight_sum += weight
            
            if data_point.ask is not None:
                weighted_ask_sum += data_point.ask * weight
                ask_weight_sum += weight
            
            sources.append(data_point.source)
            
            if latest_timestamp is None or data_point.timestamp > latest_timestamp:
                latest_timestamp = data_point.timestamp
        
        if total_weight == 0:
            return None
        
        # Calculate weighted averages
        avg_price = weighted_price_sum / total_weight
        avg_bid = weighted_bid_sum / bid_weight_sum if bid_weight_sum > 0 else None
        avg_ask = weighted_ask_sum / ask_weight_sum if ask_weight_sum > 0 else None
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(data_points, rule)
        
        return AggregatedData(
            symbol=symbol,
            timestamp=latest_timestamp or datetime.utcnow(),
            price=avg_price,
            volume=total_volume,
            bid=avg_bid,
            ask=avg_ask,
            sources=list(set(sources)),
            spread=avg_ask - avg_bid if avg_ask and avg_bid else None,
            confidence=confidence
        )
    
    def _simple_average_aggregation(self, symbol: str, data_points: List[DataPoint], rule: AggregationRule) -> AggregatedData:
        """Aggregate using simple average"""
        prices = [dp.price for dp in data_points]
        volumes = [dp.volume or 0 for dp in data_points]
        bids = [dp.bid for dp in data_points if dp.bid is not None]
        asks = [dp.ask for dp in data_points if dp.ask is not None]
        sources = [dp.source for dp in data_points]
        
        avg_price = statistics.mean(prices)
        total_volume = sum(volumes)
        avg_bid = statistics.mean(bids) if bids else None
        avg_ask = statistics.mean(asks) if asks else None
        
        latest_timestamp = max(dp.timestamp for dp in data_points)
        confidence = self._calculate_confidence(data_points, rule)
        
        return AggregatedData(
            symbol=symbol,
            timestamp=latest_timestamp,
            price=avg_price,
            volume=total_volume,
            bid=avg_bid,
            ask=avg_ask,
            sources=list(set(sources)),
            spread=avg_ask - avg_bid if avg_ask and avg_bid else None,
            confidence=confidence
        )
    
    def _median_aggregation(self, symbol: str, data_points: List[DataPoint], rule: AggregationRule) -> AggregatedData:
        """Aggregate using median (robust to outliers)"""
        prices = [dp.price for dp in data_points]
        volumes = [dp.volume or 0 for dp in data_points]
        bids = [dp.bid for dp in data_points if dp.bid is not None]
        asks = [dp.ask for dp in data_points if dp.ask is not None]
        sources = [dp.source for dp in data_points]
        
        median_price = statistics.median(prices)
        total_volume = sum(volumes)
        median_bid = statistics.median(bids) if bids else None
        median_ask = statistics.median(asks) if asks else None
        
        latest_timestamp = max(dp.timestamp for dp in data_points)
        confidence = self._calculate_confidence(data_points, rule)
        
        return AggregatedData(
            symbol=symbol,
            timestamp=latest_timestamp,
            price=median_price,
            volume=total_volume,
            bid=median_bid,
            ask=median_ask,
            sources=list(set(sources)),
            spread=median_ask - median_bid if median_ask and median_bid else None,
            confidence=confidence
        )
    
    def _latest_aggregation(self, symbol: str, data_points: List[DataPoint], rule: AggregationRule) -> AggregatedData:
        """Use latest data point (most recent timestamp)"""
        latest_data = max(data_points, key=lambda dp: dp.timestamp)
        
        confidence = self._calculate_confidence([latest_data], rule)
        
        return AggregatedData(
            symbol=symbol,
            timestamp=latest_data.timestamp,
            price=latest_data.price,
            volume=latest_data.volume or 0,
            bid=latest_data.bid,
            ask=latest_data.ask,
            sources=[latest_data.source],
            spread=latest_data.ask - latest_data.bid if latest_data.ask and latest_data.bid else None,
            confidence=confidence
        )
    
    def _best_quality_aggregation(self, symbol: str, data_points: List[DataPoint], rule: AggregationRule) -> AggregatedData:
        """Use data from highest quality source"""
        # Score data points by quality
        scored_data = []
        for data_point in data_points:
            quality_score = self._calculate_quality_score(data_point)
            scored_data.append((quality_score, data_point))
        
        # Use highest quality data point
        best_data = max(scored_data, key=lambda x: x[0])[1]
        
        confidence = self._calculate_confidence([best_data], rule)
        
        return AggregatedData(
            symbol=symbol,
            timestamp=best_data.timestamp,
            price=best_data.price,
            volume=best_data.volume or 0,
            bid=best_data.bid,
            ask=best_data.ask,
            sources=[best_data.source],
            spread=best_data.ask - best_data.bid if best_data.ask and best_data.bid else None,
            confidence=confidence
        )
    
    def _calculate_weight(self, data_point: DataPoint, rule: AggregationRule) -> float:
        """Calculate weight for data point based on rules"""
        weight = 1.0
        
        # Weight by source quality
        if rule.weight_by_source:
            source_weight = self.source_weights.get(data_point.source, 0.5)
            weight *= source_weight
        
        # Weight by latency (lower latency = higher weight)
        if rule.weight_by_latency and data_point.latency_ms is not None:
            # Normalize latency to 0-1 scale (100ms = 0.5 weight)
            latency_weight = max(0.1, 1.0 - (data_point.latency_ms / 200.0))
            weight *= latency_weight
        
        # Weight by volume (higher volume = higher weight)
        if rule.weight_by_volume and data_point.volume is not None and data_point.volume > 0:
            # Logarithmic scaling for volume weight
            import math
            volume_weight = min(2.0, 1.0 + math.log10(data_point.volume) / 10.0)
            weight *= volume_weight
        
        return max(0.01, weight)  # Minimum weight
    
    def _calculate_quality_score(self, data_point: DataPoint) -> float:
        """Calculate quality score for data point"""
        if not self.enable_quality_scoring:
            return 1.0
        
        score = 0.0
        
        # Source quality (0-40 points)
        source_quality = self.source_weights.get(data_point.source, 0.5)
        score += source_quality * 40
        
        # Latency quality (0-30 points)
        if data_point.latency_ms is not None:
            latency_score = max(0, 30 - (data_point.latency_ms / 2))  # 60ms = 0 points
            score += latency_score
        else:
            score += 15  # Average score if no latency data
        
        # Data completeness (0-20 points)
        completeness = 0
        if data_point.bid is not None:
            completeness += 5
        if data_point.ask is not None:
            completeness += 5
        if data_point.volume is not None and data_point.volume > 0:
            completeness += 10
        score += completeness
        
        # Recency (0-10 points)
        age_seconds = (datetime.utcnow() - data_point.timestamp).total_seconds()
        recency_score = max(0, 10 - age_seconds / 6)  # 60 seconds = 0 points
        score += recency_score
        
        return score / 100.0  # Normalize to 0-1
    
    def _calculate_confidence(self, data_points: List[DataPoint], rule: AggregationRule) -> float:
        """Calculate confidence level for aggregated data"""
        confidence = 0.0
        
        # Base confidence from number of sources
        unique_sources = len(set(dp.source for dp in data_points))
        source_confidence = min(1.0, unique_sources / 3.0)  # Max confidence with 3+ sources
        confidence += source_confidence * 0.4
        
        # Confidence from data consistency
        if len(data_points) > 1:
            prices = [dp.price for dp in data_points]
            mean_price = statistics.mean(prices)
            std_dev = statistics.stdev(prices) if len(prices) > 1 else 0
            
            # Lower standard deviation = higher confidence
            consistency_confidence = max(0, 1.0 - (std_dev / mean_price * 10)) if mean_price > 0 else 0
            confidence += consistency_confidence * 0.3
        else:
            confidence += 0.3  # Single source gets average consistency
        
        # Confidence from data quality
        avg_quality = statistics.mean([self._calculate_quality_score(dp) for dp in data_points])
        confidence += avg_quality * 0.3
        
        return min(1.0, confidence)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregation metrics"""
        return {
            'aggregations_performed': self.aggregations_performed,
            'symbols_aggregated': len(self.symbols_aggregated),
            'symbols_tracked': len(self.raw_data),
            'source_contributions': dict(self.source_contributions),
            'cache_size': len(self.aggregated_cache),
            'config': {
                'enable_quality_scoring': self.enable_quality_scoring,
                'enable_outlier_detection': self.enable_outlier_detection,
                'outlier_threshold_std': self.outlier_threshold_std,
                'source_weights': self.source_weights
            }
        }
    
    def clear_cache(self) -> None:
        """Clear aggregated data cache"""
        self.aggregated_cache.clear()
    
    def clear_old_data(self, max_age_hours: float = 24) -> None:
        """Clear old raw data to manage memory"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for symbol in list(self.raw_data.keys()):
            # Filter out old data points
            filtered_data = deque(
                (dp for dp in self.raw_data[symbol] if dp.timestamp >= cutoff_time),
                maxlen=1000
            )
            
            if filtered_data:
                self.raw_data[symbol] = filtered_data
            else:
                del self.raw_data[symbol]