"""Data validation and quality assurance module.

Provides comprehensive data validation, anomaly detection,
and quality metrics for real-time market data.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, Any
import statistics
import logging

from .realtime_feed import DataUpdate

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    price_min: float = 0.01
    price_max: float = 100000.0
    volume_min: float = 0
    volume_max: float = 1e12
    max_price_change_percent: float = 20.0
    max_spread_percent: float = 5.0
    required_fields: List[str] = field(default_factory=lambda: ["symbol", "price", "timestamp"])
    timestamp_tolerance_seconds: int = 60
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 2.5
    collect_metrics: bool = True
    quality_thresholds: Optional['QualityThreshold'] = None


@dataclass
class QualityThreshold:
    """Quality threshold configuration."""
    min_validity_rate: float = 0.95
    max_error_rate: float = 0.05
    evaluation_window: int = 100
    
    def __post_init__(self):
        self._results = deque(maxlen=self.evaluation_window)
        self._alerts = []
    
    def record_result(self, is_valid: bool):
        """Record validation result."""
        self._results.append(is_valid)
    
    def is_quality_acceptable(self) -> bool:
        """Check if quality meets thresholds."""
        if len(self._results) < self.evaluation_window // 2:
            return True  # Not enough data
        
        validity_rate = sum(self._results) / len(self._results)
        error_rate = 1 - validity_rate
        
        if validity_rate < self.min_validity_rate:
            self._alerts.append(f"Validity rate {validity_rate:.2%} below threshold {self.min_validity_rate:.2%}")
            return False
        
        if error_rate > self.max_error_rate:
            self._alerts.append(f"Error rate {error_rate:.2%} above threshold {self.max_error_rate:.2%}")
            return False
        
        return True
    
    def get_alerts(self) -> List[str]:
        """Get quality alerts."""
        return self._alerts.copy()


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, name: str, error_message: str):
        self.name = name
        self.error_message = error_message
    
    def validate(self, update: DataUpdate) -> bool:
        """Validate data update."""
        raise NotImplementedError


class PriceRangeRule(ValidationRule):
    """Validate price is within acceptable range."""
    
    def __init__(self, min_price: float, max_price: float):
        super().__init__("price_range", "Price out of valid range")
        self.min_price = min_price
        self.max_price = max_price
    
    def validate(self, update: DataUpdate) -> bool:
        return self.min_price <= update.price <= self.max_price


class TimestampRule(ValidationRule):
    """Validate timestamp is recent and not in future."""
    
    def __init__(self, tolerance_seconds: int):
        super().__init__("timestamp", "Invalid timestamp")
        self.tolerance_seconds = tolerance_seconds
    
    def validate(self, update: DataUpdate) -> bool:
        current_time = time.time()
        
        # Check if timestamp is in future
        if update.timestamp > current_time + 1:  # Allow 1 second tolerance
            return False
        
        # Check if timestamp is too old
        if update.timestamp < current_time - self.tolerance_seconds:
            return False
        
        return True


class AnomalyDetector:
    """Detect anomalies in data streams using statistical methods."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0, min_samples: int = 20):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_observation(self, symbol: str, price: float, volume: float = 0):
        """Add observation to history."""
        self._price_history[symbol].append(price)
        if volume > 0:
            self._volume_history[symbol].append(volume)
    
    def is_anomaly(self, symbol: str, price: float, volume: float = 0) -> bool:
        """Check if value is anomalous."""
        price_history = self._price_history.get(symbol, [])
        
        if len(price_history) < self.min_samples:
            return False  # Not enough data
        
        # Calculate statistics
        mean = statistics.mean(price_history)
        stdev = statistics.stdev(price_history)
        
        if stdev == 0:
            return abs(price - mean) > mean * 0.1  # 10% threshold if no variation
        
        # Z-score test
        z_score = abs(price - mean) / stdev
        return z_score > self.sensitivity
    
    def get_statistics(self, symbol: str) -> Dict[str, float]:
        """Get statistics for symbol."""
        price_history = self._price_history.get(symbol, [])
        
        if len(price_history) < 2:
            return {}
        
        return {
            "mean": statistics.mean(price_history),
            "stdev": statistics.stdev(price_history),
            "min": min(price_history),
            "max": max(price_history),
            "count": len(price_history)
        }


class DataQualityMetrics:
    """Collect and analyze data quality metrics."""
    
    def __init__(self):
        self._total_validations = 0
        self._valid_count = 0
        self._invalid_count = 0
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._time_series: List[Tuple[float, bool]] = []
        self._symbol_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"valid": 0, "invalid": 0})
    
    def record_validation(self, is_valid: bool, errors: List[str], symbol: str = None):
        """Record validation result."""
        self._total_validations += 1
        
        if is_valid:
            self._valid_count += 1
            if symbol:
                self._symbol_stats[symbol]["valid"] += 1
        else:
            self._invalid_count += 1
            if symbol:
                self._symbol_stats[symbol]["invalid"] += 1
            
            for error in errors:
                self._error_counts[error] += 1
        
        self._time_series.append((time.time(), is_valid))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        validity_rate = self._valid_count / self._total_validations if self._total_validations > 0 else 0
        
        return {
            "total_validations": self._total_validations,
            "valid_count": self._valid_count,
            "invalid_count": self._invalid_count,
            "validity_rate": validity_rate,
            "error_rate": 1 - validity_rate,
            "unique_errors": len(self._error_counts),
        }
    
    def get_error_frequency(self) -> Dict[str, int]:
        """Get error frequency distribution."""
        return dict(self._error_counts)
    
    def get_quality_time_series(self, interval_seconds: int = 60) -> List[Dict[str, Any]]:
        """Get quality metrics over time."""
        if not self._time_series:
            return []
        
        # Group by time intervals
        intervals = []
        current_interval_start = self._time_series[0][0]
        current_valid = 0
        current_total = 0
        
        for timestamp, is_valid in self._time_series:
            if timestamp - current_interval_start > interval_seconds:
                if current_total > 0:
                    intervals.append({
                        "timestamp": current_interval_start,
                        "validity_rate": current_valid / current_total,
                        "count": current_total
                    })
                current_interval_start = timestamp
                current_valid = 0
                current_total = 0
            
            current_total += 1
            if is_valid:
                current_valid += 1
        
        # Add last interval
        if current_total > 0:
            intervals.append({
                "timestamp": current_interval_start,
                "validity_rate": current_valid / current_total,
                "count": current_total
            })
        
        return intervals


class DataValidator:
    """Comprehensive data validator with quality monitoring."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._rules = self._create_default_rules()
        self._custom_rules: Dict[str, Callable] = {}
        self._price_history: Dict[str, float] = {}
        
        if config.enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector(
                sensitivity=config.anomaly_sensitivity
            )
        else:
            self.anomaly_detector = None
        
        if config.collect_metrics:
            self.metrics = DataQualityMetrics()
        else:
            self.metrics = None
        
        self.quality_threshold = config.quality_thresholds
        self._quality_alerts = []
    
    def _create_default_rules(self) -> List[ValidationRule]:
        """Create default validation rules."""
        return [
            PriceRangeRule(self.config.price_min, self.config.price_max),
            TimestampRule(self.config.timestamp_tolerance_seconds),
        ]
    
    def add_custom_rule(self, name: str, rule_func: Callable[[DataUpdate], ValidationResult]):
        """Add custom validation rule."""
        self._custom_rules[name] = rule_func
    
    def validate(self, update: DataUpdate) -> ValidationResult:
        """Validate single data update."""
        errors = []
        warnings = []
        
        # Check required fields
        if not update.symbol:
            errors.append("Missing symbol")
        if update.price is None:
            errors.append("Missing price")
        if update.timestamp is None:
            errors.append("Missing timestamp")
        
        # Apply built-in rules
        for rule in self._rules:
            if not rule.validate(update):
                if rule.name == "timestamp" and update.timestamp < time.time() - self.config.timestamp_tolerance_seconds:
                    errors.append(f"Data is stale (timestamp: {datetime.fromtimestamp(update.timestamp)})")
                else:
                    errors.append(rule.error_message)
        
        # Check price changes
        if update.symbol in self._price_history:
            prev_price = self._price_history[update.symbol]
            if prev_price > 0:
                change_percent = abs(update.price - prev_price) / prev_price * 100
                if change_percent > self.config.max_price_change_percent:
                    errors.append(f"Excessive price change: {change_percent:.1f}%")
        
        # Apply custom rules
        for name, rule_func in self._custom_rules.items():
            result = rule_func(update)
            if not result.is_valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        # Anomaly detection
        if self.anomaly_detector and update.symbol and update.price:
            if self.anomaly_detector.is_anomaly(update.symbol, update.price):
                warnings.append(f"Price anomaly detected for {update.symbol}")
                self._quality_alerts.append(f"Anomaly detected: {update.symbol} at {update.price}")
        
        # Update history
        if update.symbol and update.price:
            self.update_price_history(update.symbol, update.price)
            if self.anomaly_detector:
                self.anomaly_detector.add_observation(update.symbol, update.price)
        
        # Create result
        is_valid = len(errors) == 0
        result = ValidationResult(is_valid, errors, warnings)
        
        # Record metrics
        if self.metrics:
            self.metrics.record_validation(is_valid, errors, update.symbol)
        
        # Check quality thresholds
        if self.quality_threshold:
            self.quality_threshold.record_result(is_valid)
            if not self.quality_threshold.is_quality_acceptable():
                self._quality_alerts.extend(self.quality_threshold.get_alerts())
        
        return result
    
    def validate_batch(self, updates: List[DataUpdate]) -> List[ValidationResult]:
        """Validate batch of updates efficiently."""
        results = []
        for update in updates:
            results.append(self.validate(update))
        return results
    
    def update_price_history(self, symbol: str, price: float):
        """Update price history for symbol."""
        self._price_history[symbol] = price
    
    def get_quality_alerts(self) -> List[str]:
        """Get quality alerts."""
        return self._quality_alerts.copy()
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics."""
        if not self.metrics:
            return {}
        
        stats = self.metrics.get_statistics()
        stats["alerts"] = len(self._quality_alerts)
        
        if self.anomaly_detector:
            stats["anomaly_detection_enabled"] = True
        
        return stats