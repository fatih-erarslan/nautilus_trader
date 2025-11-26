"""
Data validator for ensuring data quality and integrity
"""
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from dataclasses import dataclass
from enum import Enum

from ..realtime_manager import DataPoint

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    confidence: float
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ValidationRule:
    """Data validation rule"""
    name: str
    description: str
    level: ValidationLevel
    enabled: bool = True
    threshold: Optional[float] = None
    parameters: Dict[str, Any] = None


class DataValidator:
    """Validates data quality and detects anomalies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Validation level
        self.validation_level = ValidationLevel(
            self.config.get('validation_level', 'standard')
        )
        
        # Validation rules
        self.rules = self._setup_validation_rules()
        
        # Historical data for validation
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.validation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Metrics
        self.validations_performed = 0
        self.validations_passed = 0
        self.validations_failed = 0
        self.issues_found: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.enable_statistical_validation = self.config.get('enable_statistical_validation', True)
        self.enable_cross_validation = self.config.get('enable_cross_validation', True)
        self.enable_trend_analysis = self.config.get('enable_trend_analysis', True)
        
        # Thresholds
        self.max_price_change_percent = self.config.get('max_price_change_percent', 20.0)
        self.max_spread_percent = self.config.get('max_spread_percent', 5.0)
        self.min_volume_threshold = self.config.get('min_volume_threshold', 0)
        self.max_latency_ms = self.config.get('max_latency_ms', 5000)
    
    def _setup_validation_rules(self) -> List[ValidationRule]:
        """Setup validation rules based on configuration"""
        rules = []
        
        # Basic validation rules
        rules.append(ValidationRule(
            name="price_positive",
            description="Price must be positive",
            level=ValidationLevel.BASIC,
            enabled=True
        ))
        
        rules.append(ValidationRule(
            name="volume_non_negative",
            description="Volume must be non-negative",
            level=ValidationLevel.BASIC,
            enabled=True
        ))
        
        rules.append(ValidationRule(
            name="timestamp_recent",
            description="Timestamp must be recent",
            level=ValidationLevel.BASIC,
            enabled=True,
            threshold=3600.0  # 1 hour
        ))
        
        # Standard validation rules
        rules.append(ValidationRule(
            name="price_change_reasonable",
            description="Price change must be within reasonable bounds",
            level=ValidationLevel.STANDARD,
            enabled=True,
            threshold=self.max_price_change_percent
        ))
        
        rules.append(ValidationRule(
            name="bid_ask_spread_reasonable",
            description="Bid-ask spread must be reasonable",
            level=ValidationLevel.STANDARD,
            enabled=True,
            threshold=self.max_spread_percent
        ))
        
        rules.append(ValidationRule(
            name="latency_acceptable",
            description="Data latency must be acceptable",
            level=ValidationLevel.STANDARD,
            enabled=True,
            threshold=self.max_latency_ms
        ))
        
        # Strict validation rules
        rules.append(ValidationRule(
            name="volume_pattern_consistent",
            description="Volume pattern must be consistent with history",
            level=ValidationLevel.STRICT,
            enabled=self.validation_level == ValidationLevel.STRICT
        ))
        
        rules.append(ValidationRule(
            name="price_trend_consistent",
            description="Price trend must be consistent",
            level=ValidationLevel.STRICT,
            enabled=self.validation_level == ValidationLevel.STRICT
        ))
        
        rules.append(ValidationRule(
            name="cross_source_consistency",
            description="Data must be consistent across sources",
            level=ValidationLevel.STRICT,
            enabled=self.validation_level == ValidationLevel.STRICT
        ))
        
        return rules
    
    def validate_data_point(self, data_point: DataPoint, 
                          cross_reference_data: List[DataPoint] = None) -> ValidationResult:
        """Validate a single data point"""
        self.validations_performed += 1
        
        issues = []
        warnings = []
        metadata = {}
        
        # Apply validation rules
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Skip rules above current validation level
            if rule.level.value not in [self.validation_level.value, 'basic']:
                if self.validation_level == ValidationLevel.BASIC:
                    continue
                elif self.validation_level == ValidationLevel.STANDARD and rule.level == ValidationLevel.STRICT:
                    continue
            
            rule_result = self._apply_validation_rule(rule, data_point, cross_reference_data)
            
            if rule_result['issues']:
                issues.extend(rule_result['issues'])
                self.issues_found[rule.name] += len(rule_result['issues'])
            
            if rule_result['warnings']:
                warnings.extend(rule_result['warnings'])
            
            metadata[rule.name] = rule_result
        
        # Calculate overall confidence
        confidence = self._calculate_validation_confidence(data_point, issues, warnings)
        
        # Determine if validation passed
        is_valid = len(issues) == 0
        
        if is_valid:
            self.validations_passed += 1
        else:
            self.validations_failed += 1
        
        # Update historical data
        self._update_validation_history(data_point, is_valid, confidence)
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            metadata=metadata
        )
        
        return result
    
    def validate_batch(self, data_points: List[DataPoint]) -> Dict[str, ValidationResult]:
        """Validate a batch of data points"""
        results = {}
        
        # Group by symbol for cross-validation
        symbol_groups = defaultdict(list)
        for dp in data_points:
            symbol_groups[dp.symbol].append(dp)
        
        # Validate each group
        for symbol, symbol_data in symbol_groups.items():
            for data_point in symbol_data:
                # Use other data points for cross-reference
                cross_reference = [dp for dp in symbol_data if dp != data_point]
                result = self.validate_data_point(data_point, cross_reference)
                results[f"{symbol}_{data_point.source}_{data_point.timestamp}"] = result
        
        return results
    
    def _apply_validation_rule(self, rule: ValidationRule, data_point: DataPoint,
                             cross_reference_data: List[DataPoint] = None) -> Dict[str, Any]:
        """Apply a specific validation rule"""
        issues = []
        warnings = []
        details = {}
        
        try:
            if rule.name == "price_positive":
                if data_point.price <= 0:
                    issues.append(f"Price {data_point.price} is not positive")
            
            elif rule.name == "volume_non_negative":
                if data_point.volume is not None and data_point.volume < 0:
                    issues.append(f"Volume {data_point.volume} is negative")
            
            elif rule.name == "timestamp_recent":
                age_seconds = (datetime.utcnow() - data_point.timestamp).total_seconds()
                if age_seconds > rule.threshold:
                    issues.append(f"Data is too old: {age_seconds:.1f} seconds")
                details['age_seconds'] = age_seconds
            
            elif rule.name == "price_change_reasonable":
                price_change_result = self._validate_price_change(data_point, rule.threshold)
                if price_change_result['is_extreme']:
                    issues.append(f"Extreme price change: {price_change_result['change_percent']:.2f}%")
                elif price_change_result['is_unusual']:
                    warnings.append(f"Unusual price change: {price_change_result['change_percent']:.2f}%")
                details.update(price_change_result)
            
            elif rule.name == "bid_ask_spread_reasonable":
                spread_result = self._validate_bid_ask_spread(data_point, rule.threshold)
                if spread_result['is_excessive']:
                    issues.append(f"Excessive spread: {spread_result['spread_percent']:.2f}%")
                elif spread_result['is_high']:
                    warnings.append(f"High spread: {spread_result['spread_percent']:.2f}%")
                details.update(spread_result)
            
            elif rule.name == "latency_acceptable":
                if data_point.latency_ms is not None and data_point.latency_ms > rule.threshold:
                    issues.append(f"High latency: {data_point.latency_ms:.1f}ms")
                details['latency_ms'] = data_point.latency_ms
            
            elif rule.name == "volume_pattern_consistent":
                volume_result = self._validate_volume_pattern(data_point)
                if volume_result['is_anomalous']:
                    warnings.append(f"Anomalous volume pattern: {volume_result['z_score']:.2f} std devs")
                details.update(volume_result)
            
            elif rule.name == "price_trend_consistent":
                trend_result = self._validate_price_trend(data_point)
                if trend_result['is_inconsistent']:
                    warnings.append(f"Inconsistent price trend: {trend_result['trend_break']}")
                details.update(trend_result)
            
            elif rule.name == "cross_source_consistency":
                if cross_reference_data:
                    consistency_result = self._validate_cross_source_consistency(data_point, cross_reference_data)
                    if consistency_result['is_inconsistent']:
                        warnings.append(f"Cross-source inconsistency: {consistency_result['max_deviation']:.2f}%")
                    details.update(consistency_result)
        
        except Exception as e:
            logger.error(f"Error applying validation rule {rule.name}: {e}")
            warnings.append(f"Validation rule {rule.name} failed to execute")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'details': details
        }
    
    def _validate_price_change(self, data_point: DataPoint, threshold: float) -> Dict[str, Any]:
        """Validate price change against historical data"""
        symbol = data_point.symbol
        current_price = data_point.price
        
        if symbol not in self.price_history or len(self.price_history[symbol]) == 0:
            self.price_history[symbol].append(current_price)
            return {'is_extreme': False, 'is_unusual': False, 'change_percent': 0.0}
        
        recent_prices = list(self.price_history[symbol])
        last_price = recent_prices[-1]
        
        if last_price <= 0:
            return {'is_extreme': False, 'is_unusual': False, 'change_percent': 0.0}
        
        change_percent = abs((current_price - last_price) / last_price) * 100
        
        # Update history
        self.price_history[symbol].append(current_price)
        
        return {
            'is_extreme': change_percent > threshold,
            'is_unusual': change_percent > threshold * 0.5,
            'change_percent': change_percent,
            'last_price': last_price,
            'current_price': current_price
        }
    
    def _validate_bid_ask_spread(self, data_point: DataPoint, threshold: float) -> Dict[str, Any]:
        """Validate bid-ask spread"""
        if data_point.bid is None or data_point.ask is None:
            return {'is_excessive': False, 'is_high': False, 'spread_percent': 0.0}
        
        if data_point.bid >= data_point.ask:
            return {
                'is_excessive': True,
                'is_high': True,
                'spread_percent': float('inf'),
                'bid': data_point.bid,
                'ask': data_point.ask,
                'error': 'Bid >= Ask'
            }
        
        spread = data_point.ask - data_point.bid
        mid_price = (data_point.bid + data_point.ask) / 2
        spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        return {
            'is_excessive': spread_percent > threshold,
            'is_high': spread_percent > threshold * 0.5,
            'spread_percent': spread_percent,
            'spread': spread,
            'mid_price': mid_price
        }
    
    def _validate_volume_pattern(self, data_point: DataPoint) -> Dict[str, Any]:
        """Validate volume pattern against historical data"""
        if data_point.volume is None:
            return {'is_anomalous': False, 'z_score': 0.0}
        
        symbol = data_point.symbol
        current_volume = data_point.volume
        
        if symbol not in self.volume_history or len(self.volume_history[symbol]) < 5:
            self.volume_history[symbol].append(current_volume)
            return {'is_anomalous': False, 'z_score': 0.0}
        
        historical_volumes = list(self.volume_history[symbol])
        
        if len(historical_volumes) < 2:
            return {'is_anomalous': False, 'z_score': 0.0}
        
        mean_volume = statistics.mean(historical_volumes)
        std_volume = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0
        
        if std_volume == 0:
            z_score = 0.0
        else:
            z_score = abs((current_volume - mean_volume) / std_volume)
        
        # Update history
        self.volume_history[symbol].append(current_volume)
        
        return {
            'is_anomalous': z_score > 3.0,  # 3 standard deviations
            'z_score': z_score,
            'mean_volume': mean_volume,
            'std_volume': std_volume,
            'current_volume': current_volume
        }
    
    def _validate_price_trend(self, data_point: DataPoint) -> Dict[str, Any]:
        """Validate price trend consistency"""
        symbol = data_point.symbol
        current_price = data_point.price
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 3:
            return {'is_inconsistent': False, 'trend_break': False}
        
        recent_prices = list(self.price_history[symbol])[-5:]  # Last 5 prices
        
        # Calculate trend
        if len(recent_prices) < 3:
            return {'is_inconsistent': False, 'trend_break': False}
        
        # Simple trend detection
        increasing_count = 0
        decreasing_count = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                increasing_count += 1
            elif recent_prices[i] < recent_prices[i-1]:
                decreasing_count += 1
        
        # Determine current trend
        if increasing_count > decreasing_count:
            trend = 'up'
        elif decreasing_count > increasing_count:
            trend = 'down'
        else:
            trend = 'sideways'
        
        # Check if current price breaks trend
        last_price = recent_prices[-1]
        trend_break = False
        
        if trend == 'up' and current_price < last_price * 0.95:  # 5% drop
            trend_break = True
        elif trend == 'down' and current_price > last_price * 1.05:  # 5% jump
            trend_break = True
        
        return {
            'is_inconsistent': trend_break,
            'trend_break': trend_break,
            'trend': trend,
            'last_price': last_price,
            'current_price': current_price
        }
    
    def _validate_cross_source_consistency(self, data_point: DataPoint, 
                                         cross_reference_data: List[DataPoint]) -> Dict[str, Any]:
        """Validate consistency across different sources"""
        if not cross_reference_data:
            return {'is_inconsistent': False, 'max_deviation': 0.0}
        
        current_price = data_point.price
        reference_prices = [dp.price for dp in cross_reference_data]
        
        if not reference_prices:
            return {'is_inconsistent': False, 'max_deviation': 0.0}
        
        # Calculate deviations
        deviations = []
        for ref_price in reference_prices:
            if ref_price > 0:
                deviation = abs((current_price - ref_price) / ref_price) * 100
                deviations.append(deviation)
        
        if not deviations:
            return {'is_inconsistent': False, 'max_deviation': 0.0}
        
        max_deviation = max(deviations)
        avg_deviation = statistics.mean(deviations)
        
        # Consider inconsistent if deviation > 2%
        is_inconsistent = max_deviation > 2.0
        
        return {
            'is_inconsistent': is_inconsistent,
            'max_deviation': max_deviation,
            'avg_deviation': avg_deviation,
            'reference_prices': reference_prices,
            'current_price': current_price
        }
    
    def _calculate_validation_confidence(self, data_point: DataPoint, 
                                       issues: List[str], warnings: List[str]) -> float:
        """Calculate confidence level for validation result"""
        confidence = 1.0
        
        # Reduce confidence for each issue
        confidence -= len(issues) * 0.3
        
        # Reduce confidence for each warning
        confidence -= len(warnings) * 0.1
        
        # Adjust based on data completeness
        completeness_score = 0.0
        if data_point.bid is not None:
            completeness_score += 0.2
        if data_point.ask is not None:
            completeness_score += 0.2
        if data_point.volume is not None:
            completeness_score += 0.3
        if data_point.latency_ms is not None:
            completeness_score += 0.1
        
        confidence = confidence * (0.8 + completeness_score)
        
        # Adjust based on historical consistency
        symbol = data_point.symbol
        if symbol in self.validation_history:
            recent_validations = list(self.validation_history[symbol])[-10:]
            if recent_validations:
                success_rate = sum(1 for v in recent_validations if v['is_valid']) / len(recent_validations)
                confidence = confidence * (0.7 + success_rate * 0.3)
        
        return max(0.0, min(1.0, confidence))
    
    def _update_validation_history(self, data_point: DataPoint, is_valid: bool, confidence: float) -> None:
        """Update validation history for learning"""
        symbol = data_point.symbol
        
        validation_record = {
            'timestamp': data_point.timestamp,
            'is_valid': is_valid,
            'confidence': confidence,
            'price': data_point.price,
            'source': data_point.source
        }
        
        self.validation_history[symbol].append(validation_record)
    
    def get_symbol_validation_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get validation statistics for a specific symbol"""
        if symbol not in self.validation_history:
            return None
        
        history = list(self.validation_history[symbol])
        if not history:
            return None
        
        total_validations = len(history)
        passed_validations = sum(1 for v in history if v['is_valid'])
        avg_confidence = statistics.mean([v['confidence'] for v in history])
        
        return {
            'symbol': symbol,
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'success_rate': passed_validations / total_validations,
            'avg_confidence': avg_confidence,
            'recent_validations': history[-10:]  # Last 10
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        success_rate = self.validations_passed / self.validations_performed if self.validations_performed > 0 else 0
        
        return {
            'validations_performed': self.validations_performed,
            'validations_passed': self.validations_passed,
            'validations_failed': self.validations_failed,
            'success_rate': success_rate,
            'issues_found': dict(self.issues_found),
            'symbols_tracked': len(self.validation_history),
            'validation_level': self.validation_level.value,
            'rules_enabled': len([r for r in self.rules if r.enabled]),
            'config': {
                'enable_statistical_validation': self.enable_statistical_validation,
                'enable_cross_validation': self.enable_cross_validation,
                'enable_trend_analysis': self.enable_trend_analysis,
                'max_price_change_percent': self.max_price_change_percent,
                'max_spread_percent': self.max_spread_percent
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset validation metrics"""
        self.validations_performed = 0
        self.validations_passed = 0
        self.validations_failed = 0
        self.issues_found.clear()
    
    def clear_history(self) -> None:
        """Clear validation history"""
        self.price_history.clear()
        self.volume_history.clear()
        self.validation_history.clear()