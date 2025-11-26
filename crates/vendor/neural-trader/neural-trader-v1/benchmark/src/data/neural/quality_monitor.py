"""
Neural Data Quality Monitor

This module provides comprehensive data quality monitoring specifically
designed for neural forecasting pipelines, ensuring data meets the
requirements for optimal model performance.

Key Features:
- Multi-dimensional quality assessment
- Real-time monitoring capabilities
- Neural-specific quality metrics
- Alerting and reporting
- Quality trend analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import time
import warnings

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Quality metrics for neural data."""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    NEURAL_READINESS = "neural_readiness"
    STATISTICAL_STABILITY = "statistical_stability"
    TEMPORAL_COHERENCE = "temporal_coherence"


class QualityLevel(Enum):
    """Quality levels."""
    EXCELLENT = "excellent"  # > 95%
    GOOD = "good"           # 85-95%
    ACCEPTABLE = "acceptable"  # 70-85%
    POOR = "poor"           # 50-70%
    CRITICAL = "critical"   # < 50%


@dataclass
class QualityThreshold:
    """Quality thresholds for different metrics."""
    completeness_min: float = 0.95
    consistency_min: float = 0.90
    accuracy_min: float = 0.85
    timeliness_min: float = 0.90
    validity_min: float = 0.95
    uniqueness_min: float = 0.99
    neural_readiness_min: float = 0.80
    statistical_stability_min: float = 0.85
    temporal_coherence_min: float = 0.80


@dataclass
class QualityAlert:
    """Quality alert information."""
    metric: str
    level: QualityLevel
    value: float
    threshold: float
    message: str
    timestamp: float
    data_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    overall_score: float
    metric_scores: Dict[str, float]
    quality_level: QualityLevel
    alerts: List[QualityAlert]
    recommendations: List[str]
    data_profile: Dict[str, Any]
    timestamp: float


class NeuralDataQualityMonitor:
    """
    Comprehensive data quality monitor for neural forecasting pipelines.
    
    This class provides real-time monitoring and assessment of data quality
    specifically tailored for neural time series forecasting models,
    ensuring optimal performance and reliability.
    """
    
    def __init__(self, thresholds: Optional[QualityThreshold] = None):
        self.thresholds = thresholds or QualityThreshold()
        self.logger = logging.getLogger(__name__)
        
        # Quality history
        self.quality_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)
        
        # Monitoring statistics
        self.monitoring_stats = {
            'total_assessments': 0,
            'quality_trend': deque(maxlen=50),
            'metric_trends': {metric.value: deque(maxlen=50) for metric in QualityMetric},
            'alert_counts': {level.value: 0 for level in QualityLevel},
            'last_assessment': None
        }
        
        self.logger.info("NeuralDataQualityMonitor initialized")
    
    async def assess_quality(
        self,
        formatted_data: Dict[str, np.ndarray],
        raw_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Assess overall data quality for neural forecasting.
        
        Args:
            formatted_data: Formatted data from neural preprocessor
            raw_data: Original raw data for additional checks
        
        Returns:
            Dictionary of quality metrics
        """
        try:
            self.logger.info("Assessing data quality")
            
            quality_metrics = {}
            
            # Basic validation
            if not self._validate_input(formatted_data):
                return {'overall_quality': 0.0, 'error': 'Invalid input data'}
            
            # Assess individual metrics
            quality_metrics['completeness'] = self._assess_completeness(formatted_data, raw_data)
            quality_metrics['consistency'] = self._assess_consistency(formatted_data)
            quality_metrics['validity'] = self._assess_validity(formatted_data)
            quality_metrics['neural_readiness'] = self._assess_neural_readiness(formatted_data)
            quality_metrics['statistical_stability'] = self._assess_statistical_stability(formatted_data)
            quality_metrics['temporal_coherence'] = self._assess_temporal_coherence(formatted_data)
            
            if raw_data is not None:
                quality_metrics['timeliness'] = self._assess_timeliness(raw_data)
                quality_metrics['accuracy'] = self._assess_accuracy(raw_data)
                quality_metrics['uniqueness'] = self._assess_uniqueness(raw_data)
            
            # Calculate overall quality score
            quality_metrics['overall_quality'] = self._calculate_overall_quality(quality_metrics)
            
            # Update monitoring statistics
            self._update_monitoring_stats(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing quality: {e}")
            return {'overall_quality': 0.0, 'error': str(e)}
    
    def generate_quality_report(
        self,
        formatted_data: Dict[str, np.ndarray],
        raw_data: Optional[pd.DataFrame] = None
    ) -> QualityReport:
        """Generate comprehensive quality report."""
        try:
            # Assess quality
            quality_metrics = await self.assess_quality(formatted_data, raw_data)
            
            overall_score = quality_metrics.get('overall_quality', 0.0)
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate alerts
            alerts = self._generate_alerts(quality_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(quality_metrics, alerts)
            
            # Create data profile
            data_profile = self._create_data_profile(formatted_data, raw_data)
            
            return QualityReport(
                overall_score=overall_score,
                metric_scores=quality_metrics,
                quality_level=quality_level,
                alerts=alerts,
                recommendations=recommendations,
                data_profile=data_profile,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return QualityReport(
                overall_score=0.0,
                metric_scores={'error': str(e)},
                quality_level=QualityLevel.CRITICAL,
                alerts=[],
                recommendations=["Fix data quality assessment errors"],
                data_profile={},
                timestamp=time.time()
            )
    
    def _validate_input(self, data: Dict[str, np.ndarray]) -> bool:
        """Validate input data format."""
        if not isinstance(data, dict):
            return False
        
        if 'X' not in data or 'y' not in data:
            return False
        
        if not isinstance(data['X'], np.ndarray) or not isinstance(data['y'], np.ndarray):
            return False
        
        if data['X'].ndim != 3 or data['y'].ndim != 2:
            return False
        
        if len(data['X']) != len(data['y']):
            return False
        
        return True
    
    def _assess_completeness(
        self,
        formatted_data: Dict[str, np.ndarray],
        raw_data: Optional[pd.DataFrame]
    ) -> float:
        """Assess data completeness."""
        X = formatted_data['X']
        y = formatted_data['y']
        
        # Count non-NaN values
        x_valid = np.sum(~np.isnan(X))
        y_valid = np.sum(~np.isnan(y))
        
        x_total = X.size
        y_total = y.size
        
        total_valid = x_valid + y_valid
        total_elements = x_total + y_total
        
        completeness = total_valid / total_elements if total_elements > 0 else 0
        
        return completeness
    
    def _assess_consistency(self, formatted_data: Dict[str, np.ndarray]) -> float:
        """Assess data consistency across features and time."""
        X = formatted_data['X']
        
        if X.shape[2] < 2:  # Need at least 2 features
            return 1.0
        
        consistency_scores = []
        
        # Check feature-wise consistency
        for i in range(X.shape[2]):
            for j in range(i + 1, X.shape[2]):
                # Calculate correlation between features
                feature_i = X[:, :, i].flatten()
                feature_j = X[:, :, j].flatten()
                
                # Remove NaN values
                valid_mask = ~(np.isnan(feature_i) | np.isnan(feature_j))
                if np.sum(valid_mask) > 10:
                    correlation = np.corrcoef(feature_i[valid_mask], feature_j[valid_mask])[0, 1]
                    if not np.isnan(correlation):
                        # High correlation indicates consistency
                        consistency_scores.append(abs(correlation))
        
        if consistency_scores:
            # Average correlation as consistency measure
            return np.mean(consistency_scores)
        else:
            return 0.5  # Neutral score if no comparison possible
    
    def _assess_validity(self, formatted_data: Dict[str, np.ndarray]) -> float:
        """Assess data validity (realistic ranges, no infinite values)."""
        X = formatted_data['X']
        y = formatted_data['y']
        
        validity_issues = 0
        total_checks = 0
        
        # Check for infinite values
        total_checks += 1
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            validity_issues += 1
        
        # Check for extremely large values (> 1e10)
        total_checks += 1
        if np.any(np.abs(X) > 1e10) or np.any(np.abs(y) > 1e10):
            validity_issues += 1
        
        # Check for constant sequences (all values the same)
        total_checks += 1
        has_constant = False
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                sequence = X[i, :, j]
                valid_sequence = sequence[~np.isnan(sequence)]
                if len(valid_sequence) > 1 and np.all(valid_sequence == valid_sequence[0]):
                    has_constant = True
                    break
            if has_constant:
                break
        
        if has_constant:
            validity_issues += 1
        
        # Check for reasonable variance
        total_checks += 1
        for i in range(X.shape[2]):
            feature_data = X[:, :, i].flatten()
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 10:
                variance = np.var(valid_data)
                if variance < 1e-10:  # Too little variance
                    validity_issues += 1
                    break
        
        validity_score = 1 - (validity_issues / total_checks)
        return max(0, validity_score)
    
    def _assess_neural_readiness(self, formatted_data: Dict[str, np.ndarray]) -> float:
        """Assess readiness for neural network training."""
        X = formatted_data['X']
        y = formatted_data['y']
        
        readiness_score = 0
        total_checks = 0
        
        # Check data scale (should be normalized)
        total_checks += 1
        x_mean = np.nanmean(X)
        x_std = np.nanstd(X)
        y_mean = np.nanmean(y)
        y_std = np.nanstd(y)
        
        # Good normalization: mean close to 0, std close to 1
        if abs(x_mean) < 0.1 and 0.5 < x_std < 2.0:
            readiness_score += 1
        
        # Check for balanced data distribution
        total_checks += 1
        if len(X) >= 100:  # Need sufficient data
            readiness_score += 1
        
        # Check sequence length adequacy
        total_checks += 1
        if X.shape[1] >= 20:  # Reasonable sequence length
            readiness_score += 1
        
        # Check feature dimension adequacy
        total_checks += 1
        if 1 <= X.shape[2] <= 50:  # Reasonable number of features
            readiness_score += 1
        
        # Check target distribution
        total_checks += 1
        y_flat = y.flatten()
        valid_y = y_flat[~np.isnan(y_flat)]
        if len(valid_y) > 10:
            # Check if targets have reasonable variance
            if np.var(valid_y) > 1e-6:
                readiness_score += 1
        
        return readiness_score / total_checks if total_checks > 0 else 0
    
    def _assess_statistical_stability(self, formatted_data: Dict[str, np.ndarray]) -> float:
        """Assess statistical stability over time."""
        X = formatted_data['X']
        
        if X.shape[0] < 20:  # Need sufficient samples
            return 0.5
        
        stability_scores = []
        
        # Split data into segments and compare statistics
        n_segments = min(5, X.shape[0] // 4)
        segment_size = X.shape[0] // n_segments
        
        for feature_idx in range(X.shape[2]):
            segment_means = []
            segment_stds = []
            
            for seg in range(n_segments):
                start_idx = seg * segment_size
                end_idx = (seg + 1) * segment_size if seg < n_segments - 1 else X.shape[0]
                
                segment_data = X[start_idx:end_idx, :, feature_idx].flatten()
                valid_data = segment_data[~np.isnan(segment_data)]
                
                if len(valid_data) > 5:
                    segment_means.append(np.mean(valid_data))
                    segment_stds.append(np.std(valid_data))
            
            if len(segment_means) >= 2:
                # Calculate coefficient of variation for means and stds
                mean_cv = np.std(segment_means) / (np.mean(segment_means) + 1e-8)
                std_cv = np.std(segment_stds) / (np.mean(segment_stds) + 1e-8)
                
                # Lower CV indicates higher stability
                feature_stability = 1 / (1 + mean_cv + std_cv)
                stability_scores.append(feature_stability)
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _assess_temporal_coherence(self, formatted_data: Dict[str, np.ndarray]) -> float:
        """Assess temporal coherence and smooth transitions."""
        X = formatted_data['X']
        
        coherence_scores = []
        
        for sample_idx in range(min(20, X.shape[0])):  # Sample a few sequences
            for feature_idx in range(X.shape[2]):
                sequence = X[sample_idx, :, feature_idx]
                valid_sequence = sequence[~np.isnan(sequence)]
                
                if len(valid_sequence) > 5:
                    # Calculate differences between consecutive points
                    diffs = np.diff(valid_sequence)
                    
                    # Measure smoothness (lower variance of differences = higher coherence)
                    if len(diffs) > 1:
                        diff_variance = np.var(diffs)
                        sequence_variance = np.var(valid_sequence)
                        
                        # Relative smoothness
                        if sequence_variance > 0:
                            smoothness = 1 / (1 + diff_variance / sequence_variance)
                            coherence_scores.append(smoothness)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _assess_timeliness(self, raw_data: pd.DataFrame) -> float:
        """Assess data timeliness."""
        if not isinstance(raw_data.index, pd.DatetimeIndex):
            return 0.5  # Cannot assess without timestamps
        
        # Check data freshness
        latest_timestamp = raw_data.index.max()
        current_time = pd.Timestamp.now(tz=latest_timestamp.tz)
        
        time_diff = (current_time - latest_timestamp).total_seconds()
        
        # Consider data fresh if less than 1 hour old
        if time_diff < 3600:
            freshness = 1.0
        elif time_diff < 24 * 3600:  # Less than 1 day
            freshness = 0.8
        elif time_diff < 7 * 24 * 3600:  # Less than 1 week
            freshness = 0.6
        else:
            freshness = 0.3
        
        # Check for regular intervals
        time_diffs = raw_data.index.to_series().diff().dropna()
        if len(time_diffs) > 1:
            median_interval = time_diffs.median()
            interval_variance = time_diffs.var()
            
            # Regular intervals have low variance
            regularity = 1 / (1 + interval_variance.total_seconds() / median_interval.total_seconds())
        else:
            regularity = 0.5
        
        return 0.7 * freshness + 0.3 * regularity
    
    def _assess_accuracy(self, raw_data: pd.DataFrame) -> float:
        """Assess data accuracy through consistency checks."""
        accuracy_score = 0
        total_checks = 0
        
        numeric_columns = raw_data.select_dtypes(include=[np.number]).columns
        
        # Check for reasonable ranges
        for col in numeric_columns:
            total_checks += 1
            col_data = raw_data[col].dropna()
            
            if len(col_data) > 10:
                # Check for outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_ratio = len(outliers) / len(col_data)
                
                # Low outlier ratio indicates higher accuracy
                if outlier_ratio < 0.01:
                    accuracy_score += 1
                elif outlier_ratio < 0.05:
                    accuracy_score += 0.5
        
        return accuracy_score / total_checks if total_checks > 0 else 0.5
    
    def _assess_uniqueness(self, raw_data: pd.DataFrame) -> float:
        """Assess data uniqueness (lack of duplicates)."""
        if len(raw_data) == 0:
            return 0
        
        # Check for exact duplicates
        duplicate_rows = raw_data.duplicated().sum()
        uniqueness = 1 - (duplicate_rows / len(raw_data))
        
        return uniqueness
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        # Weights for different metrics
        weights = {
            'completeness': 0.20,
            'consistency': 0.15,
            'validity': 0.20,
            'neural_readiness': 0.20,
            'statistical_stability': 0.10,
            'temporal_coherence': 0.10,
            'timeliness': 0.03,
            'accuracy': 0.02,
            'uniqueness': 0.01
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, score in metrics.items():
            if metric in weights and not np.isnan(score):
                total_score += weights[metric] * score
                total_weight += weights[metric]
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _generate_alerts(self, metrics: Dict[str, float]) -> List[QualityAlert]:
        """Generate quality alerts based on thresholds."""
        alerts = []
        threshold_map = {
            'completeness': self.thresholds.completeness_min,
            'consistency': self.thresholds.consistency_min,
            'validity': self.thresholds.validity_min,
            'neural_readiness': self.thresholds.neural_readiness_min,
            'statistical_stability': self.thresholds.statistical_stability_min,
            'temporal_coherence': self.thresholds.temporal_coherence_min,
            'timeliness': self.thresholds.timeliness_min,
            'accuracy': self.thresholds.accuracy_min,
            'uniqueness': self.thresholds.uniqueness_min
        }
        
        for metric, value in metrics.items():
            if metric in threshold_map and not np.isnan(value):
                threshold = threshold_map[metric]
                
                if value < threshold:
                    # Determine alert level
                    if value < threshold * 0.5:
                        level = QualityLevel.CRITICAL
                    elif value < threshold * 0.7:
                        level = QualityLevel.POOR
                    else:
                        level = QualityLevel.ACCEPTABLE
                    
                    alert = QualityAlert(
                        metric=metric,
                        level=level,
                        value=value,
                        threshold=threshold,
                        message=f"{metric} quality below threshold: {value:.3f} < {threshold:.3f}",
                        timestamp=time.time()
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, float],
        alerts: List[QualityAlert]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        for alert in alerts:
            if alert.metric == 'completeness':
                recommendations.append("Improve data collection to reduce missing values")
            elif alert.metric == 'consistency':
                recommendations.append("Check data sources for consistency issues")
            elif alert.metric == 'validity':
                recommendations.append("Implement data validation rules to catch invalid values")
            elif alert.metric == 'neural_readiness':
                recommendations.append("Apply proper normalization and feature scaling")
            elif alert.metric == 'statistical_stability':
                recommendations.append("Check for regime changes or data drift")
            elif alert.metric == 'temporal_coherence':
                recommendations.append("Investigate data smoothing or filtering options")
            elif alert.metric == 'timeliness':
                recommendations.append("Improve data ingestion frequency and reduce latency")
            elif alert.metric == 'accuracy':
                recommendations.append("Implement outlier detection and correction")
            elif alert.metric == 'uniqueness':
                recommendations.append("Remove duplicate records from data sources")
        
        # Add general recommendations
        overall_score = metrics.get('overall_quality', 0)
        if overall_score < 0.7:
            recommendations.append("Consider comprehensive data quality improvement initiative")
        
        return recommendations
    
    def _create_data_profile(
        self,
        formatted_data: Dict[str, np.ndarray],
        raw_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Create data profile with key statistics."""
        X = formatted_data['X']
        y = formatted_data['y']
        
        profile = {
            'sample_count': len(X),
            'sequence_length': X.shape[1],
            'feature_count': X.shape[2],
            'forecast_horizon': y.shape[1],
            'missing_ratio_X': np.sum(np.isnan(X)) / X.size,
            'missing_ratio_y': np.sum(np.isnan(y)) / y.size,
            'data_range_X': {
                'min': np.nanmin(X),
                'max': np.nanmax(X),
                'mean': np.nanmean(X),
                'std': np.nanstd(X)
            },
            'data_range_y': {
                'min': np.nanmin(y),
                'max': np.nanmax(y),
                'mean': np.nanmean(y),
                'std': np.nanstd(y)
            }
        }
        
        if raw_data is not None:
            profile['raw_data_shape'] = raw_data.shape
            profile['time_range'] = {
                'start': str(raw_data.index.min()),
                'end': str(raw_data.index.max()),
                'span_days': (raw_data.index.max() - raw_data.index.min()).days
            }
        
        return profile
    
    def _update_monitoring_stats(self, metrics: Dict[str, float]):
        """Update monitoring statistics."""
        self.monitoring_stats['total_assessments'] += 1
        self.monitoring_stats['last_assessment'] = time.time()
        
        overall_score = metrics.get('overall_quality', 0)
        self.monitoring_stats['quality_trend'].append(overall_score)
        
        for metric, value in metrics.items():
            if metric in self.monitoring_stats['metric_trends']:
                self.monitoring_stats['metric_trends'][metric].append(value)
    
    def get_quality_trend(self, metric: Optional[str] = None) -> List[float]:
        """Get quality trend over time."""
        if metric is None:
            return list(self.monitoring_stats['quality_trend'])
        elif metric in self.monitoring_stats['metric_trends']:
            return list(self.monitoring_stats['metric_trends'][metric])
        else:
            return []
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = self.monitoring_stats.copy()
        
        # Convert deques to lists for JSON serialization
        stats['quality_trend'] = list(stats['quality_trend'])
        stats['metric_trends'] = {
            metric: list(trend) for metric, trend in stats['metric_trends'].items()
        }
        
        return stats