"""ML model performance tracking implementation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class PredictionRecord:
    """Record of a single model prediction."""
    
    prediction_id: str
    model_name: str
    predicted_value: float
    actual_value: float
    confidence: float
    timestamp: datetime
    error: float = field(init=False)
    feature_importance: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Calculate error after initialization."""
        self.error = abs(self.predicted_value - self.actual_value)


class MLModelTracker:
    """Tracks ML model predictions and performance."""
    
    def __init__(self):
        """Initialize the ML model tracker."""
        self.predictions: Dict[str, List[PredictionRecord]] = {}
        self.confidence_threshold = 0.1  # Error threshold for confidence calibration
    
    def record_prediction(
        self,
        model_name: str,
        prediction_id: str,
        predicted_value: float,
        confidence: float,
        actual_value: float,
        timestamp: datetime,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a model prediction.
        
        Args:
            model_name: Name of the model
            prediction_id: Unique prediction identifier
            predicted_value: Predicted value
            confidence: Model confidence (0-1)
            actual_value: Actual observed value
            timestamp: Time of prediction
            feature_importance: Optional feature importance scores
        """
        if model_name not in self.predictions:
            self.predictions[model_name] = []
        
        record = PredictionRecord(
            prediction_id=prediction_id,
            model_name=model_name,
            predicted_value=predicted_value,
            actual_value=actual_value,
            confidence=confidence,
            timestamp=timestamp,
            feature_importance=feature_importance,
        )
        
        self.predictions[model_name].append(record)
    
    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Get performance metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of performance metrics
        """
        if model_name not in self.predictions or not self.predictions[model_name]:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "prediction_count": 0,
                "confidence_calibration": 0.0,
                "average_confidence": 0.0,
            }
        
        records = self.predictions[model_name]
        predicted_values = [r.predicted_value for r in records]
        actual_values = [r.actual_value for r in records]
        confidences = [r.confidence for r in records]
        errors = [r.error for r in records]
        
        metrics = {
            "mae": mean_absolute_error(actual_values, predicted_values),
            "rmse": np.sqrt(mean_squared_error(actual_values, predicted_values)),
            "r_squared": r2_score(actual_values, predicted_values) if len(records) > 1 else 0.0,
            "prediction_count": len(records),
            "confidence_calibration": self._calculate_confidence_calibration(confidences, errors),
            "average_confidence": np.mean(confidences),
            "std_error": np.std(errors),
            "max_error": np.max(errors),
            "percentile_90_error": np.percentile(errors, 90),
        }
        
        # Add feature importance if available
        if any(r.feature_importance for r in records):
            metrics["average_feature_importance"] = self._calculate_average_feature_importance(records)
        
        return metrics
    
    def _calculate_confidence_calibration(self, confidences: List[float], errors: List[float]) -> float:
        """Calculate how well confidence scores match actual accuracy.
        
        Args:
            confidences: List of confidence scores
            errors: List of prediction errors
            
        Returns:
            Calibration score (0-1, higher is better)
        """
        if not confidences or not errors:
            return 0.0
        
        # Group predictions by confidence level
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins
        calibration_errors = []
        
        for i in range(len(confidence_bins) - 1):
            bin_min, bin_max = confidence_bins[i], confidence_bins[i + 1]
            
            # Find predictions in this confidence bin
            bin_indices = [
                j for j, conf in enumerate(confidences)
                if bin_min <= conf < bin_max
            ]
            
            if not bin_indices:
                continue
            
            # Expected accuracy based on confidence
            expected_accuracy = np.mean([confidences[j] for j in bin_indices])
            
            # Actual accuracy (predictions within threshold)
            actual_accuracy = sum(
                1 for j in bin_indices
                if errors[j] < self.confidence_threshold
            ) / len(bin_indices)
            
            calibration_errors.append(abs(expected_accuracy - actual_accuracy))
        
        if not calibration_errors:
            return 0.0
        
        # Return calibration score (1 - average calibration error)
        return 1.0 - np.mean(calibration_errors)
    
    def _calculate_average_feature_importance(self, records: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate average feature importance across predictions.
        
        Args:
            records: List of prediction records
            
        Returns:
            Average feature importance scores
        """
        feature_sums = {}
        feature_counts = {}
        
        for record in records:
            if record.feature_importance:
                for feature, importance in record.feature_importance.items():
                    if feature not in feature_sums:
                        feature_sums[feature] = 0.0
                        feature_counts[feature] = 0
                    
                    feature_sums[feature] += importance
                    feature_counts[feature] += 1
        
        return {
            feature: feature_sums[feature] / feature_counts[feature]
            for feature in feature_sums
        }


class MLModelComparator:
    """Compares performance between multiple ML models."""
    
    def __init__(self):
        """Initialize the model comparator."""
        self.model_results: Dict[str, Tuple[List[float], List[float]]] = {}
    
    def add_model_results(
        self,
        model_name: str,
        predictions: List[float],
        actuals: List[float],
    ) -> None:
        """Add model results for comparison.
        
        Args:
            model_name: Name of the model
            predictions: List of predicted values
            actuals: List of actual values
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        self.model_results[model_name] = (predictions, actuals)
    
    def compare_models(self, include_significance: bool = False) -> Dict[str, Dict[str, float]]:
        """Compare all models.
        
        Args:
            include_significance: Whether to include statistical significance tests
            
        Returns:
            Comparison results for all models
        """
        comparison = {}
        
        for model_name, (predictions, actuals) in self.model_results.items():
            comparison[model_name] = self._calculate_model_metrics(predictions, actuals)
        
        if include_significance and len(self.model_results) > 1:
            comparison["statistical_tests"] = self._perform_statistical_tests()
        
        return comparison
    
    def _calculate_model_metrics(
        self,
        predictions: List[float],
        actuals: List[float],
    ) -> Dict[str, float]:
        """Calculate metrics for a single model.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            
        Returns:
            Dictionary of metrics
        """
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        # Calculate errors
        errors = predictions_array - actuals_array
        abs_errors = np.abs(errors)
        
        # Directional accuracy
        pred_direction = np.sign(predictions_array - 0.5)
        actual_direction = np.sign(actuals_array - 0.5)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        return {
            "mae": np.mean(abs_errors),
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "r_squared": r2_score(actuals_array, predictions_array),
            "directional_accuracy": directional_accuracy,
            "max_error": np.max(abs_errors),
            "std_error": np.std(errors),
            "bias": np.mean(errors),  # Systematic over/under prediction
        }
    
    def _perform_statistical_tests(self) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests between models.
        
        Returns:
            Statistical test results
        """
        results = {}
        model_names = list(self.model_results.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                
                preds_a, actuals_a = self.model_results[model_a]
                preds_b, actuals_b = self.model_results[model_b]
                
                # Calculate errors
                errors_a = np.abs(np.array(preds_a) - np.array(actuals_a))
                errors_b = np.abs(np.array(preds_b) - np.array(actuals_b))
                
                # Paired t-test on errors
                t_stat, p_value = stats.ttest_rel(errors_a, errors_b)
                
                results[f"{model_a}_vs_{model_b}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "better_model": model_a if np.mean(errors_a) < np.mean(errors_b) else model_b,
                }
        
        return results