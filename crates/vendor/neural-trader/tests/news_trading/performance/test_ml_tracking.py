"""Tests for ML model performance tracking."""

import pytest
from datetime import datetime
import numpy as np

from src.news_trading.performance.ml_tracking import (
    MLModelTracker,
    MLModelComparator,
    PredictionRecord,
)


class TestMLModelTracker:
    """Test ML model performance tracking."""

    def test_ml_model_performance(self):
        """Test tracking of ML model performance."""
        tracker = MLModelTracker()

        # Record model predictions
        tracker.record_prediction(
            model_name="finbert_v1",
            prediction_id="pred-123",
            predicted_value=0.75,
            confidence=0.85,
            actual_value=0.70,
            timestamp=datetime.now(),
        )

        # Get model metrics
        metrics = tracker.get_model_metrics("finbert_v1")

        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["prediction_count"] == 1
        assert 0 <= metrics["confidence_calibration"] <= 1

    def test_multiple_predictions(self):
        """Test tracking multiple predictions."""
        tracker = MLModelTracker()

        # Add multiple predictions
        predictions = [
            (0.8, 0.9, 0.75),
            (0.6, 0.7, 0.65),
            (0.3, 0.5, 0.25),
            (0.9, 0.95, 0.92),
        ]

        for i, (predicted, confidence, actual) in enumerate(predictions):
            tracker.record_prediction(
                model_name="test_model",
                prediction_id=f"pred-{i}",
                predicted_value=predicted,
                confidence=confidence,
                actual_value=actual,
                timestamp=datetime.now(),
            )

        metrics = tracker.get_model_metrics("test_model")

        assert metrics["prediction_count"] == 4
        assert metrics["mae"] < 0.1  # Should be close
        assert metrics["average_confidence"] == pytest.approx(0.7625, 0.01)

    def test_confidence_calibration(self):
        """Test confidence calibration calculation."""
        tracker = MLModelTracker()

        # Well-calibrated predictions
        for i in range(20):
            confidence = 0.8
            # 80% should be correct within threshold
            error = 0.05 if i < 16 else 0.25
            
            tracker.record_prediction(
                model_name="calibrated_model",
                prediction_id=f"pred-{i}",
                predicted_value=0.5,
                confidence=confidence,
                actual_value=0.5 + error * (1 if i % 2 == 0 else -1),
                timestamp=datetime.now(),
            )

        metrics = tracker.get_model_metrics("calibrated_model")
        
        # Confidence calibration should be good
        assert metrics["confidence_calibration"] > 0.7

    def test_feature_importance_tracking(self):
        """Test tracking feature importance over time."""
        tracker = MLModelTracker()

        # Record predictions with feature importance
        feature_importance = {
            "sentiment_score": 0.35,
            "volume_ratio": 0.25,
            "price_momentum": 0.20,
            "news_count": 0.15,
            "market_cap": 0.05,
        }

        tracker.record_prediction(
            model_name="xgboost_v1",
            prediction_id="pred-123",
            predicted_value=0.7,
            confidence=0.8,
            actual_value=0.72,
            timestamp=datetime.now(),
            feature_importance=feature_importance,
        )

        metrics = tracker.get_model_metrics("xgboost_v1")
        
        assert "average_feature_importance" in metrics
        assert metrics["average_feature_importance"]["sentiment_score"] == 0.35


class TestMLModelComparator:
    """Test ML model comparison functionality."""

    def test_model_comparison(self):
        """Test comparison between multiple models."""
        comparator = MLModelComparator()

        # Add model results
        comparator.add_model_results(
            "model_a",
            predictions=[0.7, 0.8, 0.6, 0.9],
            actuals=[0.75, 0.85, 0.55, 0.88],
        )
        
        comparator.add_model_results(
            "model_b",
            predictions=[0.6, 0.9, 0.5, 0.95],
            actuals=[0.75, 0.85, 0.55, 0.88],
        )

        comparison = comparator.compare_models()

        assert "model_a" in comparison
        assert "model_b" in comparison
        assert comparison["model_a"]["mae"] < comparison["model_b"]["mae"]

    def test_statistical_significance(self):
        """Test statistical significance of model differences."""
        comparator = MLModelComparator()

        # Model A - consistently good
        np.random.seed(42)
        actuals = np.random.uniform(0, 1, 100)
        predictions_a = actuals + np.random.normal(0, 0.05, 100)  # Small error

        # Model B - higher variance
        predictions_b = actuals + np.random.normal(0, 0.15, 100)  # Larger error

        comparator.add_model_results("model_a", predictions_a.tolist(), actuals.tolist())
        comparator.add_model_results("model_b", predictions_b.tolist(), actuals.tolist())

        comparison = comparator.compare_models(include_significance=True)
        
        assert comparison["statistical_tests"]["model_a_vs_model_b"]["significant"]
        assert comparison["statistical_tests"]["model_a_vs_model_b"]["p_value"] < 0.05

    def test_directional_accuracy(self):
        """Test directional accuracy comparison."""
        comparator = MLModelComparator()

        # Model that gets direction right but magnitude wrong
        comparator.add_model_results(
            "direction_model",
            predictions=[0.7, 0.3, 0.8, 0.2],
            actuals=[0.6, 0.4, 0.9, 0.1],  # Same direction, different magnitude
        )

        # Model that gets magnitude closer but direction sometimes wrong
        comparator.add_model_results(
            "magnitude_model",
            predictions=[0.65, 0.45, 0.45, 0.15],
            actuals=[0.6, 0.4, 0.9, 0.1],  # One wrong direction
        )

        comparison = comparator.compare_models()
        
        assert comparison["direction_model"]["directional_accuracy"] == 1.0
        assert comparison["magnitude_model"]["directional_accuracy"] < 1.0

    def test_model_ensemble_performance(self):
        """Test ensemble model performance tracking."""
        comparator = MLModelComparator()

        # Individual model predictions
        actuals = [0.7, 0.3, 0.8, 0.5, 0.6]
        
        model_predictions = {
            "model_1": [0.65, 0.35, 0.75, 0.55, 0.58],
            "model_2": [0.72, 0.28, 0.82, 0.48, 0.62],
            "model_3": [0.68, 0.32, 0.79, 0.52, 0.59],
        }

        for model_name, predictions in model_predictions.items():
            comparator.add_model_results(model_name, predictions, actuals)

        # Create ensemble predictions (simple average)
        ensemble_predictions = [
            np.mean([model_predictions[m][i] for m in model_predictions])
            for i in range(len(actuals))
        ]

        comparator.add_model_results("ensemble", ensemble_predictions, actuals)

        comparison = comparator.compare_models()
        
        # Ensemble should often perform better than individual models
        assert comparison["ensemble"]["mae"] <= min(
            comparison[m]["mae"] for m in model_predictions
        )