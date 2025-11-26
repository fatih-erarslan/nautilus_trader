"""Trade attribution system implementation."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from .models import TradeResult, Attribution


class TradeAttributor:
    """Attributes trades to news events and sources."""
    
    def __init__(self, attribution_window: timedelta = timedelta(hours=4)):
        """Initialize the trade attributor.
        
        Args:
            attribution_window: Time window for considering news impact
        """
        self.attribution_window = attribution_window
        self.decay_factor = 0.5  # Exponential decay factor
    
    def attribute_trade(self, trade: TradeResult, news_metadata: Dict[str, Dict]) -> Attribution:
        """Attribute a trade to news sources.
        
        Args:
            trade: The trade to attribute
            news_metadata: Metadata for news events
            
        Returns:
            Attribution object
        """
        source_contributions = {}
        sentiment_weights = {}
        
        if not trade.news_events:
            return Attribution(
                source_contributions={},
                primary_catalyst=None,
                news_weights={},
                confidence_score=0.0,
            )
        
        # Calculate weights for each news event
        for news_id in trade.news_events:
            if news_id not in news_metadata:
                continue
                
            news = news_metadata[news_id]
            
            # Calculate time proximity weight
            time_diff = trade.entry_time - news["published"]
            proximity_weight = self._calculate_proximity_weight(time_diff)
            
            # Calculate sentiment weight
            sentiment_magnitude = abs(news.get("sentiment", 0))
            impact_score = news.get("impact_score", 1.0)
            
            # Combined weight
            weight = proximity_weight * sentiment_magnitude * impact_score
            
            # Aggregate by source
            source = news["source"]
            if source not in source_contributions:
                source_contributions[source] = 0
            source_contributions[source] += weight
            
            sentiment_weights[news_id] = weight
        
        # Normalize contributions
        total_weight = sum(source_contributions.values())
        if total_weight > 0:
            for source in source_contributions:
                source_contributions[source] /= total_weight
        
        # Identify primary catalyst
        primary_catalyst = None
        if sentiment_weights:
            primary_catalyst = max(sentiment_weights, key=sentiment_weights.get)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(sentiment_weights, time_diff)
        
        return Attribution(
            source_contributions=source_contributions,
            primary_catalyst=primary_catalyst,
            news_weights=sentiment_weights,
            confidence_score=confidence_score,
        )
    
    def _calculate_proximity_weight(self, time_diff: timedelta) -> float:
        """Calculate weight based on time proximity.
        
        Args:
            time_diff: Time difference between news and trade
            
        Returns:
            Weight between 0 and 1
        """
        if time_diff < timedelta(0):
            return 0  # News after trade
        
        if time_diff > self.attribution_window:
            return 0  # News too old
        
        # Exponential decay
        hours_diff = time_diff.total_seconds() / 3600
        max_hours = self.attribution_window.total_seconds() / 3600
        
        # Higher weight for more recent news
        normalized_time = hours_diff / max_hours
        weight = np.exp(-normalized_time * 3)  # Decay factor of 3
        
        return weight
    
    def _calculate_confidence(self, weights: Dict[str, float], avg_time_diff: timedelta) -> float:
        """Calculate confidence score for attribution.
        
        Args:
            weights: News event weights
            avg_time_diff: Average time difference
            
        Returns:
            Confidence score between 0 and 1
        """
        if not weights:
            return 0.0
        
        # Factors for confidence
        num_sources = len(weights)
        max_weight = max(weights.values()) if weights else 0
        weight_variance = np.var(list(weights.values())) if len(weights) > 1 else 0
        
        # More sources = higher confidence
        source_factor = min(1.0, num_sources / 3)
        
        # Clear primary catalyst = higher confidence
        dominance_factor = max_weight if max_weight > 0.5 else max_weight * 2
        
        # Less variance = higher confidence
        variance_factor = 1.0 - min(1.0, weight_variance)
        
        confidence = (source_factor + dominance_factor + variance_factor) / 3
        
        return min(1.0, max(0.0, confidence))


class SentimentAccuracyTracker:
    """Tracks accuracy of sentiment predictions."""
    
    def __init__(self):
        """Initialize the accuracy tracker."""
        self.predictions = []
        self.model_predictions = {}  # Model name -> predictions
    
    def record_prediction(
        self,
        news_id: str,
        predicted_sentiment: float,
        predicted_impact: str,
        actual_price_change: float,
        model_name: Optional[str] = None,
    ) -> None:
        """Record a sentiment prediction and its outcome.
        
        Args:
            news_id: News event identifier
            predicted_sentiment: Predicted sentiment score (-1 to 1)
            predicted_impact: Predicted impact (bullish/bearish/neutral)
            actual_price_change: Actual price change percentage
            model_name: Optional model name for comparison
        """
        prediction = {
            "news_id": news_id,
            "predicted_sentiment": predicted_sentiment,
            "predicted_impact": predicted_impact,
            "actual_price_change": actual_price_change,
            "timestamp": datetime.now(),
        }
        
        self.predictions.append(prediction)
        
        if model_name:
            if model_name not in self.model_predictions:
                self.model_predictions[model_name] = []
            self.model_predictions[model_name].append(prediction)
    
    def calculate_accuracy(self) -> Dict[str, float]:
        """Calculate overall accuracy metrics.
        
        Returns:
            Dictionary of accuracy metrics
        """
        if not self.predictions:
            return {
                "direction_accuracy": 0.0,
                "magnitude_mae": 0.0,
                "total_predictions": 0,
                "correct_directions": 0,
            }
        
        correct_directions = 0
        magnitude_errors = []
        
        for pred in self.predictions:
            # Check direction accuracy
            predicted_direction = self._get_direction(pred["predicted_impact"])
            actual_direction = 1 if pred["actual_price_change"] > 0 else -1 if pred["actual_price_change"] < 0 else 0
            
            if predicted_direction == actual_direction:
                correct_directions += 1
            
            # Calculate magnitude error
            predicted_magnitude = abs(pred["predicted_sentiment"]) * 0.05  # Assume 5% max impact
            actual_magnitude = abs(pred["actual_price_change"])
            magnitude_errors.append(abs(predicted_magnitude - actual_magnitude))
        
        return {
            "direction_accuracy": correct_directions / len(self.predictions),
            "magnitude_mae": np.mean(magnitude_errors),
            "total_predictions": len(self.predictions),
            "correct_directions": correct_directions,
            "correlation": self._calculate_correlation(),
        }
    
    def get_accuracy_by_sentiment_range(self) -> Dict[str, Dict[str, float]]:
        """Get accuracy breakdown by sentiment ranges.
        
        Returns:
            Accuracy metrics for each sentiment range
        """
        ranges = {
            "strong_negative": (-1.0, -0.6),
            "moderate_negative": (-0.6, -0.3),
            "neutral": (-0.3, 0.3),
            "moderate_positive": (0.3, 0.6),
            "strong_positive": (0.6, 1.0),
        }
        
        range_accuracies = {}
        
        for range_name, (min_val, max_val) in ranges.items():
            range_predictions = [
                p for p in self.predictions
                if min_val <= p["predicted_sentiment"] <= max_val
            ]
            
            if range_predictions:
                correct = sum(
                    1 for p in range_predictions
                    if self._is_direction_correct(p)
                )
                
                range_accuracies[range_name] = {
                    "accuracy": correct / len(range_predictions),
                    "count": len(range_predictions),
                }
            else:
                range_accuracies[range_name] = {
                    "accuracy": 0.0,
                    "count": 0,
                }
        
        return range_accuracies
    
    def compare_models(self) -> Dict[str, Dict[str, float]]:
        """Compare accuracy between different models.
        
        Returns:
            Accuracy metrics for each model
        """
        model_accuracies = {}
        
        for model_name, predictions in self.model_predictions.items():
            if predictions:
                correct = sum(1 for p in predictions if self._is_direction_correct(p))
                
                model_accuracies[model_name] = {
                    "accuracy": correct / len(predictions),
                    "total_predictions": len(predictions),
                    "mae": self._calculate_mae(predictions),
                }
        
        return model_accuracies
    
    def _get_direction(self, impact: str) -> int:
        """Convert impact string to direction number."""
        if impact == "bullish":
            return 1
        elif impact == "bearish":
            return -1
        else:
            return 0
    
    def _is_direction_correct(self, prediction: Dict) -> bool:
        """Check if direction prediction is correct."""
        predicted = self._get_direction(prediction["predicted_impact"])
        actual = 1 if prediction["actual_price_change"] > 0 else -1 if prediction["actual_price_change"] < 0 else 0
        return predicted == actual
    
    def _calculate_correlation(self) -> float:
        """Calculate correlation between predicted sentiment and actual price change."""
        if len(self.predictions) < 2:
            return 0.0
        
        sentiments = [p["predicted_sentiment"] for p in self.predictions]
        price_changes = [p["actual_price_change"] for p in self.predictions]
        
        return np.corrcoef(sentiments, price_changes)[0, 1]
    
    def _calculate_mae(self, predictions: List[Dict]) -> float:
        """Calculate mean absolute error for predictions."""
        errors = []
        for pred in predictions:
            predicted_magnitude = abs(pred["predicted_sentiment"]) * 0.05
            actual_magnitude = abs(pred["actual_price_change"])
            errors.append(abs(predicted_magnitude - actual_magnitude))
        
        return np.mean(errors) if errors else 0.0