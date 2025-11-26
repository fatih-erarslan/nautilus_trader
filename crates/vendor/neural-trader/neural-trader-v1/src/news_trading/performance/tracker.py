"""Concrete implementation of performance tracking."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os

from .base import PerformanceTracker
from .models import TradeResult, PerformanceMetrics, TradeStatus
from .attribution import TradeAttributor, SentimentAccuracyTracker
from .ml_tracking import MLModelTracker
from .analytics import PerformanceAnalytics, ReportGenerator
from .persistence import PerformanceDatabase


class ConcretePerformanceTracker(PerformanceTracker):
    """Concrete implementation of performance tracking with all features."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the performance tracker.
        
        Args:
            db_path: Optional path to database file
        """
        self.trades: List[TradeResult] = []
        self.analytics = PerformanceAnalytics()
        self.attributor = TradeAttributor()
        self.sentiment_tracker = SentimentAccuracyTracker()
        self.ml_tracker = MLModelTracker()
        self.report_generator = ReportGenerator()
        
        # Initialize database if path provided
        self.db = PerformanceDatabase(db_path) if db_path else None
        
        # Load existing trades from database
        if self.db:
            self._load_from_database()
    
    def record_trade(self, trade_result: TradeResult) -> None:
        """Record a completed trade.
        
        Args:
            trade_result: The trade result to record
        """
        # Calculate P&L if needed
        if trade_result.pnl == 0 and trade_result.exit_price:
            trade_result.calculate_pnl()
        
        # Add to in-memory list
        self.trades.append(trade_result)
        
        # Add to analytics
        self.analytics.add_trade(trade_result)
        
        # Persist to database
        if self.db:
            self.db.save_trade(trade_result)
        
        # Process attributions if available
        if trade_result.news_events:
            self._process_attributions(trade_result)
    
    def calculate_metrics(self, period: str = "all") -> PerformanceMetrics:
        """Calculate performance metrics for a period.
        
        Args:
            period: Time period for metrics
            
        Returns:
            Performance metrics
        """
        # Filter trades by period
        filtered_trades = self._filter_trades_by_period(period)
        
        # Calculate metrics
        return PerformanceMetrics.from_trades(filtered_trades)
    
    def get_trade_history(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[TradeResult]:
        """Get filtered trade history.
        
        Args:
            filters: Optional filters
            
        Returns:
            List of trades matching filters
        """
        trades = self.trades
        
        if not filters:
            return trades
        
        # Apply filters
        if "asset" in filters:
            trades = [t for t in trades if t.asset == filters["asset"]]
        
        if "status" in filters:
            trades = [t for t in trades if t.status == filters["status"]]
        
        if "date_from" in filters:
            date_from = filters["date_from"]
            trades = [t for t in trades if t.entry_time >= date_from]
        
        if "date_to" in filters:
            date_to = filters["date_to"]
            trades = [t for t in trades if t.entry_time <= date_to]
        
        if "min_pnl" in filters:
            trades = [t for t in trades if t.pnl >= filters["min_pnl"]]
        
        if "max_pnl" in filters:
            trades = [t for t in trades if t.pnl <= filters["max_pnl"]]
        
        return trades
    
    def generate_report(self, format: str = "json") -> Any:
        """Generate performance report.
        
        Args:
            format: Report format
            
        Returns:
            Report in specified format
        """
        # Get comprehensive analytics
        report_data = self.analytics.generate_report()
        
        # Add attribution analysis
        report_data["source_performance"] = self.analytics.get_source_performance()
        
        # Add ML model performance
        report_data["ml_models"] = self._get_all_model_metrics()
        
        # Add sentiment accuracy
        report_data["sentiment_accuracy"] = self.sentiment_tracker.calculate_accuracy()
        
        # Format report
        if format == "json":
            return self.report_generator.generate_json(report_data)
        elif format == "csv":
            return self.report_generator.generate_csv(self.trades)
        elif format == "summary":
            metrics = self.calculate_metrics()
            return self.report_generator.generate_summary(metrics.__dict__)
        else:
            return report_data
    
    def record_ml_prediction(
        self,
        model_name: str,
        prediction_id: str,
        predicted_value: float,
        confidence: float,
        actual_value: float,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record an ML model prediction.
        
        Args:
            model_name: Name of the model
            prediction_id: Unique prediction ID
            predicted_value: Predicted value
            confidence: Model confidence
            actual_value: Actual observed value
            feature_importance: Optional feature importance scores
        """
        self.ml_tracker.record_prediction(
            model_name=model_name,
            prediction_id=prediction_id,
            predicted_value=predicted_value,
            confidence=confidence,
            actual_value=actual_value,
            timestamp=datetime.now(),
            feature_importance=feature_importance,
        )
        
        # Persist to database
        if self.db:
            self.db.save_ml_prediction({
                "model_name": model_name,
                "prediction_id": prediction_id,
                "predicted_value": predicted_value,
                "confidence": confidence,
                "actual_value": actual_value,
                "timestamp": datetime.now(),
                "feature_importance": feature_importance,
            })
    
    def record_sentiment_prediction(
        self,
        news_id: str,
        predicted_sentiment: float,
        predicted_impact: str,
        actual_price_change: float,
        model_name: Optional[str] = None,
    ) -> None:
        """Record a sentiment prediction.
        
        Args:
            news_id: News event ID
            predicted_sentiment: Predicted sentiment score
            predicted_impact: Predicted market impact
            actual_price_change: Actual price change
            model_name: Optional model name
        """
        self.sentiment_tracker.record_prediction(
            news_id=news_id,
            predicted_sentiment=predicted_sentiment,
            predicted_impact=predicted_impact,
            actual_price_change=actual_price_change,
            model_name=model_name,
        )
    
    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Get metrics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model performance metrics
        """
        return self.ml_tracker.get_model_metrics(model_name)
    
    def export_data(self, output_path: str, format: str = "json") -> None:
        """Export all performance data.
        
        Args:
            output_path: Path to save the export
            format: Export format (json, csv)
        """
        if format == "json":
            data = {
                "trades": [self._trade_to_dict(t) for t in self.trades],
                "metrics": self.calculate_metrics().__dict__,
                "report": self.analytics.generate_report(),
                "exported_at": datetime.now().isoformat(),
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format == "csv":
            csv_data = self.report_generator.generate_csv(self.trades)
            with open(output_path, 'w') as f:
                f.write(csv_data)
    
    def _filter_trades_by_period(self, period: str) -> List[TradeResult]:
        """Filter trades by time period."""
        if period == "all":
            return self.trades
        
        now = datetime.now()
        
        if period == "day":
            cutoff = now - timedelta(days=1)
        elif period == "week":
            cutoff = now - timedelta(weeks=1)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        elif period == "year":
            cutoff = now - timedelta(days=365)
        else:
            return self.trades
        
        return [
            t for t in self.trades
            if (t.exit_time or t.entry_time) >= cutoff
        ]
    
    def _process_attributions(self, trade: TradeResult) -> None:
        """Process trade attributions."""
        # This would integrate with the news system to get metadata
        # For now, we'll use mock data
        news_metadata = self._get_news_metadata(trade.news_events)
        
        if news_metadata:
            attribution = self.attributor.attribute_trade(trade, news_metadata)
            self.analytics.add_trade_with_attribution(
                trade,
                attribution.source_contributions
            )
    
    def _get_news_metadata(self, news_ids: List[str]) -> Dict[str, Dict]:
        """Get news metadata for attribution."""
        # This would fetch from the news database
        # Mock implementation for now
        metadata = {}
        
        for news_id in news_ids:
            # In real implementation, fetch from database
            metadata[news_id] = {
                "source": "mock_source",
                "sentiment": 0.7,
                "published": datetime.now() - timedelta(hours=1),
                "impact_score": 0.8,
            }
        
        return metadata
    
    def _get_all_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all ML models."""
        metrics = {}
        
        for model_name in self.ml_tracker.predictions:
            metrics[model_name] = self.ml_tracker.get_model_metrics(model_name)
        
        return metrics
    
    def _load_from_database(self) -> None:
        """Load existing data from database."""
        if not self.db:
            return
        
        # Load trades
        trades = self.db.load_trades()
        for trade_data in trades:
            trade = self._dict_to_trade(trade_data)
            self.trades.append(trade)
            self.analytics.add_trade(trade)
        
        # Load ML predictions
        predictions = self.db.load_ml_predictions()
        for pred in predictions:
            self.ml_tracker.record_prediction(**pred)
    
    def _trade_to_dict(self, trade: TradeResult) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "trade_id": trade.trade_id,
            "signal_id": trade.signal_id,
            "asset": trade.asset,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "position_size": trade.position_size,
            "pnl": trade.pnl,
            "pnl_percentage": trade.pnl_percentage,
            "status": trade.status.value,
            "news_events": trade.news_events,
            "sentiment_scores": trade.sentiment_scores,
            "fees": trade.fees,
            "metadata": trade.metadata,
        }
    
    def _dict_to_trade(self, data: Dict[str, Any]) -> TradeResult:
        """Convert dictionary to trade."""
        from datetime import datetime
        
        return TradeResult(
            trade_id=data["trade_id"],
            signal_id=data["signal_id"],
            asset=data["asset"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]) if data["exit_time"] else None,
            entry_price=data["entry_price"],
            exit_price=data.get("exit_price"),
            position_size=data["position_size"],
            pnl=data.get("pnl", 0),
            pnl_percentage=data.get("pnl_percentage", 0),
            status=TradeStatus[data["status"]],
            news_events=data.get("news_events", []),
            sentiment_scores=data.get("sentiment_scores", []),
            fees=data.get("fees", 0),
            metadata=data.get("metadata", {}),
        )