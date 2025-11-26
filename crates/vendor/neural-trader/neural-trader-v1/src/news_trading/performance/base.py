"""Base classes for performance tracking."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from .models import TradeResult, PerformanceMetrics


class PerformanceTracker(ABC):
    """Abstract base class for performance tracking."""
    
    @abstractmethod
    def record_trade(self, trade_result: TradeResult) -> None:
        """Record a completed trade.
        
        Args:
            trade_result: The trade result to record
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, period: str = "all") -> PerformanceMetrics:
        """Calculate performance metrics for a period.
        
        Args:
            period: Time period for metrics ("all", "day", "week", "month", "year")
            
        Returns:
            Performance metrics for the period
        """
        pass
    
    @abstractmethod
    def get_trade_history(self, filters: Optional[Dict[str, Any]] = None) -> List[TradeResult]:
        """Get filtered trade history.
        
        Args:
            filters: Optional filters (asset, date_range, status, etc.)
            
        Returns:
            List of trades matching filters
        """
        pass
    
    @abstractmethod
    def generate_report(self, format: str = "json") -> Dict[str, Any]:
        """Generate performance report.
        
        Args:
            format: Report format ("json", "csv", "pdf")
            
        Returns:
            Report data in specified format
        """
        pass


class AttributionEngine(ABC):
    """Abstract base class for trade attribution."""
    
    @abstractmethod
    def attribute_trade(self, trade: TradeResult, context: Dict[str, Any]) -> "Attribution":
        """Attribute trade to sources.
        
        Args:
            trade: The trade to attribute
            context: Context including news events, market data, etc.
            
        Returns:
            Attribution object with source contributions
        """
        pass
    
    @abstractmethod
    def get_source_performance(self) -> Dict[str, "SourceMetrics"]:
        """Get performance metrics by news source.
        
        Returns:
            Dictionary of source name to metrics
        """
        pass
    
    @abstractmethod
    def get_model_performance(self) -> Dict[str, "ModelMetrics"]:
        """Get performance metrics by ML model.
        
        Returns:
            Dictionary of model name to metrics
        """
        pass