"""Performance tracking module for the AI News Trading platform."""

from .models import (
    TradeStatus,
    TradeResult,
    PerformanceMetrics,
    Attribution,
    SourceMetrics,
    ModelMetrics,
)

from .base import (
    PerformanceTracker,
    AttributionEngine,
)

from .tracker import ConcretePerformanceTracker
from .attribution import TradeAttributor, SentimentAccuracyTracker
from .ml_tracking import MLModelTracker, MLModelComparator
from .ab_testing import ABTestFramework, MultiArmedBandit
from .analytics import PerformanceAnalytics, ReportGenerator, RealTimeMetrics
from .persistence import PerformanceDatabase

__all__ = [
    # Models
    "TradeStatus",
    "TradeResult",
    "PerformanceMetrics",
    "Attribution",
    "SourceMetrics",
    "ModelMetrics",
    
    # Base classes
    "PerformanceTracker",
    "AttributionEngine",
    
    # Concrete implementations
    "ConcretePerformanceTracker",
    "TradeAttributor",
    "SentimentAccuracyTracker",
    "MLModelTracker",
    "MLModelComparator",
    "ABTestFramework",
    "MultiArmedBandit",
    "PerformanceAnalytics",
    "ReportGenerator",
    "RealTimeMetrics",
    "PerformanceDatabase",
]