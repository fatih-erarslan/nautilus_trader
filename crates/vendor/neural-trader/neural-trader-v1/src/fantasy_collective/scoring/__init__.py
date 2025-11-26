"""
Fantasy Collective Scoring System

A comprehensive scoring engine for fantasy sports, prediction markets,
collective wisdom systems, and competitive leagues.

Features:
- Multiple scoring systems (fantasy sports, prediction accuracy, collective wisdom)
- Custom scoring rules per league/collective
- Real-time score calculation with caching
- Historical performance tracking
- ELO rating system for competitive leagues
- Bonus point systems and multipliers
- Achievement-based scoring boosts
- GPU-accelerated batch scoring for large datasets
"""

from .engine import (
    ScoringEngine,
    GameType,
    ScoreType,
    ScoringRule,
    Achievement,
    PlayerPerformance,
    EloRating,
    BaseScorer,
    FantasySportsScorer,
    PredictionAccuracyScorer,
    CollectiveWisdomScorer,
    EloRatingSystem,
    AchievementEngine,
    GameTypeStrategies,
    GPUAccelerator,
    CacheManager,
    HistoricalPerformanceTracker
)

__version__ = "1.0.0"
__author__ = "Fantasy Systems Developer"

__all__ = [
    "ScoringEngine",
    "GameType", 
    "ScoreType",
    "ScoringRule",
    "Achievement",
    "PlayerPerformance",
    "EloRating",
    "BaseScorer",
    "FantasySportsScorer",
    "PredictionAccuracyScorer", 
    "CollectiveWisdomScorer",
    "EloRatingSystem",
    "AchievementEngine",
    "GameTypeStrategies",
    "GPUAccelerator",
    "CacheManager",
    "HistoricalPerformanceTracker"
]