"""
Fantasy Collective System
========================

A comprehensive platform combining sports betting, prediction markets, and syndicate management
into a unified fantasy collective system supporting fantasy sports, corporate predictions,
business outcomes, news events, and custom predictions.

This system provides:
- Fantasy league management with multiple game types
- Collective intelligence and group decision-making
- Prediction markets for any type of event
- Advanced scoring algorithms with GPU acceleration
- Achievement and reward systems
- Tournament and bracket management
- Complete financial management with multi-wallet support
"""

from .database.manager import FantasyCollectiveDBManager
from .scoring.engine import FantasyCollectiveScoringEngine

__version__ = "1.0.0"
__all__ = [
    "FantasyCollectiveDBManager",
    "FantasyCollectiveScoringEngine",
]

# Configuration presets
SCORING_PRESETS = {
    "fantasy_sports": {
        "scoring_type": "points_based",
        "enable_bonuses": True,
        "enable_multipliers": True,
        "cache_ttl": 300,
        "batch_size": 100,
    },
    "prediction_markets": {
        "scoring_type": "prediction_accuracy",
        "enable_calibration": True,
        "confidence_weight": 0.3,
        "cache_ttl": 600,
        "batch_size": 500,
    },
    "business_collective": {
        "scoring_type": "collective_wisdom",
        "enable_consensus": True,
        "diversity_bonus": 0.2,
        "cache_ttl": 900,
        "batch_size": 200,
    },
    "news_betting": {
        "scoring_type": "prediction_accuracy",
        "enable_real_time": True,
        "update_frequency": 60,
        "cache_ttl": 60,
        "batch_size": 1000,
    },
}

# Quick start function
def create_fantasy_system(db_path="fantasy_collective.db", preset="fantasy_sports"):
    """
    Create a ready-to-use fantasy collective system with preset configuration.
    
    Args:
        db_path: Path to SQLite database file
        preset: Configuration preset name
        
    Returns:
        Tuple of (db_manager, scoring_engine)
    """
    from .database.manager import FantasyCollectiveDBManager
    from .scoring.engine import FantasyCollectiveScoringEngine
    
    # Initialize database
    db_manager = FantasyCollectiveDBManager(db_path)
    
    # Get preset configuration
    config = SCORING_PRESETS.get(preset, SCORING_PRESETS["fantasy_sports"])
    
    # Initialize scoring engine
    scoring_engine = FantasyCollectiveScoringEngine(
        db_path=db_path,
        config=config
    )
    
    return db_manager, scoring_engine