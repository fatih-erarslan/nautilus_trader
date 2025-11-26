"""
Configuration Management for Fantasy Collective Scoring System

This module provides configuration classes and default settings
for the scoring engine.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from .engine import GameType, ScoringRule, Achievement


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    redis_url: Optional[str] = None
    memory_cache_size: int = 10000
    default_ttl: int = 300
    enable_redis: bool = True
    redis_max_connections: int = 100
    redis_retry_on_timeout: bool = True


@dataclass
class GPUConfig:
    """GPU acceleration configuration"""
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    cuda_device_id: int = 0
    batch_size: int = 1024
    mixed_precision: bool = True
    fallback_to_cpu: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration for historical tracking"""
    db_path: str = ":memory:"
    connection_pool_size: int = 10
    enable_wal: bool = True
    synchronous_mode: str = "NORMAL"
    journal_mode: str = "WAL"
    cache_size: int = -2000  # 2MB cache


@dataclass
class EloConfig:
    """ELO rating system configuration"""
    base_k_factor: float = 32.0
    initial_rating: float = 1500.0
    min_rating: float = 100.0
    max_rating: float = 3000.0
    provisional_games: int = 30
    rating_floor: float = 600.0
    volatility_threshold: float = 100.0


@dataclass
class AchievementConfig:
    """Achievement system configuration"""
    enable_achievements: bool = True
    max_bonus_multiplier: float = 2.0
    achievement_decay_days: int = 365
    notification_enabled: bool = True
    achievement_categories: List[str] = field(default_factory=lambda: [
        "performance", "consistency", "improvement", "milestones", "special"
    ])


@dataclass
class ScoringEngineConfig:
    """Main scoring engine configuration"""
    
    # Core settings
    engine_name: str = "fantasy_collective_scoring"
    version: str = "1.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    achievements: AchievementConfig = field(default_factory=AchievementConfig)
    
    # Scoring settings
    max_score_per_rule: float = 1000.0
    min_score_per_rule: float = -100.0
    score_precision: int = 2
    enable_real_time_updates: bool = True
    
    # Performance settings
    thread_pool_size: int = 10
    batch_processing_threshold: int = 100
    async_processing: bool = True
    
    # API settings
    api_rate_limit: int = 1000  # requests per hour
    max_bulk_operations: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoringEngineConfig':
        """Create configuration from dictionary"""
        # Handle nested dataclasses
        if 'cache' in data:
            data['cache'] = CacheConfig(**data['cache'])
        if 'gpu' in data:
            data['gpu'] = GPUConfig(**data['gpu'])
        if 'database' in data:
            data['database'] = DatabaseConfig(**data['database'])
        if 'elo' in data:
            data['elo'] = EloConfig(**data['elo'])
        if 'achievements' in data:
            data['achievements'] = AchievementConfig(**data['achievements'])
        
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ScoringEngineConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration settings"""
        errors = []
        
        # Validate cache settings
        if self.cache.memory_cache_size <= 0:
            errors.append("Cache memory_cache_size must be positive")
        
        if self.cache.default_ttl <= 0:
            errors.append("Cache default_ttl must be positive")
        
        # Validate GPU settings
        if self.gpu.gpu_memory_fraction <= 0 or self.gpu.gpu_memory_fraction > 1:
            errors.append("GPU memory_fraction must be between 0 and 1")
        
        if self.gpu.batch_size <= 0:
            errors.append("GPU batch_size must be positive")
        
        # Validate ELO settings
        if self.elo.base_k_factor <= 0:
            errors.append("ELO base_k_factor must be positive")
        
        if self.elo.initial_rating <= 0:
            errors.append("ELO initial_rating must be positive")
        
        if self.elo.min_rating >= self.elo.max_rating:
            errors.append("ELO min_rating must be less than max_rating")
        
        # Validate scoring settings
        if self.max_score_per_rule <= self.min_score_per_rule:
            errors.append("max_score_per_rule must be greater than min_score_per_rule")
        
        if self.score_precision < 0:
            errors.append("score_precision must be non-negative")
        
        # Validate performance settings
        if self.thread_pool_size <= 0:
            errors.append("thread_pool_size must be positive")
        
        if self.batch_processing_threshold <= 0:
            errors.append("batch_processing_threshold must be positive")
        
        return errors


class DefaultConfigurations:
    """Predefined configurations for different use cases"""
    
    @staticmethod
    def development_config() -> ScoringEngineConfig:
        """Configuration optimized for development"""
        return ScoringEngineConfig(
            debug_mode=True,
            log_level="DEBUG",
            database=DatabaseConfig(db_path=":memory:"),
            cache=CacheConfig(
                redis_url=None,
                memory_cache_size=1000,
                enable_redis=False
            ),
            gpu=GPUConfig(
                enable_gpu=False,
                fallback_to_cpu=True
            ),
            thread_pool_size=2,
            api_rate_limit=100
        )
    
    @staticmethod
    def production_config() -> ScoringEngineConfig:
        """Configuration optimized for production"""
        return ScoringEngineConfig(
            debug_mode=False,
            log_level="INFO",
            database=DatabaseConfig(
                db_path="production_scoring.db",
                connection_pool_size=20,
                cache_size=-10000  # 10MB cache
            ),
            cache=CacheConfig(
                redis_url="redis://localhost:6379/0",
                memory_cache_size=50000,
                default_ttl=600,
                redis_max_connections=200
            ),
            gpu=GPUConfig(
                enable_gpu=True,
                gpu_memory_fraction=0.7,
                batch_size=4096,
                mixed_precision=True
            ),
            thread_pool_size=20,
            batch_processing_threshold=1000,
            api_rate_limit=10000
        )
    
    @staticmethod
    def high_performance_config() -> ScoringEngineConfig:
        """Configuration optimized for high performance"""
        return ScoringEngineConfig(
            debug_mode=False,
            log_level="WARNING",
            database=DatabaseConfig(
                db_path="performance_scoring.db",
                connection_pool_size=50,
                cache_size=-50000,  # 50MB cache
                synchronous_mode="OFF",  # Faster but less safe
            ),
            cache=CacheConfig(
                redis_url="redis://localhost:6379/0",
                memory_cache_size=100000,
                default_ttl=1200,
                redis_max_connections=500
            ),
            gpu=GPUConfig(
                enable_gpu=True,
                gpu_memory_fraction=0.9,
                batch_size=8192,
                mixed_precision=True
            ),
            thread_pool_size=50,
            batch_processing_threshold=500,
            async_processing=True,
            api_rate_limit=50000
        )
    
    @staticmethod
    def testing_config() -> ScoringEngineConfig:
        """Configuration optimized for testing"""
        return ScoringEngineConfig(
            debug_mode=True,
            log_level="ERROR",  # Reduce test output
            database=DatabaseConfig(db_path=":memory:"),
            cache=CacheConfig(
                redis_url=None,
                memory_cache_size=100,
                enable_redis=False,
                default_ttl=10
            ),
            gpu=GPUConfig(
                enable_gpu=False,
                fallback_to_cpu=True
            ),
            achievements=AchievementConfig(
                notification_enabled=False
            ),
            thread_pool_size=1,
            async_processing=False,
            api_rate_limit=10000
        )


class LeagueTemplates:
    """Predefined league configurations and scoring rules"""
    
    @staticmethod
    def nba_fantasy_rules() -> List[ScoringRule]:
        """Standard NBA fantasy scoring rules"""
        return [
            ScoringRule("points", "Points", 1.0, weight=1.0),
            ScoringRule("rebounds", "Rebounds", 1.2, weight=1.0),
            ScoringRule("assists", "Assists", 1.5, weight=1.0),
            ScoringRule("steals", "Steals", 3.0, weight=1.0),
            ScoringRule("blocks", "Blocks", 3.0, weight=1.0),
            ScoringRule("turnovers", "Turnovers", -1.0, weight=1.0),
            ScoringRule("three_pointers", "3-Pointers Made", 3.0, weight=1.0),
            ScoringRule("field_goals_made", "Field Goals Made", 2.0, weight=1.0),
            ScoringRule("field_goals_missed", "Field Goals Missed", -1.0, weight=1.0),
            ScoringRule("free_throws_made", "Free Throws Made", 1.0, weight=1.0),
            ScoringRule("free_throws_missed", "Free Throws Missed", -0.5, weight=1.0),
            ScoringRule("double_double", "Double Double", 1.5, weight=1.0, bonus_points=10.0),
            ScoringRule("triple_double", "Triple Double", 3.0, weight=1.0, bonus_points=25.0)
        ]
    
    @staticmethod
    def nfl_fantasy_rules() -> List[ScoringRule]:
        """Standard NFL fantasy scoring rules"""
        return [
            # Passing
            ScoringRule("passing_yards", "Passing Yards", 0.04, weight=1.0),
            ScoringRule("passing_touchdowns", "Passing TDs", 4.0, weight=1.0),
            ScoringRule("interceptions", "Interceptions", -2.0, weight=1.0),
            
            # Rushing
            ScoringRule("rushing_yards", "Rushing Yards", 0.1, weight=1.0),
            ScoringRule("rushing_touchdowns", "Rushing TDs", 6.0, weight=1.0),
            
            # Receiving
            ScoringRule("receiving_yards", "Receiving Yards", 0.1, weight=1.0),
            ScoringRule("receiving_touchdowns", "Receiving TDs", 6.0, weight=1.0),
            ScoringRule("receptions", "Receptions", 0.5, weight=1.0),  # PPR
            
            # Special bonuses
            ScoringRule("long_touchdown", "40+ Yard TD", 0.0, weight=1.0, bonus_points=2.0),
            ScoringRule("fumbles_lost", "Fumbles Lost", -2.0, weight=1.0)
        ]
    
    @staticmethod
    def prediction_market_rules() -> List[ScoringRule]:
        """Prediction market scoring rules"""
        return [
            ScoringRule("accuracy", "Prediction Accuracy", 100.0, weight=1.0),
            ScoringRule("calibration", "Calibration Score", 50.0, weight=1.0),
            ScoringRule("brier_score", "Brier Score", -25.0, weight=1.0),  # Lower is better
            ScoringRule("log_score", "Logarithmic Score", 30.0, weight=1.0),
            ScoringRule("early_prediction_bonus", "Early Prediction", 0.0, weight=1.0, 
                       bonus_threshold=0.8, bonus_points=20.0),
            ScoringRule("confidence_accuracy", "Confidence-Weighted Accuracy", 75.0, weight=1.0)
        ]
    
    @staticmethod
    def collective_wisdom_rules() -> List[ScoringRule]:
        """Collective wisdom scoring rules"""
        return [
            ScoringRule("individual_accuracy", "Individual Accuracy", 40.0, weight=1.0),
            ScoringRule("group_contribution", "Group Accuracy Improvement", 30.0, weight=1.0),
            ScoringRule("diversity_index", "Diversity Contribution", 20.0, weight=1.0),
            ScoringRule("consensus_quality", "Consensus Alignment Quality", 25.0, weight=1.0),
            ScoringRule("novel_insights", "Novel Insight Bonus", 0.0, weight=1.0, bonus_points=50.0),
            ScoringRule("meta_prediction", "Meta-Prediction Accuracy", 35.0, weight=1.0)
        ]
    
    @staticmethod
    def get_default_achievements() -> List[Achievement]:
        """Get default achievements for any league"""
        return [
            # Performance achievements
            Achievement(
                achievement_id="perfect_game",
                name="Perfect Game",
                description="Score 100+ points in a single game",
                criteria={"min_score": 100},
                points_bonus=25.0,
                category="performance",
                tier="bronze"
            ),
            Achievement(
                achievement_id="outstanding_performance",
                name="Outstanding Performance",
                description="Score 200+ points in a single game",
                criteria={"min_score": 200},
                points_bonus=75.0,
                multiplier_bonus=1.1,
                category="performance",
                tier="gold"
            ),
            
            # Consistency achievements
            Achievement(
                achievement_id="consistent_performer",
                name="Consistent Performer",
                description="Maintain 80%+ consistency for 10 games",
                criteria={"consistency_threshold": 0.8, "min_games": 10},
                points_bonus=50.0,
                category="consistency",
                tier="silver"
            ),
            Achievement(
                achievement_id="rock_solid",
                name="Rock Solid",
                description="Maintain 90%+ consistency for 20 games",
                criteria={"consistency_threshold": 0.9, "min_games": 20},
                points_bonus=150.0,
                multiplier_bonus=1.15,
                category="consistency",
                tier="diamond"
            ),
            
            # Streak achievements
            Achievement(
                achievement_id="hot_streak",
                name="Hot Streak",
                description="Win 5 games in a row",
                criteria={"win_streak": 5},
                points_bonus=30.0,
                category="performance",
                tier="bronze"
            ),
            Achievement(
                achievement_id="unstoppable",
                name="Unstoppable",
                description="Win 10 games in a row",
                criteria={"win_streak": 10},
                points_bonus=100.0,
                multiplier_bonus=1.2,
                category="performance",
                tier="platinum"
            ),
            
            # Improvement achievements
            Achievement(
                achievement_id="rising_star",
                name="Rising Star",
                description="Improve performance trend significantly",
                criteria={"improvement_threshold": 0.2},
                points_bonus=40.0,
                category="improvement",
                tier="silver"
            ),
            
            # Milestone achievements
            Achievement(
                achievement_id="veteran",
                name="Veteran Player",
                description="Play 100 games",
                criteria={"min_games": 100},
                points_bonus=75.0,
                multiplier_bonus=1.05,
                one_time_only=True,
                category="milestones",
                tier="gold"
            ),
            Achievement(
                achievement_id="hall_of_fame",
                name="Hall of Fame",
                description="Play 1000 games",
                criteria={"min_games": 1000},
                points_bonus=500.0,
                multiplier_bonus=1.25,
                one_time_only=True,
                category="milestones",
                tier="diamond"
            )
        ]


class ConfigurationManager:
    """Manager for loading and saving configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: ScoringEngineConfig, name: str):
        """Save a configuration with a given name"""
        filepath = self.config_dir / f"{name}.json"
        config.save_to_file(str(filepath))
    
    def load_config(self, name: str) -> ScoringEngineConfig:
        """Load a configuration by name"""
        filepath = self.config_dir / f"{name}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found")
        
        return ScoringEngineConfig.load_from_file(str(filepath))
    
    def list_configs(self) -> List[str]:
        """List available configuration names"""
        configs = []
        
        for filepath in self.config_dir.glob("*.json"):
            configs.append(filepath.stem)
        
        return sorted(configs)
    
    def create_default_configs(self):
        """Create default configuration files"""
        defaults = {
            "development": DefaultConfigurations.development_config(),
            "production": DefaultConfigurations.production_config(),
            "high_performance": DefaultConfigurations.high_performance_config(),
            "testing": DefaultConfigurations.testing_config()
        }
        
        for name, config in defaults.items():
            self.save_config(config, name)
    
    def get_config_from_env(self) -> ScoringEngineConfig:
        """Create configuration from environment variables"""
        config = DefaultConfigurations.development_config()
        
        # Override with environment variables
        if os.getenv("SCORING_DEBUG"):
            config.debug_mode = os.getenv("SCORING_DEBUG").lower() == "true"
        
        if os.getenv("SCORING_LOG_LEVEL"):
            config.log_level = os.getenv("SCORING_LOG_LEVEL")
        
        if os.getenv("REDIS_URL"):
            config.cache.redis_url = os.getenv("REDIS_URL")
        
        if os.getenv("DATABASE_PATH"):
            config.database.db_path = os.getenv("DATABASE_PATH")
        
        if os.getenv("ENABLE_GPU"):
            config.gpu.enable_gpu = os.getenv("ENABLE_GPU").lower() == "true"
        
        if os.getenv("THREAD_POOL_SIZE"):
            config.thread_pool_size = int(os.getenv("THREAD_POOL_SIZE"))
        
        return config