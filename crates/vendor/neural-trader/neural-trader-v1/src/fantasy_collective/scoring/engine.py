"""
Comprehensive Scoring Engine for Fantasy Collective Systems

This engine implements multiple scoring systems for fantasy sports, prediction accuracy,
collective wisdom, and competitive leagues with GPU acceleration support.
"""

import asyncio
import json
import logging
import math
import numpy as np
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
import sqlite3
import hashlib

try:
    import cupy as cp
    import cupyx.scipy.stats as gpu_stats
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

try:
    import numba
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None


class GameType(Enum):
    """Supported game types for fantasy scoring"""
    HEAD_TO_HEAD = "head_to_head"
    ROTISSERIE = "rotisserie"
    POINTS_BASED = "points_based"
    SURVIVOR_POOLS = "survivor_pools"
    BRACKET_TOURNAMENT = "bracket_tournament"
    PREDICTION_MARKET = "prediction_market"
    COLLECTIVE_WISDOM = "collective_wisdom"
    DYNASTY_LEAGUE = "dynasty_league"
    DAILY_FANTASY = "daily_fantasy"


class ScoreType(Enum):
    """Types of scoring calculations"""
    FANTASY_SPORTS = "fantasy_sports"
    PREDICTION_ACCURACY = "prediction_accuracy"
    COLLECTIVE_WISDOM = "collective_wisdom"
    ELO_RATING = "elo_rating"
    ACHIEVEMENT_BONUS = "achievement_bonus"
    MULTIPLIER_BONUS = "multiplier_bonus"


@dataclass
class ScoringRule:
    """Configuration for a single scoring rule"""
    rule_id: str
    name: str
    points_per_unit: float
    maximum_points: Optional[float] = None
    minimum_points: Optional[float] = None
    multiplier: float = 1.0
    bonus_threshold: Optional[float] = None
    bonus_points: float = 0.0
    decay_factor: float = 1.0
    weight: float = 1.0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Achievement:
    """Achievement configuration for bonus scoring"""
    achievement_id: str
    name: str
    description: str
    criteria: Dict[str, Any]
    points_bonus: float
    multiplier_bonus: float = 1.0
    one_time_only: bool = True
    category: str = "general"
    tier: str = "bronze"  # bronze, silver, gold, platinum, diamond


@dataclass
class PlayerPerformance:
    """Individual player performance data"""
    player_id: str
    league_id: str
    game_type: GameType
    timestamp: datetime
    stats: Dict[str, float]
    base_score: float = 0.0
    bonus_score: float = 0.0
    multiplier: float = 1.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EloRating:
    """ELO rating system implementation"""
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    peak_rating: float = 1500.0
    lowest_rating: float = 1500.0
    volatility: float = 0.0
    k_factor: float = 32.0
    last_updated: datetime = field(default_factory=datetime.now)


class GPUAccelerator:
    """GPU acceleration for batch scoring operations"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE and cp is not None
        self.logger = logging.getLogger(__name__)
        
        if self.gpu_available:
            self.logger.info("GPU acceleration enabled with CuPy")
        else:
            self.logger.warning("GPU acceleration disabled - falling back to CPU")
    
    def batch_calculate_scores(self, stats_matrix: np.ndarray, 
                             scoring_rules: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch score calculation"""
        if not self.gpu_available:
            return self._cpu_batch_calculate_scores(stats_matrix, scoring_rules)
        
        try:
            # Transfer to GPU
            gpu_stats = cp.asarray(stats_matrix)
            gpu_rules = cp.asarray(scoring_rules)
            
            # Vectorized scoring calculation
            scores = cp.sum(gpu_stats * gpu_rules, axis=1)
            
            # Transfer back to CPU
            return cp.asnumpy(scores)
        
        except Exception as e:
            self.logger.warning(f"GPU calculation failed: {e}, falling back to CPU")
            return self._cpu_batch_calculate_scores(stats_matrix, scoring_rules)
    
    def _cpu_batch_calculate_scores(self, stats_matrix: np.ndarray, 
                                  scoring_rules: np.ndarray) -> np.ndarray:
        """CPU fallback for batch score calculation"""
        return np.sum(stats_matrix * scoring_rules, axis=1)
    
    def batch_elo_updates(self, ratings: np.ndarray, outcomes: np.ndarray, 
                         k_factors: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch ELO rating updates"""
        if not self.gpu_available:
            return self._cpu_batch_elo_updates(ratings, outcomes, k_factors)
        
        try:
            gpu_ratings = cp.asarray(ratings)
            gpu_outcomes = cp.asarray(outcomes)
            gpu_k = cp.asarray(k_factors)
            
            # Expected scores calculation
            rating_diff = gpu_ratings[:, 0] - gpu_ratings[:, 1]
            expected = 1.0 / (1.0 + cp.exp(-rating_diff / 400.0))
            
            # Rating updates
            rating_change = gpu_k * (gpu_outcomes - expected)
            new_ratings = gpu_ratings.copy()
            new_ratings[:, 0] += rating_change
            new_ratings[:, 1] -= rating_change
            
            return cp.asnumpy(new_ratings)
        
        except Exception as e:
            self.logger.warning(f"GPU ELO calculation failed: {e}, falling back to CPU")
            return self._cpu_batch_elo_updates(ratings, outcomes, k_factors)
    
    def _cpu_batch_elo_updates(self, ratings: np.ndarray, outcomes: np.ndarray, 
                             k_factors: np.ndarray) -> np.ndarray:
        """CPU fallback for batch ELO updates"""
        rating_diff = ratings[:, 0] - ratings[:, 1]
        expected = 1.0 / (1.0 + np.exp(-rating_diff / 400.0))
        rating_change = k_factors * (outcomes - expected)
        
        new_ratings = ratings.copy()
        new_ratings[:, 0] += rating_change
        new_ratings[:, 1] -= rating_change
        
        return new_ratings


class CacheManager:
    """Advanced caching system for real-time scoring"""
    
    def __init__(self, redis_url: Optional[str] = None, memory_cache_size: int = 10000):
        self.memory_cache = {}
        self.cache_times = {}
        self.memory_cache_size = memory_cache_size
        self.hit_count = 0
        self.miss_count = 0
        
        # Redis cache setup
        self.redis_client = None
        if redis_url:
            try:
                import redis as redis_lib
                self.redis_client = redis_lib.from_url(redis_url)
                self.redis_client.ping()
            except Exception as e:
                logging.warning(f"Redis connection failed: {e}")
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str, ttl: int = 300) -> Optional[Any]:
        """Get cached value with TTL check"""
        # Check memory cache first
        if key in self.memory_cache:
            cache_time = self.cache_times.get(key, 0)
            if time.time() - cache_time < ttl:
                self.hit_count += 1
                return self.memory_cache[key]
            else:
                # Expired
                del self.memory_cache[key]
                del self.cache_times[key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    self.hit_count += 1
                    value = json.loads(cached_data)
                    self._set_memory_cache(key, value)
                    return value
            except Exception as e:
                self.logger.warning(f"Redis get failed: {e}")
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set cached value with TTL"""
        self._set_memory_cache(key, value)
        
        # Set in Redis with TTL
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value, default=str))
            except Exception as e:
                self.logger.warning(f"Redis set failed: {e}")
    
    def _set_memory_cache(self, key: str, value: Any):
        """Set value in memory cache with size limits"""
        if len(self.memory_cache) >= self.memory_cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache_times, key=self.cache_times.get)
            del self.memory_cache[oldest_key]
            del self.cache_times[oldest_key]
        
        self.memory_cache[key] = value
        self.cache_times[key] = time.time()
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        if pattern:
            # Pattern-based invalidation
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]
                del self.cache_times[key]
            
            if self.redis_client:
                try:
                    keys = self.redis_client.keys(f"*{pattern}*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    self.logger.warning(f"Redis invalidation failed: {e}")
        else:
            # Clear all
            self.memory_cache.clear()
            self.cache_times.clear()
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    self.logger.warning(f"Redis flush failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None
        }


class HistoricalPerformanceTracker:
    """Track and analyze historical performance data"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        # Keep a persistent connection for in-memory databases
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(db_path, check_same_thread=False)
        else:
            self._persistent_conn = None
        self._init_database()
    
    def _get_connection(self):
        """Get database connection"""
        if self._persistent_conn:
            return self._persistent_conn
        else:
            return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Initialize SQLite database for historical data"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Performance history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                league_id TEXT NOT NULL,
                game_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                base_score REAL NOT NULL,
                bonus_score REAL NOT NULL,
                final_score REAL NOT NULL,
                stats_json TEXT NOT NULL,
                metadata_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_player_league 
            ON performance_history(player_id, league_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_timestamp 
            ON performance_history(timestamp)
        """)
        
        # ELO rating history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS elo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                league_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                rating REAL NOT NULL,
                rating_change REAL NOT NULL,
                opponent_id TEXT,
                outcome TEXT,
                k_factor REAL NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_elo_player_league 
            ON elo_history(player_id, league_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_elo_timestamp 
            ON elo_history(timestamp)
        """)
        
        # Achievement history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS achievement_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                league_id TEXT NOT NULL,
                achievement_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                points_awarded REAL NOT NULL,
                metadata_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_achievement_player_league 
            ON achievement_history(player_id, league_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_achievement_id 
            ON achievement_history(achievement_id)
        """)
        
        if not self._persistent_conn:
            conn.commit()
            conn.close()
    
    def record_performance(self, performance: PlayerPerformance):
        """Record player performance in history"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_history 
            (player_id, league_id, game_type, timestamp, base_score, bonus_score, 
             final_score, stats_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            performance.player_id,
            performance.league_id,
            performance.game_type.value,
            performance.timestamp,
            performance.base_score,
            performance.bonus_score,
            performance.final_score,
            json.dumps(performance.stats),
            json.dumps(performance.metadata)
        ))
        
        if not self._persistent_conn:
            conn.commit()
            conn.close()
    
    def record_elo_change(self, player_id: str, league_id: str, 
                         old_rating: float, new_rating: float,
                         opponent_id: str = None, outcome: str = None,
                         k_factor: float = 32.0):
        """Record ELO rating change"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO elo_history 
            (player_id, league_id, timestamp, rating, rating_change, 
             opponent_id, outcome, k_factor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_id, league_id, datetime.now(), new_rating,
            new_rating - old_rating, opponent_id, outcome, k_factor
        ))
        
        if not self._persistent_conn:
            conn.commit()
            conn.close()
    
    def get_performance_trends(self, player_id: str, league_id: str,
                             days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT timestamp, base_score, bonus_score, final_score
            FROM performance_history
            WHERE player_id = ? AND league_id = ? AND timestamp > ?
            ORDER BY timestamp
        """, (player_id, league_id, cutoff_date))
        
        records = cursor.fetchall()
        if not self._persistent_conn:
            conn.close()
        
        if not records:
            return {"trend": "no_data", "games": 0}
        
        scores = [r[3] for r in records]  # final_score
        
        # Calculate trends
        if len(scores) >= 2:
            slope = np.polyfit(range(len(scores)), scores, 1)[0]
            trend = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "games": len(records),
            "avg_score": np.mean(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "recent_scores": scores[-5:],  # Last 5 games
            "consistency": 1.0 / (1.0 + np.std(scores)) if len(scores) > 1 else 1.0
        }
    
    def get_elo_progression(self, player_id: str, league_id: str,
                          days: int = 90) -> List[Tuple[datetime, float]]:
        """Get ELO rating progression over time"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT timestamp, rating
            FROM elo_history
            WHERE player_id = ? AND league_id = ? AND timestamp > ?
            ORDER BY timestamp
        """, (player_id, league_id, cutoff_date))
        
        records = cursor.fetchall()
        if not self._persistent_conn:
            conn.close()
        
        return [(datetime.fromisoformat(r[0]), r[1]) for r in records]


class BaseScorer(ABC):
    """Abstract base class for all scoring systems"""
    
    @abstractmethod
    def calculate_score(self, stats: Dict[str, float], 
                       rules: List[ScoringRule]) -> float:
        pass
    
    @abstractmethod
    def get_scorer_type(self) -> ScoreType:
        pass


class FantasySportsScorer(BaseScorer):
    """Fantasy sports scoring implementation"""
    
    def calculate_score(self, stats: Dict[str, float], 
                       rules: List[ScoringRule]) -> float:
        total_score = 0.0
        
        for rule in rules:
            if not rule.active:
                continue
            
            stat_value = stats.get(rule.rule_id, 0.0)
            
            # Apply decay factor for time-sensitive stats
            if rule.decay_factor != 1.0:
                stat_value *= rule.decay_factor
            
            # Calculate base points
            points = stat_value * rule.points_per_unit * rule.multiplier
            
            # Apply min/max limits
            if rule.maximum_points is not None:
                points = min(points, rule.maximum_points)
            if rule.minimum_points is not None:
                points = max(points, rule.minimum_points)
            
            # Apply bonus threshold
            if (rule.bonus_threshold is not None and 
                stat_value >= rule.bonus_threshold):
                points += rule.bonus_points
            
            # Apply weight
            points *= rule.weight
            
            total_score += points
        
        return max(0.0, total_score)  # Ensure non-negative score
    
    def get_scorer_type(self) -> ScoreType:
        return ScoreType.FANTASY_SPORTS


class PredictionAccuracyScorer(BaseScorer):
    """Prediction accuracy scoring for collective wisdom systems"""
    
    def __init__(self, base_points: float = 100.0):
        self.base_points = base_points
    
    def calculate_score(self, stats: Dict[str, float], 
                       rules: List[ScoringRule]) -> float:
        """
        Calculate score based on prediction accuracy
        Expected stats: {'accuracy', 'confidence', 'calibration', 'timeliness'}
        """
        accuracy = stats.get('accuracy', 0.0)
        confidence = stats.get('confidence', 0.5)
        calibration = stats.get('calibration', 0.0)  # How well-calibrated predictions are
        timeliness = stats.get('timeliness', 1.0)    # Earlier predictions get bonus
        
        # Base score from accuracy
        base_score = accuracy * self.base_points
        
        # Confidence bonus/penalty
        confidence_factor = 1.0 + (confidence - 0.5) * accuracy * 0.5
        
        # Calibration bonus
        calibration_bonus = calibration * 20.0
        
        # Timeliness multiplier
        timeliness_multiplier = 1.0 + (timeliness - 1.0) * 0.3
        
        score = (base_score * confidence_factor + calibration_bonus) * timeliness_multiplier
        
        return max(0.0, score)
    
    def get_scorer_type(self) -> ScoreType:
        return ScoreType.PREDICTION_ACCURACY


class CollectiveWisdomScorer(BaseScorer):
    """Collective wisdom scoring implementation"""
    
    def __init__(self, diversity_weight: float = 0.2, 
                 consensus_weight: float = 0.3):
        self.diversity_weight = diversity_weight
        self.consensus_weight = consensus_weight
    
    def calculate_score(self, stats: Dict[str, float], 
                       rules: List[ScoringRule]) -> float:
        """
        Score based on contribution to collective wisdom
        Expected stats: {'individual_accuracy', 'group_accuracy_improvement', 
                        'diversity_contribution', 'consensus_alignment'}
        """
        individual_acc = stats.get('individual_accuracy', 0.0)
        group_improvement = stats.get('group_accuracy_improvement', 0.0)
        diversity_contrib = stats.get('diversity_contribution', 0.0)
        consensus_align = stats.get('consensus_alignment', 0.5)
        
        # Base score from individual performance
        base_score = individual_acc * 50.0
        
        # Group improvement bonus
        improvement_bonus = group_improvement * 30.0
        
        # Diversity contribution (valuable even if wrong)
        diversity_score = diversity_contrib * self.diversity_weight * 40.0
        
        # Consensus alignment (balanced - not too high or low)
        consensus_bonus = (1.0 - abs(consensus_align - 0.7)) * self.consensus_weight * 20.0
        
        total_score = base_score + improvement_bonus + diversity_score + consensus_bonus
        
        return max(0.0, total_score)
    
    def get_scorer_type(self) -> ScoreType:
        return ScoreType.COLLECTIVE_WISDOM


class EloRatingSystem:
    """Advanced ELO rating system for competitive leagues"""
    
    def __init__(self, base_k_factor: float = 32.0):
        self.base_k_factor = base_k_factor
        self.gpu_accelerator = GPUAccelerator()
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B"""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
    
    def update_rating(self, current_rating: float, expected_score: float,
                     actual_score: float, k_factor: float = None) -> float:
        """Update ELO rating after a match"""
        if k_factor is None:
            k_factor = self.base_k_factor
        
        return current_rating + k_factor * (actual_score - expected_score)
    
    def get_k_factor(self, rating: float, games_played: int) -> float:
        """Calculate dynamic K-factor based on rating and experience"""
        # Higher K for new players and lower ratings
        if games_played < 30:
            return self.base_k_factor * 1.5
        elif rating < 1200:
            return self.base_k_factor * 1.25
        elif rating > 2400:
            return self.base_k_factor * 0.75
        else:
            return self.base_k_factor
    
    def batch_update_ratings(self, matches: List[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
        """
        Batch update ELO ratings for multiple matches
        matches: List of (rating_a, rating_b, outcome) where outcome is 1.0 (A wins), 0.0 (B wins), 0.5 (draw)
        """
        if not matches:
            return []
        
        # Prepare arrays for GPU acceleration
        ratings = np.array([[match[0], match[1]] for match in matches])
        outcomes = np.array([match[2] for match in matches])
        k_factors = np.array([self.base_k_factor] * len(matches))
        
        # Use GPU acceleration if available
        new_ratings = self.gpu_accelerator.batch_elo_updates(ratings, outcomes, k_factors)
        
        return [(new_ratings[i][0], new_ratings[i][1]) for i in range(len(matches))]
    
    def calculate_rating_volatility(self, rating_history: List[float]) -> float:
        """Calculate rating volatility (consistency metric)"""
        if len(rating_history) < 2:
            return 0.0
        
        return np.std(rating_history)
    
    def get_rating_confidence(self, games_played: int, volatility: float) -> float:
        """Calculate confidence in rating based on games played and volatility"""
        # More games = higher confidence, lower volatility = higher confidence
        game_confidence = min(1.0, games_played / 100.0)
        volatility_confidence = 1.0 / (1.0 + volatility / 100.0)
        
        return (game_confidence + volatility_confidence) / 2.0


class AchievementEngine:
    """Achievement system for bonus scoring"""
    
    def __init__(self):
        self.achievements: Dict[str, Achievement] = {}
        self.player_achievements: Dict[str, Set[str]] = defaultdict(set)
        self.logger = logging.getLogger(__name__)
    
    def register_achievement(self, achievement: Achievement):
        """Register a new achievement"""
        self.achievements[achievement.achievement_id] = achievement
    
    def check_achievements(self, player_id: str, performance: PlayerPerformance,
                          historical_data: Dict[str, Any]) -> List[Tuple[Achievement, float]]:
        """Check if player has earned any achievements"""
        earned_achievements = []
        
        for achievement in self.achievements.values():
            # Skip if already earned (for one-time achievements)
            if (achievement.one_time_only and 
                achievement.achievement_id in self.player_achievements[player_id]):
                continue
            
            if self._evaluate_achievement_criteria(achievement, performance, historical_data):
                earned_achievements.append((achievement, achievement.points_bonus))
                self.player_achievements[player_id].add(achievement.achievement_id)
                
                self.logger.info(f"Player {player_id} earned achievement: {achievement.name}")
        
        return earned_achievements
    
    def _evaluate_achievement_criteria(self, achievement: Achievement,
                                     performance: PlayerPerformance,
                                     historical_data: Dict[str, Any]) -> bool:
        """Evaluate if achievement criteria are met"""
        criteria = achievement.criteria
        
        # Score-based achievements
        if 'min_score' in criteria:
            if performance.final_score < criteria['min_score']:
                return False
        
        if 'min_base_score' in criteria:
            if performance.base_score < criteria['min_base_score']:
                return False
        
        # Stat-based achievements
        if 'required_stats' in criteria:
            for stat, min_value in criteria['required_stats'].items():
                if performance.stats.get(stat, 0) < min_value:
                    return False
        
        # Streak-based achievements
        if 'win_streak' in criteria:
            recent_games = historical_data.get('recent_games', 0)
            if recent_games < criteria['win_streak']:
                return False
        
        # Consistency achievements
        if 'consistency_threshold' in criteria:
            consistency = historical_data.get('consistency', 0)
            if consistency < criteria['consistency_threshold']:
                return False
        
        # Performance improvement achievements
        if 'improvement_threshold' in criteria:
            trend = historical_data.get('trend')
            if trend != 'improving':
                return False
        
        return True
    
    def get_achievement_multiplier(self, player_id: str, base_score: float) -> float:
        """Calculate total achievement multiplier for a player"""
        total_multiplier = 1.0
        player_achievement_ids = self.player_achievements.get(player_id, set())
        
        for achievement_id in player_achievement_ids:
            achievement = self.achievements.get(achievement_id)
            if achievement and achievement.multiplier_bonus > 1.0:
                # Apply diminishing returns for multiple multipliers
                multiplier_effect = (achievement.multiplier_bonus - 1.0) * 0.8
                total_multiplier += multiplier_effect
        
        return total_multiplier


class GameTypeStrategies:
    """Specific scoring strategies for different game types"""
    
    @staticmethod
    def head_to_head_scoring(scores: List[float]) -> List[float]:
        """Head-to-head scoring: winner takes all"""
        max_score = max(scores)
        return [1.0 if score == max_score else 0.0 for score in scores]
    
    @staticmethod
    def rotisserie_scoring(stats_matrix: np.ndarray, categories: List[str]) -> np.ndarray:
        """Rotisserie scoring: rank-based points for each category"""
        num_players, num_categories = stats_matrix.shape
        points_matrix = np.zeros_like(stats_matrix)
        
        for cat_idx in range(num_categories):
            category_stats = stats_matrix[:, cat_idx]
            
            # Rank players (higher is better)
            ranks = np.argsort(np.argsort(-category_stats))
            
            # Assign points based on rank (1st = num_players points, last = 1 point)
            points_matrix[:, cat_idx] = num_players - ranks
        
        return np.sum(points_matrix, axis=1)
    
    @staticmethod
    def survivor_pool_scoring(predictions: List[bool], outcomes: List[bool]) -> List[float]:
        """Survivor pool: eliminate incorrect predictions"""
        scores = []
        
        for pred, outcome in zip(predictions, outcomes):
            if pred == outcome:
                scores.append(1.0)
            else:
                scores.append(0.0)  # Eliminated
                break
        
        return scores
    
    @staticmethod
    def bracket_tournament_scoring(bracket_predictions: Dict[str, str],
                                 actual_results: Dict[str, str],
                                 round_multipliers: Dict[str, float]) -> float:
        """Tournament bracket scoring with round multipliers"""
        total_score = 0.0
        
        for round_name, multiplier in round_multipliers.items():
            if round_name in bracket_predictions and round_name in actual_results:
                predicted = bracket_predictions[round_name]
                actual = actual_results[round_name]
                
                if predicted == actual:
                    total_score += multiplier
        
        return total_score


class ScoringEngine:
    """Main scoring engine that orchestrates all scoring systems"""
    
    def __init__(self, redis_url: Optional[str] = None, 
                 db_path: str = ":memory:",
                 enable_gpu: bool = True):
        
        # Core components
        self.cache_manager = CacheManager(redis_url)
        self.performance_tracker = HistoricalPerformanceTracker(db_path)
        self.elo_system = EloRatingSystem()
        self.achievement_engine = AchievementEngine()
        
        # GPU acceleration
        self.gpu_accelerator = GPUAccelerator() if enable_gpu else None
        
        # Scorers
        self.scorers: Dict[ScoreType, BaseScorer] = {
            ScoreType.FANTASY_SPORTS: FantasySportsScorer(),
            ScoreType.PREDICTION_ACCURACY: PredictionAccuracyScorer(),
            ScoreType.COLLECTIVE_WISDOM: CollectiveWisdomScorer()
        }
        
        # Scoring rules per league/game type
        self.league_rules: Dict[str, Dict[GameType, List[ScoringRule]]] = defaultdict(lambda: defaultdict(list))
        
        # ELO ratings
        self.elo_ratings: Dict[Tuple[str, str], EloRating] = {}  # (player_id, league_id) -> EloRating
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default achievements
        self._setup_default_achievements()
    
    def _setup_default_achievements(self):
        """Setup default achievements"""
        achievements = [
            Achievement(
                achievement_id="perfect_week",
                name="Perfect Week",
                description="Score in top 10% for 7 consecutive days",
                criteria={"consistency_threshold": 0.9, "min_games": 7},
                points_bonus=100.0,
                category="consistency"
            ),
            Achievement(
                achievement_id="comeback_king",
                name="Comeback King",
                description="Improve rating by 200+ points from lowest point",
                criteria={"rating_improvement": 200},
                points_bonus=150.0,
                multiplier_bonus=1.1,
                category="improvement"
            ),
            Achievement(
                achievement_id="high_roller",
                name="High Roller",
                description="Achieve a single game score of 500+",
                criteria={"min_score": 500},
                points_bonus=50.0,
                category="performance"
            )
        ]
        
        for achievement in achievements:
            self.achievement_engine.register_achievement(achievement)
    
    def register_league_rules(self, league_id: str, game_type: GameType,
                            rules: List[ScoringRule]):
        """Register scoring rules for a specific league and game type"""
        self.league_rules[league_id][game_type] = rules
        
        # Invalidate related cache
        self.cache_manager.invalidate(f"league_{league_id}")
    
    def register_custom_scorer(self, score_type: ScoreType, scorer: BaseScorer):
        """Register a custom scorer implementation"""
        self.scorers[score_type] = scorer
    
    async def calculate_player_score(self, player_id: str, league_id: str,
                                   game_type: GameType, stats: Dict[str, float],
                                   score_type: ScoreType = ScoreType.FANTASY_SPORTS,
                                   use_cache: bool = True) -> PlayerPerformance:
        """Calculate comprehensive player score"""
        
        # Check cache first
        cache_key = f"score_{player_id}_{league_id}_{game_type.value}_{hash(str(sorted(stats.items())))}"
        
        if use_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return PlayerPerformance(**cached_result)
        
        # Get scoring rules for league and game type
        rules = self.league_rules[league_id][game_type]
        if not rules:
            self.logger.warning(f"No scoring rules found for league {league_id}, game type {game_type}")
            rules = []
        
        # Calculate base score
        scorer = self.scorers.get(score_type, self.scorers[ScoreType.FANTASY_SPORTS])
        base_score = scorer.calculate_score(stats, rules)
        
        # Get historical data for achievements
        historical_data = self.performance_tracker.get_performance_trends(player_id, league_id)
        
        # Create performance object
        performance = PlayerPerformance(
            player_id=player_id,
            league_id=league_id,
            game_type=game_type,
            timestamp=datetime.now(),
            stats=stats,
            base_score=base_score
        )
        
        # Check achievements
        earned_achievements = self.achievement_engine.check_achievements(
            player_id, performance, historical_data
        )
        
        # Calculate bonus score from achievements
        bonus_score = sum(bonus for _, bonus in earned_achievements)
        performance.bonus_score = bonus_score
        
        # Apply achievement multipliers
        achievement_multiplier = self.achievement_engine.get_achievement_multiplier(
            player_id, base_score
        )
        performance.multiplier = achievement_multiplier
        
        # Calculate final score
        performance.final_score = (base_score + bonus_score) * achievement_multiplier
        
        # Store in cache
        if use_cache:
            cache_data = {
                "player_id": player_id,
                "league_id": league_id,
                "game_type": game_type.value,
                "timestamp": performance.timestamp.isoformat(),
                "stats": stats,
                "base_score": base_score,
                "bonus_score": bonus_score,
                "multiplier": achievement_multiplier,
                "final_score": performance.final_score,
                "metadata": {}
            }
            self.cache_manager.set(cache_key, cache_data, ttl=300)
        
        # Record performance history
        self.performance_tracker.record_performance(performance)
        
        return performance
    
    async def batch_calculate_scores(self, player_data: List[Dict[str, Any]],
                                   use_gpu: bool = True) -> List[PlayerPerformance]:
        """GPU-accelerated batch score calculation"""
        if not player_data:
            return []
        
        if not use_gpu or not self.gpu_accelerator:
            # Process sequentially
            results = []
            for data in player_data:
                performance = await self.calculate_player_score(**data)
                results.append(performance)
            return results
        
        # Prepare data for GPU processing
        try:
            # Group by league_id and game_type for batch processing
            grouped_data = defaultdict(list)
            for data in player_data:
                key = (data['league_id'], data['game_type'])
                grouped_data[key].append(data)
            
            results = []
            
            for (league_id, game_type), group_data in grouped_data.items():
                # Get common scoring rules
                rules = self.league_rules[league_id][game_type]
                
                if not rules:
                    # Fall back to individual processing
                    for data in group_data:
                        performance = await self.calculate_player_score(**data)
                        results.append(performance)
                    continue
                
                # Prepare matrices for GPU
                stats_matrix = []
                rule_values = []
                
                for rule in rules:
                    rule_values.append(rule.points_per_unit * rule.multiplier * rule.weight)
                
                for data in group_data:
                    stats_row = [data['stats'].get(rule.rule_id, 0.0) for rule in rules]
                    stats_matrix.append(stats_row)
                
                stats_matrix = np.array(stats_matrix)
                rule_values = np.array(rule_values)
                
                # GPU batch calculation
                scores = self.gpu_accelerator.batch_calculate_scores(stats_matrix, rule_values)
                
                # Process results
                for i, data in enumerate(group_data):
                    performance = PlayerPerformance(
                        player_id=data['player_id'],
                        league_id=data['league_id'],
                        game_type=data['game_type'],
                        timestamp=datetime.now(),
                        stats=data['stats'],
                        base_score=float(scores[i]),
                        final_score=float(scores[i])  # Simplified for batch processing
                    )
                    results.append(performance)
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Batch GPU calculation failed: {e}, falling back to individual processing")
            
            # Fall back to individual processing
            results = []
            for data in player_data:
                performance = await self.calculate_player_score(**data)
                results.append(performance)
            return results
    
    def update_elo_rating(self, player_id: str, league_id: str,
                         opponent_id: str, outcome: float) -> Tuple[float, float]:
        """Update ELO ratings for both players after a match"""
        
        # Get current ratings
        player_key = (player_id, league_id)
        opponent_key = (opponent_id, league_id)
        
        player_elo = self.elo_ratings.get(player_key, EloRating())
        opponent_elo = self.elo_ratings.get(opponent_key, EloRating())
        
        # Calculate expected scores
        player_expected = self.elo_system.calculate_expected_score(
            player_elo.rating, opponent_elo.rating
        )
        opponent_expected = 1.0 - player_expected
        
        # Get K-factors
        player_k = self.elo_system.get_k_factor(player_elo.rating, player_elo.games_played)
        opponent_k = self.elo_system.get_k_factor(opponent_elo.rating, opponent_elo.games_played)
        
        # Update ratings
        old_player_rating = player_elo.rating
        old_opponent_rating = opponent_elo.rating
        
        player_elo.rating = self.elo_system.update_rating(
            player_elo.rating, player_expected, outcome, player_k
        )
        opponent_elo.rating = self.elo_system.update_rating(
            opponent_elo.rating, opponent_expected, 1.0 - outcome, opponent_k
        )
        
        # Update statistics
        player_elo.games_played += 1
        opponent_elo.games_played += 1
        
        if outcome == 1.0:  # Player wins
            player_elo.wins += 1
            opponent_elo.losses += 1
        elif outcome == 0.0:  # Opponent wins
            player_elo.losses += 1
            opponent_elo.wins += 1
        else:  # Draw
            player_elo.draws += 1
            opponent_elo.draws += 1
        
        # Update peak/lowest ratings
        player_elo.peak_rating = max(player_elo.peak_rating, player_elo.rating)
        player_elo.lowest_rating = min(player_elo.lowest_rating, player_elo.rating)
        opponent_elo.peak_rating = max(opponent_elo.peak_rating, opponent_elo.rating)
        opponent_elo.lowest_rating = min(opponent_elo.lowest_rating, opponent_elo.rating)
        
        player_elo.last_updated = datetime.now()
        opponent_elo.last_updated = datetime.now()
        
        # Store updated ratings
        self.elo_ratings[player_key] = player_elo
        self.elo_ratings[opponent_key] = opponent_elo
        
        # Record in history
        self.performance_tracker.record_elo_change(
            player_id, league_id, old_player_rating, player_elo.rating,
            opponent_id, "win" if outcome == 1.0 else "loss" if outcome == 0.0 else "draw",
            player_k
        )
        self.performance_tracker.record_elo_change(
            opponent_id, league_id, old_opponent_rating, opponent_elo.rating,
            player_id, "loss" if outcome == 1.0 else "win" if outcome == 0.0 else "draw",
            opponent_k
        )
        
        return player_elo.rating, opponent_elo.rating
    
    def get_leaderboard(self, league_id: str, game_type: GameType = None,
                       sort_by: str = "final_score", limit: int = 100,
                       time_filter: timedelta = None) -> List[Dict[str, Any]]:
        """Get leaderboard for a league"""
        
        # This would typically query the database
        # For now, return a mock implementation
        leaderboard = []
        
        # In a real implementation, this would:
        # 1. Query performance history for the league
        # 2. Filter by game_type if specified
        # 3. Apply time filter if specified
        # 4. Sort by the specified metric
        # 5. Return top N results
        
        return leaderboard
    
    def get_player_analytics(self, player_id: str, league_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a player"""
        
        # Performance trends
        trends = self.performance_tracker.get_performance_trends(player_id, league_id)
        
        # ELO progression
        elo_history = self.performance_tracker.get_elo_progression(player_id, league_id)
        
        # Current ELO rating
        player_key = (player_id, league_id)
        current_elo = self.elo_ratings.get(player_key, EloRating())
        
        # Achievements
        player_achievements = list(self.achievement_engine.player_achievements[player_id])
        
        return {
            "player_id": player_id,
            "league_id": league_id,
            "performance_trends": trends,
            "elo_rating": {
                "current": current_elo.rating,
                "peak": current_elo.peak_rating,
                "lowest": current_elo.lowest_rating,
                "games_played": current_elo.games_played,
                "win_rate": current_elo.wins / max(current_elo.games_played, 1),
                "volatility": current_elo.volatility
            },
            "elo_history": [(ts.isoformat(), rating) for ts, rating in elo_history],
            "achievements": player_achievements,
            "cache_stats": self.cache_manager.get_stats()
        }
    
    def optimize_scoring_rules(self, league_id: str, game_type: GameType,
                             target_metrics: Dict[str, float],
                             optimization_data: List[Dict[str, Any]]) -> List[ScoringRule]:
        """Optimize scoring rules to achieve target metrics"""
        
        # This is a placeholder for a complex optimization algorithm
        # In practice, this would use genetic algorithms, gradient descent,
        # or other optimization techniques to find the best scoring rules
        
        current_rules = self.league_rules[league_id][game_type]
        
        # Simulate optimization process
        self.logger.info(f"Optimizing scoring rules for league {league_id}, game type {game_type}")
        self.logger.info(f"Target metrics: {target_metrics}")
        
        # Return current rules for now
        return current_rules
    
    async def real_time_score_update(self, player_id: str, league_id: str,
                                   stat_update: Dict[str, float]) -> float:
        """Real-time score update for live games"""
        
        # Get cached current score
        cache_key = f"live_score_{player_id}_{league_id}"
        current_performance = self.cache_manager.get(cache_key)
        
        if current_performance:
            # Update stats
            for stat, value in stat_update.items():
                if stat in current_performance['stats']:
                    current_performance['stats'][stat] += value
                else:
                    current_performance['stats'][stat] = value
            
            # Recalculate score
            # This is simplified - in practice, you'd want to be more efficient
            game_type = GameType(current_performance['game_type'])
            updated_performance = await self.calculate_player_score(
                player_id, league_id, game_type, 
                current_performance['stats'], use_cache=False
            )
            
            # Update cache
            cache_data = {
                "player_id": player_id,
                "league_id": league_id,
                "game_type": game_type.value,
                "timestamp": updated_performance.timestamp.isoformat(),
                "stats": updated_performance.stats,
                "base_score": updated_performance.base_score,
                "bonus_score": updated_performance.bonus_score,
                "multiplier": updated_performance.multiplier,
                "final_score": updated_performance.final_score,
                "metadata": {}
            }
            self.cache_manager.set(cache_key, cache_data, ttl=3600)  # 1 hour TTL for live games
            
            return updated_performance.final_score
        
        return 0.0
    
    def export_league_data(self, league_id: str, format: str = "json") -> str:
        """Export league data for analysis"""
        
        # This would export all relevant data for a league
        # Including rules, performances, rankings, etc.
        
        export_data = {
            "league_id": league_id,
            "scoring_rules": {gt.value: rules for gt, rules in self.league_rules[league_id].items()},
            "export_timestamp": datetime.now().isoformat(),
            "format": format
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format == "csv":
            # Convert to CSV format
            # This would be more complex in a real implementation
            return "CSV export not implemented"
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def shutdown(self):
        """Graceful shutdown of the scoring engine"""
        self.logger.info("Shutting down scoring engine...")
        
        # Close thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear caches
        self.cache_manager.invalidate()
        
        self.logger.info("Scoring engine shutdown complete")


# Example usage and testing functions
def create_sample_league_rules() -> List[ScoringRule]:
    """Create sample scoring rules for testing"""
    return [
        ScoringRule(
            rule_id="points",
            name="Points Scored",
            points_per_unit=1.0,
            weight=1.0
        ),
        ScoringRule(
            rule_id="rebounds",
            name="Rebounds",
            points_per_unit=1.2,
            bonus_threshold=10.0,
            bonus_points=5.0,
            weight=1.0
        ),
        ScoringRule(
            rule_id="assists",
            name="Assists",
            points_per_unit=1.5,
            maximum_points=20.0,
            weight=1.0
        ),
        ScoringRule(
            rule_id="turnovers",
            name="Turnovers",
            points_per_unit=-1.0,
            weight=1.0
        )
    ]


async def demo_scoring_engine():
    """Demonstration of the scoring engine capabilities"""
    
    print("🏆 Fantasy Collective Scoring Engine Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    engine = ScoringEngine(enable_gpu=GPU_AVAILABLE)
    
    # Create sample league
    league_id = "nba_fantasy_2024"
    game_type = GameType.POINTS_BASED
    
    # Register scoring rules
    rules = create_sample_league_rules()
    engine.register_league_rules(league_id, game_type, rules)
    
    print(f"✅ Registered {len(rules)} scoring rules for league {league_id}")
    
    # Sample player performance
    player_stats = {
        "points": 25.0,
        "rebounds": 8.0,
        "assists": 6.0,
        "turnovers": 3.0
    }
    
    print(f"📊 Calculating score for player with stats: {player_stats}")
    
    # Calculate score
    performance = await engine.calculate_player_score(
        player_id="player_001",
        league_id=league_id,
        game_type=game_type,
        stats=player_stats
    )
    
    print(f"🎯 Final Score: {performance.final_score:.2f}")
    print(f"   - Base Score: {performance.base_score:.2f}")
    print(f"   - Bonus Score: {performance.bonus_score:.2f}")
    print(f"   - Multiplier: {performance.multiplier:.2f}")
    
    # Demonstrate ELO rating update
    print("\n⚡ ELO Rating System Demo")
    player1_rating, player2_rating = engine.update_elo_rating(
        "player_001", league_id, "player_002", 1.0  # Player 1 wins
    )
    
    print(f"Player 1 ELO: {player1_rating:.0f}")
    print(f"Player 2 ELO: {player2_rating:.0f}")
    
    # Batch scoring demo
    if GPU_AVAILABLE:
        print("\n🚀 GPU Batch Scoring Demo")
        
        batch_data = []
        for i in range(100):
            batch_data.append({
                "player_id": f"player_{i:03d}",
                "league_id": league_id,
                "game_type": game_type,
                "stats": {
                    "points": np.random.normal(20, 5),
                    "rebounds": np.random.normal(7, 3),
                    "assists": np.random.normal(5, 2),
                    "turnovers": np.random.normal(2.5, 1)
                },
                "use_cache": False
            })
        
        start_time = time.time()
        batch_results = await engine.batch_calculate_scores(batch_data, use_gpu=True)
        end_time = time.time()
        
        print(f"✅ Processed {len(batch_results)} players in {end_time - start_time:.3f} seconds")
        print(f"📈 Average score: {np.mean([r.final_score for r in batch_results]):.2f}")
    
    # Analytics demo
    print("\n📊 Player Analytics Demo")
    analytics = engine.get_player_analytics("player_001", league_id)
    print(f"Performance trend: {analytics['performance_trends']['trend']}")
    print(f"Games played: {analytics['performance_trends']['games']}")
    print(f"Cache hit rate: {analytics['cache_stats']['hit_rate']:.2%}")
    
    print("\n🎯 Demo completed successfully!")
    
    # Cleanup
    engine.shutdown()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_scoring_engine())