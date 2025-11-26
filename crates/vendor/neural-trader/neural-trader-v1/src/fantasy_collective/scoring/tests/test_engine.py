"""
Comprehensive tests for the Fantasy Collective Scoring Engine
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from ..engine import (
    ScoringEngine, GameType, ScoreType, ScoringRule, Achievement,
    PlayerPerformance, EloRating, FantasySportsScorer, 
    PredictionAccuracyScorer, CollectiveWisdomScorer,
    EloRatingSystem, AchievementEngine, GPUAccelerator,
    CacheManager, HistoricalPerformanceTracker
)
from ..config import DefaultConfigurations


class TestScoringRule:
    """Test ScoringRule dataclass"""
    
    def test_scoring_rule_creation(self):
        rule = ScoringRule(
            rule_id="points",
            name="Points Scored",
            points_per_unit=1.0,
            maximum_points=100.0,
            bonus_threshold=50.0,
            bonus_points=10.0
        )
        
        assert rule.rule_id == "points"
        assert rule.name == "Points Scored"
        assert rule.points_per_unit == 1.0
        assert rule.maximum_points == 100.0
        assert rule.bonus_threshold == 50.0
        assert rule.bonus_points == 10.0
        assert rule.active is True


class TestAchievement:
    """Test Achievement dataclass"""
    
    def test_achievement_creation(self):
        achievement = Achievement(
            achievement_id="perfect_game",
            name="Perfect Game",
            description="Score 100+ points",
            criteria={"min_score": 100},
            points_bonus=50.0,
            multiplier_bonus=1.1
        )
        
        assert achievement.achievement_id == "perfect_game"
        assert achievement.criteria["min_score"] == 100
        assert achievement.points_bonus == 50.0
        assert achievement.multiplier_bonus == 1.1


class TestGPUAccelerator:
    """Test GPU acceleration functionality"""
    
    def setup_method(self):
        self.gpu_accelerator = GPUAccelerator()
    
    def test_batch_calculate_scores_cpu(self):
        # Test CPU fallback
        stats_matrix = np.array([[10, 5], [20, 8], [15, 6]])
        scoring_rules = np.array([1.0, 2.0])
        
        scores = self.gpu_accelerator._cpu_batch_calculate_scores(stats_matrix, scoring_rules)
        
        expected = np.array([20.0, 36.0, 27.0])  # [10*1 + 5*2, 20*1 + 8*2, 15*1 + 6*2]
        np.testing.assert_array_equal(scores, expected)
    
    def test_batch_elo_updates_cpu(self):
        # Test CPU ELO updates
        ratings = np.array([[1500, 1400], [1600, 1500]])
        outcomes = np.array([1.0, 0.0])  # First player wins, second loses
        k_factors = np.array([32.0, 32.0])
        
        new_ratings = self.gpu_accelerator._cpu_batch_elo_updates(ratings, outcomes, k_factors)
        
        # Check that ratings changed appropriately
        assert new_ratings[0, 0] > 1500  # Winner's rating increased
        assert new_ratings[0, 1] < 1400  # Loser's rating decreased
        assert new_ratings[1, 0] < 1600  # Loser's rating decreased
        assert new_ratings[1, 1] > 1500  # Winner's rating increased


class TestCacheManager:
    """Test caching functionality"""
    
    def setup_method(self):
        self.cache_manager = CacheManager(memory_cache_size=10)
    
    def test_memory_cache_basic_operations(self):
        # Test set and get
        self.cache_manager.set("test_key", {"value": 42}, ttl=60)
        result = self.cache_manager.get("test_key", ttl=60)
        
        assert result is not None
        assert result["value"] == 42
    
    def test_memory_cache_expiration(self):
        # Test TTL expiration
        self.cache_manager.set("test_key", {"value": 42}, ttl=60)
        
        # Mock time to simulate expiration
        with patch('time.time', return_value=time.time() + 120):
            result = self.cache_manager.get("test_key", ttl=60)
            assert result is None
    
    def test_cache_size_limit(self):
        # Test cache size limits
        for i in range(15):  # More than cache_size
            self.cache_manager.set(f"key_{i}", {"value": i}, ttl=60)
        
        # Cache should not exceed size limit
        assert len(self.cache_manager.memory_cache) <= 10
    
    def test_cache_invalidation(self):
        # Test pattern-based invalidation
        self.cache_manager.set("user_1_score", {"score": 100}, ttl=60)
        self.cache_manager.set("user_2_score", {"score": 200}, ttl=60)
        self.cache_manager.set("league_settings", {"name": "test"}, ttl=60)
        
        self.cache_manager.invalidate("user")
        
        assert self.cache_manager.get("user_1_score", ttl=60) is None
        assert self.cache_manager.get("user_2_score", ttl=60) is None
        assert self.cache_manager.get("league_settings", ttl=60) is not None


class TestHistoricalPerformanceTracker:
    """Test historical performance tracking"""
    
    def setup_method(self):
        # Use in-memory database for testing
        self.tracker = HistoricalPerformanceTracker(":memory:")
    
    def test_record_and_retrieve_performance(self):
        performance = PlayerPerformance(
            player_id="player_1",
            league_id="league_1",
            game_type=GameType.POINTS_BASED,
            timestamp=datetime.now(),
            stats={"points": 25, "assists": 10},
            base_score=35.0,
            final_score=40.0
        )
        
        self.tracker.record_performance(performance)
        
        trends = self.tracker.get_performance_trends("player_1", "league_1", days=30)
        
        assert trends["games"] == 1
        assert trends["avg_score"] == 40.0
        assert trends["best_score"] == 40.0
    
    def test_elo_history_tracking(self):
        self.tracker.record_elo_change(
            "player_1", "league_1", 1500.0, 1520.0,
            "player_2", "win", 32.0
        )
        
        elo_history = self.tracker.get_elo_progression("player_1", "league_1", days=30)
        
        assert len(elo_history) == 1
        assert elo_history[0][1] == 1520.0  # New rating


class TestFantasySportsScorer:
    """Test fantasy sports scoring implementation"""
    
    def setup_method(self):
        self.scorer = FantasySportsScorer()
    
    def test_basic_scoring(self):
        stats = {"points": 25, "rebounds": 10, "assists": 8}
        rules = [
            ScoringRule("points", "Points", 1.0),
            ScoringRule("rebounds", "Rebounds", 1.2),
            ScoringRule("assists", "Assists", 1.5)
        ]
        
        score = self.scorer.calculate_score(stats, rules)
        expected = 25 * 1.0 + 10 * 1.2 + 8 * 1.5  # 25 + 12 + 12 = 49
        
        assert score == expected
    
    def test_bonus_threshold_scoring(self):
        stats = {"points": 50}
        rules = [
            ScoringRule("points", "Points", 1.0, bonus_threshold=40.0, bonus_points=10.0)
        ]
        
        score = self.scorer.calculate_score(stats, rules)
        expected = 50 * 1.0 + 10.0  # Base score + bonus
        
        assert score == expected
    
    def test_max_min_limits(self):
        stats = {"points": 100}
        rules = [
            ScoringRule("points", "Points", 1.0, maximum_points=50.0, minimum_points=10.0)
        ]
        
        score = self.scorer.calculate_score(stats, rules)
        assert score == 50.0  # Capped at maximum
        
        # Test minimum
        stats = {"points": 5}
        score = self.scorer.calculate_score(stats, rules)
        assert score == 10.0  # Raised to minimum
    
    def test_negative_scoring(self):
        stats = {"turnovers": 5}
        rules = [
            ScoringRule("turnovers", "Turnovers", -2.0)
        ]
        
        score = self.scorer.calculate_score(stats, rules)
        assert score == 0.0  # Minimum score is 0 (non-negative)


class TestPredictionAccuracyScorer:
    """Test prediction accuracy scoring"""
    
    def setup_method(self):
        self.scorer = PredictionAccuracyScorer(base_points=100.0)
    
    def test_perfect_accuracy_scoring(self):
        stats = {
            "accuracy": 1.0,
            "confidence": 0.8,
            "calibration": 1.0,
            "timeliness": 1.2
        }
        
        score = self.scorer.calculate_score(stats, [])
        
        # Should get high score for perfect accuracy with good calibration
        assert score > 100.0
    
    def test_poor_accuracy_scoring(self):
        stats = {
            "accuracy": 0.2,
            "confidence": 0.9,  # Overconfident
            "calibration": 0.1,
            "timeliness": 1.0
        }
        
        score = self.scorer.calculate_score(stats, [])
        
        # Should get low score for poor accuracy and overconfidence
        assert score < 50.0


class TestCollectiveWisdomScorer:
    """Test collective wisdom scoring"""
    
    def setup_method(self):
        self.scorer = CollectiveWisdomScorer()
    
    def test_collective_contribution_scoring(self):
        stats = {
            "individual_accuracy": 0.8,
            "group_accuracy_improvement": 0.1,
            "diversity_contribution": 0.6,
            "consensus_alignment": 0.7
        }
        
        score = self.scorer.calculate_score(stats, [])
        
        # Should reward both individual accuracy and group contribution
        assert score > 0
        
        # Test with different diversity levels
        stats["diversity_contribution"] = 0.9
        higher_diversity_score = self.scorer.calculate_score(stats, [])
        
        assert higher_diversity_score > score


class TestEloRatingSystem:
    """Test ELO rating system"""
    
    def setup_method(self):
        self.elo_system = EloRatingSystem()
    
    def test_expected_score_calculation(self):
        expected = self.elo_system.calculate_expected_score(1600, 1400)
        
        # Higher rated player should have higher expected score
        assert expected > 0.5
        
        # Test equal ratings
        expected_equal = self.elo_system.calculate_expected_score(1500, 1500)
        assert expected_equal == 0.5
    
    def test_rating_update(self):
        current_rating = 1500.0
        expected_score = 0.6
        actual_score = 1.0  # Won the match
        
        new_rating = self.elo_system.update_rating(current_rating, expected_score, actual_score)
        
        # Rating should increase when performing better than expected
        assert new_rating > current_rating
    
    def test_k_factor_calculation(self):
        # New player should have higher K-factor
        k_new = self.elo_system.get_k_factor(1500, 5)
        k_experienced = self.elo_system.get_k_factor(1500, 100)
        
        assert k_new > k_experienced
        
        # Low rated player should have higher K-factor
        k_low = self.elo_system.get_k_factor(1100, 50)
        k_high = self.elo_system.get_k_factor(2000, 50)
        
        assert k_low > k_high
    
    def test_batch_rating_updates(self):
        matches = [
            (1500, 1400, 1.0),  # First player wins
            (1600, 1500, 0.0),  # Second player wins
            (1450, 1550, 0.5)   # Draw
        ]
        
        results = self.elo_system.batch_update_ratings(matches)
        
        assert len(results) == 3
        
        # First match: winner's rating should increase
        assert results[0][0] > 1500
        assert results[0][1] < 1400
        
        # Second match: winner's rating should increase
        assert results[1][0] < 1600
        assert results[1][1] > 1500


class TestAchievementEngine:
    """Test achievement system"""
    
    def setup_method(self):
        self.achievement_engine = AchievementEngine()
        
        # Register test achievement
        self.test_achievement = Achievement(
            achievement_id="test_high_score",
            name="High Score",
            description="Score 100+ points",
            criteria={"min_score": 100},
            points_bonus=25.0,
            multiplier_bonus=1.1
        )
        self.achievement_engine.register_achievement(self.test_achievement)
    
    def test_achievement_registration(self):
        assert "test_high_score" in self.achievement_engine.achievements
        assert self.achievement_engine.achievements["test_high_score"] == self.test_achievement
    
    def test_achievement_earning(self):
        performance = PlayerPerformance(
            player_id="player_1",
            league_id="league_1",
            game_type=GameType.POINTS_BASED,
            timestamp=datetime.now(),
            stats={"points": 150},
            final_score=150.0
        )
        
        earned = self.achievement_engine.check_achievements("player_1", performance, {})
        
        assert len(earned) == 1
        assert earned[0][0] == self.test_achievement
        assert earned[0][1] == 25.0
    
    def test_one_time_achievement(self):
        performance = PlayerPerformance(
            player_id="player_1",
            league_id="league_1",
            game_type=GameType.POINTS_BASED,
            timestamp=datetime.now(),
            stats={"points": 150},
            final_score=150.0
        )
        
        # First time should earn achievement
        earned = self.achievement_engine.check_achievements("player_1", performance, {})
        assert len(earned) == 1
        
        # Second time should not earn it again (one_time_only=True)
        earned = self.achievement_engine.check_achievements("player_1", performance, {})
        assert len(earned) == 0
    
    def test_achievement_multiplier(self):
        # Award achievement to player
        self.achievement_engine.player_achievements["player_1"].add("test_high_score")
        
        multiplier = self.achievement_engine.get_achievement_multiplier("player_1", 100.0)
        
        # Should include the achievement multiplier bonus
        assert multiplier > 1.0


class TestScoringEngine:
    """Test main scoring engine"""
    
    def setup_method(self):
        # Use testing configuration
        config = DefaultConfigurations.testing_config()
        self.engine = ScoringEngine(
            redis_url=None,
            db_path=":memory:",
            enable_gpu=False
        )
        
        # Register test scoring rules
        self.test_rules = [
            ScoringRule("points", "Points", 1.0),
            ScoringRule("assists", "Assists", 1.5),
            ScoringRule("turnovers", "Turnovers", -1.0)
        ]
        
        self.engine.register_league_rules("test_league", GameType.POINTS_BASED, self.test_rules)
    
    def teardown_method(self):
        self.engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_calculate_player_score(self):
        stats = {"points": 25, "assists": 10, "turnovers": 3}
        
        performance = await self.engine.calculate_player_score(
            "player_1", "test_league", GameType.POINTS_BASED, stats
        )
        
        # Expected: 25*1.0 + 10*1.5 + 3*(-1.0) = 25 + 15 - 3 = 37
        assert performance.base_score == 37.0
        assert performance.player_id == "player_1"
        assert performance.league_id == "test_league"
        assert performance.stats == stats
    
    @pytest.mark.asyncio
    async def test_batch_score_calculation(self):
        batch_data = []
        for i in range(5):
            batch_data.append({
                "player_id": f"player_{i}",
                "league_id": "test_league",
                "game_type": GameType.POINTS_BASED,
                "stats": {"points": 20 + i, "assists": 5 + i, "turnovers": 2},
                "use_cache": False
            })
        
        results = await self.engine.batch_calculate_scores(batch_data, use_gpu=False)
        
        assert len(results) == 5
        
        # Check that scores are calculated correctly
        for i, result in enumerate(results):
            expected_base = (20 + i) * 1.0 + (5 + i) * 1.5 + 2 * (-1.0)
            assert result.base_score == expected_base
    
    def test_elo_rating_update(self):
        player1_rating, player2_rating = self.engine.update_elo_rating(
            "player_1", "test_league", "player_2", 1.0
        )
        
        # Winner should have higher rating, loser should have lower
        assert player1_rating > 1500  # Initial rating
        assert player2_rating < 1500
        
        # Check that ratings are stored
        player1_elo = self.engine.elo_ratings[("player_1", "test_league")]
        player2_elo = self.engine.elo_ratings[("player_2", "test_league")]
        
        assert player1_elo.wins == 1
        assert player1_elo.losses == 0
        assert player2_elo.wins == 0
        assert player2_elo.losses == 1
    
    @pytest.mark.asyncio
    async def test_real_time_score_update(self):
        # First, calculate initial score
        initial_stats = {"points": 20, "assists": 5, "turnovers": 2}
        await self.engine.calculate_player_score(
            "player_1", "test_league", GameType.POINTS_BASED, initial_stats
        )
        
        # Store in cache for real-time updates
        cache_key = f"live_score_player_1_test_league"
        self.engine.cache_manager.set(cache_key, {
            "player_id": "player_1",
            "league_id": "test_league",
            "game_type": GameType.POINTS_BASED.value,
            "timestamp": datetime.now().isoformat(),
            "stats": initial_stats,
            "base_score": 33.0,  # 20 + 5*1.5 - 2 = 33
            "bonus_score": 0.0,
            "multiplier": 1.0,
            "final_score": 33.0,
            "metadata": {}
        })
        
        # Update with new stats
        stat_update = {"points": 5, "assists": 2}  # Additional stats
        
        updated_score = await self.engine.real_time_score_update(
            "player_1", "test_league", stat_update
        )
        
        # Should reflect the updated stats
        assert updated_score > 33.0
    
    def test_player_analytics(self):
        # Create some performance history first
        performance = PlayerPerformance(
            player_id="player_1",
            league_id="test_league",
            game_type=GameType.POINTS_BASED,
            timestamp=datetime.now(),
            stats={"points": 25},
            base_score=25.0,
            final_score=25.0
        )
        
        self.engine.performance_tracker.record_performance(performance)
        
        analytics = self.engine.get_player_analytics("player_1", "test_league")
        
        assert "performance_trends" in analytics
        assert "elo_rating" in analytics
        assert "achievements" in analytics
        assert analytics["player_id"] == "player_1"
        assert analytics["league_id"] == "test_league"
    
    def test_export_league_data(self):
        export_data = self.engine.export_league_data("test_league", format="json")
        
        assert isinstance(export_data, str)
        
        # Should be valid JSON
        import json
        parsed = json.loads(export_data)
        
        assert "league_id" in parsed
        assert "scoring_rules" in parsed
        assert parsed["league_id"] == "test_league"


class TestIntegration:
    """Integration tests for the complete scoring system"""
    
    def setup_method(self):
        self.engine = ScoringEngine(redis_url=None, db_path=":memory:", enable_gpu=False)
        
        # Set up NBA-style league
        nba_rules = [
            ScoringRule("points", "Points", 1.0),
            ScoringRule("rebounds", "Rebounds", 1.2),
            ScoringRule("assists", "Assists", 1.5),
            ScoringRule("steals", "Steals", 3.0),
            ScoringRule("blocks", "Blocks", 3.0),
            ScoringRule("turnovers", "Turnovers", -1.0),
            ScoringRule("double_double", "Double Double", 0.0, bonus_points=5.0, bonus_threshold=1.0)
        ]
        
        self.engine.register_league_rules("nba_league", GameType.POINTS_BASED, nba_rules)
    
    def teardown_method(self):
        self.engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_complete_game_simulation(self):
        """Test a complete game simulation with multiple players"""
        
        # Player stats for a simulated game
        game_stats = {
            "lebron": {"points": 28, "rebounds": 8, "assists": 9, "steals": 2, "blocks": 1, "turnovers": 4, "double_double": 1},
            "curry": {"points": 32, "rebounds": 4, "assists": 8, "steals": 1, "blocks": 0, "turnovers": 3, "double_double": 0},
            "giannis": {"points": 35, "rebounds": 12, "assists": 6, "steals": 1, "blocks": 2, "turnovers": 2, "double_double": 1},
            "harden": {"points": 24, "rebounds": 6, "assists": 11, "steals": 2, "blocks": 0, "turnovers": 5, "double_double": 1}
        }
        
        # Calculate scores for all players
        performances = {}
        for player, stats in game_stats.items():
            performance = await self.engine.calculate_player_score(
                player, "nba_league", GameType.POINTS_BASED, stats
            )
            performances[player] = performance
        
        # Verify scores are reasonable
        assert all(p.final_score > 0 for p in performances.values())
        
        # Giannis should have highest score (highest points + rebounds + double-double bonus)
        giannis_score = performances["giannis"].final_score
        assert all(giannis_score >= p.final_score for p in performances.values())
        
        # Verify double-double bonus is applied
        lebron_performance = performances["lebron"]
        assert lebron_performance.bonus_score >= 5.0  # Should get double-double bonus
        
        # Create head-to-head matchups and update ELO ratings
        matchups = [
            ("lebron", "curry", 1.0),  # LeBron wins
            ("giannis", "harden", 1.0),  # Giannis wins
        ]
        
        elo_results = {}
        for p1, p2, outcome in matchups:
            p1_rating, p2_rating = self.engine.update_elo_rating(p1, "nba_league", p2, outcome)
            elo_results[p1] = p1_rating
            elo_results[p2] = p2_rating
        
        # Winners should have increased ratings
        assert elo_results["lebron"] > 1500
        assert elo_results["giannis"] > 1500
        assert elo_results["curry"] < 1500
        assert elo_results["harden"] < 1500
        
        # Check analytics for one player
        analytics = self.engine.get_player_analytics("lebron", "nba_league")
        
        assert analytics["performance_trends"]["games"] == 1
        assert analytics["elo_rating"]["current"] == elo_results["lebron"]
        assert analytics["elo_rating"]["games_played"] == 1
        assert analytics["elo_rating"]["win_rate"] == 1.0


@pytest.mark.asyncio
async def test_performance_under_load():
    """Test scoring engine performance with high load"""
    
    engine = ScoringEngine(redis_url=None, db_path=":memory:", enable_gpu=False)
    
    # Set up league
    rules = [
        ScoringRule("stat1", "Stat 1", 1.0),
        ScoringRule("stat2", "Stat 2", 2.0),
        ScoringRule("stat3", "Stat 3", 0.5)
    ]
    engine.register_league_rules("load_test_league", GameType.POINTS_BASED, rules)
    
    # Generate large batch of player data
    import time
    batch_size = 1000
    batch_data = []
    
    for i in range(batch_size):
        batch_data.append({
            "player_id": f"player_{i}",
            "league_id": "load_test_league",
            "game_type": GameType.POINTS_BASED,
            "stats": {
                "stat1": np.random.randint(10, 50),
                "stat2": np.random.randint(5, 20),
                "stat3": np.random.randint(0, 30)
            },
            "use_cache": False
        })
    
    # Measure performance
    start_time = time.time()
    results = await engine.batch_calculate_scores(batch_data, use_gpu=False)
    end_time = time.time()
    
    processing_time = end_time - start_time
    throughput = len(results) / processing_time
    
    # Verify results
    assert len(results) == batch_size
    assert all(r.final_score >= 0 for r in results)
    
    # Performance should be reasonable (adjust threshold as needed)
    assert throughput > 100  # At least 100 calculations per second
    
    print(f"Processed {batch_size} players in {processing_time:.3f}s ({throughput:.1f} players/sec)")
    
    engine.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])