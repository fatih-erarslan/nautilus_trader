#!/usr/bin/env python3
"""
Fantasy Collective Scoring Engine - Comprehensive Demo

This demo showcases all the key features of the scoring engine:
- Multiple game types and scoring systems
- Real-time score calculation with caching
- ELO rating system with competitive matches
- Achievement system with bonuses
- GPU-accelerated batch processing
- Historical performance tracking
- Advanced analytics and metrics
"""

import asyncio
import numpy as np
import time
from datetime import datetime
from typing import Dict, List

from fantasy_collective.scoring import (
    ScoringEngine, GameType, ScoreType, ScoringRule, Achievement,
    PlayerPerformance
)
from fantasy_collective.scoring.algorithms import AdvancedScoringAlgorithms
from fantasy_collective.scoring.config import LeagueTemplates


class ComprehensiveDemo:
    
    def __init__(self):
        self.engine = ScoringEngine(enable_gpu=False, redis_url=None, db_path=":memory:")
        
    async def run_complete_demo(self):
        """Run the complete demonstration"""
        
        print("🚀 Fantasy Collective Scoring Engine - Comprehensive Demo")
        print("=" * 70)
        
        # Demo 1: NBA Fantasy League
        await self.demo_nba_fantasy()
        
        # Demo 2: Prediction Market
        await self.demo_prediction_market()
        
        # Demo 3: Tournament Bracket
        self.demo_tournament_bracket()
        
        # Demo 4: Batch Processing Performance
        await self.demo_batch_processing()
        
        # Demo 5: Achievement System
        await self.demo_achievement_system()
        
        # Demo 6: ELO Rating Tournaments
        await self.demo_elo_tournament()
        
        # Demo 7: Advanced Analytics
        await self.demo_advanced_analytics()
        
        print("\n🎉 Complete demo finished successfully!")
        self.engine.shutdown()
    
    async def demo_nba_fantasy(self):
        """Demo 1: NBA Fantasy League with realistic player stats"""
        
        print("\n🏀 Demo 1: NBA Fantasy League")
        print("-" * 40)
        
        # Setup NBA league with comprehensive rules
        nba_rules = [
            ScoringRule("points", "Points", 1.0),
            ScoringRule("rebounds", "Rebounds", 1.2),
            ScoringRule("assists", "Assists", 1.5),
            ScoringRule("steals", "Steals", 3.0),
            ScoringRule("blocks", "Blocks", 3.0),
            ScoringRule("turnovers", "Turnovers", -1.0),
            ScoringRule("three_pointers", "3-Pointers", 3.0),
            ScoringRule("double_double", "Double Double", 0.0, bonus_points=5.0, bonus_threshold=1.0),
            ScoringRule("triple_double", "Triple Double", 0.0, bonus_points=15.0, bonus_threshold=1.0)
        ]
        
        self.engine.register_league_rules("nba_2024", GameType.POINTS_BASED, nba_rules)
        
        # Realistic NBA player performances
        players = {
            "LeBron James": {
                "points": 28, "rebounds": 8, "assists": 9, "steals": 2, "blocks": 1,
                "turnovers": 4, "three_pointers": 2, "double_double": 1, "triple_double": 0
            },
            "Stephen Curry": {
                "points": 32, "rebounds": 4, "assists": 8, "steals": 1, "blocks": 0,
                "turnovers": 3, "three_pointers": 8, "double_double": 0, "triple_double": 0
            },
            "Giannis Antetokounmpo": {
                "points": 35, "rebounds": 12, "assists": 6, "steals": 1, "blocks": 2,
                "turnovers": 2, "three_pointers": 1, "double_double": 1, "triple_double": 0
            },
            "Luka Dončić": {
                "points": 30, "rebounds": 8, "assists": 12, "steals": 1, "blocks": 0,
                "turnovers": 5, "three_pointers": 4, "double_double": 1, "triple_double": 1
            }
        }
        
        performances = {}
        
        print("Player Performances:")
        for player, stats in players.items():
            performance = await self.engine.calculate_player_score(
                player.lower().replace(" ", "_"), "nba_2024", GameType.POINTS_BASED, stats
            )
            performances[player] = performance
            
            print(f"\n{player}:")
            print(f"  Stats: Pts:{stats['points']}, Reb:{stats['rebounds']}, "
                  f"Ast:{stats['assists']}, Stl:{stats['steals']}, Blk:{stats['blocks']}")
            print(f"  Base Score: {performance.base_score:.1f}")
            print(f"  Bonus Score: {performance.bonus_score:.1f}")
            print(f"  Final Score: {performance.final_score:.1f}")
        
        # Rank players
        ranked = sorted(performances.items(), key=lambda x: x[1].final_score, reverse=True)
        
        print("\n🏆 Fantasy Rankings:")
        for i, (player, perf) in enumerate(ranked, 1):
            print(f"{i}. {player}: {perf.final_score:.1f} points")
    
    async def demo_prediction_market(self):
        """Demo 2: Prediction Market Accuracy Scoring"""
        
        print("\n🔮 Demo 2: Prediction Market Scoring")
        print("-" * 40)
        
        # Setup prediction market rules
        prediction_rules = [
            ScoringRule("accuracy", "Prediction Accuracy", 100.0),
            ScoringRule("calibration", "Calibration Score", 50.0),
            ScoringRule("brier_score", "Brier Score", -25.0),
            ScoringRule("confidence_bonus", "Confidence Bonus", 30.0)
        ]
        
        self.engine.register_league_rules("election_2024", GameType.PREDICTION_MARKET, prediction_rules)
        
        # Different types of predictors
        predictors = {
            "Expert Analyst": {
                "accuracy": 0.85, "calibration": 0.90, "brier_score": 0.15, "confidence_bonus": 0.8
            },
            "Data Scientist": {
                "accuracy": 0.78, "calibration": 0.95, "brier_score": 0.22, "confidence_bonus": 0.85
            },
            "Political Insider": {
                "accuracy": 0.82, "calibration": 0.75, "brier_score": 0.18, "confidence_bonus": 0.7
            },
            "Casual Predictor": {
                "accuracy": 0.62, "calibration": 0.45, "brier_score": 0.38, "confidence_bonus": 0.4
            }
        }
        
        print("Prediction Accuracy Results:")
        pred_performances = {}
        
        for predictor, stats in predictors.items():
            performance = await self.engine.calculate_player_score(
                predictor.lower().replace(" ", "_"), "election_2024", 
                GameType.PREDICTION_MARKET, stats, ScoreType.PREDICTION_ACCURACY
            )
            pred_performances[predictor] = performance
            
            print(f"\n{predictor}:")
            print(f"  Accuracy: {stats['accuracy']:.1%}")
            print(f"  Calibration: {stats['calibration']:.2f}")
            print(f"  Final Score: {performance.final_score:.1f}")
        
        # Rank predictors
        ranked_predictors = sorted(pred_performances.items(), 
                                 key=lambda x: x[1].final_score, reverse=True)
        
        print("\n🎯 Predictor Leaderboard:")
        for i, (predictor, perf) in enumerate(ranked_predictors, 1):
            print(f"{i}. {predictor}: {perf.final_score:.1f} points")
    
    def demo_tournament_bracket(self):
        """Demo 3: Tournament Bracket Scoring"""
        
        print("\n🏆 Demo 3: March Madness Bracket")
        print("-" * 40)
        
        # Sample NCAA tournament bracket
        my_bracket = {
            "round_1": "Duke",
            "round_2": "Duke",
            "sweet_16": "Duke",
            "elite_8": "Duke",
            "final_4": "UConn",
            "championship": "UConn"
        }
        
        actual_results = {
            "round_1": "Duke",      # ✓ Correct (1 pt)
            "round_2": "Duke",      # ✓ Correct (2 pts)
            "sweet_16": "UNC",      # ✗ Wrong (0 pts)
            "elite_8": "UConn",     # ✗ Wrong (0 pts) 
            "final_4": "UConn",     # ✓ Correct (16 pts)
            "championship": "UConn"  # ✓ Correct (32 pts)
        }
        
        scores = AdvancedScoringAlgorithms.tournament_bracket_scoring(
            my_bracket, actual_results
        )
        
        print("Bracket Performance:")
        print(f"Total Score: {scores['total']} points")
        print(f"Max Possible: {scores['max_possible']} points")
        print(f"Bracket Accuracy: {scores['percentage']:.1%}")
        
        print("\nRound-by-Round Breakdown:")
        for round_name, points in scores['rounds'].items():
            predicted = my_bracket.get(round_name, "N/A")
            actual = actual_results.get(round_name, "N/A")
            status = "✓" if points > 0 else "✗"
            print(f"  {round_name}: {predicted} vs {actual} {status} ({points} pts)")
    
    async def demo_batch_processing(self):
        """Demo 4: High-Performance Batch Processing"""
        
        print("\n🚀 Demo 4: Batch Processing Performance")
        print("-" * 40)
        
        # Setup league for batch testing
        batch_rules = [
            ScoringRule("stat1", "Stat 1", 1.0),
            ScoringRule("stat2", "Stat 2", 2.0),
            ScoringRule("stat3", "Stat 3", 0.5),
            ScoringRule("stat4", "Stat 4", 1.5)
        ]
        
        self.engine.register_league_rules("batch_test", GameType.POINTS_BASED, batch_rules)
        
        # Generate large batch of players
        batch_size = 1000
        batch_data = []
        
        print(f"Generating {batch_size} player performances...")
        
        for i in range(batch_size):
            batch_data.append({
                "player_id": f"player_{i:04d}",
                "league_id": "batch_test", 
                "game_type": GameType.POINTS_BASED,
                "stats": {
                    "stat1": np.random.randint(10, 50),
                    "stat2": np.random.randint(5, 25),
                    "stat3": np.random.randint(0, 30),
                    "stat4": np.random.randint(3, 15)
                },
                "use_cache": False
            })
        
        # Measure batch processing performance
        start_time = time.time()
        results = await self.engine.batch_calculate_scores(batch_data, use_gpu=False)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(results) / processing_time
        
        print(f"\nBatch Processing Results:")
        print(f"Players Processed: {len(results)}")
        print(f"Processing Time: {processing_time:.3f} seconds")
        print(f"Throughput: {throughput:.1f} calculations/second")
        print(f"Average Score: {np.mean([r.final_score for r in results]):.1f}")
        print(f"Score Range: {np.min([r.final_score for r in results]):.1f} - {np.max([r.final_score for r in results]):.1f}")
    
    async def demo_achievement_system(self):
        """Demo 5: Achievement System with Bonuses"""
        
        print("\n🏅 Demo 5: Achievement System")
        print("-" * 40)
        
        # Register custom achievements
        achievements = [
            Achievement(
                achievement_id="high_scorer",
                name="High Scorer",
                description="Score 100+ points in a game",
                criteria={"min_score": 100},
                points_bonus=25.0,
                category="performance"
            ),
            Achievement(
                achievement_id="consistency_king",
                name="Consistency King", 
                description="Maintain high consistency",
                criteria={"consistency_threshold": 0.8},
                points_bonus=40.0,
                multiplier_bonus=1.1,
                category="consistency"
            )
        ]
        
        for achievement in achievements:
            self.engine.achievement_engine.register_achievement(achievement)
        
        # Setup achievement test league
        achievement_rules = [
            ScoringRule("performance", "Performance", 1.0),
            ScoringRule("bonus_stat", "Bonus Stat", 2.0)
        ]
        
        self.engine.register_league_rules("achievement_test", GameType.POINTS_BASED, achievement_rules)
        
        # Test different performance levels
        test_players = {
            "Superstar Player": {"performance": 120, "bonus_stat": 15},  # Should trigger high_scorer
            "Consistent Player": {"performance": 85, "bonus_stat": 10},
            "Average Player": {"performance": 65, "bonus_stat": 8}
        }
        
        print("Testing Achievement System:")
        
        for player, stats in test_players.items():
            performance = await self.engine.calculate_player_score(
                player.lower().replace(" ", "_"), "achievement_test", 
                GameType.POINTS_BASED, stats
            )
            
            print(f"\n{player}:")
            print(f"  Base Score: {performance.base_score:.1f}")
            print(f"  Bonus Score: {performance.bonus_score:.1f}")
            print(f"  Multiplier: {performance.multiplier:.2f}x")
            print(f"  Final Score: {performance.final_score:.1f}")
            
            # Check earned achievements
            player_achievements = self.engine.achievement_engine.player_achievements.get(
                player.lower().replace(" ", "_"), set()
            )
            
            if player_achievements:
                print(f"  Achievements: {list(player_achievements)}")
            else:
                print("  No achievements earned")
    
    async def demo_elo_tournament(self):
        """Demo 6: ELO Rating Tournament"""
        
        print("\n⚡ Demo 6: ELO Rating Tournament")
        print("-" * 40)
        
        # Tournament participants
        players = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        
        print("Initial ELO Ratings: 1500 (all players)")
        
        # Tournament matches (round robin style)
        matches = [
            ("Alice", "Bob", 1.0),      # Alice wins
            ("Charlie", "Diana", 0.0),  # Diana wins
            ("Eve", "Frank", 0.5),      # Draw
            ("Alice", "Charlie", 1.0),  # Alice wins
            ("Bob", "Diana", 1.0),      # Bob wins
            ("Eve", "Alice", 0.0),      # Alice wins
            ("Frank", "Bob", 1.0),      # Frank wins
            ("Diana", "Alice", 0.0),    # Alice wins
            ("Charlie", "Eve", 1.0),    # Charlie wins
            ("Bob", "Eve", 0.0),        # Eve wins
        ]
        
        print(f"\nTournament Results ({len(matches)} matches):")
        
        # Process each match
        for i, (p1, p2, outcome) in enumerate(matches, 1):
            p1_rating, p2_rating = self.engine.update_elo_rating(
                p1.lower(), "tournament", p2.lower(), outcome
            )
            
            result = "wins" if outcome == 1.0 else "loses" if outcome == 0.0 else "draws with"
            print(f"Match {i}: {p1} {result} {p2}")
            print(f"  {p1}: {p1_rating:.0f}, {p2}: {p2_rating:.0f}")
        
        # Final tournament standings
        print("\n🏆 Final Tournament Standings:")
        
        final_ratings = []
        for player in players:
            player_key = (player.lower(), "tournament")
            if player_key in self.engine.elo_ratings:
                elo_data = self.engine.elo_ratings[player_key]
                final_ratings.append((player, elo_data.rating, elo_data.wins, elo_data.losses, elo_data.draws))
            else:
                final_ratings.append((player, 1500, 0, 0, 0))
        
        # Sort by rating
        final_ratings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (player, rating, wins, losses, draws) in enumerate(final_ratings, 1):
            win_rate = wins / max(wins + losses + draws, 1) * 100
            print(f"{i}. {player}: {rating:.0f} ELO ({wins}W-{losses}L-{draws}D, {win_rate:.1f}%)")
    
    async def demo_advanced_analytics(self):
        """Demo 7: Advanced Analytics and Performance Metrics"""
        
        print("\n📊 Demo 7: Advanced Analytics")
        print("-" * 40)
        
        # Get analytics for a player from previous demos
        test_player = "lebron_james"
        analytics = self.engine.get_player_analytics(test_player, "nba_2024")
        
        print(f"Player Analytics for {test_player}:")
        print(f"  Performance Trend: {analytics['performance_trends'].get('trend', 'N/A')}")
        print(f"  Games Played: {analytics['performance_trends'].get('games', 0)}")
        print(f"  Average Score: {analytics['performance_trends'].get('avg_score', 0):.1f}")
        print(f"  Best Score: {analytics['performance_trends'].get('best_score', 0):.1f}")
        print(f"  Consistency: {analytics['performance_trends'].get('consistency', 0):.2f}")
        
        print(f"\nELO Rating Info:")
        print(f"  Current Rating: {analytics['elo_rating']['current']:.0f}")
        print(f"  Peak Rating: {analytics['elo_rating']['peak']:.0f}")
        print(f"  Games Played: {analytics['elo_rating']['games_played']}")
        print(f"  Win Rate: {analytics['elo_rating']['win_rate']:.1%}")
        
        print(f"\nCache Performance:")
        cache_stats = analytics['cache_stats']
        print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
        print(f"  Total Hits: {cache_stats['hit_count']}")
        print(f"  Total Misses: {cache_stats['miss_count']}")
        
        # Demonstrate advanced scoring algorithms
        print(f"\nAdvanced Scoring Examples:")
        
        # Sample prediction accuracy analysis
        predictions = [0.8, 0.6, 0.9, 0.3, 0.7]
        actuals = [1.0, 0.0, 1.0, 0.0, 1.0]
        confidence = [0.9, 0.7, 0.95, 0.6, 0.8]
        
        accuracy_metrics = AdvancedScoringAlgorithms.weighted_accuracy_score(
            predictions, actuals, confidence_scores=confidence
        )
        
        print(f"  Prediction Accuracy Analysis:")
        print(f"    Hit Rate: {accuracy_metrics['hit_rate']:.1%}")
        print(f"    Calibration Score: {accuracy_metrics['calibration_score']:.2f}")
        print(f"    Weighted MAE: {accuracy_metrics['weighted_mae']:.3f}")


async def main():
    """Main demo execution"""
    demo = ComprehensiveDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Configure logging for cleaner output
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    # Run the comprehensive demo
    asyncio.run(main())