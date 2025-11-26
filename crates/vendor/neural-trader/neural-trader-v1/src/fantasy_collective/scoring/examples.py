"""
Example Usage of Fantasy Collective Scoring Engine

This file demonstrates various use cases and capabilities of the scoring system.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from .engine import (
    ScoringEngine, GameType, ScoreType, ScoringRule, Achievement,
    PlayerPerformance, EloRating
)
from .algorithms import AdvancedScoringAlgorithms, AdvancedMetrics
from .config import (
    DefaultConfigurations, LeagueTemplates, ConfigurationManager
)


class ExampleLeagues:
    """Example league implementations"""
    
    @staticmethod
    async def nba_fantasy_league_example():
        """Complete NBA fantasy league example"""
        print("🏀 NBA Fantasy League Example")
        print("=" * 50)
        
        # Initialize scoring engine
        engine = ScoringEngine(enable_gpu=False)
        
        # Set up NBA fantasy rules
        nba_rules = LeagueTemplates.nba_fantasy_rules()
        engine.register_league_rules("nba_2024", GameType.POINTS_BASED, nba_rules)
        
        # Sample player performances
        players_stats = {
            "lebron_james": {
                "points": 28, "rebounds": 8, "assists": 9, "steals": 2, "blocks": 1,
                "turnovers": 4, "three_pointers": 2, "field_goals_made": 12,
                "field_goals_missed": 8, "free_throws_made": 2, "free_throws_missed": 1,
                "double_double": 1, "triple_double": 0
            },
            "stephen_curry": {
                "points": 32, "rebounds": 4, "assists": 8, "steals": 1, "blocks": 0,
                "turnovers": 3, "three_pointers": 8, "field_goals_made": 11,
                "field_goals_missed": 12, "free_throws_made": 2, "free_throws_missed": 0,
                "double_double": 0, "triple_double": 0
            },
            "giannis_antetokounmpo": {
                "points": 35, "rebounds": 12, "assists": 6, "steals": 1, "blocks": 2,
                "turnovers": 2, "three_pointers": 1, "field_goals_made": 15,
                "field_goals_missed": 7, "free_throws_made": 4, "free_throws_missed": 2,
                "double_double": 1, "triple_double": 0
            }
        }
        
        # Calculate scores for all players
        performances = {}
        for player, stats in players_stats.items():
            performance = await engine.calculate_player_score(
                player, "nba_2024", GameType.POINTS_BASED, stats
            )
            performances[player] = performance
            
            print(f"\n{player.replace('_', ' ').title()}:")
            print(f"  Stats: {stats}")
            print(f"  Base Score: {performance.base_score:.2f}")
            print(f"  Bonus Score: {performance.bonus_score:.2f}")
            print(f"  Final Score: {performance.final_score:.2f}")
        
        # Rank players by performance
        ranked_players = sorted(performances.items(), 
                              key=lambda x: x[1].final_score, reverse=True)
        
        print("\n🏆 Player Rankings:")
        for i, (player, performance) in enumerate(ranked_players, 1):
            print(f"{i}. {player.replace('_', ' ').title()}: {performance.final_score:.2f}")
        
        # Update ELO ratings for head-to-head matchups
        print("\n⚡ ELO Rating Updates:")
        matchups = [
            ("giannis_antetokounmpo", "lebron_james", 1.0),  # Giannis wins
            ("stephen_curry", "giannis_antetokounmpo", 0.0),  # Giannis wins
            ("lebron_james", "stephen_curry", 1.0)  # LeBron wins
        ]
        
        for p1, p2, outcome in matchups:
            p1_rating, p2_rating = engine.update_elo_rating(p1, "nba_2024", p2, outcome)
            winner = p1 if outcome == 1.0 else p2 if outcome == 0.0 else "Draw"
            print(f"{p1.replace('_', ' ').title()} vs {p2.replace('_', ' ').title()}: {winner}")
            print(f"  {p1.replace('_', ' ').title()}: {p1_rating:.0f}")
            print(f"  {p2.replace('_', ' ').title()}: {p2_rating:.0f}")
        
        engine.shutdown()
        print("\n✅ NBA Fantasy League example completed!")
    
    @staticmethod
    async def prediction_market_example():
        """Prediction market scoring example"""
        print("\n🔮 Prediction Market Example")
        print("=" * 50)
        
        engine = ScoringEngine(enable_gpu=False)
        
        # Set up prediction market rules
        prediction_rules = LeagueTemplates.prediction_market_rules()
        engine.register_league_rules("election_2024", GameType.PREDICTION_MARKET, prediction_rules)
        
        # Sample predictions with different accuracy levels
        predictors = {
            "political_analyst": {
                "accuracy": 0.85, "calibration": 0.9, "brier_score": 0.15,
                "log_score": 0.8, "early_prediction_bonus": 0.9, "confidence_accuracy": 0.82
            },
            "data_scientist": {
                "accuracy": 0.78, "calibration": 0.95, "brier_score": 0.22,
                "log_score": 0.7, "early_prediction_bonus": 0.6, "confidence_accuracy": 0.88
            },
            "casual_predictor": {
                "accuracy": 0.62, "calibration": 0.4, "brier_score": 0.38,
                "log_score": 0.3, "early_prediction_bonus": 0.3, "confidence_accuracy": 0.45
            }
        }
        
        print("\nPrediction Accuracy Scores:")
        performances = {}
        for predictor, stats in predictors.items():
            performance = await engine.calculate_player_score(
                predictor, "election_2024", GameType.PREDICTION_MARKET, 
                stats, ScoreType.PREDICTION_ACCURACY
            )
            performances[predictor] = performance
            
            print(f"\n{predictor.replace('_', ' ').title()}:")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Calibration: {stats['calibration']:.2f}")
            print(f"  Final Score: {performance.final_score:.2f}")
        
        # Rank predictors
        ranked = sorted(performances.items(), key=lambda x: x[1].final_score, reverse=True)
        print("\n🎯 Predictor Rankings:")
        for i, (predictor, performance) in enumerate(ranked, 1):
            print(f"{i}. {predictor.replace('_', ' ').title()}: {performance.final_score:.2f}")
        
        engine.shutdown()
        print("\n✅ Prediction Market example completed!")
    
    @staticmethod
    async def collective_wisdom_example():
        """Collective wisdom scoring example"""
        print("\n🧠 Collective Wisdom Example")
        print("=" * 50)
        
        engine = ScoringEngine(enable_gpu=False)
        
        # Set up collective wisdom rules
        wisdom_rules = LeagueTemplates.collective_wisdom_rules()
        engine.register_league_rules("climate_forecasting", GameType.COLLECTIVE_WISDOM, wisdom_rules)
        
        # Sample collective intelligence contributions
        contributors = {
            "climate_scientist": {
                "individual_accuracy": 0.88, "group_contribution": 0.15,
                "diversity_index": 0.6, "consensus_quality": 0.85,
                "novel_insights": 1.0, "meta_prediction": 0.82
            },
            "data_modeler": {
                "individual_accuracy": 0.75, "group_contribution": 0.25,
                "diversity_index": 0.8, "consensus_quality": 0.7,
                "novel_insights": 0.0, "meta_prediction": 0.78
            },
            "policy_expert": {
                "individual_accuracy": 0.70, "group_contribution": 0.20,
                "diversity_index": 0.9, "consensus_quality": 0.6,
                "novel_insights": 1.0, "meta_prediction": 0.65
            }
        }
        
        print("\nCollective Wisdom Contributions:")
        performances = {}
        for contributor, stats in contributors.items():
            performance = await engine.calculate_player_score(
                contributor, "climate_forecasting", GameType.COLLECTIVE_WISDOM,
                stats, ScoreType.COLLECTIVE_WISDOM
            )
            performances[contributor] = performance
            
            print(f"\n{contributor.replace('_', ' ').title()}:")
            print(f"  Individual Accuracy: {stats['individual_accuracy']:.2%}")
            print(f"  Group Contribution: {stats['group_contribution']:.2%}")
            print(f"  Diversity Index: {stats['diversity_index']:.2f}")
            print(f"  Final Score: {performance.final_score:.2f}")
        
        engine.shutdown()
        print("\n✅ Collective Wisdom example completed!")


class AdvancedExamples:
    """Advanced scoring algorithm examples"""
    
    @staticmethod
    def tournament_bracket_example():
        """Tournament bracket scoring example"""
        print("\n🏆 Tournament Bracket Scoring Example")
        print("=" * 50)
        
        # Sample NCAA tournament bracket predictions vs actual results
        bracket_predictions = {
            "round_1": "Duke",
            "round_2": "Duke", 
            "sweet_16": "Duke",
            "elite_8": "UConn",
            "final_4": "UConn",
            "championship": "UConn"
        }
        
        actual_results = {
            "round_1": "Duke",
            "round_2": "Duke",
            "sweet_16": "Duke", 
            "elite_8": "Duke",  # Wrong - Duke actually lost
            "final_4": "UConn",
            "championship": "UConn"
        }
        
        scores = AdvancedScoringAlgorithms.tournament_bracket_scoring(
            bracket_predictions, actual_results
        )
        
        print("Bracket Performance:")
        print(f"Total Score: {scores['total']}")
        print(f"Max Possible: {scores['max_possible']}")
        print(f"Accuracy: {scores['percentage']:.1%}")
        print("\nRound-by-Round:")
        for round_name, score in scores['rounds'].items():
            print(f"  {round_name}: {score} points")
    
    @staticmethod
    def survivor_pool_example():
        """Survivor pool advanced scoring example"""
        print("\n🏈 Survivor Pool Example")
        print("=" * 50)
        
        from .algorithms import PoolEntry
        
        # Sample survivor pool entries
        entries = [
            PoolEntry("player_1", 1, "Chiefs", confidence=0.9),
            PoolEntry("player_1", 2, "Bills", confidence=0.8),
            PoolEntry("player_1", 3, "Ravens", confidence=0.7),
            
            PoolEntry("player_2", 1, "Chiefs", confidence=0.9),
            PoolEntry("player_2", 2, "Cowboys", confidence=0.6),  # Will lose
            
            PoolEntry("player_3", 1, "Dolphins", confidence=0.5),  # Will lose week 1
        ]
        
        # Sample weekly results
        weekly_results = {
            1: {"Chiefs": True, "Bills": True, "Ravens": True, "Cowboys": True, "Dolphins": False},
            2: {"Chiefs": True, "Bills": True, "Ravens": True, "Cowboys": False, "Dolphins": True},
            3: {"Chiefs": True, "Bills": True, "Ravens": True, "Cowboys": True, "Dolphins": True}
        }
        
        results = AdvancedScoringAlgorithms.survivor_pool_advanced_scoring(
            entries, weekly_results, confidence_bonus=True
        )
        
        print("Survivor Pool Results:")
        for player_id, score in results['scores'].items():
            elimination_week = results['elimination_week'][player_id]
            status = "SURVIVOR" if elimination_week is None else f"Eliminated Week {elimination_week}"
            print(f"{player_id}: {score:.1f} points ({status})")
        
        print(f"\nSurvivors: {results['survivors']}")
    
    @staticmethod
    def rotisserie_scoring_example():
        """Rotisserie scoring example"""
        print("\n📊 Rotisserie Scoring Example")
        print("=" * 50)
        
        # Sample player stats for rotisserie league
        player_stats = {
            "player_1": {"HR": 45, "RBI": 120, "AVG": 0.285, "SB": 15, "R": 95},
            "player_2": {"HR": 38, "RBI": 105, "AVG": 0.312, "SB": 28, "R": 105},
            "player_3": {"HR": 52, "RBI": 125, "AVG": 0.267, "SB": 8, "R": 88},
            "player_4": {"HR": 41, "RBI": 98, "AVG": 0.298, "SB": 22, "R": 102}
        }
        
        categories = ["HR", "RBI", "AVG", "SB", "R"]
        category_weights = {"HR": 1.2, "RBI": 1.1, "AVG": 1.3, "SB": 1.0, "R": 1.0}
        
        scores = AdvancedScoringAlgorithms.rotisserie_category_scoring(
            player_stats, categories, category_weights
        )
        
        print("Rotisserie Scores (Weighted):")
        for player, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{player}: {score:.1f} points")
        
        print("\nCategory Breakdown:")
        for category in categories:
            values = [(pid, stats[category]) for pid, stats in player_stats.items()]
            values.sort(key=lambda x: x[1], reverse=True)
            print(f"\n{category} (weight: {category_weights[category]}):")
            for rank, (player, value) in enumerate(values, 1):
                points = (len(player_stats) - rank + 1) * category_weights[category]
                print(f"  {rank}. {player}: {value} ({points:.1f} pts)")
    
    @staticmethod
    def advanced_metrics_example():
        """Advanced performance metrics example"""
        print("\n📈 Advanced Performance Metrics Example")
        print("=" * 50)
        
        # Sample performance data over time
        scores = [85, 92, 78, 95, 88, 76, 89, 94, 82, 91, 87, 93]
        returns = [0.02, 0.08, -0.09, 0.22, -0.07, -0.14, 0.17, 0.06, -0.13, 0.11, -0.04, 0.07]
        benchmark_returns = [0.01, 0.05, -0.05, 0.15, -0.03, -0.08, 0.12, 0.04, -0.08, 0.08, -0.02, 0.05]
        outcomes = [True, True, False, True, False, False, True, True, False, True, False, True]
        
        # Calculate Sharpe ratio
        sharpe = AdvancedMetrics.calculate_sharpe_ratio(returns)
        print(f"Sharpe Ratio: {sharpe:.3f}")
        
        # Calculate maximum drawdown
        drawdown_stats = AdvancedMetrics.calculate_maximum_drawdown(scores)
        print(f"\nDrawdown Analysis:")
        print(f"Maximum Drawdown: {drawdown_stats['max_drawdown']:.1%}")
        print(f"Drawdown Duration: {drawdown_stats['drawdown_duration']} periods")
        print(f"Recovery Time: {drawdown_stats['recovery_time']} periods")
        print(f"Current Drawdown: {drawdown_stats['current_drawdown']:.1%}")
        
        # Calculate information ratio
        info_ratio = AdvancedMetrics.calculate_information_ratio(returns, benchmark_returns)
        print(f"\nInformation Ratio: {info_ratio:.3f}")
        
        # Calculate win rate metrics
        win_stats = AdvancedMetrics.calculate_win_rate_metrics(outcomes)
        print(f"\nWin Rate Analysis:")
        print(f"Win Rate: {win_stats['win_rate']:.1%}")
        print(f"Wins/Losses: {win_stats['wins']}/{win_stats['losses']}")
        print(f"Current Streak: {win_stats['current_streak']}")
        print(f"Longest Streak: {win_stats['longest_streak']}")
    
    @staticmethod
    def collective_wisdom_aggregation_example():
        """Collective wisdom aggregation example"""
        print("\n🤝 Collective Wisdom Aggregation Example")
        print("=" * 50)
        
        # Individual expert predictions
        predictions = [
            {"temperature_rise": 2.1, "sea_level_rise": 0.43, "extreme_events": 0.75},
            {"temperature_rise": 1.8, "sea_level_rise": 0.38, "extreme_events": 0.68},
            {"temperature_rise": 2.4, "sea_level_rise": 0.51, "extreme_events": 0.82},
            {"temperature_rise": 2.0, "sea_level_rise": 0.45, "extreme_events": 0.71}
        ]
        
        # Expert confidence and expertise levels
        confidence_weights = [0.9, 0.7, 0.8, 0.85]
        expertise_weights = [0.95, 0.8, 0.9, 0.75]
        
        aggregated = AdvancedScoringAlgorithms.collective_wisdom_aggregation(
            predictions, confidence_weights, expertise_weights
        )
        
        print("Individual Predictions:")
        for i, pred in enumerate(predictions):
            conf = confidence_weights[i]
            exp = expertise_weights[i]
            print(f"Expert {i+1} (conf: {conf:.2f}, exp: {exp:.2f}): {pred}")
        
        print(f"\nAggregated Prediction:")
        for metric, value in aggregated.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.3f}")


class ConfigurationExamples:
    """Configuration management examples"""
    
    @staticmethod
    def configuration_management_example():
        """Configuration management example"""
        print("\n⚙️ Configuration Management Example")
        print("=" * 50)
        
        # Create configuration manager
        config_manager = ConfigurationManager("example_configs")
        
        # Create different configurations
        configs = {
            "development": DefaultConfigurations.development_config(),
            "production": DefaultConfigurations.production_config(),
            "high_performance": DefaultConfigurations.high_performance_config()
        }
        
        print("Creating configuration files...")
        for name, config in configs.items():
            config_manager.save_config(config, name)
            print(f"✅ Saved {name} configuration")
        
        # Load and validate a configuration
        dev_config = config_manager.load_config("development")
        validation_errors = dev_config.validate()
        
        print(f"\nDevelopment Configuration:")
        print(f"Debug Mode: {dev_config.debug_mode}")
        print(f"GPU Enabled: {dev_config.gpu.enable_gpu}")
        print(f"Thread Pool Size: {dev_config.thread_pool_size}")
        print(f"Validation Errors: {len(validation_errors)}")
        
        # List available configurations
        available_configs = config_manager.list_configs()
        print(f"\nAvailable Configurations: {available_configs}")
        
        print("\n✅ Configuration management example completed!")


async def run_all_examples():
    """Run all example demonstrations"""
    
    print("🚀 Fantasy Collective Scoring Engine - Complete Examples")
    print("=" * 70)
    
    # Core league examples
    await ExampleLeagues.nba_fantasy_league_example()
    await ExampleLeagues.prediction_market_example()
    await ExampleLeagues.collective_wisdom_example()
    
    # Advanced algorithm examples
    AdvancedExamples.tournament_bracket_example()
    AdvancedExamples.survivor_pool_example()
    AdvancedExamples.rotisserie_scoring_example()
    AdvancedExamples.advanced_metrics_example()
    AdvancedExamples.collective_wisdom_aggregation_example()
    
    # Configuration examples
    ConfigurationExamples.configuration_management_example()
    
    print("\n🎉 All examples completed successfully!")


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run all examples
    asyncio.run(run_all_examples())