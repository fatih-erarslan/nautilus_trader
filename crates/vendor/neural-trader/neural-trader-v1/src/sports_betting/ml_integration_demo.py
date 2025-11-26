"""
ML Integration Demo for Sports Betting Neural Networks.

This module demonstrates the complete integration of neural network models
for sports betting predictions, including outcome prediction, score prediction,
value detection, and the training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
import warnings
import torch

# Import our ML components
from .ml.outcome_predictor import OutcomePredictor, TeamStats, OutcomePrediction
from .ml.score_predictor import ScorePredictor, PlayerPerformance, WeatherConditions, ScorePrediction
from .ml.value_detector import ValueDetector, BookmakerOdds, ValueBet, ArbitrageOpportunity
from .ml.training_pipeline import TrainingPipeline, TrainingConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SportsMLDemoSystem:
    """
    Comprehensive demonstration system for sports betting ML models.
    
    This system showcases:
    - Neural network outcome predictions
    - Transformer-based score predictions
    - Value betting detection
    - Arbitrage opportunity identification
    - Complete training pipeline
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize the demo system."""
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Initialize models
        self.outcome_predictor = None
        self.score_predictor = None
        self.value_detector = ValueDetector()
        
        # Initialize training pipeline
        self.training_config = TrainingConfig(
            model_type="ensemble",
            use_gpu=self.use_gpu,
            epochs=50,  # Reduced for demo
            batch_size=16,
            hyperparameter_optimization=False  # Disabled for quick demo
        )
        
        self.training_pipeline = TrainingPipeline(
            config=self.training_config,
            experiment_name="sports_betting_demo"
        )
        
        logger.info(f"Initialized SportsMLDemoSystem on {self.device}")
    
    def generate_demo_data(self) -> Dict[str, Any]:
        """Generate synthetic demo data for testing."""
        logger.info("Generating synthetic demo data...")
        
        # Generate team data
        teams = ["Manchester United", "Liverpool", "Chelsea", "Arsenal", "Manchester City"]
        
        demo_data = {
            "teams": [],
            "matches": [],
            "player_data": [],
            "weather_data": [],
            "odds_data": []
        }
        
        # Generate team statistics
        for team in teams:
            team_stats = TeamStats(
                team_name=team,
                recent_form=[
                    np.random.choice([1, 0, -1], p=[0.4, 0.3, 0.3]) 
                    for _ in range(10)
                ],
                goals_scored_avg=np.random.uniform(1.0, 2.5),
                goals_conceded_avg=np.random.uniform(0.8, 2.0),
                home_performance=np.random.uniform(0.4, 0.8),
                away_performance=np.random.uniform(0.3, 0.7),
                head_to_head_record={"wins": np.random.randint(0, 5)},
                injury_list=[f"player_{i}" for i in range(np.random.randint(0, 3))],
                player_ratings={
                    f"player_{i}": np.random.uniform(6.0, 9.0) 
                    for i in range(11)
                }
            )
            demo_data["teams"].append(team_stats)
        
        # Generate match data
        for i in range(100):  # 100 demo matches
            home_team = np.random.choice(demo_data["teams"])
            away_team = np.random.choice([t for t in demo_data["teams"] if t.team_name != home_team.team_name])
            
            # Generate match context
            match_context = {
                "venue_advantage": np.random.uniform(0.0, 0.2),
                "weather_impact": np.random.uniform(-0.1, 0.1),
                "referee_home_bias": np.random.uniform(-0.05, 0.05),
                "days_since_last_match_home": np.random.randint(3, 14),
                "days_since_last_match_away": np.random.randint(3, 14),
                "league_position_home": np.random.randint(1, 20),
                "league_position_away": np.random.randint(1, 20),
                "market_pressure": np.random.uniform(-0.1, 0.1)
            }
            
            # Generate actual outcome
            outcome = np.random.choice([0, 1, 2], p=[0.3, 0.25, 0.45])  # Away win, Draw, Home win
            home_score = np.random.poisson(home_team.goals_scored_avg)
            away_score = np.random.poisson(away_team.goals_scored_avg)
            
            demo_data["matches"].append({
                "home_team": home_team,
                "away_team": away_team,
                "context": match_context,
                "outcome": outcome,
                "home_score": home_score,
                "away_score": away_score,
                "match_date": datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
        
        # Generate player performance data
        positions = ["striker", "midfielder", "defender", "goalkeeper"]
        for i in range(50):  # 50 demo players
            player = PlayerPerformance(
                player_id=f"player_{i}",
                player_name=f"Player {i}",
                position=np.random.choice(positions),
                goals_per_game=np.random.uniform(0.0, 1.2) if np.random.random() > 0.7 else 0.1,
                assists_per_game=np.random.uniform(0.0, 0.8),
                minutes_played_avg=np.random.uniform(60.0, 90.0),
                form_rating=np.random.uniform(5.0, 9.0),
                injury_status=np.random.choice(["fit", "doubt", "injured"], p=[0.8, 0.15, 0.05]),
                suspension_status=np.random.choice([True, False], p=[0.05, 0.95]),
                historical_vs_opponent={"goals": np.random.randint(0, 5)}
            )
            demo_data["player_data"].append(player)
        
        # Generate weather data
        for i in range(20):  # 20 weather scenarios
            weather = WeatherConditions(
                temperature_celsius=np.random.uniform(-5.0, 35.0),
                humidity_percentage=np.random.uniform(30.0, 95.0),
                wind_speed_kmh=np.random.uniform(0.0, 40.0),
                precipitation_chance=np.random.uniform(0.0, 100.0),
                visibility_km=np.random.uniform(5.0, 15.0),
                surface_condition=np.random.choice(["dry", "wet", "icy"], p=[0.7, 0.25, 0.05])
            )
            demo_data["weather_data"].append(weather)
        
        # Generate bookmaker odds
        bookmakers = ["Bet365", "William Hill", "Pinnacle", "Betfair", "SkyBet"]
        for i in range(50):  # 50 odds scenarios
            bookmaker_odds = BookmakerOdds(
                bookmaker=np.random.choice(bookmakers),
                market_type=np.random.choice(["1x2", "over_under_2.5", "asian_handicap"]),
                odds={
                    "home": np.random.uniform(1.5, 4.0),
                    "draw": np.random.uniform(2.8, 4.5),
                    "away": np.random.uniform(1.8, 5.0)
                },
                stake_limits={
                    "home": np.random.uniform(1000, 10000),
                    "draw": np.random.uniform(1000, 10000),
                    "away": np.random.uniform(1000, 10000)
                },
                timestamp=datetime.now().isoformat(),
                liquidity_score=np.random.uniform(0.6, 1.0)
            )
            demo_data["odds_data"].append(bookmaker_odds)
        
        logger.info(f"Generated {len(demo_data['matches'])} matches, {len(demo_data['player_data'])} players, {len(demo_data['odds_data'])} odds")
        return demo_data
    
    def demonstrate_outcome_prediction(self, demo_data: Dict[str, Any]) -> None:
        """Demonstrate outcome prediction functionality."""
        logger.info("=== OUTCOME PREDICTION DEMONSTRATION ===")
        
        # Initialize outcome predictor
        self.outcome_predictor = OutcomePredictor(
            model_type="gru",
            use_gpu=self.use_gpu,
            hidden_size=64,  # Smaller for demo
            num_layers=2
        )
        
        # Prepare training data
        training_data = []
        outcomes = []
        
        for match in demo_data["matches"][:50]:  # Use first 50 matches
            training_data.append((
                match["home_team"],
                match["away_team"],
                match["context"]
            ))
            outcomes.append(match["outcome"])
        
        # Train the model
        logger.info("Training outcome predictor...")
        training_metrics = self.outcome_predictor.train(
            training_data=training_data,
            outcomes=outcomes,
            epochs=10,  # Reduced for demo
            batch_size=8,
            early_stopping_patience=3
        )
        
        logger.info(f"Training completed. Best validation loss: {training_metrics.get('best_val_loss', 'N/A')}")
        
        # Make predictions on new matches
        test_matches = demo_data["matches"][50:55]  # Use next 5 matches
        
        for i, match in enumerate(test_matches):
            prediction = self.outcome_predictor.predict(
                home_team_stats=match["home_team"],
                away_team_stats=match["away_team"],
                match_context=match["context"]
            )
            
            logger.info(f"\nMatch {i+1}: {match['home_team'].team_name} vs {match['away_team'].team_name}")
            logger.info(f"Predicted probabilities:")
            logger.info(f"  Home win: {prediction.home_win_prob:.3f}")
            logger.info(f"  Draw: {prediction.draw_prob:.3f}")
            logger.info(f"  Away win: {prediction.away_win_prob:.3f}")
            logger.info(f"  Confidence: {prediction.confidence:.3f}")
            logger.info(f"Actual outcome: {['Away win', 'Draw', 'Home win'][match['outcome']]}")
        
        # Feature importance
        importance = self.outcome_predictor.get_feature_importance()
        logger.info(f"\nTop 5 most important features:")
        for i, (feature, score) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]):
            logger.info(f"  {i+1}. {feature}: {score:.4f}")
    
    def demonstrate_score_prediction(self, demo_data: Dict[str, Any]) -> None:
        """Demonstrate score prediction functionality."""
        logger.info("\n=== SCORE PREDICTION DEMONSTRATION ===")
        
        # Initialize score predictor
        self.score_predictor = ScorePredictor(
            use_gpu=self.use_gpu,
            d_model=128,  # Smaller for demo
            num_heads=4,
            num_layers=3
        )
        
        # Prepare training data
        training_data = []
        home_scores = []
        away_scores = []
        
        for match in demo_data["matches"][:30]:  # Use first 30 matches
            # Create simplified team stats
            home_team_stats = {
                "attack_rating": match["home_team"].goals_scored_avg * 3.5,
                "defense_rating": (3.0 - match["home_team"].goals_conceded_avg) * 2.5,
                "goals_scored_last_5": np.mean([max(0, result) for result in match["home_team"].recent_form[-5:]]) * 2,
                "goals_conceded_last_5": np.mean([max(0, -result) for result in match["home_team"].recent_form[-5:]]) * 1.5
            }
            
            away_team_stats = {
                "attack_rating": match["away_team"].goals_scored_avg * 3.0,  # Away penalty
                "defense_rating": (3.0 - match["away_team"].goals_conceded_avg) * 2.5,
                "goals_scored_last_5": np.mean([max(0, result) for result in match["away_team"].recent_form[-5:]]) * 1.8,
                "goals_conceded_last_5": np.mean([max(0, -result) for result in match["away_team"].recent_form[-5:]]) * 1.5
            }
            
            # Use sample players and weather
            home_players = demo_data["player_data"][:11]  # First 11 players
            away_players = demo_data["player_data"][11:22]  # Next 11 players
            weather = demo_data["weather_data"][0]  # First weather condition
            
            training_data.append((
                home_team_stats,
                away_team_stats,
                home_players,
                away_players,
                weather,
                match["context"]
            ))
            
            home_scores.append(match["home_score"])
            away_scores.append(match["away_score"])
        
        # Train the model
        logger.info("Training score predictor...")
        training_metrics = self.score_predictor.train(
            training_data=training_data,
            home_scores=home_scores,
            away_scores=away_scores,
            epochs=5,  # Very reduced for demo
            batch_size=4,
            early_stopping_patience=2
        )
        
        logger.info(f"Training completed. Best validation loss: {training_metrics.get('best_val_loss', 'N/A')}")
        
        # Make predictions on new matches
        test_matches = demo_data["matches"][30:33]  # Use next 3 matches
        
        for i, match in enumerate(test_matches):
            # Prepare test data the same way
            home_team_stats = {
                "attack_rating": match["home_team"].goals_scored_avg * 3.5,
                "defense_rating": (3.0 - match["home_team"].goals_conceded_avg) * 2.5
            }
            
            away_team_stats = {
                "attack_rating": match["away_team"].goals_scored_avg * 3.0,
                "defense_rating": (3.0 - match["away_team"].goals_conceded_avg) * 2.5
            }
            
            prediction = self.score_predictor.predict(
                home_team_stats=home_team_stats,
                away_team_stats=away_team_stats,
                home_players=demo_data["player_data"][:11],
                away_players=demo_data["player_data"][11:22],
                weather=demo_data["weather_data"][0],
                match_context=match["context"]
            )
            
            logger.info(f"\nMatch {i+1}: {match['home_team'].team_name} vs {match['away_team'].team_name}")
            logger.info(f"Predicted score: {prediction.home_score_expected:.1f} - {prediction.away_score_expected:.1f}")
            logger.info(f"Total goals expected: {prediction.total_goals_expected:.1f}")
            logger.info(f"Top score probabilities:")
            for score, prob in list(prediction.score_probabilities.items())[:3]:
                logger.info(f"  {score}: {prob:.3f}")
            logger.info(f"Over 2.5 goals: {prediction.over_under_probabilities.get('over_2.5', 0):.3f}")
            logger.info(f"Actual score: {match['home_score']} - {match['away_score']}")
    
    def demonstrate_value_detection(self, demo_data: Dict[str, Any]) -> None:
        """Demonstrate value betting detection."""
        logger.info("\n=== VALUE BETTING DEMONSTRATION ===")
        
        # Use sample odds data
        bookmaker_odds = demo_data["odds_data"][:10]  # First 10 odds
        
        # Generate "true" probabilities (in practice, these would come from our models)
        true_probabilities = {
            "home": 0.45,
            "draw": 0.28,
            "away": 0.27
        }
        
        confidence_scores = {
            "home": 0.85,
            "draw": 0.75,
            "away": 0.80
        }
        
        # Detect value bets
        value_bets = self.value_detector.detect_value_bets(
            bookmaker_odds=bookmaker_odds,
            true_probabilities=true_probabilities,
            confidence_scores=confidence_scores,
            bankroll=10000.0
        )
        
        logger.info(f"\nFound {len(value_bets)} value betting opportunities:")
        for i, bet in enumerate(value_bets[:5]):  # Show top 5
            logger.info(f"\nValue Bet {i+1}:")
            logger.info(f"  Bookmaker: {bet.bookmaker}")
            logger.info(f"  Market: {bet.market_type}")
            logger.info(f"  Selection: {bet.selection}")
            logger.info(f"  Odds: {bet.odds:.2f}")
            logger.info(f"  Expected Value: {bet.expected_value:.3f} ({bet.expected_value*100:.1f}%)")
            logger.info(f"  Kelly %: {bet.kelly_percentage:.3f} ({bet.kelly_percentage*100:.1f}%)")
            logger.info(f"  Max Stake: £{bet.max_stake:.2f}")
            logger.info(f"  Risk Level: {bet.risk_level}")
            logger.info(f"  Recommendation: {bet.recommendation}")
        
        # Detect arbitrage opportunities
        arbitrage_opportunities = self.value_detector.detect_arbitrage_opportunities(
            bookmaker_odds=bookmaker_odds,
            total_stake=1000.0
        )
        
        logger.info(f"\nFound {len(arbitrage_opportunities)} arbitrage opportunities:")
        for i, arb in enumerate(arbitrage_opportunities[:3]):  # Show top 3
            logger.info(f"\nArbitrage {i+1}:")
            logger.info(f"  Market: {arb.market_type}")
            logger.info(f"  Profit %: {arb.profit_percentage:.2f}%")
            logger.info(f"  Guaranteed Profit: £{arb.guaranteed_profit:.2f}")
            logger.info(f"  Bookmakers: {', '.join(arb.bookmakers)}")
            logger.info(f"  Risk Assessment: {arb.risk_assessment}")
            
            logger.info("  Optimal Stakes:")
            for selection, stake in arb.optimal_stakes.items():
                logger.info(f"    {selection}: £{stake:.2f}")
        
        # Identify market inefficiencies
        inefficiencies = self.value_detector.identify_market_inefficiencies(
            bookmaker_odds=bookmaker_odds
        )
        
        logger.info(f"\nFound {len(inefficiencies)} market inefficiencies:")
        for i, ineff in enumerate(inefficiencies[:3]):  # Show top 3
            logger.info(f"\nInefficiency {i+1}:")
            logger.info(f"  Market: {ineff.market_type}")
            logger.info(f"  Type: {ineff.inefficiency_type}")
            logger.info(f"  Price Discrepancy: {ineff.price_discrepancy:.3f}")
            logger.info(f"  Time Sensitivity: {ineff.time_sensitivity}")
            logger.info(f"  Strategy: {ineff.exploitation_strategy}")
    
    def demonstrate_training_pipeline(self, demo_data: Dict[str, Any]) -> None:
        """Demonstrate the complete training pipeline."""
        logger.info("\n=== TRAINING PIPELINE DEMONSTRATION ===")
        
        # Prepare ensemble training data
        training_data = {
            "outcome_data": {
                "features": [(match["home_team"], match["away_team"], match["context"]) 
                           for match in demo_data["matches"][:40]],
                "targets": [match["outcome"] for match in demo_data["matches"][:40]]
            },
            "score_data": {
                "features": [
                    (
                        {"attack_rating": match["home_team"].goals_scored_avg * 3.5},
                        {"attack_rating": match["away_team"].goals_scored_avg * 3.0},
                        demo_data["player_data"][:11],
                        demo_data["player_data"][11:22],
                        demo_data["weather_data"][0],
                        match["context"]
                    )
                    for match in demo_data["matches"][:40]
                ],
                "home_targets": [match["home_score"] for match in demo_data["matches"][:40]],
                "away_targets": [match["away_score"] for match in demo_data["matches"][:40]]
            }
        }
        
        # Run training experiment
        experiment_result = self.training_pipeline.run_experiment(
            training_data=training_data,
            experiment_id="sports_betting_demo"
        )
        
        logger.info(f"\nExperiment Results:")
        logger.info(f"  Experiment ID: {experiment_result.experiment_id}")
        logger.info(f"  Best Model: {experiment_result.best_model_id}")
        logger.info(f"  Best Score: {experiment_result.best_score:.3f}")
        logger.info(f"  Models Trained: {len(experiment_result.model_performances)}")
        
        # Show model performances
        logger.info(f"\nModel Performances:")
        for performance in experiment_result.model_performances:
            logger.info(f"  {performance.model_type}:")
            logger.info(f"    Accuracy: {performance.accuracy:.3f}")
            logger.info(f"    F1 Score: {performance.f1_score:.3f}")
            logger.info(f"    Training Time: {performance.training_time:.1f}s")
        
        # Get model leaderboard
        leaderboard = self.training_pipeline.get_model_leaderboard()
        if not leaderboard.empty:
            logger.info(f"\nModel Leaderboard:")
            logger.info(leaderboard.head().to_string(index=False))
    
    def run_complete_demo(self) -> None:
        """Run the complete demonstration of all components."""
        logger.info("Starting Sports Betting ML System Demonstration")
        logger.info("=" * 60)
        
        # Generate demo data
        demo_data = self.generate_demo_data()
        
        # Demonstrate each component
        try:
            self.demonstrate_outcome_prediction(demo_data)
            self.demonstrate_score_prediction(demo_data)
            self.demonstrate_value_detection(demo_data)
            self.demonstrate_training_pipeline(demo_data)
            
            logger.info("\n" + "=" * 60)
            logger.info("Sports Betting ML System Demo Completed Successfully!")
            logger.info("All components are working and integrated properly.")
            
            # Summary statistics
            logger.info(f"\nDemo Summary:")
            logger.info(f"  GPU Enabled: {self.use_gpu}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Models Demonstrated: Outcome Predictor, Score Predictor, Value Detector")
            logger.info(f"  Training Pipeline: Complete ensemble training")
            logger.info(f"  Data Generated: {len(demo_data['matches'])} matches, {len(demo_data['player_data'])} players")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


def main():
    """Main function to run the demo."""
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    logger.info(f"GPU Available: {gpu_available}")
    
    # Initialize and run demo
    demo_system = SportsMLDemoSystem(use_gpu=gpu_available)
    demo_system.run_complete_demo()


if __name__ == "__main__":
    main()