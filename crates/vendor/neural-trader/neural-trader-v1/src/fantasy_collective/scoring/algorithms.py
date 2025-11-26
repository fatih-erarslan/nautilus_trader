"""
Advanced Scoring Algorithms for Fantasy Collective Systems

This module contains specialized algorithms for different game types
and advanced scoring methodologies.
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class TournamentType(Enum):
    """Tournament bracket types"""
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"
    SWISS_SYSTEM = "swiss_system"


@dataclass
class BracketMatch:
    """Tournament bracket match"""
    match_id: str
    round_number: int
    player1_id: str
    player2_id: str
    winner_id: Optional[str] = None
    score1: Optional[float] = None
    score2: Optional[float] = None
    multiplier: float = 1.0


@dataclass
class PoolEntry:
    """Survivor pool entry"""
    player_id: str
    week: int
    pick: str
    confidence: Optional[float] = None
    eliminated: bool = False


class AdvancedScoringAlgorithms:
    """Collection of advanced scoring algorithms"""
    
    @staticmethod
    def weighted_accuracy_score(predictions: List[float], 
                              actuals: List[float],
                              weights: Optional[List[float]] = None,
                              confidence_scores: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate weighted accuracy score with confidence adjustments
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            weights: Optional weights for each prediction
            confidence_scores: Confidence levels (0-1) for each prediction
            
        Returns:
            Dictionary with various accuracy metrics
        """
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        if weights is None:
            weights = np.ones(len(predictions))
        else:
            weights = np.array(weights)
        
        if confidence_scores is None:
            confidence_scores = np.ones(len(predictions))
        else:
            confidence_scores = np.array(confidence_scores)
        
        # Calculate various metrics
        absolute_errors = np.abs(predictions - actuals)
        squared_errors = (predictions - actuals) ** 2
        
        # Weighted metrics
        weighted_mae = np.average(absolute_errors, weights=weights)
        weighted_mse = np.average(squared_errors, weights=weights)
        weighted_rmse = np.sqrt(weighted_mse)
        
        # Confidence-adjusted accuracy
        confidence_penalty = 1.0 - confidence_scores
        confidence_adjusted_errors = absolute_errors * (1.0 + confidence_penalty)
        confidence_adjusted_mae = np.average(confidence_adjusted_errors, weights=weights)
        
        # Hit rate (within threshold)
        threshold = np.std(actuals) * 0.1  # 10% of standard deviation
        hits = (absolute_errors <= threshold).astype(float)
        hit_rate = np.average(hits, weights=weights)
        
        # Calibration score (how well confidence matches accuracy)
        calibration_score = 1.0 - np.mean(np.abs(confidence_scores - (1.0 - hits)))
        
        return {
            "weighted_mae": float(weighted_mae),
            "weighted_mse": float(weighted_mse),
            "weighted_rmse": float(weighted_rmse),
            "confidence_adjusted_mae": float(confidence_adjusted_mae),
            "hit_rate": float(hit_rate),
            "calibration_score": float(calibration_score),
            "total_predictions": len(predictions)
        }
    
    @staticmethod
    def bayesian_scoring(prior_performance: Dict[str, float],
                        current_performance: Dict[str, float],
                        update_weight: float = 0.3) -> Dict[str, float]:
        """
        Bayesian update of performance scores
        
        Args:
            prior_performance: Previous performance metrics
            current_performance: Current performance metrics
            update_weight: Weight for new information (0-1)
            
        Returns:
            Updated performance scores
        """
        updated_scores = {}
        
        for metric in prior_performance.keys():
            if metric in current_performance:
                prior = prior_performance[metric]
                current = current_performance[metric]
                
                # Bayesian update
                updated = prior * (1 - update_weight) + current * update_weight
                updated_scores[metric] = updated
            else:
                updated_scores[metric] = prior_performance[metric]
        
        # Add new metrics
        for metric in current_performance.keys():
            if metric not in updated_scores:
                updated_scores[metric] = current_performance[metric] * update_weight
        
        return updated_scores
    
    @staticmethod
    def tournament_bracket_scoring(bracket_predictions: Dict[str, str],
                                 actual_results: Dict[str, str],
                                 tournament_type: TournamentType = TournamentType.SINGLE_ELIMINATION) -> Dict[str, float]:
        """
        Advanced tournament bracket scoring with different tournament types
        
        Args:
            bracket_predictions: Player's bracket predictions
            actual_results: Actual tournament results
            tournament_type: Type of tournament structure
            
        Returns:
            Scoring breakdown by round and total
        """
        scores = {}
        
        if tournament_type == TournamentType.SINGLE_ELIMINATION:
            # Standard NCAA-style bracket scoring
            round_multipliers = {
                "round_1": 1,
                "round_2": 2, 
                "sweet_16": 4,
                "elite_8": 8,
                "final_4": 16,
                "championship": 32
            }
        elif tournament_type == TournamentType.DOUBLE_ELIMINATION:
            # Double elimination with losers bracket
            round_multipliers = {
                "winners_r1": 1,
                "winners_r2": 2,
                "winners_semi": 4,
                "winners_final": 8,
                "losers_r1": 1,
                "losers_r2": 2,
                "losers_semi": 4,
                "losers_final": 8,
                "grand_final": 16
            }
        else:
            # Default multipliers
            round_multipliers = {round_name: 2 ** i for i, round_name in enumerate(bracket_predictions.keys())}
        
        total_score = 0.0
        round_scores = {}
        
        for round_name, predicted_winner in bracket_predictions.items():
            if round_name in actual_results:
                actual_winner = actual_results[round_name]
                multiplier = round_multipliers.get(round_name, 1)
                
                if predicted_winner == actual_winner:
                    round_score = multiplier
                    total_score += round_score
                    round_scores[round_name] = round_score
                else:
                    round_scores[round_name] = 0
        
        scores["total"] = total_score
        scores["rounds"] = round_scores
        scores["max_possible"] = sum(round_multipliers.values())
        scores["percentage"] = total_score / scores["max_possible"] if scores["max_possible"] > 0 else 0
        
        return scores
    
    @staticmethod
    def survivor_pool_advanced_scoring(entries: List[PoolEntry],
                                     weekly_results: Dict[int, Dict[str, bool]],
                                     confidence_bonus: bool = True) -> Dict[str, Any]:
        """
        Advanced survivor pool scoring with confidence bonuses
        
        Args:
            entries: List of pool entries
            weekly_results: Results by week {week: {pick: won}}
            confidence_bonus: Whether to apply confidence bonuses
            
        Returns:
            Detailed scoring results
        """
        player_scores = {}
        elimination_week = {}
        
        for entry in entries:
            player_id = entry.player_id
            week = entry.week
            pick = entry.pick
            confidence = entry.confidence or 1.0
            
            if player_id not in player_scores:
                player_scores[player_id] = 0
                elimination_week[player_id] = None
            
            # Skip if already eliminated
            if elimination_week[player_id] is not None:
                continue
            
            # Check if pick won
            week_results = weekly_results.get(week, {})
            pick_won = week_results.get(pick, False)
            
            if pick_won:
                # Base survival bonus
                base_score = 10
                
                # Confidence bonus if enabled
                confidence_bonus_score = 0
                if confidence_bonus and confidence is not None:
                    # Higher confidence = higher bonus if correct
                    confidence_bonus_score = base_score * (confidence - 0.5) * 0.5
                
                # Week survival bonus (later weeks worth more)
                week_bonus = week * 2
                
                total_weekly_score = base_score + confidence_bonus_score + week_bonus
                player_scores[player_id] += total_weekly_score
            else:
                # Player eliminated
                elimination_week[player_id] = week
        
        # Final survivor bonus
        survivors = [pid for pid, week in elimination_week.items() if week is None]
        if survivors:
            survivor_bonus = 100 / len(survivors)  # Split bonus among survivors
            for survivor in survivors:
                player_scores[survivor] += survivor_bonus
        
        return {
            "scores": player_scores,
            "elimination_week": elimination_week,
            "survivors": survivors,
            "total_weeks": max(weekly_results.keys()) if weekly_results else 0
        }
    
    @staticmethod
    def rotisserie_category_scoring(player_stats: Dict[str, Dict[str, float]],
                                  categories: List[str],
                                  category_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Advanced rotisserie scoring with category weighting
        
        Args:
            player_stats: {player_id: {category: value}}
            categories: List of statistical categories
            category_weights: Optional weights for each category
            
        Returns:
            Final scores for each player
        """
        if not player_stats:
            return {}
        
        if category_weights is None:
            category_weights = {cat: 1.0 for cat in categories}
        
        num_players = len(player_stats)
        player_ids = list(player_stats.keys())
        final_scores = {pid: 0.0 for pid in player_ids}
        
        for category in categories:
            # Extract category stats
            category_values = []
            for pid in player_ids:
                value = player_stats[pid].get(category, 0.0)
                category_values.append((pid, value))
            
            # Sort by value (descending - higher is better)
            category_values.sort(key=lambda x: x[1], reverse=True)
            
            # Assign points based on rank
            weight = category_weights.get(category, 1.0)
            for rank, (player_id, value) in enumerate(category_values):
                points = (num_players - rank) * weight
                final_scores[player_id] += points
        
        return final_scores
    
    @staticmethod
    def dynamic_multiplier_calculation(base_performance: float,
                                     historical_average: float,
                                     consistency_score: float,
                                     streak_length: int,
                                     difficulty_factor: float = 1.0) -> float:
        """
        Calculate dynamic multiplier based on multiple factors
        
        Args:
            base_performance: Current performance score
            historical_average: Player's historical average
            consistency_score: How consistent the player has been (0-1)
            streak_length: Current streak (positive for good, negative for bad)
            difficulty_factor: Factor for opponent/situation difficulty
            
        Returns:
            Dynamic multiplier to apply to base score
        """
        # Performance relative to average
        relative_performance = base_performance / max(historical_average, 0.1)
        performance_multiplier = 1.0 + (relative_performance - 1.0) * 0.2
        
        # Consistency bonus
        consistency_multiplier = 1.0 + consistency_score * 0.1
        
        # Streak bonus/penalty
        streak_multiplier = 1.0 + (streak_length * 0.05)
        streak_multiplier = max(0.5, min(2.0, streak_multiplier))  # Cap between 0.5x and 2.0x
        
        # Difficulty adjustment
        difficulty_multiplier = difficulty_factor
        
        # Combine all multipliers
        final_multiplier = (performance_multiplier * 
                          consistency_multiplier * 
                          streak_multiplier * 
                          difficulty_multiplier)
        
        # Apply reasonable bounds
        return max(0.1, min(3.0, final_multiplier))
    
    @staticmethod
    def collective_wisdom_aggregation(individual_predictions: List[Dict[str, float]],
                                    confidence_weights: Optional[List[float]] = None,
                                    expertise_weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Aggregate individual predictions using collective wisdom principles
        
        Args:
            individual_predictions: List of prediction dictionaries
            confidence_weights: Confidence levels for each predictor
            expertise_weights: Expertise levels for each predictor
            
        Returns:
            Aggregated predictions
        """
        if not individual_predictions:
            return {}
        
        # Get all prediction keys
        all_keys = set()
        for pred in individual_predictions:
            all_keys.update(pred.keys())
        
        aggregated = {}
        
        for key in all_keys:
            values = []
            weights = []
            
            for i, pred in enumerate(individual_predictions):
                if key in pred:
                    value = pred[key]
                    values.append(value)
                    
                    # Calculate combined weight
                    weight = 1.0
                    if confidence_weights:
                        weight *= confidence_weights[i]
                    if expertise_weights:
                        weight *= expertise_weights[i]
                    
                    weights.append(weight)
            
            if values:
                # Weighted average
                values = np.array(values)
                weights = np.array(weights)
                
                weighted_mean = np.average(values, weights=weights)
                aggregated[key] = float(weighted_mean)
        
        return aggregated
    
    @staticmethod
    def gpu_batch_elo_tournament(ratings: np.ndarray,
                               match_results: np.ndarray,
                               k_factors: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated batch ELO updates for tournament play
        
        Args:
            ratings: Array of current ratings [n_players]
            match_results: Array of match results [n_matches, 3] (player1_idx, player2_idx, outcome)
            k_factors: K-factors for each player [n_players]
            
        Returns:
            Updated ratings array
        """
        if not GPU_AVAILABLE:
            return AdvancedScoringAlgorithms._cpu_batch_elo_tournament(ratings, match_results, k_factors)
        
        try:
            # Transfer to GPU
            gpu_ratings = cp.array(ratings, dtype=cp.float32)
            gpu_matches = cp.array(match_results, dtype=cp.int32)
            gpu_k_factors = cp.array(k_factors, dtype=cp.float32)
            
            # Process each match
            for match in gpu_matches:
                p1_idx, p2_idx, outcome = match
                
                # Calculate expected scores
                rating_diff = gpu_ratings[p1_idx] - gpu_ratings[p2_idx]
                expected_p1 = 1.0 / (1.0 + cp.exp(-rating_diff / 400.0))
                expected_p2 = 1.0 - expected_p1
                
                # Update ratings
                k1 = gpu_k_factors[p1_idx]
                k2 = gpu_k_factors[p2_idx]
                
                gpu_ratings[p1_idx] += k1 * (outcome - expected_p1)
                gpu_ratings[p2_idx] += k2 * ((1.0 - outcome) - expected_p2)
            
            # Transfer back to CPU
            return cp.asnumpy(gpu_ratings)
            
        except Exception:
            # Fallback to CPU
            return AdvancedScoringAlgorithms._cpu_batch_elo_tournament(ratings, match_results, k_factors)
    
    @staticmethod
    def _cpu_batch_elo_tournament(ratings: np.ndarray,
                                match_results: np.ndarray,
                                k_factors: np.ndarray) -> np.ndarray:
        """CPU fallback for batch ELO tournament updates"""
        updated_ratings = ratings.copy()
        
        for match in match_results:
            p1_idx, p2_idx, outcome = match
            
            # Calculate expected scores
            rating_diff = updated_ratings[p1_idx] - updated_ratings[p2_idx]
            expected_p1 = 1.0 / (1.0 + math.exp(-rating_diff / 400.0))
            expected_p2 = 1.0 - expected_p1
            
            # Update ratings
            k1 = k_factors[p1_idx]
            k2 = k_factors[p2_idx]
            
            updated_ratings[p1_idx] += k1 * (outcome - expected_p1)
            updated_ratings[p2_idx] += k2 * ((1.0 - outcome) - expected_p2)
        
        return updated_ratings


class AdvancedMetrics:
    """Advanced performance metrics calculations"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio for performance evaluation"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calculate_maximum_drawdown(scores: List[float]) -> Dict[str, float]:
        """Calculate maximum drawdown and recovery statistics"""
        if not scores:
            return {"max_drawdown": 0.0, "drawdown_duration": 0, "recovery_time": 0}
        
        scores = np.array(scores)
        cumulative_max = np.maximum.accumulate(scores)
        drawdown = (cumulative_max - scores) / cumulative_max
        
        max_drawdown = np.max(drawdown)
        max_dd_idx = np.argmax(drawdown)
        
        # Calculate drawdown duration
        drawdown_start = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                drawdown_start = i
                break
        
        drawdown_duration = max_dd_idx - drawdown_start
        
        # Calculate recovery time
        recovery_time = 0
        for i in range(max_dd_idx, len(scores)):
            if scores[i] >= cumulative_max[max_dd_idx]:
                recovery_time = i - max_dd_idx
                break
        
        return {
            "max_drawdown": float(max_drawdown),
            "drawdown_duration": drawdown_duration,
            "recovery_time": recovery_time,
            "current_drawdown": float(drawdown[-1])
        }
    
    @staticmethod
    def calculate_information_ratio(returns: List[float], 
                                  benchmark_returns: List[float]) -> float:
        """Calculate information ratio vs benchmark"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        benchmark_returns = np.array(benchmark_returns)
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(excess_returns) / tracking_error
    
    @staticmethod
    def calculate_win_rate_metrics(outcomes: List[bool]) -> Dict[str, float]:
        """Calculate comprehensive win rate statistics"""
        if not outcomes:
            return {"win_rate": 0.0, "longest_streak": 0, "current_streak": 0}
        
        wins = sum(outcomes)
        total = len(outcomes)
        win_rate = wins / total
        
        # Calculate streaks
        current_streak = 0
        longest_streak = 0
        current_count = 0
        
        for outcome in reversed(outcomes):
            if outcome:
                current_count += 1
            else:
                current_streak = current_count
                break
            current_count += 1
        
        # If all recent outcomes are wins
        if current_count == len(outcomes):
            current_streak = current_count
        
        # Calculate longest streak
        max_streak = 0
        current_run = 0
        
        for outcome in outcomes:
            if outcome:
                current_run += 1
                max_streak = max(max_streak, current_run)
            else:
                current_run = 0
        
        longest_streak = max_streak
        
        return {
            "win_rate": win_rate,
            "wins": wins,
            "losses": total - wins,
            "total_games": total,
            "longest_streak": longest_streak,
            "current_streak": current_streak
        }