"""
Value Betting Detection System for Sports Betting.

This module implements advanced algorithms for detecting value betting opportunities,
calculating expected value, identifying market inefficiencies, and discovering
arbitrage opportunities across multiple bookmakers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import itertools
from scipy.optimize import minimize
from scipy.stats import norm, poisson
import math

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class BookmakerOdds:
    """Bookmaker odds information."""
    bookmaker: str
    market_type: str  # "1x2", "over_under", "asian_handicap", etc.
    odds: Dict[str, float]  # e.g., {"home": 2.1, "draw": 3.2, "away": 3.8}
    stake_limits: Dict[str, float]  # Maximum stakes
    timestamp: str
    liquidity_score: float  # 0-1 scale


@dataclass
class ValueBet:
    """Value betting opportunity."""
    bookmaker: str
    market_type: str
    selection: str
    odds: float
    true_probability: float
    expected_value: float
    kelly_percentage: float
    confidence: float
    max_stake: float
    profit_potential: float
    risk_level: str
    recommendation: str


@dataclass
class ArbitrageOpportunity:
    """Arbitrage betting opportunity."""
    market_type: str
    bookmakers: List[str]
    selections: List[str]
    odds: List[float]
    implied_probabilities: List[float]
    total_implied_probability: float
    arbitrage_percentage: float
    optimal_stakes: Dict[str, float]
    guaranteed_profit: float
    total_stake: float
    profit_percentage: float
    risk_assessment: str


@dataclass
class MarketInefficiency:
    """Market inefficiency detection result."""
    market_type: str
    inefficiency_type: str  # "overpriced", "underpriced", "inconsistent"
    affected_bookmakers: List[str]
    price_discrepancy: float
    market_sentiment: str
    volume_analysis: Dict[str, Any]
    time_sensitivity: str  # "immediate", "short_term", "long_term"
    exploitation_strategy: str


class KellyCalculator:
    """Kelly Criterion calculator for optimal bet sizing."""
    
    @staticmethod
    def calculate_kelly_percentage(
        odds: float,
        true_probability: float,
        fractional_kelly: float = 0.25
    ) -> float:
        """
        Calculate Kelly percentage for optimal bet sizing.
        
        Args:
            odds: Decimal odds
            true_probability: Estimated true probability
            fractional_kelly: Fraction of full Kelly to use (risk management)
            
        Returns:
            Optimal bet percentage of bankroll
        """
        if true_probability <= 0 or odds <= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = true probability, q = 1 - p
        b = odds - 1
        p = true_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly for risk management
        adjusted_kelly = kelly_fraction * fractional_kelly
        
        # Ensure non-negative and reasonable limits
        return max(0.0, min(adjusted_kelly, 0.1))  # Cap at 10% of bankroll
    
    @staticmethod
    def calculate_expected_value(odds: float, true_probability: float) -> float:
        """Calculate expected value of a bet."""
        return (odds * true_probability) - 1.0


class OddsAnalyzer:
    """Advanced odds analysis and market efficiency detection."""
    
    @staticmethod
    def calculate_implied_probability(odds: float) -> float:
        """Calculate implied probability from decimal odds."""
        return 1.0 / odds if odds > 0 else 0.0
    
    @staticmethod
    def calculate_market_margin(odds_list: List[float]) -> float:
        """Calculate bookmaker margin (overround)."""
        total_implied = sum(OddsAnalyzer.calculate_implied_probability(odds) for odds in odds_list)
        return total_implied - 1.0
    
    @staticmethod
    def calculate_fair_odds(odds_list: List[float]) -> List[float]:
        """Calculate fair odds removing bookmaker margin."""
        implied_probs = [OddsAnalyzer.calculate_implied_probability(odds) for odds in odds_list]
        total_implied = sum(implied_probs)
        
        # Remove margin proportionally
        fair_probs = [prob / total_implied for prob in implied_probs]
        fair_odds = [1.0 / prob if prob > 0 else 0.0 for prob in fair_probs]
        
        return fair_odds
    
    @staticmethod
    def detect_line_movement(
        historical_odds: List[Tuple[datetime, List[float]]],
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect significant line movements.
        
        Args:
            historical_odds: List of (timestamp, odds_list) tuples
            threshold: Minimum change percentage to consider significant
            
        Returns:
            Line movement analysis
        """
        if len(historical_odds) < 2:
            return {"movement_detected": False}
        
        movements = []
        for i in range(1, len(historical_odds)):
            prev_time, prev_odds = historical_odds[i-1]
            curr_time, curr_odds = historical_odds[i]
            
            for j, (prev_odd, curr_odd) in enumerate(zip(prev_odds, curr_odds)):
                if prev_odd > 0 and curr_odd > 0:
                    change_pct = (curr_odd - prev_odd) / prev_odd
                    
                    if abs(change_pct) >= threshold:
                        movements.append({
                            "selection_index": j,
                            "time": curr_time,
                            "prev_odds": prev_odd,
                            "curr_odds": curr_odd,
                            "change_percentage": change_pct,
                            "direction": "increase" if change_pct > 0 else "decrease"
                        })
        
        return {
            "movement_detected": len(movements) > 0,
            "movements": movements,
            "volatility_score": len(movements) / len(historical_odds) if historical_odds else 0
        }


class ValueDetector:
    """
    Advanced value betting detection system.
    
    Features:
    - Expected value calculations
    - Kelly criterion optimal sizing
    - Market inefficiency identification
    - Arbitrage opportunity detection
    - Risk assessment and recommendation
    """
    
    def __init__(
        self,
        min_expected_value: float = 0.05,  # 5% minimum EV
        min_kelly_percentage: float = 0.01,  # 1% minimum Kelly
        max_kelly_percentage: float = 0.1,   # 10% maximum Kelly
        arbitrage_threshold: float = 0.01,   # 1% minimum arbitrage profit
        confidence_threshold: float = 0.7    # 70% minimum confidence
    ):
        """
        Initialize Value Detector.
        
        Args:
            min_expected_value: Minimum expected value to consider
            min_kelly_percentage: Minimum Kelly percentage
            max_kelly_percentage: Maximum Kelly percentage
            arbitrage_threshold: Minimum arbitrage profit threshold
            confidence_threshold: Minimum confidence threshold
        """
        self.min_expected_value = min_expected_value
        self.min_kelly_percentage = min_kelly_percentage
        self.max_kelly_percentage = max_kelly_percentage
        self.arbitrage_threshold = arbitrage_threshold
        self.confidence_threshold = confidence_threshold
        
        self.kelly_calculator = KellyCalculator()
        self.odds_analyzer = OddsAnalyzer()
        
        logger.info("Initialized ValueDetector with advanced algorithms")
    
    def detect_value_bets(
        self,
        bookmaker_odds: List[BookmakerOdds],
        true_probabilities: Dict[str, float],
        confidence_scores: Dict[str, float],
        bankroll: float = 10000.0
    ) -> List[ValueBet]:
        """
        Detect value betting opportunities.
        
        Args:
            bookmaker_odds: List of bookmaker odds
            true_probabilities: Model-predicted true probabilities
            confidence_scores: Confidence scores for predictions
            bankroll: Available bankroll
            
        Returns:
            List of value betting opportunities
        """
        value_bets = []
        
        for odds_data in bookmaker_odds:
            for selection, odds in odds_data.odds.items():
                if selection not in true_probabilities:
                    continue
                
                true_prob = true_probabilities[selection]
                confidence = confidence_scores.get(selection, 0.5)
                
                # Skip if confidence is too low
                if confidence < self.confidence_threshold:
                    continue
                
                # Calculate expected value
                expected_value = self.kelly_calculator.calculate_expected_value(
                    odds, true_prob
                )
                
                # Calculate Kelly percentage
                kelly_pct = self.kelly_calculator.calculate_kelly_percentage(
                    odds, true_prob
                )
                
                # Check if it meets value betting criteria
                if (expected_value >= self.min_expected_value and 
                    kelly_pct >= self.min_kelly_percentage):
                    
                    # Calculate stake and profit potential
                    max_stake = min(
                        bankroll * kelly_pct,
                        odds_data.stake_limits.get(selection, float('inf'))
                    )
                    
                    profit_potential = max_stake * (odds - 1) * true_prob
                    
                    # Risk assessment
                    risk_level = self._assess_risk_level(
                        expected_value, kelly_pct, confidence, odds_data.liquidity_score
                    )
                    
                    # Generate recommendation
                    recommendation = self._generate_recommendation(
                        expected_value, kelly_pct, confidence, risk_level
                    )
                    
                    value_bet = ValueBet(
                        bookmaker=odds_data.bookmaker,
                        market_type=odds_data.market_type,
                        selection=selection,
                        odds=odds,
                        true_probability=true_prob,
                        expected_value=expected_value,
                        kelly_percentage=kelly_pct,
                        confidence=confidence,
                        max_stake=max_stake,
                        profit_potential=profit_potential,
                        risk_level=risk_level,
                        recommendation=recommendation
                    )
                    
                    value_bets.append(value_bet)
        
        # Sort by expected value descending
        value_bets.sort(key=lambda x: x.expected_value, reverse=True)
        
        return value_bets
    
    def detect_arbitrage_opportunities(
        self,
        bookmaker_odds: List[BookmakerOdds],
        total_stake: float = 1000.0
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across bookmakers.
        
        Args:
            bookmaker_odds: List of bookmaker odds
            total_stake: Total stake to allocate
            
        Returns:
            List of arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        # Group odds by market type
        market_groups = {}
        for odds_data in bookmaker_odds:
            market_type = odds_data.market_type
            if market_type not in market_groups:
                market_groups[market_type] = []
            market_groups[market_type].append(odds_data)
        
        # Check each market for arbitrage
        for market_type, odds_list in market_groups.items():
            if len(odds_list) < 2:
                continue
            
            # Get all selections available in this market
            all_selections = set()
            for odds_data in odds_list:
                all_selections.update(odds_data.odds.keys())
            
            all_selections = list(all_selections)
            
            # Find best odds for each selection
            best_odds = {}
            best_bookmakers = {}
            
            for selection in all_selections:
                best_odds[selection] = 0.0
                best_bookmakers[selection] = None
                
                for odds_data in odds_list:
                    if selection in odds_data.odds:
                        if odds_data.odds[selection] > best_odds[selection]:
                            best_odds[selection] = odds_data.odds[selection]
                            best_bookmakers[selection] = odds_data.bookmaker
            
            # Check if arbitrage exists
            if len(best_odds) >= 2:  # Need at least 2 selections
                implied_probs = [
                    self.odds_analyzer.calculate_implied_probability(odds)
                    for odds in best_odds.values()
                ]
                
                total_implied = sum(implied_probs)
                
                if total_implied < 1.0:  # Arbitrage opportunity exists
                    arbitrage_pct = (1.0 - total_implied) * 100
                    
                    if arbitrage_pct >= self.arbitrage_threshold * 100:
                        # Calculate optimal stakes
                        optimal_stakes = self._calculate_arbitrage_stakes(
                            list(best_odds.values()), total_stake
                        )
                        
                        # Calculate guaranteed profit
                        guaranteed_profit = total_stake * (1.0 - total_implied)
                        profit_percentage = (guaranteed_profit / total_stake) * 100
                        
                        # Risk assessment
                        risk_assessment = self._assess_arbitrage_risk(
                            list(best_bookmakers.values()), implied_probs
                        )
                        
                        opportunity = ArbitrageOpportunity(
                            market_type=market_type,
                            bookmakers=list(best_bookmakers.values()),
                            selections=list(best_odds.keys()),
                            odds=list(best_odds.values()),
                            implied_probabilities=implied_probs,
                            total_implied_probability=total_implied,
                            arbitrage_percentage=arbitrage_pct,
                            optimal_stakes={
                                selection: stake for selection, stake 
                                in zip(best_odds.keys(), optimal_stakes)
                            },
                            guaranteed_profit=guaranteed_profit,
                            total_stake=total_stake,
                            profit_percentage=profit_percentage,
                            risk_assessment=risk_assessment
                        )
                        
                        arbitrage_opportunities.append(opportunity)
        
        # Sort by profit percentage descending
        arbitrage_opportunities.sort(key=lambda x: x.profit_percentage, reverse=True)
        
        return arbitrage_opportunities
    
    def identify_market_inefficiencies(
        self,
        bookmaker_odds: List[BookmakerOdds],
        market_volume_data: Optional[Dict[str, Any]] = None,
        historical_odds: Optional[List[Tuple[datetime, List[float]]]] = None
    ) -> List[MarketInefficiency]:
        """
        Identify market inefficiencies and pricing discrepancies.
        
        Args:
            bookmaker_odds: List of bookmaker odds
            market_volume_data: Optional market volume information
            historical_odds: Optional historical odds data
            
        Returns:
            List of identified market inefficiencies
        """
        inefficiencies = []
        
        # Group by market type
        market_groups = {}
        for odds_data in bookmaker_odds:
            market_type = odds_data.market_type
            if market_type not in market_groups:
                market_groups[market_type] = []
            market_groups[market_type].append(odds_data)
        
        for market_type, odds_list in market_groups.items():
            if len(odds_list) < 3:  # Need at least 3 bookmakers for comparison
                continue
            
            # Analyze price discrepancies
            selection_odds = {}
            for odds_data in odds_list:
                for selection, odds in odds_data.odds.items():
                    if selection not in selection_odds:
                        selection_odds[selection] = []
                    selection_odds[selection].append({
                        'bookmaker': odds_data.bookmaker,
                        'odds': odds,
                        'liquidity': odds_data.liquidity_score
                    })
            
            # Detect inefficiencies for each selection
            for selection, odds_info in selection_odds.items():
                if len(odds_info) < 3:
                    continue
                
                odds_values = [info['odds'] for info in odds_info]
                
                # Calculate price variance
                mean_odds = np.mean(odds_values)
                std_odds = np.std(odds_values)
                cv = std_odds / mean_odds if mean_odds > 0 else 0
                
                # High coefficient of variation indicates inefficiency
                if cv > 0.05:  # 5% threshold
                    min_odds = min(odds_values)
                    max_odds = max(odds_values)
                    discrepancy = (max_odds - min_odds) / mean_odds
                    
                    # Identify overpriced and underpriced bookmakers
                    overpriced = [
                        info['bookmaker'] for info in odds_info 
                        if info['odds'] < mean_odds - std_odds
                    ]
                    underpriced = [
                        info['bookmaker'] for info in odds_info 
                        if info['odds'] > mean_odds + std_odds
                    ]
                    
                    # Determine inefficiency type
                    if len(underpriced) > 0:
                        inefficiency_type = "underpriced"
                        affected_bookmakers = underpriced
                    elif len(overpriced) > 0:
                        inefficiency_type = "overpriced"
                        affected_bookmakers = overpriced
                    else:
                        inefficiency_type = "inconsistent"
                        affected_bookmakers = [info['bookmaker'] for info in odds_info]
                    
                    # Market sentiment analysis
                    market_sentiment = self._analyze_market_sentiment(
                        odds_values, market_volume_data
                    )
                    
                    # Time sensitivity analysis
                    time_sensitivity = self._analyze_time_sensitivity(
                        historical_odds, cv
                    )
                    
                    # Exploitation strategy
                    exploitation_strategy = self._generate_exploitation_strategy(
                        inefficiency_type, discrepancy, time_sensitivity
                    )
                    
                    inefficiency = MarketInefficiency(
                        market_type=f"{market_type}_{selection}",
                        inefficiency_type=inefficiency_type,
                        affected_bookmakers=affected_bookmakers,
                        price_discrepancy=discrepancy,
                        market_sentiment=market_sentiment,
                        volume_analysis=market_volume_data or {},
                        time_sensitivity=time_sensitivity,
                        exploitation_strategy=exploitation_strategy
                    )
                    
                    inefficiencies.append(inefficiency)
        
        return inefficiencies
    
    def _assess_risk_level(
        self,
        expected_value: float,
        kelly_percentage: float,
        confidence: float,
        liquidity_score: float
    ) -> str:
        """Assess risk level of a value bet."""
        risk_score = 0
        
        # Lower EV = higher risk
        if expected_value < 0.1:
            risk_score += 1
        
        # Higher Kelly % = higher risk
        if kelly_percentage > 0.05:
            risk_score += 1
        
        # Lower confidence = higher risk
        if confidence < 0.8:
            risk_score += 1
        
        # Lower liquidity = higher risk
        if liquidity_score < 0.7:
            risk_score += 1
        
        if risk_score == 0:
            return "Low"
        elif risk_score <= 2:
            return "Medium"
        else:
            return "High"
    
    def _generate_recommendation(
        self,
        expected_value: float,
        kelly_percentage: float,
        confidence: float,
        risk_level: str
    ) -> str:
        """Generate betting recommendation."""
        if risk_level == "High":
            return "Consider with caution - high risk"
        elif expected_value > 0.15 and confidence > 0.85:
            return "Strong value bet - recommended"
        elif expected_value > 0.1 and confidence > 0.75:
            return "Good value bet - proceed"
        else:
            return "Marginal value - small stake only"
    
    def _calculate_arbitrage_stakes(
        self,
        odds_list: List[float],
        total_stake: float
    ) -> List[float]:
        """Calculate optimal stakes for arbitrage betting."""
        implied_probs = [
            self.odds_analyzer.calculate_implied_probability(odds)
            for odds in odds_list
        ]
        
        total_implied = sum(implied_probs)
        
        # Stakes proportional to implied probabilities
        stakes = [
            total_stake * (prob / total_implied)
            for prob in implied_probs
        ]
        
        return stakes
    
    def _assess_arbitrage_risk(
        self,
        bookmakers: List[str],
        implied_probs: List[float]
    ) -> str:
        """Assess risk level for arbitrage opportunity."""
        # Check for high-risk bookmakers
        high_risk_bookmakers = ["suspicious_bookie", "low_liquidity_bookie"]
        
        risk_factors = 0
        
        # Check bookmaker reliability
        for bookmaker in bookmakers:
            if bookmaker.lower() in high_risk_bookmakers:
                risk_factors += 2
        
        # Check probability distribution
        max_prob = max(implied_probs)
        if max_prob > 0.7:  # One outcome heavily favored
            risk_factors += 1
        
        # Check number of bookmakers
        if len(bookmakers) > 3:
            risk_factors += 1  # More complex to execute
        
        if risk_factors == 0:
            return "Low Risk"
        elif risk_factors <= 2:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _analyze_market_sentiment(
        self,
        odds_values: List[float],
        volume_data: Optional[Dict[str, Any]]
    ) -> str:
        """Analyze market sentiment from odds and volume."""
        if not volume_data:
            return "neutral"
        
        # Simplified sentiment analysis
        avg_odds = np.mean(odds_values)
        
        if avg_odds < 1.5:
            return "strong_favorite"
        elif avg_odds < 2.0:
            return "moderate_favorite"
        elif avg_odds < 3.0:
            return "slight_favorite"
        else:
            return "underdog"
    
    def _analyze_time_sensitivity(
        self,
        historical_odds: Optional[List[Tuple[datetime, List[float]]]],
        cv: float
    ) -> str:
        """Analyze time sensitivity of inefficiency."""
        if not historical_odds or len(historical_odds) < 2:
            return "unknown"
        
        # High variance suggests rapid changes
        if cv > 0.1:
            return "immediate"
        elif cv > 0.05:
            return "short_term"
        else:
            return "long_term"
    
    def _generate_exploitation_strategy(
        self,
        inefficiency_type: str,
        discrepancy: float,
        time_sensitivity: str
    ) -> str:
        """Generate strategy for exploiting market inefficiency."""
        if time_sensitivity == "immediate":
            return "Act quickly - place bets immediately"
        elif inefficiency_type == "underpriced" and discrepancy > 0.1:
            return "Back the underpriced selection heavily"
        elif inefficiency_type == "overpriced":
            return "Lay the overpriced selection if possible"
        else:
            return "Monitor for better entry points"
    
    def calculate_portfolio_kelly(
        self,
        value_bets: List[ValueBet],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio Kelly sizing considering correlations.
        
        Args:
            value_bets: List of value betting opportunities
            correlation_matrix: Optional correlation matrix between bets
            
        Returns:
            Adjusted Kelly percentages for portfolio
        """
        if not value_bets:
            return {}
        
        if correlation_matrix is None:
            # Use individual Kelly percentages
            return {
                f"{bet.bookmaker}_{bet.selection}": bet.kelly_percentage
                for bet in value_bets
            }
        
        # Portfolio Kelly calculation with correlations
        n_bets = len(value_bets)
        if correlation_matrix.shape != (n_bets, n_bets):
            logger.warning("Correlation matrix size mismatch, using individual Kelly")
            return {
                f"{bet.bookmaker}_{bet.selection}": bet.kelly_percentage
                for bet in value_bets
            }
        
        # Extract parameters
        odds = np.array([bet.odds for bet in value_bets])
        probs = np.array([bet.true_probability for bet in value_bets])
        
        # Calculate portfolio Kelly (simplified approach)
        individual_kelly = np.array([bet.kelly_percentage for bet in value_bets])
        
        # Adjust for correlations (reduce position sizes)
        correlation_adjustment = np.mean(np.abs(correlation_matrix))
        adjusted_kelly = individual_kelly * (1 - correlation_adjustment * 0.5)
        
        return {
            f"{bet.bookmaker}_{bet.selection}": float(kelly_pct)
            for bet, kelly_pct in zip(value_bets, adjusted_kelly)
        }