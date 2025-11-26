"""
Senator Trade Scoring Algorithm

This module implements a comprehensive scoring system for ranking senators based on:
1. Trading performance metrics (win rate, returns, Sharpe ratio)
2. Disclosure speed and transparency
3. Committee relevance and insider information potential
4. Kelly Criterion position sizing for optimal capital allocation

The scoring system combines multiple factors to identify the most successful
and timely senator trades for replication strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommitteeType(Enum):
    """Senate committee types with insider information potential"""
    BANKING = "Banking, Housing, and Urban Affairs"
    FINANCE = "Finance"
    JUDICIARY = "Judiciary"
    COMMERCE = "Commerce, Science, and Transportation"
    ENERGY = "Energy and Natural Resources"
    HEALTH = "Health, Education, Labor and Pensions"
    AGRICULTURE = "Agriculture, Nutrition, and Forestry"
    ARMED_SERVICES = "Armed Services"
    INTELLIGENCE = "Intelligence"
    HOMELAND_SECURITY = "Homeland Security and Governmental Affairs"
    OTHER = "Other"


@dataclass
class TradeRecord:
    """Individual trade record for a senator"""
    senator_name: str
    ticker: str
    transaction_date: datetime
    disclosure_date: datetime
    transaction_type: str  # 'buy', 'sell'
    amount_range: str  # e.g., "$1,001 - $15,000"
    amount_midpoint: float
    current_price: float
    entry_price: float
    return_pct: float
    holding_period_days: int
    committee_relevance: float  # 0-1 score


@dataclass
class SenatorProfile:
    """Senator profile with committee information and scoring metadata"""
    name: str
    committees: List[CommitteeType]
    party: str
    state: str
    years_in_office: int
    total_trades: int
    avg_disclosure_delay: float  # days
    committee_relevance_score: float  # 0-1


class SenatorScorer:
    """
    Main scoring engine for ranking senators based on trading performance
    and disclosure patterns with Kelly Criterion position sizing.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, 
                 lookback_period_days: int = 365):
        """
        Initialize the scorer with configurable parameters.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            lookback_period_days: Period to consider for performance metrics
        """
        self.risk_free_rate = risk_free_rate
        self.lookback_period_days = lookback_period_days
        
        # Committee relevance weights for insider information potential
        self.committee_weights = {
            CommitteeType.BANKING: 0.95,
            CommitteeType.FINANCE: 0.90,
            CommitteeType.COMMERCE: 0.85,
            CommitteeType.ENERGY: 0.80,
            CommitteeType.HEALTH: 0.75,
            CommitteeType.JUDICIARY: 0.70,
            CommitteeType.INTELLIGENCE: 0.90,
            CommitteeType.ARMED_SERVICES: 0.65,
            CommitteeType.AGRICULTURE: 0.60,
            CommitteeType.HOMELAND_SECURITY: 0.70,
            CommitteeType.OTHER: 0.30
        }
        
        # Scoring weights for final composite score
        self.scoring_weights = {
            'performance_score': 0.35,
            'win_rate_score': 0.25,
            'disclosure_speed_score': 0.20,
            'committee_relevance_score': 0.15,
            'consistency_score': 0.05
        }
    
    def calculate_performance_metrics(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a senator's trades.
        
        Args:
            trades: List of trade records for the senator
            
        Returns:
            Dictionary containing performance metrics
        """
        if not trades:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'volatility': 0.0
            }
        
        returns = [trade.return_pct for trade in trades]
        
        # Basic performance metrics
        total_return = np.sum(returns)
        avg_return = np.mean(returns)
        
        # Win rate calculation
        winning_trades = [r for r in returns if r > 0]
        win_rate = len(winning_trades) / len(returns) if returns else 0
        
        # Annualized return (assuming trades represent portfolio performance)
        avg_holding_period = np.mean([trade.holding_period_days for trade in trades])
        periods_per_year = 365 / max(avg_holding_period, 1)
        annualized_return = (1 + avg_return) ** periods_per_year - 1
        
        # Volatility and Sharpe ratio
        volatility = np.std(returns) if len(returns) > 1 else 0
        annualized_volatility = volatility * np.sqrt(periods_per_year)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown calculation
        cumulative_returns = np.cumsum(returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = rolling_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = sum(winning_trades)
        losing_trades = [r for r in returns if r < 0]
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.001  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'volatility': annualized_volatility
        }
    
    def calculate_disclosure_speed_score(self, trades: List[TradeRecord]) -> float:
        """
        Calculate disclosure speed score based on how quickly senators report trades.
        
        Args:
            trades: List of trade records
            
        Returns:
            Normalized score (0-1) where 1 is fastest disclosure
        """
        if not trades:
            return 0.0
        
        # Calculate average disclosure delay
        delays = []
        for trade in trades:
            delay = (trade.disclosure_date - trade.transaction_date).days
            delays.append(delay)
        
        avg_delay = np.mean(delays)
        
        # STOCK Act requires disclosure within 45 days
        # Score inversely proportional to delay, with bonus for early disclosure
        if avg_delay <= 1:
            return 1.0  # Same day or next day disclosure gets perfect score
        elif avg_delay <= 7:
            return 0.9  # Within a week gets high score
        elif avg_delay <= 30:
            return 0.7 - (avg_delay - 7) * 0.02  # Linear decay
        elif avg_delay <= 45:
            return 0.3 - (avg_delay - 30) * 0.01  # Slower decay for legal compliance
        else:
            return max(0.1, 0.2 - (avg_delay - 45) * 0.002)  # Penalty for late disclosure
    
    def calculate_committee_relevance_score(self, senator: SenatorProfile, 
                                          trades: List[TradeRecord]) -> float:
        """
        Calculate committee relevance score based on insider information potential.
        
        Args:
            senator: Senator profile with committee information
            trades: List of trade records
            
        Returns:
            Normalized score (0-1) based on committee relevance to trades
        """
        if not senator.committees or not trades:
            return 0.0
        
        # Calculate weighted committee score
        committee_score = 0.0
        for committee in senator.committees:
            weight = self.committee_weights.get(committee, 0.3)
            committee_score += weight
        
        # Normalize by number of committees (average)
        base_score = committee_score / len(senator.committees)
        
        # Bonus for high-value committees
        high_value_committees = [
            CommitteeType.BANKING, CommitteeType.FINANCE, 
            CommitteeType.INTELLIGENCE, CommitteeType.COMMERCE
        ]
        
        high_value_count = sum(1 for c in senator.committees if c in high_value_committees)
        high_value_bonus = min(0.2, high_value_count * 0.1)
        
        return min(1.0, base_score + high_value_bonus)
    
    def calculate_kelly_criterion(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """
        Calculate Kelly Criterion for optimal position sizing based on trade history.
        
        Args:
            trades: List of trade records
            
        Returns:
            Dictionary with Kelly percentage and related metrics
        """
        if len(trades) < 10:  # Need sufficient sample size
            return {
                'kelly_percentage': 0.0,
                'half_kelly': 0.0,
                'quarter_kelly': 0.0,
                'confidence_level': 0.0
            }
        
        returns = [trade.return_pct for trade in trades]
        
        # Calculate win probability and average win/loss
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        if not winning_returns or not losing_returns:
            return {
                'kelly_percentage': 0.0,
                'half_kelly': 0.0,
                'quarter_kelly': 0.0,
                'confidence_level': 0.0
            }
        
        win_probability = len(winning_returns) / len(returns)
        avg_win = np.mean(winning_returns)
        avg_loss = abs(np.mean(losing_returns))
        
        # Kelly Criterion: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_probability, q = 1 - p
        b = avg_win / avg_loss if avg_loss > 0 else 0
        p = win_probability
        q = 1 - p
        
        kelly_f = (b * p - q) / b if b > 0 else 0
        
        # Ensure Kelly percentage is reasonable (cap at 25%)
        kelly_percentage = max(0, min(0.25, kelly_f))
        
        # Conservative variations
        half_kelly = kelly_percentage * 0.5
        quarter_kelly = kelly_percentage * 0.25
        
        # Confidence level based on sample size and consistency
        confidence_level = min(1.0, len(trades) / 50) * (1 - np.std(returns))
        
        return {
            'kelly_percentage': kelly_percentage,
            'half_kelly': half_kelly,
            'quarter_kelly': quarter_kelly,
            'confidence_level': max(0, confidence_level)
        }
    
    def calculate_consistency_score(self, trades: List[TradeRecord]) -> float:
        """
        Calculate consistency score based on return stability and drawdown recovery.
        
        Args:
            trades: List of trade records
            
        Returns:
            Normalized consistency score (0-1)
        """
        if len(trades) < 5:
            return 0.0
        
        returns = [trade.return_pct for trade in trades]
        
        # Consistency metrics
        return_std = np.std(returns)
        return_mean = np.mean(returns)
        
        # Coefficient of variation (lower is better for positive returns)
        if return_mean > 0:
            cv = return_std / return_mean
            cv_score = max(0, 1 - cv)  # Lower CV = higher score
        else:
            cv_score = 0
        
        # Drawdown recovery analysis
        cumulative_returns = np.cumsum(returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = rolling_max - cumulative_returns
        
        # Percentage of time in drawdown
        time_in_drawdown = np.sum(drawdowns > 0) / len(drawdowns)
        drawdown_score = 1 - time_in_drawdown
        
        # Combined consistency score
        consistency_score = 0.6 * cv_score + 0.4 * drawdown_score
        
        return max(0, min(1, consistency_score))
    
    def score_senator(self, senator: SenatorProfile, 
                     trades: List[TradeRecord]) -> Dict[str, float]:
        """
        Calculate comprehensive score for a senator based on all factors.
        
        Args:
            senator: Senator profile
            trades: List of trade records
            
        Returns:
            Dictionary with component scores and final composite score
        """
        # Calculate individual component scores
        performance_metrics = self.calculate_performance_metrics(trades)
        
        # Performance score (normalized Sharpe ratio and return)
        sharpe_component = min(1, max(0, performance_metrics['sharpe_ratio'] / 2))
        return_component = min(1, max(0, performance_metrics['annualized_return'] * 2))
        performance_score = 0.6 * sharpe_component + 0.4 * return_component
        
        # Win rate score
        win_rate_score = performance_metrics['win_rate']
        
        # Disclosure speed score
        disclosure_speed_score = self.calculate_disclosure_speed_score(trades)
        
        # Committee relevance score
        committee_relevance_score = self.calculate_committee_relevance_score(senator, trades)
        
        # Consistency score
        consistency_score = self.calculate_consistency_score(trades)
        
        # Kelly Criterion analysis
        kelly_metrics = self.calculate_kelly_criterion(trades)
        
        # Calculate composite score
        composite_score = (
            self.scoring_weights['performance_score'] * performance_score +
            self.scoring_weights['win_rate_score'] * win_rate_score +
            self.scoring_weights['disclosure_speed_score'] * disclosure_speed_score +
            self.scoring_weights['committee_relevance_score'] * committee_relevance_score +
            self.scoring_weights['consistency_score'] * consistency_score
        )
        
        return {
            'composite_score': composite_score,
            'performance_score': performance_score,
            'win_rate_score': win_rate_score,
            'disclosure_speed_score': disclosure_speed_score,
            'committee_relevance_score': committee_relevance_score,
            'consistency_score': consistency_score,
            'kelly_percentage': kelly_metrics['kelly_percentage'],
            'half_kelly': kelly_metrics['half_kelly'],
            'quarter_kelly': kelly_metrics['quarter_kelly'],
            'kelly_confidence': kelly_metrics['confidence_level'],
            'sharpe_ratio': performance_metrics['sharpe_ratio'],
            'annualized_return': performance_metrics['annualized_return'],
            'win_rate': performance_metrics['win_rate'],
            'max_drawdown': performance_metrics['max_drawdown'],
            'total_trades': len(trades)
        }
    
    def rank_senators(self, senators_data: Dict[str, Tuple[SenatorProfile, List[TradeRecord]]]) -> pd.DataFrame:
        """
        Rank all senators based on their comprehensive scores.
        
        Args:
            senators_data: Dictionary mapping senator names to (profile, trades) tuples
            
        Returns:
            DataFrame with ranked senators and their scores
        """
        results = []
        
        for senator_name, (profile, trades) in senators_data.items():
            try:
                scores = self.score_senator(profile, trades)
                
                result = {
                    'senator_name': senator_name,
                    'party': profile.party,
                    'state': profile.state,
                    'years_in_office': profile.years_in_office,
                    **scores
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error scoring senator {senator_name}: {e}")
                continue
        
        # Create DataFrame and sort by composite score
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
            df['rank'] = df.index + 1
        
        return df
    
    def get_position_sizing_recommendation(self, senator_score: Dict[str, float], 
                                         portfolio_value: float,
                                         max_position_size: float = 0.10) -> Dict[str, float]:
        """
        Get position sizing recommendation based on Kelly Criterion and senator score.
        
        Args:
            senator_score: Score dictionary from score_senator method
            portfolio_value: Total portfolio value
            max_position_size: Maximum allowed position size as percentage
            
        Returns:
            Position sizing recommendations
        """
        base_kelly = senator_score.get('kelly_percentage', 0)
        confidence = senator_score.get('kelly_confidence', 0)
        composite_score = senator_score.get('composite_score', 0)
        
        # Adjust Kelly based on confidence and composite score
        adjusted_kelly = base_kelly * confidence * composite_score
        
        # Apply maximum position size constraint
        recommended_size = min(adjusted_kelly, max_position_size)
        
        # Conservative alternatives
        conservative_size = recommended_size * 0.5
        aggressive_size = min(recommended_size * 1.5, max_position_size)
        
        return {
            'recommended_percentage': recommended_size,
            'recommended_dollar_amount': portfolio_value * recommended_size,
            'conservative_percentage': conservative_size,
            'conservative_dollar_amount': portfolio_value * conservative_size,
            'aggressive_percentage': aggressive_size,
            'aggressive_dollar_amount': portfolio_value * aggressive_size,
            'base_kelly': base_kelly,
            'confidence_adjusted_kelly': adjusted_kelly
        }


# Example usage and testing functions
def create_sample_data() -> Dict[str, Tuple[SenatorProfile, List[TradeRecord]]]:
    """Create sample data for testing the scoring system"""
    
    # Sample senator profiles
    senator1 = SenatorProfile(
        name="John Smith",
        committees=[CommitteeType.BANKING, CommitteeType.FINANCE],
        party="Republican",
        state="TX",
        years_in_office=8,
        total_trades=45,
        avg_disclosure_delay=12.5,
        committee_relevance_score=0.85
    )
    
    senator2 = SenatorProfile(
        name="Jane Doe",
        committees=[CommitteeType.HEALTH, CommitteeType.COMMERCE],
        party="Democrat",
        state="CA",
        years_in_office=6,
        total_trades=32,
        avg_disclosure_delay=8.2,
        committee_relevance_score=0.75
    )
    
    # Sample trade records
    base_date = datetime(2023, 1, 1)
    
    trades1 = []
    for i in range(20):
        trade_date = base_date + timedelta(days=i*10)
        disclosure_date = trade_date + timedelta(days=np.random.randint(1, 30))
        
        trades1.append(TradeRecord(
            senator_name="John Smith",
            ticker=f"STOCK{i%5}",
            transaction_date=trade_date,
            disclosure_date=disclosure_date,
            transaction_type="buy",
            amount_range="$15,001 - $50,000",
            amount_midpoint=32500,
            current_price=100 + np.random.normal(0, 10),
            entry_price=100,
            return_pct=np.random.normal(0.05, 0.15),  # 5% average return, 15% volatility
            holding_period_days=np.random.randint(30, 180),
            committee_relevance=0.8
        ))
    
    trades2 = []
    for i in range(15):
        trade_date = base_date + timedelta(days=i*15)
        disclosure_date = trade_date + timedelta(days=np.random.randint(1, 20))
        
        trades2.append(TradeRecord(
            senator_name="Jane Doe",
            ticker=f"STOCK{i%3}",
            transaction_date=trade_date,
            disclosure_date=disclosure_date,
            transaction_type="buy",
            amount_range="$1,001 - $15,000",
            amount_midpoint=8000,
            current_price=50 + np.random.normal(0, 5),
            entry_price=50,
            return_pct=np.random.normal(0.03, 0.12),  # 3% average return, 12% volatility
            holding_period_days=np.random.randint(20, 120),
            committee_relevance=0.7
        ))
    
    return {
        "John Smith": (senator1, trades1),
        "Jane Doe": (senator2, trades2)
    }


def main():
    """Main function for testing the senator scoring system"""
    
    # Initialize scorer
    scorer = SenatorScorer(risk_free_rate=0.02, lookback_period_days=365)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Rank senators
    rankings = scorer.rank_senators(sample_data)
    
    print("Senator Rankings:")
    print("=" * 80)
    print(rankings[['rank', 'senator_name', 'party', 'state', 'composite_score', 
                   'kelly_percentage', 'win_rate', 'sharpe_ratio']].to_string(index=False))
    
    print("\n\nDetailed Scores for Top Senator:")
    print("=" * 80)
    if not rankings.empty:
        top_senator = rankings.iloc[0]
        for col in rankings.columns:
            if col not in ['rank', 'senator_name', 'party', 'state']:
                print(f"{col}: {top_senator[col]:.4f}")
    
    # Position sizing example
    if not rankings.empty:
        print("\n\nPosition Sizing Recommendations (for $100,000 portfolio):")
        print("=" * 80)
        top_score = rankings.iloc[0].to_dict()
        position_recs = scorer.get_position_sizing_recommendation(top_score, 100000)
        
        for key, value in position_recs.items():
            if 'percentage' in key:
                print(f"{key}: {value:.2%}")
            elif 'dollar' in key:
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()