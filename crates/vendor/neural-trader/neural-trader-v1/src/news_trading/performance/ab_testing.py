"""A/B testing framework for trading strategies."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats

from .models import TradeResult


@dataclass
class StrategyVariant:
    """Represents a strategy variant in an A/B test."""
    
    name: str
    config: Dict[str, Any]
    trades: List[TradeResult] = field(default_factory=list)
    
    def add_trade(self, trade: TradeResult) -> None:
        """Add a trade result to this variant."""
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for this variant."""
        if not self.trades:
            return {
                "total_trades": 0,
                "average_pnl": 0.0,
                "std_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }
        
        pnls = [t.pnl for t in self.trades]
        winning_trades = [t for t in self.trades if t.pnl > 0]
        
        # Calculate Sharpe ratio (simplified daily)
        # If pnl_percentage not set, calculate it
        returns = []
        for t in self.trades:
            if t.pnl_percentage != 0:
                returns.append(t.pnl_percentage)
            elif t.position_size > 0 and t.entry_price > 0:
                # Calculate percentage return
                pct = (t.pnl / (t.entry_price * t.position_size)) * 100
                returns.append(pct)
        
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if returns and len(returns) > 1 and np.std(returns) > 0
            else 0.0
        )
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-10)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            "total_trades": len(self.trades),
            "average_pnl": np.mean(pnls),
            "std_pnl": np.std(pnls),
            "win_rate": len(winning_trades) / len(self.trades),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_pnl": sum(pnls),
        }


class ABTestFramework:
    """Framework for A/B testing trading strategies."""
    
    def __init__(self, test_name: str, min_samples_per_variant: int = 30):
        """Initialize A/B test framework.
        
        Args:
            test_name: Name of the test
            min_samples_per_variant: Minimum samples needed for significance testing
        """
        self.test_name = test_name
        self.variants: Dict[str, StrategyVariant] = {}
        self.start_time = datetime.now()
        self.min_samples_per_variant = min_samples_per_variant
    
    def add_variant(self, name: str, config: Dict[str, Any]) -> None:
        """Add a strategy variant to the test.
        
        Args:
            name: Variant name
            config: Configuration parameters for this variant
        """
        self.variants[name] = StrategyVariant(name=name, config=config)
    
    def record_result(self, variant: str, trade: TradeResult) -> None:
        """Record a trade result for a variant.
        
        Args:
            variant: Variant name
            trade: Trade result to record
        """
        if variant in self.variants:
            self.variants[variant].add_trade(trade)
    
    def analyze(self, include_time_analysis: bool = False) -> Dict[str, Any]:
        """Analyze test results.
        
        Args:
            include_time_analysis: Whether to include time-based analysis
            
        Returns:
            Analysis results including metrics and statistical tests
        """
        results = {}
        
        # Calculate metrics for each variant
        for name, variant in self.variants.items():
            results[name] = variant.calculate_metrics()
        
        # Statistical significance test if we have two variants
        if len(self.variants) == 2:
            variant_names = list(self.variants.keys())
            variant_a = self.variants[variant_names[0]]
            variant_b = self.variants[variant_names[1]]
            
            if (len(variant_a.trades) >= self.min_samples_per_variant and
                len(variant_b.trades) >= self.min_samples_per_variant):
                
                pnls_a = [t.pnl for t in variant_a.trades]
                pnls_b = [t.pnl for t in variant_b.trades]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(pnls_a, pnls_b)
                
                # Determine winner
                avg_a = np.mean(pnls_a)
                avg_b = np.mean(pnls_b)
                winner = variant_names[0] if avg_a > avg_b else variant_names[1]
                
                results["statistical_significance"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "confidence": 1 - p_value,
                }
                results["winner"] = winner if p_value < 0.05 else "no_winner"
        
        # Time-based analysis
        if include_time_analysis:
            results["time_series"] = self._generate_time_series()
        
        return results
    
    def _generate_time_series(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate time series data for each variant."""
        time_series = {}
        
        for name, variant in self.variants.items():
            # Sort trades by time
            sorted_trades = sorted(variant.trades, key=lambda t: t.entry_time)
            
            series = []
            cumulative_pnl = 0
            
            for trade in sorted_trades:
                cumulative_pnl += trade.pnl
                series.append({
                    "timestamp": trade.entry_time.isoformat(),
                    "pnl": trade.pnl,
                    "cumulative_pnl": cumulative_pnl,
                    "trade_id": trade.trade_id,
                })
            
            time_series[name] = series
        
        return time_series


class MultiArmedBandit:
    """Multi-armed bandit for dynamic strategy selection."""
    
    def __init__(
        self,
        strategies: List[str],
        algorithm: str = "epsilon_greedy",
        exploration_rate: float = 0.1,
        use_context: bool = False,
    ):
        """Initialize multi-armed bandit.
        
        Args:
            strategies: List of strategy names
            algorithm: Algorithm to use (epsilon_greedy, thompson_sampling, ucb)
            exploration_rate: Exploration rate for epsilon-greedy
            use_context: Whether to use contextual information
        """
        self.strategies = strategies
        self.algorithm = algorithm
        self.exploration_rate = exploration_rate
        self.use_context = use_context
        
        # Initialize strategy statistics
        self.strategy_stats = {
            strategy: {
                "selections": 0,
                "total_reward": 0.0,
                "sum_squares": 0.0,  # For variance calculation
                "alpha": 1.0,  # For Thompson sampling
                "beta": 1.0,   # For Thompson sampling
            }
            for strategy in strategies
        }
        
        # Context-based models if needed
        if use_context:
            self.context_models = {strategy: [] for strategy in strategies}
    
    def select_strategy(
        self,
        context: Optional[Dict[str, float]] = None,
        exploit_only: bool = False,
    ) -> str:
        """Select a strategy using the bandit algorithm.
        
        Args:
            context: Optional context information
            exploit_only: Whether to only exploit (no exploration)
            
        Returns:
            Selected strategy name
        """
        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy_select(exploit_only, context)
        elif self.algorithm == "thompson_sampling":
            return self._thompson_sampling_select()
        elif self.algorithm == "ucb":
            return self._ucb_select()
        else:
            # Default to epsilon-greedy
            return self._epsilon_greedy_select(exploit_only, context)
    
    def update(
        self,
        strategy: str,
        reward: float,
        context: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update strategy statistics based on reward.
        
        Args:
            strategy: Strategy that was selected
            reward: Reward received
            context: Optional context when the decision was made
        """
        if strategy not in self.strategy_stats:
            return
        
        stats = self.strategy_stats[strategy]
        stats["selections"] += 1
        stats["total_reward"] += reward
        stats["sum_squares"] += reward ** 2
        
        # Update Thompson sampling parameters
        if reward > 0:
            stats["alpha"] += 1
        else:
            stats["beta"] += 1
        
        # Store context-reward pairs if using context
        if self.use_context and context:
            self.context_models[strategy].append((context, reward))
    
    def get_selection_counts(self) -> Dict[str, int]:
        """Get number of times each strategy was selected."""
        return {
            strategy: stats["selections"]
            for strategy, stats in self.strategy_stats.items()
        }
    
    def get_strategy_estimates(self) -> Dict[str, Dict[str, float]]:
        """Get estimated performance for each strategy."""
        estimates = {}
        
        for strategy, stats in self.strategy_stats.items():
            if stats["selections"] > 0:
                mean = stats["total_reward"] / stats["selections"]
                variance = (
                    stats["sum_squares"] / stats["selections"] - mean ** 2
                )
                std = np.sqrt(max(0, variance))
                
                estimates[strategy] = {
                    "mean": mean,
                    "std": std,
                    "confidence": 1 - 1 / (1 + stats["selections"] / 10),  # Confidence grows with samples
                    "selections": stats["selections"],
                }
            else:
                estimates[strategy] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "confidence": 0.0,
                    "selections": 0,
                }
        
        return estimates
    
    def _epsilon_greedy_select(self, exploit_only: bool = False, context: Optional[Dict[str, float]] = None) -> str:
        """Select using epsilon-greedy algorithm."""
        if not exploit_only and np.random.random() < self.exploration_rate:
            # Explore: random selection
            return np.random.choice(self.strategies)
        else:
            # Exploit: select best performing
            best_reward = -float('inf')
            best_strategy = self.strategies[0]
            
            # If using context, calculate expected rewards based on context
            if self.use_context and context and any(self.context_models[s] for s in self.strategies):
                for strategy in self.strategies:
                    if self.context_models[strategy]:
                        # Calculate expected reward given context
                        expected_reward = self._estimate_contextual_reward(strategy, context)
                        if expected_reward > best_reward:
                            best_reward = expected_reward
                            best_strategy = strategy
            else:
                # Standard selection without context
                for strategy, stats in self.strategy_stats.items():
                    if stats["selections"] > 0:
                        avg_reward = stats["total_reward"] / stats["selections"]
                        if avg_reward > best_reward:
                            best_reward = avg_reward
                            best_strategy = strategy
                    else:
                        # Unseen strategy gets priority
                        return strategy
            
            return best_strategy
    
    def _thompson_sampling_select(self) -> str:
        """Select using Thompson sampling."""
        samples = {}
        
        for strategy, stats in self.strategy_stats.items():
            # Sample from Beta distribution
            sample = np.random.beta(stats["alpha"], stats["beta"])
            samples[strategy] = sample
        
        # Select strategy with highest sample
        return max(samples, key=samples.get)
    
    def _ucb_select(self) -> str:
        """Select using Upper Confidence Bound algorithm."""
        total_selections = sum(
            stats["selections"] for stats in self.strategy_stats.values()
        )
        
        if total_selections == 0:
            return np.random.choice(self.strategies)
        
        ucb_scores = {}
        
        for strategy, stats in self.strategy_stats.items():
            if stats["selections"] == 0:
                # Unseen strategy gets infinite UCB
                return strategy
            
            avg_reward = stats["total_reward"] / stats["selections"]
            exploration_bonus = np.sqrt(
                2 * np.log(total_selections) / stats["selections"]
            )
            
            ucb_scores[strategy] = avg_reward + exploration_bonus
        
        # Select strategy with highest UCB
        return max(ucb_scores, key=ucb_scores.get)
    
    def _estimate_contextual_reward(self, strategy: str, context: Dict[str, float]) -> float:
        """Estimate reward for a strategy given context.
        
        Args:
            strategy: Strategy name
            context: Context features
            
        Returns:
            Estimated reward
        """
        if not self.context_models[strategy]:
            return 0.0
        
        # Simple weighted average based on context similarity
        # In a real implementation, this would use a proper ML model
        context_data = self.context_models[strategy]
        
        if not context_data:
            return 0.0
        
        # Calculate weighted average based on context similarity
        total_weight = 0.0
        weighted_reward = 0.0
        
        for past_context, reward in context_data[-50:]:  # Use last 50 observations
            # Calculate similarity (simple Euclidean distance)
            similarity = 1.0
            for key in context:
                if key in past_context:
                    diff = abs(context[key] - past_context[key])
                    similarity *= np.exp(-diff)  # Exponential decay with distance
            
            total_weight += similarity
            weighted_reward += similarity * reward
        
        if total_weight > 0:
            return weighted_reward / total_weight
        else:
            return self.strategy_stats[strategy]["total_reward"] / max(1, self.strategy_stats[strategy]["selections"])