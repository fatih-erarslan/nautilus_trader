"""Parameter optimization system for mirror trading strategy."""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import ParameterGrid
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from src.trading.strategies.mirror_trader import MirrorTradingEngine
from benchmark.src.benchmarks.strategy_benchmark import StrategyBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParameterSet:
    """Complete parameter set for mirror trading optimization."""
    
    # Institution confidence scores
    berkshire_confidence: float = 0.95
    bridgewater_confidence: float = 0.85
    renaissance_confidence: float = 0.90
    soros_confidence: float = 0.80
    tiger_confidence: float = 0.75
    third_point_confidence: float = 0.70
    pershing_confidence: float = 0.75
    appaloosa_confidence: float = 0.80
    
    # Position sizing limits
    max_position_pct: float = 0.03
    min_position_pct: float = 0.005
    
    # Confidence multipliers for different actions
    increased_position_multiplier: float = 0.8
    sold_position_multiplier: float = 0.9
    reduced_position_multiplier: float = 0.6
    
    # Entry timing thresholds
    immediate_entry_days: int = 2
    immediate_entry_price_threshold: float = 0.015
    prompt_entry_days: int = 7
    prompt_entry_price_threshold: float = 0.05
    wait_pullback_threshold: float = 0.15
    max_chase_immediate: float = 1.015
    max_chase_prompt: float = 1.03
    max_chase_pullback: float = 1.05
    
    # Risk management thresholds
    take_profit_threshold: float = 0.30
    stop_loss_threshold: float = -0.15
    long_term_profit_threshold: float = 0.15
    long_term_days: int = 365
    
    # Position scaling
    institutional_position_scale: float = 0.2
    
    # Insider transaction confidence scores
    ceo_confidence: float = 0.9
    cfo_confidence: float = 0.8
    president_confidence: float = 0.85
    director_confidence: float = 0.7
    officer_confidence: float = 0.65
    owner_10pct_confidence: float = 0.75
    
    # Transaction size multipliers
    large_transaction_threshold: int = 100000
    large_transaction_multiplier: float = 1.1
    small_transaction_threshold: int = 1000
    small_transaction_multiplier: float = 0.8
    
    # Timing score parameters
    time_penalty_window: int = 14
    price_change_penalty_multiplier: float = 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSet':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    
    best_parameters: ParameterSet
    best_score: float
    optimization_history: List[Dict]
    performance_metrics: Dict[str, float]
    improvement_over_baseline: float
    optimization_time: float

class MirrorTraderParameterOptimizer:
    """Advanced parameter optimizer for mirror trading strategy."""
    
    def __init__(self, portfolio_size: float = 100000):
        """Initialize optimizer."""
        self.portfolio_size = portfolio_size
        self.benchmark = StrategyBenchmark({})
        self.optimization_history = []
        
    def create_optimized_engine(self, params: ParameterSet) -> MirrorTradingEngine:
        """Create mirror trading engine with optimized parameters."""
        engine = MirrorTradingEngine(self.portfolio_size)
        
        # Update institution confidence scores
        engine.trusted_institutions = {
            "Berkshire Hathaway": params.berkshire_confidence,
            "Bridgewater Associates": params.bridgewater_confidence,
            "Renaissance Technologies": params.renaissance_confidence,
            "Soros Fund Management": params.soros_confidence,
            "Tiger Global": params.tiger_confidence,
            "Third Point": params.third_point_confidence,
            "Pershing Square": params.pershing_confidence,
            "Appaloosa Management": params.appaloosa_confidence
        }
        
        # Update position sizing
        engine.max_position_pct = params.max_position_pct
        engine.min_position_pct = params.min_position_pct
        
        return engine
    
    def objective_function(self, param_array: np.ndarray) -> float:
        """Objective function for optimization (negative Sharpe ratio for minimization)."""
        try:
            # Convert parameter array to ParameterSet
            params = self._array_to_parameters(param_array)
            
            # Create optimized engine
            engine = self.create_optimized_engine(params)
            
            # Run simulation with multiple scenarios
            total_score = 0
            scenarios = ['bull', 'bear', 'sideways', 'volatile']
            weights = [0.3, 0.2, 0.3, 0.2]  # Weight scenarios
            
            for scenario, weight in zip(scenarios, weights):
                score = self._evaluate_scenario(engine, params, scenario)
                total_score += score * weight
            
            # Store in history
            self.optimization_history.append({
                'parameters': params.to_dict(),
                'score': total_score,
                'timestamp': time.time()
            })
            
            # Return negative score for minimization
            return -total_score
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1000.0  # Large penalty for errors
    
    def _evaluate_scenario(self, engine: MirrorTradingEngine, params: ParameterSet, scenario: str) -> float:
        """Evaluate parameters in a specific market scenario."""
        # Generate synthetic trading data for scenario
        performance_metrics = self._simulate_mirror_trading(engine, params, scenario)
        
        # Calculate composite score
        # Focus on Sharpe ratio with penalties for excessive drawdown
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0)
        win_rate = performance_metrics.get('win_rate', 0)
        
        # Penalize high drawdown and low win rate
        drawdown_penalty = max(0, max_drawdown - 0.15) * 10  # Penalty if drawdown > 15%
        win_rate_bonus = (win_rate - 0.5) * 2  # Bonus for win rate > 50%
        
        score = sharpe_ratio + win_rate_bonus - drawdown_penalty
        return max(score, -10)  # Floor the score
    
    def _simulate_mirror_trading(self, engine: MirrorTradingEngine, params: ParameterSet, scenario: str) -> Dict[str, float]:
        """Simulate mirror trading with given parameters."""
        # Generate synthetic institutional filings and market data
        num_days = 252  # One year
        trades = []
        
        # Simulate institutional actions
        for day in range(num_days):
            if np.random.random() < 0.05:  # 5% chance of institutional action per day
                # Generate random filing
                filing = self._generate_synthetic_filing(scenario)
                signals = engine.parse_13f_filing(filing)
                
                # Simulate trades based on signals
                for signal in signals:
                    if signal['confidence'] > 0.6:  # Only trade high confidence signals
                        trade_return = self._simulate_trade_outcome(signal, scenario, params)
                        trades.append(trade_return)
        
        # Calculate performance metrics
        if not trades:
            return {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
        
        returns = np.array(trades)
        win_rate = np.sum(returns > 0) / len(returns)
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = (avg_return * 252) / (volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'volatility': volatility
        }
    
    def _generate_synthetic_filing(self, scenario: str) -> Dict:
        """Generate synthetic institutional filing."""
        institutions = ["Berkshire Hathaway", "Bridgewater Associates", "Renaissance Technologies", 
                       "Soros Fund Management", "Tiger Global", "Third Point"]
        
        institution = np.random.choice(institutions)
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        actions = ["new_positions", "increased_positions", "reduced_positions", "sold_positions"]
        action_probs = [0.3, 0.3, 0.2, 0.2] if scenario != 'bear' else [0.1, 0.2, 0.3, 0.4]
        
        filing = {
            "filer": institution,
            "quarter": "2024Q1",
            "holdings": []
        }
        
        # Add random actions
        for action in actions:
            if np.random.random() < dict(zip(actions, action_probs))[action]:
                num_tickers = np.random.randint(1, 4)
                filing[action] = np.random.choice(tickers, num_tickers, replace=False).tolist()
        
        return filing
    
    def _simulate_trade_outcome(self, signal: Dict, scenario: str, params: ParameterSet) -> float:
        """Simulate trade outcome based on signal and market scenario."""
        base_return = 0.02  # 2% base expected return
        
        # Adjust based on scenario
        scenario_multipliers = {
            'bull': 1.5,
            'bear': -0.8,
            'sideways': 0.3,
            'volatile': 1.0
        }
        
        # Adjust based on signal confidence
        confidence_factor = signal['confidence']
        
        # Adjust based on action type
        action_multipliers = {
            'buy': 1.0,
            'sell': -1.0,  # Short selling
            'reduce': -0.5
        }
        
        # Calculate expected return
        expected_return = (base_return * scenario_multipliers.get(scenario, 1.0) * 
                         confidence_factor * action_multipliers.get(signal['action'], 1.0))
        
        # Add noise
        noise = np.random.normal(0, 0.05)  # 5% volatility
        actual_return = expected_return + noise
        
        # Apply risk management
        if actual_return > params.take_profit_threshold:
            actual_return = params.take_profit_threshold  # Take profit
        elif actual_return < params.stop_loss_threshold:
            actual_return = params.stop_loss_threshold  # Stop loss
        
        return actual_return
    
    def _array_to_parameters(self, param_array: np.ndarray) -> ParameterSet:
        """Convert optimization array to ParameterSet."""
        return ParameterSet(
            berkshire_confidence=param_array[0],
            bridgewater_confidence=param_array[1],
            renaissance_confidence=param_array[2],
            soros_confidence=param_array[3],
            tiger_confidence=param_array[4],
            third_point_confidence=param_array[5],
            pershing_confidence=param_array[6],
            appaloosa_confidence=param_array[7],
            max_position_pct=param_array[8],
            min_position_pct=param_array[9],
            increased_position_multiplier=param_array[10],
            sold_position_multiplier=param_array[11],
            reduced_position_multiplier=param_array[12],
            take_profit_threshold=param_array[13],
            stop_loss_threshold=param_array[14],
            institutional_position_scale=param_array[15]
        )
    
    def _parameters_to_array(self, params: ParameterSet) -> np.ndarray:
        """Convert ParameterSet to optimization array."""
        return np.array([
            params.berkshire_confidence,
            params.bridgewater_confidence,
            params.renaissance_confidence,
            params.soros_confidence,
            params.tiger_confidence,
            params.third_point_confidence,
            params.pershing_confidence,
            params.appaloosa_confidence,
            params.max_position_pct,
            params.min_position_pct,
            params.increased_position_multiplier,
            params.sold_position_multiplier,
            params.reduced_position_multiplier,
            params.take_profit_threshold,
            params.stop_loss_threshold,
            params.institutional_position_scale
        ])
    
    def optimize_parameters(self, method: str = 'differential_evolution') -> OptimizationResult:
        """Optimize parameters using specified method."""
        start_time = time.time()
        
        # Get baseline performance
        baseline_params = ParameterSet()
        baseline_score = -self.objective_function(self._parameters_to_array(baseline_params))
        
        logger.info(f"Baseline score: {baseline_score:.4f}")
        
        # Define parameter bounds
        bounds = [
            (0.7, 1.0),   # berkshire_confidence
            (0.6, 0.95),  # bridgewater_confidence  
            (0.7, 1.0),   # renaissance_confidence
            (0.6, 0.9),   # soros_confidence
            (0.6, 0.9),   # tiger_confidence
            (0.5, 0.8),   # third_point_confidence
            (0.6, 0.9),   # pershing_confidence
            (0.6, 0.9),   # appaloosa_confidence
            (0.01, 0.05), # max_position_pct
            (0.001, 0.01), # min_position_pct
            (0.5, 1.0),   # increased_position_multiplier
            (0.7, 1.0),   # sold_position_multiplier
            (0.3, 0.8),   # reduced_position_multiplier
            (0.15, 0.5),  # take_profit_threshold
            (-0.25, -0.05), # stop_loss_threshold
            (0.1, 0.4)    # institutional_position_scale
        ]
        
        if method == 'differential_evolution':
            # Differential Evolution optimization
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                disp=True
            )
            
            optimal_params = self._array_to_parameters(result.x)
            best_score = -result.fun
            
        elif method == 'grid_search':
            # Grid search optimization (reduced grid for performance)
            best_score = -float('inf')
            optimal_params = baseline_params
            
            param_grid = {
                'berkshire_confidence': [0.90, 0.95, 0.98],
                'max_position_pct': [0.02, 0.03, 0.04],
                'take_profit_threshold': [0.25, 0.30, 0.35],
                'stop_loss_threshold': [-0.20, -0.15, -0.10]
            }
            
            grid = ParameterGrid(param_grid)
            for params_dict in grid:
                test_params = ParameterSet(**{**baseline_params.to_dict(), **params_dict})
                score = -self.objective_function(self._parameters_to_array(test_params))
                
                if score > best_score:
                    best_score = score
                    optimal_params = test_params
        
        optimization_time = time.time() - start_time
        improvement = ((best_score - baseline_score) / abs(baseline_score)) * 100 if baseline_score != 0 else 0
        
        logger.info(f"Optimization completed. Best score: {best_score:.4f}")
        logger.info(f"Improvement over baseline: {improvement:.2f}%")
        
        return OptimizationResult(
            best_parameters=optimal_params,
            best_score=best_score,
            optimization_history=self.optimization_history,
            performance_metrics=self._evaluate_final_performance(optimal_params),
            improvement_over_baseline=improvement,
            optimization_time=optimization_time
        )
    
    def _evaluate_final_performance(self, params: ParameterSet) -> Dict[str, float]:
        """Evaluate final performance with optimized parameters."""
        engine = self.create_optimized_engine(params)
        
        scenarios = ['bull', 'bear', 'sideways', 'volatile']
        metrics = {}
        
        for scenario in scenarios:
            scenario_metrics = self._simulate_mirror_trading(engine, params, scenario)
            for key, value in scenario_metrics.items():
                metrics[f"{scenario}_{key}"] = value
        
        # Calculate weighted average
        weights = {'bull': 0.3, 'bear': 0.2, 'sideways': 0.3, 'volatile': 0.2}
        for metric in ['sharpe_ratio', 'max_drawdown', 'win_rate']:
            weighted_avg = sum(metrics[f"{scenario}_{metric}"] * weights[scenario] 
                             for scenario in scenarios)
            metrics[f"weighted_avg_{metric}"] = weighted_avg
        
        return metrics

def run_parameter_optimization():
    """Run complete parameter optimization."""
    logger.info("Starting Mirror Trading Parameter Optimization")
    
    optimizer = MirrorTraderParameterOptimizer()
    
    # Run optimization with different methods
    methods = ['differential_evolution']  # Start with one method
    
    results = {}
    for method in methods:
        logger.info(f"\nRunning optimization with method: {method}")
        result = optimizer.optimize_parameters(method)
        results[method] = result
        
        logger.info(f"Method: {method}")
        logger.info(f"Best Score: {result.best_score:.4f}")
        logger.info(f"Improvement: {result.improvement_over_baseline:.2f}%")
        logger.info(f"Optimization Time: {result.optimization_time:.2f}s")
    
    # Select best result
    best_method = max(results.keys(), key=lambda k: results[k].best_score)
    best_result = results[best_method]
    
    logger.info(f"\nBest method: {best_method}")
    logger.info("Optimal parameters:")
    for key, value in best_result.best_parameters.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    return best_result

if __name__ == "__main__":
    result = run_parameter_optimization()
    
    # Save results
    with open('/workspaces/ai-news-trader/optimization_results.json', 'w') as f:
        json.dump({
            'best_parameters': result.best_parameters.to_dict(),
            'performance_metrics': result.performance_metrics,
            'improvement': result.improvement_over_baseline,
            'optimization_time': result.optimization_time
        }, f, indent=2)