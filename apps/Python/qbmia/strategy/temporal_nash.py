"""
Temporal-Biological Nash Dynamics implementation with realistic learning constraints.
"""

import numpy as np
import numba as nb
from typing import Dict, Any, List, Optional, Tuple, Deque
import logging
from collections import deque
import asyncio
from scipy import signal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BiologicalConstraints:
    """Biological learning constraints."""
    learning_rate: float = 0.01
    memory_decay: float = 0.95
    attention_span: int = 100
    reaction_delay: int = 5
    noise_level: float = 0.05
    fatigue_rate: float = 0.001

@nb.jit(nopython=True, fastmath=True, cache=True)
def _update_memory_trace_numba(current_trace: np.ndarray, new_experience: np.ndarray,
                              decay_rate: float, learning_rate: float) -> np.ndarray:
    """
    Numba-accelerated memory trace update with biological decay.

    Args:
        current_trace: Current memory trace
        new_experience: New experience to incorporate
        decay_rate: Memory decay rate
        learning_rate: Learning rate

    Returns:
        Updated memory trace
    """
    # Apply decay
    decayed_trace = current_trace * decay_rate

    # Incorporate new experience
    updated_trace = decayed_trace + learning_rate * new_experience

    # Normalize to prevent explosion
    max_val = np.max(np.abs(updated_trace))
    if max_val > 1.0:
        updated_trace = updated_trace / max_val

    return updated_trace

@nb.jit(nopython=True, fastmath=True, cache=True)
def _compute_temporal_correlation_numba(series1: np.ndarray, series2: np.ndarray,
                                      max_lag: int = 20) -> Tuple[np.ndarray, int]:
    """
    Compute temporal correlation with variable lag.

    Args:
        series1: First time series
        series2: Second time series
        max_lag: Maximum lag to consider

    Returns:
        Correlation values and optimal lag
    """
    n = min(len(series1), len(series2))
    correlations = np.zeros(2 * max_lag + 1)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # series2 leads series1
            if n + lag > 0:
                corr = np.corrcoef(series1[:n+lag], series2[-lag:n])[0, 1]
                if not np.isnan(corr):
                    correlations[lag + max_lag] = corr
        elif lag > 0:
            # series1 leads series2
            if n - lag > 0:
                corr = np.corrcoef(series1[lag:n], series2[:n-lag])[0, 1]
                if not np.isnan(corr):
                    correlations[lag + max_lag] = corr
        else:
            # No lag
            corr = np.corrcoef(series1[:n], series2[:n])[0, 1]
            if not np.isnan(corr):
                correlations[max_lag] = corr

    optimal_lag_idx = np.argmax(np.abs(correlations))
    optimal_lag = optimal_lag_idx - max_lag

    return correlations, optimal_lag

class TemporalBiologicalNash:
    """
    Nash equilibrium solver with temporal dynamics and biological constraints.
    """

    def __init__(self, memory_decay: float = 0.95,
                 learning_rate_bounds: Tuple[float, float] = (0.001, 0.1),
                 num_agents: int = 3):
        """
        Initialize Temporal-Biological Nash solver.

        Args:
            memory_decay: Memory decay rate
            learning_rate_bounds: Min and max learning rates
            num_agents: Number of agents to model
        """
        self.memory_decay = memory_decay
        self.learning_rate_bounds = learning_rate_bounds
        self.num_agents = num_agents

        # Initialize agent models
        self.agents = []
        for i in range(num_agents):
            # Heterogeneous learning rates
            learning_rate = np.random.uniform(*learning_rate_bounds)

            agent = {
                'id': i,
                'constraints': BiologicalConstraints(
                    learning_rate=learning_rate,
                    memory_decay=memory_decay,
                    attention_span=np.random.randint(50, 150),
                    reaction_delay=np.random.randint(3, 8)
                ),
                'memory_trace': np.zeros(100),  # Memory buffer
                'strategy_history': deque(maxlen=1000),
                'payoff_history': deque(maxlen=1000),
                'fatigue_level': 0.0
            }
            self.agents.append(agent)

        # Temporal dynamics tracking
        self.time_step = 0
        self.equilibrium_history = []
        self.convergence_metrics = []

        logger.info(f"Temporal-Biological Nash initialized with {num_agents} agents")

    async def find_temporal_equilibrium(self, time_series: Dict[str, np.ndarray],
                                      recent_patterns: List[np.ndarray]) -> Dict[str, Any]:
        """
        Find Nash equilibrium considering temporal dynamics and biological constraints.

        Args:
            time_series: Market time series data
            recent_patterns: Recent market patterns

        Returns:
            Temporal equilibrium analysis
        """
        # Extract relevant time series
        prices = time_series.get('prices', np.array([]))
        volumes = time_series.get('volumes', np.array([]))
        volatility = time_series.get('volatility', np.array([]))

        if len(prices) < 50:
            return {
                'equilibrium': None,
                'convergence_rate': 0.0,
                'error': 'Insufficient data'
            }

        # Initialize game dynamics
        game_state = self._initialize_game_state(prices, volumes, volatility)

        # Temporal simulation loop
        max_steps = 200
        convergence_threshold = 0.01

        for step in range(max_steps):
            self.time_step = step

            # Update each agent's strategy
            new_strategies = []
            for agent in self.agents:
                strategy = await self._update_agent_strategy(agent, game_state, step)
                new_strategies.append(strategy)

            # Calculate payoffs
            payoffs = self._calculate_payoffs(new_strategies, game_state)

            # Update agent states
            for i, agent in enumerate(self.agents):
                self._update_agent_state(agent, new_strategies[i], payoffs[i])

            # Check for equilibrium
            equilibrium_metric = self._calculate_equilibrium_metric(new_strategies, payoffs)
            self.equilibrium_history.append({
                'step': step,
                'strategies': new_strategies.copy(),
                'payoffs': payoffs.copy(),
                'metric': equilibrium_metric
            })

            # Check convergence
            if equilibrium_metric < convergence_threshold:
                logger.info(f"Temporal equilibrium reached at step {step}")
                break

            # Update game state with market feedback
            game_state = self._update_game_state(game_state, new_strategies, prices, step)

        # Analyze temporal dynamics
        analysis = self._analyze_temporal_dynamics()

        return {
            'equilibrium': self.equilibrium_history[-1] if self.equilibrium_history else None,
            'convergence_rate': self._calculate_convergence_rate(),
            'learning_confidence': self._calculate_learning_confidence(),
            'temporal_analysis': analysis,
            'predicted_action': self._predict_next_action(analysis),
            'agent_adaptations': self._summarize_agent_adaptations()
        }

    def _initialize_game_state(self, prices: np.ndarray, volumes: np.ndarray,
                             volatility: np.ndarray) -> Dict[str, Any]:
        """Initialize game state from market data."""
        # Calculate market regime
        recent_returns = np.diff(prices[-50:]) / prices[-51:-1]
        trend = np.mean(recent_returns)
        vol = np.std(recent_returns)

        regime = 'neutral'
        if trend > 0.001:
            regime = 'bullish'
        elif trend < -0.001:
            regime = 'bearish'

        if vol > 0.02:
            regime += '_volatile'

        return {
            'regime': regime,
            'trend': trend,
            'volatility': vol,
            'liquidity': np.mean(volumes[-20:]),
            'price_level': prices[-1],
            'momentum': np.mean(recent_returns[-5:])
        }

    async def _update_agent_strategy(self, agent: Dict[str, Any],
                                   game_state: Dict[str, Any],
                                   time_step: int) -> np.ndarray:
        """Update agent strategy with biological constraints."""
        constraints = agent['constraints']

        # Apply reaction delay
        if time_step < constraints.reaction_delay:
            # Use default strategy during reaction delay
            return np.array([0.25, 0.25, 0.25, 0.25])  # Uniform over 4 actions

        # Get relevant history considering attention span
        relevant_history = list(agent['strategy_history'])[-constraints.attention_span:]

        if len(relevant_history) == 0:
            # No history, use regime-based initialization
            return self._initialize_strategy_by_regime(game_state['regime'])

        # Calculate expected payoffs based on memory
        memory_trace = agent['memory_trace']
        expected_payoffs = self._calculate_expected_payoffs_from_memory(
            memory_trace, relevant_history
        )

        # Apply biological noise
        noise = np.random.normal(0, constraints.noise_level, size=4)
        expected_payoffs += noise

        # Apply fatigue
        if agent['fatigue_level'] > 0.5:
            # Fatigue causes more conservative strategies
            expected_payoffs[2] += 0.2  # Bias toward hold

        # Softmax action selection with temperature based on learning
        temperature = 1.0 / (1.0 + time_step * 0.01)  # Decreasing temperature
        exp_payoffs = np.exp(expected_payoffs / temperature)
        strategy = exp_payoffs / np.sum(exp_payoffs)

        return strategy

    def _initialize_strategy_by_regime(self, regime: str) -> np.ndarray:
        """Initialize strategy based on market regime."""
        # [buy, sell, hold, wait]
        strategies = {
            'bullish': np.array([0.6, 0.1, 0.2, 0.1]),
            'bearish': np.array([0.1, 0.6, 0.2, 0.1]),
            'neutral': np.array([0.2, 0.2, 0.4, 0.2]),
            'bullish_volatile': np.array([0.4, 0.2, 0.2, 0.2]),
            'bearish_volatile': np.array([0.2, 0.4, 0.2, 0.2]),
            'neutral_volatile': np.array([0.15, 0.15, 0.35, 0.35])
        }

        return strategies.get(regime, np.array([0.25, 0.25, 0.25, 0.25]))

    def _calculate_expected_payoffs_from_memory(self, memory_trace: np.ndarray,
                                              history: List[np.ndarray]) -> np.ndarray:
        """Calculate expected payoffs based on memory trace."""
        if len(history) == 0:
            return np.zeros(4)

        # Weighted average of historical payoffs
        weights = np.exp(-np.arange(len(history)) * 0.1)  # Exponential decay
        weights = weights / np.sum(weights)

        expected = np.zeros(4)
        for i, weight in enumerate(weights):
            if i < len(history):
                expected += weight * history[i]

        # Incorporate memory trace
        if len(memory_trace) >= 4:
            expected += memory_trace[:4] * 0.3

        return expected

    def _calculate_payoffs(self, strategies: List[np.ndarray],
                         game_state: Dict[str, Any]) -> List[float]:
        """Calculate payoffs for all agents."""
        payoffs = []

        # Market impact of collective actions
        collective_action = np.mean(strategies, axis=0)
        buy_pressure = collective_action[0]
        sell_pressure = collective_action[1]

        # Price impact
        price_change = (buy_pressure - sell_pressure) * game_state['volatility'] * 0.1

        for i, strategy in enumerate(strategies):
            # Individual payoff calculation
            action_weights = strategy

            # Base payoffs for each action
            base_payoffs = np.array([
                price_change * game_state['trend'],  # Buy payoff
                -price_change * game_state['trend'],  # Sell payoff
                0.0,  # Hold payoff
                game_state['volatility'] * 0.01  # Wait payoff (option value)
            ])

            # Adjust for market regime
            if 'volatile' in game_state['regime']:
                base_payoffs *= 1.5

            # Calculate weighted payoff
            payoff = np.sum(action_weights * base_payoffs)

            # Add competition effects
            for j, other_strategy in enumerate(strategies):
                if i != j:
                    # Negative payoff for similar strategies (competition)
                    similarity = np.dot(strategy, other_strategy)
                    payoff -= similarity * 0.1

            payoffs.append(payoff)

        return payoffs

    def _update_agent_state(self, agent: Dict[str, Any], strategy: np.ndarray, payoff: float):
        """Update agent state with new experience."""
        # Update memory trace
        experience = np.zeros_like(agent['memory_trace'])
        experience[:4] = strategy
        experience[4] = payoff

        agent['memory_trace'] = _update_memory_trace_numba(
            agent['memory_trace'],
            experience,
            agent['constraints'].memory_decay,
            agent['constraints'].learning_rate
        )

        # Update history
        agent['strategy_history'].append(strategy)
        agent['payoff_history'].append(payoff)

        # Update fatigue
        agent['fatigue_level'] += agent['constraints'].fatigue_rate
        agent['fatigue_level'] = min(1.0, agent['fatigue_level'])

        # Recovery from fatigue
        if payoff > 0:
            agent['fatigue_level'] *= 0.95

    def _calculate_equilibrium_metric(self, strategies: List[np.ndarray],
                                    payoffs: List[float]) -> float:
        """Calculate how close the system is to equilibrium."""
        # Check if any agent can improve by deviating
        total_deviation = 0.0

        for i, (strategy, payoff) in enumerate(zip(strategies, payoffs)):
            # Calculate best response
            other_strategies = [s for j, s in enumerate(strategies) if j != i]

            # Try pure strategies
            best_payoff = payoff
            for action in range(4):
                pure_strategy = np.zeros(4)
                pure_strategy[action] = 1.0

                # Calculate payoff for pure strategy
                test_strategies = strategies.copy()
                test_strategies[i] = pure_strategy
                test_payoffs = self._calculate_payoffs(test_strategies,
                                                      self.equilibrium_history[-1]['game_state']
                                                      if self.equilibrium_history else {})

                if test_payoffs[i] > best_payoff:
                    best_payoff = test_payoffs[i]

            # Deviation incentive
            deviation = max(0, best_payoff - payoff)
            total_deviation += deviation

        return total_deviation / len(strategies)

    def _update_game_state(self, current_state: Dict[str, Any],
                         strategies: List[np.ndarray],
                         prices: np.ndarray,
                         step: int) -> Dict[str, Any]:
        """Update game state based on agent actions and market feedback."""
        # Simulate market response to agent actions
        collective_action = np.mean(strategies, axis=0)

        # Update trend based on actions
        action_bias = collective_action[0] - collective_action[1]  # Buy - Sell
        new_trend = current_state['trend'] * 0.9 + action_bias * 0.1

        # Update volatility based on disagreement
        strategy_variance = np.var(strategies, axis=0)
        disagreement = np.mean(strategy_variance)
        new_volatility = current_state['volatility'] * 0.95 + disagreement * 0.05

        # Price evolution
        if step < len(prices) - 1:
            new_price = prices[step + 1]
        else:
            # Extrapolate
            new_price = current_state['price_level'] * (1 + new_trend +
                                                       np.random.normal(0, new_volatility))

        return {
            'regime': self._determine_regime(new_trend, new_volatility),
            'trend': new_trend,
            'volatility': new_volatility,
            'liquidity': current_state['liquidity'] * 0.95 + np.random.random() * 0.05,
            'price_level': new_price,
            'momentum': new_trend * 0.7 + current_state['momentum'] * 0.3
        }

    def _determine_regime(self, trend: float, volatility: float) -> str:
        """Determine market regime from trend and volatility."""
        regime = 'neutral'

        if trend > 0.001:
            regime = 'bullish'
        elif trend < -0.001:
            regime = 'bearish'

        if volatility > 0.02:
            regime += '_volatile'

        return regime

    def _analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """Analyze temporal patterns in equilibrium evolution."""
        if len(self.equilibrium_history) < 10:
            return {'insufficient_data': True}

        # Extract time series of strategies
        strategy_series = []
        for entry in self.equilibrium_history:
            avg_strategy = np.mean(entry['strategies'], axis=0)
            strategy_series.append(avg_strategy)

        strategy_series = np.array(strategy_series)

        # Analyze convergence pattern
        convergence_pattern = 'oscillating'
        if len(strategy_series) > 20:
            # Check for monotonic convergence
            diffs = np.diff(strategy_series, axis=0)
            if np.all(np.abs(diffs) < np.abs(diffs[0])):
                convergence_pattern = 'monotonic'
            # Check for cycles
            elif self._detect_cycles(strategy_series):
                convergence_pattern = 'cyclic'

        # Learning efficiency
        initial_payoffs = [entry['payoffs'] for entry in self.equilibrium_history[:10]]
        final_payoffs = [entry['payoffs'] for entry in self.equilibrium_history[-10:]]

        learning_improvement = (np.mean(final_payoffs) - np.mean(initial_payoffs)) / \
                             (np.mean(np.abs(initial_payoffs)) + 1e-8)

        return {
            'convergence_pattern': convergence_pattern,
            'learning_improvement': float(learning_improvement),
            'strategy_evolution': strategy_series.tolist(),
            'dominant_strategy': self._identify_dominant_strategy(strategy_series),
            'adaptation_speed': self._calculate_adaptation_speed()
        }

    def _detect_cycles(self, series: np.ndarray) -> bool:
        """Detect cyclic patterns in strategy evolution."""
        if len(series) < 20:
            return False

        # Use autocorrelation to detect cycles
        for lag in range(5, len(series) // 2):
            if lag < len(series):
                correlation = np.corrcoef(series[:-lag].flatten(), series[lag:].flatten())[0, 1]
                if correlation > 0.7:  # High correlation at lag indicates cycle
                    return True

        return False

    def _identify_dominant_strategy(self, strategy_series: np.ndarray) -> int:
        """Identify dominant action from strategy evolution."""
        if len(strategy_series) == 0:
            return 2  # Default to hold

        # Average strategy over recent history
        recent_avg = np.mean(strategy_series[-10:], axis=0)

        return int(np.argmax(recent_avg))

    def _calculate_adaptation_speed(self) -> float:
        """Calculate how quickly agents adapt to changes."""
        if len(self.equilibrium_history) < 20:
            return 0.5

        # Measure strategy change rate
        changes = []
        for i in range(1, len(self.equilibrium_history)):
            prev_strategies = self.equilibrium_history[i-1]['strategies']
            curr_strategies = self.equilibrium_history[i]['strategies']

            change = np.mean([np.linalg.norm(curr - prev)
                            for curr, prev in zip(curr_strategies, prev_strategies)])
            changes.append(change)

        # Adaptation speed is inverse of time to stabilize
        stabilization_time = next((i for i, c in enumerate(changes) if c < 0.01), len(changes))

        return 1.0 / (stabilization_time + 1)

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate of the temporal dynamics."""
        if len(self.equilibrium_history) < 2:
            return 0.0

        # Extract equilibrium metrics
        metrics = [entry['metric'] for entry in self.equilibrium_history]

        # Fit exponential decay to metrics
        if len(metrics) > 10:
            # Simple linear regression on log scale
            log_metrics = np.log(np.array(metrics) + 1e-8)
            x = np.arange(len(metrics))

            # Calculate slope
            slope = np.polyfit(x, log_metrics, 1)[0]

            # Convergence rate is negative slope
            return max(0.0, -slope)

        return 0.0

    def _calculate_learning_confidence(self) -> float:
        """Calculate confidence in learned equilibrium."""
        if len(self.equilibrium_history) < 10:
            return 0.0

        # Check stability of recent equilibria
        recent_strategies = [entry['strategies'] for entry in self.equilibrium_history[-10:]]

        # Calculate variance in strategies
        strategy_variance = np.var([np.mean(s, axis=0) for s in recent_strategies], axis=0)
        avg_variance = np.mean(strategy_variance)

        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + avg_variance * 10)

        return float(confidence)

    def _predict_next_action(self, analysis: Dict[str, Any]) -> int:
        """Predict next action based on temporal analysis."""
        if 'insufficient_data' in analysis:
            return 2  # Default to hold

        # Use dominant strategy from analysis
        return analysis.get('dominant_strategy', 2)

    def _summarize_agent_adaptations(self) -> Dict[str, Any]:
        """Summarize how agents have adapted over time."""
        adaptations = {}

        for agent in self.agents:
            agent_id = f"agent_{agent['id']}"

            # Calculate strategy evolution
            if len(agent['strategy_history']) > 10:
                initial_strategy = np.mean(list(agent['strategy_history'])[:10], axis=0)
                current_strategy = np.mean(list(agent['strategy_history'])[-10:], axis=0)

                strategy_shift = current_strategy - initial_strategy

                # Identify adaptation type
                adaptation_type = 'stable'
                if np.max(np.abs(strategy_shift)) > 0.2:
                    if strategy_shift[0] > 0.2:
                        adaptation_type = 'increasingly_bullish'
                    elif strategy_shift[1] > 0.2:
                        adaptation_type = 'increasingly_bearish'
                    elif strategy_shift[2] > 0.2:
                        adaptation_type = 'increasingly_passive'

                adaptations[agent_id] = {
                    'type': adaptation_type,
                    'strategy_shift': strategy_shift.tolist(),
                    'learning_rate': agent['constraints'].learning_rate,
                    'fatigue_level': agent['fatigue_level']
                }
            else:
                adaptations[agent_id] = {'type': 'insufficient_history'}

        return adaptations

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'memory_decay': self.memory_decay,
            'learning_rate_bounds': self.learning_rate_bounds,
            'num_agents': self.num_agents,
            'time_step': self.time_step,
            'equilibrium_history_length': len(self.equilibrium_history),
            'agent_states': [
                {
                    'id': agent['id'],
                    'fatigue_level': agent['fatigue_level'],
                    'memory_trace': agent['memory_trace'].tolist(),
                    'strategy_history_length': len(agent['strategy_history'])
                }
                for agent in self.agents
            ]
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization."""
        self.memory_decay = state.get('memory_decay', self.memory_decay)
        self.learning_rate_bounds = state.get('learning_rate_bounds', self.learning_rate_bounds)
        self.time_step = state.get('time_step', 0)

        # Restore agent states
        if 'agent_states' in state:
            for agent_state in state['agent_states']:
                agent_id = agent_state['id']
                if agent_id < len(self.agents):
                    self.agents[agent_id]['fatigue_level'] = agent_state.get('fatigue_level', 0.0)
                    if 'memory_trace' in agent_state:
                        self.agents[agent_id]['memory_trace'] = np.array(agent_state['memory_trace'])
