"""
Antifragile Quantum Coalition formation for benefiting from market volatility.
"""

import numpy as np
import numba as nb
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
from collections import defaultdict
import asyncio
import networkx as nx
from scipy.special import comb

logger = logging.getLogger(__name__)

@nb.jit(nopython=True, fastmath=True, cache=True)
def _calculate_volatility_benefit_numba(portfolio: np.ndarray, volatility: float,
                                      correlation_matrix: np.ndarray) -> float:
    """
    Calculate benefit from volatility for a portfolio.

    Args:
        portfolio: Portfolio weights
        volatility: Market volatility
        correlation_matrix: Asset correlation matrix

    Returns:
        Volatility benefit score
    """
    n = len(portfolio)

    # Portfolio volatility
    portfolio_var = 0.0
    for i in range(n):
        for j in range(n):
            portfolio_var += portfolio[i] * portfolio[j] * correlation_matrix[i, j]

    portfolio_vol = np.sqrt(portfolio_var) * volatility

    # Convexity benefit (simplified)
    convexity = 0.0
    for i in range(n):
        if portfolio[i] > 0:
            # Options-like payoff increases with volatility
            convexity += portfolio[i] * (volatility ** 2)

    # Antifragile benefit: gains from volatility exceed losses
    linear_loss = portfolio_vol * 0.5  # Simplified linear losses
    nonlinear_gain = convexity * 0.7  # Simplified nonlinear gains

    benefit = nonlinear_gain - linear_loss

    return max(0.0, benefit)

@nb.jit(nopython=True, fastmath=True, cache=True)
def _calculate_coalition_synergy_numba(member_strategies: np.ndarray,
                                     volatility_exposures: np.ndarray) -> float:
    """
    Calculate synergy score for coalition members.

    Args:
        member_strategies: Strategy vectors for coalition members
        volatility_exposures: Volatility exposure for each member

    Returns:
        Coalition synergy score
    """
    n_members = len(member_strategies)
    if n_members < 2:
        return 0.0

    synergy = 0.0

    # Pairwise synergies
    for i in range(n_members):
        for j in range(i + 1, n_members):
            # Strategic complementarity
            strategy_diff = np.sum(np.abs(member_strategies[i] - member_strategies[j]))

            # Volatility exposure complementarity
            vol_product = volatility_exposures[i] * volatility_exposures[j]

            # Synergy from diversity and aligned volatility benefit
            pair_synergy = strategy_diff * vol_product
            synergy += pair_synergy

    # Normalize by number of pairs
    n_pairs = n_members * (n_members - 1) / 2

    return synergy / n_pairs

class AntifragileCoalition:
    """
    Form and manage coalitions that benefit from market volatility and crises.
    """

    def __init__(self, volatility_threshold: float = 0.3,
                 min_coalition_size: int = 2,
                 max_coalition_size: int = 7):
        """
        Initialize Antifragile Coalition system.

        Args:
            volatility_threshold: Minimum volatility to activate antifragile strategies
            min_coalition_size: Minimum coalition size
            max_coalition_size: Maximum coalition size
        """
        self.volatility_threshold = volatility_threshold
        self.min_coalition_size = min_coalition_size
        self.max_coalition_size = max_coalition_size

        # Coalition management
        self.active_coalitions = {}
        self.coalition_performance = defaultdict(list)
        self.member_profiles = {}

        # Coalition formation parameters
        self.formation_criteria = {
            'synergy_threshold': 0.5,
            'volatility_benefit_threshold': 0.3,
            'trust_threshold': 0.6
        }

        # Network graph for coalition relationships
        self.coalition_network = nx.Graph()

        logger.info("Antifragile Coalition system initialized")

    async def form_coalitions(self, market_volatility: Dict[str, float],
                            crisis_indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        Form antifragile coalitions based on market conditions.

        Args:
            market_volatility: Volatility metrics for different assets/markets
            crisis_indicators: Crisis probability indicators

        Returns:
            Formed coalitions and their properties
        """
        # Assess crisis level
        crisis_level = self._assess_crisis_level(crisis_indicators)

        # Identify potential coalition members
        potential_members = await self._identify_potential_members(
            market_volatility, crisis_level
        )

        if len(potential_members) < self.min_coalition_size:
            return {
                'coalitions': [],
                'message': 'Insufficient antifragile members available'
            }

        # Generate coalition candidates
        coalition_candidates = self._generate_coalition_candidates(
            potential_members, market_volatility
        )

        # Evaluate and rank coalitions
        evaluated_coalitions = await self._evaluate_coalitions(
            coalition_candidates, market_volatility, crisis_level
        )

        # Form top coalitions
        formed_coalitions = self._form_top_coalitions(evaluated_coalitions)

        # Update coalition network
        self._update_coalition_network(formed_coalitions)

        return {
            'coalitions': formed_coalitions,
            'expected_benefit': self._calculate_total_expected_benefit(formed_coalitions),
            'crisis_exploitation_ready': crisis_level > 0.5,
            'network_metrics': self._get_network_metrics()
        }

    def _assess_crisis_level(self, crisis_indicators: Dict[str, float]) -> float:
        """Assess overall crisis level from indicators."""
        if not crisis_indicators:
            return 0.0

        # Weighted average of crisis indicators
        weights = {
            'volatility_spike': 0.3,
            'liquidity_drain': 0.25,
            'correlation_breakdown': 0.2,
            'systemic_stress': 0.25
        }

        crisis_score = 0.0
        total_weight = 0.0

        for indicator, value in crisis_indicators.items():
            weight = weights.get(indicator, 0.1)
            crisis_score += value * weight
            total_weight += weight

        return crisis_score / total_weight if total_weight > 0 else 0.0

    async def _identify_potential_members(self, market_volatility: Dict[str, float],
                                        crisis_level: float) -> List[Dict[str, Any]]:
        """Identify potential coalition members with antifragile characteristics."""
        potential_members = []

        # Simulate member identification (in practice, would query actual participants)
        for i in range(20):  # Simulated pool of participants
            member_profile = {
                'id': f'participant_{i}',
                'antifragility_score': np.random.beta(2, 3),  # Skewed toward lower values
                'volatility_exposure': np.random.random(),
                'crisis_preparedness': np.random.beta(3, 2),  # Skewed toward higher values
                'strategy_vector': np.random.dirichlet(np.ones(4)),  # 4 strategies
                'trust_score': np.random.beta(5, 2),  # Most are trustworthy
                'historical_performance': {
                    'normal_market': np.random.normal(0.05, 0.02),
                    'crisis_market': np.random.normal(0.15, 0.05)  # Better in crisis
                }
            }

            # Filter for antifragile characteristics
            if (member_profile['antifragility_score'] > 0.3 and
                member_profile['crisis_preparedness'] > 0.5):
                potential_members.append(member_profile)
                self.member_profiles[member_profile['id']] = member_profile

        return potential_members

    def _generate_coalition_candidates(self, members: List[Dict[str, Any]],
                                     market_volatility: Dict[str, float]) -> List[Set[str]]:
        """Generate potential coalition configurations."""
        member_ids = [m['id'] for m in members]
        candidates = []

        # Generate coalitions of different sizes
        for size in range(self.min_coalition_size,
                         min(len(member_ids) + 1, self.max_coalition_size + 1)):

            # Limit combinations for computational efficiency
            max_combinations = 50

            if comb(len(member_ids), size) <= max_combinations:
                # Generate all combinations
                from itertools import combinations
                for combo in combinations(member_ids, size):
                    candidates.append(set(combo))
            else:
                # Random sampling for large spaces
                for _ in range(max_combinations):
                    sample = np.random.choice(member_ids, size=size, replace=False)
                    candidates.append(set(sample))

        return candidates

    async def _evaluate_coalitions(self, candidates: List[Set[str]],
                                 market_volatility: Dict[str, float],
                                 crisis_level: float) -> List[Dict[str, Any]]:
        """Evaluate coalition candidates for antifragile properties."""
        evaluated = []

        for coalition_members in candidates:
            # Get member profiles
            members = [self.member_profiles[mid] for mid in coalition_members]

            # Calculate coalition metrics
            synergy = self._calculate_coalition_synergy(members)
            volatility_benefit = self._calculate_volatility_benefit(members, market_volatility)
            crisis_readiness = self._calculate_crisis_readiness(members, crisis_level)
            trust_level = self._calculate_trust_level(members)

            # Overall coalition score
            score = (synergy * 0.3 +
                    volatility_benefit * 0.35 +
                    crisis_readiness * 0.25 +
                    trust_level * 0.1)

            # Check formation criteria
            if (synergy >= self.formation_criteria['synergy_threshold'] and
                volatility_benefit >= self.formation_criteria['volatility_benefit_threshold'] and
                trust_level >= self.formation_criteria['trust_threshold']):

                evaluated.append({
                    'members': coalition_members,
                    'score': score,
                    'synergy': synergy,
                    'volatility_benefit': volatility_benefit,
                    'crisis_readiness': crisis_readiness,
                    'trust_level': trust_level,
                    'expected_performance': self._estimate_coalition_performance(
                        members, market_volatility, crisis_level
                    )
                })

        # Sort by score
        evaluated.sort(key=lambda x: x['score'], reverse=True)

        return evaluated

    def _calculate_coalition_synergy(self, members: List[Dict[str, Any]]) -> float:
        """Calculate synergy between coalition members."""
        if len(members) < 2:
            return 0.0

        # Extract data for Numba function
        strategies = np.array([m['strategy_vector'] for m in members])
        vol_exposures = np.array([m['volatility_exposure'] for m in members])

        base_synergy = _calculate_coalition_synergy_numba(strategies, vol_exposures)

        # Additional synergy factors
        antifragility_diversity = np.std([m['antifragility_score'] for m in members])

        # Diversity bonus
        diversity_bonus = min(0.3, antifragility_diversity * 2)

        return float(base_synergy + diversity_bonus)

    def _calculate_volatility_benefit(self, members: List[Dict[str, Any]],
                                    market_volatility: Dict[str, float]) -> float:
        """Calculate expected benefit from volatility."""
        # Average market volatility
        avg_volatility = np.mean(list(market_volatility.values()))

        if avg_volatility < self.volatility_threshold:
            return 0.0

        # Coalition portfolio (equal weight for simplicity)
        n_members = len(members)
        portfolio = np.ones(n_members) / n_members

        # Correlation matrix (simplified - would use actual correlations)
        correlations = np.eye(n_members) * 0.5 + 0.5

        benefit = _calculate_volatility_benefit_numba(portfolio, avg_volatility, correlations)

        # Scale by member antifragility
        avg_antifragility = np.mean([m['antifragility_score'] for m in members])

        return float(benefit * avg_antifragility)

    def _calculate_crisis_readiness(self, members: List[Dict[str, Any]],
                                  crisis_level: float) -> float:
        """Calculate coalition's crisis readiness."""
        # Average crisis preparedness
        avg_preparedness = np.mean([m['crisis_preparedness'] for m in members])

        # Historical crisis performance
        crisis_performances = [m['historical_performance']['crisis_market'] for m in members]
        avg_crisis_performance = np.mean(crisis_performances)

        # Readiness score
        readiness = avg_preparedness * 0.6 + (avg_crisis_performance + 1) / 2 * 0.4

        # Scale by current crisis level
        return float(readiness * (0.5 + crisis_level * 0.5))

    def _calculate_trust_level(self, members: List[Dict[str, Any]]) -> float:
        """Calculate trust level within coalition."""
        trust_scores = [m['trust_score'] for m in members]

        # Minimum trust (weakest link)
        min_trust = np.min(trust_scores)

        # Average trust
        avg_trust = np.mean(trust_scores)

        # Combined trust metric
        return float(min_trust * 0.7 + avg_trust * 0.3)

    def _estimate_coalition_performance(self, members: List[Dict[str, Any]],
                                      market_volatility: Dict[str, float],
                                      crisis_level: float) -> Dict[str, float]:
        """Estimate coalition performance under different scenarios."""
        # Normal market performance
        normal_perfs = [m['historical_performance']['normal_market'] for m in members]
        coalition_normal = np.mean(normal_perfs) * 1.1  # 10% synergy bonus

        # Crisis market performance
        crisis_perfs = [m['historical_performance']['crisis_market'] for m in members]
        coalition_crisis = np.mean(crisis_perfs) * 1.3  # 30% synergy bonus in crisis

        # Volatility-weighted expected performance
        avg_volatility = np.mean(list(market_volatility.values()))

        if avg_volatility > self.volatility_threshold:
            # High volatility scenario
            expected = coalition_normal * 0.3 + coalition_crisis * 0.7
        else:
            # Normal volatility
            expected = coalition_normal * 0.7 + coalition_crisis * 0.3

        return {
            'normal_market': float(coalition_normal),
            'crisis_market': float(coalition_crisis),
            'expected': float(expected)
        }

    def _form_top_coalitions(self, evaluated_coalitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Form the top-scoring non-overlapping coalitions."""
        formed = []
        used_members = set()

        for coalition in evaluated_coalitions:
            # Check if any members are already in a coalition
            if not any(member in used_members for member in coalition['members']):
                # Form coalition
                coalition_id = f"coalition_{len(self.active_coalitions)}"

                formed_coalition = {
                    'id': coalition_id,
                    'members': list(coalition['members']),
                    'formation_time': asyncio.get_event_loop().time(),
                    'properties': {
                        'synergy': coalition['synergy'],
                        'volatility_benefit': coalition['volatility_benefit'],
                        'crisis_readiness': coalition['crisis_readiness'],
                        'expected_performance': coalition['expected_performance']
                    }
                }

                formed.append(formed_coalition)
                self.active_coalitions[coalition_id] = formed_coalition

                # Mark members as used
                used_members.update(coalition['members'])

                # Limit number of coalitions
                if len(formed) >= 5:
                    break

        return formed

    def _update_coalition_network(self, formed_coalitions: List[Dict[str, Any]]):
        """Update the coalition relationship network."""
        for coalition in formed_coalitions:
            coalition_id = coalition['id']

            # Add coalition node
            self.coalition_network.add_node(
                coalition_id,
                type='coalition',
                formation_time=coalition['formation_time'],
                properties=coalition['properties']
            )

            # Add member nodes and edges
            for member_id in coalition['members']:
                if member_id not in self.coalition_network:
                    self.coalition_network.add_node(
                        member_id,
                        type='member',
                        profile=self.member_profiles.get(member_id, {})
                    )

                # Add edge between member and coalition
                self.coalition_network.add_edge(
                    member_id,
                    coalition_id,
                    relationship='member_of'
                )

    def _calculate_total_expected_benefit(self, coalitions: List[Dict[str, Any]]) -> float:
        """Calculate total expected benefit from all coalitions."""
        if not coalitions:
            return 0.0

        total_benefit = 0.0

        for coalition in coalitions:
            volatility_benefit = coalition['properties']['volatility_benefit']
            expected_performance = coalition['properties']['expected_performance']['expected']

            # Coalition benefit
            coalition_benefit = volatility_benefit + max(0, expected_performance)
            total_benefit += coalition_benefit

        return float(total_benefit)

    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get metrics about the coalition network."""
        if len(self.coalition_network) == 0:
            return {'nodes': 0, 'edges': 0}

        return {
            'total_nodes': self.coalition_network.number_of_nodes(),
            'total_edges': self.coalition_network.number_of_edges(),
            'num_coalitions': len([n for n, d in self.coalition_network.nodes(data=True)
                                 if d.get('type') == 'coalition']),
            'num_members': len([n for n, d in self.coalition_network.nodes(data=True)
                              if d.get('type') == 'member']),
            'avg_coalition_size': np.mean([len(c['members']) for c in self.active_coalitions.values()])
                                 if self.active_coalitions else 0,
            'network_density': nx.density(self.coalition_network)
        }

    def execute_antifragile_strategy(self, coalition_id: str,
                                   market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute antifragile strategy for a coalition.

        Args:
            coalition_id: Coalition identifier
            market_conditions: Current market conditions

        Returns:
            Execution results
        """
        if coalition_id not in self.active_coalitions:
            return {'error': 'Coalition not found'}

        coalition = self.active_coalitions[coalition_id]

        # Determine strategy based on market conditions
        volatility = market_conditions.get('volatility', 0.01)
        trend = market_conditions.get('trend', 0.0)

        strategy = {
            'type': 'antifragile',
            'actions': []
        }

        if volatility > self.volatility_threshold:
            # High volatility - activate antifragile positions
            strategy['actions'].extend([
                'increase_convex_positions',
                'add_volatility_hedges',
                'prepare_crisis_liquidity'
            ])

            if trend < -0.01:  # Downtrend
                strategy['actions'].append('activate_tail_hedges')

        else:
            # Low volatility - prepare for future volatility
            strategy['actions'].extend([
                'accumulate_options',
                'reduce_linear_exposure',
                'build_crisis_reserves'
            ])

        # Calculate expected benefit
        expected_benefit = coalition['properties']['volatility_benefit'] * volatility / 0.02

        # Track performance
        performance = {
            'coalition_id': coalition_id,
            'timestamp': asyncio.get_event_loop().time(),
            'volatility': volatility,
            'expected_benefit': expected_benefit,
            'strategy': strategy
        }

        self.coalition_performance[coalition_id].append(performance)

        return {
            'strategy': strategy,
            'expected_benefit': float(expected_benefit),
            'coalition_status': 'active',
            'members': coalition['members']
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'volatility_threshold': self.volatility_threshold,
            'formation_criteria': self.formation_criteria.copy(),
            'active_coalitions': list(self.active_coalitions.keys()),
            'num_coalitions': len(self.active_coalitions),
            'network_metrics': self._get_network_metrics()
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization."""
        self.volatility_threshold = state.get('volatility_threshold', self.volatility_threshold)
        self.formation_criteria = state.get('formation_criteria', self.formation_criteria).copy()
