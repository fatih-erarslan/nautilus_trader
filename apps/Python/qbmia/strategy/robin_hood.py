"""
Robin Hood Protocol for ethical market intervention and wealth redistribution.
"""

import numpy as np
import numba as nb
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import asyncio

logger = logging.getLogger(__name__)

@nb.jit(nopython=True, fastmath=True, cache=True)
def _calculate_gini_coefficient_numba(wealth_values: np.ndarray) -> float:
    """
    Numba-accelerated Gini coefficient calculation.

    Args:
        wealth_values: Array of wealth values

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    n = len(wealth_values)
    if n == 0:
        return 0.0

    # Sort wealth values
    sorted_wealth = np.sort(wealth_values)

    # Calculate Gini coefficient
    cumsum = 0.0
    for i in range(n):
        cumsum += (2 * (i + 1) - n - 1) * sorted_wealth[i]

    gini = cumsum / (n * np.sum(sorted_wealth))

    return max(0.0, min(1.0, gini))

@nb.jit(nopython=True, fastmath=True, cache=True)
def _identify_wealth_concentration_numba(wealth_values: np.ndarray,
                                       threshold_percentile: float = 90.0) -> np.ndarray:
    """
    Identify concentrated wealth holders.

    Args:
        wealth_values: Array of wealth values
        threshold_percentile: Percentile threshold for concentration

    Returns:
        Boolean mask of concentrated wealth holders
    """
    if len(wealth_values) == 0:
        return np.array([], dtype=np.bool_)

    threshold = np.percentile(wealth_values, threshold_percentile)
    return wealth_values > threshold

class RobinHoodProtocol:
    """
    Protocol for identifying and executing ethical market interventions.
    """

    def __init__(self, wealth_threshold: float = 0.8, intervention_threshold: float = 0.7):
        """
        Initialize Robin Hood Protocol.

        Args:
            wealth_threshold: Gini coefficient threshold for intervention
            intervention_threshold: Confidence threshold for executing interventions
        """
        self.wealth_threshold = wealth_threshold
        self.intervention_threshold = intervention_threshold

        # Intervention strategies
        self.intervention_strategies = {
            'liquidity_provision': {
                'enabled': True,
                'intensity': 0.3,
                'target_spread': 0.001
            },
            'market_making': {
                'enabled': True,
                'depth': 0.2,
                'symmetric': True
            },
            'arbitrage_disruption': {
                'enabled': False,
                'delay': 0.1
            }
        }

        # Tracking
        self.intervention_history = []
        self.wealth_redistribution_total = 0.0
        self.market_health_improvements = []

        logger.info("Robin Hood Protocol initialized")

    async def analyze_wealth_distribution(self, participant_wealth: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze wealth distribution among market participants.

        Args:
            participant_wealth: Dictionary of participant IDs to wealth values

        Returns:
            Wealth distribution analysis
        """
        if not participant_wealth:
            return {
                'gini_coefficient': 0.0,
                'wealth_concentration': {},
                'intervention_needed': False
            }

        # Convert to numpy array
        participants = list(participant_wealth.keys())
        wealth_values = np.array(list(participant_wealth.values()), dtype=np.float64)

        # Calculate Gini coefficient
        gini = _calculate_gini_coefficient_numba(wealth_values)

        # Identify wealth concentration
        concentrated_mask = _identify_wealth_concentration_numba(wealth_values)
        concentrated_participants = [
            participants[i] for i in range(len(participants)) if concentrated_mask[i]
        ]

        # Calculate additional metrics
        top_10_percent = int(len(wealth_values) * 0.1)
        if top_10_percent > 0:
            top_10_wealth_share = np.sum(np.sort(wealth_values)[-top_10_percent:]) / np.sum(wealth_values)
        else:
            top_10_wealth_share = 0.0

        # Determine if intervention is needed
        intervention_needed = gini > self.wealth_threshold

        return {
            'gini_coefficient': float(gini),
            'wealth_concentration': {
                'concentrated_participants': concentrated_participants,
                'top_10_percent_share': float(top_10_wealth_share),
                'concentration_ratio': float(len(concentrated_participants) / len(participants))
            },
            'intervention_needed': intervention_needed,
            'distribution_metrics': {
                'mean_wealth': float(np.mean(wealth_values)),
                'median_wealth': float(np.median(wealth_values)),
                'std_wealth': float(np.std(wealth_values)),
                'wealth_skewness': float(stats.skew(wealth_values))
            }
        }

    async def identify_interventions(self, wealth_analysis: Dict[str, Any],
                                   market_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential market interventions to improve wealth distribution.

        Args:
            wealth_analysis: Results from wealth distribution analysis
            market_structure: Current market structure information

        Returns:
            List of potential interventions
        """
        interventions = []

        if not wealth_analysis.get('intervention_needed', False):
            return interventions

        # Analyze market structure for intervention opportunities
        liquidity_gaps = self._identify_liquidity_gaps(market_structure)
        arbitrage_opportunities = self._identify_arbitrage_opportunities(market_structure)
        market_inefficiencies = self._identify_inefficiencies(market_structure)

        # Generate intervention proposals
        if liquidity_gaps and self.intervention_strategies['liquidity_provision']['enabled']:
            interventions.append({
                'type': 'liquidity_provision',
                'target_markets': liquidity_gaps,
                'expected_impact': self._estimate_liquidity_impact(liquidity_gaps),
                'resource_requirement': 'medium',
                'risk_level': 'low'
            })

        if market_inefficiencies and self.intervention_strategies['market_making']['enabled']:
            interventions.append({
                'type': 'market_making',
                'inefficiencies': market_inefficiencies,
                'expected_spread_reduction': 0.002,
                'resource_requirement': 'high',
                'risk_level': 'medium'
            })

        if arbitrage_opportunities and self.intervention_strategies['arbitrage_disruption']['enabled']:
            interventions.append({
                'type': 'arbitrage_disruption',
                'opportunities': arbitrage_opportunities[:3],  # Top 3
                'expected_redistribution': self._estimate_arbitrage_impact(arbitrage_opportunities),
                'resource_requirement': 'low',
                'risk_level': 'medium'
            })

        # Score and rank interventions
        scored_interventions = self._score_interventions(interventions, wealth_analysis)

        return scored_interventions

    def _identify_liquidity_gaps(self, market_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify markets with poor liquidity."""
        gaps = []

        markets = market_structure.get('markets', {})
        for market_id, market_data in markets.items():
            spread = market_data.get('bid_ask_spread', 0)
            volume = market_data.get('volume', 0)
            depth = market_data.get('order_book_depth', 0)

            # Poor liquidity indicators
            if spread > 0.005 or volume < 1000 or depth < 10:
                gaps.append({
                    'market_id': market_id,
                    'spread': spread,
                    'volume': volume,
                    'depth': depth,
                    'severity': self._calculate_liquidity_severity(spread, volume, depth)
                })

        # Sort by severity
        gaps.sort(key=lambda x: x['severity'], reverse=True)

        return gaps

    def _identify_arbitrage_opportunities(self, market_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify arbitrage opportunities that extract value."""
        opportunities = []

        # Simplified arbitrage detection
        markets = market_structure.get('markets', {})
        market_pairs = []

        # Find correlated markets
        for market1_id, market1_data in markets.items():
            for market2_id, market2_data in markets.items():
                if market1_id < market2_id:  # Avoid duplicates
                    price1 = market1_data.get('price', 0)
                    price2 = market2_data.get('price', 0)

                    if price1 > 0 and price2 > 0:
                        # Check for price divergence
                        expected_ratio = market1_data.get('expected_ratio', 1.0)
                        actual_ratio = price1 / price2
                        divergence = abs(actual_ratio - expected_ratio) / expected_ratio

                        if divergence > 0.01:  # 1% divergence
                            opportunities.append({
                                'market_pair': (market1_id, market2_id),
                                'divergence': divergence,
                                'profit_potential': divergence * min(
                                    market1_data.get('volume', 0),
                                    market2_data.get('volume', 0)
                                )
                            })

        # Sort by profit potential
        opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)

        return opportunities

    def _identify_inefficiencies(self, market_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify market inefficiencies."""
        inefficiencies = []

        markets = market_structure.get('markets', {})
        for market_id, market_data in markets.items():
            # Price volatility vs volume (inefficiency indicator)
            volatility = market_data.get('volatility', 0)
            volume = market_data.get('volume', 1)

            inefficiency_score = volatility / np.log(volume + 1)

            if inefficiency_score > 0.1:
                inefficiencies.append({
                    'market_id': market_id,
                    'inefficiency_score': inefficiency_score,
                    'volatility': volatility,
                    'volume': volume
                })

        return inefficiencies

    def _calculate_liquidity_severity(self, spread: float, volume: float, depth: int) -> float:
        """Calculate severity score for liquidity gap."""
        spread_score = min(1.0, spread / 0.01)  # Normalize by 1% spread
        volume_score = max(0.0, 1.0 - volume / 10000)  # Normalize by 10k volume
        depth_score = max(0.0, 1.0 - depth / 100)  # Normalize by 100 order depth

        # Weighted average
        severity = 0.5 * spread_score + 0.3 * volume_score + 0.2 * depth_score

        return severity

    def _estimate_liquidity_impact(self, liquidity_gaps: List[Dict[str, Any]]) -> float:
        """Estimate impact of liquidity provision."""
        if not liquidity_gaps:
            return 0.0

        # Estimate spread reduction impact
        total_impact = 0.0
        for gap in liquidity_gaps[:5]:  # Top 5 gaps
            current_spread = gap['spread']
            target_spread = self.intervention_strategies['liquidity_provision']['target_spread']
            spread_reduction = max(0, current_spread - target_spread)

            # Impact proportional to volume and spread reduction
            impact = spread_reduction * gap.get('volume', 0) * 0.001
            total_impact += impact

        return total_impact

    def _estimate_arbitrage_impact(self, opportunities: List[Dict[str, Any]]) -> float:
        """Estimate redistribution impact of arbitrage disruption."""
        if not opportunities:
            return 0.0

        # Sum profit potential that would be redistributed
        total_redistribution = sum(
            opp['profit_potential'] * 0.5  # Assume 50% capture
            for opp in opportunities[:5]  # Top 5 opportunities
        )

        return total_redistribution

    def _score_interventions(self, interventions: List[Dict[str, Any]],
                           wealth_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score and rank interventions by expected impact."""
        gini = wealth_analysis['gini_coefficient']

        for intervention in interventions:
            # Score based on multiple factors
            impact_score = 0.0

            if intervention['type'] == 'liquidity_provision':
                impact_score = intervention['expected_impact'] * 0.5
                impact_score *= (1.0 if intervention['risk_level'] == 'low' else 0.7)

            elif intervention['type'] == 'market_making':
                spread_impact = intervention['expected_spread_reduction'] * 1000
                impact_score = spread_impact * 0.4
                impact_score *= (0.8 if intervention['risk_level'] == 'medium' else 0.6)

            elif intervention['type'] == 'arbitrage_disruption':
                redistribution = intervention['expected_redistribution']
                impact_score = redistribution * 0.6
                impact_score *= (0.8 if intervention['risk_level'] == 'medium' else 0.6)

            # Adjust by current wealth inequality
            impact_score *= gini

            intervention['impact_score'] = impact_score
            intervention['priority'] = self._calculate_priority(intervention)

        # Sort by priority
        interventions.sort(key=lambda x: x['priority'], reverse=True)

        return interventions

    def _calculate_priority(self, intervention: Dict[str, Any]) -> float:
        """Calculate intervention priority."""
        impact = intervention.get('impact_score', 0)

        # Adjust by resource requirement
        resource_multiplier = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.8
        }.get(intervention.get('resource_requirement', 'medium'), 1.0)

        # Adjust by risk
        risk_multiplier = {
            'low': 1.1,
            'medium': 1.0,
            'high': 0.9
        }.get(intervention.get('risk_level', 'medium'), 1.0)

        return impact * resource_multiplier * risk_multiplier

    def execute_intervention(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a market intervention.

        Args:
            intervention: Intervention to execute

        Returns:
            Execution results
        """
        intervention_type = intervention['type']

        if intervention_type == 'liquidity_provision':
            result = self._execute_liquidity_provision(intervention)
        elif intervention_type == 'market_making':
            result = self._execute_market_making(intervention)
        elif intervention_type == 'arbitrage_disruption':
            result = self._execute_arbitrage_disruption(intervention)
        else:
            result = {'success': False, 'error': 'Unknown intervention type'}

        # Track intervention
        self.intervention_history.append({
            'timestamp': asyncio.get_event_loop().time(),
            'intervention': intervention,
            'result': result
        })

        # Update metrics
        if result.get('success', False):
            self.wealth_redistribution_total += result.get('value_redistributed', 0)

            if 'market_health_improvement' in result:
                self.market_health_improvements.append(result['market_health_improvement'])

        return result

    def _execute_liquidity_provision(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Execute liquidity provision intervention."""
        # Simulated execution
        markets_improved = len(intervention.get('target_markets', []))
        average_spread_reduction = 0.002

        return {
            'success': True,
            'markets_improved': markets_improved,
            'spread_reduction': average_spread_reduction,
            'value_redistributed': markets_improved * average_spread_reduction * 1000,
            'market_health_improvement': 0.1
        }

    def _execute_market_making(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market making intervention."""
        # Simulated execution
        inefficiencies_addressed = len(intervention.get('inefficiencies', []))

        return {
            'success': True,
            'inefficiencies_addressed': inefficiencies_addressed,
            'depth_added': 0.2,
            'value_redistributed': inefficiencies_addressed * 0.5,
            'market_health_improvement': 0.15
        }

    def _execute_arbitrage_disruption(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Execute arbitrage disruption intervention."""
        # Simulated execution
        opportunities_disrupted = len(intervention.get('opportunities', []))

        return {
            'success': True,
            'opportunities_disrupted': opportunities_disrupted,
            'arbitrage_prevented': intervention.get('expected_redistribution', 0) * 0.7,
            'value_redistributed': intervention.get('expected_redistribution', 0) * 0.3,
            'market_health_improvement': 0.05
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'wealth_threshold': self.wealth_threshold,
            'intervention_threshold': self.intervention_threshold,
            'intervention_strategies': self.intervention_strategies.copy(),
            'wealth_redistribution_total': self.wealth_redistribution_total,
            'intervention_count': len(self.intervention_history)
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization."""
        self.wealth_threshold = state.get('wealth_threshold', self.wealth_threshold)
        self.intervention_threshold = state.get('intervention_threshold', self.intervention_threshold)
        self.intervention_strategies = state.get('intervention_strategies', self.intervention_strategies).copy()
        self.wealth_redistribution_total = state.get('wealth_redistribution_total', 0.0)
