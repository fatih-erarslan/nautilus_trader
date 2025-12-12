"""
Interface for QBMIA integration with Decision App boardroom.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class BoardroomInterface:
    """
    Interface for QBMIA participation in the Panarchy Adaptive Decision System boardroom.
    """

    def __init__(self, qbmia_agent: Any, boardroom_config: Dict[str, Any]):
        """
        Initialize boardroom interface.

        Args:
            qbmia_agent: QBMIA agent instance
            boardroom_config: Boardroom configuration
        """
        self.agent = qbmia_agent
        self.config = boardroom_config

        # Boardroom communication
        self.agent_id = self.config.get('agent_id', 'QBMIA')
        self.other_agents = ['QAR', 'Quasar', 'Quantum_Amos', 'CDFA']

        # Message handlers
        self.message_handlers = {
            'market_update': self._handle_market_update,
            'decision_request': self._handle_decision_request,
            'agent_communication': self._handle_agent_communication,
            'consensus_request': self._handle_consensus_request,
            'performance_review': self._handle_performance_review
        }

        # Boardroom state
        self.current_decision_context = {}
        self.agent_opinions = {}
        self.consensus_history = []

        # Performance tracking
        self.boardroom_metrics = {
            'decisions_participated': 0,
            'consensus_achieved': 0,
            'influence_scores': [],
            'collaboration_scores': {}
        }

    async def join_boardroom(self):
        """Join the boardroom and announce presence."""
        announcement = {
            'agent_id': self.agent_id,
            'agent_type': 'Quantum-Biological Market Intuition Agent',
            'capabilities': [
                'quantum_nash_equilibrium',
                'market_manipulation_detection',
                'ethical_market_intervention',
                'temporal_learning',
                'antifragile_strategies'
            ],
            'status': 'active',
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"{self.agent_id} joining boardroom")

        # Initialize collaboration scores
        for agent in self.other_agents:
            self.boardroom_metrics['collaboration_scores'][agent] = 0.5

        return announcement

    async def process_boardroom_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming boardroom message.

        Args:
            message: Boardroom message

        Returns:
            Response message if needed
        """
        message_type = message.get('type', 'unknown')

        if message_type in self.message_handlers:
            return await self.message_handlers[message_type](message)
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return None

    async def _handle_market_update(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle market update message."""
        market_data = message.get('market_data', {})

        # Perform QBMIA analysis
        analysis = await self.agent.analyze_market(market_data, self.current_decision_context)

        # Share key insights with boardroom
        insights = {
            'agent_id': self.agent_id,
            'type': 'market_analysis',
            'timestamp': datetime.utcnow().isoformat(),
            'insights': {
                'quantum_equilibrium': analysis['market_intelligence'].get('quantum_nash', {}),
                'manipulation_detected': analysis['market_intelligence'].get('machiavellian', {}).get('manipulation_detected', False),
                'ecosystem_health': analysis['market_intelligence'].get('robin_hood', {}).get('ecosystem_health', {}),
                'volatility_opportunity': analysis['market_intelligence'].get('antifragile', {}).get('volatility_benefit', 0.0)
            },
            'confidence': analysis.get('agent_confidence', 0.5)
        }

        return insights

    async def _handle_decision_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle decision request from boardroom."""
        decision_context = message.get('context', {})
        self.current_decision_context = decision_context

        # Gather other agents' preliminary views
        other_agent_views = message.get('agent_views', {})

        # Generate QBMIA recommendation
        recommendation = self.agent.generate_boardroom_recommendation(
            {'market_intelligence': self.agent.last_decision},
            other_agent_views
        )

        # Format for boardroom
        boardroom_response = {
            'agent_id': self.agent_id,
            'type': 'decision_recommendation',
            'timestamp': datetime.utcnow().isoformat(),
            'recommendation': {
                'action': recommendation['strategic_perspective'].get('action', 'hold'),
                'confidence': recommendation['confidence_level'],
                'reasoning': recommendation['unique_insights'],
                'risk_assessment': recommendation['risk_assessment'],
                'timing': recommendation['timing_recommendations']
            },
            'collaborative_suggestions': self._generate_collaborative_suggestions(other_agent_views),
            'influence_weight': recommendation.get('boardroom_influence_weight', 0.2)
        }

        self.boardroom_metrics['decisions_participated'] += 1

        return boardroom_response

    async def _handle_agent_communication(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle direct communication from another agent."""
        sender = message.get('sender_id')
        content = message.get('content', {})

        if sender not in self.other_agents:
            return None

        # Process based on communication type
        comm_type = content.get('type')

        if comm_type == 'information_request':
            # Share specific information
            requested_info = content.get('requested_info')
            response_content = self._prepare_information_response(requested_info)

        elif comm_type == 'collaboration_proposal':
            # Evaluate collaboration opportunity
            proposal = content.get('proposal')
            response_content = self._evaluate_collaboration(sender, proposal)

        elif comm_type == 'strategy_coordination':
            # Coordinate strategies
            strategy = content.get('strategy')
            response_content = self._coordinate_strategy(sender, strategy)

        else:
            response_content = {'acknowledged': True}

        # Update collaboration score
        self._update_collaboration_score(sender, 0.1)

        return {
            'agent_id': self.agent_id,
            'type': 'agent_response',
            'recipient': sender,
            'content': response_content,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def _handle_consensus_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle consensus building request."""
        proposals = message.get('proposals', [])
        other_positions = message.get('agent_positions', {})

        # Evaluate each proposal
        evaluations = []
        for proposal in proposals:
            score = self._evaluate_proposal(proposal, other_positions)
            evaluations.append({
                'proposal_id': proposal['id'],
                'score': score,
                'concerns': self._identify_concerns(proposal),
                'improvements': self._suggest_improvements(proposal)
            })

        # Select preferred proposal
        best_proposal_idx = np.argmax([e['score'] for e in evaluations])

        # Check for consensus opportunity
        consensus_possible = self._check_consensus_possibility(evaluations, other_positions)

        response = {
            'agent_id': self.agent_id,
            'type': 'consensus_response',
            'evaluations': evaluations,
            'preferred_proposal': proposals[best_proposal_idx]['id'],
            'consensus_possible': consensus_possible,
            'compromise_suggestions': self._generate_compromise_suggestions(proposals, other_positions),
            'timestamp': datetime.utcnow().isoformat()
        }

        if consensus_possible:
            self.boardroom_metrics['consensus_achieved'] += 1

        return response

    async def _handle_performance_review(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance review request."""
        review_period = message.get('period', 'session')

        # Get QBMIA performance metrics
        agent_metrics = self.agent.get_status()

        # Prepare performance report
        performance_report = {
            'agent_id': self.agent_id,
            'type': 'performance_report',
            'period': review_period,
            'metrics': {
                'decision_accuracy': agent_metrics['performance'].get('prediction_accuracy', 0.0),
                'strategic_advantage': agent_metrics['performance'].get('strategic_advantage', 0.0),
                'unique_contributions': self._calculate_unique_contributions(),
                'collaboration_effectiveness': np.mean(list(self.boardroom_metrics['collaboration_scores'].values())),
                'boardroom_influence': np.mean(self.boardroom_metrics['influence_scores']) if self.boardroom_metrics['influence_scores'] else 0.2
            },
            'key_achievements': self._summarize_achievements(),
            'improvement_areas': self._identify_improvement_areas(),
            'timestamp': datetime.utcnow().isoformat()
        }

        return performance_report

    def _generate_collaborative_suggestions(self, other_views: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions for collaboration with other agents."""
        suggestions = []

        # Analyze alignment with each agent
        for agent, view in other_views.items():
            if agent in self.other_agents:
                alignment = self._calculate_alignment(view)

                if alignment > 0.7:
                    # High alignment - suggest coordination
                    suggestions.append({
                        'type': 'coordination',
                        'agent': agent,
                        'reason': 'strategic_alignment',
                        'proposed_action': 'joint_execution'
                    })
                elif alignment < 0.3:
                    # Low alignment - suggest information sharing
                    suggestions.append({
                        'type': 'information_sharing',
                        'agent': agent,
                        'reason': 'perspective_diversity',
                        'proposed_action': 'exchange_analysis'
                    })

        # Suggest multi-agent strategies
        if len(suggestions) > 1:
            suggestions.append({
                'type': 'coalition',
                'agents': [s['agent'] for s in suggestions[:3]],
                'reason': 'synergistic_opportunity',
                'proposed_action': 'form_strategic_coalition'
            })

        return suggestions

    def _calculate_alignment(self, other_view: Dict[str, Any]) -> float:
        """Calculate strategic alignment with another agent's view."""
        if not self.agent.last_decision:
            return 0.5

        # Compare key decision elements
        alignment_factors = []

        # Action alignment
        our_action = self.agent.last_decision.get('integrated_decision', {}).get('action', 'hold')
        their_action = other_view.get('recommendation', {}).get('action', 'hold')
        action_alignment = 1.0 if our_action == their_action else 0.0
        alignment_factors.append(action_alignment)

        # Confidence alignment
        our_confidence = self.agent.last_decision.get('integrated_decision', {}).get('confidence', 0.5)
        their_confidence = other_view.get('confidence', 0.5)
        confidence_alignment = 1.0 - abs(our_confidence - their_confidence)
        alignment_factors.append(confidence_alignment)

        # Risk assessment alignment
        our_risk = self._extract_risk_level(self.agent.last_decision)
        their_risk = other_view.get('risk_level', 0.5)
        risk_alignment = 1.0 - abs(our_risk - their_risk)
        alignment_factors.append(risk_alignment)

        return np.mean(alignment_factors)

    def _extract_risk_level(self, analysis: Dict[str, Any]) -> float:
        """Extract risk level from analysis."""
        # Simplified risk extraction
        if 'integrated_decision' in analysis:
            decision = analysis['integrated_decision']
            if decision.get('action') in ['buy', 'sell']:
                return 0.7  # Higher risk for active positions
            else:
                return 0.3  # Lower risk for hold/wait
        return 0.5

    def _prepare_information_response(self, requested_info: str) -> Dict[str, Any]:
        """Prepare response to information request."""
        if not self.agent.last_decision:
            return {'status': 'no_data_available'}

        response_map = {
            'quantum_analysis': self.agent.last_decision.get('market_intelligence', {}).get('quantum_nash', {}),
            'manipulation_status': self.agent.last_decision.get('market_intelligence', {}).get('machiavellian', {}),
            'market_health': self.agent.last_decision.get('market_intelligence', {}).get('robin_hood', {}),
            'volatility_assessment': self.agent.last_decision.get('market_intelligence', {}).get('antifragile', {})
        }

        return response_map.get(requested_info, {'status': 'unknown_request'})

    def _evaluate_collaboration(self, agent: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate collaboration proposal from another agent."""
        collaboration_score = 0.0

        # Evaluate based on historical collaboration
        historical_score = self.boardroom_metrics['collaboration_scores'].get(agent, 0.5)
        collaboration_score += historical_score * 0.3

        # Evaluate based on proposal quality
        if 'expected_benefit' in proposal:
            benefit_score = min(1.0, proposal['expected_benefit'])
            collaboration_score += benefit_score * 0.4

        # Evaluate based on strategic fit
        strategic_fit = self._assess_strategic_fit(proposal)
        collaboration_score += strategic_fit * 0.3

        accept = collaboration_score > 0.6

        return {
            'decision': 'accept' if accept else 'decline',
            'score': collaboration_score,
            'conditions': self._generate_collaboration_conditions(proposal) if accept else None
        }

    def _assess_strategic_fit(self, proposal: Dict[str, Any]) -> float:
        """Assess strategic fit of a proposal."""
        fit_score = 0.5  # Neutral default

        proposal_type = proposal.get('type', '')

        # QBMIA excels at certain strategies
        if 'antifragile' in proposal_type or 'volatility' in proposal_type:
            fit_score += 0.3

        if 'quantum' in proposal_type or 'nash' in proposal_type:
            fit_score += 0.2

        if 'ethical' in proposal_type or 'ecosystem' in proposal_type:
            fit_score += 0.2

        return min(1.0, fit_score)

    def _generate_collaboration_conditions(self, proposal: Dict[str, Any]) -> List[str]:
        """Generate conditions for collaboration."""
        conditions = []

        # Risk-based conditions
        if proposal.get('risk_level', 0) > 0.7:
            conditions.append('risk_mitigation_required')

        # Resource-based conditions
        if proposal.get('resource_requirement', '') == 'high':
            conditions.append('resource_sharing_agreement')

        # Information sharing conditions
        conditions.append('bidirectional_information_flow')

        return conditions

    def _coordinate_strategy(self, agent: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate strategy with another agent."""
        # Evaluate strategy compatibility
        our_strategy = self._get_current_strategy()
        compatibility = self._assess_strategy_compatibility(our_strategy, strategy)

        if compatibility > 0.7:
            # High compatibility - full coordination
            coordination_plan = {
                'level': 'full',
                'adjustments': self._calculate_strategy_adjustments(our_strategy, strategy),
                'timing': 'synchronized'
            }
        elif compatibility > 0.4:
            # Moderate compatibility - partial coordination
            coordination_plan = {
                'level': 'partial',
                'adjustments': self._calculate_minimal_adjustments(our_strategy, strategy),
                'timing': 'sequential'
            }
        else:
            # Low compatibility - maintain independence
            coordination_plan = {
                'level': 'independent',
                'adjustments': None,
                'timing': 'independent'
            }

        return {
            'coordination_plan': coordination_plan,
            'compatibility_score': compatibility,
            'expected_synergy': compatibility * 0.3  # Synergy benefit estimate
        }

    def _get_current_strategy(self) -> Dict[str, Any]:
        """Get QBMIA's current strategy."""
        if not self.agent.last_decision:
            return {'type': 'adaptive', 'focus': 'analysis'}

        decision = self.agent.last_decision.get('integrated_decision', {})

        return {
            'type': 'quantum_biological',
            'action': decision.get('action', 'hold'),
            'focus': self._determine_strategic_focus(self.agent.last_decision),
            'time_horizon': 'medium',
            'risk_tolerance': 'adaptive'
        }

    def _determine_strategic_focus(self, analysis: Dict[str, Any]) -> str:
        """Determine current strategic focus."""
        # Check which component had highest influence
        components = analysis.get('component_results', {})

        if not components:
            return 'balanced'

        # Simplified determination
        if 'machiavellian' in components and components['machiavellian'].get('manipulation_detected', {}).get('detected'):
            return 'defensive'

        if 'antifragile' in components and components['antifragile'].get('volatility_benefit', 0) > 0.5:
            return 'volatility_exploitation'

        if 'robin_hood' in components and components['robin_hood'].get('intervention_needed', False):
            return 'market_intervention'

        return 'equilibrium_seeking'

    def _assess_strategy_compatibility(self, our_strategy: Dict[str, Any],
                                     their_strategy: Dict[str, Any]) -> float:
        """Assess compatibility between strategies."""
        compatibility = 0.0

        # Action compatibility
        if our_strategy.get('action') == their_strategy.get('action'):
            compatibility += 0.4
        elif (our_strategy.get('action') in ['buy', 'hold'] and
              their_strategy.get('action') in ['buy', 'hold']):
            compatibility += 0.2

        # Focus compatibility
        our_focus = our_strategy.get('focus', '')
        their_focus = their_strategy.get('focus', '')

        compatible_focuses = {
            'defensive': ['risk_management', 'protection'],
            'volatility_exploitation': ['antifragile', 'opportunity'],
            'equilibrium_seeking': ['balanced', 'stable'],
            'market_intervention': ['ethical', 'ecosystem']
        }

        if their_focus in compatible_focuses.get(our_focus, []):
            compatibility += 0.3

        # Time horizon compatibility
        if our_strategy.get('time_horizon') == their_strategy.get('time_horizon'):
            compatibility += 0.3

        return compatibility

    def _calculate_strategy_adjustments(self, our_strategy: Dict[str, Any],
                                      their_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate strategy adjustments for coordination."""
        adjustments = {}

        # Timing adjustments
        if our_strategy.get('action') == their_strategy.get('action'):
            adjustments['execution_timing'] = 'staggered_entry'  # Avoid market impact

        # Risk adjustments
        our_risk = our_strategy.get('risk_tolerance', 'adaptive')
        their_risk = their_strategy.get('risk_tolerance', 'moderate')

        if our_risk == 'adaptive':
            adjustments['risk_adjustment'] = f'align_with_{their_risk}'

        # Position sizing adjustments
        adjustments['position_sizing'] = 'proportional_scaling'

        return adjustments

    def _calculate_minimal_adjustments(self, our_strategy: Dict[str, Any],
                                     their_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate minimal adjustments for partial coordination."""
        return {
            'information_sharing': 'key_signals_only',
            'execution_coordination': 'avoid_conflicts',
            'risk_management': 'independent'
        }

    def _evaluate_proposal(self, proposal: Dict[str, Any],
                         other_positions: Dict[str, Any]) -> float:
        """Evaluate a consensus proposal."""
        score = 0.0

        # Evaluate based on QBMIA's analysis
        if self.agent.last_decision:
            decision = self.agent.last_decision.get('integrated_decision', {})

            # Action alignment
            if proposal.get('action') == decision.get('action'):
                score += 0.4

            # Risk alignment
            proposal_risk = proposal.get('risk_level', 0.5)
            our_risk = self._extract_risk_level(self.agent.last_decision)
            risk_diff = abs(proposal_risk - our_risk)
            score += 0.3 * (1.0 - risk_diff)

        # Evaluate based on unique QBMIA insights
        if self._incorporates_quantum_insights(proposal):
            score += 0.15

        if self._addresses_market_health(proposal):
            score += 0.15

        return score

    def _incorporates_quantum_insights(self, proposal: Dict[str, Any]) -> bool:
        """Check if proposal incorporates quantum insights."""
        rationale = proposal.get('rationale', '').lower()
        return any(term in rationale for term in ['quantum', 'superposition', 'equilibrium'])

    def _addresses_market_health(self, proposal: Dict[str, Any]) -> bool:
        """Check if proposal addresses market health concerns."""
        rationale = proposal.get('rationale', '').lower()
        objectives = proposal.get('objectives', [])

        return (any(term in rationale for term in ['ecosystem', 'health', 'intervention']) or
                'market_health' in objectives)

    def _identify_concerns(self, proposal: Dict[str, Any]) -> List[str]:
        """Identify concerns with a proposal."""
        concerns = []

        # Risk concerns
        if proposal.get('risk_level', 0) > 0.8:
            concerns.append('excessive_risk')

        # Market manipulation concerns
        if self.agent.last_decision:
            machiavellian = self.agent.last_decision.get('market_intelligence', {}).get('machiavellian', {})
            if machiavellian.get('manipulation_detected', {}).get('detected', False):
                concerns.append('market_manipulation_risk')

        # Ecosystem health concerns
        if not self._addresses_market_health(proposal) and self._market_health_poor():
            concerns.append('ignores_market_health')

        return concerns

    def _market_health_poor(self) -> bool:
        """Check if market health is poor."""
        if not self.agent.last_decision:
            return False

        robin_hood = self.agent.last_decision.get('market_intelligence', {}).get('robin_hood', {})
        health_score = robin_hood.get('ecosystem_health', {}).get('overall_health_score', 1.0)

        return health_score < 0.6

    def _suggest_improvements(self, proposal: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest improvements to a proposal."""
        improvements = []

        # Add quantum insights if missing
        if not self._incorporates_quantum_insights(proposal):
            improvements.append({
                'aspect': 'strategic_depth',
                'suggestion': 'incorporate_quantum_equilibrium_analysis'
            })

        # Add market health considerations
        if not self._addresses_market_health(proposal) and self._market_health_poor():
            improvements.append({
                'aspect': 'ecosystem_impact',
                'suggestion': 'add_market_health_objectives'
            })

        # Add antifragile elements for high volatility
        if self._high_volatility_detected() and 'antifragile' not in proposal.get('strategy', ''):
            improvements.append({
                'aspect': 'volatility_strategy',
                'suggestion': 'incorporate_antifragile_elements'
            })

        return improvements

    def _high_volatility_detected(self) -> bool:
        """Check if high volatility is detected."""
        if not self.agent.last_decision:
            return False

        antifragile = self.agent.last_decision.get('market_intelligence', {}).get('antifragile', {})
        return antifragile.get('volatility_benefit', 0) > 0.5

    def _check_consensus_possibility(self, evaluations: List[Dict[str, Any]],
                                   other_positions: Dict[str, Any]) -> bool:
        """Check if consensus is possible."""
        # Get best proposal score
        best_score = max(e['score'] for e in evaluations)

        if best_score < 0.6:
            return False  # No acceptable proposal

        # Check if others are converging
        position_variance = self._calculate_position_variance(other_positions)

        return position_variance < 0.3  # Low variance indicates convergence

    def _calculate_position_variance(self, positions: Dict[str, Any]) -> float:
        """Calculate variance in agent positions."""
        if not positions:
            return 1.0

        # Extract key position elements
        actions = []
        confidences = []

        for agent, position in positions.items():
            if isinstance(position, dict):
                action = position.get('action', 'hold')
                confidence = position.get('confidence', 0.5)

                # Convert action to numeric
                action_map = {'buy': 1.0, 'sell': -1.0, 'hold': 0.0, 'wait': 0.0}
                actions.append(action_map.get(action, 0.0))
                confidences.append(confidence)

        if actions:
            action_variance = np.var(actions)
            confidence_variance = np.var(confidences)

            return (action_variance + confidence_variance) / 2

        return 1.0

    def _generate_compromise_suggestions(self, proposals: List[Dict[str, Any]],
                                       other_positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate compromise suggestions for consensus building."""
        suggestions = []

        # Identify common ground
        common_elements = self._find_common_ground(proposals)

        if common_elements:
            suggestions.append({
                'type': 'build_on_common_ground',
                'elements': common_elements,
                'approach': 'incremental_agreement'
            })

        # Suggest phased approach for divergent views
        if self._calculate_position_variance(other_positions) > 0.5:
            suggestions.append({
                'type': 'phased_execution',
                'phases': [
                    'information_gathering',
                    'small_scale_test',
                    'full_implementation'
                ],
                'approach': 'risk_reduction'
            })

        # Suggest coalition approach
        aligned_agents = self._find_aligned_agents(other_positions)
        if len(aligned_agents) >= 2:
            suggestions.append({
                'type': 'coalition_formation',
                'aligned_agents': aligned_agents,
                'approach': 'majority_influence'
            })

        return suggestions

    def _find_common_ground(self, proposals: List[Dict[str, Any]]) -> List[str]:
        """Find common elements across proposals."""
        if not proposals:
            return []

        common = []

        # Check for common objectives
        all_objectives = [set(p.get('objectives', [])) for p in proposals]
        if all_objectives:
            common_objectives = set.intersection(*all_objectives)
            common.extend(list(common_objectives))

        # Check for common risk management approaches
        risk_approaches = [p.get('risk_management', '') for p in proposals]
        if len(set(risk_approaches)) == 1 and risk_approaches[0]:
            common.append(f'risk_management_{risk_approaches[0]}')

        return common

    def _find_aligned_agents(self, positions: Dict[str, Any]) -> List[str]:
        """Find agents with aligned positions."""
        if not self.agent.last_decision:
            return []

        our_decision = self.agent.last_decision.get('integrated_decision', {})
        our_action = our_decision.get('action', 'hold')

        aligned = []

        for agent, position in positions.items():
            if isinstance(position, dict):
                their_action = position.get('action', 'hold')
                if their_action == our_action:
                    aligned.append(agent)

        return aligned

    def _update_collaboration_score(self, agent: str, delta: float):
        """Update collaboration score with another agent."""
        if agent in self.boardroom_metrics['collaboration_scores']:
            current = self.boardroom_metrics['collaboration_scores'][agent]
            new_score = max(0.0, min(1.0, current + delta))
            self.boardroom_metrics['collaboration_scores'][agent] = new_score

    def _calculate_unique_contributions(self) -> float:
        """Calculate unique contribution score."""
        # Based on how often QBMIA's insights differ from consensus
        if not self.consensus_history:
            return 0.5

        unique_count = sum(1 for c in self.consensus_history[-10:]
                          if c.get('unique_insight', False))

        return unique_count / min(10, len(self.consensus_history))

    def _summarize_achievements(self) -> List[str]:
        """Summarize key achievements."""
        achievements = []

        # High decision participation
        if self.boardroom_metrics['decisions_participated'] > 10:
            achievements.append(f"participated_in_{self.boardroom_metrics['decisions_participated']}_decisions")

        # High consensus rate
        consensus_rate = (self.boardroom_metrics['consensus_achieved'] /
                         max(1, self.boardroom_metrics['decisions_participated']))
        if consensus_rate > 0.7:
            achievements.append(f"achieved_{consensus_rate:.0%}_consensus_rate")

        # Strong collaboration
        avg_collab = np.mean(list(self.boardroom_metrics['collaboration_scores'].values()))
        if avg_collab > 0.7:
            achievements.append("maintained_strong_collaboration")

        # Unique insights
        if hasattr(self.agent, 'performance_tracker'):
            unique_insights = getattr(self.agent.performance_tracker, 'unique_insights_count', 0)
            if unique_insights > 5:
                achievements.append(f"provided_{unique_insights}_unique_insights")

        return achievements

    def _identify_improvement_areas(self) -> List[str]:
        """Identify areas for improvement."""
        areas = []

        # Low consensus achievement
        consensus_rate = (self.boardroom_metrics['consensus_achieved'] /
                         max(1, self.boardroom_metrics['decisions_participated']))
        if consensus_rate < 0.5:
            areas.append("consensus_building")

        # Weak collaboration with specific agents
        for agent, score in self.boardroom_metrics['collaboration_scores'].items():
            if score < 0.3:
                areas.append(f"collaboration_with_{agent}")

        # Low influence
        if self.boardroom_metrics['influence_scores']:
            avg_influence = np.mean(self.boardroom_metrics['influence_scores'])
            if avg_influence < 0.15:
                areas.append("boardroom_influence")

        return areas

    def get_boardroom_status(self) -> Dict[str, Any]:
        """Get current boardroom participation status."""
        return {
            'agent_id': self.agent_id,
            'active': True,
            'current_context': self.current_decision_context,
            'metrics': self.boardroom_metrics,
            'last_recommendation': self.agent.last_decision.get('integrated_decision') if self.agent.last_decision else None,
            'collaboration_health': {
                agent: 'strong' if score > 0.7 else 'moderate' if score > 0.4 else 'weak'
                for agent, score in self.boardroom_metrics['collaboration_scores'].items()
            }
        }
