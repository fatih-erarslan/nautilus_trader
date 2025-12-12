"""
PADS (Panarchy Adaptive Decision System) integration for QBMIA.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import numpy as np
from collections import deque
import sys
import os

# Add parent directory to path for unified messaging import
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from unified_messaging import (
        UnifiedMessenger, Message, MessageType, AgentType,
        create_qbmia_messenger
    )
    UNIFIED_MESSAGING_AVAILABLE = True
except ImportError:
    UNIFIED_MESSAGING_AVAILABLE = False
    logger.warning("Unified messaging not available, falling back to simulated communication")

logger = logging.getLogger(__name__)

class PADSConnector:
    """
    Connector for integrating QBMIA with the Panarchy Adaptive Decision System.
    """

    def __init__(self, qbmia_agent: Any, pads_config: Dict[str, Any]):
        """
        Initialize PADS connector.

        Args:
            qbmia_agent: QBMIA agent instance
            pads_config: PADS configuration
        """
        self.agent = qbmia_agent
        self.config = pads_config

        # PADS connection state
        self.connected = False
        self.pads_endpoint = self.config.get('endpoint', 'http://localhost:9090')

        # Real messaging implementation
        self.use_real_messaging = UNIFIED_MESSAGING_AVAILABLE and self.config.get('use_real_messaging', True)
        self.messenger: Optional[UnifiedMessenger] = None
        
        if self.use_real_messaging:
            # Initialize unified messenger for QBMIA
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            zmq_ports = self.config.get('zmq_ports', {})
            self.messenger = create_qbmia_messenger(
                redis_url=redis_url,
                zmq_connect_ports=zmq_ports
            )
            logger.info("QBMIA PADS connector initialized with real messaging")
        else:
            logger.warning("QBMIA PADS connector using simulated messaging")

        self.system_id = self.config.get('system_id', 'QBMIA_PADS_001')

        # Panarchy levels
        self.panarchy_levels = {
            'micro': {'scale': 'individual', 'time_horizon': 'minutes'},
            'meso': {'scale': 'group', 'time_horizon': 'hours'},
            'macro': {'scale': 'system', 'time_horizon': 'days'}
        }

        # Current system state in adaptive cycle
        self.adaptive_cycle_phase = 'exploitation'  # growth, conservation, release, reorganization
        self.panarchy_position = {
            'level': 'meso',
            'resilience': 0.7,
            'potential': 0.5,
            'connectedness': 0.6
        }

        # Cross-scale interactions
        self.cross_scale_memory = deque(maxlen=100)
        self.revolt_connections = []  # Bottom-up effects
        self.remember_connections = []  # Top-down effects

        # System metrics
        self.pads_metrics = {
            'adaptive_capacity': 0.7,
            'transformability': 0.5,
            'system_resilience': 0.6,
            'phase_transitions': 0,
            'cross_scale_interactions': 0
        }

    async def connect_to_pads(self) -> bool:
        """
        Establish connection to PADS.

        Returns:
            Connection success status
        """
        try:
            logger.info(f"Connecting to PADS at {self.pads_endpoint}")

            # Connect using real messaging if available
            if self.use_real_messaging and self.messenger:
                # Connect to unified messaging system
                messaging_connected = await self.messenger.connect()
                
                if messaging_connected:
                    # Register message handlers
                    self._register_message_handlers()
                    
                    # Start listening for messages
                    asyncio.create_task(self.messenger.start_listening())
                    
                    logger.info("Connected to PADS via unified messaging")
                else:
                    logger.error("Failed to connect to unified messaging system")
                    self.use_real_messaging = False  # Fall back to simulated

            # Register with PADS (either real or simulated)
            registration = await self._register_with_pads()

            if registration['status'] == 'success':
                self.connected = True

                # Initialize cross-scale connections
                await self._initialize_cross_scale_connections()

                # Start monitoring tasks
                asyncio.create_task(self._monitor_adaptive_cycle())
                asyncio.create_task(self._monitor_cross_scale_effects())

                logger.info(f"Successfully connected to PADS as {self.system_id}")
                return True
            else:
                logger.error(f"PADS registration failed: {registration.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to PADS: {e}")
            return False

    async def _register_with_pads(self) -> Dict[str, Any]:
        """Register QBMIA with PADS."""
        registration_data = {
            'system_id': self.system_id,
            'system_type': 'quantum_biological_agent',
            'capabilities': {
                'quantum_processing': True,
                'biological_learning': True,
                'market_analysis': True,
                'adaptive_strategies': True
            },
            'panarchy_level': 'meso',
            'initial_phase': self.adaptive_cycle_phase,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Simulated registration response
        return {
            'status': 'success',
            'assigned_role': 'adaptive_agent',
            'panarchy_position': self.panarchy_position
        }

    async def _initialize_cross_scale_connections(self):
        """Initialize connections across panarchy scales."""
        # Connect to lower scale (revolt connections)
        self.revolt_connections = [
            {
                'source_level': 'micro',
                'target_level': 'meso',
                'connection_type': 'innovation_cascade',
                'strength': 0.6
            },
            {
                'source_level': 'micro',
                'target_level': 'meso',
                'connection_type': 'crisis_propagation',
                'strength': 0.8
            }
        ]

        # Connect to higher scale (remember connections)
        self.remember_connections = [
            {
                'source_level': 'macro',
                'target_level': 'meso',
                'connection_type': 'institutional_memory',
                'strength': 0.7
            },
            {
                'source_level': 'macro',
                'target_level': 'meso',
                'connection_type': 'resource_subsidy',
                'strength': 0.5
            }
        ]

    async def submit_decision_to_pads(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit QBMIA decision to PADS for system-wide coordination.

        Args:
            decision: QBMIA decision

        Returns:
            PADS response
        """
        if not self.connected:
            raise ConnectionError("Not connected to PADS")

        # Enrich decision with panarchy context
        pads_decision = {
            'system_id': self.system_id,
            'decision': decision,
            'panarchy_context': {
                'current_phase': self.adaptive_cycle_phase,
                'level': self.panarchy_position['level'],
                'resilience': self.panarchy_position['resilience'],
                'cross_scale_effects': self._analyze_cross_scale_effects(decision)
            },
            'adaptive_capacity': self.pads_metrics['adaptive_capacity'],
            'timestamp': datetime.utcnow().isoformat()
        }

        # Submit to PADS
        response = await self._send_to_pads('decision_submission', pads_decision)

        # Process PADS feedback
        if response['status'] == 'accepted':
            # Update system state based on PADS feedback
            await self._process_pads_feedback(response['feedback'])

        return response

    def _analyze_cross_scale_effects(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential cross-scale effects of decision."""
        effects = {
            'upward_effects': [],
            'downward_effects': [],
            'lateral_effects': []
        }

        # Analyze upward effects (revolt)
        if decision.get('confidence', 0) > 0.8:
            effects['upward_effects'].append({
                'type': 'confidence_cascade',
                'target_level': 'macro',
                'potential_impact': 'stabilizing',
                'magnitude': decision['confidence'] * 0.5
            })

        # Check for crisis indicators
        if 'component_results' in decision:
            if decision['component_results'].get('machiavellian', {}).get('manipulation_detected', {}).get('detected'):
                effects['upward_effects'].append({
                    'type': 'manipulation_alert',
                    'target_level': 'macro',
                    'potential_impact': 'destabilizing',
                    'magnitude': 0.7
                })

        # Analyze downward effects (remember)
        if self.adaptive_cycle_phase in ['release', 'reorganization']:
            effects['downward_effects'].append({
                'type': 'innovation_opportunity',
                'target_level': 'micro',
                'potential_impact': 'transformative',
                'magnitude': 0.6
            })

        return effects

    async def _monitor_adaptive_cycle(self):
        """Monitor and update adaptive cycle phase."""
        while self.connected:
            try:
                # Get system state indicators
                indicators = await self._collect_system_indicators()

                # Determine current phase
                new_phase = self._determine_adaptive_phase(indicators)

                if new_phase != self.adaptive_cycle_phase:
                    # Phase transition detected
                    await self._handle_phase_transition(
                        self.adaptive_cycle_phase,
                        new_phase,
                        indicators
                    )

                    self.adaptive_cycle_phase = new_phase
                    self.pads_metrics['phase_transitions'] += 1

                # Update panarchy position
                self._update_panarchy_position(indicators)

                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Adaptive cycle monitoring error: {e}")
                await asyncio.sleep(60)

    async def _collect_system_indicators(self) -> Dict[str, float]:
        """Collect indicators for adaptive cycle assessment."""
        indicators = {}

        # Get QBMIA performance metrics
        agent_status = self.agent.get_status()

        # Resource availability (potential)
        memory_usage = agent_status.get('memory_usage', {})
        indicators['resource_availability'] = 1.0 - memory_usage.get('capacity_percentage', 0) / 100

        # System complexity (connectedness)
        if hasattr(self.agent, 'boardroom_interface'):
            boardroom_status = self.agent.boardroom_interface.get_boardroom_status()
            collaboration_scores = boardroom_status['metrics']['collaboration_scores']
            indicators['connectedness'] = np.mean(list(collaboration_scores.values()))
        else:
            indicators['connectedness'] = 0.5

        # Performance stability (resilience)
        performance = agent_status.get('performance', {})
        if 'recent_decisions' in performance:
            decision_variance = np.var([d['confidence'] for d in performance['recent_decisions'][-10:]])
            indicators['stability'] = 1.0 - min(1.0, decision_variance * 10)
        else:
            indicators['stability'] = 0.7

        # Innovation rate
        if hasattr(self.agent, 'strategy_innovation_rate'):
            indicators['innovation'] = self.agent.strategy_innovation_rate
        else:
            indicators['innovation'] = 0.3

        # Crisis indicators
        if self.agent.last_decision:
            crisis_level = 0.0

            # Check manipulation
            machiavellian = self.agent.last_decision.get('market_intelligence', {}).get('machiavellian', {})
            if machiavellian.get('manipulation_detected', {}).get('detected'):
                crisis_level += 0.3

            # Check volatility
            antifragile = self.agent.last_decision.get('market_intelligence', {}).get('antifragile', {})
            if antifragile.get('volatility_benefit', 0) > 0.7:
                crisis_level += 0.2

            indicators['crisis_level'] = crisis_level
        else:
            indicators['crisis_level'] = 0.0

        return indicators

    def _determine_adaptive_phase(self, indicators: Dict[str, float]) -> str:
        """
        Determine current phase in adaptive cycle based on indicators.

        Phases:
        - growth (r): High innovation, low connectedness, increasing resources
        - conservation (K): Low innovation, high connectedness, high resources
        - release (Ω): Crisis/disruption, declining stability
        - reorganization (α): High innovation potential, low connectedness
        """
        innovation = indicators.get('innovation', 0.3)
        connectedness = indicators.get('connectedness', 0.5)
        resources = indicators.get('resource_availability', 0.5)
        stability = indicators.get('stability', 0.7)
        crisis = indicators.get('crisis_level', 0.0)

        # Current phase
        current = self.adaptive_cycle_phase

        # Phase transition logic
        if current == 'growth':
            # Transition to conservation when connectedness increases
            if connectedness > 0.7 and resources > 0.6:
                return 'conservation'

        elif current == 'conservation':
            # Transition to release when crisis occurs or resources depleted
            if crisis > 0.5 or resources < 0.3 or stability < 0.4:
                return 'release'

        elif current == 'release':
            # Transition to reorganization when release completes
            if stability < 0.3 and connectedness < 0.4:
                return 'reorganization'

        elif current == 'reorganization':
            # Transition to growth when innovation increases
            if innovation > 0.5 and resources > 0.4:
                return 'growth'

        # Check for forced transitions
        if crisis > 0.7 and current != 'release':
            return 'release'

        return current

    async def _handle_phase_transition(self, old_phase: str, new_phase: str,
                                     indicators: Dict[str, float]):
        """Handle adaptive cycle phase transition."""
        logger.info(f"PADS phase transition: {old_phase} -> {new_phase}")

        # Notify PADS of phase transition
        transition_event = {
            'system_id': self.system_id,
            'transition': f"{old_phase}_to_{new_phase}",
            'indicators': indicators,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self._send_to_pads('phase_transition', transition_event)

        # Adjust QBMIA behavior based on new phase
        if new_phase == 'growth':
            # Increase exploration and innovation
            if hasattr(self.agent, 'set_exploration_rate'):
                self.agent.set_exploration_rate(0.3)

        elif new_phase == 'conservation':
            # Focus on efficiency and optimization
            if hasattr(self.agent, 'set_exploration_rate'):
                self.agent.set_exploration_rate(0.1)

        elif new_phase == 'release':
            # Activate crisis management strategies
            logger.warning("Entering release phase - activating crisis strategies")
            if hasattr(self.agent, 'activate_crisis_mode'):
                self.agent.activate_crisis_mode()

        elif new_phase == 'reorganization':
            # Enable radical innovation and restructuring
            if hasattr(self.agent, 'enable_reorganization_mode'):
                self.agent.enable_reorganization_mode()

    def _update_panarchy_position(self, indicators: Dict[str, float]):
        """Update position in panarchy space."""
        # Update resilience
        self.panarchy_position['resilience'] = indicators.get('stability', 0.7)

        # Update potential
        self.panarchy_position['potential'] = indicators.get('resource_availability', 0.5)

        # Update connectedness
        self.panarchy_position['connectedness'] = indicators.get('connectedness', 0.5)

        # Calculate adaptive capacity
        self.pads_metrics['adaptive_capacity'] = (
            self.panarchy_position['resilience'] * 0.4 +
            self.panarchy_position['potential'] * 0.3 +
            (1 - self.panarchy_position['connectedness']) * 0.3  # Lower connectedness = higher adaptability
        )

        # Calculate transformability
        if self.adaptive_cycle_phase in ['release', 'reorganization']:
            self.pads_metrics['transformability'] = 0.8
        else:
            self.pads_metrics['transformability'] = 0.3

    async def _monitor_cross_scale_effects(self):
        """Monitor cross-scale interactions in panarchy."""
        while self.connected:
            try:
                # Check for revolt (bottom-up) effects
                revolt_signals = await self._detect_revolt_signals()

                if revolt_signals:
                    for signal in revolt_signals:
                        await self._process_revolt_signal(signal)

                # Check for remember (top-down) effects
                remember_signals = await self._detect_remember_signals()

                if remember_signals:
                    for signal in remember_signals:
                        await self._process_remember_signal(signal)

                # Update cross-scale metrics
                self.pads_metrics['cross_scale_interactions'] = (
                    len(revolt_signals) + len(remember_signals)
                )

                # Store in memory
                self.cross_scale_memory.append({
                    'timestamp': datetime.utcnow(),
                    'revolt_count': len(revolt_signals),
                    'remember_count': len(remember_signals),
                    'phase': self.adaptive_cycle_phase
                })

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Cross-scale monitoring error: {e}")
                await asyncio.sleep(30)

    async def _detect_revolt_signals(self) -> List[Dict[str, Any]]:
        """Detect bottom-up (revolt) signals from lower scales."""
        signals = []

        # Check for micro-level disruptions
        if hasattr(self.agent, 'detect_micro_disruptions'):
            disruptions = self.agent.detect_micro_disruptions()

            for disruption in disruptions:
                if disruption['severity'] > 0.6:
                    signals.append({
                        'type': 'revolt',
                        'source': 'micro',
                        'nature': disruption['type'],
                        'severity': disruption['severity'],
                        'potential_cascade': disruption['severity'] > 0.8
                    })

        # Check for innovation cascades
        if self.agent.last_decision:
            temporal_nash = self.agent.last_decision.get('market_intelligence', {}).get('temporal_nash', {})
            if temporal_nash.get('learning_improvement', 0) > 0.5:
                signals.append({
                    'type': 'revolt',
                    'source': 'micro',
                    'nature': 'innovation_cascade',
                    'severity': 0.6,
                    'potential_cascade': True
                })

        return signals

    async def _detect_remember_signals(self) -> List[Dict[str, Any]]:
        """Detect top-down (remember) signals from higher scales."""
        signals = []

        # Check for institutional constraints
        if self.adaptive_cycle_phase == 'reorganization':
            # Higher scales provide memory and resources during reorganization
            signals.append({
                'type': 'remember',
                'source': 'macro',
                'nature': 'institutional_memory',
                'strength': 0.7,
                'resources': {
                    'knowledge': 'historical_patterns',
                    'constraints': 'regulatory_framework'
                }
            })

        # Check for resource subsidies
        if self.panarchy_position['potential'] < 0.3:
            signals.append({
                'type': 'remember',
                'source': 'macro',
                'nature': 'resource_subsidy',
                'strength': 0.5,
                'resources': {
                    'computational': 'additional_capacity',
                    'informational': 'macro_trends'
                }
            })

        return signals

    def _register_message_handlers(self) -> None:
        """Register message handlers for real communication."""
        if not self.messenger:
            return
            
        # Register handlers for different message types
        self.messenger.register_handler(MessageType.DECISION_REQUEST, self._handle_decision_request)
        self.messenger.register_handler(MessageType.PHASE_TRANSITION, self._handle_phase_transition)
        self.messenger.register_handler(MessageType.RISK_ALERT, self._handle_risk_alert)
        self.messenger.register_handler(MessageType.SYSTEM_COMMAND, self._handle_system_command)
        
        logger.info("Registered QBMIA message handlers")
    
    async def _handle_decision_request(self, message: Message) -> None:
        """Handle decision request from PADS."""
        try:
            request_data = message.data
            market_data = request_data.get('market_data', {})
            position_state = request_data.get('position_state', {})
            
            # Get decision from QBMIA agent
            if hasattr(self.agent, 'make_decision'):
                decision_result = await self.agent.make_decision(market_data, position_state)
                
                # Send response back to PADS
                response = Message(
                    message_type=MessageType.DECISION_RESPONSE,
                    sender=AgentType.QBMIA,
                    recipient=AgentType.PADS,
                    data={
                        'decision': decision_result,
                        'system_id': self.system_id,
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    correlation_id=message.id,
                    priority=1
                )
                
                await self.messenger.send_message(response)
                logger.info(f"Sent decision response: {decision_result}")
                
        except Exception as e:
            logger.error(f"Error handling decision request: {e}")
    
    async def _handle_phase_transition(self, message: Message) -> None:
        """Handle market phase transition notification."""
        try:
            phase_data = message.data
            new_phase = phase_data.get('new_phase')
            old_phase = phase_data.get('old_phase')
            
            logger.info(f"QBMIA received phase transition: {old_phase} -> {new_phase}")
            
            # Update internal state
            if new_phase:
                self.adaptive_cycle_phase = new_phase
                
            # Notify agent if it has a phase change handler
            if hasattr(self.agent, 'handle_phase_transition'):
                await self.agent.handle_phase_transition(old_phase, new_phase, phase_data)
                
        except Exception as e:
            logger.error(f"Error handling phase transition: {e}")
    
    async def _handle_risk_alert(self, message: Message) -> None:
        """Handle risk alert from PADS or other agents."""
        try:
            alert_data = message.data
            alert_type = alert_data.get('alert_type')
            severity = alert_data.get('severity', 0.5)
            
            logger.warning(f"QBMIA received risk alert: {alert_type} (severity: {severity})")
            
            # Adjust agent behavior based on risk level
            if hasattr(self.agent, 'handle_risk_alert'):
                await self.agent.handle_risk_alert(alert_type, severity, alert_data)
                
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_system_command(self, message: Message) -> None:
        """Handle system commands from PADS."""
        try:
            command_data = message.data
            command = command_data.get('command')
            
            logger.info(f"QBMIA received system command: {command}")
            
            if command == 'status_request':
                # Send status response
                status = self.get_pads_status()
                response = Message(
                    message_type=MessageType.AGENT_STATUS,
                    sender=AgentType.QBMIA,
                    recipient=AgentType.PADS,
                    data=status,
                    correlation_id=message.id
                )
                await self.messenger.send_message(response)
                
            elif command == 'shutdown':
                logger.info("QBMIA received shutdown command")
                await self.disconnect_from_pads()
                
        except Exception as e:
            logger.error(f"Error handling system command: {e}")

    async def _process_revolt_signal(self, signal: Dict[str, Any]):
        """Process bottom-up revolt signal."""
        logger.info(f"Processing revolt signal: {signal['nature']}")

        if signal['potential_cascade']:
            # Alert higher scales
            cascade_alert = {
                'system_id': self.system_id,
                'signal_type': 'revolt_cascade',
                'signal': signal,
                'current_phase': self.adaptive_cycle_phase,
                'recommended_action': self._recommend_cascade_response(signal)
            }

            await self._send_to_pads('cascade_alert', cascade_alert)

        # Adjust local behavior
        if signal['nature'] == 'innovation_cascade':
            # Increase receptivity to innovation
            if hasattr(self.agent, 'increase_innovation_receptivity'):
                self.agent.increase_innovation_receptivity()

        elif signal['severity'] > 0.8:
            # Prepare for potential phase transition
            self.pads_metrics['system_resilience'] *= 0.9

    async def _process_remember_signal(self, signal: Dict[str, Any]):
        """Process top-down remember signal."""
        logger.info(f"Processing remember signal: {signal['nature']}")

        if signal['nature'] == 'institutional_memory':
            # Incorporate historical patterns
            if hasattr(self.agent, 'memory'):
                # Load historical patterns into memory
                historical_patterns = signal['resources'].get('knowledge', {})
                if historical_patterns:
                    self.agent.memory.apply_attention(['historical_patterns'])

        elif signal['nature'] == 'resource_subsidy':
            # Accept resource support
            resources = signal['resources']

            if 'computational' in resources:
                # Increase computational capacity
                if hasattr(self.agent, 'hw_optimizer'):
                    self.agent.hw_optimizer.allocate_memory_pool(
                        'subsidized_compute',
                        2048  # 2GB additional
                    )

            # Update metrics
            self.panarchy_position['potential'] += 0.1

    def _recommend_cascade_response(self, signal: Dict[str, Any]) -> str:
        """Recommend response to cascade signal."""
        if signal['severity'] > 0.8:
            if self.adaptive_cycle_phase == 'conservation':
                return 'prepare_for_release'
            else:
                return 'increase_resilience'
        else:
            return 'monitor_closely'

    async def _send_to_pads(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to PADS using real or simulated communication."""
        
        if self.use_real_messaging and self.messenger and self.connected:
            try:
                # Map string message types to MessageType enum
                msg_type_map = {
                    'decision_submission': MessageType.DECISION_RESPONSE,
                    'phase_transition': MessageType.PHASE_TRANSITION,
                    'cascade_alert': MessageType.RISK_ALERT,
                    'disconnection': MessageType.AGENT_STATUS
                }
                
                enum_message_type = msg_type_map.get(message_type, MessageType.SYSTEM_COMMAND)
                
                # Create unified message
                message = Message(
                    message_type=enum_message_type,
                    sender=AgentType.QBMIA,
                    recipient=AgentType.PADS,
                    data=data,
                    priority=1 if message_type in ['cascade_alert', 'decision_submission'] else 2
                )
                
                # Send message
                success = await self.messenger.send_message(message)
                
                if success:
                    logger.debug(f"Sent real message to PADS: {message_type}")
                    return {
                        'status': 'accepted',
                        'message_id': message.id,
                        'feedback': {
                            'recommendation': 'continue_monitoring',
                            'system_health': 'stable'
                        }
                    }
                else:
                    logger.error(f"Failed to send real message to PADS: {message_type}")
                    # Fall back to simulated
                    
            except Exception as e:
                logger.error(f"Error sending real message to PADS: {e}")
                # Fall back to simulated
        
        # Simulated PADS communication (fallback)
        message = {
            'type': message_type,
            'source': self.system_id,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Simulate response
        return {
            'status': 'accepted',
            'feedback': {
                'recommendation': 'continue_monitoring',
                'system_health': 'stable'
            }
        }

    async def _process_pads_feedback(self, feedback: Dict[str, Any]):
        """Process feedback from PADS."""
        recommendation = feedback.get('recommendation', '')

        if recommendation == 'increase_adaptability':
            self.pads_metrics['adaptive_capacity'] = min(1.0,
                self.pads_metrics['adaptive_capacity'] * 1.1)

        elif recommendation == 'strengthen_connections':
            # Increase collaboration with other systems
            if hasattr(self.agent, 'boardroom_interface'):
                # Increase collaboration efforts
                pass

        elif recommendation == 'prepare_transformation':
            self.pads_metrics['transformability'] = min(1.0,
                self.pads_metrics['transformability'] * 1.2)

    def get_pads_status(self) -> Dict[str, Any]:
        """Get current PADS integration status."""
        return {
            'connected': self.connected,
            'system_id': self.system_id,
            'adaptive_cycle_phase': self.adaptive_cycle_phase,
            'panarchy_position': self.panarchy_position.copy(),
            'metrics': self.pads_metrics.copy(),
            'cross_scale_activity': {
                'revolt_connections': len(self.revolt_connections),
                'remember_connections': len(self.remember_connections),
                'recent_interactions': len([m for m in self.cross_scale_memory
                                          if (datetime.utcnow() - m['timestamp']).seconds < 3600])
            }
        }

    async def disconnect_from_pads(self):
        """Disconnect from PADS."""
        if self.connected:
            # Send disconnection notice
            await self._send_to_pads('disconnection', {
                'reason': 'graceful_shutdown',
                'final_state': self.get_pads_status()
            })

            # Disconnect unified messenger if using real communication
            if self.use_real_messaging and self.messenger:
                await self.messenger.disconnect()
                logger.info("Disconnected from unified messaging system")

            self.connected = False
            logger.info(f"Disconnected from PADS")
