#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PADS messaging integration for real-time communication with quantum-biological trading agents.

This module enhances the PADS (Panarchy Adaptive Decision System) with real
messaging capabilities for coordinating decisions across multiple agents.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
import threading

try:
    from unified_messaging import (
        UnifiedMessenger, Message, MessageType, AgentType,
        create_pads_messenger
    )
    UNIFIED_MESSAGING_AVAILABLE = True
except ImportError:
    UNIFIED_MESSAGING_AVAILABLE = False

logger = logging.getLogger("PADSMessaging")

class PADSMessagingIntegration:
    """
    Messaging integration for PADS orchestrator.
    
    Handles real-time communication with QBMIA, QUASAR, Quantum AMOS,
    and other trading agents for coordinated decision making.
    """
    
    def __init__(self, pads_instance, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PADS messaging integration.
        
        Args:
            pads_instance: PanarchyAdaptiveDecisionSystem instance
            config: Messaging configuration
        """
        self.pads = pads_instance
        self.config = config or {}
        
        # Communication state
        self.connected = False
        self.use_real_messaging = UNIFIED_MESSAGING_AVAILABLE and self.config.get('use_real_messaging', True)
        self.messenger: Optional[UnifiedMessenger] = None
        
        if self.use_real_messaging:
            # Initialize unified messenger for PADS
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            zmq_ports = self.config.get('zmq_ports', {})
            self.messenger = create_pads_messenger(
                redis_url=redis_url,
                zmq_bind_port=9090,  # PADS is the central coordinator
                zmq_connect_ports=zmq_ports
            )
            logger.info("PADS messaging integration initialized with real messaging")
        else:
            logger.warning("PADS messaging integration using simulated messaging")
            
        # Agent coordination state
        self.agent_responses = {}
        self.decision_requests = {}
        self.active_agents = set()
        
        # Message processing
        self._processing_lock = threading.RLock()
        
    async def connect(self) -> bool:
        """Connect to the messaging system."""
        if self.use_real_messaging and self.messenger:
            try:
                # Connect to unified messaging
                messaging_connected = await self.messenger.connect()
                
                if messaging_connected:
                    # Register message handlers
                    self._register_message_handlers()
                    
                    # Start listening for messages
                    asyncio.create_task(self.messenger.start_listening())
                    
                    self.connected = True
                    logger.info("PADS connected to unified messaging system")
                    
                    # Send startup notification
                    await self._announce_pads_startup()
                    
                    return True
                else:
                    logger.error("Failed to connect PADS to unified messaging")
                    self.use_real_messaging = False
                    
            except Exception as e:
                logger.error(f"Error connecting PADS to messaging: {e}")
                self.use_real_messaging = False
        
        # Mark as connected even if using simulated messaging
        self.connected = True
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from the messaging system."""
        if self.use_real_messaging and self.messenger:
            # Send shutdown notification
            await self._announce_pads_shutdown()
            
            await self.messenger.disconnect()
            logger.info("PADS disconnected from unified messaging")
            
        self.connected = False
        
    def _register_message_handlers(self) -> None:
        """Register message handlers for PADS."""
        if not self.messenger:
            return
            
        # Register handlers for different message types
        self.messenger.register_handler(MessageType.DECISION_RESPONSE, self._handle_decision_response)
        self.messenger.register_handler(MessageType.AGENT_STATUS, self._handle_agent_status)
        self.messenger.register_handler(MessageType.MARKET_UPDATE, self._handle_market_update)
        self.messenger.register_handler(MessageType.RISK_ALERT, self._handle_risk_alert)
        self.messenger.register_handler(MessageType.SYSTEM_COMMAND, self._handle_system_command)
        
        logger.info("Registered PADS message handlers")
    
    async def _handle_decision_response(self, message: Message) -> None:
        """Handle decision responses from agents."""
        try:
            with self._processing_lock:
                response_data = message.data
                agent_type = message.sender.value
                correlation_id = message.correlation_id
                
                logger.debug(f"PADS received decision response from {agent_type}")
                
                # Store response for correlation
                if correlation_id:
                    if correlation_id not in self.agent_responses:
                        self.agent_responses[correlation_id] = {}
                    
                    self.agent_responses[correlation_id][agent_type] = {
                        'decision': response_data.get('decision'),
                        'confidence': response_data.get('confidence', 0.5),
                        'timestamp': response_data.get('timestamp'),
                        'system_id': response_data.get('system_id')
                    }
                
                # Track active agents
                self.active_agents.add(agent_type)
                
                # Update PADS board member votes if this is a board decision request
                await self._update_board_votes(agent_type, response_data)
                
        except Exception as e:
            logger.error(f"Error handling decision response: {e}")
    
    async def _handle_agent_status(self, message: Message) -> None:
        """Handle agent status updates."""
        try:
            status_data = message.data
            agent_type = message.sender.value
            
            logger.debug(f"PADS received status update from {agent_type}")
            
            # Update agent tracking
            if status_data.get('connected', False):
                self.active_agents.add(agent_type)
            else:
                self.active_agents.discard(agent_type)
                
            # Update PADS board member reputation if available
            if hasattr(self.pads, 'reputation_scores') and agent_type in self.pads.reputation_scores:
                # Positive status update slightly improves reputation
                self.pads.reputation_scores[agent_type] = min(1.0, 
                    self.pads.reputation_scores[agent_type] + 0.01)
                
        except Exception as e:
            logger.error(f"Error handling agent status: {e}")
    
    async def _handle_market_update(self, message: Message) -> None:
        """Handle market updates from agents."""
        try:
            update_data = message.data
            agent_type = message.sender.value
            
            logger.debug(f"PADS received market update from {agent_type}")
            
            # Process different types of market updates
            if 'intention_signal' in update_data:
                # Quantum AMOS intention signal
                await self._process_intention_signal(agent_type, update_data)
            elif 'analysis' in update_data:
                # QUASAR market analysis
                await self._process_market_analysis(agent_type, update_data)
            elif 'beliefs' in update_data:
                # Agent belief updates
                await self._process_belief_update(agent_type, update_data)
                
        except Exception as e:
            logger.error(f"Error handling market update: {e}")
    
    async def _handle_risk_alert(self, message: Message) -> None:
        """Handle risk alerts from agents."""
        try:
            alert_data = message.data
            agent_type = message.sender.value
            alert_type = alert_data.get('alert_type', 'unknown')
            severity = alert_data.get('severity', 0.5)
            
            logger.warning(f"PADS received risk alert from {agent_type}: {alert_type} (severity: {severity})")
            
            # Escalate high-severity alerts
            if severity > 0.8:
                # Broadcast to all agents
                await self._broadcast_risk_alert(alert_data, exclude_sender=message.sender)
                
                # Adjust PADS decision making
                if hasattr(self.pads, 'board_state'):
                    self.pads.board_state['risk_appetite'] = max(0.1, 
                        self.pads.board_state.get('risk_appetite', 0.5) * 0.8)
                    logger.info(f"PADS reduced risk appetite due to high-severity alert")
                
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_system_command(self, message: Message) -> None:
        """Handle system commands."""
        try:
            command_data = message.data
            command = command_data.get('command')
            agent_type = message.sender.value
            
            logger.info(f"PADS received system command from {agent_type}: {command}")
            
            if command == 'request_decision':
                # Agent requesting a decision from PADS
                market_data = command_data.get('market_data', {})
                position_state = command_data.get('position_state', {})
                
                # Make PADS decision
                decision = self.pads.make_decision(market_data, {}, position_state)
                
                # Send response
                if decision:
                    response = Message(
                        message_type=MessageType.DECISION_RESPONSE,
                        sender=AgentType.PADS,
                        recipient=message.sender,
                        data={
                            'decision': {
                                'decision_type': decision.decision_type.name,
                                'confidence': decision.confidence,
                                'reasoning': decision.reasoning
                            },
                            'system_id': 'PADS_001',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        },
                        correlation_id=message.id,
                        priority=1
                    )
                    await self.messenger.send_message(response)
            
            elif command == 'status_request':
                # Send PADS status
                status = self._get_pads_status()
                response = Message(
                    message_type=MessageType.AGENT_STATUS,
                    sender=AgentType.PADS,
                    recipient=message.sender,
                    data=status,
                    correlation_id=message.id
                )
                await self.messenger.send_message(response)
                
        except Exception as e:
            logger.error(f"Error handling system command: {e}")
    
    async def _update_board_votes(self, agent_type: str, response_data: Dict[str, Any]) -> None:
        """Update PADS board member votes with agent decision."""
        try:
            if not hasattr(self.pads, 'board_members'):
                return
                
            # Map agent types to board members
            agent_board_map = {
                'qbmia': 'qar',
                'quasar': 'qstar', 
                'quantum_amos': 'antifragility'
            }
            
            board_member = agent_board_map.get(agent_type)
            if not board_member or board_member not in self.pads.board_members:
                return
                
            # Extract decision information
            decision = response_data.get('decision', {})
            confidence = response_data.get('confidence', 0.5)
            
            # Update board quantities (for LMSR)
            if hasattr(self.pads, 'board_quantities'):
                # Convert decision to market signal
                decision_type = decision.get('decision_type', 'HOLD')
                signal_value = self._decision_to_signal(decision_type)
                
                # Update board market quantities
                current_qty = self.pads.board_quantities.get(board_member, 0.0)
                weight = self.pads.board_members.get(board_member, 0.0)
                
                # Weighted update
                update = signal_value * confidence * weight
                self.pads.board_quantities[board_member] = current_qty + update
                
                logger.debug(f"Updated board vote for {board_member}: {signal_value} (confidence: {confidence})")
                
        except Exception as e:
            logger.error(f"Error updating board votes: {e}")
    
    def _decision_to_signal(self, decision_type: str) -> float:
        """Convert decision type to signal value (-1 to 1)."""
        signal_map = {
            'BUY': 1.0,
            'INCREASE': 0.5,
            'HOLD': 0.0,
            'DECREASE': -0.25,
            'HEDGE': -0.5,
            'SELL': -0.75,
            'EXIT': -1.0
        }
        return signal_map.get(decision_type.upper(), 0.0)
    
    async def _process_intention_signal(self, agent_type: str, update_data: Dict[str, Any]) -> None:
        """Process intention signal from Quantum AMOS."""
        try:
            intention = update_data.get('intention_signal', 0.0)
            agent_name = update_data.get('agent_name', agent_type)
            
            # Update PADS internal state with intention information
            # This could influence future decision making
            logger.debug(f"Processed intention signal from {agent_name}: {intention}")
            
        except Exception as e:
            logger.error(f"Error processing intention signal: {e}")
    
    async def _process_market_analysis(self, agent_type: str, update_data: Dict[str, Any]) -> None:
        """Process market analysis from QUASAR."""
        try:
            analysis = update_data.get('analysis', {})
            
            # Extract useful information for PADS decision making
            # This could update factor values or market state
            logger.debug(f"Processed market analysis from {agent_type}")
            
        except Exception as e:
            logger.error(f"Error processing market analysis: {e}")
    
    async def _process_belief_update(self, agent_type: str, update_data: Dict[str, Any]) -> None:
        """Process belief updates from agents."""
        try:
            beliefs = update_data.get('beliefs', {})
            agent_name = update_data.get('agent_name', agent_type)
            
            # Store agent beliefs for potential consensus building
            logger.debug(f"Processed belief update from {agent_name}")
            
        except Exception as e:
            logger.error(f"Error processing belief update: {e}")
    
    async def _broadcast_risk_alert(self, alert_data: Dict[str, Any], exclude_sender: Optional[AgentType] = None) -> None:
        """Broadcast risk alert to all active agents."""
        try:
            message = Message(
                message_type=MessageType.RISK_ALERT,
                sender=AgentType.PADS,
                recipient=None,  # Broadcast
                data=alert_data,
                priority=1
            )
            
            await self.messenger.send_message(message)
            logger.info("PADS broadcasted risk alert to all agents")
            
        except Exception as e:
            logger.error(f"Error broadcasting risk alert: {e}")
    
    async def _announce_pads_startup(self) -> None:
        """Announce PADS startup to all agents."""
        try:
            message = Message(
                message_type=MessageType.SYSTEM_COMMAND,
                sender=AgentType.PADS,
                recipient=None,  # Broadcast
                data={
                    'command': 'pads_startup',
                    'pads_config': {
                        'decision_styles': getattr(self.pads, 'decision_styles', []),
                        'board_members': getattr(self.pads, 'board_members', {}),
                        'confidence_thresholds': getattr(self.pads, 'confidence_thresholds', {})
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                priority=2
            )
            
            await self.messenger.send_message(message)
            logger.info("PADS announced startup to all agents")
            
        except Exception as e:
            logger.error(f"Error announcing PADS startup: {e}")
    
    async def _announce_pads_shutdown(self) -> None:
        """Announce PADS shutdown to all agents."""
        try:
            message = Message(
                message_type=MessageType.SYSTEM_COMMAND,
                sender=AgentType.PADS,
                recipient=None,  # Broadcast
                data={
                    'command': 'pads_shutdown',
                    'final_status': self._get_pads_status(),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                priority=1
            )
            
            await self.messenger.send_message(message)
            logger.info("PADS announced shutdown to all agents")
            
        except Exception as e:
            logger.error(f"Error announcing PADS shutdown: {e}")
    
    def _get_pads_status(self) -> Dict[str, Any]:
        """Get PADS system status."""
        try:
            status = {
                'system_type': 'PADS',
                'connected': self.connected,
                'use_real_messaging': self.use_real_messaging,
                'active_agents': list(self.active_agents),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Add PADS-specific status
            if hasattr(self.pads, 'panarchy_state'):
                status['panarchy_state'] = self.pads.panarchy_state.copy()
                
            if hasattr(self.pads, 'board_state'):
                status['board_state'] = self.pads.board_state.copy()
                
            if hasattr(self.pads, 'performance_metrics'):
                status['performance_metrics'] = self.pads.performance_metrics.copy()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting PADS status: {e}")
            return {
                'system_type': 'PADS',
                'connected': self.connected,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def request_agent_decisions(self, market_data: Dict[str, Any], 
                                    position_state: Optional[Dict[str, Any]] = None,
                                    agents: Optional[List[AgentType]] = None,
                                    timeout: float = 5.0) -> Dict[str, Any]:
        """Request decisions from specific agents."""
        if not self.connected:
            logger.error("PADS not connected to messaging system")
            return {}
            
        request_id = f"decision_req_{datetime.now(timezone.utc).timestamp()}"
        
        try:
            # Default to all active agents if none specified
            if agents is None:
                agents = [AgentType(agent) for agent in self.active_agents 
                         if agent in ['qbmia', 'quasar', 'quantum_amos']]
            
            # Send decision requests
            for agent in agents:
                message = Message(
                    message_type=MessageType.DECISION_REQUEST,
                    sender=AgentType.PADS,
                    recipient=agent,
                    data={
                        'market_data': market_data,
                        'position_state': position_state or {},
                        'request_id': request_id,
                        'timeout': timeout
                    },
                    correlation_id=request_id,
                    priority=1
                )
                
                await self.messenger.send_message(message)
                logger.debug(f"PADS sent decision request to {agent.value}")
            
            # Wait for responses
            await asyncio.sleep(min(timeout, 3.0))
            
            # Collect responses
            responses = self.agent_responses.get(request_id, {})
            
            # Clean up
            if request_id in self.agent_responses:
                del self.agent_responses[request_id]
            
            return responses
            
        except Exception as e:
            logger.error(f"Error requesting agent decisions: {e}")
            return {}
    
    async def broadcast_phase_transition(self, old_phase: str, new_phase: str, 
                                       phase_data: Optional[Dict[str, Any]] = None) -> bool:
        """Broadcast market phase transition to all agents."""
        if not self.connected:
            return False
            
        try:
            message = Message(
                message_type=MessageType.PHASE_TRANSITION,
                sender=AgentType.PADS,
                recipient=None,  # Broadcast
                data={
                    'old_phase': old_phase,
                    'new_phase': new_phase,
                    'phase_data': phase_data or {},
                    'transition_time': datetime.now(timezone.utc).isoformat()
                },
                priority=1
            )
            
            await self.messenger.send_message(message)
            logger.info(f"PADS broadcasted phase transition: {old_phase} -> {new_phase}")
            return True
            
        except Exception as e:
            logger.error(f"Error broadcasting phase transition: {e}")
            return False

# Integration helper function
def integrate_pads_messaging(pads_instance, config: Optional[Dict[str, Any]] = None) -> PADSMessagingIntegration:
    """
    Integrate PADS with the unified messaging system.
    
    Args:
        pads_instance: PanarchyAdaptiveDecisionSystem instance
        config: Messaging configuration
        
    Returns:
        Configured messaging integration
    """
    integration = PADSMessagingIntegration(pads_instance, config)
    
    # Add messaging methods to PADS instance
    pads_instance.messaging_integration = integration
    pads_instance.request_agent_decisions = integration.request_agent_decisions
    pads_instance.broadcast_phase_transition = integration.broadcast_phase_transition
    
    logger.info("PADS messaging integration completed")
    return integration