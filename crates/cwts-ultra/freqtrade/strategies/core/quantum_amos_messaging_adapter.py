#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum AMOS messaging adapter for real-time communication with PADS and other agents.

This module provides the communication layer for Quantum AMOS agents to integrate
with the unified messaging system for multi-agent coordination and decision making.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import numpy as np

try:
    from unified_messaging import (
        UnifiedMessenger, Message, MessageType, AgentType,
        create_quantum_amos_messenger
    )
    UNIFIED_MESSAGING_AVAILABLE = True
except ImportError:
    UNIFIED_MESSAGING_AVAILABLE = False

logger = logging.getLogger("QuantumAMOSMessaging")

class QuantumAMOSMessagingAdapter:
    """
    Messaging adapter for Quantum AMOS agent integration with PADS.
    
    Handles real-time communication for intention signals, belief updates,
    and network coordination between Quantum AMOS agents and other trading systems.
    """
    
    def __init__(self, amos_agent_or_network, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Quantum AMOS messaging adapter.
        
        Args:
            amos_agent_or_network: QuantumAmosAgent or QuantumAmosNetwork instance
            config: Configuration for messaging
        """
        self.amos = amos_agent_or_network
        self.config = config or {}
        
        # Determine if we have a single agent or network
        self.is_network = hasattr(amos_agent_or_network, 'agents')
        self.agent_name = getattr(amos_agent_or_network, 'name', 'QuantumAMOS_001')
        
        # Communication state
        self.connected = False
        self.use_real_messaging = UNIFIED_MESSAGING_AVAILABLE and self.config.get('use_real_messaging', True)
        self.messenger: Optional[UnifiedMessenger] = None
        
        if self.use_real_messaging:
            # Initialize unified messenger for Quantum AMOS
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            zmq_ports = self.config.get('zmq_ports', {})
            self.messenger = create_quantum_amos_messenger(
                redis_url=redis_url,
                zmq_connect_ports=zmq_ports
            )
            logger.info("Quantum AMOS messaging adapter initialized with real messaging")
        else:
            logger.warning("Quantum AMOS messaging adapter using simulated messaging")
            
        # Decision coordination state
        self.network_consensus = {}
        self.agent_beliefs = {}
        
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
                    logger.info("Quantum AMOS connected to unified messaging system")
                    return True
                else:
                    logger.error("Failed to connect Quantum AMOS to unified messaging")
                    self.use_real_messaging = False
                    
            except Exception as e:
                logger.error(f"Error connecting Quantum AMOS to messaging: {e}")
                self.use_real_messaging = False
        
        # Mark as connected even if using simulated messaging
        self.connected = True
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from the messaging system."""
        if self.use_real_messaging and self.messenger:
            await self.messenger.disconnect()
            logger.info("Quantum AMOS disconnected from unified messaging")
            
        self.connected = False
        
    def _register_message_handlers(self) -> None:
        """Register message handlers for Quantum AMOS."""
        if not self.messenger:
            return
            
        # Register handlers for different message types
        self.messenger.register_handler(MessageType.DECISION_REQUEST, self._handle_decision_request)
        self.messenger.register_handler(MessageType.MARKET_UPDATE, self._handle_market_update)
        self.messenger.register_handler(MessageType.PHASE_TRANSITION, self._handle_phase_transition)
        self.messenger.register_handler(MessageType.RISK_ALERT, self._handle_risk_alert)
        self.messenger.register_handler(MessageType.PERFORMANCE_FEEDBACK, self._handle_performance_feedback)
        self.messenger.register_handler(MessageType.SYSTEM_COMMAND, self._handle_system_command)
        
        logger.info("Registered Quantum AMOS message handlers")
    
    async def _handle_decision_request(self, message: Message) -> None:
        """Handle decision request from PADS or other agents."""
        try:
            request_data = message.data
            market_data = request_data.get('market_data', {})
            expected_outcome = request_data.get('expected_outcome', 0.0)
            probability = request_data.get('probability', 0.5)
            
            logger.debug(f"Quantum AMOS received decision request: {message.id}")
            
            # Get decision from Quantum AMOS
            if self.is_network:
                # Network decision
                decision = self.amos.network_decide(market_data, expected_outcome, probability)
                decision_data = {
                    'decision_type': decision.name,
                    'intention_signals': [agent.compute_intention(market_data, expected_outcome, probability) 
                                        for agent in self.amos.agents],
                    'network_size': len(self.amos.agents),
                    'market_phase': getattr(self.amos, 'market_phase', 'unknown'),
                    'agent_weights': getattr(self.amos, 'agent_weights', {})
                }
            else:
                # Single agent decision
                decision = self.amos.decide(market_data, expected_outcome, probability)
                intention = self.amos.compute_intention(market_data, expected_outcome, probability)
                decision_data = {
                    'decision_type': decision.name,
                    'intention_signal': intention,
                    'agent_name': self.agent_name,
                    'beliefs': self.amos.compute_beliefs(market_data),
                    'desire': getattr(self.amos, 'desire', 0.0)
                }
            
            # Send response back
            response = Message(
                message_type=MessageType.DECISION_RESPONSE,
                sender=AgentType.QUANTUM_AMOS,
                recipient=message.sender,
                data={
                    'decision': decision_data,
                    'system_id': self.agent_name,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'confidence': 0.8  # Quantum AMOS typically has high conviction
                },
                correlation_id=message.id,
                priority=1
            )
            
            await self.messenger.send_message(response)
            logger.info(f"Quantum AMOS sent decision response: {decision.name}")
                
        except Exception as e:
            logger.error(f"Error in Quantum AMOS decision request handler: {e}")
    
    async def _handle_market_update(self, message: Message) -> None:
        """Handle market data updates."""
        try:
            update_data = message.data
            symbol = update_data.get('symbol', 'UNKNOWN')
            
            logger.debug(f"Quantum AMOS received market update for {symbol}")
            
            # Update beliefs if we have new market data
            if self.is_network:
                # Update all agents in network
                for agent in self.amos.agents:
                    if hasattr(agent, 'update_beliefs'):
                        await agent.update_beliefs(update_data)
            else:
                # Update single agent
                if hasattr(self.amos, 'update_beliefs'):
                    await self.amos.update_beliefs(update_data)
                    
        except Exception as e:
            logger.error(f"Error handling market update: {e}")
    
    async def _handle_phase_transition(self, message: Message) -> None:
        """Handle market phase transition notifications."""
        try:
            phase_data = message.data
            new_phase = phase_data.get('new_phase')
            old_phase = phase_data.get('old_phase')
            
            logger.info(f"Quantum AMOS received phase transition: {old_phase} -> {new_phase}")
            
            # Update market phase in agents
            if self.is_network:
                # Update network market phase
                if hasattr(self.amos, 'market_phase'):
                    try:
                        # Map string to MarketPhase enum if available
                        from quantum_amos import MarketPhase
                        if new_phase in [phase.value for phase in MarketPhase]:
                            self.amos.market_phase = MarketPhase(new_phase)
                            # Update all agents
                            for agent in self.amos.agents:
                                agent.market_phase = MarketPhase(new_phase)
                            logger.debug(f"Updated network market phase to {new_phase}")
                    except ImportError:
                        logger.warning("Could not import MarketPhase enum")
            else:
                # Update single agent
                if hasattr(self.amos, 'market_phase'):
                    try:
                        from quantum_amos import MarketPhase
                        if new_phase in [phase.value for phase in MarketPhase]:
                            self.amos.market_phase = MarketPhase(new_phase)
                            logger.debug(f"Updated agent market phase to {new_phase}")
                    except ImportError:
                        logger.warning("Could not import MarketPhase enum")
                        
        except Exception as e:
            logger.error(f"Error handling phase transition: {e}")
    
    async def _handle_risk_alert(self, message: Message) -> None:
        """Handle risk alerts from PADS or other agents."""
        try:
            alert_data = message.data
            alert_type = alert_data.get('alert_type')
            severity = alert_data.get('severity', 0.5)
            
            logger.warning(f"Quantum AMOS received risk alert: {alert_type} (severity: {severity})")
            
            # Adjust agent behavior based on risk level
            if severity > 0.8:
                # High risk - reduce desire (more conservative)
                if self.is_network:
                    for agent in self.amos.agents:
                        if hasattr(agent, 'desire'):
                            agent.desire *= 0.8  # Reduce desire temporarily
                else:
                    if hasattr(self.amos, 'desire'):
                        self.amos.desire *= 0.8
                        
                logger.info("Quantum AMOS reduced desire due to high risk alert")
                    
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_performance_feedback(self, message: Message) -> None:
        """Handle performance feedback for cognitive reappraisal."""
        try:
            feedback_data = message.data
            predicted_return = feedback_data.get('predicted_return', 0.0)
            actual_return = feedback_data.get('actual_return', 0.0)
            market_data = feedback_data.get('market_data', {})
            
            logger.info(f"Quantum AMOS received performance feedback: predicted={predicted_return}, actual={actual_return}")
            
            # Apply cognitive reappraisal
            if self.is_network:
                # Update all agents in network
                if hasattr(self.amos, 'update_agents'):
                    self.amos.update_agents(market_data, predicted_return, actual_return)
                else:
                    # Update agents individually
                    for agent in self.amos.agents:
                        if hasattr(agent, 'cognitive_reappraisal'):
                            agent.cognitive_reappraisal(market_data, predicted_return, actual_return)
            else:
                # Update single agent
                if hasattr(self.amos, 'cognitive_reappraisal'):
                    self.amos.cognitive_reappraisal(market_data, predicted_return, actual_return)
                    
            logger.debug("Applied cognitive reappraisal based on performance feedback")
                    
        except Exception as e:
            logger.error(f"Error handling performance feedback: {e}")
    
    async def _handle_system_command(self, message: Message) -> None:
        """Handle system commands."""
        try:
            command_data = message.data
            command = command_data.get('command')
            
            logger.info(f"Quantum AMOS received system command: {command}")
            
            if command == 'status_request':
                # Send status response
                status = self._get_system_status()
                response = Message(
                    message_type=MessageType.AGENT_STATUS,
                    sender=AgentType.QUANTUM_AMOS,
                    recipient=message.sender,
                    data=status,
                    correlation_id=message.id
                )
                await self.messenger.send_message(response)
                
            elif command == 'reset_weights':
                logger.info("Quantum AMOS received reset weights command")
                await self._reset_agent_weights()
                
            elif command == 'shutdown':
                logger.info("Quantum AMOS received shutdown command")
                await self.disconnect()
                
        except Exception as e:
            logger.error(f"Error handling system command: {e}")
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get Quantum AMOS system status."""
        try:
            status = {
                'agent_type': 'QUANTUM_AMOS',
                'agent_name': self.agent_name,
                'is_network': self.is_network,
                'connected': self.connected,
                'use_real_messaging': self.use_real_messaging,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if self.is_network:
                # Network status
                status['network_status'] = {
                    'num_agents': len(self.amos.agents),
                    'agent_names': [agent.name for agent in self.amos.agents],
                    'agent_weights': getattr(self.amos, 'agent_weights', {}),
                    'market_phase': getattr(self.amos, 'market_phase', 'unknown'),
                    'decision_history_length': len(getattr(self.amos, 'decision_history', []))
                }
            else:
                # Single agent status
                status['agent_status'] = {
                    'name': self.amos.name,
                    'desire': getattr(self.amos, 'desire', 0.0),
                    'weights': getattr(self.amos, 'weights', {}),
                    'market_phase': getattr(self.amos, 'market_phase', 'unknown')
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting Quantum AMOS status: {e}")
            return {
                'agent_type': 'QUANTUM_AMOS',
                'connected': self.connected,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _reset_agent_weights(self) -> None:
        """Reset agent weights to defaults."""
        try:
            if self.is_network:
                # Reset network agent weights
                if hasattr(self.amos, 'agent_weights'):
                    num_agents = len(self.amos.agents)
                    default_weight = 1.0 / num_agents
                    self.amos.agent_weights = {agent.name: default_weight for agent in self.amos.agents}
                    logger.info("Reset network agent weights to equal distribution")
            else:
                # Reset single agent weights
                if hasattr(self.amos, 'weights'):
                    try:
                        from quantum_amos import StandardFactors
                        default_weights = StandardFactors.get_default_weights()
                        self.amos.weights = default_weights
                        logger.info("Reset agent factor weights to defaults")
                    except ImportError:
                        logger.warning("Could not import StandardFactors for weight reset")
                        
        except Exception as e:
            logger.error(f"Error resetting agent weights: {e}")
    
    async def send_intention_signal(self, intention: float, market_data: Dict[str, Any], 
                                  correlation_id: Optional[str] = None) -> bool:
        """Send intention signal to PADS or other agents."""
        if not self.connected:
            logger.error("Quantum AMOS not connected to messaging system")
            return False
            
        try:
            message = Message(
                message_type=MessageType.MARKET_UPDATE,
                sender=AgentType.QUANTUM_AMOS,
                recipient=AgentType.PADS,
                data={
                    'intention_signal': intention,
                    'market_data': market_data,
                    'agent_name': self.agent_name,
                    'system_id': self.agent_name,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id,
                priority=2
            )
            
            if self.use_real_messaging and self.messenger:
                success = await self.messenger.send_message(message)
                if success:
                    logger.debug("Quantum AMOS sent intention signal via real messaging")
                    return True
                else:
                    logger.error("Failed to send intention signal via real messaging")
            
            # Fallback: simulated sending
            logger.debug("Quantum AMOS sent intention signal (simulated)")
            return True
            
        except Exception as e:
            logger.error(f"Error sending intention signal: {e}")
            return False
    
    async def broadcast_beliefs(self, beliefs: Dict[str, float], 
                              correlation_id: Optional[str] = None) -> bool:
        """Broadcast belief state to other agents."""
        if not self.connected:
            return False
            
        try:
            message = Message(
                message_type=MessageType.MARKET_UPDATE,
                sender=AgentType.QUANTUM_AMOS,
                recipient=None,  # Broadcast
                data={
                    'beliefs': beliefs,
                    'agent_name': self.agent_name,
                    'belief_type': 'market_factors',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id,
                priority=3
            )
            
            if self.use_real_messaging and self.messenger:
                success = await self.messenger.send_message(message)
                return success
            
            return True  # Simulated success
            
        except Exception as e:
            logger.error(f"Error broadcasting beliefs: {e}")
            return False
    
    async def request_network_consensus(self, market_data: Dict[str, Any],
                                      timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Request consensus decision from other Quantum AMOS agents."""
        if not self.connected or not self.use_real_messaging:
            return None
            
        try:
            # Send consensus request
            request_id = f"consensus_{datetime.now(timezone.utc).timestamp()}"
            message = Message(
                message_type=MessageType.DECISION_REQUEST,
                sender=AgentType.QUANTUM_AMOS,
                recipient=None,  # Broadcast to all AMOS agents
                data={
                    'request_type': 'network_consensus',
                    'market_data': market_data,
                    'agent_name': self.agent_name,
                    'request_id': request_id,
                    'timeout': timeout
                },
                priority=1
            )
            
            await self.messenger.send_message(message)
            
            # Wait for responses (simplified - in practice you'd collect multiple responses)
            await asyncio.sleep(min(timeout, 2.0))
            
            # Return collected consensus (placeholder implementation)
            return {
                'consensus_available': False,
                'message': 'Network consensus not fully implemented'
            }
            
        except Exception as e:
            logger.error(f"Error requesting network consensus: {e}")
            return None

# Integration helper function
def integrate_quantum_amos_messaging(amos_instance, config: Optional[Dict[str, Any]] = None) -> QuantumAMOSMessagingAdapter:
    """
    Integrate Quantum AMOS with the unified messaging system.
    
    Args:
        amos_instance: QuantumAmosAgent or QuantumAmosNetwork instance
        config: Messaging configuration
        
    Returns:
        Configured messaging adapter
    """
    adapter = QuantumAMOSMessagingAdapter(amos_instance, config)
    
    # Add messaging methods to AMOS instance
    amos_instance.messaging_adapter = adapter
    amos_instance.send_intention_signal = adapter.send_intention_signal
    amos_instance.broadcast_beliefs = adapter.broadcast_beliefs
    amos_instance.request_network_consensus = adapter.request_network_consensus
    
    logger.info("Quantum AMOS messaging integration completed")
    return adapter