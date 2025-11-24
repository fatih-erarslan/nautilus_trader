#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUASAR messaging adapter for real-time communication with PADS and other agents.

This module provides the communication layer for QUASAR to integrate with
the unified messaging system and receive/send trading decisions and market updates.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
import numpy as np

try:
    from unified_messaging import (
        UnifiedMessenger, Message, MessageType, AgentType,
        create_quasar_messenger
    )
    UNIFIED_MESSAGING_AVAILABLE = True
except ImportError:
    UNIFIED_MESSAGING_AVAILABLE = False

logger = logging.getLogger("QUASARMessaging")

class QUASARMessagingAdapter:
    """
    Messaging adapter for QUASAR system integration with PADS.
    
    Handles real-time communication for decision requests, market updates,
    and system coordination between QUASAR and other trading agents.
    """
    
    def __init__(self, quasar_instance, config: Optional[Dict[str, Any]] = None):
        """
        Initialize QUASAR messaging adapter.
        
        Args:
            quasar_instance: QUASAR system instance
            config: Configuration for messaging
        """
        self.quasar = quasar_instance
        self.config = config or {}
        
        # Communication state
        self.connected = False
        self.use_real_messaging = UNIFIED_MESSAGING_AVAILABLE and self.config.get('use_real_messaging', True)
        self.messenger: Optional[UnifiedMessenger] = None
        
        if self.use_real_messaging:
            # Initialize unified messenger for QUASAR
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            zmq_ports = self.config.get('zmq_ports', {})
            self.messenger = create_quasar_messenger(
                redis_url=redis_url,
                zmq_connect_ports=zmq_ports
            )
            logger.info("QUASAR messaging adapter initialized with real messaging")
        else:
            logger.warning("QUASAR messaging adapter using simulated messaging")
            
        # Message response tracking
        self.pending_requests = {}
        
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
                    logger.info("QUASAR connected to unified messaging system")
                    return True
                else:
                    logger.error("Failed to connect QUASAR to unified messaging")
                    self.use_real_messaging = False
                    
            except Exception as e:
                logger.error(f"Error connecting QUASAR to messaging: {e}")
                self.use_real_messaging = False
        
        # Mark as connected even if using simulated messaging
        self.connected = True
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from the messaging system."""
        if self.use_real_messaging and self.messenger:
            await self.messenger.disconnect()
            logger.info("QUASAR disconnected from unified messaging")
            
        self.connected = False
        
    def _register_message_handlers(self) -> None:
        """Register message handlers for QUASAR."""
        if not self.messenger:
            return
            
        # Register handlers for different message types
        self.messenger.register_handler(MessageType.DECISION_REQUEST, self._handle_decision_request)
        self.messenger.register_handler(MessageType.MARKET_UPDATE, self._handle_market_update)
        self.messenger.register_handler(MessageType.PHASE_TRANSITION, self._handle_phase_transition)
        self.messenger.register_handler(MessageType.RISK_ALERT, self._handle_risk_alert)
        self.messenger.register_handler(MessageType.PERFORMANCE_FEEDBACK, self._handle_performance_feedback)
        self.messenger.register_handler(MessageType.SYSTEM_COMMAND, self._handle_system_command)
        
        logger.info("Registered QUASAR message handlers")
    
    async def _handle_decision_request(self, message: Message) -> None:
        """Handle decision request from PADS or other agents."""
        try:
            request_data = message.data
            market_data = request_data.get('market_data', {})
            position_state = request_data.get('position_state', {})
            
            logger.debug(f"QUASAR received decision request: {message.id}")
            
            # Convert market data to DataFrame format if needed
            dataframe = self._convert_to_dataframe(market_data)
            
            if dataframe is not None:
                # Make decision using QUASAR
                decision_result = self.quasar.make_decision(
                    dataframe=dataframe,
                    position_state=position_state
                )
                
                # Send response back
                response = Message(
                    message_type=MessageType.DECISION_RESPONSE,
                    sender=AgentType.QUASAR,
                    recipient=message.sender,
                    data={
                        'decision': decision_result,
                        'system_id': 'QUASAR_001',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'confidence': decision_result.get('confidence', 0.5) if decision_result else 0.0
                    },
                    correlation_id=message.id,
                    priority=1
                )
                
                await self.messenger.send_message(response)
                logger.info(f"QUASAR sent decision response: {decision_result.get('action_name', 'UNKNOWN') if decision_result else 'NONE'}")
            else:
                logger.warning("QUASAR could not process decision request - invalid market data")
                
        except Exception as e:
            logger.error(f"Error in QUASAR decision request handler: {e}")
    
    async def _handle_market_update(self, message: Message) -> None:
        """Handle market data updates."""
        try:
            update_data = message.data
            symbol = update_data.get('symbol', 'UNKNOWN')
            
            logger.debug(f"QUASAR received market update for {symbol}")
            
            # Update internal market state if QUASAR has this capability
            if hasattr(self.quasar, 'update_market_data'):
                await self.quasar.update_market_data(update_data)
                
        except Exception as e:
            logger.error(f"Error handling market update: {e}")
    
    async def _handle_phase_transition(self, message: Message) -> None:
        """Handle market phase transition notifications."""
        try:
            phase_data = message.data
            new_phase = phase_data.get('new_phase')
            old_phase = phase_data.get('old_phase')
            
            logger.info(f"QUASAR received phase transition: {old_phase} -> {new_phase}")
            
            # Update QUASAR's market regime state if applicable
            if hasattr(self.quasar, 'state') and hasattr(self.quasar.state, 'market_regime'):
                # Map phase string to MarketPhase enum if needed
                try:
                    from qar import MarketPhase
                    if new_phase in [phase.value for phase in MarketPhase]:
                        phase_enum = MarketPhase(new_phase)
                        self.quasar.state.market_regime.phase = phase_enum
                        logger.debug(f"Updated QUASAR market regime to {new_phase}")
                except ImportError:
                    logger.warning("Could not import MarketPhase enum for phase update")
                    
        except Exception as e:
            logger.error(f"Error handling phase transition: {e}")
    
    async def _handle_risk_alert(self, message: Message) -> None:
        """Handle risk alerts from PADS or other agents."""
        try:
            alert_data = message.data
            alert_type = alert_data.get('alert_type')
            severity = alert_data.get('severity', 0.5)
            
            logger.warning(f"QUASAR received risk alert: {alert_type} (severity: {severity})")
            
            # Adjust QUASAR behavior based on risk level
            if severity > 0.8:
                # High risk - be more conservative
                if hasattr(self.quasar, 'config'):
                    # Temporarily increase decision threshold
                    original_threshold = self.quasar.config.decision_threshold
                    self.quasar.config.decision_threshold = min(0.9, original_threshold * 1.2)
                    logger.info(f"QUASAR increased decision threshold due to high risk: {original_threshold} -> {self.quasar.config.decision_threshold}")
                    
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_performance_feedback(self, message: Message) -> None:
        """Handle performance feedback for learning."""
        try:
            feedback_data = message.data
            decision_id = feedback_data.get('decision_id')
            outcome = feedback_data.get('outcome')
            profit_loss = feedback_data.get('profit_loss', 0.0)
            
            logger.info(f"QUASAR received performance feedback: {outcome} (P/L: {profit_loss})")
            
            # Provide feedback to QUASAR if it supports learning
            if hasattr(self.quasar, 'provide_feedback'):
                success = self.quasar.provide_feedback(
                    decision_id=decision_id,
                    outcome=outcome,
                    profit_loss=profit_loss
                )
                
                if success:
                    logger.debug("Successfully provided feedback to QUASAR")
                else:
                    logger.warning("Failed to provide feedback to QUASAR")
                    
        except Exception as e:
            logger.error(f"Error handling performance feedback: {e}")
    
    async def _handle_system_command(self, message: Message) -> None:
        """Handle system commands."""
        try:
            command_data = message.data
            command = command_data.get('command')
            
            logger.info(f"QUASAR received system command: {command}")
            
            if command == 'status_request':
                # Send status response
                status = self._get_system_status()
                response = Message(
                    message_type=MessageType.AGENT_STATUS,
                    sender=AgentType.QUASAR,
                    recipient=message.sender,
                    data=status,
                    correlation_id=message.id
                )
                await self.messenger.send_message(response)
                
            elif command == 'reset':
                logger.info("QUASAR received reset command")
                await self._reset_system()
                
            elif command == 'shutdown':
                logger.info("QUASAR received shutdown command")
                await self.disconnect()
                
        except Exception as e:
            logger.error(f"Error handling system command: {e}")
    
    def _convert_to_dataframe(self, market_data: Dict[str, Any]):
        """Convert market data to DataFrame format for QUASAR."""
        try:
            import pandas as pd
            
            # Check if market_data contains DataFrame-compatible data
            if 'dataframe' in market_data:
                return market_data['dataframe']
            
            # Try to construct DataFrame from raw data
            if 'prices' in market_data and isinstance(market_data['prices'], list):
                prices = market_data['prices']
                df_data = {}
                
                # Extract OHLCV data
                for i, price_data in enumerate(prices):
                    if isinstance(price_data, dict):
                        for key in ['open', 'high', 'low', 'close', 'volume']:
                            if key in price_data:
                                if key not in df_data:
                                    df_data[key] = []
                                df_data[key].append(price_data[key])
                
                if df_data:
                    return pd.DataFrame(df_data)
            
            # If we have individual price points, create a simple DataFrame
            if 'close' in market_data:
                return pd.DataFrame({
                    'close': [market_data['close']],
                    'volume': [market_data.get('volume', 0)]
                })
                
            return None
            
        except Exception as e:
            logger.error(f"Error converting market data to DataFrame: {e}")
            return None
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get QUASAR system status."""
        try:
            status = {
                'agent_type': 'QUASAR',
                'connected': self.connected,
                'use_real_messaging': self.use_real_messaging,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Add QUASAR-specific status
            if hasattr(self.quasar, 'state'):
                status['quasar_state'] = {
                    'is_initialized': getattr(self.quasar.state, 'is_initialized', False),
                    'is_trained': getattr(self.quasar.state, 'is_trained', False),
                    'qstar_ready': getattr(self.quasar.state, 'qstar_ready', False),
                    'qar_ready': getattr(self.quasar.state, 'qar_ready', False)
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting QUASAR status: {e}")
            return {
                'agent_type': 'QUASAR',
                'connected': self.connected,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _reset_system(self) -> None:
        """Reset QUASAR system state."""
        try:
            # Reset any relevant QUASAR state
            if hasattr(self.quasar, 'decision_history'):
                self.quasar.decision_history.clear()
                
            # Clear pending requests
            self.pending_requests.clear()
            
            logger.info("QUASAR system reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting QUASAR system: {e}")
    
    async def send_decision_to_pads(self, decision: Dict[str, Any], correlation_id: Optional[str] = None) -> bool:
        """Send a decision to PADS."""
        if not self.connected:
            logger.error("QUASAR not connected to messaging system")
            return False
            
        try:
            message = Message(
                message_type=MessageType.DECISION_RESPONSE,
                sender=AgentType.QUASAR,
                recipient=AgentType.PADS,
                data={
                    'decision': decision,
                    'system_id': 'QUASAR_001',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id,
                priority=1
            )
            
            if self.use_real_messaging and self.messenger:
                success = await self.messenger.send_message(message)
                if success:
                    logger.info("QUASAR sent decision to PADS via real messaging")
                    return True
                else:
                    logger.error("Failed to send decision via real messaging")
            
            # Fallback: simulated sending
            logger.info("QUASAR sent decision to PADS (simulated)")
            return True
            
        except Exception as e:
            logger.error(f"Error sending decision to PADS: {e}")
            return False
    
    async def broadcast_market_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Broadcast market analysis to interested agents."""
        if not self.connected:
            return False
            
        try:
            message = Message(
                message_type=MessageType.MARKET_UPDATE,
                sender=AgentType.QUASAR,
                recipient=None,  # Broadcast
                data={
                    'analysis': analysis,
                    'system_id': 'QUASAR_001',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                priority=2
            )
            
            if self.use_real_messaging and self.messenger:
                success = await self.messenger.send_message(message)
                return success
            
            return True  # Simulated success
            
        except Exception as e:
            logger.error(f"Error broadcasting market analysis: {e}")
            return False

# Integration helper function
def integrate_quasar_messaging(quasar_instance, config: Optional[Dict[str, Any]] = None) -> QUASARMessagingAdapter:
    """
    Integrate QUASAR with the unified messaging system.
    
    Args:
        quasar_instance: QUASAR system instance
        config: Messaging configuration
        
    Returns:
        Configured messaging adapter
    """
    adapter = QUASARMessagingAdapter(quasar_instance, config)
    
    # Add messaging methods to QUASAR instance
    quasar_instance.messaging_adapter = adapter
    quasar_instance.send_decision_to_pads = adapter.send_decision_to_pads
    quasar_instance.broadcast_analysis = adapter.broadcast_market_analysis
    
    logger.info("QUASAR messaging integration completed")
    return adapter