#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified messaging protocol for quantum-biological trading system integration.

This module provides Redis pub/sub and ZeroMQ messaging capabilities for
seamless communication between PADS, QBMIA, QUASAR, and Quantum AMOS agents.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor

# Redis for pub/sub messaging
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ZeroMQ for high-performance messaging
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

logger = logging.getLogger("UnifiedMessaging")

class MessageType(Enum):
    """Types of messages in the system."""
    DECISION_REQUEST = "decision_request"
    DECISION_RESPONSE = "decision_response"
    PHASE_TRANSITION = "phase_transition"
    RISK_ALERT = "risk_alert"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    MARKET_UPDATE = "market_update"
    AGENT_STATUS = "agent_status"
    SYSTEM_COMMAND = "system_command"

class AgentType(Enum):
    """Types of agents in the system."""
    PADS = "pads"
    QBMIA = "qbmia"
    QUASAR = "quasar"
    QUANTUM_AMOS = "quantum_amos"
    PREDICTION = "prediction"
    CDFA = "cdfa"
    PAIRLIST = "pairlist"

@dataclass
class Message:
    """Standard message format for inter-agent communication."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    message_type: MessageType = MessageType.SYSTEM_COMMAND
    sender: AgentType = AgentType.PADS
    recipient: Optional[AgentType] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    ttl: Optional[float] = None  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'id': self.id,
            'message_type': self.message_type.value,
            'sender': self.sender.value,
            'recipient': self.recipient.value if self.recipient else None,
            'data': self.data,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'priority': self.priority,
            'ttl': self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            id=data.get('id', uuid.uuid4().hex),
            message_type=MessageType(data.get('message_type', 'system_command')),
            sender=AgentType(data.get('sender', 'pads')),
            recipient=AgentType(data['recipient']) if data.get('recipient') else None,
            data=data.get('data', {}),
            timestamp=data.get('timestamp', time.time()),
            correlation_id=data.get('correlation_id'),
            priority=data.get('priority', 1),
            ttl=data.get('ttl')
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

class RedisMessenger:
    """Redis-based pub/sub messaging for the quantum trading system."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 agent_type: AgentType = AgentType.PADS):
        self.redis_url = redis_url
        self.agent_type = agent_type
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.channels = {
            'decisions': f'quantum_trading.decisions',
            'phases': f'quantum_trading.phases',
            'risks': f'quantum_trading.risks',
            'feedback': f'quantum_trading.feedback',
            'status': f'quantum_trading.status',
            'commands': f'quantum_trading.commands',
            'agent_specific': f'quantum_trading.{agent_type.value}'
        }
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.error("Redis not available for messaging")
            return False
            
        try:
            self.redis_client = redis.from_url(self.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            
            # Create pubsub client
            self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to relevant channels
            await self.pubsub.subscribe(
                self.channels['decisions'],
                self.channels['phases'],
                self.channels['risks'],
                self.channels['feedback'],
                self.channels['commands'],
                self.channels['agent_specific']
            )
            
            self.connected = True
            logger.info(f"Redis messenger connected for {self.agent_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        try:
            if self.pubsub:
                await self.pubsub.unsubscribe()
                await self.pubsub.close()
            
            if self.redis_client:
                await self.redis_client.close()
                
            self.connected = False
            logger.info(f"Redis messenger disconnected for {self.agent_type.value}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type.value}")
    
    async def publish_message(self, message: Message) -> bool:
        """Publish a message to the appropriate channel."""
        if not self.connected or not self.redis_client:
            logger.error("Not connected to Redis")
            return False
            
        try:
            # Determine channel based on message type
            channel = self._get_channel_for_message_type(message.message_type)
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Publish
            result = await self.redis_client.publish(channel, message_data)
            
            logger.debug(f"Published message {message.id} to {channel}, subscribers: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    async def listen_for_messages(self) -> None:
        """Listen for incoming messages and handle them."""
        if not self.connected or not self.pubsub:
            logger.error("Not connected to Redis pubsub")
            return
            
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    await self._handle_received_message(message)
                    
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")
    
    async def _handle_received_message(self, raw_message: Dict[str, Any]) -> None:
        """Handle a received message."""
        try:
            # Parse message
            message_data = json.loads(raw_message['data'])
            message = Message.from_dict(message_data)
            
            # Check if message is for us or is broadcast
            if message.recipient and message.recipient != self.agent_type:
                return  # Not for us
                
            # Check if message has expired
            if message.is_expired():
                logger.warning(f"Discarding expired message {message.id}")
                return
            
            # Handle message
            handler = self.message_handlers.get(message.message_type)
            if handler:
                try:
                    # Call handler (may be sync or async)
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler for {message.message_type.value}: {e}")
            else:
                logger.warning(f"No handler registered for message type {message.message_type.value}")
                
        except Exception as e:
            logger.error(f"Error handling received message: {e}")
    
    def _get_channel_for_message_type(self, message_type: MessageType) -> str:
        """Get the appropriate channel for a message type."""
        channel_map = {
            MessageType.DECISION_REQUEST: self.channels['decisions'],
            MessageType.DECISION_RESPONSE: self.channels['decisions'],
            MessageType.PHASE_TRANSITION: self.channels['phases'],
            MessageType.RISK_ALERT: self.channels['risks'],
            MessageType.PERFORMANCE_FEEDBACK: self.channels['feedback'],
            MessageType.AGENT_STATUS: self.channels['status'],
            MessageType.SYSTEM_COMMAND: self.channels['commands']
        }
        return channel_map.get(message_type, self.channels['commands'])

class ZeroMQMessenger:
    """ZeroMQ-based high-performance messaging for critical trading decisions."""
    
    def __init__(self, 
                 agent_type: AgentType = AgentType.PADS,
                 bind_port: Optional[int] = None,
                 connect_ports: Optional[Dict[AgentType, int]] = None):
        self.agent_type = agent_type
        self.context: Optional[zmq.asyncio.Context] = None
        self.router_socket: Optional[zmq.asyncio.Socket] = None
        self.dealer_sockets: Dict[AgentType, zmq.asyncio.Socket] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.connected = False
        
        # Default ports for each agent type
        self.default_ports = {
            AgentType.PADS: 9090,
            AgentType.QBMIA: 9091,
            AgentType.QUASAR: 9092,
            AgentType.QUANTUM_AMOS: 9093,
            AgentType.PREDICTION: 9094,
            AgentType.CDFA: 9095,
            AgentType.PAIRLIST: 9096
        }
        
        self.bind_port = bind_port or self.default_ports[agent_type]
        self.connect_ports = connect_ports or {
            agent: port for agent, port in self.default_ports.items() 
            if agent != agent_type
        }
        
    async def connect(self) -> bool:
        """Connect to ZeroMQ network."""
        if not ZMQ_AVAILABLE:
            logger.error("ZeroMQ not available for messaging")
            return False
            
        try:
            self.context = zmq.asyncio.Context()
            
            # Create ROUTER socket for receiving messages
            self.router_socket = self.context.socket(zmq.ROUTER)
            self.router_socket.bind(f"tcp://*:{self.bind_port}")
            
            # Create DEALER sockets for sending to other agents
            for agent_type, port in self.connect_ports.items():
                dealer = self.context.socket(zmq.DEALER)
                dealer.identity = f"{self.agent_type.value}".encode()
                try:
                    dealer.connect(f"tcp://localhost:{port}")
                    self.dealer_sockets[agent_type] = dealer
                    logger.debug(f"Connected to {agent_type.value} on port {port}")
                except Exception as e:
                    logger.warning(f"Could not connect to {agent_type.value} on port {port}: {e}")
            
            self.connected = True
            logger.info(f"ZeroMQ messenger connected for {self.agent_type.value} on port {self.bind_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ZeroMQ: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from ZeroMQ."""
        try:
            # Close dealer sockets
            for dealer in self.dealer_sockets.values():
                dealer.close()
            self.dealer_sockets.clear()
            
            # Close router socket
            if self.router_socket:
                self.router_socket.close()
            
            # Terminate context
            if self.context:
                self.context.term()
                
            self.connected = False
            logger.info(f"ZeroMQ messenger disconnected for {self.agent_type.value}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from ZeroMQ: {e}")
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered ZMQ handler for {message_type.value}")
    
    async def send_message(self, message: Message) -> bool:
        """Send a message to a specific agent."""
        if not self.connected:
            logger.error("Not connected to ZeroMQ")
            return False
            
        if not message.recipient:
            logger.error("Cannot send ZMQ message without recipient")
            return False
            
        dealer = self.dealer_sockets.get(message.recipient)
        if not dealer:
            logger.error(f"No connection to {message.recipient.value}")
            return False
            
        try:
            # Serialize message
            message_data = json.dumps(message.to_dict()).encode()
            
            # Send message
            await dealer.send(message_data)
            
            logger.debug(f"Sent ZMQ message {message.id} to {message.recipient.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending ZMQ message: {e}")
            return False
    
    async def listen_for_messages(self) -> None:
        """Listen for incoming ZMQ messages."""
        if not self.connected or not self.router_socket:
            logger.error("Not connected to ZeroMQ")
            return
            
        try:
            while self.connected:
                # Receive message (identity, message)
                identity, message_data = await self.router_socket.recv_multipart()
                
                # Parse message
                message_dict = json.loads(message_data.decode())
                message = Message.from_dict(message_dict)
                
                # Handle message
                await self._handle_received_message(message)
                
        except Exception as e:
            logger.error(f"Error listening for ZMQ messages: {e}")
    
    async def _handle_received_message(self, message: Message) -> None:
        """Handle a received ZMQ message."""
        try:
            # Check if message has expired
            if message.is_expired():
                logger.warning(f"Discarding expired ZMQ message {message.id}")
                return
            
            # Handle message
            handler = self.message_handlers.get(message.message_type)
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in ZMQ message handler for {message.message_type.value}: {e}")
            else:
                logger.warning(f"No ZMQ handler registered for message type {message.message_type.value}")
                
        except Exception as e:
            logger.error(f"Error handling received ZMQ message: {e}")

class UnifiedMessenger:
    """Unified messaging interface that combines Redis and ZeroMQ."""
    
    def __init__(self, 
                 agent_type: AgentType,
                 redis_url: str = "redis://localhost:6379",
                 zmq_bind_port: Optional[int] = None,
                 zmq_connect_ports: Optional[Dict[AgentType, int]] = None):
        self.agent_type = agent_type
        self.redis_messenger = RedisMessenger(redis_url, agent_type)
        self.zmq_messenger = ZeroMQMessenger(agent_type, zmq_bind_port, zmq_connect_ports)
        self.connected = False
        
        # Message routing: high-priority messages go via ZMQ, others via Redis
        self.zmq_message_types = {
            MessageType.DECISION_REQUEST,
            MessageType.DECISION_RESPONSE,
            MessageType.RISK_ALERT
        }
        
    async def connect(self) -> bool:
        """Connect both Redis and ZeroMQ messengers."""
        redis_ok = await self.redis_messenger.connect()
        zmq_ok = await self.zmq_messenger.connect()
        
        self.connected = redis_ok or zmq_ok  # At least one should work
        
        if redis_ok and zmq_ok:
            logger.info(f"Unified messenger fully connected for {self.agent_type.value}")
        elif redis_ok:
            logger.warning(f"Unified messenger connected with Redis only for {self.agent_type.value}")
        elif zmq_ok:
            logger.warning(f"Unified messenger connected with ZeroMQ only for {self.agent_type.value}")
        else:
            logger.error(f"Failed to connect unified messenger for {self.agent_type.value}")
            
        return self.connected
    
    async def disconnect(self) -> None:
        """Disconnect both messengers."""
        await self.redis_messenger.disconnect()
        await self.zmq_messenger.disconnect()
        self.connected = False
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a handler for both messaging systems."""
        self.redis_messenger.register_handler(message_type, handler)
        self.zmq_messenger.register_handler(message_type, handler)
    
    async def send_message(self, message: Message) -> bool:
        """Send message via the appropriate transport."""
        if not self.connected:
            logger.error("Unified messenger not connected")
            return False
            
        # Route high-priority messages via ZeroMQ
        if (message.message_type in self.zmq_message_types and 
            message.recipient and 
            self.zmq_messenger.connected):
            return await self.zmq_messenger.send_message(message)
        
        # Route other messages via Redis
        elif self.redis_messenger.connected:
            return await self.redis_messenger.publish_message(message)
        
        else:
            logger.error(f"No available transport for message type {message.message_type.value}")
            return False
    
    async def start_listening(self) -> None:
        """Start listening on both transports."""
        if not self.connected:
            logger.error("Not connected to start listening")
            return
            
        # Start listening tasks
        tasks = []
        
        if self.redis_messenger.connected:
            tasks.append(asyncio.create_task(self.redis_messenger.listen_for_messages()))
            
        if self.zmq_messenger.connected:
            tasks.append(asyncio.create_task(self.zmq_messenger.listen_for_messages()))
        
        if tasks:
            logger.info(f"Started listening for messages on {len(tasks)} transports")
            # Wait for all listening tasks
            await asyncio.gather(*tasks, return_exceptions=True)

# Convenience functions for creating messengers
def create_pads_messenger(**kwargs) -> UnifiedMessenger:
    """Create a messenger for the PADS orchestrator."""
    return UnifiedMessenger(AgentType.PADS, **kwargs)

def create_qbmia_messenger(**kwargs) -> UnifiedMessenger:
    """Create a messenger for the QBMIA agent."""
    return UnifiedMessenger(AgentType.QBMIA, **kwargs)

def create_quasar_messenger(**kwargs) -> UnifiedMessenger:
    """Create a messenger for the QUASAR system."""
    return UnifiedMessenger(AgentType.QUASAR, **kwargs)

def create_quantum_amos_messenger(**kwargs) -> UnifiedMessenger:
    """Create a messenger for the Quantum AMOS agent."""
    return UnifiedMessenger(AgentType.QUANTUM_AMOS, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    async def test_messaging():
        """Test the unified messaging system."""
        # Create messengers for PADS and QBMIA
        pads_messenger = create_pads_messenger()
        qbmia_messenger = create_qbmia_messenger()
        
        # Connect both
        await pads_messenger.connect()
        await qbmia_messenger.connect()
        
        # Register handlers
        async def handle_decision_request(message: Message):
            logger.info(f"QBMIA received decision request: {message.data}")
            
            # Send response
            response = Message(
                message_type=MessageType.DECISION_RESPONSE,
                sender=AgentType.QBMIA,
                recipient=AgentType.PADS,
                data={'decision': 'BUY', 'confidence': 0.8},
                correlation_id=message.id
            )
            await qbmia_messenger.send_message(response)
        
        qbmia_messenger.register_handler(MessageType.DECISION_REQUEST, handle_decision_request)
        
        # Start listening (in background)
        listen_task = asyncio.create_task(qbmia_messenger.start_listening())
        
        # Send test message from PADS
        test_message = Message(
            message_type=MessageType.DECISION_REQUEST,
            sender=AgentType.PADS,
            recipient=AgentType.QBMIA,
            data={'market_state': {'price': 100, 'volume': 1000}},
            priority=1
        )
        
        await pads_messenger.send_message(test_message)
        
        # Wait a bit for message processing
        await asyncio.sleep(2)
        
        # Cleanup
        listen_task.cancel()
        await pads_messenger.disconnect()
        await qbmia_messenger.disconnect()
    
    # Run test
    asyncio.run(test_messaging())