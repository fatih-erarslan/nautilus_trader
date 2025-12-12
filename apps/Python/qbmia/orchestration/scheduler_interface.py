"""
Interface for Quantum Component Orchestrator integration.
"""

import asyncio
import json
import websockets
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class OrchestrationMessage:
    """Message format for orchestrator communication."""
    message_type: str  # 'request', 'response', 'notification', 'heartbeat'
    source: str
    destination: str
    payload: Dict[str, Any]
    timestamp: str = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str):
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

class SchedulerInterface:
    """
    Interface for communication with Quantum Component Orchestrator.
    """

    def __init__(self, agent_id: str, orchestrator_url: str = "ws://localhost:8080"):
        """
        Initialize scheduler interface.

        Args:
            agent_id: QBMIA agent identifier
            orchestrator_url: WebSocket URL for orchestrator
        """
        self.agent_id = agent_id
        self.orchestrator_url = orchestrator_url

        # Connection state
        self.websocket = None
        self.connected = False
        self.connection_task = None

        # Message handling
        self.message_handlers = {}
        self.pending_responses = {}
        self.message_queue = asyncio.Queue()

        # Callbacks
        self.on_connected = None
        self.on_disconnected = None
        self.on_resource_granted = None
        self.on_preemption = None

        # Performance tracking
        self.message_stats = {
            'sent': 0,
            'received': 0,
            'errors': 0
        }

    async def connect(self):
        """Connect to orchestrator."""
        self.connection_task = asyncio.create_task(self._maintain_connection())
        logger.info(f"Scheduler interface connecting to {self.orchestrator_url}")

    async def disconnect(self):
        """Disconnect from orchestrator."""
        self.connected = False

        if self.websocket:
            await self.websocket.close()

        if self.connection_task:
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass

        logger.info("Scheduler interface disconnected")

    async def _maintain_connection(self):
        """Maintain WebSocket connection with reconnection logic."""
        while True:
            try:
                async with websockets.connect(self.orchestrator_url) as websocket:
                    self.websocket = websocket
                    self.connected = True

                    # Send registration message
                    await self._register_with_orchestrator()

                    # Notify connected callback
                    if self.on_connected:
                        await self.on_connected()

                    # Start message handlers
                    receive_task = asyncio.create_task(self._receive_messages())
                    send_task = asyncio.create_task(self._send_messages())
                    heartbeat_task = asyncio.create_task(self._send_heartbeats())

                    # Wait for disconnect
                    await asyncio.gather(
                        receive_task,
                        send_task,
                        heartbeat_task,
                        return_exceptions=True
                    )

            except Exception as e:
                logger.error(f"Orchestrator connection error: {e}")
                self.connected = False

                # Notify disconnected callback
                if self.on_disconnected:
                    await self.on_disconnected()

                # Reconnection backoff
                await asyncio.sleep(5)

    async def _register_with_orchestrator(self):
        """Register QBMIA agent with orchestrator."""
        registration = OrchestrationMessage(
            message_type='request',
            source=self.agent_id,
            destination='orchestrator',
            payload={
                'action': 'register',
                'agent_type': 'QBMIA',
                'capabilities': {
                    'quantum_simulation': True,
                    'gpu_acceleration': True,
                    'state_serialization': True,
                    'preemption_support': True
                },
                'resource_requirements': {
                    'min_memory_mb': 2048,
                    'preferred_gpu_memory_mb': 4096,
                    'max_qubits': 20
                }
            }
        )

        await self._send_message(registration)
        logger.info(f"Registered {self.agent_id} with orchestrator")

    async def _receive_messages(self):
        """Receive messages from orchestrator."""
        try:
            async for message in self.websocket:
                try:
                    msg = OrchestrationMessage.from_json(message)
                    self.message_stats['received'] += 1

                    # Handle based on message type
                    if msg.message_type == 'response':
                        await self._handle_response(msg)
                    elif msg.message_type == 'notification':
                        await self._handle_notification(msg)
                    elif msg.message_type == 'request':
                        await self._handle_request(msg)

                except Exception as e:
                    logger.error(f"Message handling error: {e}")
                    self.message_stats['errors'] += 1

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False

    async def _send_messages(self):
        """Send queued messages to orchestrator."""
        while self.connected:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Send message
                await self.websocket.send(message.to_json())
                self.message_stats['sent'] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message send error: {e}")
                self.message_stats['errors'] += 1

    async def _send_heartbeats(self):
        """Send periodic heartbeats to orchestrator."""
        while self.connected:
            try:
                heartbeat = OrchestrationMessage(
                    message_type='heartbeat',
                    source=self.agent_id,
                    destination='orchestrator',
                    payload={
                        'status': 'alive',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )

                await self._send_message(heartbeat)
                await asyncio.sleep(30)  # 30 second heartbeat interval

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _send_message(self, message: OrchestrationMessage):
        """Queue message for sending."""
        await self.message_queue.put(message)

    async def _handle_response(self, message: OrchestrationMessage):
        """Handle response message."""
        correlation_id = message.correlation_id

        if correlation_id and correlation_id in self.pending_responses:
            # Resolve pending response
            future = self.pending_responses[correlation_id]
            future.set_result(message.payload)
            del self.pending_responses[correlation_id]

    async def _handle_notification(self, message: OrchestrationMessage):
        """Handle notification from orchestrator."""
        notification_type = message.payload.get('type')

        if notification_type == 'resource_granted':
            if self.on_resource_granted:
                await self.on_resource_granted(message.payload)

        elif notification_type == 'preemption_request':
            if self.on_preemption:
                await self.on_preemption(message.payload)

        elif notification_type == 'resource_revoked':
            logger.warning(f"Resources revoked: {message.payload}")

    async def _handle_request(self, message: OrchestrationMessage):
        """Handle request from orchestrator."""
        action = message.payload.get('action')

        if action == 'get_status':
            # Respond with current status
            response = OrchestrationMessage(
                message_type='response',
                source=self.agent_id,
                destination=message.source,
                correlation_id=message.correlation_id,
                payload={
                    'status': 'active',
                    'resource_usage': await self._get_resource_usage()
                }
            )
            await self._send_message(response)

        elif action == 'prepare_checkpoint':
            # Prepare for checkpointing
            response = OrchestrationMessage(
                message_type='response',
                source=self.agent_id,
                destination=message.source,
                correlation_id=message.correlation_id,
                payload={
                    'ready': True,
                    'checkpoint_size_estimate': 1024 * 1024  # 1MB estimate
                }
            )
            await self._send_message(response)

    async def request_resources(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request resources from orchestrator.

        Args:
            requirements: Resource requirements

        Returns:
            Orchestrator response
        """
        correlation_id = f"{self.agent_id}_{datetime.utcnow().timestamp()}"

        request = OrchestrationMessage(
            message_type='request',
            source=self.agent_id,
            destination='orchestrator',
            correlation_id=correlation_id,
            payload={
                'action': 'request_resources',
                'requirements': requirements
            }
        )

        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[correlation_id] = response_future

        # Send request
        await self._send_message(request)

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(
                response_future,
                timeout=requirements.get('timeout', 30.0)
            )
            return response
        except asyncio.TimeoutError:
            del self.pending_responses[correlation_id]
            raise TimeoutError("Resource request timeout")

    async def release_resources(self, allocation_id: str):
        """Release allocated resources."""
        release_msg = OrchestrationMessage(
            message_type='notification',
            source=self.agent_id,
            destination='orchestrator',
            payload={
                'type': 'resource_release',
                'allocation_id': allocation_id
            }
        )

        await self._send_message(release_msg)

    async def report_performance(self, metrics: Dict[str, Any]):
        """Report performance metrics to orchestrator."""
        report = OrchestrationMessage(
            message_type='notification',
            source=self.agent_id,
            destination='orchestrator',
            payload={
                'type': 'performance_report',
                'metrics': metrics
            }
        )

        await self._send_message(report)

    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        # This would interface with resource manager
        return {
            'cpu_usage': 50.0,
            'memory_mb': 2048,
            'gpu_memory_mb': 1024
        }

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status and statistics."""
        return {
            'connected': self.connected,
            'url': self.orchestrator_url,
            'message_stats': self.message_stats.copy(),
            'pending_responses': len(self.pending_responses)
        }
