"""WebSocket Server for Real-time Model Updates and Streaming Predictions."""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from collections import defaultdict, deque
import websockets
from websockets.server import WebSocketServerProtocol
import traceback

# Import storage components
from ..storage.model_storage import ModelStorage
from ..storage.metadata_manager import MetadataManager, ModelStatus
from ..storage.version_control import ModelVersionControl

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PREDICTION_REQUEST = "prediction_request"
    PREDICTION_RESPONSE = "prediction_response"
    MODEL_UPDATE = "model_update"
    STRATEGY_UPDATE = "strategy_update"
    PERFORMANCE_UPDATE = "performance_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    NOTIFICATION = "notification"


class SubscriptionType(Enum):
    """Types of subscriptions."""
    MODEL_UPDATES = "model_updates"
    STRATEGY_UPDATES = "strategy_updates"
    PERFORMANCE_UPDATES = "performance_updates"
    REAL_TIME_PREDICTIONS = "real_time_predictions"
    SYSTEM_STATUS = "system_status"


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format."""
    message_type: MessageType
    message_id: str
    timestamp: datetime
    data: Dict[str, Any]
    client_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'message_type': self.message_type.value,
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'client_id': self.client_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WebSocketMessage':
        """Create from dictionary."""
        return cls(
            message_type=MessageType(data['message_type']),
            message_id=data['message_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            client_id=data.get('client_id')
        )


@dataclass
class ClientConnection:
    """Client connection information."""
    client_id: str
    websocket: WebSocketServerProtocol
    connected_at: datetime
    subscriptions: Set[str]
    last_heartbeat: datetime
    metadata: Dict[str, Any]
    
    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if connection is still alive."""
        return (datetime.now() - self.last_heartbeat).total_seconds() < timeout_seconds


class ModelWebSocketServer:
    """WebSocket server for real-time model management and streaming predictions."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002,
                 storage_path: str = "model_management"):
        """
        Initialize WebSocket server.
        
        Args:
            host: Server host
            port: Server port
            storage_path: Path to model storage
        """
        self.host = host
        self.port = port
        self.storage_path = storage_path
        
        # Initialize storage components
        self.model_storage = ModelStorage(f"{storage_path}/models")
        self.metadata_manager = MetadataManager(f"{storage_path}/storage")
        self.version_control = ModelVersionControl(f"{storage_path}/models/versions")
        
        # Client management
        self.clients: Dict[str, ClientConnection] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # subscription_type -> client_ids
        self.client_lock = threading.Lock()
        
        # Model cache for predictions
        self.model_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = timedelta(minutes=30)
        
        # Message history for replay
        self.message_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.stats = {
            'connections_total': 0,
            'connections_active': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'predictions_made': 0,
            'errors_count': 0,
            'start_time': datetime.now()
        }
        
        # Background tasks
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info(f"WebSocket server initialized on {host}:{port}")
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._model_update_monitor())
        ]
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        ):
            await asyncio.Future()  # Run forever
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close all client connections
        with self.client_lock:
            for client in list(self.clients.values()):
                try:
                    await client.websocket.close()
                except:
                    pass
            self.clients.clear()
            self.subscriptions.clear()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        client_id = str(uuid.uuid4())
        
        # Create client connection
        client = ClientConnection(
            client_id=client_id,
            websocket=websocket,
            connected_at=datetime.now(),
            subscriptions=set(),
            last_heartbeat=datetime.now(),
            metadata={}
        )
        
        # Register client
        with self.client_lock:
            self.clients[client_id] = client
            self.stats['connections_total'] += 1
            self.stats['connections_active'] += 1
        
        logger.info(f"Client {client_id} connected")
        
        try:
            # Send welcome message
            await self._send_message(client, WebSocketMessage(
                message_type=MessageType.NOTIFICATION,
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                data={
                    'type': 'welcome',
                    'client_id': client_id,
                    'server_info': {
                        'version': '1.0.0',
                        'capabilities': [t.value for t in SubscriptionType],
                        'message_types': [t.value for t in MessageType]
                    }
                },
                client_id=client_id
            ))
            
            # Handle messages
            async for message_str in websocket:
                try:
                    await self._handle_message(client, message_str)
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    await self._send_error(client, str(e))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error with client {client_id}: {e}")
        finally:
            # Cleanup client
            with self.client_lock:
                if client_id in self.clients:
                    # Remove from subscriptions
                    for subscription_type in client.subscriptions:
                        self.subscriptions[subscription_type].discard(client_id)
                    
                    del self.clients[client_id]
                    self.stats['connections_active'] -= 1
    
    async def _handle_message(self, client: ClientConnection, message_str: str):
        """Handle incoming message from client."""
        try:
            message_data = json.loads(message_str)
            message = WebSocketMessage.from_dict(message_data)
            
            # Update heartbeat
            client.last_heartbeat = datetime.now()
            self.stats['messages_received'] += 1
            
            # Handle different message types
            if message.message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(client, message)
            
            elif message.message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(client, message)
            
            elif message.message_type == MessageType.PREDICTION_REQUEST:
                await self._handle_prediction_request(client, message)
            
            elif message.message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(client, message)
            
            else:
                await self._send_error(client, f"Unknown message type: {message.message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(client, "Invalid JSON message")
        except Exception as e:
            await self._send_error(client, f"Message processing error: {str(e)}")
    
    async def _handle_subscribe(self, client: ClientConnection, message: WebSocketMessage):
        """Handle subscription request."""
        subscription_type = message.data.get('subscription_type')
        filters = message.data.get('filters', {})
        
        if not subscription_type:
            await self._send_error(client, "Subscription type is required")
            return
        
        try:
            # Validate subscription type
            SubscriptionType(subscription_type)
            
            # Add to subscriptions
            with self.client_lock:
                client.subscriptions.add(subscription_type)
                self.subscriptions[subscription_type].add(client.client_id)
                
                # Store filters in client metadata
                client.metadata[f'{subscription_type}_filters'] = filters
            
            # Send subscription confirmation
            await self._send_message(client, WebSocketMessage(
                message_type=MessageType.NOTIFICATION,
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                data={
                    'type': 'subscription_confirmed',
                    'subscription_type': subscription_type,
                    'filters': filters
                },
                client_id=client.client_id
            ))
            
            # Send recent history if available
            await self._send_subscription_history(client, subscription_type)
            
            logger.info(f"Client {client.client_id} subscribed to {subscription_type}")
            
        except ValueError:
            await self._send_error(client, f"Invalid subscription type: {subscription_type}")
    
    async def _handle_unsubscribe(self, client: ClientConnection, message: WebSocketMessage):
        """Handle unsubscribe request."""
        subscription_type = message.data.get('subscription_type')
        
        if not subscription_type:
            await self._send_error(client, "Subscription type is required")
            return
        
        # Remove from subscriptions
        with self.client_lock:
            client.subscriptions.discard(subscription_type)
            self.subscriptions[subscription_type].discard(client.client_id)
            
            # Remove filters
            client.metadata.pop(f'{subscription_type}_filters', None)
        
        # Send confirmation
        await self._send_message(client, WebSocketMessage(
            message_type=MessageType.NOTIFICATION,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            data={
                'type': 'unsubscription_confirmed',
                'subscription_type': subscription_type
            },
            client_id=client.client_id
        ))
        
        logger.info(f"Client {client.client_id} unsubscribed from {subscription_type}")
    
    async def _handle_prediction_request(self, client: ClientConnection, message: WebSocketMessage):
        """Handle real-time prediction request."""
        try:
            model_id = message.data.get('model_id')
            input_data = message.data.get('input_data', {})
            
            if not model_id:
                await self._send_error(client, "Model ID is required for prediction")
                return
            
            # Load model if not cached
            model, metadata = await self._get_or_load_model(model_id)
            
            # Make prediction
            prediction = await self._make_prediction(model, input_data, metadata)
            
            # Send prediction response
            await self._send_message(client, WebSocketMessage(
                message_type=MessageType.PREDICTION_RESPONSE,
                message_id=message.message_id,  # Keep same ID for request/response matching
                timestamp=datetime.now(),
                data={
                    'model_id': model_id,
                    'prediction': prediction,
                    'input_data': input_data,
                    'model_metadata': {
                        'name': metadata.name,
                        'strategy_name': metadata.strategy_name,
                        'version': metadata.version
                    }
                },
                client_id=client.client_id
            ))
            
            self.stats['predictions_made'] += 1
            
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            await self._send_error(client, f"Prediction failed: {str(e)}")
    
    async def _handle_heartbeat(self, client: ClientConnection, message: WebSocketMessage):
        """Handle heartbeat message."""
        # Update heartbeat timestamp
        client.last_heartbeat = datetime.now()
        
        # Send heartbeat response
        await self._send_message(client, WebSocketMessage(
            message_type=MessageType.HEARTBEAT,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            data={
                'type': 'heartbeat_response',
                'server_time': datetime.now().isoformat()
            },
            client_id=client.client_id
        ))
    
    async def _send_message(self, client: ClientConnection, message: WebSocketMessage):
        """Send message to client."""
        try:
            message_str = json.dumps(message.to_dict(), default=str)
            await client.websocket.send(message_str)
            self.stats['messages_sent'] += 1
        except Exception as e:
            logger.error(f"Failed to send message to {client.client_id}: {e}")
            self.stats['errors_count'] += 1
    
    async def _send_error(self, client: ClientConnection, error_message: str):
        """Send error message to client."""
        await self._send_message(client, WebSocketMessage(
            message_type=MessageType.ERROR,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            data={
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            },
            client_id=client.client_id
        ))
        self.stats['errors_count'] += 1
    
    async def _send_subscription_history(self, client: ClientConnection, subscription_type: str):
        """Send recent history for subscription."""
        if subscription_type in self.message_history:
            history = list(self.message_history[subscription_type])
            
            for message_data in history[-10:]:  # Send last 10 messages
                await self._send_message(client, WebSocketMessage(
                    message_type=MessageType.NOTIFICATION,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={
                        'type': 'history',
                        'subscription_type': subscription_type,
                        'historical_data': message_data
                    },
                    client_id=client.client_id
                ))
    
    async def _get_or_load_model(self, model_id: str) -> tuple:
        """Get model from cache or load from storage."""
        with self.cache_lock:
            # Check cache
            if model_id in self.model_cache:
                cache_entry = self.model_cache[model_id]
                if datetime.now() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['model'], cache_entry['metadata']
                else:
                    # Remove expired entry
                    del self.model_cache[model_id]
        
        # Load from storage
        model, metadata = self.model_storage.load_model(model_id)
        
        # Cache the model
        with self.cache_lock:
            self.model_cache[model_id] = {
                'model': model,
                'metadata': metadata,
                'timestamp': datetime.now()
            }
        
        return model, metadata
    
    async def _make_prediction(self, model: Any, input_data: Dict, metadata: Any) -> Dict:
        """Make prediction using the model."""
        # Simplified prediction logic (same as in MCP server)
        if isinstance(model, dict):
            # Parameter-based prediction
            return {
                'action': 'hold',
                'confidence': 0.5,
                'position_size': model.get('base_position_size', 0.05),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'parameters'
            }
        
        # If model has predict method
        if hasattr(model, 'predict'):
            result = model.predict(input_data)
            return {
                'prediction': result,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'ml_model'
            }
        
        # Fallback
        return {
            'action': 'hold',
            'confidence': 0.5,
            'position_size': 0.05,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'fallback'
        }
    
    async def broadcast_model_update(self, model_id: str, update_data: Dict):
        """Broadcast model update to subscribed clients."""
        message_data = {
            'type': 'model_update',
            'model_id': model_id,
            'update_data': update_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.message_history[SubscriptionType.MODEL_UPDATES.value].append(message_data)
        
        # Send to subscribed clients
        await self._broadcast_to_subscription(SubscriptionType.MODEL_UPDATES.value, message_data)
    
    async def broadcast_strategy_update(self, strategy_name: str, analytics_data: Dict):
        """Broadcast strategy analytics update."""
        message_data = {
            'type': 'strategy_update',
            'strategy_name': strategy_name,
            'analytics_data': analytics_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.message_history[SubscriptionType.STRATEGY_UPDATES.value].append(message_data)
        
        # Send to subscribed clients
        await self._broadcast_to_subscription(SubscriptionType.STRATEGY_UPDATES.value, message_data)
    
    async def broadcast_performance_update(self, performance_data: Dict):
        """Broadcast performance metrics update."""
        message_data = {
            'type': 'performance_update',
            'performance_data': performance_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.message_history[SubscriptionType.PERFORMANCE_UPDATES.value].append(message_data)
        
        # Send to subscribed clients
        await self._broadcast_to_subscription(SubscriptionType.PERFORMANCE_UPDATES.value, message_data)
    
    async def _broadcast_to_subscription(self, subscription_type: str, data: Dict):
        """Broadcast data to clients subscribed to a specific type."""
        client_ids = self.subscriptions.get(subscription_type, set()).copy()
        
        for client_id in client_ids:
            client = self.clients.get(client_id)
            if client and client.websocket.open:
                try:
                    # Apply filters if any
                    filters = client.metadata.get(f'{subscription_type}_filters', {})
                    if self._data_matches_filters(data, filters):
                        await self._send_message(client, WebSocketMessage(
                            message_type=MessageType.NOTIFICATION,
                            message_id=str(uuid.uuid4()),
                            timestamp=datetime.now(),
                            data=data,
                            client_id=client_id
                        ))
                except Exception as e:
                    logger.error(f"Failed to broadcast to client {client_id}: {e}")
    
    def _data_matches_filters(self, data: Dict, filters: Dict) -> bool:
        """Check if data matches client filters."""
        if not filters:
            return True
        
        # Apply strategy filter
        if 'strategy_name' in filters:
            if data.get('strategy_name') != filters['strategy_name']:
                return False
        
        # Apply model filter
        if 'model_id' in filters:
            if data.get('model_id') != filters['model_id']:
                return False
        
        # Apply performance threshold filters
        if 'min_performance' in filters:
            performance_data = data.get('performance_data', {})
            for metric, min_value in filters['min_performance'].items():
                if performance_data.get(metric, 0) < min_value:
                    return False
        
        return True
    
    async def _heartbeat_monitor(self):
        """Monitor client heartbeats and remove stale connections."""
        while self.running:
            try:
                stale_clients = []
                
                with self.client_lock:
                    for client_id, client in self.clients.items():
                        if not client.is_alive():
                            stale_clients.append(client_id)
                
                # Remove stale clients
                for client_id in stale_clients:
                    logger.info(f"Removing stale client {client_id}")
                    with self.client_lock:
                        client = self.clients.pop(client_id, None)
                        if client:
                            # Remove from subscriptions
                            for subscription_type in client.subscriptions:
                                self.subscriptions[subscription_type].discard(client_id)
                            
                            self.stats['connections_active'] -= 1
                            
                            try:
                                await client.websocket.close()
                            except:
                                pass
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitor(self):
        """Monitor and broadcast system performance."""
        while self.running:
            try:
                # Collect performance data
                performance_data = {
                    'server_stats': self.stats.copy(),
                    'active_connections': len(self.clients),
                    'cached_models': len(self.model_cache),
                    'subscription_counts': {
                        sub_type: len(client_ids) 
                        for sub_type, client_ids in self.subscriptions.items()
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Broadcast to performance subscribers
                await self.broadcast_performance_update(performance_data)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def _model_update_monitor(self):
        """Monitor for model updates and broadcast changes."""
        last_check = datetime.now()
        
        while self.running:
            try:
                # Check for new models since last check
                models = self.metadata_manager.search_models(
                    created_after=last_check,
                    limit=100
                )
                
                for model in models:
                    await self.broadcast_model_update(
                        model.model_id,
                        {
                            'action': 'created',
                            'model_data': model.to_dict()
                        }
                    )
                
                last_check = datetime.now()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in model update monitor: {e}")
                await asyncio.sleep(300)
    
    def get_server_stats(self) -> Dict:
        """Get current server statistics."""
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'active_connections': len(self.clients),
            'subscriptions': {
                sub_type: len(client_ids) 
                for sub_type, client_ids in self.subscriptions.items()
            },
            'cached_models': len(self.model_cache)
        }


# Convenience function to start server
async def start_websocket_server(host: str = "0.0.0.0", port: int = 8002):
    """Start WebSocket server."""
    server = ModelWebSocketServer(host=host, port=port)
    await server.start_server()


if __name__ == "__main__":
    # Run the server
    asyncio.run(start_websocket_server())