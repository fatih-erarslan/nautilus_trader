#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulsar Connector for CDFA Extensions

Provides a communication interface between CDFA and the Pulsar module,
which contains Q*, River, Cerebellar SNN, and Narrative Forecaster components.

Author: Created on May 6, 2025
"""

import logging
import time
import json
import threading
import queue
import asyncio
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
import os
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
import traceback

# ---- Optional dependencies with graceful fallbacks ----

# Redis for message passing
try:
    import redis
    import msgpack
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Direct Redis communication will be disabled.")
    
    # Create dummy msgpack module
    class DummyMsgpack:
        def packb(self, data, *args, **kwargs):
            return json.dumps(data).encode()
            
        def unpackb(self, data, *args, **kwargs):
            return json.loads(data.decode())
    
    msgpack = DummyMsgpack()

# ZeroMQ for direct communication
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    warnings.warn("ZeroMQ not available. Direct ZMQ communication will be disabled.")

class CommunicationMode(Enum):
    """Communication modes for the Pulsar connector."""
    REDIS = auto()
    ZMQ = auto()
    IPC = auto()
    HTTP = auto()
    DIRECT = auto()  # Direct in-memory calls when Pulsar is in the same process

class MessagePriority(Enum):
    """Priority levels for Pulsar messages."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class PulsarMessage:
    """Message format for communication with Pulsar module."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "cdfa"
    target: str = "pulsar"
    component: str = ""  # Target Pulsar component (q_star, river, cerebellum, narrative)
    action: str = ""  # Requested action
    priority: MessagePriority = MessagePriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    response_required: bool = True
    timeout: float = 30.0  # Timeout in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
            "target": self.target,
            "component": self.component,
            "action": self.action,
            "priority": self.priority.value,
            "data": self.data,
            "response_required": self.response_required,
            "timeout": self.timeout,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PulsarMessage':
        """Create message from dictionary."""
        # Convert priority from int to enum
        if "priority" in data and isinstance(data["priority"], int):
            data["priority"] = MessagePriority(data["priority"])
            
        return cls(**data)

class PulsarConnector:
    """
    Communication interface with the Pulsar module for CDFA.
    
    This connector handles message passing between CDFA and Pulsar components:
    - Q* reinforcement learning
    - River online machine learning
    - Cerebellar SNN
    - Narrative Forecaster
    
    It supports multiple communication modes, auto-reconnection, and asynchronous requests.
    """
    
    def __init__(self, mode: Union[str, CommunicationMode] = "redis",
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the Pulsar connector.
        
        Args:
            mode: Communication mode ('redis', 'zmq', 'ipc', 'http', or 'direct')
            config: Configuration options
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Parse mode
        if isinstance(mode, str):
            try:
                self.mode = CommunicationMode[mode.upper()]
            except KeyError:
                self.logger.warning(f"Invalid mode '{mode}', defaulting to REDIS")
                self.mode = CommunicationMode.REDIS
        else:
            self.mode = mode
            
        # Default configuration
        self.default_config = {
            # Redis configuration
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            "redis_password": None,
            "redis_channel_prefix": "pulsar:",
            
            # ZMQ configuration
            "zmq_address": "tcp://localhost:5555",
            "zmq_timeout": 5000,  # ms
            
            # IPC configuration
            "ipc_socket_path": "/tmp/pulsar_socket",
            
            # HTTP configuration
            "http_url": "http://localhost:8000/pulsar",
            "http_timeout": 30,  # seconds
            
            # General configuration
            "response_timeout": 30,  # seconds
            "reconnect_interval": 5,  # seconds
            "max_retries": 3,
            "keep_alive": True,
            "compress_messages": True,
            "message_queue_size": 100,
            "background_processing": True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize state
        self._is_connected = False
        self._is_running = True
        self._lock = threading.RLock()
        self._response_callbacks = {}
        self._message_queue = queue.PriorityQueue(maxsize=self.config["message_queue_size"])
        self._responses = {}
        
        # Initialize communication clients
        self._redis_client = None
        self._zmq_context = None
        self._zmq_socket = None
        self._http_session = None
        
        # Initialize event loop
        self._loop = None
        self._worker_thread = None
        
        # Initialize message handling
        if self.config["background_processing"]:
            self._initialize_background_processing()
        
        # Try to connect
        self.connect()
        
    def _initialize_background_processing(self):
        """Initialize background processing thread."""
        self._worker_thread = threading.Thread(
            target=self._background_worker,
            daemon=True,
            name="PulsarConnectorWorker"
        )
        self._worker_thread.start()
    
    def _background_worker(self):
        """Background worker for processing message queue."""
        self.logger.info("Starting background worker thread")
        
        while self._is_running:
            try:
                # Get next message from queue
                try:
                    # Format: (priority, message)
                    _, message = self._message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process message
                try:
                    self._process_outgoing_message(message)
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                
                # Mark task as done
                self._message_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in background worker: {e}")
                time.sleep(0.1)
                
        self.logger.info("Background worker thread stopped")
    
    def connect(self) -> bool:
        """
        Connect to the Pulsar module using the configured mode.
        
        Returns:
            Success flag
        """
        try:
            with self._lock:
                if self.mode == CommunicationMode.REDIS:
                    success = self._connect_redis()
                elif self.mode == CommunicationMode.ZMQ:
                    success = self._connect_zmq()
                elif self.mode == CommunicationMode.IPC:
                    success = self._connect_ipc()
                elif self.mode == CommunicationMode.HTTP:
                    success = self._connect_http()
                elif self.mode == CommunicationMode.DIRECT:
                    success = self._connect_direct()
                else:
                    self.logger.error(f"Unsupported communication mode: {self.mode}")
                    return False
                    
                self._is_connected = success
                return success
                
        except Exception as e:
            self.logger.error(f"Error connecting to Pulsar: {e}")
            self._is_connected = False
            return False
    
    def _connect_redis(self) -> bool:
        """Connect using Redis."""
        if not REDIS_AVAILABLE:
            self.logger.error("Redis is not available")
            return False
            
        try:
            # Create Redis client
            self._redis_client = redis.Redis(
                host=self.config["redis_host"],
                port=self.config["redis_port"],
                db=self.config["redis_db"],
                password=self.config["redis_password"],
                socket_timeout=5.0,
                retry_on_timeout=True,
                decode_responses=False
            )
            
            # Test connection
            self._redis_client.ping()
            
            # Setup PubSub for responses
            self._redis_pubsub = self._redis_client.pubsub(ignore_subscribe_messages=True)
            
            # Subscribe to response channel
            response_channel = f"{self.config['redis_channel_prefix']}response"
            self._redis_pubsub.subscribe(response_channel)
            
            # Start listener thread
            self._redis_listener_thread = threading.Thread(
                target=self._redis_listener,
                daemon=True,
                name="PulsarRedisListener"
            )
            self._redis_listener_thread.start()
            
            self.logger.info(f"Connected to Redis at {self.config['redis_host']}:{self.config['redis_port']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def _redis_listener(self):
        """Listen for Redis messages in background thread."""
        self.logger.debug("Redis listener thread started")
        
        while self._is_running:
            try:
                if self._redis_pubsub is None:
                    time.sleep(1.0)
                    continue
                    
                # Get message with timeout
                message = self._redis_pubsub.get_message(timeout=0.5)
                if message:
                    try:
                        # Process response
                        data = message.get("data")
                        if data:
                            # Decode data with MessagePack
                            try:
                                data = msgpack.unpackb(data, raw=False)
                                self._handle_response(data)
                            except Exception as e:
                                self.logger.error(f"Error decoding message: {e}")
                    except Exception as e:
                        self.logger.error(f"Error handling Redis message: {e}")
                        
            except redis.RedisError as e:
                self.logger.error(f"Redis error in listener: {e}")
                time.sleep(1.0)
                # Try to reconnect
                with self._lock:
                    self._connect_redis()
                    
            except Exception as e:
                self.logger.error(f"Error in Redis listener: {e}")
                time.sleep(0.1)
                
        self.logger.debug("Redis listener thread stopped")
    
    def _connect_zmq(self) -> bool:
        """Connect using ZeroMQ."""
        if not ZMQ_AVAILABLE:
            self.logger.error("ZeroMQ is not available")
            return False
            
        try:
            # Create ZeroMQ context and socket
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.REQ)
            
            # Set timeout
            self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.config["zmq_timeout"])
            self._zmq_socket.setsockopt(zmq.SNDTIMEO, self.config["zmq_timeout"])
            
            # Connect to server
            self._zmq_socket.connect(self.config["zmq_address"])
            
            self.logger.info(f"Connected to ZeroMQ at {self.config['zmq_address']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to ZeroMQ: {e}")
            return False
    
    def _connect_ipc(self) -> bool:
        """Connect using IPC socket."""
        if not ZMQ_AVAILABLE:
            self.logger.error("ZeroMQ is not available for IPC")
            return False
            
        try:
            # Create ZeroMQ context and socket
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.REQ)
            
            # Set timeout
            self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.config["zmq_timeout"])
            self._zmq_socket.setsockopt(zmq.SNDTIMEO, self.config["zmq_timeout"])
            
            # Connect to server
            socket_path = f"ipc://{self.config['ipc_socket_path']}"
            self._zmq_socket.connect(socket_path)
            
            self.logger.info(f"Connected to IPC socket at {socket_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IPC socket: {e}")
            return False
    
    def _connect_http(self) -> bool:
        """Connect using HTTP."""
        try:
            import requests
            self._http_session = requests.Session()
            
            # Test connection
            response = self._http_session.get(
                f"{self.config['http_url']}/ping",
                timeout=self.config["http_timeout"]
            )
            
            if response.status_code == 200:
                self.logger.info(f"Connected to HTTP API at {self.config['http_url']}")
                return True
            else:
                self.logger.error(f"HTTP connection failed with status code {response.status_code}")
                return False
                
        except ImportError:
            self.logger.error("Requests library not available for HTTP communication")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to connect to HTTP API: {e}")
            return False
    
    def _connect_direct(self) -> bool:
        """Connect using direct in-memory calls."""
        try:
            # Try to import Pulsar module
            try:
                import pulsar
                self._pulsar_module = pulsar
                
                # Check for required components
                required_components = [
                    "QStarLearningAgent",
                    "RiverOnlineML",
                    "CerebellumSNN",
                    "QuantumOptimizer",
                    "NarrativeForecaster"
                ]
                
                for component in required_components:
                    if not hasattr(pulsar, component):
                        self.logger.warning(f"Pulsar module missing required component: {component}")
                        
                self.logger.info("Connected to Pulsar module directly")
                return True
                
            except ImportError:
                self.logger.error("Pulsar module not found for direct connection")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect directly to Pulsar: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the Pulsar module."""
        with self._lock:
            self._is_running = False
            
            # Clean up resources based on mode
            if self.mode == CommunicationMode.REDIS and self._redis_client:
                try:
                    if hasattr(self, '_redis_pubsub') and self._redis_pubsub:
                        self._redis_pubsub.close()
                    self._redis_client.close()
                except:
                    pass
                    
            elif self.mode in (CommunicationMode.ZMQ, CommunicationMode.IPC) and self._zmq_socket:
                try:
                    self._zmq_socket.close()
                    if self._zmq_context:
                        self._zmq_context.term()
                except:
                    pass
                    
            elif self.mode == CommunicationMode.HTTP and self._http_session:
                try:
                    self._http_session.close()
                except:
                    pass
                    
            self._is_connected = False
            self.logger.info("Disconnected from Pulsar")
    
    def is_connected(self) -> bool:
        """
        Check if connected to the Pulsar module.
        
        Returns:
            Connection status
        """
        return self._is_connected
    
    def _handle_response(self, response: Dict[str, Any]):
        """Handle response from Pulsar."""
        message_id = response.get("id")
        if not message_id:
            self.logger.warning("Received response without message ID")
            return
            
        # Check if waiting for this response
        with self._lock:
            callback = self._response_callbacks.pop(message_id, None)
            
            # Store response
            self._responses[message_id] = response
            
        # Call callback if registered
        if callback and callable(callback):
            try:
                callback(response)
            except Exception as e:
                self.logger.error(f"Error in response callback: {e}")
    
    def _process_outgoing_message(self, message: PulsarMessage):
        """Process outgoing message to Pulsar."""
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError("Not connected to Pulsar")
                
        message_dict = message.to_dict()
        
        # Send based on communication mode
        if self.mode == CommunicationMode.REDIS:
            self._send_redis(message_dict)
        elif self.mode in (CommunicationMode.ZMQ, CommunicationMode.IPC):
            self._send_zmq(message_dict)
        elif self.mode == CommunicationMode.HTTP:
            self._send_http(message_dict)
        elif self.mode == CommunicationMode.DIRECT:
            self._send_direct(message_dict)
        else:
            raise ValueError(f"Unsupported communication mode: {self.mode}")
    
    def _send_redis(self, message: Dict[str, Any]):
        """Send message via Redis."""
        if not self._redis_client:
            raise ConnectionError("Redis client not initialized")
            
        try:
            # Determine channel based on target component
            component = message.get("component", "").lower()
            
            # Default channel
            channel = f"{self.config['redis_channel_prefix']}request"
            
            # Component-specific channel
            if component:
                component_channel = f"{self.config['redis_channel_prefix']}{component}"
                channel = component_channel
                
            # Encode message with MessagePack
            packed = msgpack.packb(message, use_bin_type=True)
            
            # Publish message
            self._redis_client.publish(channel, packed)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message via Redis: {e}")
            # Mark as disconnected to trigger reconnect
            self._is_connected = False
            raise
    
    def _send_zmq(self, message: Dict[str, Any]):
        """Send message via ZeroMQ."""
        if not self._zmq_socket:
            raise ConnectionError("ZeroMQ socket not initialized")
            
        try:
            # Encode message with MessagePack
            packed = msgpack.packb(message, use_bin_type=True)
            
            # Send message
            self._zmq_socket.send(packed)
            
            # Wait for response only if required
            if message.get("response_required", True):
                # Set timeout
                timeout = message.get("timeout", self.config["response_timeout"]) * 1000
                self._zmq_socket.setsockopt(zmq.RCVTIMEO, int(timeout))
                
                # Receive response
                response_data = self._zmq_socket.recv()
                
                # Decode response
                response = msgpack.unpackb(response_data, raw=False)
                
                # Handle response
                self._handle_response(response)
                
            return True
            
        except zmq.ZMQError as e:
            self.logger.error(f"ZeroMQ error: {e}")
            # Mark as disconnected to trigger reconnect
            self._is_connected = False
            raise
            
        except Exception as e:
            self.logger.error(f"Error sending message via ZeroMQ: {e}")
            raise
    
    def _send_http(self, message: Dict[str, Any]):
        """Send message via HTTP."""
        if not self._http_session:
            raise ConnectionError("HTTP session not initialized")
            
        try:
            # Determine endpoint based on component
            component = message.get("component", "").lower()
            action = message.get("action", "").lower()
            
            # Default endpoint
            endpoint = f"{self.config['http_url']}/request"
            
            # Component-specific endpoint
            if component:
                endpoint = f"{self.config['http_url']}/{component}"
                
                # Action-specific endpoint
                if action:
                    endpoint = f"{endpoint}/{action}"
                    
            # Send request
            timeout = message.get("timeout", self.config["http_timeout"])
            response = self._http_session.post(
                endpoint,
                json=message,
                timeout=timeout
            )
            
            # Check response
            if response.status_code == 200:
                # Parse response
                response_data = response.json()
                
                # Handle response
                self._handle_response(response_data)
                return True
            else:
                self.logger.error(f"HTTP request failed with status code {response.status_code}")
                raise RuntimeError(f"HTTP request failed: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending message via HTTP: {e}")
            # Mark as disconnected to trigger reconnect
            self._is_connected = False
            raise
    
    def _send_direct(self, message: Dict[str, Any]):
        """Send message via direct in-memory call."""
        if not hasattr(self, '_pulsar_module') or self._pulsar_module is None:
            raise ConnectionError("Pulsar module not initialized")
            
        try:
            # Extract component and action
            component_name = message.get("component", "").lower()
            action = message.get("action", "").lower()
            data = message.get("data", {})
            
            # Get component
            if component_name == "q_star":
                component = self._pulsar_module.QStarLearningAgent()
            elif component_name == "river":
                component = self._pulsar_module.RiverOnlineML()
            elif component_name == "cerebellum":
                component = self._pulsar_module.CerebellumSNN()
            elif component_name == "narrative":
                component = self._pulsar_module.NarrativeForecaster()
            elif component_name == "optimizer":
                component = self._pulsar_module.QuantumOptimizer()
            else:
                raise ValueError(f"Unknown component: {component_name}")
                
            # Call method
            if not hasattr(component, action):
                raise ValueError(f"Unknown action '{action}' for component '{component_name}'")
                
            method = getattr(component, action)
            
            # Call method with data
            result = method(**data)
            
            # Create response
            response = {
                "id": message.get("id"),
                "timestamp": time.time(),
                "source": "pulsar",
                "target": "cdfa",
                "component": component_name,
                "action": action,
                "result": result,
                "status": "success",
                "metadata": {}
            }
            
            # Handle response
            self._handle_response(response)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message directly to Pulsar: {e}")
            # Create error response
            error_response = {
                "id": message.get("id"),
                "timestamp": time.time(),
                "source": "pulsar",
                "target": "cdfa",
                "component": message.get("component", ""),
                "action": message.get("action", ""),
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": {}
            }
            
            # Handle error response
            self._handle_response(error_response)
            
            raise
    
    def send_message(self, message: PulsarMessage, callback: Optional[Callable] = None) -> str:
        """
        Send a message to the Pulsar module.
        
        Args:
            message: Message to send
            callback: Optional callback for response
            
        Returns:
            Message ID
        """
        if callback and callable(callback):
            with self._lock:
                self._response_callbacks[message.id] = callback
                
        # Add to message queue with priority
        priority = message.priority.value
        try:
            self._message_queue.put((priority, message), block=False)
        except queue.Full:
            self.logger.warning("Message queue full, dropping message")
            if callback:
                error_response = {
                    "id": message.id,
                    "timestamp": time.time(),
                    "source": "cdfa",
                    "target": "pulsar",
                    "status": "error",
                    "error": "Message queue full",
                    "metadata": {}
                }
                callback(error_response)
                
        return message.id
    
    def send_and_receive(self, message: PulsarMessage, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Send a message and wait for the response.
        
        Args:
            message: Message to send
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Response data
        """
        if timeout is None:
            timeout = message.timeout
            
        # Set up response event
        response_event = threading.Event()
        response_data = [None]
        
        def response_callback(response):
            response_data[0] = response
            response_event.set()
            
        # Send message
        message.response_required = True
        self.send_message(message, callback=response_callback)
        
        # Wait for response
        if response_event.wait(timeout):
            return response_data[0]
        else:
            # Timeout occurred
            with self._lock:
                self._response_callbacks.pop(message.id, None)
                
            raise TimeoutError(f"No response received within {timeout} seconds")
    
    # ----- High-level component access methods -----
    
    def query_q_star(self, state: Dict[str, Any], action_space: Optional[List[str]] = None, 
                    timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Query the Q* reinforcement learning component.
        
        Args:
            state: Current state representation
            action_space: Optional list of available actions
            timeout: Optional timeout override
            
        Returns:
            Q* response with recommended action
        """
        message = PulsarMessage(
            component="q_star",
            action="get_action",
            data={
                "state": state,
                "action_space": action_space
            }
        )
        
        return self.send_and_receive(message, timeout)
    
    def train_q_star(self, state: Dict[str, Any], action: str, reward: float, 
                    next_state: Dict[str, Any], done: bool = False) -> Dict[str, Any]:
        """
        Train the Q* reinforcement learning component.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether the episode is done
            
        Returns:
            Training result
        """
        message = PulsarMessage(
            component="q_star",
            action="train",
            data={
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }
        )
        
        return self.send_and_receive(message)
    
    def query_river_ml(self, features: Dict[str, Any], model_name: Optional[str] = None,
                      timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Query the River online machine learning component.
        
        Args:
            features: Input features
            model_name: Optional specific model to query
            timeout: Optional timeout override
            
        Returns:
            Prediction result
        """
        message = PulsarMessage(
            component="river",
            action="predict",
            data={
                "features": features,
                "model_name": model_name
            }
        )
        
        return self.send_and_receive(message, timeout)
    
    def train_river_ml(self, features: Dict[str, Any], target: Any, 
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the River online machine learning component.
        
        Args:
            features: Input features
            target: Target value
            model_name: Optional specific model to train
            
        Returns:
            Training result
        """
        message = PulsarMessage(
            component="river",
            action="learn",
            data={
                "features": features,
                "target": target,
                "model_name": model_name
            }
        )
        
        return self.send_and_receive(message)
    
    def query_cerebellum(self, pattern: List[float], pattern_type: Optional[str] = None,
                       timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Query the Cerebellar SNN for pattern recognition.
        
        Args:
            pattern: Input pattern
            pattern_type: Optional pattern type
            timeout: Optional timeout override
            
        Returns:
            Pattern recognition result
        """
        message = PulsarMessage(
            component="cerebellum",
            action="recognize_pattern",
            data={
                "pattern": pattern,
                "pattern_type": pattern_type
            }
        )
        
        return self.send_and_receive(message, timeout)
    
    def train_cerebellum(self, pattern: List[float], label: str, 
                       pattern_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the Cerebellar SNN with a new pattern.
        
        Args:
            pattern: Input pattern
            label: Pattern label
            pattern_type: Optional pattern type
            
        Returns:
            Training result
        """
        message = PulsarMessage(
            component="cerebellum",
            action="learn_pattern",
            data={
                "pattern": pattern,
                "label": label,
                "pattern_type": pattern_type
            }
        )
        
        return self.send_and_receive(message)

    # Add to cdfa_extensions/pulsar_connector.py
    
    def analyze_market_narrative(self, symbols: Union[str, List[str]], 
                                timeframe: str = "medium",
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive market narrative analysis using Pulsar's NLP system.
        
        Args:
            symbols: Symbol or list of symbols to analyze
            timeframe: Analysis timeframe ('short', 'medium', 'long')
            context: Additional context for the analysis
            
        Returns:
            Narrative analysis results
        """
        # Format symbols list
        if isinstance(symbols, str):
            symbols = [symbols]
            
        symbols_str = ", ".join(symbols)
        
        # Create context if not provided
        if context is None:
            context = {}
            
        # Add required context
        context.update({
            "symbols": symbols,
            "timeframe": timeframe,
            "analysis_type": "market_narrative",
            "request_time": datetime.datetime.now().isoformat()
        })
        
        # Query narrative forecaster with specific request
        query = f"Provide a comprehensive market narrative analysis for these symbols: {symbols_str}"
        
        result = self.query_narrative_forecaster(query, context=context)
        
        if not result or "error" in result:
            self.logger.error(f"Error from Narrative Forecaster: {result.get('error', 'Unknown error')}")
            return {"error": "Failed to get narrative analysis from Pulsar"}
            
        return result

    
    def query_narrative_forecaster(self, text: str, context: Optional[Dict[str, Any]] = None,
                                 timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Query the Narrative Forecaster for market sentiment analysis.
        
        Args:
            text: Input text
            context: Optional additional context
            timeout: Optional timeout override
            
        Returns:
            Narrative analysis result
        """
        message = PulsarMessage(
            component="narrative",
            action="analyze_text",
            data={
                "text": text,
                "context": context or {}
            }
        )
        
        return self.send_and_receive(message, timeout)
    
    def async_query_narrative_forecaster(self, text: str, context: Optional[Dict[str, Any]] = None,
                                       callback: Optional[Callable] = None) -> str:
        """
        Asynchronously query the Narrative Forecaster.
        
        Args:
            text: Input text
            context: Optional additional context
            callback: Optional callback for response
            
        Returns:
            Message ID
        """
        message = PulsarMessage(
            component="narrative",
            action="analyze_text",
            data={
                "text": text,
                "context": context or {}
            },
            response_required=True
        )
        
        return self.send_message(message, callback)
    
    def optimize_weights(self, weights: Dict[str, float], constraints: Optional[Dict[str, Any]] = None,
                      objective: Optional[str] = "sharpe", timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Use the Quantum Optimizer to optimize weights.
        
        Args:
            weights: Initial weights
            constraints: Optional constraints
            objective: Optimization objective
            timeout: Optional timeout override
            
        Returns:
            Optimized weights
        """
        message = PulsarMessage(
            component="optimizer",
            action="optimize",
            data={
                "weights": weights,
                "constraints": constraints or {},
                "objective": objective
            }
        )
        
        return self.send_and_receive(message, timeout)
