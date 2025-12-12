#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PADS Reporter for CDFA Extensions

Provides a communication interface between CDFA and the Panarchy Adaptive Decision System (PADS),
enabling structured reporting of signals, analysis results, and recommendations.

Author: Created on May 6, 2025
"""

import logging
import time
import json
import threading
import queue
import numpy as np
import uuid
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import os
import traceback

# ---- Optional dependencies with graceful fallbacks ----

# Redis for communication
try:
    import redis
    import msgpack
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Direct Redis communication will be disabled.", DeprecationWarning, DeprecationWarning)
    
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
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    warnings.warn("ZeroMQ not available. Direct ZMQ communication will be disabled.", DeprecationWarning, DeprecationWarning)

class SignalType(Enum):
    """Types of signals that can be reported to PADS."""
    MARKET_SIGNAL = auto()        # General market signal
    MARKET_REGIME = auto()        # Market regime identification
    TRADING_SIGNAL = auto()       # Specific trading signal
    RISK_ALERT = auto()           # Risk assessment alert
    CORRELATION_SIGNAL = auto()   # Asset correlation signal
    VOLATILITY_SIGNAL = auto()    # Volatility regime signal
    BLACK_SWAN = auto()           # Black swan event detection
    WHALE_ACTIVITY = auto()       # Whale activity detection
    NARRATIVE_SHIFT = auto()      # Market narrative shift
    CUSTOM = auto()               # Custom signal type

class SignalDirection(Enum):
    """Direction of the signal."""
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    
    @classmethod
    def from_value(cls, value: float) -> 'SignalDirection':
        """Convert a float value to signal direction."""
        if value > 0.6:
            return cls.BULLISH
        elif value < 0.4:
            return cls.BEARISH
        else:
            return cls.NEUTRAL

class SignalConfidence(Enum):
    """Confidence level of the signal."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4
    
    @classmethod
    def from_value(cls, value: float) -> 'SignalConfidence':
        """Convert a float value to confidence level."""
        if value > 0.9:
            return cls.VERY_HIGH
        elif value > 0.7:
            return cls.HIGH
        elif value > 0.5:
            return cls.MEDIUM
        else:
            return cls.LOW

class SignalTimeframe(Enum):
    """Timeframe of the signal."""
    ULTRA_SHORT = auto()  # < 1 hour
    SHORT = auto()        # 1 hour - 1 day
    MEDIUM = auto()       # 1 day - 1 week
    LONG = auto()         # 1 week - 1 month
    VERY_LONG = auto()    # > 1 month

class CommunicationMode(Enum):
    """Communication modes for the PADS reporter."""
    REDIS = auto()
    ZMQ = auto()
    IPC = auto()
    HTTP = auto()
    FILE = auto()     # Write to file
    DIRECT = auto()   # Direct in-memory calls

@dataclass
class PADSSignal:
    """Signal format for communication with PADS."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "cdfa"
    source_component: str = ""  # Source component within CDFA
    symbol: str = ""           # Market symbol
    timeframe: SignalTimeframe = SignalTimeframe.MEDIUM
    signal_type: SignalType = SignalType.MARKET_SIGNAL
    direction: SignalDirection = SignalDirection.NEUTRAL
    strength: float = 0.5      # 0.0 to 1.0
    confidence: SignalConfidence = SignalConfidence.MEDIUM
    data: Dict[str, Any] = field(default_factory=dict)  # Signal-specific data
    related_signals: List[str] = field(default_factory=list)  # Related signal IDs
    priority: int = 1          # 1 (lowest) to 10 (highest)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiration: Optional[float] = None  # Timestamp when signal expires
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
            "source_component": self.source_component,
            "symbol": self.symbol,
            "timeframe": self.timeframe.name if isinstance(self.timeframe, SignalTimeframe) else self.timeframe,
            "signal_type": self.signal_type.name if isinstance(self.signal_type, SignalType) else self.signal_type,
            "direction": self.direction.name if isinstance(self.direction, SignalDirection) else self.direction,
            "strength": self.strength,
            "confidence": self.confidence.name if isinstance(self.confidence, SignalConfidence) else self.confidence,
            "data": self.data,
            "related_signals": self.related_signals,
            "priority": self.priority,
            "tags": self.tags,
            "metadata": self.metadata,
            "expiration": self.expiration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PADSSignal':
        """Create signal from dictionary."""
        # Convert enum strings to enums
        if "timeframe" in data and isinstance(data["timeframe"], str):
            data["timeframe"] = SignalTimeframe[data["timeframe"]]
            
        if "signal_type" in data and isinstance(data["signal_type"], str):
            data["signal_type"] = SignalType[data["signal_type"]]
            
        if "direction" in data and isinstance(data["direction"], str):
            data["direction"] = SignalDirection[data["direction"]]
            
        if "confidence" in data and isinstance(data["confidence"], str):
            data["confidence"] = SignalConfidence[data["confidence"]]
            
        return cls(**data)

@dataclass
class PADSFeedback:
    """Feedback from PADS to CDFA."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "pads"
    target: str = "cdfa"
    signal_id: str = ""  # ID of the original signal
    feedback_type: str = ""  # Type of feedback
    value: float = 0.0     # Feedback value (e.g., score)
    comments: str = ""     # Optional comments
    data: Dict[str, Any] = field(default_factory=dict)  # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
            "target": self.target,
            "signal_id": self.signal_id,
            "feedback_type": self.feedback_type,
            "value": self.value,
            "comments": self.comments,
            "data": self.data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PADSFeedback':
        """Create feedback from dictionary."""
        return cls(**data)

class PADSReporter:
    """
    Communication interface with the Panarchy Adaptive Decision System (PADS).
    
    This reporter handles structured reporting of signals, analysis results, and
    recommendations from CDFA to PADS, as well as processing feedback from PADS.
    """
    
    def __init__(self, mode: Union[str, CommunicationMode] = "redis",
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the PADS reporter.
        
        Args:
            mode: Communication mode ('redis', 'zmq', 'ipc', 'http', 'file', 'direct')
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
            "redis_channel_prefix": "pads:",
            
            # ZMQ configuration
            "zmq_address": "tcp://localhost:5556",
            "zmq_timeout": 5000,  # ms
            
            # IPC configuration
            "ipc_socket_path": "/tmp/pads_socket",
            
            # HTTP configuration
            "http_url": "http://localhost:8001/pads",
            "http_timeout": 30,  # seconds
            
            # File configuration
            "file_path": "/tmp/pads_signals",
            "file_format": "jsonl",  # jsonl or msgpack
            "file_rotate_size": 10*1024*1024,  # 10 MB
            "file_rotate_count": 5,
            
            # General configuration
            "max_queue_size": 1000,
            "flush_interval": 1.0,  # seconds
            "compress_messages": True,
            "batch_size": 10,  # Number of signals to send in one batch
            "history_size": 1000,  # Number of signals to keep in history
            "feedback_enabled": True,
            "include_metadata": True,
            "auto_reconnect": True,
            "reconnect_interval": 5.0,  # seconds
            "signal_history_ttl": 86400  # 24 hours
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize state
        self._is_connected = False
        self._is_running = True
        self._lock = threading.RLock()
        self._signal_queue = queue.PriorityQueue(maxsize=self.config["max_queue_size"])
        self._signal_history = {}  # id -> signal
        self._signal_timestamps = []  # [(timestamp, id), ...] for TTL management
        self._feedback_callbacks = {}  # signal_type -> [callback, ...]
        self._feedback_history = {}  # id -> feedback
        
        # Initialize communication clients
        self._redis_client = None
        self._zmq_context = None
        self._zmq_socket = None
        self._http_session = None
        self._file_handle = None
        self._file_size = 0
        
        # Initialize background threads
        self._flush_thread = None
        self._feedback_thread = None
        self._cleanup_thread = None
        
        # Start background threads
        self._start_background_threads()
        
        # Try to connect
        self.connect()
        
    def _start_background_threads(self):
        """Start background processing threads."""
        # Flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            daemon=True,
            name="PADSReporterFlushWorker"
        )
        self._flush_thread.start()
        
        # Feedback thread for Redis
        if self.config["feedback_enabled"] and self.mode == CommunicationMode.REDIS:
            self._feedback_thread = threading.Thread(
                target=self._feedback_worker,
                daemon=True,
                name="PADSReporterFeedbackWorker"
            )
            self._feedback_thread.start()
            
        # Cleanup thread for history TTL
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="PADSReporterCleanupWorker"
        )
        self._cleanup_thread.start()
    
    def _flush_worker(self):
        """Background worker for flushing signal queue."""
        self.logger.info("Starting flush worker thread")
        
        while self._is_running:
            try:
                batch = []
                # Collect batch of signals
                while len(batch) < self.config["batch_size"]:
                    try:
                        # Get next signal from queue
                        priority, signal = self._signal_queue.get(block=True, timeout=self.config["flush_interval"])
                        batch.append(signal)
                        self._signal_queue.task_done()
                    except queue.Empty:
                        break
                
                # If we have signals to send
                if batch:
                    try:
                        # Send batch
                        self._send_signals(batch)
                    except Exception as e:
                        self.logger.error(f"Error sending signals: {e}")
                        # Put signals back in queue
                        for signal in batch:
                            try:
                                priority = 11 - signal.priority  # Invert for priority queue
                                self._signal_queue.put((priority, signal), block=False)
                            except queue.Full:
                                self.logger.error(f"Signal queue full, dropping signal: {signal.id}")
                
            except Exception as e:
                self.logger.error(f"Error in flush worker: {e}")
                time.sleep(0.1)
                
        self.logger.info("Flush worker thread stopped")
    
    def _feedback_worker(self):
        """Background worker for processing feedback from PADS."""
        self.logger.info("Starting feedback worker thread")
        
        if not REDIS_AVAILABLE:
            self.logger.error("Redis not available for feedback worker")
            return
            
        # Initialize Redis PubSub
        pubsub = None
        
        while self._is_running:
            try:
                # Check if Redis is connected
                if not self._is_connected or self._redis_client is None:
                    time.sleep(self.config["reconnect_interval"])
                    continue
                    
                # Initialize PubSub if needed
                if pubsub is None:
                    pubsub = self._redis_client.pubsub(ignore_subscribe_messages=True)
                    feedback_channel = f"{self.config['redis_channel_prefix']}feedback"
                    pubsub.subscribe(feedback_channel)
                    
                # Get message with timeout
                message = pubsub.get_message(timeout=1.0)
                if message:
                    try:
                        # Process feedback
                        data = message.get("data")
                        if data:
                            # Decode data with MessagePack
                            try:
                                data = msgpack.unpackb(data, raw=False)
                                self._handle_feedback(data)
                            except Exception as e:
                                self.logger.error(f"Error decoding feedback: {e}")
                    except Exception as e:
                        self.logger.error(f"Error handling feedback: {e}")
                
            except redis.RedisError as e:
                self.logger.error(f"Redis error in feedback worker: {e}")
                pubsub = None
                time.sleep(self.config["reconnect_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in feedback worker: {e}")
                time.sleep(0.1)
                
        self.logger.info("Feedback worker thread stopped")
    
    def _cleanup_worker(self):
        """Background worker for cleaning up signal history."""
        self.logger.info("Starting cleanup worker thread")
        
        while self._is_running:
            try:
                # Sleep for 5 minutes
                time.sleep(300)
                
                # Cleanup expired signals
                self._cleanup_expired_signals()
                
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                time.sleep(10)
                
        self.logger.info("Cleanup worker thread stopped")
    
    def _cleanup_expired_signals(self):
        """Remove expired signals from history."""
        with self._lock:
            # Current time
            current_time = time.time()
            ttl = self.config["signal_history_ttl"]
            
            # Find expired signals
            cutoff_time = current_time - ttl
            expired_count = 0
            
            # Find index of first non-expired timestamp
            keep_index = 0
            for i, (timestamp, signal_id) in enumerate(self._signal_timestamps):
                if timestamp >= cutoff_time:
                    keep_index = i
                    break
                    
                # Remove from history
                self._signal_history.pop(signal_id, None)
                expired_count += 1
                
            # Remove expired timestamps
            if keep_index > 0:
                self._signal_timestamps = self._signal_timestamps[keep_index:]
                
            # Log result
            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired signals")
    
    def connect(self) -> bool:
        """
        Connect to PADS using the configured mode.
        
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
                elif self.mode == CommunicationMode.FILE:
                    success = self._connect_file()
                elif self.mode == CommunicationMode.DIRECT:
                    success = self._connect_direct()
                else:
                    self.logger.error(f"Unsupported communication mode: {self.mode}")
                    return False
                    
                self._is_connected = success
                return success
                
        except Exception as e:
            self.logger.error(f"Error connecting to PADS: {e}")
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
            
            self.logger.info(f"Connected to Redis at {self.config['redis_host']}:{self.config['redis_port']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def _connect_zmq(self) -> bool:
        """Connect using ZeroMQ."""
        if not ZMQ_AVAILABLE:
            self.logger.error("ZeroMQ is not available")
            return False
            
        try:
            # Create ZeroMQ context and socket
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
            
            # Set timeout
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
            self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
            
            # Set timeout
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
    
    def _connect_file(self) -> bool:
        """Connect using file output."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config["file_path"]), exist_ok=True)
            
            # Determine current file
            base_path = self.config["file_path"]
            extension = ".jsonl" if self.config["file_format"] == "jsonl" else ".msgpack"
            file_path = f"{base_path}{extension}"
            
            # Check if file exists and get size
            if os.path.exists(file_path):
                self._file_size = os.path.getsize(file_path)
                
                # Rotate if needed
                if self._file_size >= self.config["file_rotate_size"]:
                    self._rotate_file(file_path)
            else:
                self._file_size = 0
                
            # Open file in append mode
            self._file_handle = open(file_path, "ab")
            
            self.logger.info(f"Connected to file at {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to file: {e}")
            return False
    
    def _rotate_file(self, file_path: str):
        """Rotate signal log file."""
        try:
            # Close current file if open
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
                
            # Rotate files
            max_count = self.config["file_rotate_count"]
            
            # Remove oldest file if it exists
            oldest_file = f"{file_path}.{max_count}"
            if os.path.exists(oldest_file):
                os.remove(oldest_file)
                
            # Shift existing files
            for i in range(max_count - 1, 0, -1):
                old_file = f"{file_path}.{i}"
                new_file = f"{file_path}.{i+1}"
                if os.path.exists(old_file):
                    os.rename(old_file, new_file)
                    
            # Rename current file
            if os.path.exists(file_path):
                os.rename(file_path, f"{file_path}.1")
                
            # Reset file size
            self._file_size = 0
            
            self.logger.info(f"Rotated signal log file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error rotating file: {e}")
    
    def _connect_direct(self) -> bool:
        """Connect using direct in-memory calls."""
        try:
            # Try to import PADS module
            try:
                import pads
                self._pads_module = pads
                
                # Check for required components
                if not hasattr(pads, "add_signal"):
                    self.logger.warning("PADS module missing add_signal method")
                
                if self.config["feedback_enabled"] and not hasattr(pads, "register_feedback_handler"):
                    self.logger.warning("PADS module missing register_feedback_handler method")
                    
                # Register feedback handler if enabled
                if self.config["feedback_enabled"] and hasattr(pads, "register_feedback_handler"):
                    pads.register_feedback_handler(self._handle_direct_feedback)
                    
                self.logger.info("Connected to PADS module directly")
                return True
                
            except ImportError:
                self.logger.error("PADS module not found for direct connection")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect directly to PADS: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PADS."""
        with self._lock:
            self._is_running = False
            
            # Clean up resources based on mode
            if self.mode == CommunicationMode.REDIS and self._redis_client:
                try:
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
                    
            elif self.mode == CommunicationMode.FILE and self._file_handle:
                try:
                    self._file_handle.close()
                except:
                    pass
                    
            self._is_connected = False
            self.logger.info("Disconnected from PADS")
            
    def is_connected(self) -> bool:
        """
        Check if connected to PADS.
        
        Returns:
            Connection status
        """
        return self._is_connected
    
    def _send_signals(self, signals: List[PADSSignal]):
        """Send signals to PADS."""
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError("Not connected to PADS")
                
        # Convert signals to dictionaries
        signal_dicts = [signal.to_dict() for signal in signals]
        
        # Add to signal history
        with self._lock:
            current_time = time.time()
            for i, signal in enumerate(signals):
                self._signal_history[signal.id] = signal
                self._signal_timestamps.append((current_time, signal.id))
                
        # Send based on communication mode
        if self.mode == CommunicationMode.REDIS:
            self._send_redis(signal_dicts)
        elif self.mode in (CommunicationMode.ZMQ, CommunicationMode.IPC):
            self._send_zmq(signal_dicts)
        elif self.mode == CommunicationMode.HTTP:
            self._send_http(signal_dicts)
        elif self.mode == CommunicationMode.FILE:
            self._send_file(signal_dicts)
        elif self.mode == CommunicationMode.DIRECT:
            self._send_direct(signal_dicts)
        else:
            raise ValueError(f"Unsupported communication mode: {self.mode}")
            
        self.logger.debug(f"Sent {len(signals)} signals to PADS")
    
    def _send_redis(self, signal_dicts: List[Dict[str, Any]]):
        """Send signals via Redis."""
        if not self._redis_client:
            raise ConnectionError("Redis client not initialized")
            
        try:
            # Determine channel
            channel = f"{self.config['redis_channel_prefix']}signals"
            
            # Encode message with MessagePack
            packed = msgpack.packb(signal_dicts, use_bin_type=True)
            
            # Publish message
            self._redis_client.publish(channel, packed)
            
        except Exception as e:
            self.logger.error(f"Error sending signals via Redis: {e}")
            # Mark as disconnected to trigger reconnect
            self._is_connected = False
            raise
    
    def _send_zmq(self, signal_dicts: List[Dict[str, Any]]):
        """Send signals via ZeroMQ."""
        if not self._zmq_socket:
            raise ConnectionError("ZeroMQ socket not initialized")
            
        try:
            # Encode message with MessagePack
            packed = msgpack.packb(signal_dicts, use_bin_type=True)
            
            # Send message
            self._zmq_socket.send(packed)
            
        except zmq.ZMQError as e:
            self.logger.error(f"ZeroMQ error: {e}")
            # Mark as disconnected to trigger reconnect
            self._is_connected = False
            raise
            
        except Exception as e:
            self.logger.error(f"Error sending signals via ZeroMQ: {e}")
            raise
    
    def _send_http(self, signal_dicts: List[Dict[str, Any]]):
        """Send signals via HTTP."""
        if not self._http_session:
            raise ConnectionError("HTTP session not initialized")
            
        try:
            # Send request
            response = self._http_session.post(
                f"{self.config['http_url']}/signals",
                json=signal_dicts,
                timeout=self.config["http_timeout"]
            )
            
            # Check response
            if response.status_code != 200:
                self.logger.error(f"HTTP request failed with status code {response.status_code}")
                raise RuntimeError(f"HTTP request failed: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending signals via HTTP: {e}")
            # Mark as disconnected to trigger reconnect
            self._is_connected = False
            raise
    
    def _send_file(self, signal_dicts: List[Dict[str, Any]]):
        """Send signals to file."""
        if not self._file_handle:
            raise ConnectionError("File not initialized")
            
        try:
            # Write signals to file based on format
            if self.config["file_format"] == "jsonl":
                # Write as JSONL
                for signal_dict in signal_dicts:
                    line = json.dumps(signal_dict) + "\n"
                    data = line.encode("utf-8")
                    self._file_handle.write(data)
                    self._file_size += len(data)
            else:
                # Write as MessagePack
                for signal_dict in signal_dicts:
                    packed = msgpack.packb(signal_dict, use_bin_type=True)
                    self._file_handle.write(packed)
                    self._file_size += len(packed)
                    
            # Flush to disk
            self._file_handle.flush()
            
            # Check if rotation needed
            if self._file_size >= self.config["file_rotate_size"]:
                base_path = self.config["file_path"]
                extension = ".jsonl" if self.config["file_format"] == "jsonl" else ".msgpack"
                file_path = f"{base_path}{extension}"
                self._rotate_file(file_path)
                
                # Open new file
                self._file_handle = open(file_path, "ab")
                
        except Exception as e:
            self.logger.error(f"Error sending signals to file: {e}")
            # Mark as disconnected to trigger reconnect
            self._is_connected = False
            raise
    
    def _send_direct(self, signal_dicts: List[Dict[str, Any]]):
        """Send signals via direct in-memory call."""
        if not hasattr(self, '_pads_module') or self._pads_module is None:
            raise ConnectionError("PADS module not initialized")
            
        try:
            # Call add_signal method
            if hasattr(self._pads_module, "add_signals"):
                # Batch add
                self._pads_module.add_signals(signal_dicts)
            elif hasattr(self._pads_module, "add_signal"):
                # Add one by one
                for signal_dict in signal_dicts:
                    self._pads_module.add_signal(signal_dict)
            else:
                raise NotImplementedError("PADS module does not have add_signal(s) method")
                
        except Exception as e:
            self.logger.error(f"Error sending signals directly to PADS: {e}")
            raise
    
    def _handle_feedback(self, feedback_data: Dict[str, Any]):
        """Handle feedback from PADS."""
        try:
            # Convert to feedback object
            feedback = PADSFeedback.from_dict(feedback_data)
            
            # Store feedback
            with self._lock:
                self._feedback_history[feedback.id] = feedback
                
            # Get original signal
            signal = self.get_signal(feedback.signal_id)
            
            # Call callbacks
            if signal:
                self._call_feedback_callbacks(signal.signal_type, signal, feedback)
                
        except Exception as e:
            self.logger.error(f"Error handling feedback: {e}")
    
    def _handle_direct_feedback(self, feedback_data: Dict[str, Any]):
        """Handle direct feedback from PADS module."""
        self._handle_feedback(feedback_data)
    
    def _call_feedback_callbacks(self, signal_type: SignalType, signal: PADSSignal, feedback: PADSFeedback):
        """Call registered feedback callbacks."""
        with self._lock:
            # Get callbacks for this signal type
            callbacks = self._feedback_callbacks.get(signal_type, [])
            
            # Also get generic callbacks
            generic_callbacks = self._feedback_callbacks.get(None, [])
            
            # Combine callbacks
            all_callbacks = callbacks + generic_callbacks
            
        # Call callbacks
        for callback in all_callbacks:
            try:
                callback(signal, feedback)
            except Exception as e:
                self.logger.error(f"Error in feedback callback: {e}")
    
    def register_feedback_callback(self, callback: Callable[[PADSSignal, PADSFeedback], None], 
                                signal_type: Optional[SignalType] = None):
        """
        Register a callback for feedback from PADS.
        
        Args:
            callback: Callback function to call when feedback is received
            signal_type: Optional signal type to filter feedback
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
            
        with self._lock:
            if signal_type not in self._feedback_callbacks:
                self._feedback_callbacks[signal_type] = []
                
            self._feedback_callbacks[signal_type].append(callback)
    
    def unregister_feedback_callback(self, callback: Callable[[PADSSignal, PADSFeedback], None], 
                                  signal_type: Optional[SignalType] = None):
        """
        Unregister a feedback callback.
        
        Args:
            callback: Callback function to unregister
            signal_type: Optional signal type to unregister from
        """
        with self._lock:
            if signal_type in self._feedback_callbacks:
                callbacks = self._feedback_callbacks[signal_type]
                if callback in callbacks:
                    callbacks.remove(callback)
                    
                # Remove empty list
                if not callbacks:
                    del self._feedback_callbacks[signal_type]
    
    def get_signal(self, signal_id: str) -> Optional[PADSSignal]:
        """
        Get a signal from history.
        
        Args:
            signal_id: ID of the signal
            
        Returns:
            Signal or None if not found
        """
        with self._lock:
            return self._signal_history.get(signal_id)
    
    def get_feedback(self, feedback_id: str) -> Optional[PADSFeedback]:
        """
        Get feedback from history.
        
        Args:
            feedback_id: ID of the feedback
            
        Returns:
            Feedback or None if not found
        """
        with self._lock:
            return self._feedback_history.get(feedback_id)
    
    def report_signal(self, signal: PADSSignal):
        """
        Report a signal to PADS.
        
        Args:
            signal: Signal to report
        """
        # Check if queue is full
        if self._signal_queue.full():
            self.logger.warning("Signal queue full")
            return
            
        # Calculate priority for queue (invert for priority queue)
        queue_priority = 11 - signal.priority
        
        # Add to queue
        try:
            self._signal_queue.put((queue_priority, signal), block=False)
        except queue.Full:
            self.logger.warning(f"Signal queue full, dropping signal: {signal.id}")
    
    # ----- Convenience methods for creating signals -----
    
    def create_market_signal(self, symbol: str, value: float, confidence: float = 0.7, 
                          timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                          source_component: str = "", data: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a market signal.
        
        Args:
            symbol: Market symbol
            value: Signal value (0.0 to 1.0)
            confidence: Signal confidence (0.0 to 1.0)
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine direction
        direction = SignalDirection.from_value(value)
        
        # Determine confidence level
        confidence_level = SignalConfidence.from_value(confidence)
        
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.MARKET_SIGNAL,
            direction=direction,
            strength=value,
            confidence=confidence_level,
            data=data or {},
            tags=tags or []
        )
        
        return signal
    
    def create_trading_signal(self, symbol: str, action: str, strength: float, confidence: float = 0.7,
                           timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                           source_component: str = "", data: Optional[Dict[str, Any]] = None,
                           tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a trading signal.
        
        Args:
            symbol: Market symbol
            action: Trading action ('buy', 'sell', 'hold')
            strength: Signal strength (0.0 to 1.0)
            confidence: Signal confidence (0.0 to 1.0)
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine direction based on action
        if action.lower() == "buy":
            direction = SignalDirection.BULLISH
            value = 0.5 + (strength * 0.5)  # 0.5 to 1.0
        elif action.lower() == "sell":
            direction = SignalDirection.BEARISH
            value = 0.5 - (strength * 0.5)  # 0.0 to 0.5
        else:  # hold
            direction = SignalDirection.NEUTRAL
            value = 0.5
            
        # Determine confidence level
        confidence_level = SignalConfidence.from_value(confidence)
        
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.TRADING_SIGNAL,
            direction=direction,
            strength=strength,
            confidence=confidence_level,
            data={
                "action": action,
                "value": value,
                **(data or {})
            },
            tags=tags or [],
            priority=3  # Higher priority for trading signals
        )
        
        return signal
    
    def create_market_regime_signal(self, symbol: str, regime: str, transition_probability: float = 0.0,
                                 timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                                 source_component: str = "", data: Optional[Dict[str, Any]] = None,
                                 tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a market regime signal.
        
        Args:
            symbol: Market symbol
            regime: Market regime ('growth', 'conservation', 'release', 'reorganization')
            transition_probability: Probability of regime transition
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine direction based on regime
        regime_lower = regime.lower()
        if regime_lower in ("growth", "reorganization"):
            direction = SignalDirection.BULLISH
            value = 0.7
        elif regime_lower == "conservation":
            direction = SignalDirection.NEUTRAL
            value = 0.5
        elif regime_lower in ("release", "collapse"):
            direction = SignalDirection.BEARISH
            value = 0.3
        else:
            direction = SignalDirection.NEUTRAL
            value = 0.5
            
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.MARKET_REGIME,
            direction=direction,
            strength=value,
            confidence=SignalConfidence.HIGH,
            data={
                "regime": regime,
                "transition_probability": transition_probability,
                **(data or {})
            },
            tags=tags or [],
            priority=2  # Higher priority for regime signals
        )
        
        return signal
    
    def create_risk_alert(self, symbol: str, risk_level: float, risk_type: str,
                       timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                       source_component: str = "", data: Optional[Dict[str, Any]] = None,
                       tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a risk alert signal.
        
        Args:
            symbol: Market symbol
            risk_level: Risk level (0.0 to 1.0)
            risk_type: Type of risk ('liquidity', 'volatility', 'correlation', etc.)
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine direction and priority based on risk level
        if risk_level > 0.7:
            direction = SignalDirection.BEARISH
            priority = 4  # High priority for high risk
            confidence = SignalConfidence.HIGH
        elif risk_level > 0.4:
            direction = SignalDirection.NEUTRAL
            priority = 3
            confidence = SignalConfidence.MEDIUM
        else:
            direction = SignalDirection.BULLISH
            priority = 2
            confidence = SignalConfidence.LOW
            
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.RISK_ALERT,
            direction=direction,
            strength=risk_level,
            confidence=confidence,
            data={
                "risk_type": risk_type,
                "risk_level": risk_level,
                **(data or {})
            },
            tags=tags or [],
            priority=priority
        )
        
        return signal
    
    def create_correlation_signal(self, symbol: str, correlated_symbols: Dict[str, float],
                               timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                               source_component: str = "", data: Optional[Dict[str, Any]] = None,
                               tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a correlation signal.
        
        Args:
            symbol: Market symbol
            correlated_symbols: Dictionary of symbol -> correlation
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Calculate average correlation
        correlations = list(correlated_symbols.values())
        avg_correlation = sum(map(abs, correlations)) / len(correlations) if correlations else 0.0
        
        # Determine direction based on correlation
        # High correlation can be a risk factor
        if avg_correlation > 0.7:
            direction = SignalDirection.BEARISH
            value = 0.3
        elif avg_correlation > 0.4:
            direction = SignalDirection.NEUTRAL
            value = 0.5
        else:
            direction = SignalDirection.BULLISH
            value = 0.7
            
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.CORRELATION_SIGNAL,
            direction=direction,
            strength=value,
            confidence=SignalConfidence.MEDIUM,
            data={
                "correlated_symbols": correlated_symbols,
                "average_correlation": avg_correlation,
                **(data or {})
            },
            tags=tags or []
        )
        
        return signal
    
    def create_volatility_signal(self, symbol: str, volatility: float, vol_regime: str,
                              timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                              source_component: str = "", data: Optional[Dict[str, Any]] = None,
                              tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a volatility signal.
        
        Args:
            symbol: Market symbol
            volatility: Volatility level (0.0 to 1.0)
            vol_regime: Volatility regime ('low', 'medium', 'high', 'extreme')
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine direction and priority based on volatility
        if volatility > 0.7:
            direction = SignalDirection.BEARISH
            priority = 3
            confidence = SignalConfidence.HIGH
        elif volatility > 0.4:
            direction = SignalDirection.NEUTRAL
            priority = 2
            confidence = SignalConfidence.MEDIUM
        else:
            direction = SignalDirection.BULLISH
            priority = 1
            confidence = SignalConfidence.HIGH
            
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.VOLATILITY_SIGNAL,
            direction=direction,
            strength=volatility,
            confidence=confidence,
            data={
                "volatility": volatility,
                "vol_regime": vol_regime,
                **(data or {})
            },
            tags=tags or [],
            priority=priority
        )
        
        return signal
    
    def create_black_swan_signal(self, symbol: str, probability: float, potential_impact: float,
                              timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                              source_component: str = "", data: Optional[Dict[str, Any]] = None,
                              tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a black swan event signal.
        
        Args:
            symbol: Market symbol
            probability: Event probability (0.0 to 1.0)
            potential_impact: Potential impact (0.0 to 1.0)
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine priority based on probability and impact
        risk_score = probability * potential_impact
        
        if risk_score > 0.5:
            priority = 5  # Highest priority
            confidence = SignalConfidence.HIGH
        elif risk_score > 0.2:
            priority = 4
            confidence = SignalConfidence.MEDIUM
        else:
            priority = 3
            confidence = SignalConfidence.LOW
            
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.BLACK_SWAN,
            direction=SignalDirection.BEARISH,  # Black swans are typically negative
            strength=potential_impact,
            confidence=confidence,
            data={
                "probability": probability,
                "potential_impact": potential_impact,
                "risk_score": risk_score,
                **(data or {})
            },
            tags=tags or [],
            priority=priority
        )
        
        return signal
    
    def create_whale_activity_signal(self, symbol: str, activity: float, direction: float,
                                  timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                                  source_component: str = "", data: Optional[Dict[str, Any]] = None,
                                  tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a whale activity signal.
        
        Args:
            symbol: Market symbol
            activity: Activity level (0.0 to 1.0)
            direction: Direction (-1.0 to 1.0, negative for selling, positive for buying)
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine signal direction
        if direction > 0.2:
            signal_direction = SignalDirection.BULLISH
            value = 0.5 + (direction * 0.5)  # 0.5 to 1.0
        elif direction < -0.2:
            signal_direction = SignalDirection.BEARISH
            value = 0.5 + (direction * 0.5)  # 0.0 to 0.5
        else:
            signal_direction = SignalDirection.NEUTRAL
            value = 0.5
            
        # Determine priority based on activity
        if activity > 0.7:
            priority = 4
            confidence = SignalConfidence.HIGH
        elif activity > 0.4:
            priority = 3
            confidence = SignalConfidence.MEDIUM
        else:
            priority = 2
            confidence = SignalConfidence.LOW
            
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.WHALE_ACTIVITY,
            direction=signal_direction,
            strength=activity,
            confidence=confidence,
            data={
                "activity": activity,
                "direction": direction,
                "value": value,
                **(data or {})
            },
            tags=tags or [],
            priority=priority
        )
        
        return signal
    
    def create_narrative_shift_signal(self, symbol: str, sentiment: float, shift_magnitude: float,
                                   timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                                   source_component: str = "", data: Optional[Dict[str, Any]] = None,
                                   tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a narrative shift signal.
        
        Args:
            symbol: Market symbol
            sentiment: Sentiment (-1.0 to 1.0)
            shift_magnitude: Magnitude of narrative shift (0.0 to 1.0)
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Normalize sentiment to 0.0-1.0 range
        norm_sentiment = (sentiment + 1.0) / 2.0
        
        # Determine direction
        signal_direction = SignalDirection.from_value(norm_sentiment)
        
        # Determine priority based on shift magnitude
        if shift_magnitude > 0.7:
            priority = 4
            confidence = SignalConfidence.HIGH
        elif shift_magnitude > 0.4:
            priority = 3
            confidence = SignalConfidence.MEDIUM
        else:
            priority = 2
            confidence = SignalConfidence.LOW
            
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.NARRATIVE_SHIFT,
            direction=signal_direction,
            strength=shift_magnitude,
            confidence=confidence,
            data={
                "sentiment": sentiment,
                "shift_magnitude": shift_magnitude,
                "norm_sentiment": norm_sentiment,
                **(data or {})
            },
            tags=tags or [],
            priority=priority
        )
        
        return signal
    
    def create_custom_signal(self, symbol: str, signal_type: str, value: float,
                          timeframe: SignalTimeframe = SignalTimeframe.MEDIUM,
                          source_component: str = "", data: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> PADSSignal:
        """
        Create a custom signal.
        
        Args:
            symbol: Market symbol
            signal_type: Custom signal type
            value: Signal value (0.0 to 1.0)
            timeframe: Signal timeframe
            source_component: Source component within CDFA
            data: Additional data
            tags: Signal tags
            
        Returns:
            Created signal
        """
        # Determine direction
        direction = SignalDirection.from_value(value)
        
        # Create signal
        signal = PADSSignal(
            source="cdfa",
            source_component=source_component,
            symbol=symbol,
            timeframe=timeframe,
            signal_type=SignalType.CUSTOM,
            direction=direction,
            strength=value,
            confidence=SignalConfidence.MEDIUM,
            data={
                "custom_type": signal_type,
                "value": value,
                **(data or {})
            },
            tags=tags or []
        )
        
        return signal
