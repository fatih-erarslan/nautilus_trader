#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:35:47 2025

@author: ashina
"""
import logging
import time
import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import os
import json
from datetime import datetime, timedelta
import uuid
import tempfile
import redis
import msgpack
try:
    import msgpack_numpy as m
    MSGPACK_AVAILABLE = True
except ImportError:
    m = None
    MSGPACK_AVAILABLE = False
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial


@dataclass
class AdvancedCDFAConfig:
    """Configuration for the Advanced CDFA module"""
    # Hardware acceleration
    use_gpu: bool = True
    gpu_vendor: str = "auto"  # "auto", "nvidia", "amd", "apple"
    torch_device: str = "auto"  # "auto", "cuda", "rocm", "mps", "cpu"
    
    # TorchScript
    use_torchscript: bool = True
    enable_quantization: bool = True
    
    # PyWavelets
    wavelet_family: str = "sym8"
    wavelet_mode: str = "symmetric"
    wavelet_level: int = 4
    
    # Numba
    use_numba: bool = True
    parallel_threshold: int = 1000  # Data size threshold for parallel processing
    
    # Neuromorphic
    use_snn: bool = True
    stdp_learning_rate: float = 0.01
    snn_timesteps: int = 100
    snn_hidden_size: int = 128
    
    # Cross-asset
    max_assets: int = 50
    correlation_window: int = 60
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_channel_prefix: str = "adv_cdfa:"
    pulsar_channel: str = "pulsar:"
    pads_channel: str = "pads:"
    message_ttl: int = 3600  # Seconds
    
    # Performance
    num_threads: int = 8
    cache_size: int = 1000
    
    # Other
    log_level: int = logging.INFO

class RedisConnector:
    """
    Provides communication with external systems via Redis,
    including Pulsar (Q*, River, Cerebellar SNN) and PADS.
    """
    
    def __init__(self, config: AdvancedCDFAConfig):
        self.config = config
        self.logger = logging.getLogger("AdvancedCDFA.RedisConnector")
        
        # Initialize Redis client
        self.redis = self._initialize_redis()
        
        # Set up callbacks
        self.callbacks = {}
        
        # Message queue for outgoing messages
        self.outgoing_queue = []
        
        # Start worker thread if Redis is connected
        if self.redis is not None:
            self.worker_thread = ThreadPoolExecutor(max_workers=1)
            self.worker_thread.submit(self._process_queue)
    
    def get_config_value(self, key, default):
        """
        Helper to get config value from either dict or object configuration
        
        Args:
            key: Configuration key to get
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)
    
    def _initialize_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        try:
            # Connect to Redis
            r = redis.Redis(
                host=self.get_config_value('redis_host', 'localhost'),
                port=self.get_config_value('redis_port', 6379),
                db=self.get_config_value('redis_db', 0),
                password=self.get_config_value('redis_password', None),
                decode_responses=False  # We need binary for msgpack
            )
            
            # Test connection
            r.ping()
            host = self.get_config_value('redis_host', 'localhost')
            port = self.get_config_value('redis_port', 6379)
            self.logger.info(f"Connected to Redis at {host}:{port}")
            return r
        
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return None
        
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Redis: {e}")
            return None
    
    def _process_queue(self):
        """Process outgoing message queue"""
        while True:
            try:
                # Process messages
                if self.outgoing_queue:
                    message = self.outgoing_queue.pop(0)
                    self._publish_message(message["channel"], message["data"])
                
                # Check for incoming messages
                self._check_subscriptions()
                
                # Sleep to prevent high CPU usage
                time.sleep(0.01)
            
            except Exception as e:
                self.logger.error(f"Error in Redis worker thread: {e}")
                time.sleep(1)  # Avoid rapid error loops
    
    def _check_subscriptions(self):
        """Check for messages on subscribed channels"""
        if self.redis is None:
            return
        
        try:
            # Get all subscribed channels
            channels = list(self.callbacks.keys())
            
            # Check each channel
            for channel in channels:
                # Get messages without blocking
                result = self.redis.xread({channel: "0-0"}, count=10, block=0)
                
                if result:
                    # Process messages
                    for stream_name, messages in result:
                        for message_id, data in messages:
                            # Decode message
                            if b"data" in data:
                                try:
                                    message_data = msgpack.unpackb(data[b"data"], raw=False)
                                    
                                    # Call the callback
                                    if channel in self.callbacks:
                                        self.callbacks[channel](message_data)
                                
                                except Exception as e:
                                    self.logger.error(f"Error processing message: {e}")
        
        except Exception as e:
            self.logger.error(f"Error checking subscriptions: {e}")
    
    def _publish_message(self, channel: str, data: Any) -> bool:
        """Publish a message to a Redis channel"""
        if self.redis is None:
            self.logger.warning("Redis not connected, cannot publish message")
            return False
        
        try:
            # Pack the data
            packed_data = msgpack.packb(data, use_bin_type=True)
            
            # Add to Redis stream
            self.redis.xadd(
                channel,
                {"data": packed_data},
                maxlen=1000  # Limit stream length
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error publishing message to {channel}: {e}")
            return False
    
    def publish(self, channel: str, data: Any, priority: int = 0) -> bool:
        """
        Queue a message for publishing to a Redis channel
        
        Args:
            channel: Channel name
            data: Data to publish
            priority: Message priority (higher = more important)
            
        Returns:
            Success flag
        """
        if self.redis is None:
            self.logger.warning("Redis not connected, cannot queue message")
            return False
        
        # Add to outgoing queue
        message = {
            "channel": channel,
            "data": data,
            "priority": priority,
            "timestamp": time.time()
        }
        
        # Insert based on priority
        if priority > 0:
            # Find position based on priority
            for i, msg in enumerate(self.outgoing_queue):
                if msg["priority"] < priority:
                    self.outgoing_queue.insert(i, message)
                    return True
        
        # Add to end of queue
        self.outgoing_queue.append(message)
        return True
    
    def subscribe(self, channel: str, callback: Callable[[Any], None]) -> bool:
        """
        Subscribe to a Redis channel
        
        Args:
            channel: Channel name
            callback: Function to call with received messages
            
        Returns:
            Success flag
        """
        if self.redis is None:
            self.logger.warning("Redis not connected, cannot subscribe")
            return False
        
        try:
            # Create Redis stream if it doesn't exist
            self.redis.xadd(channel, {"init": "true"}, maxlen=1000)
            
            # Register callback
            self.callbacks[channel] = callback
            
            self.logger.info(f"Subscribed to Redis channel: {channel}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error subscribing to {channel}: {e}")
            return False
    
    def unsubscribe(self, channel: str) -> bool:
        """
        Unsubscribe from a Redis channel
        
        Args:
            channel: Channel name
            
        Returns:
            Success flag
        """
        if channel in self.callbacks:
            del self.callbacks[channel]
            self.logger.info(f"Unsubscribed from Redis channel: {channel}")
            return True
        
        return False
    
    def publish_to_pulsar(self, data: Dict[str, Any], component: str = None) -> bool:
        """
        Publish data to the Pulsar system
        
        Args:
            data: Data to publish
            component: Specific Pulsar component (None for main channel)
            
        Returns:
            Success flag
        """
        # Prepare message with metadata
        message = {
            "source": "advanced_cdfa",
            "timestamp": time.time(),
            "data": data
        }
        
        # Get pulsar_channel using the helper
        pulsar_channel = self.get_config_value('pulsar_channel', "pulsar:")
        
        # Add component if specified
        if component:
            message["component"] = component
            channel = f"{pulsar_channel}{component}"
        else:
            channel = pulsar_channel
        
        return self.publish(channel, message, priority=1)
    
    def publish_to_pads(self, signal_type: str, data: Dict[str, Any], 
                         confidence: float = 0.5, priority: int = 0) -> bool:
        """
        Publish a signal to the PADS decision system
        
        Args:
            signal_type: Type of signal (e.g., 'trade', 'regime', 'risk')
            data: Signal data
            confidence: Signal confidence (0-1)
            priority: Message priority
            
        Returns:
            Success flag
        """
        # Prepare message with metadata
        message = {
            "source": "advanced_cdfa",
            "timestamp": time.time(),
            "signal_type": signal_type,
            "confidence": confidence,
            "data": data
        }
        
        # Get pads_channel using helper
        pads_channel = self.get_config_value('pads_channel', "pads:")
        
        # Determine channel based on signal type
        channel = f"{pads_channel}{signal_type}"
        
        return self.publish(channel, message, priority=priority)
    
    def request_from_pulsar(self, query: Dict[str, Any], component: str, 
                             timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Request data from a Pulsar component
        
        Args:
            query: Query data
            component: Pulsar component to query
            timeout: Timeout in seconds
            
        Returns:
            Response data or None if failed
        """
        if self.redis is None:
            self.logger.warning("Redis not connected, cannot request from Pulsar")
            return None
        
        try:
            # Create unique request ID
            request_id = str(uuid.uuid4())
            
            # Get channel prefix using helper
            channel_prefix = self.get_config_value('redis_channel_prefix', "adv_cdfa:")
            
            # Prepare request message
            message = {
                "source": "advanced_cdfa",
                "timestamp": time.time(),
                "request_id": request_id,
                "component": component,
                "query": query,
                "response_channel": f"{channel_prefix}response:{request_id}"
            }
            
            # Create response channel and prepare for receiving response
            response_channel = message["response_channel"]
            response_data = None
            response_received = False
            
            # Create a callback to handle the response
            def response_callback(data):
                nonlocal response_data, response_received
                if data.get("request_id") == request_id:
                    response_data = data.get("response")
                    response_received = True
            
            # Subscribe to response channel
            self.subscribe(response_channel, response_callback)
            
            # Get pulsar_channel using helper
            pulsar_channel = self.get_config_value('pulsar_channel', "pulsar:")
            
            # Publish request
            request_channel = f"{pulsar_channel}{component}"
            success = self.publish(request_channel, message, priority=2)
            
            if not success:
                self.logger.error(f"Failed to publish request to {component}")
                self.unsubscribe(response_channel)
                return None
            
            # Wait for response
            start_time = time.time()
            while not response_received and (time.time() - start_time) < timeout:
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
            
            # Cleanup
            self.unsubscribe(response_channel)
            
            # Check if response was received
            if not response_received:
                self.logger.warning(f"Request to {component} timed out after {timeout} seconds")
                return None
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error requesting data from Pulsar component {component}: {e}")
            return None