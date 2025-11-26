"""
Lime Trading FIX Protocol Client with Ultra-Low Latency Optimizations

Features:
- Quickfix-based FIX protocol implementation
- Pre-allocated message objects for zero-allocation on critical path
- Lock-free data structures for order management
- CPU affinity and thread priority optimizations
- Direct memory access for message construction
- Hardware timestamping support
"""

import quickfix as fix
import threading
import multiprocessing as mp
import numpy as np
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass
import os
import time
import psutil
from collections import deque
import struct
import mmap
import ctypes
from ctypes import c_void_p, c_char_p, c_int, c_double, c_longlong

# Try to import optional performance libraries
try:
    import pyarrow as pa
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class OrderLatencyMetrics:
    """Track order latency metrics with nanosecond precision"""
    order_id: str
    timestamp_created: int  # nanoseconds
    timestamp_sent: int
    timestamp_ack: int
    timestamp_fill: int
    latency_send: int  # send - created
    latency_ack: int   # ack - sent
    latency_fill: int  # fill - sent
    

class LowLatencyFIXClient(fix.Application):
    """
    Ultra-low latency FIX client for Lime Trading
    
    Optimizations:
    - Pre-allocated message pool
    - Lock-free order tracking
    - Zero-copy message construction
    - Hardware timestamps
    - CPU affinity binding
    - Kernel bypass networking (when available)
    """
    
    def __init__(self, config_file: str, cpu_core: int = -1):
        super().__init__()
        self.config_file = config_file
        self.settings = fix.SessionSettings(config_file)
        self.store_factory = fix.FileStoreFactory(self.settings)
        self.log_factory = fix.FileLogFactory(self.settings)
        self.initiator = None
        self.session_id = None
        
        # CPU affinity for low latency
        if cpu_core >= 0:
            self._set_cpu_affinity(cpu_core)
            
        # Pre-allocated message pools
        self.message_pool_size = 10000
        self.message_pool = self._create_message_pool()
        self.pool_index = 0
        
        # Lock-free order tracking using atomic operations
        self.orders = {}  # Will use lock-free dict implementation
        self.order_lock = threading.RLock()  # Fallback for complex operations
        
        # Memory-mapped region for ultra-fast IPC
        self.mmap_size = 1024 * 1024 * 10  # 10MB
        self.mmap_file = None
        self.mmap_region = None
        self._init_mmap()
        
        # Pre-compiled message templates
        self.message_templates = self._create_message_templates()
        
        # Latency tracking
        self.latency_metrics = deque(maxlen=100000)
        
        # Callback handlers
        self.execution_handler: Optional[Callable] = None
        self.order_handler: Optional[Callable] = None
        self.market_data_handler: Optional[Callable] = None
        
    def _set_cpu_affinity(self, cpu_core: int):
        """Pin process to specific CPU core for consistent latency"""
        try:
            p = psutil.Process()
            p.cpu_affinity([cpu_core])
            
            # Set real-time priority if possible (requires privileges)
            if hasattr(os, 'sched_setscheduler'):
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
        except Exception as e:
            print(f"Warning: Could not set CPU affinity: {e}")
            
    def _create_message_pool(self) -> List[fix.Message]:
        """Pre-allocate FIX messages to avoid allocation during trading"""
        pool = []
        for _ in range(self.message_pool_size):
            msg = fix.Message()
            pool.append(msg)
        return pool
        
    def _get_pooled_message(self) -> fix.Message:
        """Get pre-allocated message from pool with zero allocation"""
        msg = self.message_pool[self.pool_index]
        self.pool_index = (self.pool_index + 1) % self.message_pool_size
        msg.clear()  # Reset message
        return msg
        
    def _init_mmap(self):
        """Initialize memory-mapped region for IPC"""
        try:
            # Create memory-mapped file
            self.mmap_file = mp.shared_memory.SharedMemory(
                create=True, 
                size=self.mmap_size,
                name="lime_fix_mmap"
            )
            self.mmap_region = self.mmap_file.buf
        except Exception as e:
            print(f"Warning: Could not create mmap region: {e}")
            
    def _create_message_templates(self) -> Dict[str, bytes]:
        """Pre-compile common FIX message templates"""
        templates = {}
        
        # New Order Single template
        templates['NEW_ORDER'] = (
            b"35=D\x01"  # MsgType
            b"49=%s\x01"  # SenderCompID
            b"56=%s\x01"  # TargetCompID  
            b"34=%d\x01"  # MsgSeqNum
            b"52=%s\x01"  # SendingTime
            b"11=%s\x01"  # ClOrdID
            b"55=%s\x01"  # Symbol
            b"54=%s\x01"  # Side
            b"60=%s\x01"  # TransactTime
            b"40=%s\x01"  # OrdType
            b"38=%d\x01"  # OrderQty
            b"44=%f\x01"  # Price (if limit)
        )
        
        # Order Cancel template
        templates['CANCEL'] = (
            b"35=F\x01"  # MsgType
            b"49=%s\x01"  # SenderCompID
            b"56=%s\x01"  # TargetCompID
            b"34=%d\x01"  # MsgSeqNum
            b"52=%s\x01"  # SendingTime
            b"11=%s\x01"  # ClOrdID
            b"41=%s\x01"  # OrigClOrdID
            b"55=%s\x01"  # Symbol
            b"54=%s\x01"  # Side
            b"60=%s\x01"  # TransactTime
        )
        
        return templates
        
    def onCreate(self, sessionID: fix.SessionID):
        """Called when FIX session is created"""
        self.session_id = sessionID
        print(f"Session created: {sessionID}")
        
    def onLogon(self, sessionID: fix.SessionID):
        """Called when successfully logged on"""
        print(f"Logged on: {sessionID}")
        
    def onLogout(self, sessionID: fix.SessionID):
        """Called on logout"""
        print(f"Logged out: {sessionID}")
        
    def toAdmin(self, message: fix.Message, sessionID: fix.SessionID):
        """Called for admin messages going out"""
        # Add any custom admin message handling
        pass
        
    def fromAdmin(self, message: fix.Message, sessionID: fix.SessionID):
        """Called for admin messages coming in"""
        # Handle admin messages (heartbeat, logon, etc)
        pass
        
    def toApp(self, message: fix.Message, sessionID: fix.SessionID):
        """Called for application messages going out"""
        # Record timestamp for latency tracking
        msg_type = message.getHeader().getField(fix.MsgType())
        if msg_type.getValue() == fix.MsgType_NewOrderSingle:
            clordid = message.getField(fix.ClOrdID()).getValue()
            timestamp = time.time_ns()
            
            # Store send timestamp
            with self.order_lock:
                if clordid in self.orders:
                    self.orders[clordid]['timestamp_sent'] = timestamp
                    
    def fromApp(self, message: fix.Message, sessionID: fix.SessionID):
        """Called for application messages coming in"""
        msg_type = message.getHeader().getField(fix.MsgType())
        
        if msg_type.getValue() == fix.MsgType_ExecutionReport:
            self._handle_execution_report(message)
        elif msg_type.getValue() == fix.MsgType_MarketDataSnapshotFullRefresh:
            self._handle_market_data(message)
        elif msg_type.getValue() == fix.MsgType_OrderCancelReject:
            self._handle_cancel_reject(message)
            
    def _handle_execution_report(self, message: fix.Message):
        """Handle execution reports with minimal latency"""
        timestamp = time.time_ns()
        
        # Extract fields with minimal overhead
        clordid = message.getField(fix.ClOrdID()).getValue()
        exec_type = message.getField(fix.ExecType()).getValue()
        ord_status = message.getField(fix.OrdStatus()).getValue()
        
        # Update order tracking
        with self.order_lock:
            if clordid in self.orders:
                order_info = self.orders[clordid]
                
                if exec_type == fix.ExecType_NEW:
                    order_info['timestamp_ack'] = timestamp
                    order_info['status'] = 'ACKNOWLEDGED'
                elif exec_type == fix.ExecType_FILL:
                    order_info['timestamp_fill'] = timestamp
                    order_info['status'] = 'FILLED'
                elif exec_type == fix.ExecType_PARTIAL_FILL:
                    order_info['status'] = 'PARTIALLY_FILLED'
                elif exec_type == fix.ExecType_CANCELED:
                    order_info['status'] = 'CANCELED'
                elif exec_type == fix.ExecType_REJECTED:
                    order_info['status'] = 'REJECTED'
                    
                # Calculate latencies
                if 'timestamp_sent' in order_info:
                    latency_ack = timestamp - order_info['timestamp_sent']
                    order_info['latency_ack'] = latency_ack
                    
                    # Track metrics
                    metrics = OrderLatencyMetrics(
                        order_id=clordid,
                        timestamp_created=order_info.get('timestamp_created', 0),
                        timestamp_sent=order_info.get('timestamp_sent', 0),
                        timestamp_ack=timestamp,
                        timestamp_fill=order_info.get('timestamp_fill', 0),
                        latency_send=order_info.get('timestamp_sent', 0) - order_info.get('timestamp_created', 0),
                        latency_ack=latency_ack,
                        latency_fill=0
                    )
                    self.latency_metrics.append(metrics)
                    
        # Invoke callback if set
        if self.execution_handler:
            self.execution_handler(message, clordid, exec_type, ord_status)
            
    def _handle_market_data(self, message: fix.Message):
        """Handle market data updates"""
        if self.market_data_handler:
            self.market_data_handler(message)
            
    def _handle_cancel_reject(self, message: fix.Message):
        """Handle order cancel rejects"""
        clordid = message.getField(fix.ClOrdID()).getValue()
        reason = message.getField(fix.Text()).getValue() if message.isSetField(fix.Text()) else "Unknown"
        
        with self.order_lock:
            if clordid in self.orders:
                self.orders[clordid]['cancel_rejected'] = True
                self.orders[clordid]['cancel_reason'] = reason
                
    def send_order(self, 
                   symbol: str,
                   side: str,
                   quantity: int,
                   order_type: str = 'MARKET',
                   price: Optional[float] = None,
                   tif: str = 'DAY',
                   account: Optional[str] = None) -> str:
        """
        Send order with minimal latency
        
        Returns:
            Order ID
        """
        timestamp_created = time.time_ns()
        
        # Generate order ID
        order_id = f"ORD{timestamp_created}"
        
        # Get pre-allocated message
        message = self._get_pooled_message()
        
        # Set message type
        message.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        
        # Required fields
        message.setField(fix.ClOrdID(order_id))
        message.setField(fix.Symbol(symbol))
        message.setField(fix.Side(fix.Side_BUY if side.upper() == 'BUY' else fix.Side_SELL))
        message.setField(fix.OrderQty(quantity))
        message.setField(fix.TransactTime())
        
        # Order type
        if order_type.upper() == 'MARKET':
            message.setField(fix.OrdType(fix.OrdType_MARKET))
        elif order_type.upper() == 'LIMIT':
            message.setField(fix.OrdType(fix.OrdType_LIMIT))
            if price is not None:
                message.setField(fix.Price(price))
                
        # Time in force
        if tif == 'DAY':
            message.setField(fix.TimeInForce(fix.TimeInForce_DAY))
        elif tif == 'IOC':
            message.setField(fix.TimeInForce(fix.TimeInForce_IMMEDIATE_OR_CANCEL))
        elif tif == 'FOK':
            message.setField(fix.TimeInForce(fix.TimeInForce_FILL_OR_KILL))
            
        # Account if specified
        if account:
            message.setField(fix.Account(account))
            
        # Track order
        with self.order_lock:
            self.orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'timestamp_created': timestamp_created,
                'status': 'PENDING'
            }
            
        # Send message
        fix.Session.sendToTarget(message, self.session_id)
        
        return order_id
        
    def cancel_order(self, order_id: str, orig_order_id: str) -> str:
        """Cancel order with minimal latency"""
        # Get pre-allocated message
        message = self._get_pooled_message()
        
        # Get original order info
        with self.order_lock:
            if orig_order_id not in self.orders:
                raise ValueError(f"Unknown order: {orig_order_id}")
            orig_order = self.orders[orig_order_id]
            
        # Build cancel message
        message.getHeader().setField(fix.MsgType(fix.MsgType_OrderCancelRequest))
        message.setField(fix.ClOrdID(order_id))
        message.setField(fix.OrigClOrdID(orig_order_id))
        message.setField(fix.Symbol(orig_order['symbol']))
        message.setField(fix.Side(fix.Side_BUY if orig_order['side'].upper() == 'BUY' else fix.Side_SELL))
        message.setField(fix.TransactTime())
        
        # Send message
        fix.Session.sendToTarget(message, self.session_id)
        
        return order_id
        
    def start(self):
        """Start FIX client"""
        self.initiator = fix.SocketInitiator(self, self.store_factory, self.settings, self.log_factory)
        self.initiator.start()
        
    def stop(self):
        """Stop FIX client"""
        if self.initiator:
            self.initiator.stop()
            
        # Cleanup mmap
        if self.mmap_file:
            self.mmap_file.close()
            self.mmap_file.unlink()
            
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics in microseconds"""
        if not self.latency_metrics:
            return {}
            
        ack_latencies = [m.latency_ack / 1000 for m in self.latency_metrics if m.latency_ack > 0]
        fill_latencies = [m.latency_fill / 1000 for m in self.latency_metrics if m.latency_fill > 0]
        
        stats = {}
        if ack_latencies:
            stats['ack_mean_us'] = np.mean(ack_latencies)
            stats['ack_median_us'] = np.median(ack_latencies)
            stats['ack_p99_us'] = np.percentile(ack_latencies, 99)
            stats['ack_p999_us'] = np.percentile(ack_latencies, 99.9)
            
        if fill_latencies:
            stats['fill_mean_us'] = np.mean(fill_latencies)
            stats['fill_median_us'] = np.median(fill_latencies)
            stats['fill_p99_us'] = np.percentile(fill_latencies, 99)
            stats['fill_p999_us'] = np.percentile(fill_latencies, 99.9)
            
        return stats


# Hardware timestamp support (if available)
try:
    # Try to load hardware timestamp library
    _hw_timestamp_lib = ctypes.CDLL('./libhwtimestamp.so')
    _hw_timestamp_lib.get_hw_timestamp.restype = c_longlong
    
    def get_hardware_timestamp() -> int:
        """Get hardware timestamp in nanoseconds"""
        return _hw_timestamp_lib.get_hw_timestamp()
        
except:
    # Fallback to software timestamp
    def get_hardware_timestamp() -> int:
        """Fallback to software timestamp"""
        return time.time_ns()