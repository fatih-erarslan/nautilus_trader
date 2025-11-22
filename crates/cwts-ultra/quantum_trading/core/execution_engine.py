"""
Execution Engine - High-performance order execution
"""

import asyncio
import logging
import time
from typing import Dict, Any
import numpy as np

class ExecutionEngine:
    """
    High-performance order execution engine
    Handles market orders, limit orders, and advanced order types
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_count = 0
        self.latency_history = []
        
    async def initialize(self):
        """Initialize execution engine"""
        self.logger.info("Execution Engine initialized")
        
    async def execute_order(self, order: Dict[str, Any]) -> bool:
        """Execute order in the market"""
        start_time = time.perf_counter()
        
        try:
            # Simulate market execution
            execution_delay = np.random.exponential(0.001)  # Average 1ms
            await asyncio.sleep(execution_delay)
            
            # Update order with execution details
            order['executed_at'] = time.time()
            order['execution_price'] = order.get('price', 0) * (1 + np.random.normal(0, 0.0001))
            order['status'] = 'EXECUTED'
            
            self.execution_count += 1
            
            # Record latency
            latency = time.perf_counter() - start_time
            self.latency_history.append(latency)
            
            self.logger.debug(f"Order executed: {order.get('order_id')} in {latency*1000:.2f}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return False
            
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.latency_history:
            return {'executions': 0, 'avg_latency_ms': 0}
            
        return {
            'executions': self.execution_count,
            'avg_latency_ms': np.mean(self.latency_history) * 1000,
            'p95_latency_ms': np.percentile(self.latency_history, 95) * 1000
        }