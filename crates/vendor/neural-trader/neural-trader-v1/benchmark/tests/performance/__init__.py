"""
Performance validation test suite for AI News Trading benchmark system.

This module contains tests that validate performance targets:
- Signal generation latency < 100ms (P99)
- Throughput > 1000 trades/second
- Memory usage < 2GB for 8-hour simulation
- 100+ concurrent symbols supported
"""

__all__ = [
    'test_latency_targets',
    'test_throughput_targets',
    'test_memory_usage', 
    'test_scalability'
]