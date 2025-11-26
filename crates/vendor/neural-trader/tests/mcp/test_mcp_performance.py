"""MCP Performance Benchmarking Tests.

Tests MCP server performance, scalability, and resource usage under various load conditions.
"""

import pytest
import asyncio
import json
import time
import statistics
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
import websockets
import psutil
import gc
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from model_management.mcp_integration.trading_mcp_server import TradingMCPServer


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    min_latency: float
    max_latency: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    
    def __str__(self):
        return f"""
Performance Test Results:
------------------------
Total Requests: {self.total_requests}
Successful: {self.successful_requests} ({self.successful_requests/self.total_requests*100:.1f}%)
Failed: {self.failed_requests}
Total Time: {self.total_time:.2f}s
Throughput: {self.throughput:.2f} req/s

Latency Statistics:
  Min: {self.min_latency*1000:.2f}ms
  Avg: {self.avg_latency*1000:.2f}ms
  Median: {self.median_latency*1000:.2f}ms
  P95: {self.p95_latency*1000:.2f}ms
  P99: {self.p99_latency*1000:.2f}ms
  Max: {self.max_latency*1000:.2f}ms

Resource Usage:
  CPU: {self.cpu_usage:.1f}%
  Memory: {self.memory_usage:.1f} MB
"""


class TestMCPPerformance:
    """Test MCP server performance and scalability."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create MCP server for testing."""
        server = TradingMCPServer(
            host="127.0.0.1",
            port=8894,
            model_storage_path="test_models"
        )
        return server
    
    @pytest.fixture
    def test_client(self, mcp_server):
        """Create test client."""
        return TestClient(mcp_server.app)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for consistent testing."""
        model_params = {
            "z_score_entry_threshold": 2.0,
            "base_position_size": 0.05,
            "z_score_exit_threshold": 0.5
        }
        
        metadata = MagicMock()
        metadata.strategy_name = "mean_reversion"
        metadata.parameters = model_params
        metadata.performance_metrics = {
            "sharpe_ratio": 1.85,
            "win_rate": 0.62
        }
        
        return model_params, metadata
    
    def measure_system_resources(self):
        """Measure current system resource usage."""
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_mb": process.memory_info().rss / 1024 / 1024
        }
    
    def calculate_metrics(self, latencies: List[float], total_time: float,
                         successful: int, failed: int,
                         initial_resources: Dict, final_resources: Dict) -> PerformanceMetrics:
        """Calculate performance metrics from test results."""
        if not latencies:
            latencies = [0]
        
        sorted_latencies = sorted(latencies)
        
        return PerformanceMetrics(
            total_requests=successful + failed,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            min_latency=min(latencies) if latencies else 0,
            max_latency=max(latencies) if latencies else 0,
            avg_latency=statistics.mean(latencies) if latencies else 0,
            median_latency=statistics.median(latencies) if latencies else 0,
            p95_latency=sorted_latencies[int(len(sorted_latencies) * 0.95)] if latencies else 0,
            p99_latency=sorted_latencies[int(len(sorted_latencies) * 0.99)] if latencies else 0,
            throughput=successful / total_time if total_time > 0 else 0,
            cpu_usage=final_resources["cpu_percent"],
            memory_usage=final_resources["memory_mb"]
        )
    
    def test_single_request_baseline(self, test_client, mock_model):
        """Test single request performance baseline."""
        model_params, metadata = mock_model
        
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            mock_storage.load_model.return_value = (model_params, metadata)
            
            request_data = {
                "model_id": "test_model",
                "input_data": {
                    "z_score": 2.5,
                    "price": 105.0,
                    "moving_average": 100.0,
                    "volatility": 0.2,
                    "volume_ratio": 1.2,
                    "rsi": 65.0,
                    "market_regime": 0.7
                }
            }
            
            # Warm up
            test_client.post("/models/predict", json=request_data)
            
            # Measure single request
            start_time = time.time()
            response = test_client.post("/models/predict", json=request_data)
            latency = time.time() - start_time
            
            assert response.status_code == 200
            print(f"Single request latency: {latency*1000:.2f}ms")
            
            # Baseline should be fast
            assert latency < 0.1  # Less than 100ms
    
    def test_concurrent_request_handling(self, test_client, mock_model):
        """Test concurrent request handling performance."""
        model_params, metadata = mock_model
        concurrent_levels = [10, 50, 100]
        
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            mock_storage.load_model.return_value = (model_params, metadata)
            
            for num_concurrent in concurrent_levels:
                latencies = []
                successful = 0
                failed = 0
                
                initial_resources = self.measure_system_resources()
                start_time = time.time()
                
                def make_request(i):
                    request_data = {
                        "model_id": "test_model",
                        "input_data": {
                            "z_score": np.random.normal(0, 2),
                            "price": 100 + np.random.normal(0, 5),
                            "moving_average": 100,
                            "volatility": 0.2,
                            "volume_ratio": 1.0,
                            "rsi": 50,
                            "market_regime": 0.5
                        }
                    }
                    
                    req_start = time.time()
                    try:
                        response = test_client.post("/models/predict", json=request_data)
                        req_latency = time.time() - req_start
                        
                        if response.status_code == 200:
                            return True, req_latency
                        else:
                            return False, req_latency
                    except Exception:
                        return False, time.time() - req_start
                
                # Execute concurrent requests
                with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                    futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
                    
                    for future in as_completed(futures):
                        success, latency = future.result()
                        if success:
                            successful += 1
                            latencies.append(latency)
                        else:
                            failed += 1
                
                total_time = time.time() - start_time
                final_resources = self.measure_system_resources()
                
                metrics = self.calculate_metrics(
                    latencies, total_time, successful, failed,
                    initial_resources, final_resources
                )
                
                print(f"\nConcurrent Requests: {num_concurrent}")
                print(metrics)
                
                # Performance assertions
                assert metrics.successful_requests >= num_concurrent * 0.95  # 95% success rate
                assert metrics.avg_latency < 0.5  # Average under 500ms
                assert metrics.throughput > num_concurrent * 0.5  # At least 50% of concurrent level
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, test_client, mock_model):
        """Test performance under sustained load."""
        model_params, metadata = mock_model
        duration_seconds = 10
        requests_per_second = 50
        
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            mock_storage.load_model.return_value = (model_params, metadata)
            
            latencies = []
            successful = 0
            failed = 0
            
            initial_resources = self.measure_system_resources()
            start_time = time.time()
            
            async def make_async_request(session, i):
                request_data = {
                    "model_id": "test_model",
                    "input_data": {
                        "z_score": np.random.normal(0, 2),
                        "price": 100 + np.random.normal(0, 5),
                        "moving_average": 100,
                        "volatility": 0.2,
                        "volume_ratio": 1.0,
                        "rsi": 50,
                        "market_regime": 0.5
                    }
                }
                
                req_start = time.time()
                try:
                    response = await session.post(
                        "http://127.0.0.1:8894/models/predict",
                        json=request_data
                    )
                    req_latency = time.time() - req_start
                    
                    if response.status_code == 200:
                        return True, req_latency
                    else:
                        return False, req_latency
                except Exception:
                    return False, time.time() - req_start
            
            # Generate sustained load
            async with httpx.AsyncClient() as client:
                request_count = 0
                interval = 1.0 / requests_per_second
                
                while time.time() - start_time < duration_seconds:
                    # Launch request
                    task = asyncio.create_task(make_async_request(client, request_count))
                    
                    # Don't wait for completion, just track
                    async def track_result(t):
                        nonlocal successful, failed
                        success, latency = await t
                        if success:
                            successful += 1
                            latencies.append(latency)
                        else:
                            failed += 1
                    
                    asyncio.create_task(track_result(task))
                    request_count += 1
                    
                    # Wait for next request slot
                    await asyncio.sleep(interval)
                
                # Wait for remaining requests to complete
                await asyncio.sleep(2)
            
            total_time = time.time() - start_time
            final_resources = self.measure_system_resources()
            
            metrics = self.calculate_metrics(
                latencies, total_time, successful, failed,
                initial_resources, final_resources
            )
            
            print(f"\nSustained Load Test ({requests_per_second} req/s for {duration_seconds}s)")
            print(metrics)
            
            # Performance assertions for sustained load
            assert metrics.successful_requests > request_count * 0.9  # 90% success rate
            assert metrics.avg_latency < 1.0  # Average under 1 second
            assert metrics.p95_latency < 2.0  # 95th percentile under 2 seconds
    
    def test_memory_leak_detection(self, test_client, mock_model):
        """Test for memory leaks under repeated requests."""
        model_params, metadata = mock_model
        
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            mock_storage.load_model.return_value = (model_params, metadata)
            
            # Measure initial memory
            gc.collect()
            initial_memory = self.measure_system_resources()["memory_mb"]
            
            # Make many requests
            request_data = {
                "model_id": "test_model",
                "input_data": {
                    "z_score": 2.0,
                    "price": 100.0,
                    "moving_average": 100.0,
                    "volatility": 0.2,
                    "volume_ratio": 1.0,
                    "rsi": 50.0,
                    "market_regime": 0.5
                }
            }
            
            for i in range(1000):
                response = test_client.post("/models/predict", json=request_data)
                assert response.status_code == 200
                
                if i % 100 == 0:
                    gc.collect()
                    current_memory = self.measure_system_resources()["memory_mb"]
                    memory_growth = current_memory - initial_memory
                    print(f"After {i} requests: Memory growth = {memory_growth:.2f} MB")
            
            # Final memory check
            gc.collect()
            final_memory = self.measure_system_resources()["memory_mb"]
            total_growth = final_memory - initial_memory
            
            print(f"Total memory growth: {total_growth:.2f} MB")
            
            # Memory growth should be minimal (less than 50MB for 1000 requests)
            assert total_growth < 50
    
    def test_large_payload_handling(self, test_client, mock_model):
        """Test performance with large request payloads."""
        model_params, metadata = mock_model
        payload_sizes = [1, 10, 100]  # KB
        
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            mock_storage.load_model.return_value = (model_params, metadata)
            
            for size_kb in payload_sizes:
                # Create large input data
                num_features = int(size_kb * 1024 / 8)  # 8 bytes per float
                large_data = {
                    f"feature_{i}": float(i) 
                    for i in range(num_features)
                }
                
                request_data = {
                    "model_id": "test_model",
                    "input_data": large_data
                }
                
                # Measure request time
                start_time = time.time()
                response = test_client.post("/models/predict", json=request_data)
                latency = time.time() - start_time
                
                print(f"Payload size: {size_kb}KB, Latency: {latency*1000:.2f}ms")
                
                assert response.status_code == 200
                # Latency should scale reasonably with payload size
                assert latency < size_kb * 0.1  # Less than 100ms per KB
    
    @pytest.mark.asyncio
    async def test_websocket_throughput(self):
        """Test WebSocket message throughput."""
        uri = "ws://127.0.0.1:8892/ws"
        num_messages = 1000
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Measure throughput
                start_time = time.time()
                
                # Send messages
                for i in range(num_messages):
                    message = {
                        "message_type": "heartbeat",
                        "message_id": f"perf_test_{i}",
                        "timestamp": datetime.now().isoformat(),
                        "data": {"index": i}
                    }
                    await websocket.send(json.dumps(message))
                
                # Receive responses
                received = 0
                while received < num_messages:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        received += 1
                    except asyncio.TimeoutError:
                        break
                
                total_time = time.time() - start_time
                throughput = received / total_time
                
                print(f"WebSocket throughput: {throughput:.2f} messages/second")
                print(f"Received {received}/{num_messages} messages")
                
                # Should handle at least 100 messages per second
                assert throughput > 100
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    def test_cache_performance(self, test_client, mock_model):
        """Test model cache performance impact."""
        model_params, metadata = mock_model
        
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            # Simulate slow model loading
            def slow_load(model_id):
                time.sleep(0.1)  # 100ms load time
                return model_params, metadata
            
            mock_storage.load_model = slow_load
            
            request_data = {
                "model_id": "cache_test_model",
                "input_data": {
                    "z_score": 2.0,
                    "price": 100.0,
                    "moving_average": 100.0,
                    "volatility": 0.2,
                    "volume_ratio": 1.0,
                    "rsi": 50.0,
                    "market_regime": 0.5
                }
            }
            
            # First request (cache miss)
            start_time = time.time()
            response = test_client.post("/models/predict", json=request_data)
            first_latency = time.time() - start_time
            assert response.status_code == 200
            
            # Subsequent requests (cache hits)
            cache_latencies = []
            for _ in range(10):
                start_time = time.time()
                response = test_client.post("/models/predict", json=request_data)
                cache_latencies.append(time.time() - start_time)
                assert response.status_code == 200
            
            avg_cache_latency = statistics.mean(cache_latencies)
            
            print(f"First request (cache miss): {first_latency*1000:.2f}ms")
            print(f"Cached requests average: {avg_cache_latency*1000:.2f}ms")
            print(f"Cache speedup: {first_latency/avg_cache_latency:.1f}x")
            
            # Cache should provide significant speedup
            assert avg_cache_latency < first_latency * 0.5  # At least 2x faster
    
    def test_error_handling_performance(self, test_client):
        """Test performance of error handling paths."""
        error_scenarios = [
            # Invalid model ID
            {
                "model_id": "non_existent",
                "input_data": {"x": 1.0}
            },
            # Invalid input data
            {
                "model_id": "test_model",
                "input_data": "invalid"
            },
            # Missing required fields
            {
                "input_data": {"x": 1.0}
            }
        ]
        
        for scenario in error_scenarios:
            start_time = time.time()
            response = test_client.post("/models/predict", json=scenario)
            error_latency = time.time() - start_time
            
            # Error responses should be fast
            assert error_latency < 0.05  # Less than 50ms
            print(f"Error scenario latency: {error_latency*1000:.2f}ms")
    
    def test_scaling_with_model_count(self, test_client, mock_model):
        """Test performance scaling with number of models."""
        model_params, metadata = mock_model
        model_counts = [10, 50, 100]
        
        for count in model_counts:
            with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
                # Simulate multiple models in storage
                mock_storage._model_registry = {
                    f"model_{i}": True for i in range(count)
                }
                mock_storage.load_model.return_value = (model_params, metadata)
                
                # Test list models performance
                start_time = time.time()
                response = test_client.get("/models")
                list_latency = time.time() - start_time
                
                print(f"List {count} models latency: {list_latency*1000:.2f}ms")
                
                # Listing should scale linearly or better
                assert list_latency < count * 0.001  # Less than 1ms per model


class TestMCPStressTests:
    """Stress tests for MCP server limits."""
    
    @pytest.mark.skip(reason="Stress test - run manually")
    @pytest.mark.asyncio
    async def test_maximum_concurrent_connections(self):
        """Test maximum concurrent WebSocket connections."""
        uri = "ws://127.0.0.1:8892/ws"
        max_connections = 1000
        connections = []
        
        try:
            for i in range(max_connections):
                ws = await websockets.connect(uri)
                connections.append(ws)
                
                if i % 100 == 0:
                    print(f"Established {i} connections")
            
            print(f"Successfully established {len(connections)} concurrent connections")
            
            # Cleanup
            for ws in connections:
                await ws.close()
                
        except Exception as e:
            print(f"Connection limit reached at {len(connections)} connections: {e}")
    
    @pytest.mark.skip(reason="Stress test - run manually")
    def test_request_flood(self, test_client):
        """Test server behavior under request flood."""
        flood_size = 10000
        batch_size = 100
        
        request_data = {
            "model_id": "test_model",
            "input_data": {"x": 1.0}
        }
        
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            
            for i in range(0, flood_size, batch_size):
                batch_futures = [
                    executor.submit(test_client.post, "/models/predict", json=request_data)
                    for _ in range(batch_size)
                ]
                futures.extend(batch_futures)
                
                # Small delay between batches
                time.sleep(0.01)
            
            # Wait for all requests
            for future in as_completed(futures):
                try:
                    response = future.result()
                    if response.status_code == 200:
                        successful += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
        
        total_time = time.time() - start_time
        
        print(f"Request flood test: {flood_size} requests in {total_time:.2f}s")
        print(f"Successful: {successful}, Failed: {failed}")
        print(f"Throughput: {successful/total_time:.2f} req/s")