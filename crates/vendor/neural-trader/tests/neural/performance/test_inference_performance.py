"""
Neural Inference Performance Tests

Performance tests for neural forecasting inference including latency,
throughput, and memory efficiency tests.
"""

import pytest
import torch
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import statistics
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Test utilities
from tests.neural.utils.fixtures import (
    basic_nhits_config, high_performance_nhits_config,
    gpu_available, device, benchmark_config, performance_baseline
)
from tests.neural.utils.mock_objects import (
    MockNHITSModel, MockRealTimeEngine, create_mock_nhits_model
)
from tests.neural.utils.gpu_utils import (
    skip_if_no_gpu, require_gpu_memory, GPUMemoryTracker,
    gpu_memory_context, GPUPerformanceBenchmark
)
from tests.neural.utils.performance_utils import (
    LatencyBenchmark, ThroughputBenchmark, MemoryBenchmark,
    BenchmarkConfig, PerformanceMonitor, benchmark_latency,
    assert_performance_regression
)
from tests.neural.utils.data_generators import ModelTestDataGenerator

# Neural components (use mocks if not available)
try:
    from plans.neuralforecast.NHITS_Implementation_Guide import (
        OptimizedNHITS, RealTimeNHITSEngine, MultiAssetNHITSProcessor
    )
    NEURAL_COMPONENTS_AVAILABLE = True
except ImportError:
    NEURAL_COMPONENTS_AVAILABLE = False
    OptimizedNHITS = MockNHITSModel
    RealTimeNHITSEngine = MockRealTimeEngine


class TestInferenceLatency:
    """Test neural inference latency performance."""
    
    def test_single_prediction_latency(self, basic_nhits_config, device):
        """Test latency of single predictions."""
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        # Prepare input
        input_tensor = torch.randn(1, basic_nhits_config.input_size, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Analyze results
        mean_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Single prediction latency ({device.type}):")
        print(f"  Mean: {mean_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms") 
        print(f"  P99: {p99_latency:.2f}ms")
        
        # Performance assertions
        assert mean_latency < 100, f"Mean latency too high: {mean_latency:.2f}ms"
        assert p99_latency < 200, f"P99 latency too high: {p99_latency:.2f}ms"
    
    def test_batch_prediction_latency(self, basic_nhits_config, device):
        """Test latency of batch predictions."""
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        batch_sizes = [1, 4, 16, 32, 64]
        results = {}
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, basic_nhits_config.input_size, device=device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Measure
            times = []
            for _ in range(50):
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            mean_time = statistics.mean(times)
            latency_per_sample = mean_time / batch_size
            
            results[batch_size] = {
                'total_time_ms': mean_time,
                'per_sample_ms': latency_per_sample
            }
            
            print(f"Batch size {batch_size}: {mean_time:.2f}ms total, {latency_per_sample:.2f}ms per sample")
        
        # Check that batching is efficient (per-sample latency should decrease)
        per_sample_latencies = [results[bs]['per_sample_ms'] for bs in batch_sizes]
        
        # Larger batches should be more efficient per sample
        assert per_sample_latencies[0] > per_sample_latencies[-1], \
            "Batching should improve per-sample efficiency"
    
    @skip_if_no_gpu
    def test_gpu_vs_cpu_latency(self, basic_nhits_config):
        """Compare GPU vs CPU inference latency."""
        # CPU model
        cpu_model = OptimizedNHITS(basic_nhits_config)
        cpu_model.eval()
        
        # GPU model
        gpu_model = OptimizedNHITS(basic_nhits_config)
        gpu_model = gpu_model.cuda()
        gpu_model.eval()
        
        batch_size = 16
        cpu_input = torch.randn(batch_size, basic_nhits_config.input_size)
        gpu_input = cpu_input.cuda()
        
        # Benchmark CPU
        cpu_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = cpu_model(cpu_input)
            end_time = time.perf_counter()
            cpu_times.append((end_time - start_time) * 1000)
        
        # Benchmark GPU
        gpu_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = gpu_model(gpu_input)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            gpu_times.append((end_time - start_time) * 1000)
        
        cpu_mean = statistics.mean(cpu_times)
        gpu_mean = statistics.mean(gpu_times)
        speedup = cpu_mean / gpu_mean
        
        print(f"CPU latency: {cpu_mean:.2f}ms")
        print(f"GPU latency: {gpu_mean:.2f}ms")
        print(f"GPU speedup: {speedup:.2f}x")
        
        # GPU should be faster for reasonable batch sizes
        assert gpu_mean < cpu_mean, "GPU should be faster than CPU"
        assert speedup > 1.0, f"GPU speedup too low: {speedup:.2f}x"
    
    def test_real_time_engine_latency(self, basic_nhits_config):
        """Test real-time engine latency."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            # Create and save mock model
            mock_model = MockNHITSModel(basic_nhits_config)
            torch.save(mock_model.state_dict(), tmp_file.name)
            
            engine = RealTimeNHITSEngine(tmp_file.name, basic_nhits_config)
            
            # Test prediction latency
            input_data = np.random.randn(basic_nhits_config.input_size)
            
            # Warmup
            for _ in range(10):
                engine.predict(input_data)
            
            # Measure
            latencies = []
            for _ in range(100):
                start_time = time.perf_counter()
                result = engine.predict(input_data)
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)
                
                # Verify result structure
                assert 'predictions' in result
                assert 'inference_time_ms' in result
            
            mean_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            print(f"Real-time engine latency:")
            print(f"  Mean: {mean_latency:.2f}ms")
            print(f"  P95: {p95_latency:.2f}ms")
            
            # Real-time requirements
            assert mean_latency < 50, f"Real-time latency too high: {mean_latency:.2f}ms"
            assert p95_latency < 100, f"P95 latency too high: {p95_latency:.2f}ms"


class TestInferenceThroughput:
    """Test neural inference throughput performance."""
    
    def test_predictions_per_second(self, basic_nhits_config, device):
        """Test predictions per second throughput."""
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        batch_size = 32
        test_duration = 5.0  # seconds
        
        def data_generator():
            while True:
                return torch.randn(batch_size, basic_nhits_config.input_size, device=device)
        
        # Throughput test
        benchmark = ThroughputBenchmark(BenchmarkConfig())
        
        def inference_func(data):
            with torch.no_grad():
                return model(data)
        
        stats = benchmark.benchmark_throughput(
            inference_func, 
            data_generator, 
            duration_seconds=test_duration
        )
        
        print(f"Inference throughput ({device.type}):")
        print(f"  Predictions/sec: {stats['items_per_second']:.0f}")
        print(f"  Operations/sec: {stats['operations_per_second']:.0f}")
        print(f"  Error rate: {stats['error_rate']:.3f}")
        
        # Performance requirements
        assert stats['items_per_second'] > 100, \
            f"Throughput too low: {stats['items_per_second']:.0f} predictions/sec"
        assert stats['error_rate'] < 0.01, \
            f"Error rate too high: {stats['error_rate']:.3f}"
    
    def test_multi_asset_throughput(self, basic_nhits_config):
        """Test multi-asset processing throughput.""" 
        from tests.neural.utils.mock_objects import MockMultiAssetProcessor
        
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        processor = MockMultiAssetProcessor(assets, basic_nhits_config)
        
        async def throughput_test():
            """Async throughput test."""
            start_time = time.time()
            completed_batches = 0
            test_duration = 10.0
            
            while time.time() - start_time < test_duration:
                # Generate data for all assets
                asset_data = {
                    asset: np.random.randn(basic_nhits_config.batch_size, basic_nhits_config.input_size)
                    for asset in assets
                }
                
                # Process batch
                results = await processor.process_batch(asset_data)
                
                assert len(results) == len(assets)
                completed_batches += 1
            
            actual_duration = time.time() - start_time
            batches_per_sec = completed_batches / actual_duration
            assets_per_sec = batches_per_sec * len(assets)
            
            return {
                'batches_per_sec': batches_per_sec,
                'assets_per_sec': assets_per_sec,
                'completed_batches': completed_batches
            }
        
        # Run async test
        import asyncio
        stats = asyncio.run(throughput_test())
        
        print(f"Multi-asset throughput:")
        print(f"  Assets/sec: {stats['assets_per_sec']:.1f}")
        print(f"  Batches/sec: {stats['batches_per_sec']:.1f}")
        print(f"  Total batches: {stats['completed_batches']}")
        
        assert stats['assets_per_sec'] > 10, \
            f"Multi-asset throughput too low: {stats['assets_per_sec']:.1f} assets/sec"
    
    def test_concurrent_requests_throughput(self, basic_nhits_config):
        """Test throughput under concurrent requests."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            # Create engine
            mock_model = MockNHITSModel(basic_nhits_config)
            torch.save(mock_model.state_dict(), tmp_file.name)
            
            engine = RealTimeNHITSEngine(tmp_file.name, basic_nhits_config)
            
            def single_prediction():
                """Single prediction function."""
                input_data = np.random.randn(basic_nhits_config.input_size)
                return engine.predict(input_data)
            
            # Test with different numbers of concurrent threads
            thread_counts = [1, 2, 4, 8]
            results = {}
            
            for num_threads in thread_counts:
                start_time = time.time()
                test_duration = 5.0
                completed_predictions = 0
                
                def worker():
                    nonlocal completed_predictions
                    end_time = start_time + test_duration
                    
                    while time.time() < end_time:
                        single_prediction()
                        completed_predictions += 1
                
                # Run with thread pool
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(worker) for _ in range(num_threads)]
                    
                    # Wait for completion
                    for future in futures:
                        future.result()
                
                actual_duration = time.time() - start_time
                predictions_per_sec = completed_predictions / actual_duration
                
                results[num_threads] = {
                    'predictions_per_sec': predictions_per_sec,
                    'total_predictions': completed_predictions
                }
                
                print(f"Threads: {num_threads}, Predictions/sec: {predictions_per_sec:.0f}")
            
            # Check scaling
            single_thread_perf = results[1]['predictions_per_sec']
            
            for threads in [2, 4, 8]:
                if threads in results:
                    multi_thread_perf = results[threads]['predictions_per_sec']
                    scaling_factor = multi_thread_perf / single_thread_perf
                    
                    print(f"Scaling factor ({threads} threads): {scaling_factor:.2f}x")
                    
                    # Should scale reasonably well
                    assert scaling_factor > 1.0, f"No improvement with {threads} threads"


class TestMemoryPerformance:
    """Test memory usage and efficiency."""
    
    def test_inference_memory_usage(self, basic_nhits_config, device):
        """Test memory usage during inference."""
        memory_benchmark = MemoryBenchmark()
        memory_benchmark.start_monitoring()
        
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        # Checkpoint after model creation
        memory_benchmark.checkpoint("model_loaded")
        
        # Test with different batch sizes
        batch_sizes = [1, 8, 32, 64]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, basic_nhits_config.input_size, device=device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            memory_benchmark.checkpoint(f"batch_size_{batch_size}")
            
            # Clean up
            del input_tensor, output
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Final checkpoint
        memory_benchmark.checkpoint("final")
        
        # Analyze memory usage
        stats = memory_benchmark.get_memory_stats()
        
        print("Memory usage analysis:")
        print(f"  Baseline: {stats['baseline']['cpu_memory_mb']:.1f}MB CPU, {stats['baseline']['gpu_memory_mb']:.1f}MB GPU")
        print(f"  Peak: {stats['peak']['cpu_memory_mb']:.1f}MB CPU, {stats['peak']['gpu_memory_mb']:.1f}MB GPU")
        print(f"  Increase: {stats['increase']['cpu_memory_mb']:.1f}MB CPU, {stats['increase']['gpu_memory_mb']:.1f}MB GPU")
        
        # Memory usage should be reasonable
        total_increase = stats['increase']['cpu_memory_mb'] + stats['increase']['gpu_memory_mb']
        assert total_increase < 1000, f"Memory usage too high: {total_increase:.1f}MB"
    
    @skip_if_no_gpu 
    def test_gpu_memory_efficiency(self, basic_nhits_config):
        """Test GPU memory efficiency."""
        with gpu_memory_context("inference_memory_test") as tracker:
            model = OptimizedNHITS(basic_nhits_config)
            model = model.cuda()
            model.eval()
            
            tracker.checkpoint("model_loaded")
            
            # Test memory usage scaling with batch size
            batch_sizes = [1, 2, 4, 8, 16, 32]
            memory_usage = []
            
            for batch_size in batch_sizes:
                # Clear cache before each test
                torch.cuda.empty_cache()
                baseline_memory = torch.cuda.memory_allocated()
                
                input_tensor = torch.randn(batch_size, basic_nhits_config.input_size, device='cuda')
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_per_sample = (peak_memory - baseline_memory) / batch_size
                
                memory_usage.append({
                    'batch_size': batch_size,
                    'total_memory_mb': peak_memory / (1024**2),
                    'memory_per_sample_mb': memory_per_sample / (1024**2)
                })
                
                # Cleanup
                del input_tensor, output
            
            # Analyze memory scaling
            print("GPU memory scaling:")
            for usage in memory_usage:
                print(f"  Batch {usage['batch_size']}: {usage['total_memory_mb']:.1f}MB total, "
                      f"{usage['memory_per_sample_mb']:.2f}MB per sample")
            
            # Memory per sample should be relatively stable
            per_sample_memories = [u['memory_per_sample_mb'] for u in memory_usage]
            memory_variance = np.var(per_sample_memories)
            
            assert memory_variance < 1.0, f"Memory per sample too variable: variance={memory_variance:.3f}"
    
    def test_memory_leak_detection(self, basic_nhits_config, device):
        """Test for memory leaks during repeated inference."""
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        memory_tracker = GPUMemoryTracker() if device.type == 'cuda' else None
        initial_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
        
        if memory_tracker:
            memory_tracker.checkpoint("initial")
        
        # Run many inference iterations
        num_iterations = 100
        
        for i in range(num_iterations):
            input_tensor = torch.randn(16, basic_nhits_config.input_size, device=device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Cleanup
            del input_tensor, output
            
            # Check memory every 20 iterations
            if i % 20 == 0:
                if memory_tracker:
                    memory_tracker.checkpoint(f"iteration_{i}")
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Final memory check
        final_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
        cpu_memory_increase = final_cpu_memory - initial_cpu_memory
        
        print(f"Memory leak test ({num_iterations} iterations):")
        print(f"  CPU memory increase: {cpu_memory_increase:.1f}MB")
        
        if memory_tracker:
            gpu_leak = memory_tracker.detect_memory_leak(tolerance_mb=50.0)
            usage = memory_tracker.get_memory_usage()
            print(f"  GPU memory: {usage['allocated']:.1f}MB allocated, {usage['peak']:.1f}MB peak")
            assert not gpu_leak, "GPU memory leak detected"
        
        # CPU memory should not increase significantly
        assert cpu_memory_increase < 100, f"CPU memory leak detected: {cpu_memory_increase:.1f}MB"


class TestPerformanceRegression:
    """Test for performance regressions against baseline."""
    
    def test_latency_regression(self, basic_nhits_config, device, performance_baseline):
        """Test for latency regression."""
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        config = BenchmarkConfig(benchmark_iterations=50)
        benchmark = LatencyBenchmark(config)
        
        def inference_func():
            input_tensor = torch.randn(16, basic_nhits_config.input_size, device=device)
            with torch.no_grad():
                return model(input_tensor)
        
        current_stats = benchmark.benchmark_function(inference_func)
        
        # Compare against baseline
        baseline_stats = performance_baseline['inference_latency']
        
        print("Latency regression test:")
        print(f"  Current mean: {current_stats['mean_ms']:.2f}ms")
        print(f"  Baseline mean: {baseline_stats['mean_ms']:.2f}ms")
        print(f"  Current P95: {current_stats['p95_ms']:.2f}ms")
        print(f"  Baseline P95: {baseline_stats['p95_ms']:.2f}ms")
        
        # Assert no significant regression (20% tolerance)
        assert_performance_regression(current_stats, baseline_stats, tolerance_percent=20.0)
    
    def test_throughput_regression(self, basic_nhits_config, device, performance_baseline):
        """Test for throughput regression."""
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        def data_generator():
            return torch.randn(32, basic_nhits_config.input_size, device=device)
        
        def inference_func(data):
            with torch.no_grad():
                return model(data)
        
        benchmark = ThroughputBenchmark(BenchmarkConfig())
        current_stats = benchmark.benchmark_throughput(
            inference_func, data_generator, duration_seconds=5.0
        )
        
        baseline_stats = performance_baseline['throughput']
        
        print("Throughput regression test:")
        print(f"  Current: {current_stats['items_per_second']:.0f} predictions/sec")
        print(f"  Baseline: {baseline_stats['predictions_per_second']:.0f} predictions/sec")
        
        # Should not regress significantly
        current_throughput = current_stats['items_per_second']
        baseline_throughput = baseline_stats['predictions_per_second']
        
        if baseline_throughput > 0:
            regression_percent = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
            
            assert regression_percent < 20, \
                f"Throughput regression: {regression_percent:.1f}% decrease"
    
    def test_memory_regression(self, basic_nhits_config, device, performance_baseline):
        """Test for memory usage regression."""
        memory_benchmark = MemoryBenchmark()
        memory_benchmark.start_monitoring()
        
        model = OptimizedNHITS(basic_nhits_config)
        model = model.to(device)
        model.eval()
        
        # Run inference
        input_tensor = torch.randn(32, basic_nhits_config.input_size, device=device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        current_stats = memory_benchmark.get_memory_stats()
        baseline_stats = performance_baseline['memory_usage']
        
        current_total = (current_stats['current']['cpu_memory_mb'] + 
                        current_stats['current']['gpu_memory_mb'])
        baseline_total = (baseline_stats['cpu_memory_mb'] + 
                         baseline_stats['gpu_memory_mb'])
        
        print("Memory regression test:")
        print(f"  Current: {current_total:.1f}MB total")
        print(f"  Baseline: {baseline_total:.1f}MB total")
        
        if baseline_total > 0:
            memory_increase_percent = ((current_total - baseline_total) / baseline_total) * 100
            
            assert memory_increase_percent < 50, \
                f"Memory usage regression: {memory_increase_percent:.1f}% increase"


class TestScalabilityPerformance:
    """Test performance scalability."""
    
    def test_input_size_scalability(self, device):
        """Test performance scaling with input size."""
        input_sizes = [60, 120, 240, 480, 720]  # Different input sizes
        horizon = 24
        
        results = []
        
        for input_size in input_sizes:
            from tests.neural.utils.mock_objects import MockNHITSConfig
            config = MockNHITSConfig(h=horizon, input_size=input_size)
            
            model = OptimizedNHITS(config)
            model = model.to(device)
            model.eval()
            
            # Benchmark latency
            input_tensor = torch.randn(16, input_size, device=device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            # Measure
            times = []
            for _ in range(20):
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            mean_time = statistics.mean(times)
            
            results.append({
                'input_size': input_size,
                'latency_ms': mean_time,
                'latency_per_input': mean_time / input_size
            })
            
            print(f"Input size {input_size}: {mean_time:.2f}ms ({mean_time/input_size:.4f}ms per input)")
        
        # Check scaling is reasonable (should be roughly linear or sub-linear)
        latencies = [r['latency_ms'] for r in results]
        input_sizes_list = [r['input_size'] for r in results]
        
        # Calculate scaling factor (last vs first)
        size_ratio = input_sizes_list[-1] / input_sizes_list[0]
        latency_ratio = latencies[-1] / latencies[0]
        
        print(f"Scaling: {size_ratio:.1f}x input size -> {latency_ratio:.1f}x latency")
        
        # Latency should not scale worse than quadratically
        assert latency_ratio <= size_ratio ** 1.5, \
            f"Poor scaling: {latency_ratio:.1f}x latency for {size_ratio:.1f}x input"
    
    def test_horizon_size_scalability(self, device):
        """Test performance scaling with forecast horizon."""
        horizons = [12, 24, 48, 96, 168]  # Different forecast horizons
        input_size = 168
        
        results = []
        
        for horizon in horizons:
            from tests.neural.utils.mock_objects import MockNHITSConfig
            config = MockNHITSConfig(h=horizon, input_size=input_size)
            
            model = OptimizedNHITS(config)
            model = model.to(device)
            model.eval()
            
            input_tensor = torch.randn(16, input_size, device=device)
            
            # Benchmark
            times = []
            for _ in range(20):
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            mean_time = statistics.mean(times)
            
            results.append({
                'horizon': horizon,
                'latency_ms': mean_time
            })
            
            print(f"Horizon {horizon}: {mean_time:.2f}ms")
        
        # Check reasonable scaling
        latencies = [r['latency_ms'] for r in results]
        horizons_list = [r['horizon'] for r in results]
        
        # Should scale reasonably with horizon
        for i in range(1, len(results)):
            horizon_ratio = horizons_list[i] / horizons_list[0]
            latency_ratio = latencies[i] / latencies[0]
            
            assert latency_ratio <= horizon_ratio * 2, \
                f"Poor horizon scaling: {latency_ratio:.1f}x latency for {horizon_ratio:.1f}x horizon"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])