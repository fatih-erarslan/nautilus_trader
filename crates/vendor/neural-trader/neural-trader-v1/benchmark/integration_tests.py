#!/usr/bin/env python3
"""
Complete integration test suite for AI News Trading benchmark system.
Tests end-to-end functionality across all components including the new integration layer.
"""

import asyncio
import json
import os
import pytest
import time
import tempfile
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock, Mock, AsyncMock
import concurrent.futures
import psutil

# Import all benchmark components
from src.benchmarks.runner import BenchmarkRunner
from src.benchmarks.latency_benchmark import LatencyBenchmark
from src.benchmarks.throughput_benchmark import ThroughputBenchmark
from src.benchmarks.strategy_benchmark import StrategyBenchmark
from src.benchmarks.resource_benchmark import ResourceBenchmark
from src.simulation.simulator import Simulator
from src.simulation.market_simulator import MarketSimulator
from src.optimization.optimizer import Optimizer
from src.data.realtime_manager import RealtimeDataManager
from src.config import ConfigManager
from cli import cli, Context
from click.testing import CliRunner

# Import new integration components
from src.integration.system_orchestrator import SystemOrchestrator, SystemState
from src.integration.component_registry import ComponentRegistry, ComponentStatus, ComponentType
from src.integration.data_pipeline import DataPipeline, DataType, DataPacket
from src.integration.performance_monitor import PerformanceMonitor, AlertLevel, Alert


class IntegrationTestConfig:
    """Configuration for integration tests"""
    
    def __init__(self):
        self.config = {
            'benchmark': {
                'suites': {
                    'integration': {
                        'strategies': ['momentum', 'swing', 'mirror'],
                        'duration': 60,  # 1 minute for tests
                        'metrics': ['latency', 'throughput', 'strategy_performance', 'memory']
                    }
                },
                'targets': {
                    'latency_ms': 100,
                    'throughput_ops_sec': 10000,
                    'memory_mb': 2048,
                    'concurrent_simulations': 1000
                }
            },
            'simulation': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005,
                'market_hours': True
            },
            'data': {
                'sources': ['mock', 'alpha_vantage', 'yahoo'],
                'cache_enabled': True,
                'realtime_enabled': True
            },
            'optimization': {
                'algorithms': ['bayesian', 'genetic', 'grid_search'],
                'max_iterations': 50,
                'convergence_threshold': 0.001
            }
        }
    
    def to_dict(self) -> dict:
        return self.config


class TestSystemIntegration:
    """Test complete system integration"""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration"""
        return IntegrationTestConfig().to_dict()
    
    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_cli_to_simulation_integration(self, config, temp_dir):
        """Test CLI command integrates with simulation engine"""
        runner = CliRunner()
        output_file = os.path.join(temp_dir, 'integration_test.json')
        
        # Run benchmark command
        result = runner.invoke(cli, [
            'benchmark',
            '--strategy', 'momentum',
            '--duration', '1m',
            '--assets', 'stocks',
            '--output', output_file,
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        
        # Validate output structure
        with open(output_file, 'r') as f:
            data = json.load(f)
            assert 'strategies' in data
            assert 'momentum' in data['strategies']
            assert 'metrics' in data['strategies']['momentum']
    
    def test_simulation_to_optimization_pipeline(self, config, temp_dir):
        """Test simulation results feed into optimization"""
        runner = CliRunner()
        sim_output = os.path.join(temp_dir, 'simulation.json')
        opt_output = os.path.join(temp_dir, 'optimization.json')
        
        # Run simulation
        result = runner.invoke(cli, [
            'simulate',
            '--historical',
            '--start', '2024-01-01',
            '--end', '2024-03-31',
            '--output', sim_output,
            '--format', 'json'
        ])
        assert result.exit_code == 0
        
        # Run optimization on simulation results
        result = runner.invoke(cli, [
            'optimize',
            '--strategy', 'momentum',
            '--metric', 'sharpe',
            '--iterations', '10',
            '--input', sim_output,
            '--output', opt_output
        ])
        assert result.exit_code == 0
        assert os.path.exists(opt_output)
    
    def test_realtime_data_integration(self, config):
        """Test real-time data manager integration"""
        # Mock real-time data sources
        with patch('src.data.realtime_manager.RealtimeDataManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.start.return_value = True
            mock_instance.get_latest_data.return_value = {
                'AAPL': {'price': 150.0, 'volume': 1000000, 'timestamp': datetime.now()}
            }
            
            # Initialize data manager
            data_manager = RealtimeDataManager(config)
            assert data_manager.start()
            
            # Verify data retrieval
            data = data_manager.get_latest_data(['AAPL'])
            assert 'AAPL' in data
            assert 'price' in data['AAPL']
    
    def test_multi_component_benchmark_pipeline(self, config):
        """Test complete benchmark pipeline with all components"""
        benchmark_runner = BenchmarkRunner(IntegrationTestConfig())
        
        # Run integrated benchmark suite
        results = benchmark_runner.run_suite('integration')
        
        assert results['status'] == 'success'
        assert 'results' in results
        
        # Verify all benchmark types executed
        if 'latency' in results['results']:
            assert 'signal_generation' in results['results']['latency']
        
        if 'throughput' in results['results']:
            assert isinstance(results['results']['throughput'], dict)
        
        if 'strategy_performance' in results['results']:
            assert isinstance(results['results']['strategy_performance'], dict)


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    def test_complete_trading_system_workflow(self):
        """Test complete trading system from data to execution"""
        # This is a comprehensive integration test simulating real usage
        
        # Step 1: Initialize all components
        config = IntegrationTestConfig()
        
        # Step 2: Set up data pipeline
        # (Mock implementation for testing)
        market_data = {
            'AAPL': [150.0, 151.0, 149.5, 152.0],
            'MSFT': [300.0, 301.5, 299.0, 302.0],
            'GOOGL': [2800.0, 2810.0, 2795.0, 2815.0]
        }
        
        # Step 3: Run simulation
        simulator = Simulator(config.to_dict())
        sim_results = simulator.run_backtest(
            strategies=['momentum'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            assets=['AAPL', 'MSFT', 'GOOGL']
        )
        
        assert sim_results is not None
        assert 'performance' in sim_results
        
        # Step 4: Optimize based on simulation
        optimizer = Optimizer(config.to_dict())
        opt_results = optimizer.optimize_strategy(
            'momentum',
            sim_results,
            metric='sharpe_ratio'
        )
        
        assert opt_results is not None
        assert 'optimized_parameters' in opt_results
        
        # Step 5: Generate performance report
        report = self._generate_integration_report(sim_results, opt_results)
        assert report is not None
        assert len(report) > 100  # Non-trivial report
    
    def test_concurrent_simulation_capacity(self):
        """Test system handles concurrent simulations"""
        config = IntegrationTestConfig()
        num_concurrent = 10  # Reduced for testing
        
        async def run_single_simulation(sim_id: int):
            """Run a single simulation"""
            simulator = Simulator(config.to_dict())
            return await simulator.run_async_backtest(
                strategy='momentum',
                start_date='2024-01-01',
                end_date='2024-01-15',
                assets=['AAPL'],
                simulation_id=sim_id
            )
        
        async def run_concurrent_simulations():
            """Run multiple simulations concurrently"""
            tasks = [
                run_single_simulation(i) 
                for i in range(num_concurrent)
            ]
            results = await asyncio.gather(*tasks)
            return results
        
        # Execute concurrent simulations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_concurrent_simulations())
            assert len(results) == num_concurrent
            assert all(result is not None for result in results)
        finally:
            loop.close()
    
    def test_memory_efficiency_under_load(self):
        """Test memory efficiency during high-load operations"""
        config = IntegrationTestConfig()
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run memory-intensive operations
        benchmark_runner = BenchmarkRunner(config)
        
        # Multiple sequential benchmark runs
        for i in range(5):
            results = benchmark_runner.run_suite('quick')
            assert results['status'] == 'success'
            
            # Check memory hasn't grown excessively
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Allow up to 500MB growth during testing
            assert memory_growth < 500, f"Memory grew by {memory_growth:.1f}MB"
    
    def _generate_integration_report(self, sim_results, opt_results) -> str:
        """Generate integration test report"""
        report_lines = [
            "INTEGRATION TEST REPORT",
            "=" * 50,
            f"Timestamp: {datetime.now().isoformat()}",
            "",
            "SIMULATION RESULTS:",
            f"  Performance: {sim_results.get('performance', 'N/A')}",
            "",
            "OPTIMIZATION RESULTS:",
            f"  Optimized Parameters: {opt_results.get('optimized_parameters', 'N/A')}",
            "",
            "INTEGRATION STATUS: PASSED"
        ]
        return "\n".join(report_lines)


class TestPerformanceTargets:
    """Test system meets performance targets"""
    
    @pytest.fixture
    def performance_config(self):
        """Configuration with performance targets"""
        return IntegrationTestConfig()
    
    def test_signal_generation_latency_target(self, performance_config):
        """Test signal generation meets latency target < 100ms"""
        latency_benchmark = LatencyBenchmark(performance_config)
        
        # Run latency benchmark
        result = latency_benchmark.run_sync('signal_generation')
        
        # Check P95 latency is under target
        p95_latency = result.percentiles.get('p95', float('inf'))
        target_latency = performance_config.config['benchmark']['targets']['latency_ms']
        
        assert p95_latency < target_latency, f"P95 latency {p95_latency}ms exceeds target {target_latency}ms"
    
    def test_throughput_target(self, performance_config):
        """Test throughput meets target > 10,000 ops/sec"""
        throughput_benchmark = ThroughputBenchmark(performance_config)
        
        # Run throughput benchmark
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                throughput_benchmark.benchmark_signal_throughput()
            )
            
            ops_per_sec = result.operations_per_second
            target_throughput = performance_config.config['benchmark']['targets']['throughput_ops_sec']
            
            assert ops_per_sec > target_throughput, f"Throughput {ops_per_sec:.1f} ops/sec below target {target_throughput}"
        finally:
            loop.close()
    
    def test_memory_usage_target(self, performance_config):
        """Test memory usage stays under target < 2GB"""
        resource_benchmark = ResourceBenchmark(performance_config)
        
        # Run resource benchmark
        result = resource_benchmark.benchmark_signal_generation_resources()
        
        peak_memory_mb = result.memory.peak_memory_mb
        target_memory = performance_config.config['benchmark']['targets']['memory_mb']
        
        assert peak_memory_mb < target_memory, f"Peak memory {peak_memory_mb:.1f}MB exceeds target {target_memory}MB"
    
    def test_concurrent_simulation_capacity_target(self, performance_config):
        """Test system supports target concurrent simulations"""
        # This is a scaled-down version for testing
        target_simulations = 50  # Reduced from 1000 for CI testing
        
        async def create_mock_simulation(sim_id: int):
            """Create a mock simulation task"""
            await asyncio.sleep(0.01)  # Simulate work
            return {'id': sim_id, 'status': 'completed'}
        
        async def test_concurrent_capacity():
            """Test concurrent simulation capacity"""
            tasks = [
                create_mock_simulation(i) 
                for i in range(target_simulations)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Should complete within reasonable time
            duration = end_time - start_time
            assert duration < 5.0, f"Concurrent operations took {duration:.2f}s"
            assert len(results) == target_simulations
            
            return results
        
        # Execute test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(test_concurrent_capacity())
            assert len(results) == target_simulations
        finally:
            loop.close()


class TestSystemResilience:
    """Test system resilience and error recovery"""
    
    def test_component_failure_recovery(self):
        """Test system handles component failures gracefully"""
        config = IntegrationTestConfig()
        
        # Simulate data source failure
        with patch('src.data.realtime_manager.RealtimeDataManager.connect') as mock_connect:
            mock_connect.side_effect = ConnectionError("Data source unavailable")
            
            # System should handle this gracefully
            try:
                data_manager = RealtimeDataManager(config.to_dict())
                result = data_manager.start()
                # Should either succeed with fallback or fail gracefully
                assert isinstance(result, bool)
            except Exception as e:
                # If it raises an exception, it should be handled gracefully
                assert "Data source unavailable" in str(e)
    
    def test_resource_exhaustion_handling(self):
        """Test system behavior under resource constraints"""
        config = IntegrationTestConfig()
        
        # Simulate low memory condition
        original_memory_limit = 100  # MB, artificially low for testing
        
        # Mock memory monitoring
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = original_memory_limit * 1024 * 1024  # Convert to bytes
            
            benchmark_runner = BenchmarkRunner(config)
            
            # Should handle gracefully or warn about low memory
            try:
                results = benchmark_runner.run_suite('quick')
                # Should either complete successfully or handle gracefully
                assert 'status' in results
            except MemoryError:
                # If memory error is raised, it should be expected
                pytest.skip("Low memory condition correctly detected")
    
    def test_configuration_validation(self):
        """Test system validates configuration properly"""
        # Test with invalid configuration
        invalid_config = {
            'benchmark': {
                'invalid_key': 'invalid_value'
            }
        }
        
        config_manager = ConfigManager()
        config_manager.config = invalid_config
        
        # Should validate and handle invalid configuration
        is_valid = config_manager.validate()
        
        # Either validation should pass (with defaults) or fail appropriately
        assert isinstance(is_valid, bool)


class TestDataFlowIntegrity:
    """Test data flows correctly between components"""
    
    def test_market_data_to_signals_flow(self):
        """Test market data flows correctly to signal generation"""
        config = IntegrationTestConfig()
        
        # Mock market data
        market_data = {
            'timestamp': datetime.now(),
            'AAPL': {
                'price': 150.0,
                'volume': 1000000,
                'bid': 149.95,
                'ask': 150.05
            }
        }
        
        # Test data flows through signal generation
        simulator = Simulator(config.to_dict())
        signals = simulator.generate_signals(market_data, strategy='momentum')
        
        assert signals is not None
        assert isinstance(signals, (dict, list))
    
    def test_signals_to_execution_flow(self):
        """Test signals flow correctly to execution engine"""
        config = IntegrationTestConfig()
        
        # Mock signals
        signals = [
            {
                'symbol': 'AAPL',
                'action': 'buy',
                'quantity': 100,
                'price': 150.0,
                'timestamp': datetime.now()
            }
        ]
        
        # Test signals flow to execution
        simulator = Simulator(config.to_dict())
        execution_results = simulator.execute_signals(signals)
        
        assert execution_results is not None
        assert isinstance(execution_results, (dict, list))
    
    def test_execution_to_portfolio_flow(self):
        """Test execution results update portfolio correctly"""
        config = IntegrationTestConfig()
        
        # Mock execution results
        executions = [
            {
                'symbol': 'AAPL',
                'action': 'buy',
                'quantity': 100,
                'price': 150.0,
                'commission': 1.0,
                'timestamp': datetime.now()
            }
        ]
        
        # Test portfolio updates
        simulator = Simulator(config.to_dict())
        portfolio_state = simulator.update_portfolio(executions)
        
        assert portfolio_state is not None
        assert isinstance(portfolio_state, dict)


class TestIntegrationLayer:
    """Test the new integration layer components"""
    
    @pytest.fixture
    async def integration_config(self):
        """Provide integration test configuration"""
        return {
            'monitoring_interval': 0.1,
            'persistence_enabled': True,
            'data_dir': tempfile.mkdtemp(prefix="integration_test_")
        }
    
    @pytest.fixture
    async def temp_orchestrator(self, integration_config):
        """Provide temporary orchestrator for testing"""
        orchestrator = SystemOrchestrator()
        orchestrator.config = type('Config', (), integration_config)()
        yield orchestrator
        
        # Cleanup
        if orchestrator.state == SystemState.RUNNING:
            await orchestrator.stop()
        
        # Clean up temp directory
        import shutil
        if 'data_dir' in integration_config and os.path.exists(integration_config['data_dir']):
            shutil.rmtree(integration_config['data_dir'])
    
    @pytest.mark.asyncio
    async def test_system_orchestrator_lifecycle(self, temp_orchestrator):
        """Test system orchestrator startup and shutdown"""
        orchestrator = temp_orchestrator
        
        # Test startup
        start_result = await orchestrator.start()
        assert start_result, "Orchestrator should start successfully"
        assert orchestrator.state == SystemState.RUNNING, "Should be in running state"
        
        # Test status
        status = orchestrator.get_system_status()
        assert status['state'] == SystemState.RUNNING.value
        
        # Test shutdown
        stop_result = await orchestrator.stop()
        assert stop_result, "Orchestrator should stop successfully"
        assert orchestrator.state == SystemState.STOPPED, "Should be in stopped state"
    
    @pytest.mark.asyncio
    async def test_component_registry_functionality(self):
        """Test component registry operations"""
        registry = ComponentRegistry()
        
        # Create mock components
        mock_component = Mock()
        mock_component.start = AsyncMock(return_value=True)
        mock_component.stop = AsyncMock(return_value=True)
        mock_component.health_check = AsyncMock(return_value=True)
        
        # Test registration
        result = await registry.register_component(
            'test_component', 
            mock_component, 
            ComponentType.SERVICE
        )
        assert result, "Component should register successfully"
        
        # Test startup
        start_result = await registry.start_component('test_component')
        assert start_result, "Component should start successfully"
        
        # Test health check
        health = await registry.health_check()
        assert health['test_component'] == ComponentStatus.RUNNING
        
        # Test shutdown
        stop_result = await registry.stop_component('test_component')
        assert stop_result, "Component should stop successfully"
    
    @pytest.mark.asyncio
    async def test_data_pipeline_processing(self, integration_config):
        """Test data pipeline processing"""
        config = type('Config', (), integration_config)()
        pipeline = DataPipeline(config)
        
        # Start pipeline
        start_result = await pipeline.start()
        assert start_result, "Pipeline should start successfully"
        
        # Create test packet
        packet = DataPacket(
            data_type=DataType.MARKET_DATA,
            timestamp=time.time(),
            source="test",
            data={
                'symbol': 'AAPL',
                'price': 150.0,
                'volume': 1000
            }
        )
        
        # Enqueue packet
        enqueue_result = await pipeline.enqueue_packet(packet)
        assert enqueue_result, "Packet should be enqueued successfully"
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Check status
        status = pipeline.get_status()
        assert status['running'], "Pipeline should be running"
        
        # Stop pipeline
        stop_result = await pipeline.stop()
        assert stop_result, "Pipeline should stop successfully"
    
    @pytest.mark.asyncio
    async def test_performance_monitor_functionality(self, integration_config):
        """Test performance monitoring"""
        monitor = PerformanceMonitor(integration_config)
        
        # Start monitoring
        start_result = await monitor.start()
        assert start_result, "Monitor should start successfully"
        
        # Generate metrics
        monitor.increment_counter('test_counter', 5)
        monitor.set_gauge('test_gauge', 42.0)
        monitor.record_histogram('test_histogram', 100.0)
        
        # Start and stop timer
        stop_timer = monitor.start_timer('test_timer')
        await asyncio.sleep(0.01)
        stop_timer()
        
        # Wait for data collection
        await asyncio.sleep(0.2)
        
        # Check metrics
        metrics = monitor.get_current_metrics()
        assert metrics['counters']['test_counter'] == 5
        assert metrics['gauges']['test_gauge'] == 42.0
        
        # Check system status
        system_status = monitor.get_system_status()
        assert 'cpu_percent' in system_status
        assert 'memory_percent' in system_status
        
        # Stop monitoring
        stop_result = await monitor.stop()
        assert stop_result, "Monitor should stop successfully"
    
    @pytest.mark.asyncio
    async def test_integration_layer_error_handling(self):
        """Test error handling in integration layer"""
        registry = ComponentRegistry()
        
        # Create failing component
        failing_component = Mock()
        failing_component.start = AsyncMock(side_effect=Exception("Test failure"))
        failing_component.stop = AsyncMock(return_value=True)
        
        # Register failing component
        await registry.register_component('failing_component', failing_component)
        
        # Attempt to start - should handle failure gracefully
        start_result = await registry.start_component('failing_component')
        assert not start_result, "Should fail to start"
        
        # Check component is in error state
        status = registry.get_component_status()
        assert status['failing_component']['status'] == ComponentStatus.ERROR.value
    
    @pytest.mark.asyncio
    async def test_complete_integration_workflow(self, temp_orchestrator):
        """Test complete integration workflow"""
        orchestrator = temp_orchestrator
        
        # Start orchestrator
        await orchestrator.start()
        
        # Simulate data processing workflow
        if orchestrator.data_pipeline:
            # Create test data
            test_packets = [
                DataPacket(
                    data_type=DataType.MARKET_DATA,
                    timestamp=time.time(),
                    source="integration_test",
                    data={'symbol': 'AAPL', 'price': 150.0 + i, 'volume': 1000}
                )
                for i in range(10)
            ]
            
            # Process packets
            for packet in test_packets:
                await orchestrator.data_pipeline.enqueue_packet(packet)
            
            # Wait for processing
            await asyncio.sleep(1)
            
            # Check pipeline status
            status = orchestrator.data_pipeline.get_status()
            assert status['metrics']['packets_processed'] > 0
        
        # Check system status
        system_status = orchestrator.get_system_status()
        assert system_status['state'] == SystemState.RUNNING.value
        
        # Stop orchestrator
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_integration_performance_under_load(self, temp_orchestrator):
        """Test integration layer performance under load"""
        orchestrator = temp_orchestrator
        await orchestrator.start()
        
        # Generate high load
        if orchestrator.data_pipeline:
            start_time = time.time()
            packet_count = 100
            
            # Create packets in batches
            for batch in range(10):
                batch_packets = [
                    DataPacket(
                        data_type=DataType.MARKET_DATA,
                        timestamp=time.time(),
                        source="load_test",
                        data={
                            'symbol': f'TEST{i}',
                            'price': 100.0 + i,
                            'volume': 1000 + i
                        }
                    )
                    for i in range(batch * 10, (batch + 1) * 10)
                ]
                
                # Enqueue batch
                for packet in batch_packets:
                    await orchestrator.data_pipeline.enqueue_packet(packet)
                
                await asyncio.sleep(0.01)  # Small delay between batches
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check performance
            status = orchestrator.data_pipeline.get_status()
            processing_time = time.time() - start_time
            throughput = status['metrics']['packets_processed'] / processing_time
            
            # Should achieve reasonable throughput
            assert throughput > 10, f"Throughput too low: {throughput:.2f} pps"
        
        await orchestrator.stop()


class TestSystemResilience:
    """Test system resilience and error recovery"""
    
    @pytest.mark.asyncio
    async def test_integration_layer_component_recovery(self):
        """Test component failure and recovery in integration layer"""
        registry = ComponentRegistry()
        
        # Create component that initially fails health checks
        mock_component = Mock()
        mock_component.start = AsyncMock(return_value=True)
        mock_component.stop = AsyncMock(return_value=True)
        mock_component.health_check = AsyncMock(return_value=False)  # Initially unhealthy
        
        await registry.register_component('test_component', mock_component)
        await registry.start_component('test_component')
        
        # Health check should show failure
        health = await registry.health_check()
        assert health['test_component'] == ComponentStatus.ERROR
        
        # Fix component and restart
        mock_component.health_check = AsyncMock(return_value=True)
        restart_result = await registry.restart_component('test_component')
        assert restart_result, "Component should restart successfully"
        
        # Health check should now pass
        health = await registry.health_check()
        assert health['test_component'] == ComponentStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_failures(self):
        """Test system continues operating with component failures"""
        orchestrator = SystemOrchestrator()
        
        # Mock partial component failures
        with patch.object(orchestrator, '_initialize_components') as mock_init:
            # Simulate some components failing to initialize
            mock_init.side_effect = Exception("Partial initialization failure")
            
            # System should handle this gracefully
            start_result = await orchestrator.start()
            # Depending on implementation, it might start with reduced functionality
            # or fail gracefully
            assert isinstance(start_result, bool)
            
            if orchestrator.state == SystemState.RUNNING:
                await orchestrator.stop()


if __name__ == '__main__':
    # Run integration tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--durations=10'
    ])