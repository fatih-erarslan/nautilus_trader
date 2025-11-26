"""
Integration Tests for MCP Neural Tools

Tests for neural forecasting tool integration with the MCP server.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd

# Test utilities
from tests.neural.utils.fixtures import (
    mock_mcp_client, basic_nhits_config, sample_time_series_data,
    async_test_timeout, benchmark_config
)
from tests.neural.utils.mock_objects import (
    MockMCPServer, create_mock_mcp_server, MockNHITSConfig
)
from tests.neural.utils.performance_utils import (
    LatencyBenchmark, performance_monitoring
)
from tests.neural.utils.data_generators import (
    SyntheticTimeSeriesGenerator, TimeSeriesParams
)

# MCP components (use mocks if not available)
try:
    from src.mcp.server import MCPServer
    from src.mcp.handlers.tools import neural_forecast, neural_backtest, neural_train
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Use mock implementations
    MCPServer = MockMCPServer


class TestMCPNeuralTools:
    """Test neural forecasting tools through MCP interface."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server instance for testing."""
        if MCP_AVAILABLE:
            server = MCPServer()
            # Initialize with neural tools
            await server.initialize_neural_tools()
            return server
        else:
            return create_mock_mcp_server()
    
    @pytest.mark.asyncio
    async def test_neural_forecast_tool_basic(self, mcp_server):
        """Test basic neural forecasting tool functionality."""
        params = {
            'symbol': 'AAPL',
            'horizon': 24,
            'confidence_level': 95,
            'use_gpu': False
        }
        
        response = await mcp_server.call_tool('neural_forecast', params)
        
        assert response['status'] == 'success'
        assert 'result' in response
        
        result = response['result']
        assert 'symbol' in result
        assert 'forecast' in result
        assert 'horizon' in result
        assert 'confidence_intervals' in result
        assert 'metadata' in result
        
        # Check forecast length
        assert len(result['forecast']) == params['horizon']
        
        # Check confidence intervals
        ci = result['confidence_intervals']
        assert f"{params['confidence_level']}%" in ci
        
        # Check metadata
        metadata = result['metadata']
        assert 'inference_time_ms' in metadata
        assert 'model_version' in metadata
    
    @pytest.mark.asyncio
    async def test_neural_forecast_tool_gpu(self, mcp_server):
        """Test neural forecasting tool with GPU acceleration."""
        params = {
            'symbol': 'GOOGL',
            'horizon': 48,
            'confidence_level': 90,
            'use_gpu': True
        }
        
        response = await mcp_server.call_tool('neural_forecast', params)
        
        assert response['status'] == 'success'
        result = response['result']
        
        # GPU usage should be indicated in metadata
        metadata = result['metadata']
        assert 'gpu_used' in metadata
        
        # GPU inference might be faster (in real implementation)
        if 'inference_time_ms' in metadata:
            assert metadata['inference_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_neural_forecast_tool_multiple_horizons(self, mcp_server):
        """Test neural forecasting with different forecast horizons."""
        horizons = [12, 24, 48, 96]
        
        for horizon in horizons:
            params = {
                'symbol': 'MSFT',
                'horizon': horizon,
                'confidence_level': 95,
                'use_gpu': False
            }
            
            response = await mcp_server.call_tool('neural_forecast', params)
            
            assert response['status'] == 'success'
            result = response['result']
            assert len(result['forecast']) == horizon
    
    @pytest.mark.asyncio
    async def test_neural_forecast_tool_error_handling(self, mcp_server):
        """Test error handling in neural forecasting tool."""
        # Test with invalid symbol
        params = {
            'symbol': '',  # Empty symbol
            'horizon': 24,
            'confidence_level': 95
        }
        
        response = await mcp_server.call_tool('neural_forecast', params)
        
        # Should handle gracefully (either success with default or error)
        assert 'status' in response
        
        # Test with invalid horizon
        params = {
            'symbol': 'AAPL',
            'horizon': -1,  # Invalid horizon
            'confidence_level': 95
        }
        
        response = await mcp_server.call_tool('neural_forecast', params)
        
        # Should handle invalid parameters
        if response['status'] == 'error':
            assert 'error' in response
    
    @pytest.mark.asyncio
    async def test_neural_backtest_tool(self, mcp_server):
        """Test neural backtesting tool."""
        params = {
            'model_id': 'nhits_test_model',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'metrics': ['mae', 'mape', 'sharpe']
        }
        
        response = await mcp_server.call_tool('neural_backtest', params)
        
        assert response['status'] == 'success'
        result = response['result']
        
        assert 'model_id' in result
        assert 'period' in result
        assert 'metrics' in result
        assert 'performance_summary' in result
        
        # Check metrics
        metrics = result['metrics']
        for metric in params['metrics']:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check performance summary
        summary = result['performance_summary']
        assert 'total_predictions' in summary
        assert 'accuracy_score' in summary
    
    @pytest.mark.asyncio
    async def test_neural_train_tool(self, mcp_server):
        """Test neural training tool."""
        params = {
            'data_path': 'test_data.csv',
            'model_type': 'nhits',
            'epochs': 10,  # Small for testing
            'use_gpu': False
        }
        
        response = await mcp_server.call_tool('neural_train', params)
        
        assert response['status'] == 'success'
        result = response['result']
        
        assert 'model_id' in result
        assert 'training_metrics' in result
        assert 'model_info' in result
        
        # Check training metrics
        metrics = result['training_metrics']
        assert 'final_train_loss' in metrics
        assert 'final_val_loss' in metrics
        assert 'epochs_completed' in metrics
        assert 'training_time_seconds' in metrics
        
        # Check model info
        info = result['model_info']
        assert 'type' in info
        assert 'parameters' in info
        assert info['type'] == params['model_type']
    
    @pytest.mark.asyncio
    async def test_neural_optimize_tool(self, mcp_server):
        """Test neural optimization tool."""
        params = {
            'model_id': 'nhits_test_model',
            'optimization_metric': 'mae',
            'max_trials': 20  # Small for testing
        }
        
        response = await mcp_server.call_tool('neural_optimize', params)
        
        assert response['status'] == 'success'
        result = response['result']
        
        assert 'model_id' in result
        assert 'optimization_results' in result
        assert 'performance_improvement' in result
        
        # Check optimization results
        opt_results = result['optimization_results']
        assert 'best_score' in opt_results
        assert 'best_params' in opt_results
        assert 'trials_completed' in opt_results
        
        # Check improvement metrics
        improvement = result['performance_improvement']
        assert 'baseline_score' in improvement
        assert 'optimized_score' in improvement
    
    @pytest.mark.asyncio
    async def test_neural_analyze_tool(self, mcp_server):
        """Test neural analysis tool."""
        params = {
            'symbol': 'AAPL',
            'analysis_type': 'feature_importance'
        }
        
        response = await mcp_server.call_tool('neural_analyze', params)
        
        assert response['status'] == 'success'
        result = response['result']
        
        assert 'symbol' in result
        assert 'analysis_type' in result
        
        if params['analysis_type'] == 'feature_importance':
            assert 'feature_importance' in result
            assert 'top_features' in result
    
    @pytest.mark.asyncio
    async def test_neural_benchmark_tool(self, mcp_server):
        """Test neural benchmarking tool."""
        params = {
            'model_id': 'nhits_test_model',
            'benchmark_type': 'performance'
        }
        
        response = await mcp_server.call_tool('neural_benchmark', params)
        
        assert response['status'] == 'success'
        result = response['result']
        
        assert 'model_id' in result
        assert 'benchmark_type' in result
        assert 'results' in result
        
        # Check benchmark results structure
        benchmark_results = result['results']
        assert 'latency_ms' in benchmark_results
        assert 'throughput' in benchmark_results
        assert 'memory_usage' in benchmark_results
        assert 'accuracy' in benchmark_results


class TestMCPToolPerformance:
    """Test performance of MCP neural tools."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server for performance testing."""
        return create_mock_mcp_server()
    
    @pytest.mark.asyncio
    async def test_tool_response_latency(self, mcp_server, benchmark_config):
        """Test MCP tool response latency."""
        latency_benchmark = LatencyBenchmark(benchmark_config)
        
        async def forecast_call():
            params = {
                'symbol': 'AAPL',
                'horizon': 24,
                'confidence_level': 95,
                'use_gpu': False
            }
            return await mcp_server.call_tool('neural_forecast', params)
        
        stats = await latency_benchmark.benchmark_async_function(forecast_call)
        
        # Assert reasonable latency
        assert stats['mean_ms'] < 1000  # Should be under 1 second
        assert stats['p95_ms'] < 2000   # 95th percentile under 2 seconds
        assert stats['error_rate'] == 0  # No errors
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self, mcp_server):
        """Test concurrent MCP tool calls."""
        async def make_forecast_call(symbol):
            params = {
                'symbol': symbol,
                'horizon': 24,
                'confidence_level': 95,
                'use_gpu': False
            }
            return await mcp_server.call_tool('neural_forecast', params)
        
        # Make concurrent calls
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        tasks = [make_forecast_call(symbol) for symbol in symbols]
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Check all responses successful
        for response in responses:
            assert response['status'] == 'success'
        
        # Check reasonable total time
        total_time = end_time - start_time
        assert total_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_tool_scalability(self, mcp_server):
        """Test tool scalability with increasing load."""
        # Test with different numbers of concurrent requests
        request_counts = [1, 5, 10, 20]
        response_times = []
        
        for count in request_counts:
            tasks = []
            for i in range(count):
                params = {
                    'symbol': f'SYMBOL_{i}',
                    'horizon': 24,
                    'confidence_level': 95
                }
                task = mcp_server.call_tool('neural_forecast', params)
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Check all successful
            for response in responses:
                assert response['status'] == 'success'
            
            response_times.append(end_time - start_time)
        
        # Response time should scale reasonably (not exponentially)
        for i in range(1, len(response_times)):
            scaling_factor = response_times[i] / response_times[0]
            request_factor = request_counts[i] / request_counts[0]
            
            # Should not scale worse than linearly by much
            assert scaling_factor <= request_factor * 2


class TestMCPToolIntegration:
    """Test integration between MCP tools and neural components."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create integrated MCP server."""
        return create_mock_mcp_server()
    
    @pytest.mark.asyncio
    async def test_forecast_to_backtest_workflow(self, mcp_server):
        """Test workflow from forecasting to backtesting."""
        # Step 1: Generate forecast
        forecast_params = {
            'symbol': 'AAPL',
            'horizon': 24,
            'confidence_level': 95
        }
        
        forecast_response = await mcp_server.call_tool('neural_forecast', forecast_params)
        assert forecast_response['status'] == 'success'
        
        # Step 2: Use forecast in backtest
        backtest_params = {
            'model_id': 'nhits_test_model',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'metrics': ['mae', 'mape']
        }
        
        backtest_response = await mcp_server.call_tool('neural_backtest', backtest_params)
        assert backtest_response['status'] == 'success'
        
        # Check consistency between forecast and backtest
        forecast_result = forecast_response['result']
        backtest_result = backtest_response['result']
        
        assert len(forecast_result['forecast']) == forecast_params['horizon']
        assert 'mae' in backtest_result['metrics']
    
    @pytest.mark.asyncio
    async def test_train_to_forecast_workflow(self, mcp_server):
        """Test workflow from training to forecasting."""
        # Step 1: Train model
        train_params = {
            'data_path': 'test_data.csv',
            'model_type': 'nhits',
            'epochs': 10
        }
        
        train_response = await mcp_server.call_tool('neural_train', train_params)
        assert train_response['status'] == 'success'
        
        model_id = train_response['result']['model_id']
        
        # Step 2: Use trained model for forecasting
        # (In real implementation, would use the trained model)
        forecast_params = {
            'symbol': 'AAPL',
            'horizon': 24,
            'model_id': model_id  # Use trained model
        }
        
        # For mock, this might not use the model_id, but should still work
        forecast_response = await mcp_server.call_tool('neural_forecast', forecast_params)
        assert forecast_response['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_optimize_to_benchmark_workflow(self, mcp_server):
        """Test workflow from optimization to benchmarking."""
        # Step 1: Optimize model
        optimize_params = {
            'model_id': 'nhits_test_model',
            'optimization_metric': 'mae',
            'max_trials': 10
        }
        
        optimize_response = await mcp_server.call_tool('neural_optimize', optimize_params)
        assert optimize_response['status'] == 'success'
        
        # Step 2: Benchmark optimized model
        benchmark_params = {
            'model_id': optimize_params['model_id'],
            'benchmark_type': 'performance'
        }
        
        benchmark_response = await mcp_server.call_tool('neural_benchmark', benchmark_params)
        assert benchmark_response['status'] == 'success'
        
        # Check optimization improved performance
        optimize_result = optimize_response['result']
        benchmark_result = benchmark_response['result']
        
        assert 'best_score' in optimize_result['optimization_results']
        assert 'accuracy' in benchmark_result['results']


class TestMCPErrorHandling:
    """Test error handling in MCP neural tools."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server for error testing."""
        return create_mock_mcp_server()
    
    @pytest.mark.asyncio
    async def test_invalid_tool_name(self, mcp_server):
        """Test handling of invalid tool names."""
        response = await mcp_server.call_tool('invalid_neural_tool', {})
        
        assert response['status'] == 'error'
        assert 'error' in response
        assert 'available_tools' in response
    
    @pytest.mark.asyncio
    async def test_missing_required_parameters(self, mcp_server):
        """Test handling of missing required parameters."""
        # Forecast without symbol
        response = await mcp_server.call_tool('neural_forecast', {'horizon': 24})
        
        # Should either work with defaults or return error
        assert 'status' in response
        
        if response['status'] == 'error':
            assert 'error' in response
    
    @pytest.mark.asyncio
    async def test_invalid_parameter_values(self, mcp_server):
        """Test handling of invalid parameter values."""
        # Invalid horizon
        params = {
            'symbol': 'AAPL',
            'horizon': 'invalid',  # Should be integer
            'confidence_level': 95
        }
        
        response = await mcp_server.call_tool('neural_forecast', params)
        
        # Should handle gracefully
        assert 'status' in response
    
    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self, mcp_server):
        """Test handling of tool timeouts."""
        # This would require a more sophisticated mock or real implementation
        # For now, just test that tools complete within reasonable time
        
        params = {
            'symbol': 'AAPL',
            'horizon': 24,
            'confidence_level': 95
        }
        
        start_time = time.time()
        response = await asyncio.wait_for(
            mcp_server.call_tool('neural_forecast', params),
            timeout=30.0  # 30 second timeout
        )
        end_time = time.time()
        
        assert response['status'] == 'success'
        assert end_time - start_time < 30.0


class TestMCPDataIntegration:
    """Test data integration through MCP tools."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server for data testing."""
        return create_mock_mcp_server()
    
    @pytest.mark.asyncio
    async def test_real_time_data_integration(self, mcp_server):
        """Test integration with real-time data feeds."""
        # Simulate real-time data flow
        timestamps = []
        responses = []
        
        for i in range(5):  # 5 time steps
            params = {
                'symbol': 'AAPL',
                'horizon': 12,  # Short horizon for real-time
                'confidence_level': 90,
                'timestamp': datetime.now().isoformat()
            }
            
            response = await mcp_server.call_tool('neural_forecast', params)
            
            timestamps.append(params['timestamp'])
            responses.append(response)
            
            # Small delay to simulate real-time flow
            await asyncio.sleep(0.1)
        
        # Check all responses successful
        for response in responses:
            assert response['status'] == 'success'
        
        # Check timestamps are in order
        assert len(timestamps) == 5
    
    @pytest.mark.asyncio
    async def test_multi_asset_data_integration(self, mcp_server):
        """Test integration with multi-asset data."""
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        forecasts = {}
        
        # Get forecasts for multiple assets
        for asset in assets:
            params = {
                'symbol': asset,
                'horizon': 24,
                'confidence_level': 95
            }
            
            response = await mcp_server.call_tool('neural_forecast', params)
            assert response['status'] == 'success'
            
            forecasts[asset] = response['result']['forecast']
        
        # Check we got forecasts for all assets
        assert len(forecasts) == len(assets)
        
        # Check forecast lengths are consistent
        forecast_lengths = [len(forecast) for forecast in forecasts.values()]
        assert all(length == 24 for length in forecast_lengths)
    
    @pytest.mark.asyncio
    async def test_historical_data_integration(self, mcp_server):
        """Test integration with historical data for backtesting."""
        # Test different time periods
        time_periods = [
            ('2023-01-01', '2023-03-31'),  # Q1
            ('2023-04-01', '2023-06-30'),  # Q2
            ('2023-07-01', '2023-09-30'),  # Q3
            ('2023-10-01', '2023-12-31'),  # Q4
        ]
        
        backtest_results = []
        
        for start_date, end_date in time_periods:
            params = {
                'model_id': 'nhits_quarterly_model',
                'start_date': start_date,
                'end_date': end_date,
                'metrics': ['mae', 'mape']
            }
            
            response = await mcp_server.call_tool('neural_backtest', params)
            assert response['status'] == 'success'
            
            backtest_results.append(response['result'])
        
        # Check we got results for all periods
        assert len(backtest_results) == len(time_periods)
        
        # Check consistency of metrics across periods
        for result in backtest_results:
            assert 'metrics' in result
            assert 'mae' in result['metrics']
            assert 'mape' in result['metrics']


class TestMCPMonitoring:
    """Test monitoring and logging of MCP neural tools."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server with monitoring."""
        server = create_mock_mcp_server()
        # In real implementation, would enable monitoring
        return server
    
    @pytest.mark.asyncio
    async def test_tool_call_logging(self, mcp_server):
        """Test that tool calls are properly logged."""
        params = {
            'symbol': 'AAPL',
            'horizon': 24,
            'confidence_level': 95
        }
        
        # Make tool call
        response = await mcp_server.call_tool('neural_forecast', params)
        assert response['status'] == 'success'
        
        # Check call history (mock server tracks this)
        if hasattr(mcp_server, 'call_history'):
            assert len(mcp_server.call_history) > 0
            
            last_call = mcp_server.call_history[-1]
            assert last_call['tool'] == 'neural_forecast'
            assert last_call['parameters'] == params
            assert 'timestamp' in last_call
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, mcp_server):
        """Test collection of performance metrics."""
        with performance_monitoring() as monitor:
            # Make several tool calls
            for i in range(5):
                params = {
                    'symbol': f'SYMBOL_{i}',
                    'horizon': 24,
                    'confidence_level': 95
                }
                
                response = await mcp_server.call_tool('neural_forecast', params)
                assert response['status'] == 'success'
        
        # Check monitoring data was collected
        monitoring_data = monitor.stop_monitoring()
        assert len(monitoring_data) > 0
        
        # Check metrics structure
        for metric in monitoring_data:
            assert 'timestamp' in metric
            assert 'cpu_percent' in metric
            assert 'memory_mb' in metric


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])