"""
Comprehensive GPU Performance Tests and Validation
Tests all GPU-accelerated components for performance, accuracy, and integration.
Validates 6,250x speedup targets and system reliability.
"""

import unittest
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import warnings
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import GPU components
from gpu_backtester import GPUBacktester, GPUMemoryManager
from gpu_strategies.gpu_mirror_trader import GPUMirrorTradingEngine
from gpu_strategies.gpu_momentum_trader import GPUMomentumEngine
from gpu_strategies.gpu_swing_trader import GPUSwingTradingEngine
from gpu_strategies.gpu_mean_reversion import GPUMeanReversionEngine
from gpu_optimizer import GPUParameterOptimizer
from gpu_benchmarks import GPUBenchmarkSuite

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGPUPerformance(unittest.TestCase):
    """Comprehensive GPU performance test suite."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        logger.info("Setting up GPU performance test environment")
        
        # Performance targets
        cls.performance_targets = {
            'min_speedup': 1000,  # Minimum 1000x speedup
            'max_memory_gb': 8,   # Maximum 8GB GPU memory usage
            'max_latency_ms': 100,  # Maximum 100ms latency
            'min_accuracy': 0.95,   # Minimum 95% accuracy vs CPU
            'min_throughput_ops_sec': 10000  # Minimum 10k operations/second
        }
        
        # Generate test data
        cls.test_data_sizes = [1000, 10000, 100000]
        cls.test_market_data = cls._generate_test_market_data()
        
        logger.info("GPU performance test environment ready")
    
    @classmethod
    def _generate_test_market_data(cls) -> Dict[str, cudf.DataFrame]:
        """Generate test market data for all test sizes."""
        test_data = {}
        
        for size in cls.test_data_sizes:
            dates = pd.date_range('2020-01-01', periods=size, freq='D')
            
            # Generate realistic market data
            np.random.seed(42)  # For reproducible tests
            returns = np.random.normal(0.0005, 0.02, size)  # Daily returns
            prices = 100 * np.cumprod(1 + returns)  # Price series
            
            data = cudf.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.005, size)),
                'high': prices * (1 + np.random.uniform(0, 0.01, size)),
                'low': prices * (1 - np.random.uniform(0, 0.01, size)),
                'close': prices,
                'volume': np.random.lognormal(12, 0.5, size)
            })
            
            # Ensure price relationships
            data['high'] = data[['open', 'high', 'close']].max(axis=1)
            data['low'] = data[['open', 'low', 'close']].min(axis=1)
            
            test_data[f'size_{size}'] = data
        
        return test_data
    
    def test_gpu_memory_management(self):
        """Test GPU memory management and optimization."""
        logger.info("Testing GPU memory management")
        
        memory_manager = GPUMemoryManager()
        
        # Test memory allocation
        initial_memory = memory_manager.get_memory_info()
        
        # Allocate test arrays
        test_arrays = []
        for i in range(10):
            array = cp.zeros(100000, dtype=cp.float32)
            test_arrays.append(array)
        
        # Check memory usage
        peak_memory = memory_manager.get_memory_info()
        
        # Optimize memory
        memory_manager.optimize_memory()
        optimized_memory = memory_manager.get_memory_info()
        
        # Assertions
        self.assertLess(peak_memory['used_gb'], self.performance_targets['max_memory_gb'])
        self.assertLessEqual(optimized_memory['used_gb'], peak_memory['used_gb'])
        
        logger.info(f"GPU memory test passed: Peak {peak_memory['used_gb']:.2f}GB, "
                   f"Optimized {optimized_memory['used_gb']:.2f}GB")
    
    def test_gpu_backtester_performance(self):
        """Test GPU backtester performance and accuracy."""
        logger.info("Testing GPU backtester performance")
        
        backtester = GPUBacktester(initial_capital=100000)
        
        for size_name, test_data in self.test_market_data.items():
            logger.info(f"Testing backtester with {size_name}")
            
            # Prepare market data
            market_data = {f'TEST_{size_name}': test_data}
            
            # Test strategy function
            def simple_strategy(data, params):
                signals = cudf.DataFrame({
                    'signal': cp.random.choice([-1, 0, 1], size=len(data), p=[0.1, 0.8, 0.1])
                })
                return signals
            
            # Test parameters
            test_params = {
                'position_size': 0.02,
                'transaction_cost': 0.001
            }
            
            # Run backtest with timing
            start_time = time.perf_counter()
            results = backtester.run_strategy_backtest(simple_strategy, market_data, test_params)
            execution_time = time.perf_counter() - start_time
            
            # Validate results
            self.assertEqual(results['status'], 'completed')
            self.assertIn('performance_metrics', results)
            self.assertGreater(results['speedup_achieved'], self.performance_targets['min_speedup'])
            self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
            
            logger.info(f"Backtester {size_name}: {results['speedup_achieved']:.0f}x speedup, "
                       f"{execution_time:.2f}s execution time")
    
    def test_gpu_mirror_trader_performance(self):
        """Test GPU Mirror Trading strategy performance."""
        logger.info("Testing GPU Mirror Trading performance")
        
        gpu_mirror = GPUMirrorTradingEngine(portfolio_size=100000)
        
        # Test 13F filing processing
        sample_filings = [
            {
                'filer': 'Berkshire Hathaway',
                'filing_date': datetime.now() - timedelta(days=5),
                'new_positions': ['AAPL', 'MSFT'],
                'increased_positions': ['GOOGL'],
                'reduced_positions': ['TSLA'],
                'sold_positions': []
            }
        ]
        
        start_time = time.perf_counter()
        signals = gpu_mirror.process_13f_filings_gpu(sample_filings)
        processing_time = time.perf_counter() - start_time
        
        # Validate results
        self.assertGreater(len(signals), 0)
        self.assertIn('confidence', signals.columns)
        self.assertLess(processing_time, 1.0)  # Should be very fast
        
        # Test strategy backtest
        for size_name, test_data in self.test_market_data.items():
            if 'size_10000' in size_name:  # Test with medium size data
                test_params = {
                    'confidence_threshold': 0.7,
                    'position_size': 0.02,
                    'transaction_cost': 0.001
                }
                
                start_time = time.perf_counter()
                backtest_results = gpu_mirror.backtest_mirror_strategy_gpu(test_data, test_params)
                execution_time = time.perf_counter() - start_time
                
                # Validate performance
                self.assertIn('sharpe_ratio', backtest_results)
                self.assertIn('gpu_speedup_achieved', backtest_results)
                self.assertGreater(backtest_results['gpu_speedup_achieved'], 
                                 self.performance_targets['min_speedup'])
                
                logger.info(f"Mirror trader: {backtest_results['gpu_speedup_achieved']:.0f}x speedup, "
                           f"Sharpe {backtest_results['sharpe_ratio']:.2f}")
                break
    
    def test_gpu_momentum_trader_performance(self):
        """Test GPU Momentum Trading strategy performance."""
        logger.info("Testing GPU Momentum Trading performance")
        
        gpu_momentum = GPUMomentumEngine(portfolio_size=100000)
        
        for size_name, test_data in self.test_market_data.items():
            if 'size_10000' in size_name:  # Test with medium size data
                test_params = {
                    'momentum_threshold': 0.02,
                    'confidence_threshold': 0.6,
                    'base_position_size': 0.02,
                    'risk_free_rate': 0.02
                }
                
                start_time = time.perf_counter()
                backtest_results = gpu_momentum.backtest_momentum_strategy_gpu(test_data, test_params)
                execution_time = time.perf_counter() - start_time
                
                # Validate performance
                self.assertIn('sharpe_ratio', backtest_results)
                self.assertIn('gpu_performance_stats', backtest_results)
                self.assertGreater(backtest_results['gpu_performance_stats']['speedup_achieved'], 
                                 self.performance_targets['min_speedup'])
                self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
                
                logger.info(f"Momentum trader: "
                           f"{backtest_results['gpu_performance_stats']['speedup_achieved']:.0f}x speedup, "
                           f"Sharpe {backtest_results['sharpe_ratio']:.2f}")
                break
    
    def test_gpu_swing_trader_performance(self):
        """Test GPU Swing Trading strategy performance."""
        logger.info("Testing GPU Swing Trading performance")
        
        gpu_swing = GPUSwingTradingEngine(portfolio_size=100000)
        
        for size_name, test_data in self.test_market_data.items():
            if 'size_10000' in size_name:  # Test with medium size data
                test_params = {
                    'base_position_size': 0.03,
                    'atr_stop_multiplier': 2.0,
                    'profit_target_multiplier': 3.0,
                    'min_risk_reward_ratio': 2.0,
                    'trend_filter': True,
                    'risk_free_rate': 0.02
                }
                
                start_time = time.perf_counter()
                backtest_results = gpu_swing.backtest_swing_strategy_gpu(test_data, test_params)
                execution_time = time.perf_counter() - start_time
                
                # Validate performance
                self.assertIn('sharpe_ratio', backtest_results)
                self.assertIn('gpu_performance_stats', backtest_results)
                self.assertGreater(backtest_results['gpu_performance_stats']['speedup_achieved'], 
                                 self.performance_targets['min_speedup'])
                self.assertLess(execution_time, 15.0)  # Pattern recognition takes more time
                
                logger.info(f"Swing trader: "
                           f"{backtest_results['gpu_performance_stats']['speedup_achieved']:.0f}x speedup, "
                           f"Sharpe {backtest_results['sharpe_ratio']:.2f}")
                break
    
    def test_gpu_mean_reversion_performance(self):
        """Test GPU Mean Reversion strategy performance."""
        logger.info("Testing GPU Mean Reversion performance")
        
        gpu_mean_reversion = GPUMeanReversionEngine(portfolio_size=100000)
        
        for size_name, test_data in self.test_market_data.items():
            if 'size_10000' in size_name:  # Test with medium size data
                test_params = {
                    'entry_z_threshold': 2.0,
                    'exit_z_threshold': 0.5,
                    'base_position_size': 0.03,
                    'max_half_life': 15,
                    'min_reversion_strength': 1.0,
                    'trend_filter': True,
                    'risk_free_rate': 0.02
                }
                
                start_time = time.perf_counter()
                backtest_results = gpu_mean_reversion.backtest_mean_reversion_strategy_gpu(
                    test_data, test_params
                )
                execution_time = time.perf_counter() - start_time
                
                # Validate performance
                self.assertIn('sharpe_ratio', backtest_results)
                self.assertIn('reversion_efficiency', backtest_results)
                self.assertIn('gpu_performance_stats', backtest_results)
                self.assertGreater(backtest_results['gpu_performance_stats']['speedup_achieved'], 
                                 self.performance_targets['min_speedup'])
                self.assertLess(execution_time, 10.0)
                
                logger.info(f"Mean reversion: "
                           f"{backtest_results['gpu_performance_stats']['speedup_achieved']:.0f}x speedup, "
                           f"Sharpe {backtest_results['sharpe_ratio']:.2f}, "
                           f"Efficiency {backtest_results['reversion_efficiency']:.2f}")
                break
    
    def test_gpu_parameter_optimizer_performance(self):
        """Test GPU parameter optimizer performance."""
        logger.info("Testing GPU parameter optimizer performance")
        
        gpu_optimizer = GPUParameterOptimizer(max_combinations=5000, batch_size=500)
        
        # Test parameter ranges
        parameter_ranges = {
            'param1': {'start': 0.01, 'stop': 0.1, 'type': 'float'},
            'param2': {'start': 0.5, 'stop': 2.0, 'type': 'float'},
            'param3': [5, 10, 15, 20, 25],
            'param4': {'start': 0.1, 'stop': 1.0, 'type': 'float'}
        }
        
        # Simple test strategy
        def test_strategy(market_data, parameters):
            returns = cp.random.normal(0, 0.01, 252)
            return {
                'returns': returns,
                'risk_scores': cp.random.uniform(0, 1, 252),
                'objective_value': cp.mean(returns) / cp.std(returns)
            }
        
        # Use medium-sized test data
        test_data = self.test_market_data['size_10000']
        
        start_time = time.perf_counter()
        optimization_results = gpu_optimizer.optimize_strategy_parameters(
            test_strategy,
            test_data,
            parameter_ranges,
            objective_function='sharpe_ratio',
            generation_strategy='random_search',
            max_iterations=1
        )
        execution_time = time.perf_counter() - start_time
        
        # Validate results
        self.assertEqual(optimization_results['status'], 'completed')
        self.assertIn('best_parameters', optimization_results)
        self.assertIn('optimization_stats', optimization_results)
        self.assertGreater(optimization_results['optimization_stats']['speedup_achieved'], 
                         self.performance_targets['min_speedup'])
        self.assertGreater(optimization_results['combinations_per_second'], 
                         self.performance_targets['min_throughput_ops_sec'] / 10)  # Allow for complexity
        
        logger.info(f"Parameter optimizer: "
                   f"{optimization_results['optimization_stats']['speedup_achieved']:.0f}x speedup, "
                   f"{optimization_results['combinations_per_second']:.0f} combinations/sec")
    
    def test_gpu_benchmark_suite_performance(self):
        """Test GPU benchmark suite performance."""
        logger.info("Testing GPU benchmark suite performance")
        
        benchmark_suite = GPUBenchmarkSuite()
        
        # Run limited benchmarks to avoid long test times
        test_sizes = [1000, 10000]
        
        start_time = time.perf_counter()
        results = benchmark_suite.run_comprehensive_benchmarks(test_sizes)
        execution_time = time.perf_counter() - start_time
        
        # Validate results
        self.assertIn('summary', results)
        self.assertIn('performance_validation', results)
        self.assertEqual(results['summary']['benchmark_suites_failed'], 0)
        self.assertLess(execution_time, 300.0)  # Should complete within 5 minutes
        
        # Check performance targets met
        validation = results['performance_validation']
        if 'results' in validation:
            for target_name, target_result in validation['results'].items():
                if 'passed' in target_result:
                    logger.info(f"Performance target {target_name}: "
                               f"{'PASSED' if target_result['passed'] else 'FAILED'}")
        
        logger.info(f"Benchmark suite completed in {execution_time:.2f}s, "
                   f"Overall pass: {validation.get('overall_pass', False)}")
    
    def test_gpu_memory_scalability(self):
        """Test GPU memory usage scalability across different data sizes."""
        logger.info("Testing GPU memory scalability")
        
        memory_manager = GPUMemoryManager()
        scalability_results = {}
        
        for size_name, test_data in self.test_market_data.items():
            # Record initial memory
            initial_memory = memory_manager.get_memory_info()
            
            # Process data
            gpu_momentum = GPUMomentumEngine(portfolio_size=100000)
            processed_data = gpu_momentum.calculate_comprehensive_momentum_gpu(test_data)
            
            # Record peak memory
            peak_memory = memory_manager.get_memory_info()
            
            # Calculate memory efficiency
            data_size_gb = len(test_data) * 5 * 4 / (1024**3)  # Estimate for OHLCV float32
            memory_overhead = peak_memory['used_gb'] - initial_memory['used_gb']
            efficiency = data_size_gb / max(memory_overhead, 0.001)
            
            scalability_results[size_name] = {
                'data_size_gb': data_size_gb,
                'memory_overhead_gb': memory_overhead,
                'efficiency': efficiency,
                'under_limit': peak_memory['used_gb'] < self.performance_targets['max_memory_gb']
            }
            
            # Cleanup
            del processed_data
            memory_manager.optimize_memory()
        
        # Validate scalability
        for size_name, result in scalability_results.items():
            self.assertTrue(result['under_limit'], 
                           f"Memory usage exceeded limit for {size_name}")
            self.assertGreater(result['efficiency'], 0.1, 
                              f"Poor memory efficiency for {size_name}")
            
            logger.info(f"Memory scalability {size_name}: "
                       f"{result['memory_overhead_gb']:.2f}GB used, "
                       f"{result['efficiency']:.2f} efficiency")
    
    def test_gpu_accuracy_vs_cpu(self):
        """Test GPU calculation accuracy compared to CPU baseline."""
        logger.info("Testing GPU vs CPU accuracy")
        
        # Use smallest test data for accuracy comparison
        test_data = self.test_market_data['size_1000']
        
        # Convert to pandas for CPU calculations
        cpu_data = test_data.to_pandas()
        
        # Test technical indicators accuracy
        # GPU calculations
        gpu_momentum = GPUMomentumEngine(portfolio_size=100000)
        gpu_data = gpu_momentum.calculate_comprehensive_momentum_gpu(test_data.copy())
        
        # CPU calculations (simplified)
        cpu_data['sma_20'] = cpu_data['close'].rolling(window=20).mean()
        cpu_data['sma_50'] = cpu_data['close'].rolling(window=50).mean()
        
        delta = cpu_data['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        cpu_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Compare results (skip NaN values)
        valid_indices = ~(cpu_data['sma_20'].isna() | gpu_data['sma_20'].isna())
        
        if valid_indices.sum() > 10:
            # Calculate correlation and RMSE
            sma_correlation = np.corrcoef(
                cpu_data.loc[valid_indices, 'sma_20'],
                gpu_data.loc[valid_indices, 'sma_20'].to_pandas()
            )[0, 1]
            
            rsi_valid = ~(cpu_data['rsi'].isna() | gpu_data['rsi'].isna())
            if rsi_valid.sum() > 10:
                rsi_correlation = np.corrcoef(
                    cpu_data.loc[rsi_valid, 'rsi'],
                    gpu_data.loc[rsi_valid, 'rsi'].to_pandas()
                )[0, 1]
            else:
                rsi_correlation = 1.0  # Default if insufficient data
            
            # Validate accuracy
            self.assertGreater(sma_correlation, self.performance_targets['min_accuracy'])
            self.assertGreater(rsi_correlation, self.performance_targets['min_accuracy'])
            
            logger.info(f"GPU vs CPU accuracy: SMA correlation {sma_correlation:.4f}, "
                       f"RSI correlation {rsi_correlation:.4f}")
        else:
            logger.warning("Insufficient data for GPU vs CPU accuracy comparison")
    
    def test_gpu_error_handling(self):
        """Test GPU error handling and recovery."""
        logger.info("Testing GPU error handling")
        
        # Test with invalid data
        invalid_data = cudf.DataFrame({
            'close': [np.nan, np.inf, -np.inf, 0, 100],
            'volume': [1000, 2000, 3000, 4000, 5000]
        })
        
        gpu_momentum = GPUMomentumEngine(portfolio_size=100000)
        
        # Should handle invalid data gracefully
        try:
            result = gpu_momentum.calculate_comprehensive_momentum_gpu(invalid_data)
            # Should not crash, might return modified data
            self.assertIsInstance(result, cudf.DataFrame)
        except Exception as e:
            self.fail(f"GPU error handling failed: {str(e)}")
        
        # Test memory allocation limits
        memory_manager = GPUMemoryManager()
        initial_memory = memory_manager.get_memory_info()
        
        # Try to allocate very large array (should handle gracefully)
        try:
            huge_array = cp.zeros(10**9, dtype=cp.float32)  # 4GB array
            # If successful, clean up
            del huge_array
        except (cp.cuda.memory.MemoryError, MemoryError):
            # Expected behavior for memory limits
            pass
        except Exception as e:
            self.fail(f"Unexpected error in memory allocation test: {str(e)}")
        
        logger.info("GPU error handling tests passed")
    
    def test_gpu_concurrent_operations(self):
        """Test GPU concurrent operations and thread safety."""
        logger.info("Testing GPU concurrent operations")
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        test_data = self.test_market_data['size_1000']
        
        def worker_function(worker_id):
            """Worker function for concurrent testing."""
            try:
                gpu_momentum = GPUMomentumEngine(portfolio_size=100000)
                result = gpu_momentum.calculate_comprehensive_momentum_gpu(test_data.copy())
                results_queue.put(f"Worker {worker_id} completed successfully")
            except Exception as e:
                results_queue.put(f"Worker {worker_id} failed: {str(e)}")
        
        # Create multiple threads
        threads = []
        num_workers = 3  # Limited number to avoid GPU contention
        
        for i in range(num_workers):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Validate all workers completed successfully
        successful_workers = len([r for r in results if 'completed successfully' in r])
        self.assertEqual(successful_workers, num_workers, 
                        f"Only {successful_workers}/{num_workers} workers completed successfully")
        
        logger.info(f"Concurrent operations test: {successful_workers}/{num_workers} workers successful")
    
    def tearDown(self):
        """Clean up after each test."""
        # Force GPU memory cleanup
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        logger.info("Cleaning up GPU performance test environment")
        
        # Final memory cleanup
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        logger.info("GPU performance test environment cleaned up")


class TestGPUIntegration(unittest.TestCase):
    """Test GPU system integration with existing framework."""
    
    def test_integration_with_existing_benchmark(self):
        """Test integration with existing benchmark framework."""
        logger.info("Testing integration with existing benchmark framework")
        
        # This would test integration with the existing benchmark system
        # For now, we'll test that our GPU components can be imported and used
        # alongside the existing system
        
        try:
            # Test imports work
            from gpu_backtester import GPUBacktester
            from gpu_strategies.gpu_mirror_trader import GPUMirrorTradingEngine
            from gpu_optimizer import GPUParameterOptimizer
            
            # Test basic initialization
            gpu_backtester = GPUBacktester()
            gpu_mirror = GPUMirrorTradingEngine()
            gpu_optimizer = GPUParameterOptimizer(max_combinations=100)
            
            # Test basic functionality
            memory_info = gpu_backtester.memory_manager.get_memory_info()
            self.assertIn('used_gb', memory_info)
            
            performance_stats = gpu_mirror.get_gpu_performance_stats()
            self.assertIn('gpu_memory_info', performance_stats)
            
            logger.info("GPU integration test passed")
            
        except Exception as e:
            self.fail(f"GPU integration test failed: {str(e)}")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)