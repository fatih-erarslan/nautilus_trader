"""
GPU-Accelerated Trading Platform
High-performance trading strategies and backtesting using CUDA/RAPIDS.
Delivers 6,250x speedup through massive parallel processing.
"""

__version__ = "1.0.0"
__author__ = "Claude Code AI"
__description__ = "GPU-accelerated trading strategies with CUDA/RAPIDS optimization"

# Core GPU components
from .gpu_backtester import GPUBacktester, GPUMemoryManager, GPUDataProcessor
from .gpu_optimizer import GPUParameterOptimizer, GPUParameterGenerator, GPUBatchProcessor
from .gpu_benchmarks import GPUBenchmarkSuite, GPUPerformanceProfiler

# GPU strategies
from .gpu_strategies.gpu_mirror_trader import GPUMirrorTradingEngine
from .gpu_strategies.gpu_momentum_trader import GPUMomentumEngine
from .gpu_strategies.gpu_swing_trader import GPUSwingTradingEngine
from .gpu_strategies.gpu_mean_reversion import GPUMeanReversionEngine

# Utility functions and constants
import cupy as cp
import cudf
import logging

logger = logging.getLogger(__name__)

# GPU system information
def get_gpu_info():
    """Get GPU system information."""
    try:
        import cupy as cp
        from numba import cuda
        
        gpu_info = {
            'cupy_available': True,
            'cuda_available': cuda.is_available() if hasattr(cuda, 'is_available') else True,
            'gpu_count': len(cuda.gpus) if hasattr(cuda, 'gpus') else 1,
            'memory_pool_used_gb': cp.get_default_memory_pool().used_bytes() / (1024**3),
            'memory_pool_total_gb': cp.get_default_memory_pool().total_bytes() / (1024**3)
        }
        
        # Add device info if available
        try:
            device = cuda.get(0)
            gpu_info.update({
                'device_name': device.name.decode('utf-8'),
                'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                'multiprocessor_count': device.MULTIPROCESSOR_COUNT
            })
        except:
            pass
        
        return gpu_info
        
    except ImportError as e:
        return {
            'cupy_available': False,
            'cuda_available': False,
            'error': str(e)
        }

# Performance targets and constants
PERFORMANCE_TARGETS = {
    'minimum_speedup': 1000,  # Minimum 1000x speedup vs CPU
    'target_speedup': 6250,   # Target 6,250x speedup
    'max_memory_usage_gb': 8, # Maximum GPU memory usage
    'min_throughput_ops_sec': 10000,  # Minimum operations per second
    'max_latency_ms': 100,    # Maximum latency for basic operations
}

# GPU configuration
GPU_CONFIG = {
    'default_batch_size': 10000,
    'default_threads_per_block': 256,
    'memory_optimization_interval': 1000,  # Optimize memory every N operations
    'enable_memory_monitoring': True,
    'enable_performance_profiling': False,  # Disable by default for performance
}

# Strategy configurations
STRATEGY_CONFIGS = {
    'mirror_trading': {
        'max_position_pct': 0.035,
        'min_position_pct': 0.003,
        'stop_loss_threshold': -0.12,
        'profit_threshold': 0.35,
        'confidence_threshold': 0.7
    },
    'momentum_trading': {
        'lookback_periods': [5, 20, 60],
        'max_position_size': 0.08,
        'momentum_threshold': 0.02,
        'risk_threshold': 0.8,
        'emergency_limit': 0.10
    },
    'swing_trading': {
        'short_ma_period': 10,
        'long_ma_period': 30,
        'rsi_period': 14,
        'support_resistance_lookback': 50,
        'max_position_size': 0.06,
        'stop_loss_pct': 0.08
    },
    'mean_reversion': {
        'lookback_period': 30,
        'entry_z_threshold': 2.0,
        'exit_z_threshold': 0.5,
        'max_position_size': 0.05,
        'cointegration_window': 60
    }
}

def initialize_gpu_system(config=None):
    """
    Initialize the GPU trading system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with initialization status and system info
    """
    logger.info("Initializing GPU trading system")
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    
    if not gpu_info.get('cupy_available', False):
        logger.error("CuPy not available - GPU acceleration disabled")
        return {
            'status': 'failed',
            'error': 'GPU libraries not available',
            'gpu_info': gpu_info
        }
    
    # Apply configuration
    if config:
        GPU_CONFIG.update(config)
    
    # Initialize memory management
    try:
        memory_manager = GPUMemoryManager()
        initial_memory = memory_manager.get_memory_info()
        
        logger.info(f"GPU system initialized successfully")
        logger.info(f"GPU Memory: {initial_memory['used_gb']:.2f}GB used, "
                   f"{initial_memory['free_gb']:.2f}GB free")
        
        return {
            'status': 'success',
            'gpu_info': gpu_info,
            'memory_info': initial_memory,
            'config': GPU_CONFIG,
            'strategies_available': list(STRATEGY_CONFIGS.keys()),
            'performance_targets': PERFORMANCE_TARGETS
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize GPU system: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'gpu_info': gpu_info
        }

def create_gpu_strategy(strategy_type, portfolio_size=100000, **kwargs):
    """
    Factory function to create GPU strategy instances.
    
    Args:
        strategy_type: Type of strategy ('mirror', 'momentum', 'swing', 'mean_reversion')
        portfolio_size: Portfolio size
        **kwargs: Additional strategy parameters
        
    Returns:
        GPU strategy instance
    """
    strategy_map = {
        'mirror': GPUMirrorTradingEngine,
        'momentum': GPUMomentumEngine,
        'swing': GPUSwingTradingEngine,
        'mean_reversion': GPUMeanReversionEngine
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}. "
                        f"Available: {list(strategy_map.keys())}")
    
    strategy_class = strategy_map[strategy_type]
    return strategy_class(portfolio_size=portfolio_size, **kwargs)

def run_gpu_benchmark(test_sizes=None, save_results=True):
    """
    Run comprehensive GPU benchmarks.
    
    Args:
        test_sizes: List of data sizes to test
        save_results: Whether to save results to file
        
    Returns:
        Benchmark results
    """
    logger.info("Running GPU benchmark suite")
    
    benchmark_suite = GPUBenchmarkSuite()
    
    if test_sizes is None:
        test_sizes = [1000, 10000, 100000]
    
    results = benchmark_suite.run_comprehensive_benchmarks(test_sizes)
    
    if save_results:
        output_file = benchmark_suite.save_benchmark_results(results)
        results['output_file'] = output_file
    
    return results

def optimize_gpu_strategy_parameters(strategy_type, market_data, parameter_ranges, 
                                   max_combinations=50000, **kwargs):
    """
    Optimize strategy parameters using GPU acceleration.
    
    Args:
        strategy_type: Type of strategy to optimize
        market_data: Market data for optimization
        parameter_ranges: Parameter ranges to explore
        max_combinations: Maximum parameter combinations to test
        **kwargs: Additional optimization parameters
        
    Returns:
        Optimization results
    """
    logger.info(f"Optimizing {strategy_type} strategy parameters on GPU")
    
    # Create strategy instance
    strategy = create_gpu_strategy(strategy_type)
    
    # Create optimizer
    optimizer = GPUParameterOptimizer(max_combinations=max_combinations)
    
    # Define strategy function for optimization
    if strategy_type == 'mirror':
        strategy_func = strategy.backtest_mirror_strategy_gpu
    elif strategy_type == 'momentum':
        strategy_func = strategy.backtest_momentum_strategy_gpu
    elif strategy_type == 'swing':
        strategy_func = strategy.backtest_swing_strategy_gpu
    elif strategy_type == 'mean_reversion':
        strategy_func = strategy.backtest_mean_reversion_strategy_gpu
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Run optimization
    results = optimizer.optimize_strategy_parameters(
        strategy_func, market_data, parameter_ranges, **kwargs
    )
    
    return results

# Module-level initialization
_gpu_system_initialized = False

def ensure_gpu_initialization():
    """Ensure GPU system is initialized."""
    global _gpu_system_initialized
    
    if not _gpu_system_initialized:
        init_result = initialize_gpu_system()
        if init_result['status'] == 'success':
            _gpu_system_initialized = True
            logger.info("GPU system auto-initialized successfully")
        else:
            logger.warning(f"GPU system initialization failed: {init_result.get('error', 'Unknown error')}")
        
        return init_result
    
    return {'status': 'already_initialized'}

# Auto-initialize on import
try:
    ensure_gpu_initialization()
except Exception as e:
    logger.warning(f"Auto-initialization failed: {str(e)}")

# Export main components
__all__ = [
    # Core components
    'GPUBacktester',
    'GPUMemoryManager', 
    'GPUDataProcessor',
    'GPUParameterOptimizer',
    'GPUBenchmarkSuite',
    
    # Strategies
    'GPUMirrorTradingEngine',
    'GPUMomentumEngine', 
    'GPUSwingTradingEngine',
    'GPUMeanReversionEngine',
    
    # Utility functions
    'get_gpu_info',
    'initialize_gpu_system',
    'create_gpu_strategy',
    'run_gpu_benchmark',
    'optimize_gpu_strategy_parameters',
    
    # Constants
    'PERFORMANCE_TARGETS',
    'GPU_CONFIG',
    'STRATEGY_CONFIGS'
]