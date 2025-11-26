"""Latency benchmark implementation for AI News Trading platform."""

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LatencyResult:
    """Result container for latency measurements."""
    
    component: str
    operation: str
    samples: List[float]
    percentiles: Dict[str, float]
    mean: float
    std_dev: float
    min_latency: float
    max_latency: float
    total_operations: int
    duration: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "operation": self.operation,
            "samples": self.samples,
            "percentiles": self.percentiles,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "min": self.min_latency,
            "max": self.max_latency,
            "total_operations": self.total_operations,
            "duration": self.duration,
        }


class LatencyBenchmark:
    """Measures operation latency for various components."""
    
    def __init__(self, config):
        """Initialize latency benchmark."""
        self.config = config
        
        # Handle both ConfigManager and dict configurations
        if hasattr(config, 'config'):
            config_dict = config.config
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config if isinstance(config, dict) else {}
        
        # Extract configuration values with defaults
        metrics_config = config_dict.get("metrics", {}).get("latency", {})
        benchmark_config = config_dict.get("benchmark", {})
        
        self.percentiles = metrics_config.get("percentiles", [50, 90, 95, 99, 99.9])
        self.window_size = metrics_config.get("window_size", 1000)
        self.warmup_duration = benchmark_config.get("warmup_duration", 10)
        self.measurement_duration = benchmark_config.get("measurement_duration", 60)
    
    async def benchmark_signal_generation(self, strategy_name: str = "momentum") -> LatencyResult:
        """Benchmark signal generation latency."""
        return await self._benchmark_operation(
            component="signal_generator",
            operation="generate_signal",
            func=self._mock_signal_generation,
            args=(strategy_name,)
        )
    
    async def benchmark_order_execution(self, order_type: str = "market") -> LatencyResult:
        """Benchmark order execution latency."""
        return await self._benchmark_operation(
            component="order_executor",
            operation="execute_order",
            func=self._mock_order_execution,
            args=(order_type,)
        )
    
    async def benchmark_data_processing(self, data_size: int = 1000) -> LatencyResult:
        """Benchmark data processing latency."""
        return await self._benchmark_operation(
            component="data_processor",
            operation="process_data",
            func=self._mock_data_processing,
            args=(data_size,)
        )
    
    async def benchmark_risk_calculation(self, portfolio_size: int = 100) -> LatencyResult:
        """Benchmark risk calculation latency."""
        return await self._benchmark_operation(
            component="risk_calculator",
            operation="calculate_risk",
            func=self._mock_risk_calculation,
            args=(portfolio_size,)
        )
    
    async def _benchmark_operation(
        self,
        component: str,
        operation: str,
        func,
        args: tuple = ()
    ) -> LatencyResult:
        """Generic operation latency benchmark."""
        # Warmup phase
        warmup_start = time.time()
        while time.time() - warmup_start < self.warmup_duration:
            await func(*args)
        
        # Measurement phase
        latencies = []
        start_time = time.time()
        
        while time.time() - start_time < self.measurement_duration:
            op_start = time.perf_counter()
            await func(*args)
            op_end = time.perf_counter()
            
            latency_ms = (op_end - op_start) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)
        
        duration = time.time() - start_time
        
        # Calculate statistics
        percentile_values = {}
        for p in self.percentiles:
            percentile_values[f"p{p}"] = np.percentile(latencies, p)
        
        return LatencyResult(
            component=component,
            operation=operation,
            samples=latencies,
            percentiles=percentile_values,
            mean=statistics.mean(latencies),
            std_dev=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            min_latency=min(latencies),
            max_latency=max(latencies),
            total_operations=len(latencies),
            duration=duration
        )
    
    async def _mock_signal_generation(self, strategy_name: str):
        """Mock signal generation for benchmarking."""
        # Simulate signal generation work
        await asyncio.sleep(0.001 + np.random.exponential(0.002))
        
        # Simulate some CPU work
        x = np.random.randn(100)
        y = np.convolve(x, np.ones(10)/10, mode='valid')
        
        return {"signal": "buy" if np.mean(y) > 0 else "sell", "confidence": 0.75}
    
    async def _mock_order_execution(self, order_type: str):
        """Mock order execution for benchmarking."""
        # Simulate order execution latency
        base_latency = 0.005 if order_type == "market" else 0.010
        await asyncio.sleep(base_latency + np.random.exponential(0.003))
        
        return {"order_id": f"order_{int(time.time()*1000000)}", "status": "filled"}
    
    async def _mock_data_processing(self, data_size: int):
        """Mock data processing for benchmarking."""
        # Simulate data processing work
        await asyncio.sleep(0.0001 * data_size / 1000)
        
        # Simulate some computation
        data = np.random.randn(data_size)
        processed = np.fft.fft(data)
        
        return {"processed_samples": len(processed)}
    
    async def _mock_risk_calculation(self, portfolio_size: int):
        """Mock risk calculation for benchmarking."""
        # Simulate risk calculation complexity
        await asyncio.sleep(0.001 + 0.0001 * portfolio_size / 100)
        
        # Simulate risk matrix computation
        weights = np.random.randn(portfolio_size)
        covariance = np.random.randn(portfolio_size, portfolio_size)
        risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        
        return {"portfolio_risk": float(risk), "var_95": float(risk * 1.65)}
    
    def run_sync(self, benchmark_name: str, **kwargs) -> LatencyResult:
        """Run benchmark synchronously."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if benchmark_name == "signal_generation":
                return loop.run_until_complete(
                    self.benchmark_signal_generation(kwargs.get("strategy", "momentum"))
                )
            elif benchmark_name == "order_execution":
                return loop.run_until_complete(
                    self.benchmark_order_execution(kwargs.get("order_type", "market"))
                )
            elif benchmark_name == "data_processing":
                return loop.run_until_complete(
                    self.benchmark_data_processing(kwargs.get("data_size", 1000))
                )
            elif benchmark_name == "risk_calculation":
                return loop.run_until_complete(
                    self.benchmark_risk_calculation(kwargs.get("portfolio_size", 100))
                )
            else:
                raise ValueError(f"Unknown benchmark: {benchmark_name}")
        finally:
            loop.close()


class LatencyProfiler:
    """Profiles latency across multiple components."""
    
    def __init__(self, config):
        """Initialize latency profiler."""
        self.config = config
        self.benchmark = LatencyBenchmark(config)
    
    async def profile_all_components(self) -> Dict[str, LatencyResult]:
        """Profile latency for all components."""
        results = {}
        
        # Signal generation
        results["signal_generation"] = await self.benchmark.benchmark_signal_generation()
        
        # Order execution
        results["order_execution_market"] = await self.benchmark.benchmark_order_execution("market")
        results["order_execution_limit"] = await self.benchmark.benchmark_order_execution("limit")
        
        # Data processing
        results["data_processing_small"] = await self.benchmark.benchmark_data_processing(500)
        results["data_processing_large"] = await self.benchmark.benchmark_data_processing(5000)
        
        # Risk calculation
        results["risk_calculation_small"] = await self.benchmark.benchmark_risk_calculation(50)
        results["risk_calculation_large"] = await self.benchmark.benchmark_risk_calculation(500)
        
        return results
    
    def analyze_latency_trends(self, results: Dict[str, LatencyResult]) -> Dict[str, Dict]:
        """Analyze latency trends and identify bottlenecks."""
        analysis = {
            "bottlenecks": [],
            "performance_summary": {},
            "recommendations": []
        }
        
        # Identify bottlenecks (components with high p99 latency)
        for name, result in results.items():
            p99_latency = result.percentiles.get("p99", 0)
            if p99_latency > 100:  # > 100ms is considered high
                analysis["bottlenecks"].append({
                    "component": name,
                    "p99_latency": p99_latency,
                    "severity": "high" if p99_latency > 500 else "medium"
                })
        
        # Performance summary
        for name, result in results.items():
            analysis["performance_summary"][name] = {
                "mean_latency": result.mean,
                "p95_latency": result.percentiles.get("p95", 0),
                "throughput": result.total_operations / result.duration,
                "stability": result.std_dev / result.mean if result.mean > 0 else 0
            }
        
        # Generate recommendations
        if analysis["bottlenecks"]:
            analysis["recommendations"].append(
                "Consider optimizing components with high p99 latency"
            )
        
        # Check for high variability
        high_variability = [
            name for name, summary in analysis["performance_summary"].items()
            if summary["stability"] > 0.5
        ]
        if high_variability:
            analysis["recommendations"].append(
                f"Components with high latency variability: {', '.join(high_variability)}"
            )
        
        return analysis