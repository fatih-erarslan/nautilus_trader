"""Throughput benchmark implementation for AI News Trading platform."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ThroughputResult:
    """Result container for throughput measurements."""
    
    component: str
    operation: str
    total_operations: int
    duration: float
    operations_per_second: float
    peak_ops_per_second: float
    min_ops_per_second: float
    average_batch_size: float
    concurrent_workers: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "operation": self.operation,
            "total_operations": self.total_operations,
            "duration": self.duration,
            "operations_per_second": self.operations_per_second,
            "peak_ops_per_second": self.peak_ops_per_second,
            "min_ops_per_second": self.min_ops_per_second,
            "average_batch_size": self.average_batch_size,
            "concurrent_workers": self.concurrent_workers,
        }


class ThroughputBenchmark:
    """Measures operation throughput for various components."""
    
    def __init__(self, config):
        """Initialize throughput benchmark."""
        self.config = config
        
        # Handle both ConfigManager and dict configurations
        if hasattr(config, 'config'):
            config_dict = config.config
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config if isinstance(config, dict) else {}
        
        # Extract configuration values with defaults
        benchmark_config = config_dict.get("benchmark", {})
        global_config = config_dict.get("global", {})
        
        self.measurement_duration = benchmark_config.get("measurement_duration", 60)
        self.warmup_duration = benchmark_config.get("warmup_duration", 10)
        self.parallel_workers = global_config.get("parallel_workers", 4)
        self.sample_interval = 1.0  # Sample throughput every 1 second
    
    async def benchmark_signal_throughput(
        self,
        strategy_name: str = "momentum",
        concurrent_workers: Optional[int] = None
    ) -> ThroughputResult:
        """Benchmark signal generation throughput."""
        workers = concurrent_workers or self.parallel_workers
        
        return await self._benchmark_async_throughput(
            component="signal_generator",
            operation="generate_signals",
            func=self._mock_signal_generation,
            args=(strategy_name,),
            concurrent_workers=workers
        )
    
    async def benchmark_order_throughput(
        self,
        order_type: str = "market",
        concurrent_workers: Optional[int] = None
    ) -> ThroughputResult:
        """Benchmark order processing throughput."""
        workers = concurrent_workers or self.parallel_workers
        
        return await self._benchmark_async_throughput(
            component="order_processor",
            operation="process_orders",
            func=self._mock_order_processing,
            args=(order_type,),
            concurrent_workers=workers
        )
    
    def benchmark_data_ingestion_throughput(
        self,
        data_source: str = "market_data",
        concurrent_workers: Optional[int] = None
    ) -> ThroughputResult:
        """Benchmark data ingestion throughput (synchronous)."""
        workers = concurrent_workers or self.parallel_workers
        
        return self._benchmark_sync_throughput(
            component="data_ingester",
            operation="ingest_data",
            func=self._mock_data_ingestion,
            args=(data_source,),
            concurrent_workers=workers
        )
    
    def benchmark_portfolio_calculation_throughput(
        self,
        portfolio_size: int = 100,
        concurrent_workers: Optional[int] = None
    ) -> ThroughputResult:
        """Benchmark portfolio calculation throughput."""
        workers = concurrent_workers or self.parallel_workers
        
        return self._benchmark_sync_throughput(
            component="portfolio_calculator",
            operation="calculate_portfolio",
            func=self._mock_portfolio_calculation,
            args=(portfolio_size,),
            concurrent_workers=workers
        )
    
    async def _benchmark_async_throughput(
        self,
        component: str,
        operation: str,
        func,
        args: tuple = (),
        concurrent_workers: int = 4
    ) -> ThroughputResult:
        """Generic async operation throughput benchmark."""
        # Warmup phase
        await self._warmup_async(func, args, concurrent_workers)
        
        # Measurement phase
        operation_counts = []
        start_time = time.time()
        sample_start = start_time
        current_operations = 0
        
        # Create worker tasks
        tasks = []
        stop_event = asyncio.Event()
        
        for _ in range(concurrent_workers):
            task = asyncio.create_task(
                self._async_worker(func, args, stop_event)
            )
            tasks.append(task)
        
        # Monitor throughput
        while time.time() - start_time < self.measurement_duration:
            await asyncio.sleep(self.sample_interval)
            
            # Count completed operations in this interval
            sample_operations = sum(task.done() for task in tasks if not task.cancelled())
            current_operations += sample_operations
            
            # Calculate ops per second for this interval
            interval_ops = sample_operations / self.sample_interval
            operation_counts.append(interval_ops)
            
            # Restart completed tasks
            for i, task in enumerate(tasks):
                if task.done():
                    tasks[i] = asyncio.create_task(
                        self._async_worker(func, args, stop_event)
                    )
        
        # Stop all workers
        stop_event.set()
        
        # Wait for tasks to complete
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        duration = time.time() - start_time
        total_operations = current_operations
        
        return ThroughputResult(
            component=component,
            operation=operation,
            total_operations=total_operations,
            duration=duration,
            operations_per_second=total_operations / duration,
            peak_ops_per_second=max(operation_counts) if operation_counts else 0,
            min_ops_per_second=min(operation_counts) if operation_counts else 0,
            average_batch_size=1.0,  # Single operations
            concurrent_workers=concurrent_workers
        )
    
    def _benchmark_sync_throughput(
        self,
        component: str,
        operation: str,
        func,
        args: tuple = (),
        concurrent_workers: int = 4
    ) -> ThroughputResult:
        """Generic sync operation throughput benchmark."""
        # Warmup phase
        self._warmup_sync(func, args, concurrent_workers)
        
        # Measurement phase
        operation_counts = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = []
            operations_completed = 0
            
            # Submit initial batch of work
            for _ in range(concurrent_workers):
                future = executor.submit(self._sync_worker, func, args, start_time, self.measurement_duration)
                futures.append(future)
            
            # Monitor completion
            while time.time() - start_time < self.measurement_duration:
                time.sleep(self.sample_interval)
                
                # Count completed operations
                completed_in_interval = 0
                for future in as_completed(futures, timeout=0.1):
                    try:
                        ops = future.result()
                        completed_in_interval += ops
                        operations_completed += ops
                        
                        # Submit new work
                        if time.time() - start_time < self.measurement_duration:
                            new_future = executor.submit(
                                self._sync_worker, func, args, start_time, self.measurement_duration
                            )
                            futures.append(new_future)
                    except:
                        pass
                
                # Calculate throughput for this interval
                interval_ops = completed_in_interval / self.sample_interval
                operation_counts.append(interval_ops)
        
        duration = time.time() - start_time
        
        return ThroughputResult(
            component=component,
            operation=operation,
            total_operations=operations_completed,
            duration=duration,
            operations_per_second=operations_completed / duration,
            peak_ops_per_second=max(operation_counts) if operation_counts else 0,
            min_ops_per_second=min(operation_counts) if operation_counts else 0,
            average_batch_size=1.0,
            concurrent_workers=concurrent_workers
        )
    
    async def _warmup_async(self, func, args: tuple, workers: int):
        """Warmup phase for async benchmarks."""
        tasks = []
        stop_event = asyncio.Event()
        
        for _ in range(workers):
            task = asyncio.create_task(self._async_worker(func, args, stop_event))
            tasks.append(task)
        
        await asyncio.sleep(self.warmup_duration)
        stop_event.set()
        
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def _warmup_sync(self, func, args: tuple, workers: int):
        """Warmup phase for sync benchmarks."""
        with ThreadPoolExecutor(max_workers=workers) as executor:
            start_time = time.time()
            futures = [
                executor.submit(self._sync_worker, func, args, start_time, self.warmup_duration)
                for _ in range(workers)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except:
                    pass
    
    async def _async_worker(self, func, args: tuple, stop_event: asyncio.Event):
        """Async worker that runs until stop event is set."""
        operations = 0
        while not stop_event.is_set():
            try:
                await func(*args)
                operations += 1
            except asyncio.CancelledError:
                break
            except Exception:
                continue  # Continue on errors
        return operations
    
    def _sync_worker(self, func, args: tuple, start_time: float, duration: float):
        """Sync worker that runs for specified duration."""
        operations = 0
        while time.time() - start_time < duration:
            try:
                func(*args)
                operations += 1
            except Exception:
                continue  # Continue on errors
        return operations
    
    async def _mock_signal_generation(self, strategy_name: str):
        """Mock signal generation for throughput testing."""
        # Simulate signal generation work (faster than latency test)
        await asyncio.sleep(0.0001 + np.random.exponential(0.0002))
        return {"signal": "buy", "confidence": 0.75}
    
    async def _mock_order_processing(self, order_type: str):
        """Mock order processing for throughput testing."""
        # Simulate order processing
        await asyncio.sleep(0.0005 + np.random.exponential(0.0001))
        return {"order_id": int(time.time() * 1000000), "status": "processed"}
    
    def _mock_data_ingestion(self, data_source: str):
        """Mock data ingestion for throughput testing."""
        # Simulate data ingestion work
        time.sleep(0.0001 + np.random.exponential(0.00005))
        return {"records_ingested": np.random.randint(1, 10)}
    
    def _mock_portfolio_calculation(self, portfolio_size: int):
        """Mock portfolio calculation for throughput testing."""
        # Simulate portfolio calculation
        time.sleep(0.001 + 0.00001 * portfolio_size)
        return {"portfolio_value": np.random.uniform(100000, 200000)}


class ThroughputProfiler:
    """Profiles throughput across multiple components and configurations."""
    
    def __init__(self, config):
        """Initialize throughput profiler."""
        self.config = config
        self.benchmark = ThroughputBenchmark(config)
    
    async def profile_all_components(self) -> Dict[str, ThroughputResult]:
        """Profile throughput for all components."""
        results = {}
        
        # Signal generation throughput
        results["signal_generation"] = await self.benchmark.benchmark_signal_throughput()
        
        # Order processing throughput
        results["order_processing"] = await self.benchmark.benchmark_order_throughput()
        
        # Data ingestion throughput
        results["data_ingestion"] = self.benchmark.benchmark_data_ingestion_throughput()
        
        # Portfolio calculations
        results["portfolio_calculation"] = self.benchmark.benchmark_portfolio_calculation_throughput()
        
        return results
    
    async def profile_scalability(self, component: str, max_workers: int = 16) -> Dict[int, ThroughputResult]:
        """Profile how throughput scales with number of workers."""
        results = {}
        
        for workers in [1, 2, 4, 8, max_workers]:
            if component == "signal_generation":
                result = await self.benchmark.benchmark_signal_throughput(
                    concurrent_workers=workers
                )
            elif component == "order_processing":
                result = await self.benchmark.benchmark_order_throughput(
                    concurrent_workers=workers
                )
            elif component == "data_ingestion":
                result = self.benchmark.benchmark_data_ingestion_throughput(
                    concurrent_workers=workers
                )
            elif component == "portfolio_calculation":
                result = self.benchmark.benchmark_portfolio_calculation_throughput(
                    concurrent_workers=workers
                )
            else:
                continue
            
            results[workers] = result
        
        return results
    
    def analyze_throughput_trends(self, results: Dict[str, ThroughputResult]) -> Dict[str, Dict]:
        """Analyze throughput trends and identify performance characteristics."""
        analysis = {
            "performance_summary": {},
            "bottlenecks": [],
            "optimization_opportunities": [],
            "scaling_analysis": {}
        }
        
        # Performance summary
        for name, result in results.items():
            ops_per_sec = result.operations_per_second
            peak_ops = result.peak_ops_per_second
            
            analysis["performance_summary"][name] = {
                "average_throughput": ops_per_sec,
                "peak_throughput": peak_ops,
                "throughput_efficiency": ops_per_sec / peak_ops if peak_ops > 0 else 0,
                "workers_utilized": result.concurrent_workers,
                "throughput_per_worker": ops_per_sec / result.concurrent_workers
            }
            
            # Identify bottlenecks (low throughput components)
            if ops_per_sec < 100:  # Less than 100 ops/sec
                analysis["bottlenecks"].append({
                    "component": name,
                    "throughput": ops_per_sec,
                    "severity": "high" if ops_per_sec < 10 else "medium"
                })
        
        # Optimization opportunities
        for name, summary in analysis["performance_summary"].items():
            efficiency = summary["throughput_efficiency"]
            if efficiency < 0.7:  # Less than 70% efficiency
                analysis["optimization_opportunities"].append({
                    "component": name,
                    "current_efficiency": efficiency,
                    "recommendation": "Consider optimizing worker utilization or reducing contention"
                })
        
        return analysis
    
    def generate_performance_report(self, results: Dict[str, ThroughputResult]) -> str:
        """Generate a human-readable performance report."""
        analysis = self.analyze_throughput_trends(results)
        
        report = ["THROUGHPUT PERFORMANCE REPORT", "=" * 40, ""]
        
        # Summary
        report.append("PERFORMANCE SUMMARY:")
        for component, summary in analysis["performance_summary"].items():
            report.append(f"  {component}:")
            report.append(f"    Average Throughput: {summary['average_throughput']:.2f} ops/sec")
            report.append(f"    Peak Throughput: {summary['peak_throughput']:.2f} ops/sec")
            report.append(f"    Efficiency: {summary['throughput_efficiency']:.1%}")
            report.append(f"    Workers: {summary['workers_utilized']}")
            report.append("")
        
        # Bottlenecks
        if analysis["bottlenecks"]:
            report.append("IDENTIFIED BOTTLENECKS:")
            for bottleneck in analysis["bottlenecks"]:
                report.append(f"  - {bottleneck['component']}: {bottleneck['throughput']:.2f} ops/sec ({bottleneck['severity']} severity)")
            report.append("")
        
        # Recommendations
        if analysis["optimization_opportunities"]:
            report.append("OPTIMIZATION OPPORTUNITIES:")
            for opp in analysis["optimization_opportunities"]:
                report.append(f"  - {opp['component']}: {opp['recommendation']}")
            report.append("")
        
        return "\n".join(report)