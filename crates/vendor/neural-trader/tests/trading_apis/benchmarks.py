"""
Latency Benchmarks for Trading APIs

Comprehensive benchmarking suite for measuring and analyzing:
- API response times
- Order execution latency
- Throughput under load
- Failover performance
- System scalability
"""

import asyncio
import time
import statistics
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark measurement result"""
    operation: str
    latency_us: float
    success: bool
    timestamp: datetime
    api_name: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkStats:
    """Statistical summary of benchmark results"""
    operation: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_latency_us: float
    min_latency_us: float
    max_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    std_latency_us: float
    throughput_rps: float
    error_messages: List[str] = field(default_factory=list)


class LatencyBenchmark:
    """
    Comprehensive latency benchmarking for trading APIs
    """
    
    def __init__(self, apis: Dict[str, Any]):
        """
        Initialize benchmark suite
        
        Args:
            apis: Dictionary of API instances to benchmark
        """
        self.apis = apis
        self.results: List[BenchmarkResult] = []
        self.executor = ThreadPoolExecutor(max_workers=100)
        
    async def benchmark_health_checks(self, 
                                    duration_seconds: int = 60,
                                    requests_per_second: int = 10) -> Dict[str, BenchmarkStats]:
        """
        Benchmark health check latency for all APIs
        
        Args:
            duration_seconds: Test duration
            requests_per_second: Target RPS per API
            
        Returns:
            Statistics for each API
        """
        logger.info(f"Starting health check benchmark: {duration_seconds}s @ {requests_per_second} RPS")
        
        # Create tasks for each API
        tasks = []
        for api_name, api_instance in self.apis.items():
            task = self._benchmark_single_operation(
                api_name,
                api_instance,
                "health_check",
                self._health_check_operation,
                duration_seconds,
                requests_per_second
            )
            tasks.append(task)
        
        # Run all benchmarks concurrently
        results = await asyncio.gather(*tasks)
        
        # Compile statistics
        stats = {}
        for api_name, api_results in zip(self.apis.keys(), results):
            stats[api_name] = self._calculate_stats("health_check", api_results)
        
        return stats
    
    async def benchmark_market_data(self,
                                  symbols: List[str],
                                  duration_seconds: int = 60,
                                  requests_per_second: int = 5) -> Dict[str, BenchmarkStats]:
        """
        Benchmark market data retrieval latency
        
        Args:
            symbols: List of trading symbols to test
            duration_seconds: Test duration
            requests_per_second: Target RPS per API
            
        Returns:
            Statistics for each API
        """
        logger.info(f"Starting market data benchmark: {len(symbols)} symbols, {duration_seconds}s @ {requests_per_second} RPS")
        
        # Create tasks for each API
        tasks = []
        for api_name, api_instance in self.apis.items():
            operation = lambda api: self._market_data_operation(api, symbols)
            task = self._benchmark_single_operation(
                api_name,
                api_instance,
                "market_data",
                operation,
                duration_seconds,
                requests_per_second
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Compile statistics
        stats = {}
        for api_name, api_results in zip(self.apis.keys(), results):
            stats[api_name] = self._calculate_stats("market_data", api_results)
        
        return stats
    
    async def benchmark_order_placement(self,
                                      symbol: str = "BTCUSD",
                                      quantity: float = 100.0,
                                      duration_seconds: int = 60,
                                      requests_per_second: int = 2) -> Dict[str, BenchmarkStats]:
        """
        Benchmark order placement latency (using test orders)
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            duration_seconds: Test duration
            requests_per_second: Target RPS per API
            
        Returns:
            Statistics for each API
        """
        logger.info(f"Starting order placement benchmark: {symbol} {quantity}, {duration_seconds}s @ {requests_per_second} RPS")
        
        # Create tasks for each API
        tasks = []
        for api_name, api_instance in self.apis.items():
            operation = lambda api: self._order_placement_operation(api, symbol, quantity)
            task = self._benchmark_single_operation(
                api_name,
                api_instance,
                "order_placement",
                operation,
                duration_seconds,
                requests_per_second
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Compile statistics
        stats = {}
        for api_name, api_results in zip(self.apis.keys(), results):
            stats[api_name] = self._calculate_stats("order_placement", api_results)
        
        return stats
    
    async def benchmark_throughput(self,
                                 operation_type: str = "health_check",
                                 max_rps: int = 100,
                                 step_size: int = 10,
                                 step_duration: int = 30) -> Dict[str, List[Tuple[int, float, float]]]:
        """
        Benchmark throughput scaling for each API
        
        Args:
            operation_type: Type of operation to benchmark
            max_rps: Maximum RPS to test
            step_size: RPS increment for each step
            step_duration: Duration of each step in seconds
            
        Returns:
            For each API: List of (rps, avg_latency, success_rate) tuples
        """
        logger.info(f"Starting throughput benchmark: {operation_type}, up to {max_rps} RPS")
        
        results = {}
        
        for api_name, api_instance in self.apis.items():
            api_results = []
            
            for target_rps in range(step_size, max_rps + 1, step_size):
                logger.info(f"Testing {api_name} at {target_rps} RPS")
                
                # Run benchmark at this RPS
                operation = self._get_operation_function(operation_type)
                step_results = await self._benchmark_single_operation(
                    api_name,
                    api_instance,
                    operation_type,
                    operation,
                    step_duration,
                    target_rps
                )
                
                # Calculate metrics
                stats = self._calculate_stats(operation_type, step_results)
                api_results.append((target_rps, stats.avg_latency_us, stats.success_rate))
                
                # Stop if success rate drops too low
                if stats.success_rate < 0.8:
                    logger.warning(f"{api_name} success rate dropped to {stats.success_rate:.2%} at {target_rps} RPS")
                    break
            
            results[api_name] = api_results
        
        return results
    
    async def benchmark_failover_performance(self,
                                           failover_manager,
                                           duration_seconds: int = 300) -> Dict[str, Any]:
        """
        Benchmark failover detection and recovery performance
        
        Args:
            failover_manager: FailoverManager instance
            duration_seconds: Test duration
            
        Returns:
            Failover performance metrics
        """
        logger.info(f"Starting failover performance benchmark: {duration_seconds}s")
        
        # Start monitoring
        await failover_manager.start_monitoring()
        
        # Record initial state
        initial_healthy = failover_manager.get_healthy_apis()
        
        # Simulate API failures and recoveries
        failure_events = []
        recovery_events = []
        
        start_time = time.time()
        
        # Run for specified duration
        while time.time() - start_time < duration_seconds:
            # Pick random API to fail
            apis_to_test = list(self.apis.keys())
            
            for api_name in apis_to_test:
                if api_name in self.apis:
                    # Simulate failure
                    failure_start = time.time()
                    await failover_manager.manual_failover(api_name, "Benchmark test failure")
                    failure_detection_time = time.time() - failure_start
                    
                    failure_events.append({
                        'api': api_name,
                        'detection_time_ms': failure_detection_time * 1000,
                        'timestamp': datetime.now()
                    })
                    
                    # Wait a bit
                    await asyncio.sleep(10)
                    
                    # Simulate recovery
                    recovery_start = time.time()
                    await failover_manager.manual_recovery(api_name, "Benchmark test recovery")
                    recovery_time = time.time() - recovery_start
                    
                    recovery_events.append({
                        'api': api_name,
                        'recovery_time_ms': recovery_time * 1000,
                        'timestamp': datetime.now()
                    })
                    
                    # Wait before next test
                    await asyncio.sleep(30)
        
        # Stop monitoring
        await failover_manager.stop_monitoring()
        
        # Calculate metrics
        avg_detection_time = np.mean([e['detection_time_ms'] for e in failure_events])
        avg_recovery_time = np.mean([e['recovery_time_ms'] for e in recovery_events])
        
        return {
            'total_failures': len(failure_events),
            'total_recoveries': len(recovery_events),
            'avg_detection_time_ms': avg_detection_time,
            'avg_recovery_time_ms': avg_recovery_time,
            'failure_events': failure_events,
            'recovery_events': recovery_events,
            'final_healthy_apis': failover_manager.get_healthy_apis()
        }
    
    async def benchmark_concurrent_operations(self,
                                            operation_type: str = "health_check",
                                            concurrent_requests: int = 50,
                                            total_requests: int = 1000) -> Dict[str, BenchmarkStats]:
        """
        Benchmark concurrent operation performance
        
        Args:
            operation_type: Type of operation
            concurrent_requests: Number of concurrent requests
            total_requests: Total requests to make
            
        Returns:
            Statistics for each API
        """
        logger.info(f"Starting concurrent operations benchmark: {operation_type}, "
                   f"{concurrent_requests} concurrent, {total_requests} total")
        
        stats = {}
        
        for api_name, api_instance in self.apis.items():
            logger.info(f"Testing {api_name} with {concurrent_requests} concurrent requests")
            
            operation = self._get_operation_function(operation_type)
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def limited_operation():
                async with semaphore:
                    return await self._single_operation_call(api_instance, operation)
            
            # Run all requests
            start_time = time.time()
            tasks = [limited_operation() for _ in range(total_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Process results
            benchmark_results = []
            for result in results:
                if isinstance(result, BenchmarkResult):
                    benchmark_results.append(result)
                else:
                    # Handle exception
                    benchmark_results.append(BenchmarkResult(
                        operation=operation_type,
                        latency_us=0,
                        success=False,
                        timestamp=datetime.now(),
                        api_name=api_name,
                        error_message=str(result)
                    ))
            
            # Calculate statistics
            api_stats = self._calculate_stats(operation_type, benchmark_results)
            api_stats.throughput_rps = total_requests / total_time
            
            stats[api_name] = api_stats
        
        return stats
    
    async def _benchmark_single_operation(self,
                                        api_name: str,
                                        api_instance: Any,
                                        operation_type: str,
                                        operation_func: callable,
                                        duration_seconds: int,
                                        requests_per_second: int) -> List[BenchmarkResult]:
        """Benchmark a single operation type for one API"""
        
        results = []
        end_time = time.time() + duration_seconds
        request_interval = 1.0 / requests_per_second
        
        while time.time() < end_time:
            try:
                result = await self._single_operation_call(api_instance, operation_func)
                result.api_name = api_name
                results.append(result)
                
                # Wait for next request
                await asyncio.sleep(request_interval)
                
            except Exception as e:
                logger.error(f"Benchmark error for {api_name}: {e}")
                results.append(BenchmarkResult(
                    operation=operation_type,
                    latency_us=0,
                    success=False,
                    timestamp=datetime.now(),
                    api_name=api_name,
                    error_message=str(e)
                ))
        
        return results
    
    async def _single_operation_call(self, api_instance: Any, operation_func: callable) -> BenchmarkResult:
        """Perform a single operation call with timing"""
        start_time = time.perf_counter()
        
        try:
            await operation_func(api_instance)
            latency_us = (time.perf_counter() - start_time) * 1_000_000
            
            return BenchmarkResult(
                operation="generic",
                latency_us=latency_us,
                success=True,
                timestamp=datetime.now(),
                api_name=""
            )
            
        except Exception as e:
            latency_us = (time.perf_counter() - start_time) * 1_000_000
            
            return BenchmarkResult(
                operation="generic",
                latency_us=latency_us,
                success=False,
                timestamp=datetime.now(),
                api_name="",
                error_message=str(e)
            )
    
    def _get_operation_function(self, operation_type: str) -> callable:
        """Get operation function by type"""
        if operation_type == "health_check":
            return self._health_check_operation
        elif operation_type == "market_data":
            return lambda api: self._market_data_operation(api, ["BTCUSD"])
        elif operation_type == "order_placement":
            return lambda api: self._order_placement_operation(api, "BTCUSD", 100.0)
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
    
    async def _health_check_operation(self, api_instance: Any):
        """Health check operation"""
        if hasattr(api_instance, 'health_check'):
            return await api_instance.health_check()
        elif hasattr(api_instance, 'get_server_time'):
            return await api_instance.get_server_time()
        else:
            return await api_instance.get_account_info()
    
    async def _market_data_operation(self, api_instance: Any, symbols: List[str]):
        """Market data operation"""
        symbol = symbols[0]  # Use first symbol
        
        if hasattr(api_instance, 'get_order_book'):
            return await api_instance.get_order_book(symbol)
        elif hasattr(api_instance, 'get_market_data'):
            return await api_instance.get_market_data(symbol)
        else:
            return await api_instance.get_ticker(symbol)
    
    async def _order_placement_operation(self, api_instance: Any, symbol: str, quantity: float):
        """Order placement operation (test mode)"""
        if hasattr(api_instance, 'place_test_order'):
            return await api_instance.place_test_order(
                symbol=symbol,
                quantity=quantity,
                order_type="market"
            )
        elif hasattr(api_instance, 'place_order'):
            # Use test flag if available
            return await api_instance.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="market",
                test=True
            )
        else:
            # Simulate order placement
            await asyncio.sleep(0.001)  # 1ms simulated processing
            return {"status": "test_order", "order_id": "test_123"}
    
    def _calculate_stats(self, operation: str, results: List[BenchmarkResult]) -> BenchmarkStats:
        """Calculate statistics from benchmark results"""
        if not results:
            return BenchmarkStats(
                operation=operation,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0.0,
                avg_latency_us=0.0,
                min_latency_us=0.0,
                max_latency_us=0.0,
                p50_latency_us=0.0,
                p95_latency_us=0.0,
                p99_latency_us=0.0,
                std_latency_us=0.0,
                throughput_rps=0.0
            )
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            latencies = [r.latency_us for r in successful_results]
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = min_latency = max_latency = std_latency = 0
            p50_latency = p95_latency = p99_latency = 0
        
        # Calculate throughput
        if results:
            time_span = (results[-1].timestamp - results[0].timestamp).total_seconds()
            throughput_rps = len(results) / time_span if time_span > 0 else 0
        else:
            throughput_rps = 0
        
        return BenchmarkStats(
            operation=operation,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            success_rate=len(successful_results) / len(results) if results else 0,
            avg_latency_us=avg_latency,
            min_latency_us=min_latency,
            max_latency_us=max_latency,
            p50_latency_us=p50_latency,
            p95_latency_us=p95_latency,
            p99_latency_us=p99_latency,
            std_latency_us=std_latency,
            throughput_rps=throughput_rps,
            error_messages=[r.error_message for r in failed_results if r.error_message]
        )
    
    def generate_report(self, 
                       stats: Dict[str, BenchmarkStats],
                       output_file: Optional[str] = None) -> str:
        """Generate a formatted benchmark report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TRADING API LATENCY BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        for api_name, api_stats in stats.items():
            report_lines.append(f"API: {api_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"Operation: {api_stats.operation}")
            report_lines.append(f"Total Requests: {api_stats.total_requests:,}")
            report_lines.append(f"Successful: {api_stats.successful_requests:,} ({api_stats.success_rate:.1%})")
            report_lines.append(f"Failed: {api_stats.failed_requests:,}")
            report_lines.append("")
            
            if api_stats.successful_requests > 0:
                report_lines.append("Latency Statistics (microseconds):")
                report_lines.append(f"  Average: {api_stats.avg_latency_us:,.0f} μs")
                report_lines.append(f"  Minimum: {api_stats.min_latency_us:,.0f} μs")
                report_lines.append(f"  Maximum: {api_stats.max_latency_us:,.0f} μs")
                report_lines.append(f"  P50: {api_stats.p50_latency_us:,.0f} μs")
                report_lines.append(f"  P95: {api_stats.p95_latency_us:,.0f} μs")
                report_lines.append(f"  P99: {api_stats.p99_latency_us:,.0f} μs")
                report_lines.append(f"  Std Dev: {api_stats.std_latency_us:,.0f} μs")
                report_lines.append("")
                
                report_lines.append(f"Throughput: {api_stats.throughput_rps:.1f} RPS")
            
            if api_stats.error_messages:
                report_lines.append("")
                report_lines.append("Error Messages:")
                for error in set(api_stats.error_messages[:5]):  # Show unique errors
                    report_lines.append(f"  - {error}")
            
            report_lines.append("")
        
        # Add ranking
        report_lines.append("RANKING BY AVERAGE LATENCY:")
        report_lines.append("-" * 40)
        
        sorted_apis = sorted(stats.items(), key=lambda x: x[1].avg_latency_us)
        for i, (api_name, api_stats) in enumerate(sorted_apis, 1):
            if api_stats.successful_requests > 0:
                report_lines.append(f"{i}. {api_name}: {api_stats.avg_latency_us:,.0f} μs "
                                  f"({api_stats.success_rate:.1%} success)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text
    
    def export_results(self, output_file: str):
        """Export raw benchmark results to JSON"""
        export_data = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'operation': result.operation,
                    'latency_us': result.latency_us,
                    'success': result.success,
                    'timestamp': result.timestamp.isoformat(),
                    'api_name': result.api_name,
                    'error_message': result.error_message,
                    'metadata': result.metadata
                }
                for result in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to {output_file}")


# Convenience functions for quick benchmarking
async def quick_latency_test(apis: Dict[str, Any], duration: int = 30) -> Dict[str, BenchmarkStats]:
    """Quick latency test for all APIs"""
    benchmark = LatencyBenchmark(apis)
    return await benchmark.benchmark_health_checks(duration_seconds=duration)


async def comprehensive_benchmark(apis: Dict[str, Any], 
                                symbols: List[str] = ["BTCUSD", "ETHUSD"],
                                duration: int = 120) -> Dict[str, Dict[str, BenchmarkStats]]:
    """Comprehensive benchmark suite"""
    benchmark = LatencyBenchmark(apis)
    
    results = {}
    
    # Health checks
    logger.info("Running health check benchmark...")
    results['health_checks'] = await benchmark.benchmark_health_checks(duration)
    
    # Market data
    logger.info("Running market data benchmark...")
    results['market_data'] = await benchmark.benchmark_market_data(symbols, duration)
    
    # Order placement
    logger.info("Running order placement benchmark...")
    results['order_placement'] = await benchmark.benchmark_order_placement(
        symbol=symbols[0], duration_seconds=duration
    )
    
    # Concurrent operations
    logger.info("Running concurrent operations benchmark...")
    results['concurrent'] = await benchmark.benchmark_concurrent_operations(
        operation_type="health_check", concurrent_requests=20, total_requests=200
    )
    
    return results


async def stress_test(apis: Dict[str, Any], max_rps: int = 50) -> Dict[str, List[Tuple[int, float, float]]]:
    """Stress test to find maximum throughput"""
    benchmark = LatencyBenchmark(apis)
    return await benchmark.benchmark_throughput(max_rps=max_rps)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Mock APIs for testing
    class MockAPI:
        def __init__(self, name: str, latency_ms: float = 1.0):
            self.name = name
            self.latency_ms = latency_ms
        
        async def health_check(self):
            await asyncio.sleep(self.latency_ms / 1000)
            return {"status": "ok"}
        
        async def get_order_book(self, symbol: str):
            await asyncio.sleep(self.latency_ms / 1000)
            return {"bids": [], "asks": []}
        
        async def place_test_order(self, **kwargs):
            await asyncio.sleep(self.latency_ms / 1000)
            return {"order_id": "test", "status": "filled"}
    
    # Create test APIs
    test_apis = {
        "fast_api": MockAPI("fast_api", 0.5),
        "medium_api": MockAPI("medium_api", 2.0),
        "slow_api": MockAPI("slow_api", 10.0)
    }
    
    async def main():
        # Run comprehensive benchmark
        results = await comprehensive_benchmark(test_apis)
        
        # Generate reports
        benchmark = LatencyBenchmark(test_apis)
        for test_type, stats in results.items():
            print(f"\n{test_type.upper()} BENCHMARK:")
            print(benchmark.generate_report(stats))
    
    asyncio.run(main())