"""Comprehensive benchmark runner implementation."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .latency_benchmark import LatencyBenchmark, LatencyProfiler
from .throughput_benchmark import ThroughputBenchmark, ThroughputProfiler
from .strategy_benchmark import StrategyBenchmark, StrategyProfiler
from .resource_benchmark import ResourceBenchmark, ResourceProfiler


class BenchmarkRunner:
    """Comprehensive benchmark runner for AI News Trading platform."""
    
    def __init__(self, config):
        """Initialize benchmark runner."""
        self.config = config
        
        # Initialize benchmark components
        self.latency_benchmark = LatencyBenchmark(config)
        self.throughput_benchmark = ThroughputBenchmark(config)
        self.strategy_benchmark = StrategyBenchmark(config)
        self.resource_benchmark = ResourceBenchmark(config)
        
        # Initialize profilers
        self.latency_profiler = LatencyProfiler(config)
        self.throughput_profiler = ThroughputProfiler(config)
        self.strategy_profiler = StrategyProfiler(config)
        self.resource_profiler = ResourceProfiler(config)
    
    def run_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a predefined benchmark suite."""
        # Convert to dict to access suite configurations
        benchmark_dict = self.config.to_dict().get("benchmark", {})
        suite_config = benchmark_dict.get("suites", {}).get(suite_name, {})
        
        if not suite_config:
            # Default suite configurations
            if suite_name == "quick":
                return self._run_quick_suite()
            elif suite_name == "standard":
                return self._run_standard_suite()
            elif suite_name == "comprehensive":
                return self._run_comprehensive_suite()
            else:
                raise ValueError(f"Unknown suite: {suite_name}")
        
        # Run custom suite based on configuration
        return self._run_custom_suite(suite_config)
    
    def run_strategies(
        self,
        strategies: List[str],
        duration: int = 300,
        parallel: Optional[int] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run benchmarks for specific strategies."""
        results = {
            "status": "success",
            "strategies": strategies,
            "duration": duration,
            "parallel": parallel,
            "metrics": metrics or ["latency", "throughput", "strategy_performance"],
            "benchmark_results": {}
        }
        
        # Run benchmarks for each strategy
        for strategy in strategies:
            strategy_results = {}
            
            # Latency benchmarks
            if "latency" in results["metrics"]:
                latency_result = self.latency_benchmark.run_sync(
                    "signal_generation", strategy=strategy
                )
                strategy_results["latency"] = latency_result.to_dict()
            
            # Throughput benchmarks
            if "throughput" in results["metrics"]:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    throughput_result = loop.run_until_complete(
                        self.throughput_benchmark.benchmark_signal_throughput(strategy)
                    )
                    strategy_results["throughput"] = throughput_result.to_dict()
                finally:
                    loop.close()
            
            # Strategy performance
            if "strategy_performance" in results["metrics"]:
                strategy_result = self.strategy_benchmark.benchmark_strategy(
                    strategy, duration_days=duration // 24 if duration > 24 else 30
                )
                strategy_results["strategy_performance"] = strategy_result.to_dict()
            
            # Resource usage
            if "memory" in results["metrics"] or "cpu" in results["metrics"]:
                resource_result = self.resource_benchmark.benchmark_signal_generation_resources(
                    strategy, num_signals=100
                )
                strategy_results["resource_usage"] = resource_result.to_dict()
            
            results["benchmark_results"][strategy] = strategy_results
        
        return results
    
    def _run_quick_suite(self) -> Dict[str, Any]:
        """Run quick benchmark suite (< 2 minutes)."""
        start_time = time.time()
        results = {
            "status": "success",
            "suite": "quick",
            "start_time": start_time,
            "results": {}
        }
        
        # Quick latency test
        latency_result = self.latency_benchmark.run_sync("signal_generation")
        results["results"]["latency"] = {
            "signal_generation": latency_result.to_dict()
        }
        
        # Quick throughput test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            throughput_result = loop.run_until_complete(
                self.throughput_benchmark.benchmark_signal_throughput()
            )
            results["results"]["throughput"] = {
                "signal_generation": throughput_result.to_dict()
            }
        finally:
            loop.close()
        
        # Quick strategy performance (30 days)
        strategy_result = self.strategy_benchmark.benchmark_strategy(
            "momentum", duration_days=30
        )
        results["results"]["strategy_performance"] = {
            "momentum": strategy_result.to_dict()
        }
        
        results["duration"] = time.time() - start_time
        return results
    
    def _run_standard_suite(self) -> Dict[str, Any]:
        """Run standard benchmark suite (5-10 minutes)."""
        start_time = time.time()
        results = {
            "status": "success",
            "suite": "standard",
            "start_time": start_time,
            "results": {}
        }
        
        # Comprehensive latency tests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            latency_results = loop.run_until_complete(
                self.latency_profiler.profile_all_components()
            )
            results["results"]["latency"] = {
                name: result.to_dict() for name, result in latency_results.items()
            }
        finally:
            loop.close()
        
        # Comprehensive throughput tests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            throughput_results = loop.run_until_complete(
                self.throughput_profiler.profile_all_components()
            )
            results["results"]["throughput"] = {
                name: result.to_dict() for name, result in throughput_results.items()
            }
        finally:
            loop.close()
        
        # Strategy comparison
        strategies = ["momentum", "mean_reversion", "buy_and_hold"]
        strategy_results = self.strategy_benchmark.compare_strategies(
            strategies, duration_days=252
        )
        results["results"]["strategy_performance"] = {
            name: result.to_dict() for name, result in strategy_results.items()
        }
        
        # Resource usage analysis
        resource_results = {}
        resource_results["signal_generation"] = self.resource_benchmark.benchmark_signal_generation_resources()
        resource_results["data_processing"] = self.resource_benchmark.benchmark_data_processing_resources()
        
        results["results"]["resource_usage"] = {
            name: result.to_dict() for name, result in resource_results.items()
        }
        
        results["duration"] = time.time() - start_time
        return results
    
    def _run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite (15-30 minutes)."""
        start_time = time.time()
        results = {
            "status": "success",
            "suite": "comprehensive",
            "start_time": start_time,
            "results": {}
        }
        
        # Full latency analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            latency_results = loop.run_until_complete(
                self.latency_profiler.profile_all_components()
            )
            latency_analysis = self.latency_profiler.analyze_latency_trends(latency_results)
            
            results["results"]["latency"] = {
                "measurements": {name: result.to_dict() for name, result in latency_results.items()},
                "analysis": latency_analysis
            }
        finally:
            loop.close()
        
        # Full throughput analysis including scalability
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            throughput_results = loop.run_until_complete(
                self.throughput_profiler.profile_all_components()
            )
            
            # Scalability analysis for signal generation
            scalability_results = loop.run_until_complete(
                self.throughput_profiler.profile_scalability("signal_generation", max_workers=16)
            )
            
            throughput_analysis = self.throughput_profiler.analyze_throughput_trends(throughput_results)
            
            results["results"]["throughput"] = {
                "measurements": {name: result.to_dict() for name, result in throughput_results.items()},
                "scalability": {workers: result.to_dict() for workers, result in scalability_results.items()},
                "analysis": throughput_analysis
            }
        finally:
            loop.close()
        
        # Comprehensive strategy analysis across market conditions
        strategies = ["momentum", "mean_reversion", "arbitrage", "buy_and_hold"]
        strategy_results = {}
        
        for strategy in strategies:
            # Base performance
            base_result = self.strategy_benchmark.benchmark_strategy(strategy, duration_days=252)
            strategy_results[strategy] = base_result
            
            # Market condition analysis
            condition_results = self.strategy_profiler.profile_strategy_across_conditions(strategy)
            strategy_results.update(condition_results)
        
        results["results"]["strategy_performance"] = {
            name: result.to_dict() for name, result in strategy_results.items()
        }
        
        # Comprehensive resource analysis
        resource_results = {}
        
        # Individual component analysis
        resource_results["signal_generation"] = self.resource_benchmark.benchmark_signal_generation_resources()
        resource_results["data_processing"] = self.resource_benchmark.benchmark_data_processing_resources()
        resource_results["portfolio_optimization"] = self.resource_benchmark.benchmark_portfolio_optimization_resources()
        resource_results["risk_calculation"] = self.resource_benchmark.benchmark_risk_calculation_resources()
        
        # Scaling analysis
        scaling_results = self.resource_profiler.profile_scaling_behavior(
            "data_processing", [1000, 5000, 10000, 50000]
        )
        
        resource_analysis = self.resource_profiler.analyze_resource_efficiency(resource_results)
        
        results["results"]["resource_usage"] = {
            "measurements": {name: result.to_dict() for name, result in resource_results.items()},
            "scaling": {scale: result.to_dict() for scale, result in scaling_results.items()},
            "analysis": resource_analysis
        }
        
        results["duration"] = time.time() - start_time
        return results
    
    def _run_custom_suite(self, suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run custom benchmark suite based on configuration."""
        start_time = time.time()
        results = {
            "status": "success",
            "suite": "custom",
            "start_time": start_time,
            "config": suite_config,
            "results": {}
        }
        
        # Run specified strategies
        if "strategies" in suite_config:
            strategy_results = self.run_strategies(
                suite_config["strategies"],
                duration=suite_config.get("duration", 300),
                metrics=suite_config.get("metrics", ["latency", "throughput", "strategy_performance"])
            )
            results["results"].update(strategy_results["benchmark_results"])
        
        results["duration"] = time.time() - start_time
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary report."""
        report = [
            f"BENCHMARK SUITE REPORT: {results.get('suite', 'Unknown').upper()}",
            "=" * 60,
            f"Duration: {results.get('duration', 0):.2f} seconds",
            f"Status: {results.get('status', 'Unknown')}",
            ""
        ]
        
        # Latency summary
        if "latency" in results.get("results", {}):
            report.append("LATENCY PERFORMANCE:")
            latency_data = results["results"]["latency"]
            
            if "measurements" in latency_data:
                for component, data in latency_data["measurements"].items():
                    p95 = data.get("percentiles", {}).get("p95", 0)
                    report.append(f"  {component}: P95 = {p95:.2f}ms")
            else:
                for component, data in latency_data.items():
                    p95 = data.get("percentiles", {}).get("p95", 0)
                    report.append(f"  {component}: P95 = {p95:.2f}ms")
            report.append("")
        
        # Throughput summary
        if "throughput" in results.get("results", {}):
            report.append("THROUGHPUT PERFORMANCE:")
            throughput_data = results["results"]["throughput"]
            
            if "measurements" in throughput_data:
                for component, data in throughput_data["measurements"].items():
                    ops_per_sec = data.get("operations_per_second", 0)
                    report.append(f"  {component}: {ops_per_sec:.2f} ops/sec")
            else:
                for component, data in throughput_data.items():
                    ops_per_sec = data.get("operations_per_second", 0)
                    report.append(f"  {component}: {ops_per_sec:.2f} ops/sec")
            report.append("")
        
        # Strategy performance summary
        if "strategy_performance" in results.get("results", {}):
            report.append("STRATEGY PERFORMANCE:")
            strategy_data = results["results"]["strategy_performance"]
            
            for strategy, data in strategy_data.items():
                if "bull" in strategy or "bear" in strategy:
                    continue  # Skip market condition variants for summary
                
                sharpe = data.get("sharpe_ratio", 0)
                annual_return = data.get("annualized_return", 0)
                max_dd = data.get("max_drawdown", 0)
                
                report.append(f"  {strategy}:")
                report.append(f"    Annual Return: {annual_return:.2%}")
                report.append(f"    Sharpe Ratio: {sharpe:.2f}")
                report.append(f"    Max Drawdown: {max_dd:.2%}")
            report.append("")
        
        # Resource usage summary
        if "resource_usage" in results.get("results", {}):
            report.append("RESOURCE USAGE:")
            resource_data = results["results"]["resource_usage"]
            
            if "measurements" in resource_data:
                for component, data in resource_data["measurements"].items():
                    cpu = data.get("cpu", {}).get("peak_cpu_percent", 0)
                    memory = data.get("memory", {}).get("peak_memory_mb", 0)
                    report.append(f"  {component}: CPU {cpu:.1f}%, Memory {memory:.1f}MB")
            else:
                for component, data in resource_data.items():
                    cpu = data.get("cpu", {}).get("peak_cpu_percent", 0)
                    memory = data.get("memory", {}).get("peak_memory_mb", 0)
                    report.append(f"  {component}: CPU {cpu:.1f}%, Memory {memory:.1f}MB")
        
        return "\n".join(report)