#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Runner
Orchestrates all performance optimization validations
Scientific methodology for zero computational waste validation
"""

import time
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
import psutil
import concurrent.futures
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark execution results"""
    benchmark_name: str
    execution_time: float
    success: bool
    output: str
    error: str
    metrics: Dict[str, Any]
    timestamp: float

@dataclass
class SystemInfo:
    """System information for benchmark context"""
    cpu_info: str
    memory_total_gb: float
    python_version: str
    platform: str
    core_count: int
    timestamp: float

class ComprehensiveBenchmarkRunner:
    """Orchestrates comprehensive performance benchmarking"""
    
    def __init__(self, output_dir: str = "/home/kutlu/CWTS/cwts-ultra/performance/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.system_info = self._collect_system_info()
        
    def _collect_system_info(self) -> SystemInfo:
        """Collect system information for benchmark context"""
        cpu_info = "Unknown"
        try:
            # Try to get CPU info from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('model name'):
                        cpu_info = line.split(':')[1].strip()
                        break
        except:
            pass
            
        return SystemInfo(
            cpu_info=cpu_info,
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            python_version=sys.version.split()[0],
            platform=sys.platform,
            core_count=psutil.cpu_count(),
            timestamp=time.time()
        )
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Execute all performance benchmarks"""
        logger.info("🚀 Starting comprehensive performance benchmark suite")
        logger.info(f"System: {self.system_info.cpu_info}")
        logger.info(f"Memory: {self.system_info.memory_total_gb:.1f} GB")
        logger.info(f"Cores: {self.system_info.core_count}")
        
        benchmarks = [
            {
                'name': 'IEEE754 Compliance Validation',
                'script': 'ieee754_compliance_validator.py',
                'timeout': 300,
                'critical': True
            },
            {
                'name': 'GIL Elimination Benchmarking',
                'script': 'gil_elimination_benchmarker.py',
                'timeout': 600,
                'critical': True
            },
            {
                'name': 'Scientific Data Processing',
                'script': 'scientific_data_processor.py',
                'timeout': 400,
                'critical': True
            },
            {
                'name': 'Comprehensive Performance Analysis',
                'script': 'comprehensive_performance_analyzer.py',
                'timeout': 900,
                'critical': False
            },
            {
                'name': 'Cython GIL Profiler',
                'script': 'cython_gil_profiler.py',
                'timeout': 300,
                'critical': False
            }
        ]
        
        # Execute benchmarks in parallel where possible
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Run critical benchmarks first
            critical_benchmarks = [b for b in benchmarks if b['critical']]
            non_critical_benchmarks = [b for b in benchmarks if not b['critical']]
            
            logger.info("⚡ Executing critical benchmarks...")
            critical_futures = []
            for benchmark in critical_benchmarks:
                future = executor.submit(self._run_single_benchmark, benchmark)
                critical_futures.append(future)
                
            # Wait for critical benchmarks to complete
            critical_results = []
            for future in concurrent.futures.as_completed(critical_futures, timeout=1200):
                try:
                    result = future.result()
                    critical_results.append(result)
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Critical benchmark failed: {e}")
                    
            logger.info("🔧 Executing non-critical benchmarks...")
            non_critical_futures = []
            for benchmark in non_critical_benchmarks:
                future = executor.submit(self._run_single_benchmark, benchmark)
                non_critical_futures.append(future)
                
            # Wait for non-critical benchmarks (allow failures)
            for future in concurrent.futures.as_completed(non_critical_futures, timeout=1800):
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    logger.warning(f"Non-critical benchmark failed: {e}")
                    
        # Generate comprehensive report
        return self._generate_comprehensive_report()
        
    def _run_single_benchmark(self, benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Execute a single benchmark with timeout and error handling"""
        name = benchmark_config['name']
        script = benchmark_config['script']
        timeout = benchmark_config.get('timeout', 300)
        
        logger.info(f"🔄 Running {name}...")
        
        start_time = time.time()
        
        try:
            # Execute benchmark script
            result = subprocess.run(
                [sys.executable, script],
                cwd=str(self.output_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            # Extract metrics from output if available
            metrics = self._extract_metrics(result.stdout, name)
            
            logger.info(f"✅ {name} completed in {execution_time:.1f}s")
            
            return BenchmarkResult(
                benchmark_name=name,
                execution_time=execution_time,
                success=success,
                output=result.stdout,
                error=result.stderr,
                metrics=metrics,
                timestamp=time.time()
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"❌ {name} timed out after {timeout}s")
            
            return BenchmarkResult(
                benchmark_name=name,
                execution_time=execution_time,
                success=False,
                output="",
                error=f"Benchmark timed out after {timeout} seconds",
                metrics={},
                timestamp=time.time()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {name} failed: {e}")
            
            return BenchmarkResult(
                benchmark_name=name,
                execution_time=execution_time,
                success=False,
                output="",
                error=str(e),
                metrics={},
                timestamp=time.time()
            )
            
    def _extract_metrics(self, output: str, benchmark_name: str) -> Dict[str, Any]:
        """Extract performance metrics from benchmark output"""
        metrics = {}
        
        # Common patterns to extract
        patterns = {
            'compliance_rate': r'compliance.*?(\d+\.?\d*)%',
            'throughput': r'(\d+\.?\d*)\s*ops/sec',
            'latency_p99': r'P99.*?(\d+\.?\d*)\s*(?:ns|ms|s)',
            'memory_usage': r'Memory.*?(\d+\.?\d*)\s*(?:MB|GB)',
            'cpu_usage': r'CPU.*?(\d+\.?\d*)%',
            'error_rate': r'error.*?(\d+\.?\d*)%',
            'processing_time': r'completed in (\d+\.?\d*)\s*(?:s|ms)',
        }
        
        import re
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    # Take the first match and convert to float
                    value = float(matches[0])
                    metrics[metric_name] = value
                except (ValueError, IndexError):
                    pass
                    
        # Benchmark-specific metric extraction
        if 'IEEE754' in benchmark_name:
            # Extract IEEE 754 specific metrics
            if 'Total tests:' in output:
                lines = output.split('\n')
                for line in lines:
                    if 'Total tests:' in line:
                        metrics['total_tests'] = int(re.findall(r'\d+', line)[0])
                    elif 'Max relative error:' in line:
                        match = re.search(r'(\d+\.?\d*e?-?\d*)', line)
                        if match:
                            metrics['max_relative_error'] = float(match.group(1))
                            
        elif 'GIL' in benchmark_name:
            # Extract GIL-specific metrics
            if 'ops/sec' in output:
                throughput_values = re.findall(r'(\d+)\s*ops/sec', output)
                if throughput_values:
                    metrics['max_throughput'] = max(int(t) for t in throughput_values)
                    metrics['avg_throughput'] = np.mean([int(t) for t in throughput_values])
                    
        elif 'Scientific Data' in benchmark_name:
            # Extract data processing metrics
            volatility_matches = re.findall(r'Volatility:\s*(\d+\.?\d*)%', output)
            if volatility_matches:
                metrics['avg_volatility'] = np.mean([float(v) for v in volatility_matches])
                
            sharpe_matches = re.findall(r'Sharpe Ratio:\s*(\d+\.?\d*)', output)
            if sharpe_matches:
                metrics['avg_sharpe_ratio'] = np.mean([float(s) for s in sharpe_matches])
                
        return metrics
        
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        successful_benchmarks = [r for r in self.results if r.success]
        failed_benchmarks = [r for r in self.results if not r.success]
        
        total_execution_time = sum(r.execution_time for r in self.results)
        
        # Aggregate metrics
        all_metrics = {}
        for result in successful_benchmarks:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
                
        # Calculate aggregated statistics
        aggregated_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                aggregated_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 0.0,
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
                
        # Performance score calculation
        performance_score = self._calculate_performance_score(successful_benchmarks)
        
        report = {
            'summary': {
                'total_benchmarks': len(self.results),
                'successful_benchmarks': len(successful_benchmarks),
                'failed_benchmarks': len(failed_benchmarks),
                'success_rate': len(successful_benchmarks) / len(self.results) * 100,
                'total_execution_time': total_execution_time,
                'performance_score': performance_score,
                'timestamp': datetime.now().isoformat()
            },
            'system_info': asdict(self.system_info),
            'benchmark_results': [asdict(r) for r in self.results],
            'aggregated_metrics': aggregated_metrics,
            'recommendations': self._generate_recommendations(successful_benchmarks, failed_benchmarks)
        }
        
        return report
        
    def _calculate_performance_score(self, successful_benchmarks: List[BenchmarkResult]) -> float:
        """Calculate overall performance score (0-100)"""
        if not successful_benchmarks:
            return 0.0
            
        score_components = []
        
        # IEEE 754 compliance score
        ieee_benchmarks = [b for b in successful_benchmarks if 'IEEE754' in b.benchmark_name]
        if ieee_benchmarks:
            for bench in ieee_benchmarks:
                compliance_rate = bench.metrics.get('compliance_rate', 0)
                score_components.append(compliance_rate)
                
        # Throughput score (normalized)
        gil_benchmarks = [b for b in successful_benchmarks if 'GIL' in b.benchmark_name]
        if gil_benchmarks:
            for bench in gil_benchmarks:
                max_throughput = bench.metrics.get('max_throughput', 0)
                # Normalize throughput to 0-100 scale (assuming max of 100k ops/sec)
                throughput_score = min(100, (max_throughput / 100000) * 100)
                score_components.append(throughput_score)
                
        # Execution time penalty
        avg_execution_time = np.mean([b.execution_time for b in successful_benchmarks])
        time_penalty = max(0, min(20, (avg_execution_time - 60) / 30 * 20))  # Penalty for >60s
        
        base_score = np.mean(score_components) if score_components else 50
        final_score = max(0, base_score - time_penalty)
        
        return round(final_score, 2)
        
    def _generate_recommendations(self, successful: List[BenchmarkResult], 
                                failed: List[BenchmarkResult]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze successful benchmarks
        for result in successful:
            if 'IEEE754' in result.benchmark_name:
                compliance_rate = result.metrics.get('compliance_rate', 0)
                if compliance_rate < 95:
                    recommendations.append(f"IEEE 754 compliance at {compliance_rate}% - review arithmetic operations")
                    
            elif 'GIL' in result.benchmark_name:
                max_throughput = result.metrics.get('max_throughput', 0)
                if max_throughput < 10000:
                    recommendations.append("Low GIL elimination throughput - consider C extensions or multiprocessing")
                    
        # Analyze failed benchmarks
        if failed:
            recommendations.append(f"{len(failed)} benchmarks failed - check system dependencies and resources")
            
        # General recommendations
        if not recommendations:
            recommendations.append("All benchmarks passed - system is optimally configured")
            
        recommendations.extend([
            "Consider implementing Cython extensions for critical paths",
            "Monitor memory usage patterns in production",
            "Validate numerical stability in financial calculations",
            "Use SIMD optimizations for vectorized operations"
        ])
        
        return recommendations
        
    def save_report(self, report: Dict[str, Any], filename: str = "comprehensive_benchmark_report.json"):
        """Save comprehensive report to file"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"📊 Report saved to: {filepath}")
        return filepath
        
    def print_summary(self, report: Dict[str, Any]):
        """Print executive summary of benchmark results"""
        summary = report['summary']
        
        print("\n" + "="*80)
        print("🎯 CWTS PERFORMANCE OPTIMIZATION VALIDATION SUMMARY")
        print("="*80)
        
        print(f"📈 Overall Performance Score: {summary['performance_score']:.1f}/100")
        print(f"✅ Successful Benchmarks: {summary['successful_benchmarks']}/{summary['total_benchmarks']}")
        print(f"⏱️  Total Execution Time: {summary['total_execution_time']:.1f}s")
        print(f"🎯 Success Rate: {summary['success_rate']:.1f}%")
        
        # Key metrics
        if 'aggregated_metrics' in report:
            print("\n📊 Key Performance Metrics:")
            metrics = report['aggregated_metrics']
            
            if 'compliance_rate' in metrics:
                print(f"  IEEE 754 Compliance: {metrics['compliance_rate']['mean']:.1f}%")
                
            if 'max_throughput' in metrics:
                print(f"  Max Throughput: {metrics['max_throughput']['max']:,.0f} ops/sec")
                
            if 'avg_volatility' in metrics:
                print(f"  Avg Market Volatility: {metrics['avg_volatility']['mean']:.1f}%")
                
        # Recommendations
        print(f"\n🎯 Top Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
            
        print("\n" + "="*80)

def main():
    """Execute comprehensive benchmark suite"""
    runner = ComprehensiveBenchmarkRunner()
    
    try:
        # Run all benchmarks
        report = runner.run_all_benchmarks()
        
        # Save report
        report_path = runner.save_report(report)
        
        # Print summary
        runner.print_summary(report)
        
        # Exit with appropriate code
        success_rate = report['summary']['success_rate']
        if success_rate >= 80:
            logger.info("🎉 Benchmark suite completed successfully")
            sys.exit(0)
        else:
            logger.warning(f"⚠️ Benchmark suite completed with {success_rate:.1f}% success rate")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Benchmark suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Benchmark suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()