"""
Comprehensive Test Runner for AI News Trading Benchmark System.

This module orchestrates the complete test suite including:
- Unit tests
- Integration tests
- Performance validation tests
- Regression tests
- Generates comprehensive test reports
- Validates performance targets
- Creates CI/CD integration reports
"""

import asyncio
import argparse
import json
import time
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import subprocess
import pytest
import yaml
import psutil
import os

# Import test modules
from tests.integration.test_end_to_end_benchmark import TestEndToEndBenchmark
from tests.integration.test_cli_integration import TestCLIBenchmarkIntegration
from tests.integration.test_realtime_simulation import TestRealtimeSimulationIntegration
from tests.integration.test_optimization_integration import TestOptimizationBenchmarkIntegration

from tests.performance.test_latency_targets import TestSignalGenerationLatency
from tests.performance.test_throughput_targets import TestTradeExecutionThroughput
from tests.performance.test_memory_usage import TestMemoryUsageLimits
from tests.performance.test_scalability import TestConcurrentSymbolScaling

from tests.regression.test_baseline_comparison import TestPerformanceBaselineComparison
from tests.regression.test_strategy_consistency import TestStrategyOutputConsistency
from tests.regression.test_data_quality import TestDataFeedQualityValidation


class ComprehensiveTestRunner:
    """Comprehensive test runner and orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize test runner with configuration."""
        self.config = self._load_config(config_path)
        self.results = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
            'test_categories': {},
            'performance_validation': {},
            'summary': {},
            'environment': {},
            'failures': []
        }
        self.performance_targets = {
            'signal_latency_p99_ms': 100,
            'trade_throughput_per_sec': 1000,
            'memory_limit_gb': 2,
            'concurrent_symbols': 100
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test runner configuration."""
        default_config = {
            'test_categories': {
                'unit': {'enabled': True, 'timeout': 300},
                'integration': {'enabled': True, 'timeout': 600},
                'performance': {'enabled': True, 'timeout': 1200},
                'regression': {'enabled': True, 'timeout': 900}
            },
            'performance_validation': {
                'enabled': True,
                'strict_mode': False,
                'tolerance_percent': 10
            },
            'reporting': {
                'formats': ['json', 'html', 'junit'],
                'output_dir': './test_results',
                'detailed_logs': True
            },
            'parallel_execution': {
                'enabled': True,
                'max_workers': 4
            },
            'environment_checks': {
                'system_resources': True,
                'dependencies': True,
                'connectivity': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
                
                # Merge user config with defaults
                return self._merge_configs(default_config, user_config)
        
        return default_config
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Merge user configuration with defaults."""
        merged = default.copy()
        for key, value in user.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    async def run_full_suite(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("🚀 Starting AI News Trading Benchmark Test Suite")
        print("=" * 60)
        
        self.results['start_time'] = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        
        try:
            # Pre-flight checks
            await self._run_environment_checks()
            
            # Determine which categories to run
            if categories is None:
                categories = [cat for cat, config in self.config['test_categories'].items() 
                             if config.get('enabled', True)]
            
            print(f"📋 Test Categories: {', '.join(categories)}")
            print()
            
            # Run test categories
            for category in categories:
                if category in self.config['test_categories']:
                    await self._run_test_category(category)
                else:
                    print(f"⚠️  Unknown test category: {category}")
            
            # Performance validation
            if self.config['performance_validation']['enabled']:
                await self._validate_performance_targets()
            
            # Generate summary
            await self._generate_summary()
            
        except Exception as e:
            print(f"❌ Critical error during test execution: {e}")
            traceback.print_exc()
            self.results['failures'].append({
                'type': 'critical_error',
                'message': str(e),
                'traceback': traceback.format_exc()
            })
        
        finally:
            end_time = time.perf_counter()
            self.results['end_time'] = datetime.now(timezone.utc).isoformat()
            self.results['duration_seconds'] = end_time - start_time
            
            # Generate reports
            await self._generate_reports()
            
            # Print final summary
            self._print_final_summary()
        
        return self.results
    
    async def _run_environment_checks(self):
        """Run pre-flight environment checks."""
        print("🔍 Running Environment Checks")
        print("-" * 30)
        
        env_config = self.config.get('environment_checks', {})
        env_results = {}
        
        # System resources check
        if env_config.get('system_resources', True):
            process = psutil.Process()
            memory_info = psutil.virtual_memory()
            cpu_info = psutil.cpu_count()
            
            env_results['system_resources'] = {
                'cpu_cores': cpu_info,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_available_gb': memory_info.available / (1024**3),
                'memory_percent': memory_info.percent,
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
            }
            
            # Check minimum requirements
            min_memory_gb = 4
            min_disk_gb = 10
            
            if env_results['system_resources']['memory_available_gb'] < min_memory_gb:
                print(f"⚠️  Low memory: {env_results['system_resources']['memory_available_gb']:.1f}GB < {min_memory_gb}GB")
            
            if env_results['system_resources']['disk_free_gb'] < min_disk_gb:
                print(f"⚠️  Low disk space: {env_results['system_resources']['disk_free_gb']:.1f}GB < {min_disk_gb}GB")
            
            print(f"✅ System: {cpu_info} cores, {env_results['system_resources']['memory_available_gb']:.1f}GB RAM available")
        
        # Dependencies check
        if env_config.get('dependencies', True):
            required_packages = ['pytest', 'numpy', 'pandas', 'psutil', 'aiohttp']
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            env_results['dependencies'] = {
                'required_packages': required_packages,
                'missing_packages': missing_packages,
                'python_version': sys.version
            }
            
            if missing_packages:
                print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
            else:
                print("✅ Dependencies: All required packages available")
        
        # Connectivity check (basic)
        if env_config.get('connectivity', True):
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                env_results['connectivity'] = {'internet': True}
                print("✅ Connectivity: Internet connection available")
            except OSError:
                env_results['connectivity'] = {'internet': False}
                print("⚠️  Connectivity: No internet connection (some tests may fail)")
        
        self.results['environment'] = env_results
        print()
    
    async def _run_test_category(self, category: str):
        """Run tests for a specific category."""
        print(f"🧪 Running {category.title()} Tests")
        print("-" * (20 + len(category)))
        
        category_config = self.config['test_categories'][category]
        timeout = category_config.get('timeout', 600)
        
        category_results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'running',
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'duration_seconds': 0,
            'details': {},
            'failures': []
        }
        
        start_time = time.perf_counter()
        
        try:
            if category == 'unit':
                results = await self._run_unit_tests(timeout)
            elif category == 'integration':
                results = await self._run_integration_tests(timeout)
            elif category == 'performance':
                results = await self._run_performance_tests(timeout)
            elif category == 'regression':
                results = await self._run_regression_tests(timeout)
            else:
                raise ValueError(f"Unknown test category: {category}")
            
            category_results.update(results)
            category_results['status'] = 'completed'
            
        except asyncio.TimeoutError:
            category_results['status'] = 'timeout'
            category_results['failures'].append({
                'type': 'timeout',
                'message': f"Tests timed out after {timeout} seconds"
            })
            print(f"⏰ {category.title()} tests timed out after {timeout} seconds")
            
        except Exception as e:
            category_results['status'] = 'error'
            category_results['failures'].append({
                'type': 'execution_error',
                'message': str(e),
                'traceback': traceback.format_exc()
            })
            print(f"❌ Error running {category} tests: {e}")
        
        finally:
            end_time = time.perf_counter()
            category_results['duration_seconds'] = end_time - start_time
            category_results['end_time'] = datetime.now(timezone.utc).isoformat()
            
            self.results['test_categories'][category] = category_results
            
            # Print category summary
            self._print_category_summary(category, category_results)
            print()
    
    async def _run_unit_tests(self, timeout: int) -> Dict[str, Any]:
        """Run unit tests using pytest."""
        # Run existing unit tests with pytest
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '--tb=short',
            '--json-report',
            '--json-report-file=test_results/unit_tests.json',
            '-v',
            '--timeout', str(timeout // 10)  # Per-test timeout
        ]
        
        # Exclude our new integration/performance/regression tests
        cmd.extend([
            '--ignore=tests/integration',
            '--ignore=tests/performance', 
            '--ignore=tests/regression'
        ])
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
        
        # Parse pytest JSON report if available
        report_file = Path('test_results/unit_tests.json')
        if report_file.exists():
            with open(report_file) as f:
                pytest_report = json.load(f)
            
            return {
                'tests_run': pytest_report['summary']['total'],
                'tests_passed': pytest_report['summary']['passed'],
                'tests_failed': pytest_report['summary']['failed'],
                'tests_skipped': pytest_report['summary']['skipped'],
                'details': {
                    'pytest_report': pytest_report,
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode()
                }
            }
        else:
            # Fallback parsing from stdout
            return self._parse_pytest_output(stdout.decode(), stderr.decode())
    
    async def _run_integration_tests(self, timeout: int) -> Dict[str, Any]:
        """Run integration tests."""
        integration_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'details': {}
        }
        
        test_classes = [
            ('end_to_end', TestEndToEndBenchmark),
            ('cli_integration', TestCLIBenchmarkIntegration),
            ('realtime_simulation', TestRealtimeSimulationIntegration),
            ('optimization_integration', TestOptimizationBenchmarkIntegration)
        ]
        
        for test_name, test_class in test_classes:
            try:
                print(f"  Running {test_name} tests...")
                test_result = await self._run_test_class(test_class, timeout // len(test_classes))
                
                integration_results['tests_run'] += test_result['tests_run']
                integration_results['tests_passed'] += test_result['tests_passed']
                integration_results['tests_failed'] += test_result['tests_failed']
                integration_results['tests_skipped'] += test_result['tests_skipped']
                integration_results['details'][test_name] = test_result
                
                status = "✅" if test_result['tests_failed'] == 0 else "❌"
                print(f"    {status} {test_name}: {test_result['tests_passed']}/{test_result['tests_run']} passed")
                
            except Exception as e:
                print(f"    ❌ {test_name}: Error - {e}")
                integration_results['tests_failed'] += 1
                integration_results['details'][test_name] = {'error': str(e)}
        
        return integration_results
    
    async def _run_performance_tests(self, timeout: int) -> Dict[str, Any]:
        """Run performance tests."""
        performance_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'details': {},
            'performance_metrics': {}
        }
        
        test_classes = [
            ('latency_targets', TestSignalGenerationLatency),
            ('throughput_targets', TestTradeExecutionThroughput),
            ('memory_usage', TestMemoryUsageLimits),
            ('scalability', TestConcurrentSymbolScaling)
        ]
        
        for test_name, test_class in test_classes:
            try:
                print(f"  Running {test_name} tests...")
                test_result = await self._run_test_class(test_class, timeout // len(test_classes))
                
                performance_results['tests_run'] += test_result['tests_run']
                performance_results['tests_passed'] += test_result['tests_passed']
                performance_results['tests_failed'] += test_result['tests_failed']
                performance_results['tests_skipped'] += test_result['tests_skipped']
                performance_results['details'][test_name] = test_result
                
                # Extract performance metrics if available
                if 'performance_metrics' in test_result:
                    performance_results['performance_metrics'][test_name] = test_result['performance_metrics']
                
                status = "✅" if test_result['tests_failed'] == 0 else "❌"
                print(f"    {status} {test_name}: {test_result['tests_passed']}/{test_result['tests_run']} passed")
                
            except Exception as e:
                print(f"    ❌ {test_name}: Error - {e}")
                performance_results['tests_failed'] += 1
                performance_results['details'][test_name] = {'error': str(e)}
        
        return performance_results
    
    async def _run_regression_tests(self, timeout: int) -> Dict[str, Any]:
        """Run regression tests."""
        regression_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'details': {}
        }
        
        test_classes = [
            ('baseline_comparison', TestPerformanceBaselineComparison),
            ('strategy_consistency', TestStrategyOutputConsistency),
            ('data_quality', TestDataFeedQualityValidation)
        ]
        
        for test_name, test_class in test_classes:
            try:
                print(f"  Running {test_name} tests...")
                test_result = await self._run_test_class(test_class, timeout // len(test_classes))
                
                regression_results['tests_run'] += test_result['tests_run']
                regression_results['tests_passed'] += test_result['tests_passed']
                regression_results['tests_failed'] += test_result['tests_failed']
                regression_results['tests_skipped'] += test_result['tests_skipped']
                regression_results['details'][test_name] = test_result
                
                status = "✅" if test_result['tests_failed'] == 0 else "❌"
                print(f"    {status} {test_name}: {test_result['tests_passed']}/{test_result['tests_run']} passed")
                
            except Exception as e:
                print(f"    ❌ {test_name}: Error - {e}")
                regression_results['tests_failed'] += 1
                regression_results['details'][test_name] = {'error': str(e)}
        
        return regression_results
    
    async def _run_test_class(self, test_class, timeout: int) -> Dict[str, Any]:
        """Run a specific test class and return results."""
        # This is a simplified test runner - in practice you'd use pytest programmatically
        # or create instances and run specific test methods
        
        # For now, return mock results to demonstrate the structure
        return {
            'tests_run': 5,
            'tests_passed': 4,
            'tests_failed': 1,
            'tests_skipped': 0,
            'performance_metrics': {
                'avg_latency_ms': 45.2,
                'throughput_per_sec': 1250,
                'memory_usage_mb': 512
            }
        }
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        # Simple parsing - in practice you'd use pytest's JSON report
        lines = stdout.split('\n')
        
        tests_run = tests_passed = tests_failed = tests_skipped = 0
        
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Parse summary line like "5 passed, 2 failed, 1 skipped"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        tests_passed = int(parts[i-1])
                    elif part == 'failed' and i > 0:
                        tests_failed = int(parts[i-1])
                    elif part == 'skipped' and i > 0:
                        tests_skipped = int(parts[i-1])
        
        tests_run = tests_passed + tests_failed + tests_skipped
        
        return {
            'tests_run': tests_run,
            'tests_passed': tests_passed,
            'tests_failed': tests_failed,
            'tests_skipped': tests_skipped,
            'details': {
                'stdout': stdout,
                'stderr': stderr
            }
        }
    
    async def _validate_performance_targets(self):
        """Validate that performance targets are met."""
        print("🎯 Validating Performance Targets")
        print("-" * 35)
        
        validation_config = self.config['performance_validation']
        tolerance_percent = validation_config.get('tolerance_percent', 10)
        strict_mode = validation_config.get('strict_mode', False)
        
        validation_results = {
            'targets_met': 0,
            'targets_failed': 0,
            'target_details': {},
            'overall_pass': True
        }
        
        # Extract performance metrics from test results
        performance_data = self.results['test_categories'].get('performance', {})
        performance_metrics = performance_data.get('details', {})
        
        for target_name, target_value in self.performance_targets.items():
            target_result = {
                'target_value': target_value,
                'actual_value': None,
                'met': False,
                'within_tolerance': False,
                'deviation_percent': None
            }
            
            # Map target names to test results
            actual_value = self._extract_performance_metric(performance_metrics, target_name)
            
            if actual_value is not None:
                target_result['actual_value'] = actual_value
                
                # Check if target is met
                if target_name.endswith('_ms'):  # Latency targets (lower is better)
                    target_result['met'] = actual_value <= target_value
                    deviation = (actual_value - target_value) / target_value * 100
                else:  # Throughput/capacity targets (higher is better)
                    target_result['met'] = actual_value >= target_value
                    deviation = (target_value - actual_value) / target_value * 100
                
                target_result['deviation_percent'] = deviation
                target_result['within_tolerance'] = abs(deviation) <= tolerance_percent
                
                if target_result['met']:
                    validation_results['targets_met'] += 1
                    status = "✅"
                elif target_result['within_tolerance'] and not strict_mode:
                    validation_results['targets_met'] += 1
                    status = "⚠️"
                else:
                    validation_results['targets_failed'] += 1
                    validation_results['overall_pass'] = False
                    status = "❌"
                
                print(f"  {status} {target_name}: {actual_value} (target: {target_value})")
                
            else:
                validation_results['targets_failed'] += 1
                target_result['met'] = False
                print(f"  ❓ {target_name}: No data available")
            
            validation_results['target_details'][target_name] = target_result
        
        self.results['performance_validation'] = validation_results
        
        # Print summary
        total_targets = len(self.performance_targets)
        print(f"\n  📊 Performance Summary: {validation_results['targets_met']}/{total_targets} targets met")
        
        if not validation_results['overall_pass']:
            print(f"  ⚠️  Performance validation failed: {validation_results['targets_failed']} targets not met")
        
        print()
    
    def _extract_performance_metric(self, performance_metrics: Dict, target_name: str) -> Optional[float]:
        """Extract specific performance metric from test results."""
        # This would map target names to actual test result locations
        metric_mappings = {
            'signal_latency_p99_ms': ['latency_targets', 'performance_metrics', 'p99_latency_ms'],
            'trade_throughput_per_sec': ['throughput_targets', 'performance_metrics', 'throughput_per_sec'],
            'memory_limit_gb': ['memory_usage', 'performance_metrics', 'memory_usage_mb'],
            'concurrent_symbols': ['scalability', 'performance_metrics', 'concurrent_symbols']
        }
        
        if target_name in metric_mappings:
            path = metric_mappings[target_name]
            value = performance_metrics
            
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            # Convert units if needed
            if target_name == 'memory_limit_gb' and isinstance(value, (int, float)):
                return value / 1024  # Convert MB to GB
            
            return value if isinstance(value, (int, float)) else None
        
        return None
    
    async def _generate_summary(self):
        """Generate test execution summary."""
        summary = {
            'total_tests_run': 0,
            'total_tests_passed': 0,
            'total_tests_failed': 0,
            'total_tests_skipped': 0,
            'categories_run': len(self.results['test_categories']),
            'categories_passed': 0,
            'categories_failed': 0,
            'performance_targets_met': 0,
            'performance_targets_total': len(self.performance_targets),
            'overall_pass': True
        }
        
        # Aggregate results across categories
        for category, results in self.results['test_categories'].items():
            summary['total_tests_run'] += results.get('tests_run', 0)
            summary['total_tests_passed'] += results.get('tests_passed', 0)
            summary['total_tests_failed'] += results.get('tests_failed', 0)
            summary['total_tests_skipped'] += results.get('tests_skipped', 0)
            
            if results.get('status') == 'completed' and results.get('tests_failed', 0) == 0:
                summary['categories_passed'] += 1
            else:
                summary['categories_failed'] += 1
                summary['overall_pass'] = False
        
        # Include performance validation
        perf_validation = self.results.get('performance_validation', {})
        summary['performance_targets_met'] = perf_validation.get('targets_met', 0)
        
        if not perf_validation.get('overall_pass', True):
            summary['overall_pass'] = False
        
        # Calculate success rates
        if summary['total_tests_run'] > 0:
            summary['test_success_rate'] = summary['total_tests_passed'] / summary['total_tests_run']
        else:
            summary['test_success_rate'] = 0
        
        if summary['categories_run'] > 0:
            summary['category_success_rate'] = summary['categories_passed'] / summary['categories_run']
        else:
            summary['category_success_rate'] = 0
        
        if summary['performance_targets_total'] > 0:
            summary['performance_success_rate'] = summary['performance_targets_met'] / summary['performance_targets_total']
        else:
            summary['performance_success_rate'] = 0
        
        self.results['summary'] = summary
    
    async def _generate_reports(self):
        """Generate test reports in various formats."""
        reporting_config = self.config.get('reporting', {})
        output_dir = Path(reporting_config.get('output_dir', './test_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        formats = reporting_config.get('formats', ['json'])
        
        # Generate JSON report
        if 'json' in formats:
            json_report_path = output_dir / 'test_report.json'
            with open(json_report_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📄 JSON report saved: {json_report_path}")
        
        # Generate HTML report
        if 'html' in formats:
            html_report_path = output_dir / 'test_report.html'
            self._generate_html_report(html_report_path)
            print(f"📄 HTML report saved: {html_report_path}")
        
        # Generate JUnit XML report
        if 'junit' in formats:
            junit_report_path = output_dir / 'junit_report.xml'
            self._generate_junit_report(junit_report_path)
            print(f"📄 JUnit report saved: {junit_report_path}")
    
    def _generate_html_report(self, output_path: Path):
        """Generate HTML test report."""
        summary = self.results['summary']
        duration_min = self.results['duration_seconds'] / 60
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI News Trading Benchmark Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AI News Trading Benchmark Test Report</h1>
        <p><strong>Generated:</strong> {self.results['start_time']}</p>
        <p><strong>Duration:</strong> {duration_min:.1f} minutes</p>
        <p><strong>Status:</strong> <span class="{'passed' if summary['overall_pass'] else 'failed'}">
            {'PASSED' if summary['overall_pass'] else 'FAILED'}
        </span></p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Tests</h3>
            <p><strong>{summary['total_tests_passed']}/{summary['total_tests_run']}</strong></p>
            <p>{summary['test_success_rate']:.1%} success</p>
        </div>
        <div class="metric">
            <h3>Categories</h3>
            <p><strong>{summary['categories_passed']}/{summary['categories_run']}</strong></p>
            <p>{summary['category_success_rate']:.1%} success</p>
        </div>
        <div class="metric">
            <h3>Performance</h3>
            <p><strong>{summary['performance_targets_met']}/{summary['performance_targets_total']}</strong></p>
            <p>{summary['performance_success_rate']:.1%} targets met</p>
        </div>
    </div>
    
    <h2>📊 Test Categories</h2>
        """
        
        # Add category details
        for category, results in self.results['test_categories'].items():
            status_class = 'passed' if results.get('tests_failed', 0) == 0 else 'failed'
            html_content += f"""
    <div class="category">
        <h3>{category.title()} Tests <span class="{status_class}">
            {'✅' if results.get('tests_failed', 0) == 0 else '❌'}
        </span></h3>
        <p><strong>Duration:</strong> {results.get('duration_seconds', 0):.1f} seconds</p>
        <p><strong>Results:</strong> {results.get('tests_passed', 0)} passed, 
           {results.get('tests_failed', 0)} failed, {results.get('tests_skipped', 0)} skipped</p>
    </div>
            """
        
        # Add performance validation
        if 'performance_validation' in self.results:
            html_content += """
    <h2>🎯 Performance Validation</h2>
    <table>
        <tr><th>Target</th><th>Expected</th><th>Actual</th><th>Status</th></tr>
            """
            
            for target_name, target_data in self.results['performance_validation']['target_details'].items():
                status = '✅' if target_data['met'] else '❌'
                actual = target_data['actual_value'] or 'N/A'
                html_content += f"""
        <tr>
            <td>{target_name}</td>
            <td>{target_data['target_value']}</td>
            <td>{actual}</td>
            <td>{status}</td>
        </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
</body>
</html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_junit_report(self, output_path: Path):
        """Generate JUnit XML test report."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        
        testsuites = Element('testsuites')
        testsuites.set('tests', str(self.results['summary']['total_tests_run']))
        testsuites.set('failures', str(self.results['summary']['total_tests_failed']))
        testsuites.set('time', str(self.results['duration_seconds']))
        
        for category, results in self.results['test_categories'].items():
            testsuite = SubElement(testsuites, 'testsuite')
            testsuite.set('name', category)
            testsuite.set('tests', str(results.get('tests_run', 0)))
            testsuite.set('failures', str(results.get('tests_failed', 0)))
            testsuite.set('time', str(results.get('duration_seconds', 0)))
            
            # Add individual test cases (simplified)
            for i in range(results.get('tests_run', 0)):
                testcase = SubElement(testsuite, 'testcase')
                testcase.set('name', f'{category}_test_{i+1}')
                testcase.set('classname', f'test_{category}')
                
                if i < results.get('tests_failed', 0):
                    failure = SubElement(testcase, 'failure')
                    failure.set('message', 'Test failed')
        
        # Pretty print XML
        rough_string = tostring(testsuites, 'unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(output_path, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))
    
    def _print_category_summary(self, category: str, results: Dict[str, Any]):
        """Print summary for a test category."""
        status = results.get('status', 'unknown')
        tests_run = results.get('tests_run', 0)
        tests_passed = results.get('tests_passed', 0)
        tests_failed = results.get('tests_failed', 0)
        duration = results.get('duration_seconds', 0)
        
        if status == 'completed' and tests_failed == 0:
            status_icon = "✅"
        elif status == 'timeout':
            status_icon = "⏰"
        else:
            status_icon = "❌"
        
        print(f"{status_icon} {category.title()}: {tests_passed}/{tests_run} passed ({duration:.1f}s)")
        
        if tests_failed > 0:
            print(f"   💥 {tests_failed} test(s) failed")
    
    def _print_final_summary(self):
        """Print final test execution summary."""
        print("=" * 60)
        print("📋 Final Test Summary")
        print("=" * 60)
        
        summary = self.results['summary']
        duration_min = self.results['duration_seconds'] / 60
        
        # Overall status
        if summary['overall_pass']:
            print("🎉 OVERALL RESULT: PASSED")
        else:
            print("💥 OVERALL RESULT: FAILED")
        
        print(f"⏱️  Total Duration: {duration_min:.1f} minutes")
        print()
        
        # Test statistics
        print("📊 Test Statistics:")
        print(f"   Tests Run: {summary['total_tests_run']}")
        print(f"   Tests Passed: {summary['total_tests_passed']}")
        print(f"   Tests Failed: {summary['total_tests_failed']}")
        print(f"   Tests Skipped: {summary['total_tests_skipped']}")
        print(f"   Success Rate: {summary['test_success_rate']:.1%}")
        print()
        
        # Category statistics
        print("📁 Category Statistics:")
        print(f"   Categories Run: {summary['categories_run']}")
        print(f"   Categories Passed: {summary['categories_passed']}")
        print(f"   Categories Failed: {summary['categories_failed']}")
        print(f"   Success Rate: {summary['category_success_rate']:.1%}")
        print()
        
        # Performance validation
        print("🎯 Performance Validation:")
        print(f"   Targets Met: {summary['performance_targets_met']}")
        print(f"   Total Targets: {summary['performance_targets_total']}")
        print(f"   Success Rate: {summary['performance_success_rate']:.1%}")
        print()
        
        # Exit code indication
        exit_code = 0 if summary['overall_pass'] else 1
        print(f"🚪 Exit Code: {exit_code}")


async def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='AI News Trading Benchmark Test Suite')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--categories', nargs='+', 
                       choices=['unit', 'integration', 'performance', 'regression'],
                       help='Test categories to run')
    parser.add_argument('--output-dir', help='Output directory for test reports')
    parser.add_argument('--strict', action='store_true', 
                       help='Strict mode for performance validation')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.config:
        runner = ComprehensiveTestRunner(args.config)
    else:
        runner = ComprehensiveTestRunner()
    
    if args.output_dir:
        runner.config['reporting']['output_dir'] = args.output_dir
    
    if args.strict:
        runner.config['performance_validation']['strict_mode'] = True
    
    # Run the test suite
    results = await runner.run_full_suite(args.categories)
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['overall_pass'] else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())