#!/usr/bin/env python3
"""
Performance Target Validation Script for AI News Trading Benchmark System.

This script validates that all performance targets are met:
- Signal generation latency < 100ms (P99)
- Throughput > 1000 trades/second
- Memory usage < 2GB for 8-hour simulation
- 100+ concurrent symbols supported
- Real-time data latency < 50ms

Usage:
    python validate_performance_targets.py [--config CONFIG_FILE] [--strict] [--output OUTPUT_FILE]
"""

import asyncio
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import yaml
import traceback

# Import the test runner
from run_full_suite import ComprehensiveTestRunner


class PerformanceTargetValidator:
    """Validates that all performance targets are met."""
    
    def __init__(self, config_path: Optional[str] = None, strict_mode: bool = False):
        """Initialize the performance validator."""
        self.config = self._load_config(config_path)
        self.strict_mode = strict_mode
        
        # Define the official performance targets
        self.performance_targets = {
            'signal_generation_latency_p99_ms': {
                'target': 100,
                'description': 'Signal generation latency P99 must be < 100ms',
                'type': 'latency',
                'critical': True
            },
            'trade_throughput_per_sec': {
                'target': 1000,
                'description': 'Trade execution throughput must be > 1000 trades/second',
                'type': 'throughput',
                'critical': True
            },
            'memory_usage_8hr_simulation_gb': {
                'target': 2.0,
                'description': 'Memory usage during 8-hour simulation must be < 2GB',
                'type': 'memory',
                'critical': True
            },
            'concurrent_symbols_supported': {
                'target': 100,
                'description': 'System must support 100+ concurrent symbols',
                'type': 'scalability',
                'critical': True
            },
            'realtime_data_latency_ms': {
                'target': 50,
                'description': 'Real-time data processing latency must be < 50ms',
                'type': 'latency',
                'critical': True
            },
            'data_processing_throughput_per_sec': {
                'target': 5000,
                'description': 'Data processing throughput must be > 5000 updates/second',
                'type': 'throughput',
                'critical': False  # Secondary target
            },
            'system_availability_percent': {
                'target': 99.9,
                'description': 'System availability must be > 99.9%',
                'type': 'reliability',
                'critical': True
            },
            'error_rate_percent': {
                'target': 1.0,
                'description': 'Error rate must be < 1%',
                'type': 'reliability',
                'critical': True
            }
        }
        
        self.validation_results = {
            'validation_timestamp': None,
            'targets_validated': 0,
            'targets_passed': 0,
            'targets_failed': 0,
            'critical_failures': 0,
            'overall_pass': False,
            'target_details': {},
            'test_execution_summary': {},
            'recommendations': []
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration."""
        default_config = {
            'validation': {
                'tolerance_percent': 5,  # 5% tolerance for targets
                'require_all_critical': True,
                'max_test_duration_minutes': 60
            },
            'test_execution': {
                'categories': ['performance', 'integration'],
                'performance_focus': True,
                'detailed_metrics': True
            },
            'reporting': {
                'detailed_breakdown': True,
                'include_recommendations': True,
                'output_format': 'json'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
                
                # Merge configurations
                return self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    async def validate_all_targets(self) -> Dict[str, Any]:
        """Run comprehensive validation of all performance targets."""
        print("🎯 AI News Trading Benchmark - Performance Target Validation")
        print("=" * 70)
        print()
        
        self.validation_results['validation_timestamp'] = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        
        try:
            # Step 1: Run performance-focused test suite
            print("📊 Step 1: Executing Performance Test Suite")
            print("-" * 45)
            test_results = await self._run_performance_tests()
            self.validation_results['test_execution_summary'] = test_results
            
            # Step 2: Extract and validate performance metrics
            print("\n🔍 Step 2: Extracting Performance Metrics")
            print("-" * 42)
            await self._extract_performance_metrics(test_results)
            
            # Step 3: Validate each target
            print("\n✅ Step 3: Validating Performance Targets")
            print("-" * 42)
            await self._validate_individual_targets()
            
            # Step 4: Generate overall assessment
            print("\n📋 Step 4: Generating Assessment")
            print("-" * 35)
            await self._generate_overall_assessment()
            
            # Step 5: Provide recommendations
            print("\n💡 Step 5: Generating Recommendations")
            print("-" * 39)
            await self._generate_recommendations()
            
        except Exception as e:
            print(f"❌ Critical error during validation: {e}")
            traceback.print_exc()
            self.validation_results['critical_error'] = {
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        
        finally:
            end_time = time.perf_counter()
            self.validation_results['validation_duration_seconds'] = end_time - start_time
            
            # Print final results
            self._print_validation_summary()
        
        return self.validation_results
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run the performance-focused test suite."""
        # Use performance test configuration
        config_path = Path(__file__).parent / "test_configs" / "performance_test_config.yaml"
        
        runner = ComprehensiveTestRunner(str(config_path))
        
        # Focus on performance and related integration tests
        test_categories = self.config['test_execution']['categories']
        
        print(f"  Running test categories: {', '.join(test_categories)}")
        print("  This may take several minutes...")
        print()
        
        test_results = await runner.run_full_suite(test_categories)
        
        # Extract relevant results
        return {
            'overall_status': test_results['summary']['overall_pass'],
            'duration_seconds': test_results['duration_seconds'],
            'categories': test_results['test_categories'],
            'performance_data': self._extract_performance_data(test_results)
        }
    
    def _extract_performance_data(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance data from test results."""
        performance_data = {}
        
        # Extract from performance test category
        if 'performance' in test_results['test_categories']:
            perf_category = test_results['test_categories']['performance']
            performance_data['performance_tests'] = perf_category.get('details', {})
        
        # Extract from integration tests that include performance metrics
        if 'integration' in test_results['test_categories']:
            integration_category = test_results['test_categories']['integration']
            performance_data['integration_performance'] = integration_category.get('details', {})
        
        return performance_data
    
    async def _extract_performance_metrics(self, test_results: Dict[str, Any]):
        """Extract specific performance metrics for validation."""
        performance_data = test_results.get('performance_data', {})
        
        print("  Extracting metrics from test results...")
        
        # Map test results to validation targets
        metric_extractors = {
            'signal_generation_latency_p99_ms': self._extract_signal_latency,
            'trade_throughput_per_sec': self._extract_trade_throughput,
            'memory_usage_8hr_simulation_gb': self._extract_memory_usage,
            'concurrent_symbols_supported': self._extract_concurrent_symbols,
            'realtime_data_latency_ms': self._extract_realtime_latency,
            'data_processing_throughput_per_sec': self._extract_data_throughput,
            'system_availability_percent': self._extract_availability,
            'error_rate_percent': self._extract_error_rate
        }
        
        for target_name, extractor in metric_extractors.items():
            try:
                actual_value = await extractor(performance_data)
                self.validation_results['target_details'][target_name] = {
                    'actual_value': actual_value,
                    'extraction_successful': actual_value is not None
                }
                
                if actual_value is not None:
                    print(f"    ✅ {target_name}: {actual_value}")
                else:
                    print(f"    ⚠️  {target_name}: No data found")
                    
            except Exception as e:
                print(f"    ❌ {target_name}: Extraction error - {e}")
                self.validation_results['target_details'][target_name] = {
                    'actual_value': None,
                    'extraction_successful': False,
                    'error': str(e)
                }
    
    async def _extract_signal_latency(self, performance_data: Dict) -> Optional[float]:
        """Extract signal generation P99 latency."""
        # Look for latency data in performance tests
        latency_tests = performance_data.get('performance_tests', {}).get('latency_targets', {})
        
        if 'performance_metrics' in latency_tests:
            return latency_tests['performance_metrics'].get('p99_latency_ms')
        
        # Fallback: look in integration tests
        integration_perf = performance_data.get('integration_performance', {})
        for test_name, test_data in integration_perf.items():
            if 'latency' in test_name.lower() and 'performance_metrics' in test_data:
                return test_data['performance_metrics'].get('p99_latency_ms')
        
        return None
    
    async def _extract_trade_throughput(self, performance_data: Dict) -> Optional[float]:
        """Extract trade execution throughput."""
        throughput_tests = performance_data.get('performance_tests', {}).get('throughput_targets', {})
        
        if 'performance_metrics' in throughput_tests:
            return throughput_tests['performance_metrics'].get('throughput_per_sec')
        
        return None
    
    async def _extract_memory_usage(self, performance_data: Dict) -> Optional[float]:
        """Extract memory usage from 8-hour simulation."""
        memory_tests = performance_data.get('performance_tests', {}).get('memory_usage', {})
        
        if 'performance_metrics' in memory_tests:
            memory_mb = memory_tests['performance_metrics'].get('peak_memory_mb')
            return memory_mb / 1024 if memory_mb else None  # Convert to GB
        
        return None
    
    async def _extract_concurrent_symbols(self, performance_data: Dict) -> Optional[int]:
        """Extract concurrent symbols capability."""
        scalability_tests = performance_data.get('performance_tests', {}).get('scalability', {})
        
        if 'performance_metrics' in scalability_tests:
            return scalability_tests['performance_metrics'].get('max_concurrent_symbols')
        
        return None
    
    async def _extract_realtime_latency(self, performance_data: Dict) -> Optional[float]:
        """Extract real-time data processing latency."""
        # Look in integration tests for real-time performance
        integration_perf = performance_data.get('integration_performance', {})
        
        realtime_test = integration_perf.get('realtime_simulation', {})
        if 'performance_metrics' in realtime_test:
            return realtime_test['performance_metrics'].get('avg_latency_ms')
        
        return None
    
    async def _extract_data_throughput(self, performance_data: Dict) -> Optional[float]:
        """Extract data processing throughput."""
        performance_tests = performance_data.get('performance_tests', {})
        
        # Look for data processing metrics
        for test_name, test_data in performance_tests.items():
            if 'data' in test_name.lower() and 'performance_metrics' in test_data:
                return test_data['performance_metrics'].get('throughput_per_sec')
        
        return None
    
    async def _extract_availability(self, performance_data: Dict) -> Optional[float]:
        """Extract system availability percentage."""
        # Calculate from error rates and uptime
        error_rate = await self._extract_error_rate(performance_data)
        if error_rate is not None:
            return 100 - error_rate  # Simple approximation
        
        return None
    
    async def _extract_error_rate(self, performance_data: Dict) -> Optional[float]:
        """Extract error rate percentage."""
        # Look across all tests for error rates
        total_tests = 0
        total_errors = 0
        
        for category_data in performance_data.values():
            if isinstance(category_data, dict):
                for test_data in category_data.values():
                    if isinstance(test_data, dict):
                        tests_run = test_data.get('tests_run', 0)
                        tests_failed = test_data.get('tests_failed', 0)
                        total_tests += tests_run
                        total_errors += tests_failed
        
        if total_tests > 0:
            return (total_errors / total_tests) * 100
        
        return None
    
    async def _validate_individual_targets(self):
        """Validate each performance target."""
        tolerance_percent = self.config['validation']['tolerance_percent']
        
        for target_name, target_config in self.performance_targets.items():
            target_value = target_config['target']
            target_type = target_config['type']
            is_critical = target_config['critical']
            
            target_detail = self.validation_results['target_details'].get(target_name, {})
            actual_value = target_detail.get('actual_value')
            
            validation_result = {
                'target_value': target_value,
                'actual_value': actual_value,
                'target_type': target_type,
                'is_critical': is_critical,
                'description': target_config['description'],
                'passed': False,
                'within_tolerance': False,
                'deviation_percent': None,
                'status': 'no_data'
            }
            
            if actual_value is not None:
                # Calculate deviation and determine pass/fail
                if target_type in ['latency', 'memory']:
                    # Lower is better
                    passed = actual_value <= target_value
                    deviation = ((actual_value - target_value) / target_value) * 100
                elif target_type in ['throughput', 'scalability']:
                    # Higher is better
                    passed = actual_value >= target_value
                    deviation = ((target_value - actual_value) / target_value) * 100
                elif target_type == 'reliability':
                    if 'error_rate' in target_name:
                        # Lower is better for error rates
                        passed = actual_value <= target_value
                        deviation = ((actual_value - target_value) / target_value) * 100
                    else:
                        # Higher is better for availability
                        passed = actual_value >= target_value
                        deviation = ((target_value - actual_value) / target_value) * 100
                else:
                    passed = False
                    deviation = 0
                
                validation_result.update({
                    'passed': passed,
                    'within_tolerance': abs(deviation) <= tolerance_percent,
                    'deviation_percent': deviation,
                    'status': 'passed' if passed else 'failed'
                })
                
                # Update counters
                self.validation_results['targets_validated'] += 1
                
                if passed or (validation_result['within_tolerance'] and not self.strict_mode):
                    self.validation_results['targets_passed'] += 1
                    status_icon = "✅" if passed else "⚠️"
                else:
                    self.validation_results['targets_failed'] += 1
                    if is_critical:
                        self.validation_results['critical_failures'] += 1
                    status_icon = "❌"
                
                print(f"  {status_icon} {target_name}: {actual_value} (target: {target_value})")
                if abs(deviation) > 0:
                    print(f"      Deviation: {deviation:+.1f}%")
                
            else:
                print(f"  ❓ {target_name}: No measurement data available")
                validation_result['status'] = 'no_data'
                self.validation_results['targets_failed'] += 1
                if is_critical:
                    self.validation_results['critical_failures'] += 1
            
            self.validation_results['target_details'][target_name] = validation_result
    
    async def _generate_overall_assessment(self):
        """Generate overall validation assessment."""
        total_targets = len(self.performance_targets)
        critical_targets = sum(1 for t in self.performance_targets.values() if t['critical'])
        
        # Determine overall pass/fail
        require_all_critical = self.config['validation']['require_all_critical']
        
        if require_all_critical:
            # All critical targets must pass
            critical_passed = critical_targets - self.validation_results['critical_failures']
            overall_pass = (self.validation_results['critical_failures'] == 0 and 
                          self.validation_results['targets_failed'] <= (total_targets - critical_targets))
        else:
            # Majority of targets must pass
            overall_pass = self.validation_results['targets_passed'] > (total_targets / 2)
        
        self.validation_results['overall_pass'] = overall_pass
        
        print(f"  📊 Validation Summary:")
        print(f"     Total targets: {total_targets}")
        print(f"     Targets validated: {self.validation_results['targets_validated']}")
        print(f"     Targets passed: {self.validation_results['targets_passed']}")
        print(f"     Targets failed: {self.validation_results['targets_failed']}")
        print(f"     Critical failures: {self.validation_results['critical_failures']}")
        print(f"     Overall result: {'PASSED' if overall_pass else 'FAILED'}")
    
    async def _generate_recommendations(self):
        """Generate optimization recommendations based on validation results."""
        recommendations = []
        
        # Analyze failed targets and generate specific recommendations
        for target_name, target_detail in self.validation_results['target_details'].items():
            if target_detail.get('status') == 'failed':
                target_config = self.performance_targets[target_name]
                
                if 'latency' in target_name:
                    recommendations.append({
                        'type': 'performance_optimization',
                        'priority': 'high' if target_config['critical'] else 'medium',
                        'target': target_name,
                        'recommendation': 'Optimize signal generation pipeline, consider caching and parallel processing',
                        'specific_actions': [
                            'Profile signal generation code for bottlenecks',
                            'Implement result caching for repeated calculations',
                            'Consider asynchronous processing where possible',
                            'Optimize database queries and data structures'
                        ]
                    })
                
                elif 'throughput' in target_name:
                    recommendations.append({
                        'type': 'scalability_improvement',
                        'priority': 'high' if target_config['critical'] else 'medium',
                        'target': target_name,
                        'recommendation': 'Increase system throughput capacity',
                        'specific_actions': [
                            'Implement connection pooling',
                            'Add load balancing across multiple workers',
                            'Optimize batch processing capabilities',
                            'Consider horizontal scaling architecture'
                        ]
                    })
                
                elif 'memory' in target_name:
                    recommendations.append({
                        'type': 'memory_optimization',
                        'priority': 'high' if target_config['critical'] else 'medium',
                        'target': target_name,
                        'recommendation': 'Reduce memory usage and improve memory management',
                        'specific_actions': [
                            'Implement memory pooling for frequently used objects',
                            'Optimize data structures for memory efficiency',
                            'Improve garbage collection tuning',
                            'Consider streaming processing for large datasets'
                        ]
                    })
        
        # General recommendations based on overall performance
        if self.validation_results['critical_failures'] > 0:
            recommendations.append({
                'type': 'architecture_review',
                'priority': 'critical',
                'target': 'overall_system',
                'recommendation': 'Critical performance targets not met - architecture review recommended',
                'specific_actions': [
                    'Conduct comprehensive performance profiling',
                    'Review system architecture for scalability bottlenecks',
                    'Consider microservices architecture for better resource isolation',
                    'Implement performance monitoring and alerting'
                ]
            })
        
        self.validation_results['recommendations'] = recommendations
        
        # Print recommendations
        if recommendations:
            print(f"  💡 Generated {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"     {i}. {rec['recommendation']} (Priority: {rec['priority']})")
        else:
            print("  🎉 No optimization recommendations - all targets met!")
    
    def _print_validation_summary(self):
        """Print final validation summary."""
        print("\n" + "=" * 70)
        print("🎯 Performance Target Validation Summary")
        print("=" * 70)
        
        overall_pass = self.validation_results['overall_pass']
        duration_min = self.validation_results['validation_duration_seconds'] / 60
        
        if overall_pass:
            print("🎉 VALIDATION RESULT: PASSED")
            print("   All critical performance targets have been met!")
        else:
            print("💥 VALIDATION RESULT: FAILED")
            print("   Some critical performance targets were not met.")
        
        print(f"⏱️  Validation Duration: {duration_min:.1f} minutes")
        print()
        
        # Print target summary
        print("📊 Target Summary:")
        critical_count = sum(1 for t in self.performance_targets.values() if t['critical'])
        print(f"   Total Targets: {len(self.performance_targets)} ({critical_count} critical)")
        print(f"   Targets Passed: {self.validation_results['targets_passed']}")
        print(f"   Targets Failed: {self.validation_results['targets_failed']}")
        print(f"   Critical Failures: {self.validation_results['critical_failures']}")
        print()
        
        # Print failed targets
        failed_targets = [
            name for name, detail in self.validation_results['target_details'].items()
            if detail.get('status') == 'failed'
        ]
        
        if failed_targets:
            print("❌ Failed Targets:")
            for target_name in failed_targets:
                target_detail = self.validation_results['target_details'][target_name]
                target_config = self.performance_targets[target_name]
                actual = target_detail.get('actual_value', 'N/A')
                expected = target_config['target']
                critical_marker = " (CRITICAL)" if target_config['critical'] else ""
                print(f"   • {target_name}: {actual} (expected: {expected}){critical_marker}")
        
        # Print recommendations count
        rec_count = len(self.validation_results.get('recommendations', []))
        if rec_count > 0:
            print(f"\n💡 {rec_count} optimization recommendations generated")
        
        print(f"\n🚪 Exit Code: {0 if overall_pass else 1}")
    
    async def save_results(self, output_path: str):
        """Save validation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"📄 Validation results saved to: {output_file}")


async def main():
    """Main entry point for performance target validation."""
    parser = argparse.ArgumentParser(
        description='Validate AI News Trading Benchmark Performance Targets'
    )
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--strict', action='store_true',
                       help='Strict mode - no tolerance for target deviations')
    parser.add_argument('--output', default='./test_results/performance_validation.json',
                       help='Output file for validation results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create validator
    validator = PerformanceTargetValidator(
        config_path=args.config,
        strict_mode=args.strict
    )
    
    try:
        # Run validation
        results = await validator.validate_all_targets()
        
        # Save results
        await validator.save_results(args.output)
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_pass'] else 1
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        if args.verbose:
            traceback.print_exc()
        exit_code = 2
    
    sys.exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())