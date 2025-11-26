"""
Baseline comparison regression test suite for AI News Trading benchmark system.

This module detects performance regressions by comparing current results against established baselines.
Tests include:
- Performance baseline comparison
- Strategy performance regression detection
- System resource usage regression
- API response time regression
- Data quality regression
"""

import json
import time
import statistics
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta
import hashlib

from benchmark.src.benchmarks.runner import BenchmarkRunner
from benchmark.src.analysis.comparator import BaselineComparator
from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.profiling.profiler import RegressionProfiler


class TestPerformanceBaselineComparison:
    """Test performance regression against established baselines."""
    
    @pytest.fixture
    async def baseline_system(self):
        """Create system for baseline comparison testing."""
        config = {
            'baseline_mode': True,
            'performance_tracking': True,
            'regression_detection': True,
            'baseline_storage_path': '/tmp/test_baselines'
        }
        
        system = {
            'benchmark_runner': BenchmarkRunner(config),
            'comparator': BaselineComparator(config),
            'profiler': RegressionProfiler()
        }
        
        for component in system.values():
            await component.initialize()
        
        yield system
        
        for component in system.values():
            await component.shutdown()
    
    @pytest.fixture
    def sample_baseline(self):
        """Create sample baseline for testing."""
        return {
            'metadata': {
                'version': '1.0.0',
                'timestamp': '2024-01-15T10:00:00Z',
                'environment': 'test',
                'git_commit': 'abc123',
                'system_specs': {
                    'cpu_cores': 8,
                    'memory_gb': 16,
                    'python_version': '3.11'
                }
            },
            'performance_metrics': {
                'signal_generation': {
                    'avg_latency_ms': 25.5,
                    'p95_latency_ms': 45.2,
                    'p99_latency_ms': 78.1,
                    'throughput_per_sec': 1250
                },
                'trade_execution': {
                    'avg_latency_ms': 15.8,
                    'p95_latency_ms': 28.3,
                    'p99_latency_ms': 52.7,
                    'throughput_per_sec': 1800
                },
                'data_processing': {
                    'avg_latency_ms': 5.2,
                    'p95_latency_ms': 12.1,
                    'p99_latency_ms': 25.4,
                    'throughput_per_sec': 5000
                }
            },
            'strategy_performance': {
                'momentum': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': 0.08,
                    'win_rate': 0.62,
                    'profit_factor': 1.45
                },
                'arbitrage': {
                    'total_return': 0.08,
                    'sharpe_ratio': 2.2,
                    'max_drawdown': 0.04,
                    'win_rate': 0.78,
                    'profit_factor': 2.15
                },
                'news_sentiment': {
                    'total_return': 0.12,
                    'sharpe_ratio': 1.6,
                    'max_drawdown': 0.06,
                    'win_rate': 0.58,
                    'profit_factor': 1.32
                }
            },
            'resource_usage': {
                'peak_memory_mb': 1024,
                'avg_cpu_percent': 45.2,
                'disk_io_mb': 125.8,
                'network_io_mb': 89.4
            }
        }
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, baseline_system, sample_baseline):
        """Test detection of performance regressions."""
        benchmark_runner = baseline_system['benchmark_runner']
        comparator = baseline_system['comparator']
        
        # Load baseline
        await comparator.load_baseline('test_baseline', sample_baseline)
        
        # Run current benchmark
        current_results = await benchmark_runner.run_suite('regression_test')
        
        # Mock current results that show some regressions
        mock_current_results = {
            'performance_metrics': {
                'signal_generation': {
                    'avg_latency_ms': 35.2,  # Regression: +38% latency
                    'p95_latency_ms': 52.8,  # Regression: +17% latency
                    'p99_latency_ms': 95.3,  # Regression: +22% latency
                    'throughput_per_sec': 1050  # Regression: -16% throughput
                },
                'trade_execution': {
                    'avg_latency_ms': 14.2,  # Improvement: -10% latency
                    'p95_latency_ms': 25.1,  # Improvement: -11% latency
                    'p99_latency_ms': 48.9,  # Improvement: -7% latency
                    'throughput_per_sec': 1950  # Improvement: +8% throughput
                },
                'data_processing': {
                    'avg_latency_ms': 5.8,   # Small regression: +12% latency
                    'p95_latency_ms': 13.5,  # Small regression: +12% latency
                    'p99_latency_ms': 27.1,  # Small regression: +7% latency
                    'throughput_per_sec': 4800  # Small regression: -4% throughput
                }
            }
        }
        
        # Compare against baseline
        comparison_result = await comparator.compare_performance(
            baseline_name='test_baseline',
            current_results=mock_current_results,
            regression_threshold=0.15  # 15% regression threshold
        )
        
        # Validate regression detection
        assert 'regressions' in comparison_result
        assert 'improvements' in comparison_result
        assert 'status' in comparison_result
        
        # Should detect signal generation regressions
        regressions = comparison_result['regressions']
        assert any('signal_generation' in reg['component'] for reg in regressions)
        
        # Should detect trade execution improvements
        improvements = comparison_result['improvements']
        assert any('trade_execution' in imp['component'] for imp in improvements)
        
        # Overall status should indicate regressions found
        assert comparison_result['status'] in ['regression_detected', 'mixed_results']
        
        # Validate regression details
        signal_gen_regressions = [r for r in regressions if 'signal_generation' in r['component']]
        assert len(signal_gen_regressions) > 0, "Should detect signal generation regressions"
        
        for regression in signal_gen_regressions:
            assert regression['severity'] in ['minor', 'major', 'critical']
            assert 'baseline_value' in regression
            assert 'current_value' in regression
            assert 'regression_percent' in regression
            
            # Major regressions should be flagged
            if regression['regression_percent'] > 20:
                assert regression['severity'] in ['major', 'critical']
        
        print(f"\nRegression Detection Results:")
        print(f"  Status: {comparison_result['status']}")
        print(f"  Regressions found: {len(regressions)}")
        print(f"  Improvements found: {len(improvements)}")
        
        for reg in regressions[:3]:  # Show first 3 regressions
            print(f"  - {reg['component']}: {reg['regression_percent']:.1f}% regression ({reg['severity']})")
    
    @pytest.mark.asyncio
    async def test_strategy_performance_regression(self, baseline_system, sample_baseline):
        """Test strategy performance regression detection."""
        comparator = baseline_system['comparator']
        
        # Load baseline
        await comparator.load_baseline('strategy_baseline', sample_baseline)
        
        # Mock current strategy results with some regressions
        current_strategy_results = {
            'strategy_performance': {
                'momentum': {
                    'total_return': 0.08,   # Regression: -47% return
                    'sharpe_ratio': 1.3,    # Regression: -28% sharpe
                    'max_drawdown': 0.12,   # Regression: +50% drawdown
                    'win_rate': 0.55,       # Regression: -11% win rate
                    'profit_factor': 1.15   # Regression: -21% profit factor
                },
                'arbitrage': {
                    'total_return': 0.085,  # Improvement: +6% return
                    'sharpe_ratio': 2.35,   # Improvement: +7% sharpe
                    'max_drawdown': 0.035,  # Improvement: -12% drawdown
                    'win_rate': 0.82,       # Improvement: +5% win rate
                    'profit_factor': 2.28   # Improvement: +6% profit factor
                },
                'news_sentiment': {
                    'total_return': 0.10,   # Regression: -17% return
                    'sharpe_ratio': 1.4,    # Regression: -12% sharpe
                    'max_drawdown': 0.08,   # Regression: +33% drawdown
                    'win_rate': 0.52,       # Regression: -10% win rate
                    'profit_factor': 1.18   # Regression: -11% profit factor
                }
            }
        }
        
        # Compare strategy performance
        strategy_comparison = await comparator.compare_strategy_performance(
            baseline_name='strategy_baseline',
            current_results=current_strategy_results,
            critical_metrics=['total_return', 'sharpe_ratio', 'max_drawdown']
        )
        
        # Validate strategy regression detection
        assert 'strategy_regressions' in strategy_comparison
        assert 'strategy_improvements' in strategy_comparison
        assert 'overall_assessment' in strategy_comparison
        
        strategy_regressions = strategy_comparison['strategy_regressions']
        
        # Should detect momentum strategy regressions
        momentum_regressions = [r for r in strategy_regressions if r['strategy'] == 'momentum']
        assert len(momentum_regressions) > 0, "Should detect momentum strategy regressions"
        
        # Should detect news_sentiment strategy regressions
        news_regressions = [r for r in strategy_regressions if r['strategy'] == 'news_sentiment']
        assert len(news_regressions) > 0, "Should detect news sentiment strategy regressions"
        
        # Arbitrage should show improvements
        strategy_improvements = strategy_comparison['strategy_improvements']
        arbitrage_improvements = [i for i in strategy_improvements if i['strategy'] == 'arbitrage']
        assert len(arbitrage_improvements) > 0, "Should detect arbitrage strategy improvements"
        
        # Validate critical metric analysis
        for regression in strategy_regressions:
            if regression['metric'] in ['total_return', 'sharpe_ratio']:
                assert regression['severity'] in ['major', 'critical'], \
                    f"Critical metric {regression['metric']} regression should be major/critical"
        
        print(f"\nStrategy Regression Analysis:")
        print(f"  Overall assessment: {strategy_comparison['overall_assessment']}")
        print(f"  Strategy regressions: {len(strategy_regressions)}")
        print(f"  Strategy improvements: {len(strategy_improvements)}")
    
    @pytest.mark.asyncio
    async def test_resource_usage_regression(self, baseline_system, sample_baseline):
        """Test resource usage regression detection."""
        comparator = baseline_system['comparator']
        
        # Load baseline
        await comparator.load_baseline('resource_baseline', sample_baseline)
        
        # Mock current resource usage with regressions
        current_resource_usage = {
            'resource_usage': {
                'peak_memory_mb': 1450,    # Regression: +42% memory usage
                'avg_cpu_percent': 38.7,   # Improvement: -14% CPU usage
                'disk_io_mb': 165.3,       # Regression: +31% disk I/O
                'network_io_mb': 78.2      # Improvement: -13% network I/O
            }
        }
        
        # Compare resource usage
        resource_comparison = await comparator.compare_resource_usage(
            baseline_name='resource_baseline',
            current_results=current_resource_usage,
            memory_threshold=0.25,  # 25% memory increase threshold
            cpu_threshold=0.20,     # 20% CPU increase threshold
            io_threshold=0.30       # 30% I/O increase threshold
        )
        
        # Validate resource regression detection
        assert 'resource_regressions' in resource_comparison
        assert 'resource_improvements' in resource_comparison
        assert 'resource_status' in resource_comparison
        
        resource_regressions = resource_comparison['resource_regressions']
        
        # Should detect memory regression
        memory_regressions = [r for r in resource_regressions if 'memory' in r['metric']]
        assert len(memory_regressions) > 0, "Should detect memory usage regression"
        
        # Should detect disk I/O regression
        disk_regressions = [r for r in resource_regressions if 'disk' in r['metric']]
        assert len(disk_regressions) > 0, "Should detect disk I/O regression"
        
        # Should detect CPU and network improvements
        resource_improvements = resource_comparison['resource_improvements']
        cpu_improvements = [i for i in resource_improvements if 'cpu' in i['metric']]
        network_improvements = [i for i in resource_improvements if 'network' in i['metric']]
        
        assert len(cpu_improvements) > 0, "Should detect CPU usage improvement"
        assert len(network_improvements) > 0, "Should detect network I/O improvement"
        
        # Resource status should indicate regressions
        assert resource_comparison['resource_status'] in ['regression_detected', 'mixed_results']
        
        print(f"\nResource Usage Analysis:")
        print(f"  Resource status: {resource_comparison['resource_status']}")
        print(f"  Resource regressions: {len(resource_regressions)}")
        print(f"  Resource improvements: {len(resource_improvements)}")


class TestTimeSeriesRegressionAnalysis:
    """Test time-series regression analysis for trend detection."""
    
    @pytest.fixture
    def historical_performance_data(self):
        """Create historical performance data for trend analysis."""
        base_date = datetime.now() - timedelta(days=30)
        
        data = []
        for i in range(30):  # 30 days of data
            # Simulate gradual performance degradation
            degradation_factor = 1 + (i * 0.005)  # 0.5% degradation per day
            
            daily_data = {
                'date': (base_date + timedelta(days=i)).isoformat(),
                'performance_metrics': {
                    'avg_latency_ms': 25.0 * degradation_factor + np.random.normal(0, 2),
                    'p99_latency_ms': 75.0 * degradation_factor + np.random.normal(0, 5),
                    'throughput_per_sec': 1200 / degradation_factor + np.random.normal(0, 50),
                    'memory_usage_mb': 800 * degradation_factor + np.random.normal(0, 50)
                },
                'strategy_performance': {
                    'momentum': {
                        'daily_return': np.random.normal(0.001, 0.02),  # 0.1% daily return with volatility
                        'sharpe_ratio': max(0.5, 1.8 / degradation_factor + np.random.normal(0, 0.1))
                    }
                }
            }
            data.append(daily_data)
        
        return data
    
    @pytest.mark.asyncio
    async def test_performance_trend_regression_detection(self, historical_performance_data):
        """Test detection of performance trend regressions."""
        comparator = BaselineComparator({'trend_analysis': True})
        await comparator.initialize()
        
        # Analyze performance trends
        trend_analysis = await comparator.analyze_performance_trends(
            historical_data=historical_performance_data,
            trend_window_days=7,    # 7-day moving average
            regression_threshold=0.10,  # 10% degradation threshold
            min_data_points=14      # Minimum 14 days for analysis
        )
        
        # Validate trend analysis
        assert 'trends' in trend_analysis
        assert 'regression_trends' in trend_analysis
        assert 'trend_status' in trend_analysis
        
        trends = trend_analysis['trends']
        regression_trends = trend_analysis['regression_trends']
        
        # Should detect latency degradation trend
        latency_trends = [t for t in trends if 'latency' in t['metric']]
        assert len(latency_trends) > 0, "Should detect latency trends"
        
        # Should detect throughput degradation trend
        throughput_trends = [t for t in trends if 'throughput' in t['metric']]
        assert len(throughput_trends) > 0, "Should detect throughput trends"
        
        # Should identify negative trends as regressions
        for trend in regression_trends:
            assert trend['direction'] in ['degrading', 'declining']
            assert abs(trend['trend_slope']) > 0.01  # Meaningful trend
            assert 'projected_impact' in trend
        
        # Validate trend severity classification
        for trend in regression_trends:
            if abs(trend['trend_slope']) > 0.05:  # >5% change per week
                assert trend['severity'] in ['major', 'critical']
            else:
                assert trend['severity'] in ['minor', 'moderate']
        
        print(f"\nTrend Analysis Results:")
        print(f"  Trend status: {trend_analysis['trend_status']}")
        print(f"  Total trends: {len(trends)}")
        print(f"  Regression trends: {len(regression_trends)}")
        
        for trend in regression_trends[:3]:
            print(f"  - {trend['metric']}: {trend['direction']} trend, {trend['severity']} severity")
        
        await comparator.shutdown()
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_in_performance(self, historical_performance_data):
        """Test anomaly detection in performance metrics."""
        comparator = BaselineComparator({'anomaly_detection': True})
        await comparator.initialize()
        
        # Add some anomalous data points
        anomalous_data = historical_performance_data.copy()
        
        # Inject anomalies
        anomalous_data[10]['performance_metrics']['avg_latency_ms'] = 150.0  # 6x normal latency
        anomalous_data[15]['performance_metrics']['throughput_per_sec'] = 300  # 4x lower throughput
        anomalous_data[20]['performance_metrics']['memory_usage_mb'] = 2500   # 3x memory usage
        
        # Detect anomalies
        anomaly_analysis = await comparator.detect_performance_anomalies(
            data=anomalous_data,
            z_score_threshold=2.5,  # 2.5 standard deviations
            isolation_forest_contamination=0.1  # 10% expected anomalies
        )
        
        # Validate anomaly detection
        assert 'anomalies' in anomaly_analysis
        assert 'anomaly_summary' in anomaly_analysis
        assert 'detection_methods' in anomaly_analysis
        
        anomalies = anomaly_analysis['anomalies']
        
        # Should detect the injected anomalies
        latency_anomalies = [a for a in anomalies if 'latency' in a['metric']]
        throughput_anomalies = [a for a in anomalies if 'throughput' in a['metric']]
        memory_anomalies = [a for a in anomalies if 'memory' in a['metric']]
        
        assert len(latency_anomalies) > 0, "Should detect latency anomalies"
        assert len(throughput_anomalies) > 0, "Should detect throughput anomalies"
        assert len(memory_anomalies) > 0, "Should detect memory anomalies"
        
        # Validate anomaly details
        for anomaly in anomalies:
            assert 'timestamp' in anomaly
            assert 'metric' in anomaly
            assert 'value' in anomaly
            assert 'expected_range' in anomaly
            assert 'severity' in anomaly
            assert 'detection_method' in anomaly
        
        # Check anomaly severity classification
        severe_anomalies = [a for a in anomalies if a['severity'] in ['major', 'critical']]
        assert len(severe_anomalies) > 0, "Should classify some anomalies as severe"
        
        print(f"\nAnomaly Detection Results:")
        print(f"  Total anomalies: {len(anomalies)}")
        print(f"  Severe anomalies: {len(severe_anomalies)}")
        print(f"  Detection methods used: {anomaly_analysis['detection_methods']}")
        
        await comparator.shutdown()


class TestBaselineManagement:
    """Test baseline creation, storage, and management."""
    
    @pytest.mark.asyncio
    async def test_baseline_creation_and_storage(self):
        """Test creating and storing performance baselines."""
        config = {
            'baseline_storage_path': '/tmp/test_baselines',
            'baseline_versioning': True,
            'automatic_baseline_updates': False
        }
        
        comparator = BaselineComparator(config)
        await comparator.initialize()
        
        # Create new baseline from current performance
        simulator = MarketSimulator({'performance_mode': True})
        benchmark_runner = BenchmarkRunner({'baseline_creation': True})
        
        await simulator.initialize()
        await benchmark_runner.initialize()
        
        # Run performance benchmarks
        performance_results = await benchmark_runner.run_baseline_suite()
        
        # Create baseline
        baseline_metadata = {
            'name': 'test_baseline_v1',
            'version': '1.0.0',
            'description': 'Test baseline for regression testing',
            'environment': 'test',
            'created_by': 'automated_test',
            'git_commit': 'test_commit_123'
        }
        
        baseline_id = await comparator.create_baseline(
            results=performance_results,
            metadata=baseline_metadata
        )
        
        # Validate baseline creation
        assert baseline_id is not None
        assert isinstance(baseline_id, str)
        
        # Verify baseline can be retrieved
        stored_baseline = await comparator.get_baseline(baseline_id)
        
        assert stored_baseline is not None
        assert stored_baseline['metadata']['name'] == 'test_baseline_v1'
        assert stored_baseline['metadata']['version'] == '1.0.0'
        assert 'performance_metrics' in stored_baseline
        assert 'timestamp' in stored_baseline['metadata']
        
        # Test baseline listing
        baselines = await comparator.list_baselines()
        
        test_baselines = [b for b in baselines if b['name'] == 'test_baseline_v1']
        assert len(test_baselines) == 1
        
        # Test baseline validation
        validation_result = await comparator.validate_baseline(baseline_id)
        
        assert validation_result['valid'] is True
        assert 'validation_errors' in validation_result
        assert len(validation_result['validation_errors']) == 0
        
        print(f"\nBaseline Management Test:")
        print(f"  Baseline ID: {baseline_id}")
        print(f"  Baseline valid: {validation_result['valid']}")
        print(f"  Total baselines: {len(baselines)}")
        
        # Cleanup
        await comparator.delete_baseline(baseline_id)
        await simulator.shutdown()
        await benchmark_runner.shutdown()
        await comparator.shutdown()
    
    @pytest.mark.asyncio
    async def test_baseline_comparison_report_generation(self):
        """Test comprehensive baseline comparison report generation."""
        config = {
            'report_generation': True,
            'detailed_analysis': True,
            'export_formats': ['json', 'html', 'csv']
        }
        
        comparator = BaselineComparator(config)
        await comparator.initialize()
        
        # Mock baseline and current results
        baseline_data = {
            'performance_metrics': {
                'latency_p99': 75.0,
                'throughput': 1200,
                'memory_mb': 800
            },
            'strategy_performance': {
                'momentum': {'sharpe_ratio': 1.8},
                'arbitrage': {'sharpe_ratio': 2.2}
            }
        }
        
        current_results = {
            'performance_metrics': {
                'latency_p99': 85.0,  # 13% regression
                'throughput': 1350,   # 12% improvement
                'memory_mb': 920      # 15% regression
            },
            'strategy_performance': {
                'momentum': {'sharpe_ratio': 1.65},  # 8% regression
                'arbitrage': {'sharpe_ratio': 2.35}  # 7% improvement
            }
        }
        
        # Generate comprehensive comparison report
        report = await comparator.generate_comparison_report(
            baseline_data=baseline_data,
            current_results=current_results,
            include_charts=True,
            include_recommendations=True
        )
        
        # Validate report structure
        assert 'executive_summary' in report
        assert 'detailed_analysis' in report
        assert 'performance_comparison' in report
        assert 'strategy_comparison' in report
        assert 'recommendations' in report
        assert 'metadata' in report
        
        # Validate executive summary
        exec_summary = report['executive_summary']
        assert 'overall_status' in exec_summary
        assert 'key_findings' in exec_summary
        assert 'critical_issues' in exec_summary
        
        # Validate detailed analysis
        detailed = report['detailed_analysis']
        assert 'regressions' in detailed
        assert 'improvements' in detailed
        assert 'stability_metrics' in detailed
        
        # Validate recommendations
        recommendations = report['recommendations']
        assert 'immediate_actions' in recommendations
        assert 'long_term_improvements' in recommendations
        assert 'monitoring_suggestions' in recommendations
        
        print(f"\nComparison Report Generated:")
        print(f"  Overall status: {exec_summary['overall_status']}")
        print(f"  Key findings: {len(exec_summary['key_findings'])}")
        print(f"  Recommendations: {len(recommendations['immediate_actions'])}")
        
        await comparator.shutdown()


if __name__ == '__main__':
    pytest.main([__file__])