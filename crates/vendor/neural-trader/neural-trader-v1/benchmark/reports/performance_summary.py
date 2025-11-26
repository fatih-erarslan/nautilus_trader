#!/usr/bin/env python3
"""
Performance Summary Generator for AI News Trading Platform.

This module generates comprehensive performance summaries including:
- Key performance indicators
- Performance trends analysis
- Bottleneck identification
- Resource utilization analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    target: float
    status: str  # PASS, FAIL, WARNING
    category: str
    critical: bool = False
    trend: Optional[str] = None  # IMPROVING, DEGRADING, STABLE


@dataclass
class CategorySummary:
    """Summary for a performance category"""
    category: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    pass_rate: float
    average_performance: float
    critical_failures: List[str] = field(default_factory=list)
    key_metrics: List[PerformanceMetric] = field(default_factory=list)


@dataclass
class PerformanceBottleneck:
    """Identified performance bottleneck"""
    component: str
    severity: str  # HIGH, MEDIUM, LOW
    impact: str
    description: str
    affected_metrics: List[str]
    recommended_actions: List[str] = field(default_factory=list)


class PerformanceSummaryGenerator:
    """Generates comprehensive performance summaries"""
    
    def __init__(self):
        """Initialize performance summary generator"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_summary(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance summary
        
        Args:
            validation_data: Performance validation results
            
        Returns:
            Dictionary containing performance summary
        """
        self.logger.info("Generating performance summary...")
        
        try:
            results = validation_data.get('results', [])
            targets = validation_data.get('targets', {})
            summary_data = validation_data.get('summary', {})
            
            # Generate key metrics
            key_metrics = self._extract_key_metrics(results, targets)
            
            # Analyze by category
            category_summaries = self._analyze_by_category(results)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(results)
            
            # Analyze trends
            trends = self._analyze_performance_trends(results)
            
            # Generate insights
            insights = self._generate_performance_insights(
                key_metrics, category_summaries, bottlenecks, trends
            )
            
            # Calculate overall scores
            scores = self._calculate_performance_scores(results, category_summaries)
            
            return {
                'key_metrics': self._serialize_metrics(key_metrics),
                'category_summaries': self._serialize_category_summaries(category_summaries),
                'bottlenecks': self._serialize_bottlenecks(bottlenecks),
                'trends': trends,
                'insights': insights,
                'scores': scores,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}")
            raise
    
    def _extract_key_metrics(self, results: List[Dict], targets: Dict) -> List[PerformanceMetric]:
        """Extract key performance metrics from results"""
        key_metrics = []
        
        # Define which metrics are considered "key"
        key_metric_names = [
            'Signal Generation P99 Latency',
            'Order Execution P95 Latency', 
            'Trading Throughput',
            'Signal Generation Throughput',
            'Sustained Memory Usage',
            'CPU Usage Under Load',
            'Strategy Sharpe Ratio',
            'Optimization Convergence Time'
        ]
        
        for result in results:
            test_name = result.get('test_name', '')
            if test_name in key_metric_names:
                target_info = result.get('target', {})
                
                metric = PerformanceMetric(
                    name=test_name,
                    value=result.get('measured_value', 0),
                    unit=target_info.get('unit', ''),
                    target=target_info.get('target_value', 0),
                    status=result.get('status', 'UNKNOWN'),
                    category=result.get('category', 'unknown'),
                    critical=target_info.get('critical', False)
                )
                
                key_metrics.append(metric)
        
        return key_metrics
    
    def _analyze_by_category(self, results: List[Dict]) -> Dict[str, CategorySummary]:
        """Analyze performance results by category"""
        categories = defaultdict(lambda: {
            'total': 0, 'passed': 0, 'failed': 0, 'warning': 0, 
            'critical_failures': [], 'metrics': []
        })
        
        # Group results by category
        for result in results:
            category = result.get('category', 'unknown')
            status = result.get('status', 'UNKNOWN')
            
            categories[category]['total'] += 1
            categories[category]['metrics'].append(result)
            
            if status == 'PASS':
                categories[category]['passed'] += 1
            elif status == 'FAIL':
                categories[category]['failed'] += 1
                if result.get('target', {}).get('critical', False):
                    categories[category]['critical_failures'].append(result.get('test_name', ''))
            elif status == 'WARNING':
                categories[category]['warning'] += 1
        
        # Create category summaries
        summaries = {}
        for category, data in categories.items():
            pass_rate = data['passed'] / data['total'] if data['total'] > 0 else 0
            
            # Calculate average performance (how close to targets)
            performance_scores = []
            key_metrics = []
            
            for metric in data['metrics']:
                measured = metric.get('measured_value')
                target_info = metric.get('target', {})
                target_value = target_info.get('target_value')
                comparison = target_info.get('comparison_operator', 'eq')
                
                if measured is not None and target_value is not None:
                    score = self._calculate_performance_score(measured, target_value, comparison)
                    performance_scores.append(score)
                    
                    # Create key metric
                    key_metric = PerformanceMetric(
                        name=metric.get('test_name', ''),
                        value=measured,
                        unit=target_info.get('unit', ''),
                        target=target_value,
                        status=metric.get('status', 'UNKNOWN'),
                        category=category,
                        critical=target_info.get('critical', False)
                    )
                    key_metrics.append(key_metric)
            
            average_performance = np.mean(performance_scores) if performance_scores else 0
            
            summaries[category] = CategorySummary(
                category=category,
                total_tests=data['total'],
                passed_tests=data['passed'],
                failed_tests=data['failed'],
                warning_tests=data['warning'],
                pass_rate=pass_rate,
                average_performance=average_performance,
                critical_failures=data['critical_failures'],
                key_metrics=key_metrics
            )
        
        return summaries
    
    def _identify_bottlenecks(self, results: List[Dict]) -> List[PerformanceBottleneck]:
        """Identify performance bottlenecks from results"""
        bottlenecks = []
        
        # Analyze latency bottlenecks
        latency_results = [r for r in results if 'latency' in r.get('category', '').lower()]
        if latency_results:
            failed_latency = [r for r in latency_results if r.get('status') == 'FAIL']
            if failed_latency:
                bottleneck = PerformanceBottleneck(
                    component='Latency Processing',
                    severity='HIGH' if len(failed_latency) > len(latency_results) / 2 else 'MEDIUM',
                    impact='Delayed signal generation and order execution',
                    description=f'{len(failed_latency)} out of {len(latency_results)} latency tests failed',
                    affected_metrics=[r.get('test_name', '') for r in failed_latency],
                    recommended_actions=[
                        'Optimize critical path algorithms',
                        'Implement asynchronous processing',
                        'Review database query performance',
                        'Consider caching frequently accessed data'
                    ]
                )
                bottlenecks.append(bottleneck)
        
        # Analyze throughput bottlenecks
        throughput_results = [r for r in results if 'throughput' in r.get('category', '').lower()]
        if throughput_results:
            failed_throughput = [r for r in throughput_results if r.get('status') == 'FAIL']
            if failed_throughput:
                bottleneck = PerformanceBottleneck(
                    component='Throughput Processing',
                    severity='HIGH' if len(failed_throughput) > len(throughput_results) / 2 else 'MEDIUM',
                    impact='Reduced system capacity and scalability',
                    description=f'{len(failed_throughput)} out of {len(throughput_results)} throughput tests failed',
                    affected_metrics=[r.get('test_name', '') for r in failed_throughput],
                    recommended_actions=[
                        'Scale horizontally with more workers',
                        'Implement batch processing',
                        'Optimize resource utilization',
                        'Review concurrent processing limits'
                    ]
                )
                bottlenecks.append(bottleneck)
        
        # Analyze resource bottlenecks
        resource_results = [r for r in results if 'resource' in r.get('category', '').lower()]
        if resource_results:
            failed_resources = [r for r in resource_results if r.get('status') == 'FAIL']
            if failed_resources:
                bottleneck = PerformanceBottleneck(
                    component='Resource Management',
                    severity='MEDIUM',
                    impact='Resource constraints limiting performance',
                    description=f'{len(failed_resources)} out of {len(resource_results)} resource tests failed',
                    affected_metrics=[r.get('test_name', '') for r in failed_resources],
                    recommended_actions=[
                        'Optimize memory usage patterns',
                        'Implement efficient garbage collection',
                        'Review CPU-intensive operations',
                        'Consider resource pooling'
                    ]
                )
                bottlenecks.append(bottleneck)
        
        # Analyze strategy bottlenecks
        strategy_results = [r for r in results if 'strategy' in r.get('category', '').lower()]
        if strategy_results:
            failed_strategies = [r for r in strategy_results if r.get('status') == 'FAIL']
            if failed_strategies:
                bottleneck = PerformanceBottleneck(
                    component='Trading Strategies',
                    severity='HIGH',
                    impact='Poor trading performance and profitability',
                    description=f'{len(failed_strategies)} out of {len(strategy_results)} strategy tests failed',
                    affected_metrics=[r.get('test_name', '') for r in failed_strategies],
                    recommended_actions=[
                        'Review strategy parameters',
                        'Enhance risk management',
                        'Add more diverse data sources',
                        'Implement ensemble methods'
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _analyze_performance_trends(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends from results"""
        trends = {
            'overall_trend': 'STABLE',
            'category_trends': {},
            'metric_trends': {},
            'trend_analysis': {}
        }
        
        # For now, simulate trend analysis
        # In a real implementation, this would compare with historical data
        
        categories = set(r.get('category', 'unknown') for r in results)
        
        for category in categories:
            category_results = [r for r in results if r.get('category') == category]
            pass_rate = len([r for r in category_results if r.get('status') == 'PASS']) / len(category_results)
            
            if pass_rate >= 0.9:
                trends['category_trends'][category] = 'EXCELLENT'
            elif pass_rate >= 0.7:
                trends['category_trends'][category] = 'GOOD'
            elif pass_rate >= 0.5:
                trends['category_trends'][category] = 'FAIR'
            else:
                trends['category_trends'][category] = 'POOR'
        
        # Overall trend based on category trends
        category_trend_values = list(trends['category_trends'].values())
        if all(t in ['EXCELLENT', 'GOOD'] for t in category_trend_values):
            trends['overall_trend'] = 'IMPROVING'
        elif any(t == 'POOR' for t in category_trend_values):
            trends['overall_trend'] = 'DEGRADING'
        else:
            trends['overall_trend'] = 'STABLE'
        
        # Detailed trend analysis
        trends['trend_analysis'] = {
            'performance_stability': 'HIGH' if trends['overall_trend'] != 'DEGRADING' else 'LOW',
            'improvement_areas': [
                cat for cat, trend in trends['category_trends'].items() 
                if trend in ['FAIR', 'POOR']
            ],
            'strong_areas': [
                cat for cat, trend in trends['category_trends'].items() 
                if trend in ['EXCELLENT', 'GOOD']
            ]
        }
        
        return trends
    
    def _generate_performance_insights(self, key_metrics: List[PerformanceMetric],
                                     category_summaries: Dict[str, CategorySummary],
                                     bottlenecks: List[PerformanceBottleneck],
                                     trends: Dict[str, Any]) -> List[str]:
        """Generate performance insights and observations"""
        insights = []
        
        # Overall performance insight
        total_tests = sum(cs.total_tests for cs in category_summaries.values())
        total_passed = sum(cs.passed_tests for cs in category_summaries.values())
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
        
        if overall_pass_rate >= 0.9:
            insights.append(f"Excellent overall performance with {overall_pass_rate:.1%} pass rate across all tests.")
        elif overall_pass_rate >= 0.7:
            insights.append(f"Good overall performance with {overall_pass_rate:.1%} pass rate, with room for optimization.")
        else:
            insights.append(f"Performance needs improvement with only {overall_pass_rate:.1%} pass rate across tests.")
        
        # Critical failures insight
        critical_failures = []
        for cs in category_summaries.values():
            critical_failures.extend(cs.critical_failures)
        
        if critical_failures:
            insights.append(f"Critical attention needed: {len(critical_failures)} critical test failures detected.")
        else:
            insights.append("No critical failures detected - system meets minimum production requirements.")
        
        # Category-specific insights
        best_category = max(category_summaries.items(), key=lambda x: x[1].pass_rate)
        worst_category = min(category_summaries.items(), key=lambda x: x[1].pass_rate)
        
        insights.append(f"Best performing category: {best_category[0]} ({best_category[1].pass_rate:.1%} pass rate)")
        
        if worst_category[1].pass_rate < 0.7:
            insights.append(f"Category needing attention: {worst_category[0]} ({worst_category[1].pass_rate:.1%} pass rate)")
        
        # Bottleneck insights
        high_severity_bottlenecks = [b for b in bottlenecks if b.severity == 'HIGH']
        if high_severity_bottlenecks:
            insights.append(f"High priority: {len(high_severity_bottlenecks)} high-severity bottlenecks identified.")
        
        # Key metrics insights
        failed_key_metrics = [m for m in key_metrics if m.status == 'FAIL']
        if failed_key_metrics:
            insights.append(f"Key metrics concern: {len(failed_key_metrics)} critical performance metrics below targets.")
        
        # Trend insights
        if trends['overall_trend'] == 'IMPROVING':
            insights.append("Positive trend: Performance is improving across most categories.")
        elif trends['overall_trend'] == 'DEGRADING':
            insights.append("Warning: Performance trend is degrading and requires immediate attention.")
        
        return insights
    
    def _calculate_performance_scores(self, results: List[Dict], 
                                    category_summaries: Dict[str, CategorySummary]) -> Dict[str, float]:
        """Calculate various performance scores"""
        scores = {}
        
        # Overall performance score (0-100)
        total_tests = sum(cs.total_tests for cs in category_summaries.values())
        total_passed = sum(cs.passed_tests for cs in category_summaries.values())
        scores['overall_performance_score'] = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Category scores
        for category, summary in category_summaries.items():
            scores[f'{category}_score'] = summary.pass_rate * 100
        
        # Reliability score (inverse of critical failures)
        critical_failure_count = sum(len(cs.critical_failures) for cs in category_summaries.values())
        critical_test_count = sum(1 for r in results if r.get('target', {}).get('critical', False))
        scores['reliability_score'] = (
            ((critical_test_count - critical_failure_count) / critical_test_count * 100) 
            if critical_test_count > 0 else 100
        )
        
        # Efficiency score (based on how close to targets)
        efficiency_scores = []
        for result in results:
            measured = result.get('measured_value')
            target_info = result.get('target', {})
            target_value = target_info.get('target_value')
            comparison = target_info.get('comparison_operator', 'eq')
            
            if measured is not None and target_value is not None:
                score = self._calculate_performance_score(measured, target_value, comparison)
                efficiency_scores.append(score)
        
        scores['efficiency_score'] = np.mean(efficiency_scores) * 100 if efficiency_scores else 0
        
        return scores
    
    def _calculate_performance_score(self, measured: float, target: float, comparison: str) -> float:
        """Calculate performance score for a metric (0-1 scale)"""
        if comparison == 'lt':  # measured should be less than target
            if measured <= target:
                return 1.0
            else:
                # Gradual degradation beyond target
                return max(0, 1 - (measured - target) / target)
        elif comparison == 'gt':  # measured should be greater than target
            if measured >= target:
                return 1.0
            else:
                # Gradual degradation below target
                return max(0, measured / target)
        elif comparison == 'eq':  # measured should equal target
            difference = abs(measured - target)
            tolerance = target * 0.05  # 5% tolerance
            if difference <= tolerance:
                return 1.0
            else:
                return max(0, 1 - difference / target)
        else:
            return 0.5  # Unknown comparison
    
    def _serialize_metrics(self, metrics: List[PerformanceMetric]) -> List[Dict]:
        """Serialize performance metrics to dictionary format"""
        return [
            {
                'name': m.name,
                'value': m.value,
                'unit': m.unit,
                'target': m.target,
                'status': m.status,
                'category': m.category,
                'critical': m.critical,
                'trend': m.trend
            }
            for m in metrics
        ]
    
    def _serialize_category_summaries(self, summaries: Dict[str, CategorySummary]) -> Dict[str, Dict]:
        """Serialize category summaries to dictionary format"""
        return {
            category: {
                'category': summary.category,
                'total_tests': summary.total_tests,
                'passed_tests': summary.passed_tests,
                'failed_tests': summary.failed_tests,
                'warning_tests': summary.warning_tests,
                'pass_rate': summary.pass_rate,
                'average_performance': summary.average_performance,
                'critical_failures': summary.critical_failures,
                'key_metrics': self._serialize_metrics(summary.key_metrics)
            }
            for category, summary in summaries.items()
        }
    
    def _serialize_bottlenecks(self, bottlenecks: List[PerformanceBottleneck]) -> List[Dict]:
        """Serialize bottlenecks to dictionary format"""
        return [
            {
                'component': b.component,
                'severity': b.severity,
                'impact': b.impact,
                'description': b.description,
                'affected_metrics': b.affected_metrics,
                'recommended_actions': b.recommended_actions
            }
            for b in bottlenecks
        ]


def main():
    """Main entry point for testing performance summary generation"""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Summary Generator")
    parser.add_argument('--input', required=True, help='Input validation results JSON file')
    parser.add_argument('--output', help='Output summary JSON file')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load validation data
    with open(args.input, 'r') as f:
        validation_data = json.load(f)
    
    # Generate summary
    generator = PerformanceSummaryGenerator()
    summary = generator.generate_summary(validation_data)
    
    # Save or print summary
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved to {args.output}")
    else:
        print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()