#!/usr/bin/env python3
"""
Benchmark Comparison Analyzer for AI News Trading Platform.

This module compares current performance validation results with historical
benchmarks and industry standards to provide context and trend analysis.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import statistics


class TrendDirection(Enum):
    """Trend direction indicators"""
    IMPROVING = "IMPROVING"
    DEGRADING = "DEGRADING"
    STABLE = "STABLE"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_name: str
    current_value: float
    historical_average: float
    trend_direction: TrendDirection
    percentage_change: float
    volatility: float
    confidence_level: float
    significance: str  # HIGH, MEDIUM, LOW


@dataclass
class BenchmarkComparison:
    """Benchmark comparison result"""
    metric_name: str
    current_value: float
    benchmark_value: float
    benchmark_type: str  # INTERNAL, INDUSTRY, TARGET
    comparison_result: str  # BETTER, WORSE, SIMILAR
    percentage_difference: float
    ranking_percentile: Optional[float] = None
    context: str = ""


@dataclass
class HistoricalAnalysis:
    """Historical performance analysis"""
    timeframe: str
    data_points: int
    trends: List[PerformanceTrend]
    comparisons: List[BenchmarkComparison]
    summary: Dict[str, Any]
    insights: List[str] = field(default_factory=list)


class BenchmarkComparisonAnalyzer:
    """Analyzes performance trends and comparisons with benchmarks"""
    
    def __init__(self):
        """Initialize benchmark comparison analyzer"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Industry benchmarks for trading systems
        self.industry_benchmarks = {
            'signal_generation_latency_p99': 150.0,  # ms
            'order_execution_latency_p95': 100.0,    # ms
            'data_processing_latency_p95': 50.0,     # ms
            'trading_throughput': 500.0,             # trades/sec
            'signal_throughput': 5000.0,             # signals/sec
            'memory_usage_sustained': 4096.0,        # MB
            'cpu_usage_under_load': 70.0,            # %
            'strategy_sharpe_ratio': 1.5,            # ratio
            'system_uptime': 99.5,                   # %
            'error_rate': 0.5                        # %
        }
        
        # Performance categories for analysis
        self.performance_categories = {
            'latency': ['signal_generation_latency', 'order_execution_latency', 'data_processing_latency'],
            'throughput': ['trading_throughput', 'signal_throughput', 'data_processing_throughput'],
            'resource': ['memory_usage', 'cpu_usage', 'disk_io'],
            'strategy': ['sharpe_ratio', 'returns', 'volatility', 'drawdown'],
            'reliability': ['uptime', 'error_rate', 'availability']
        }
    
    def compare_with_historical(self, current_data: Dict[str, Any], 
                              historical_data: List[Dict[str, Any]]) -> HistoricalAnalysis:
        """Compare current performance with historical data
        
        Args:
            current_data: Current validation results
            historical_data: List of historical validation results
            
        Returns:
            HistoricalAnalysis with trends and comparisons
        """
        self.logger.info("Analyzing performance trends and historical comparisons...")
        
        try:
            # Prepare data for analysis
            current_metrics = self._extract_metrics(current_data)
            historical_metrics = self._extract_historical_metrics(historical_data)
            
            # Analyze trends
            trends = self._analyze_trends(current_metrics, historical_metrics)
            
            # Compare with benchmarks
            comparisons = self._compare_with_benchmarks(current_metrics, historical_metrics)
            
            # Generate summary
            summary = self._generate_historical_summary(trends, comparisons, len(historical_data))
            
            # Generate insights
            insights = self._generate_historical_insights(trends, comparisons)
            
            # Determine timeframe
            timeframe = self._determine_timeframe(historical_data)
            
            return HistoricalAnalysis(
                timeframe=timeframe,
                data_points=len(historical_data) + 1,  # +1 for current
                trends=trends,
                comparisons=comparisons,
                summary=summary,
                insights=insights
            )
            
        except Exception as e:
            self.logger.error(f"Historical comparison analysis failed: {e}")
            raise
    
    def compare_with_industry_standards(self, current_data: Dict[str, Any]) -> List[BenchmarkComparison]:
        """Compare current performance with industry standards
        
        Args:
            current_data: Current validation results
            
        Returns:
            List of benchmark comparisons
        """
        self.logger.info("Comparing with industry standards...")
        
        current_metrics = self._extract_metrics(current_data)
        comparisons = []
        
        for metric_name, current_value in current_metrics.items():
            # Find matching industry benchmark
            benchmark_key = self._find_benchmark_key(metric_name)
            if benchmark_key and benchmark_key in self.industry_benchmarks:
                benchmark_value = self.industry_benchmarks[benchmark_key]
                
                comparison = self._create_benchmark_comparison(
                    metric_name, current_value, benchmark_value, "INDUSTRY"
                )
                comparisons.append(comparison)
        
        return comparisons
    
    def _extract_metrics(self, validation_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from validation data"""
        metrics = {}
        
        results = validation_data.get('results', [])
        for result in results:
            test_name = result.get('test_name', '')
            measured_value = result.get('measured_value')
            
            if measured_value is not None:
                # Normalize metric name
                metric_key = self._normalize_metric_name(test_name)
                metrics[metric_key] = float(measured_value)
        
        return metrics
    
    def _extract_historical_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract metrics from historical data"""
        historical_metrics = {}
        
        for data_point in historical_data:
            metrics = self._extract_metrics(data_point)
            
            for metric_name, value in metrics.items():
                if metric_name not in historical_metrics:
                    historical_metrics[metric_name] = []
                historical_metrics[metric_name].append(value)
        
        return historical_metrics
    
    def _analyze_trends(self, current_metrics: Dict[str, float], 
                       historical_metrics: Dict[str, List[float]]) -> List[PerformanceTrend]:
        """Analyze performance trends"""
        trends = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in historical_metrics and len(historical_metrics[metric_name]) >= 2:
                historical_values = historical_metrics[metric_name]
                
                # Calculate trend statistics
                historical_average = np.mean(historical_values)
                percentage_change = ((current_value - historical_average) / historical_average * 100) if historical_average != 0 else 0
                volatility = np.std(historical_values) / historical_average if historical_average != 0 else 0
                
                # Determine trend direction
                trend_direction = self._determine_trend_direction(
                    current_value, historical_values, metric_name
                )
                
                # Calculate confidence and significance
                confidence_level = self._calculate_confidence_level(historical_values, current_value)
                significance = self._determine_significance(abs(percentage_change), volatility)
                
                trend = PerformanceTrend(
                    metric_name=metric_name,
                    current_value=current_value,
                    historical_average=historical_average,
                    trend_direction=trend_direction,
                    percentage_change=percentage_change,
                    volatility=volatility,
                    confidence_level=confidence_level,
                    significance=significance
                )
                
                trends.append(trend)
        
        return trends
    
    def _compare_with_benchmarks(self, current_metrics: Dict[str, float], 
                               historical_metrics: Dict[str, List[float]]) -> List[BenchmarkComparison]:
        """Compare with internal and industry benchmarks"""
        comparisons = []
        
        # Internal benchmarks (historical averages)
        for metric_name, current_value in current_metrics.items():
            if metric_name in historical_metrics and len(historical_metrics[metric_name]) >= 3:
                internal_benchmark = np.mean(historical_metrics[metric_name])
                
                comparison = self._create_benchmark_comparison(
                    metric_name, current_value, internal_benchmark, "INTERNAL"
                )
                comparisons.append(comparison)
        
        # Industry benchmarks
        industry_comparisons = self.compare_with_industry_standards({'results': [
            {'test_name': name, 'measured_value': value} 
            for name, value in current_metrics.items()
        ]})
        comparisons.extend(industry_comparisons)
        
        return comparisons
    
    def _create_benchmark_comparison(self, metric_name: str, current_value: float, 
                                   benchmark_value: float, benchmark_type: str) -> BenchmarkComparison:
        """Create a benchmark comparison"""
        percentage_difference = ((current_value - benchmark_value) / benchmark_value * 100) if benchmark_value != 0 else 0
        
        # Determine if better or worse (depends on metric type)
        comparison_result = self._determine_comparison_result(
            metric_name, current_value, benchmark_value
        )
        
        # Generate context
        context = self._generate_comparison_context(
            metric_name, comparison_result, abs(percentage_difference)
        )
        
        return BenchmarkComparison(
            metric_name=metric_name,
            current_value=current_value,
            benchmark_value=benchmark_value,
            benchmark_type=benchmark_type,
            comparison_result=comparison_result,
            percentage_difference=percentage_difference,
            context=context
        )
    
    def _determine_trend_direction(self, current_value: float, 
                                 historical_values: List[float], 
                                 metric_name: str) -> TrendDirection:
        """Determine trend direction based on recent data"""
        if len(historical_values) < 2:
            return TrendDirection.UNKNOWN
        
        # Use recent trend (last 3-5 data points)
        recent_values = historical_values[-min(5, len(historical_values)):]
        recent_values.append(current_value)
        
        # Calculate trend using linear regression slope
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        if len(x) >= 2:
            slope = np.polyfit(x, y, 1)[0]
            
            # Determine if improvement or degradation based on metric type
            is_lower_better = self._is_lower_better_metric(metric_name)
            
            threshold = np.std(historical_values) * 0.1  # 10% of standard deviation
            
            if abs(slope) < threshold:
                return TrendDirection.STABLE
            elif slope > 0:
                return TrendDirection.DEGRADING if is_lower_better else TrendDirection.IMPROVING
            else:
                return TrendDirection.IMPROVING if is_lower_better else TrendDirection.DEGRADING
        
        return TrendDirection.UNKNOWN
    
    def _determine_comparison_result(self, metric_name: str, current_value: float, 
                                   benchmark_value: float) -> str:
        """Determine if current value is better, worse, or similar to benchmark"""
        if abs(current_value - benchmark_value) / benchmark_value < 0.05:  # Within 5%
            return "SIMILAR"
        
        is_lower_better = self._is_lower_better_metric(metric_name)
        
        if current_value < benchmark_value:
            return "BETTER" if is_lower_better else "WORSE"
        else:
            return "WORSE" if is_lower_better else "BETTER"
    
    def _is_lower_better_metric(self, metric_name: str) -> bool:
        """Determine if lower values are better for this metric"""
        lower_better_indicators = [
            'latency', 'memory', 'cpu', 'error', 'drawdown', 'volatility'
        ]
        
        metric_lower = metric_name.lower()
        return any(indicator in metric_lower for indicator in lower_better_indicators)
    
    def _calculate_confidence_level(self, historical_values: List[float], current_value: float) -> float:
        """Calculate confidence level in trend analysis"""
        if len(historical_values) < 3:
            return 0.3  # Low confidence with limited data
        
        # Calculate z-score to determine how unusual current value is
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return 0.5  # Medium confidence if no variation
        
        z_score = abs(current_value - mean) / std
        
        # Convert z-score to confidence (inverse relationship)
        if z_score < 1:
            return 0.9  # High confidence - value is typical
        elif z_score < 2:
            return 0.7  # Medium-high confidence
        elif z_score < 3:
            return 0.5  # Medium confidence
        else:
            return 0.3  # Low confidence - value is unusual
    
    def _determine_significance(self, percentage_change: float, volatility: float) -> str:
        """Determine significance of change"""
        if volatility > 0.5:  # High volatility
            threshold_high = 30
            threshold_medium = 15
        elif volatility > 0.2:  # Medium volatility
            threshold_high = 20
            threshold_medium = 10
        else:  # Low volatility
            threshold_high = 10
            threshold_medium = 5
        
        if percentage_change >= threshold_high:
            return "HIGH"
        elif percentage_change >= threshold_medium:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_historical_summary(self, trends: List[PerformanceTrend], 
                                   comparisons: List[BenchmarkComparison], 
                                   historical_count: int) -> Dict[str, Any]:
        """Generate summary of historical analysis"""
        summary = {
            'historical_data_points': historical_count,
            'metrics_analyzed': len(trends),
            'trend_analysis': {},
            'benchmark_analysis': {},
            'overall_assessment': {}
        }
        
        # Trend analysis summary
        trend_directions = [t.trend_direction.value for t in trends]
        summary['trend_analysis'] = {
            'improving_metrics': trend_directions.count('IMPROVING'),
            'degrading_metrics': trend_directions.count('DEGRADING'),
            'stable_metrics': trend_directions.count('STABLE'),
            'volatile_metrics': trend_directions.count('VOLATILE'),
            'dominant_trend': max(set(trend_directions), key=trend_directions.count) if trend_directions else 'UNKNOWN'
        }
        
        # Benchmark analysis summary
        benchmark_results = [c.comparison_result for c in comparisons]
        summary['benchmark_analysis'] = {
            'better_than_benchmark': benchmark_results.count('BETTER'),
            'worse_than_benchmark': benchmark_results.count('WORSE'),
            'similar_to_benchmark': benchmark_results.count('SIMILAR'),
            'total_comparisons': len(comparisons)
        }
        
        # Overall assessment
        improving_ratio = summary['trend_analysis']['improving_metrics'] / max(len(trends), 1)
        better_ratio = summary['benchmark_analysis']['better_than_benchmark'] / max(len(comparisons), 1)
        
        if improving_ratio >= 0.6 and better_ratio >= 0.5:
            overall_status = "EXCELLENT"
        elif improving_ratio >= 0.4 and better_ratio >= 0.3:
            overall_status = "GOOD"
        elif improving_ratio >= 0.2 or better_ratio >= 0.2:
            overall_status = "FAIR"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        summary['overall_assessment'] = {
            'status': overall_status,
            'improving_ratio': improving_ratio,
            'better_than_benchmark_ratio': better_ratio,
            'confidence_score': np.mean([t.confidence_level for t in trends]) if trends else 0
        }
        
        return summary
    
    def _generate_historical_insights(self, trends: List[PerformanceTrend], 
                                    comparisons: List[BenchmarkComparison]) -> List[str]:
        """Generate insights from historical analysis"""
        insights = []
        
        # Trend insights
        improving_trends = [t for t in trends if t.trend_direction == TrendDirection.IMPROVING]
        degrading_trends = [t for t in trends if t.trend_direction == TrendDirection.DEGRADING]
        
        if improving_trends:
            top_improvement = max(improving_trends, key=lambda x: abs(x.percentage_change))
            insights.append(
                f"Best improvement: {top_improvement.metric_name} improved by "
                f"{abs(top_improvement.percentage_change):.1f}% compared to historical average"
            )
        
        if degrading_trends:
            worst_degradation = max(degrading_trends, key=lambda x: abs(x.percentage_change))
            insights.append(
                f"Concerning trend: {worst_degradation.metric_name} degraded by "
                f"{abs(worst_degradation.percentage_change):.1f}% compared to historical average"
            )
        
        # Benchmark insights
        better_comparisons = [c for c in comparisons if c.comparison_result == "BETTER"]
        worse_comparisons = [c for c in comparisons if c.comparison_result == "WORSE"]
        
        if better_comparisons:
            best_benchmark = max(better_comparisons, key=lambda x: abs(x.percentage_difference))
            insights.append(
                f"Strong performance: {best_benchmark.metric_name} is "
                f"{abs(best_benchmark.percentage_difference):.1f}% better than {best_benchmark.benchmark_type.lower()} benchmark"
            )
        
        if worse_comparisons:
            worst_benchmark = max(worse_comparisons, key=lambda x: abs(x.percentage_difference))
            insights.append(
                f"Underperforming: {worst_benchmark.metric_name} is "
                f"{abs(worst_benchmark.percentage_difference):.1f}% worse than {worst_benchmark.benchmark_type.lower()} benchmark"
            )
        
        # Volatility insights
        high_volatility = [t for t in trends if t.volatility > 0.3]
        if high_volatility:
            insights.append(
                f"High volatility detected in {len(high_volatility)} metrics, indicating potential instability"
            )
        
        # Confidence insights
        low_confidence = [t for t in trends if t.confidence_level < 0.5]
        if low_confidence:
            insights.append(
                f"Low confidence in trend analysis for {len(low_confidence)} metrics due to limited or inconsistent data"
            )
        
        return insights
    
    def _determine_timeframe(self, historical_data: List[Dict[str, Any]]) -> str:
        """Determine timeframe of historical data"""
        if not historical_data:
            return "No historical data"
        
        # Try to extract timestamps
        timestamps = []
        for data in historical_data:
            timestamp_str = data.get('summary', {}).get('validation_timestamp')
            if timestamp_str:
                try:
                    timestamps.append(datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')))
                except:
                    pass
        
        if len(timestamps) >= 2:
            timespan = max(timestamps) - min(timestamps)
            days = timespan.days
            
            if days < 7:
                return f"{days} days"
            elif days < 30:
                weeks = days // 7
                return f"{weeks} weeks"
            elif days < 365:
                months = days // 30
                return f"{months} months"
            else:
                years = days // 365
                return f"{years} years"
        
        return f"{len(historical_data)} data points"
    
    def _normalize_metric_name(self, test_name: str) -> str:
        """Normalize metric name for consistent comparison"""
        return test_name.lower().replace(' ', '_').replace('-', '_')
    
    def _find_benchmark_key(self, metric_name: str) -> Optional[str]:
        """Find matching benchmark key for metric"""
        normalized_name = metric_name.lower()
        
        for benchmark_key in self.industry_benchmarks.keys():
            # Check for partial matches
            key_parts = benchmark_key.split('_')
            if any(part in normalized_name for part in key_parts if len(part) > 3):
                return benchmark_key
        
        return None
    
    def _generate_comparison_context(self, metric_name: str, comparison_result: str, 
                                   percentage_difference: float) -> str:
        """Generate context for benchmark comparison"""
        if comparison_result == "SIMILAR":
            return f"Performance is within acceptable range of benchmark"
        elif comparison_result == "BETTER":
            if percentage_difference > 20:
                return f"Significantly outperforming benchmark by {percentage_difference:.1f}%"
            else:
                return f"Moderately outperforming benchmark by {percentage_difference:.1f}%"
        else:  # WORSE
            if percentage_difference > 20:
                return f"Significantly underperforming benchmark by {percentage_difference:.1f}%"
            else:
                return f"Moderately underperforming benchmark by {percentage_difference:.1f}%"


def main():
    """Main entry point for testing benchmark comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Comparison Analyzer")
    parser.add_argument('--current', required=True, help='Current validation results JSON file')
    parser.add_argument('--historical', help='Historical validation results JSON file (array)')
    parser.add_argument('--output', help='Output comparison JSON file')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load current data
    with open(args.current, 'r') as f:
        current_data = json.load(f)
    
    # Load historical data
    historical_data = []
    if args.historical:
        with open(args.historical, 'r') as f:
            historical_data = json.load(f)
    
    # Analyze
    analyzer = BenchmarkComparisonAnalyzer()
    
    if historical_data:
        analysis = analyzer.compare_with_historical(current_data, historical_data)
        result = {
            'historical_analysis': {
                'timeframe': analysis.timeframe,
                'data_points': analysis.data_points,
                'trends': [
                    {
                        'metric_name': t.metric_name,
                        'current_value': t.current_value,
                        'historical_average': t.historical_average,
                        'trend_direction': t.trend_direction.value,
                        'percentage_change': t.percentage_change,
                        'volatility': t.volatility,
                        'confidence_level': t.confidence_level,
                        'significance': t.significance
                    }
                    for t in analysis.trends
                ],
                'comparisons': [
                    {
                        'metric_name': c.metric_name,
                        'current_value': c.current_value,
                        'benchmark_value': c.benchmark_value,
                        'benchmark_type': c.benchmark_type,
                        'comparison_result': c.comparison_result,
                        'percentage_difference': c.percentage_difference,
                        'context': c.context
                    }
                    for c in analysis.comparisons
                ],
                'summary': analysis.summary,
                'insights': analysis.insights
            }
        }
    else:
        industry_comparisons = analyzer.compare_with_industry_standards(current_data)
        result = {
            'industry_comparison': [
                {
                    'metric_name': c.metric_name,
                    'current_value': c.current_value,
                    'benchmark_value': c.benchmark_value,
                    'benchmark_type': c.benchmark_type,
                    'comparison_result': c.comparison_result,
                    'percentage_difference': c.percentage_difference,
                    'context': c.context
                }
                for c in industry_comparisons
            ]
        }
    
    # Save or print result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Comparison analysis saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()