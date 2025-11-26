#!/usr/bin/env python3
"""
Performance Reports Package for AI News Trading Platform.

This package provides comprehensive reporting capabilities for performance
validation results including:
- Report generation in multiple formats
- Performance summaries and analytics
- Optimization recommendations
- Historical benchmark comparisons

Modules:
    generate_report: Main report generator with HTML, JSON, PDF output
    performance_summary: Performance analysis and summary generation
    optimization_recommendations: Actionable optimization recommendations
    benchmark_comparison: Historical and industry benchmark comparisons
"""

from .generate_report import ReportGenerator, ReportConfiguration
from .performance_summary import PerformanceSummaryGenerator, PerformanceMetric, CategorySummary
from .optimization_recommendations import OptimizationRecommendationEngine, OptimizationRecommendation, Priority, ImplementationComplexity
from .benchmark_comparison import BenchmarkComparisonAnalyzer, PerformanceTrend, BenchmarkComparison, TrendDirection

__all__ = [
    'ReportGenerator',
    'ReportConfiguration',
    'PerformanceSummaryGenerator',
    'PerformanceMetric',
    'CategorySummary',
    'OptimizationRecommendationEngine',
    'OptimizationRecommendation',
    'Priority',
    'ImplementationComplexity',
    'BenchmarkComparisonAnalyzer',
    'PerformanceTrend',
    'BenchmarkComparison',
    'TrendDirection'
]

__version__ = "1.0.0"