#!/usr/bin/env python3
"""
Main Performance Validation Orchestrator for AI News Trading Platform.

This module coordinates all performance validation tests and ensures the platform
meets production-ready performance targets across all components.

Performance Targets:
- Signal Generation: < 100ms (P99)
- Order Execution: < 50ms (P95) 
- Data Processing: < 25ms per tick
- Throughput: > 1000 trades/second
- Memory Usage: < 2GB sustained
- CPU Usage: < 80% under load
- Strategy Performance: Sharpe > 2.0
- Optimization: Convergence in < 30 minutes
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import numpy as np

from .latency_validator import LatencyValidator
from .throughput_validator import ThroughputValidator
from .resource_validator import ResourceValidator
from .strategy_validator import StrategyValidator

class ValidationStatus(Enum):
    """Validation test status"""
    PASS = "PASS"
    FAIL = "FAIL" 
    WARNING = "WARNING"
    SKIP = "SKIP"
    ERROR = "ERROR"

@dataclass
class PerformanceTarget:
    """Performance target specification"""
    name: str
    description: str
    target_value: Union[int, float]
    unit: str
    comparison_operator: str  # 'lt', 'gt', 'lte', 'gte', 'eq'
    critical: bool = True
    category: str = "performance"
    
@dataclass 
class ValidationResult:
    """Result of a single validation test"""
    test_name: str
    category: str
    target: PerformanceTarget
    measured_value: Union[int, float, None]
    status: ValidationStatus
    message: str
    duration_seconds: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_name': self.test_name,
            'category': self.category,
            'target': {
                'name': self.target.name,
                'description': self.target.description,
                'target_value': self.target.target_value,
                'unit': self.target.unit,
                'comparison_operator': self.target.comparison_operator,
                'critical': self.target.critical
            },
            'measured_value': self.measured_value,
            'status': self.status.value,
            'message': self.message,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'recommendations': self.recommendations
        }

@dataclass
class ValidationSummary:
    """Summary of all validation results"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    error_tests: int
    skipped_tests: int
    critical_failures: List[str]
    overall_status: ValidationStatus
    total_duration_seconds: float
    validation_timestamp: datetime
    platform_version: str = "1.0.0"

class PerformanceValidator:
    """Main performance validation orchestrator"""
    
    def __init__(self, config: Optional[Dict] = None, output_dir: str = None):
        """Initialize the performance validator
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for validation outputs
        """
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Initialize component validators
        self.latency_validator = LatencyValidator(config)
        self.throughput_validator = ThroughputValidator(config)
        self.resource_validator = ResourceValidator(config)
        self.strategy_validator = StrategyValidator(config)
        
        # Performance targets
        self.targets = self._define_performance_targets()
        
        # Results storage
        self.results: List[ValidationResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_file = self.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Configure logger
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _define_performance_targets(self) -> Dict[str, PerformanceTarget]:
        """Define all performance targets"""
        return {
            # Latency Targets
            'signal_generation_p99': PerformanceTarget(
                name='Signal Generation P99 Latency',
                description='99th percentile latency for signal generation must be under 100ms',
                target_value=100.0,
                unit='ms',
                comparison_operator='lt',
                critical=True,
                category='latency'
            ),
            'order_execution_p95': PerformanceTarget(
                name='Order Execution P95 Latency',
                description='95th percentile latency for order execution must be under 50ms',
                target_value=50.0,
                unit='ms',
                comparison_operator='lt',
                critical=True,
                category='latency'
            ),
            'data_processing_tick': PerformanceTarget(
                name='Data Processing Per Tick',
                description='Data processing latency per tick must be under 25ms',
                target_value=25.0,
                unit='ms',
                comparison_operator='lt',
                critical=True,
                category='latency'
            ),
            
            # Throughput Targets
            'trades_per_second': PerformanceTarget(
                name='Trading Throughput',
                description='System must handle over 1000 trades per second',
                target_value=1000.0,
                unit='trades/sec',
                comparison_operator='gt',
                critical=True,
                category='throughput'
            ),
            'signals_per_second': PerformanceTarget(
                name='Signal Generation Throughput',
                description='System must generate over 10000 signals per second',
                target_value=10000.0,
                unit='signals/sec',
                comparison_operator='gt',
                critical=True,
                category='throughput'
            ),
            
            # Resource Targets
            'memory_sustained': PerformanceTarget(
                name='Sustained Memory Usage',
                description='Sustained memory usage must be under 2GB',
                target_value=2048.0,
                unit='MB',
                comparison_operator='lt',
                critical=True,
                category='resource'
            ),
            'cpu_under_load': PerformanceTarget(
                name='CPU Usage Under Load',
                description='CPU usage under load must be under 80%',
                target_value=80.0,
                unit='%',
                comparison_operator='lt',
                critical=False,
                category='resource'
            ),
            
            # Strategy Performance Targets
            'strategy_sharpe_ratio': PerformanceTarget(
                name='Strategy Sharpe Ratio',
                description='Strategy Sharpe ratio must be over 2.0',
                target_value=2.0,
                unit='ratio',
                comparison_operator='gt',
                critical=True,
                category='strategy'
            ),
            'optimization_convergence': PerformanceTarget(
                name='Optimization Convergence Time',
                description='Parameter optimization must converge in under 30 minutes',
                target_value=30.0,
                unit='minutes',
                comparison_operator='lt',
                critical=False,
                category='optimization'
            )
        }
    
    async def validate_all(self, 
                          quick_mode: bool = False,
                          categories: Optional[List[str]] = None) -> ValidationSummary:
        """Run complete performance validation
        
        Args:
            quick_mode: Run only critical tests for faster validation
            categories: List of categories to test (latency, throughput, resource, strategy)
            
        Returns:
            ValidationSummary with results
        """
        self.logger.info("Starting comprehensive performance validation")
        self.start_time = datetime.now()
        self.results.clear()
        
        try:
            # Determine which tests to run
            test_categories = categories or ['latency', 'throughput', 'resource', 'strategy']
            
            # Run validation tests by category
            if 'latency' in test_categories:
                await self._run_latency_validation(quick_mode)
                
            if 'throughput' in test_categories:
                await self._run_throughput_validation(quick_mode)
                
            if 'resource' in test_categories:
                await self._run_resource_validation(quick_mode)
                
            if 'strategy' in test_categories:
                await self._run_strategy_validation(quick_mode)
            
            # Generate summary
            summary = self._generate_summary()
            
            # Save results
            await self._save_results(summary)
            
            self.logger.info(f"Performance validation completed. Overall status: {summary.overall_status.value}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {str(e)}")
            raise
        finally:
            self.end_time = datetime.now()
    
    async def _run_latency_validation(self, quick_mode: bool = False):
        """Run latency validation tests"""
        self.logger.info("Running latency validation tests...")
        
        try:
            # Signal generation latency
            result = await self.latency_validator.validate_signal_generation_latency()
            self._add_result('signal_generation_p99', result)
            
            if not quick_mode:
                # Order execution latency
                result = await self.latency_validator.validate_order_execution_latency()
                self._add_result('order_execution_p95', result)
                
                # Data processing latency
                result = await self.latency_validator.validate_data_processing_latency()
                self._add_result('data_processing_tick', result)
                
        except Exception as e:
            self.logger.error(f"Latency validation failed: {e}")
            self._add_error_result('latency_validation', str(e))
    
    async def _run_throughput_validation(self, quick_mode: bool = False):
        """Run throughput validation tests"""
        self.logger.info("Running throughput validation tests...")
        
        try:
            # Trading throughput
            result = await self.throughput_validator.validate_trading_throughput()
            self._add_result('trades_per_second', result)
            
            if not quick_mode:
                # Signal generation throughput
                result = await self.throughput_validator.validate_signal_throughput()
                self._add_result('signals_per_second', result)
                
        except Exception as e:
            self.logger.error(f"Throughput validation failed: {e}")
            self._add_error_result('throughput_validation', str(e))
    
    async def _run_resource_validation(self, quick_mode: bool = False):
        """Run resource usage validation tests"""
        self.logger.info("Running resource validation tests...")
        
        try:
            # Memory usage validation
            result = await self.resource_validator.validate_memory_usage()
            self._add_result('memory_sustained', result)
            
            if not quick_mode:
                # CPU usage validation
                result = await self.resource_validator.validate_cpu_usage()
                self._add_result('cpu_under_load', result)
                
        except Exception as e:
            self.logger.error(f"Resource validation failed: {e}")
            self._add_error_result('resource_validation', str(e))
    
    async def _run_strategy_validation(self, quick_mode: bool = False):
        """Run strategy performance validation tests"""
        self.logger.info("Running strategy validation tests...")
        
        try:
            # Strategy performance validation
            result = await self.strategy_validator.validate_strategy_performance()
            self._add_result('strategy_sharpe_ratio', result)
            
            if not quick_mode:
                # Optimization convergence validation
                result = await self.strategy_validator.validate_optimization_convergence()
                self._add_result('optimization_convergence', result)
                
        except Exception as e:
            self.logger.error(f"Strategy validation failed: {e}")
            self._add_error_result('strategy_validation', str(e))
    
    def _add_result(self, target_key: str, validation_data: Dict[str, Any]):
        """Add validation result from validator"""
        target = self.targets[target_key]
        
        # Determine status based on comparison
        measured_value = validation_data.get('measured_value')
        status = self._compare_against_target(measured_value, target)
        
        # Generate recommendations if failed
        recommendations = []
        if status == ValidationStatus.FAIL:
            recommendations = self._generate_recommendations(target_key, validation_data)
        
        result = ValidationResult(
            test_name=target.name,
            category=target.category,
            target=target,
            measured_value=measured_value,
            status=status,
            message=validation_data.get('message', ''),
            duration_seconds=validation_data.get('duration_seconds', 0.0),
            timestamp=datetime.now(),
            metadata=validation_data.get('metadata', {}),
            recommendations=recommendations
        )
        
        self.results.append(result)
        
        # Log result
        log_level = logging.ERROR if status == ValidationStatus.FAIL else logging.INFO
        self.logger.log(
            log_level, 
            f"{status.value}: {target.name} - {result.message}"
        )
    
    def _add_error_result(self, test_name: str, error_message: str):
        """Add error result for failed test"""
        result = ValidationResult(
            test_name=test_name,
            category='error',
            target=PerformanceTarget(
                name=test_name,
                description=f"Test failed with error: {error_message}",
                target_value=0,
                unit='',
                comparison_operator='eq',
                critical=True
            ),
            measured_value=None,
            status=ValidationStatus.ERROR,
            message=f"Test failed: {error_message}",
            duration_seconds=0.0,
            timestamp=datetime.now(),
            metadata={'error': error_message}
        )
        
        self.results.append(result)
        self.logger.error(f"ERROR: {test_name} - {error_message}")
    
    def _compare_against_target(self, measured_value: Union[int, float, None], 
                               target: PerformanceTarget) -> ValidationStatus:
        """Compare measured value against target"""
        if measured_value is None:
            return ValidationStatus.ERROR
        
        try:
            if target.comparison_operator == 'lt':
                return ValidationStatus.PASS if measured_value < target.target_value else ValidationStatus.FAIL
            elif target.comparison_operator == 'gt':
                return ValidationStatus.PASS if measured_value > target.target_value else ValidationStatus.FAIL
            elif target.comparison_operator == 'lte':
                return ValidationStatus.PASS if measured_value <= target.target_value else ValidationStatus.FAIL
            elif target.comparison_operator == 'gte':
                return ValidationStatus.PASS if measured_value >= target.target_value else ValidationStatus.FAIL
            elif target.comparison_operator == 'eq':
                return ValidationStatus.PASS if measured_value == target.target_value else ValidationStatus.FAIL
            else:
                return ValidationStatus.ERROR
        except (TypeError, ValueError):
            return ValidationStatus.ERROR
    
    def _generate_recommendations(self, target_key: str, validation_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for failed targets"""
        recommendations = []
        
        if 'latency' in target_key:
            recommendations.extend([
                "Consider implementing asynchronous processing for non-critical operations",
                "Review and optimize database query patterns",
                "Implement result caching for frequently accessed data",
                "Consider using faster data structures or algorithms"
            ])
        elif 'throughput' in target_key:
            recommendations.extend([
                "Scale horizontally by adding more processing workers",
                "Implement batch processing for bulk operations",
                "Optimize bottleneck components identified in profiling",
                "Consider using connection pooling and resource reuse"
            ])
        elif 'memory' in target_key:
            recommendations.extend([
                "Implement streaming processing for large datasets",
                "Review memory leaks and optimize object lifecycle",
                "Use memory-efficient data structures",
                "Implement garbage collection tuning"
            ])
        elif 'strategy' in target_key:
            recommendations.extend([
                "Review strategy parameters and risk management",
                "Add more diverse market data sources",
                "Implement ensemble methods for better performance",
                "Consider advanced machine learning techniques"
            ])
        
        return recommendations
    
    def _generate_summary(self) -> ValidationSummary:
        """Generate validation summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        failed_tests = sum(1 for r in self.results if r.status == ValidationStatus.FAIL)
        warning_tests = sum(1 for r in self.results if r.status == ValidationStatus.WARNING)
        error_tests = sum(1 for r in self.results if r.status == ValidationStatus.ERROR)
        skipped_tests = sum(1 for r in self.results if r.status == ValidationStatus.SKIP)
        
        # Critical failures
        critical_failures = [
            r.test_name for r in self.results 
            if r.status == ValidationStatus.FAIL and r.target.critical
        ]
        
        # Determine overall status
        if critical_failures:
            overall_status = ValidationStatus.FAIL
        elif failed_tests > 0 or error_tests > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASS
        
        # Calculate duration
        duration = 0.0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return ValidationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests,
            critical_failures=critical_failures,
            overall_status=overall_status,
            total_duration_seconds=duration,
            validation_timestamp=self.start_time or datetime.now()
        )
    
    async def _save_results(self, summary: ValidationSummary):
        """Save validation results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = self.output_dir / f"validation_results_{timestamp}.json"
        detailed_results = {
            'summary': {
                'total_tests': summary.total_tests,
                'passed_tests': summary.passed_tests,
                'failed_tests': summary.failed_tests,
                'warning_tests': summary.warning_tests,
                'error_tests': summary.error_tests,
                'skipped_tests': summary.skipped_tests,
                'critical_failures': summary.critical_failures,
                'overall_status': summary.overall_status.value,
                'total_duration_seconds': summary.total_duration_seconds,
                'validation_timestamp': summary.validation_timestamp.isoformat(),
                'platform_version': summary.platform_version
            },
            'results': [result.to_dict() for result in self.results],
            'targets': {key: {
                'name': target.name,
                'description': target.description,
                'target_value': target.target_value,
                'unit': target.unit,
                'comparison_operator': target.comparison_operator,
                'critical': target.critical,
                'category': target.category
            } for key, target in self.targets.items()}
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        self.logger.info(f"Detailed results saved to {results_file}")
        
        # Save summary report
        summary_file = self.output_dir / f"validation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'summary': detailed_results['summary']
            }, f, indent=2, default=str)
        
        self.logger.info(f"Summary saved to {summary_file}")

    def generate_human_readable_report(self, summary: ValidationSummary) -> str:
        """Generate human-readable validation report"""
        lines = [
            "="*80,
            "AI NEWS TRADING PLATFORM - PERFORMANCE VALIDATION REPORT",
            "="*80,
            f"Validation Timestamp: {summary.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Platform Version: {summary.platform_version}",
            f"Total Duration: {summary.total_duration_seconds:.2f} seconds",
            "",
            "SUMMARY:",
            f"  Overall Status: {summary.overall_status.value}",
            f"  Total Tests: {summary.total_tests}",
            f"  Passed: {summary.passed_tests}",
            f"  Failed: {summary.failed_tests}",
            f"  Warnings: {summary.warning_tests}",
            f"  Errors: {summary.error_tests}",
            f"  Skipped: {summary.skipped_tests}",
            ""
        ]

        if summary.critical_failures:
            lines.extend([
                "CRITICAL FAILURES:",
                *[f"  ❌ {failure}" for failure in summary.critical_failures],
                ""
            ])

        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        for category, results in categories.items():
            lines.extend([
                f"{category.upper()} VALIDATION RESULTS:",
                "-" * 40
            ])
            
            for result in results:
                status_icon = {
                    ValidationStatus.PASS: "✅",
                    ValidationStatus.FAIL: "❌", 
                    ValidationStatus.WARNING: "⚠️",
                    ValidationStatus.ERROR: "🔴",
                    ValidationStatus.SKIP: "⏭️"
                }.get(result.status, "❓")
                
                lines.append(f"  {status_icon} {result.test_name}")
                lines.append(f"     Target: {result.target.comparison_operator} {result.target.target_value} {result.target.unit}")
                if result.measured_value is not None:
                    lines.append(f"     Measured: {result.measured_value} {result.target.unit}")
                lines.append(f"     Status: {result.status.value}")
                lines.append(f"     Duration: {result.duration_seconds:.3f}s")
                
                if result.recommendations:
                    lines.append("     Recommendations:")
                    for rec in result.recommendations:
                        lines.append(f"       - {rec}")
                lines.append("")

        lines.append("="*80)
        return "\n".join(lines)


async def main():
    """Main entry point for performance validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI News Trading Performance Validator")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (critical tests only)')
    parser.add_argument('--categories', nargs='+', choices=['latency', 'throughput', 'resource', 'strategy'],
                       help='Categories to validate')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Initialize validator
        validator = PerformanceValidator(
            config=None,  # Load from file if needed
            output_dir=args.output_dir
        )
        
        # Run validation
        print("Starting AI News Trading Platform Performance Validation...")
        summary = await validator.validate_all(
            quick_mode=args.quick,
            categories=args.categories
        )
        
        # Generate and display report
        report = validator.generate_human_readable_report(summary)
        print(report)
        
        # Store progress in memory
        progress_data = {
            'validation_completed': True,
            'overall_status': summary.overall_status.value,
            'total_tests': summary.total_tests,
            'passed_tests': summary.passed_tests,
            'failed_tests': summary.failed_tests,
            'critical_failures': summary.critical_failures,
            'timestamp': summary.validation_timestamp.isoformat(),
            'duration_seconds': summary.total_duration_seconds
        }
        
        memory_file = Path('/workspaces/ai-news-trader/memory/data/swarm-benchmark-validation-progress.json')
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(memory_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Exit with appropriate code
        if summary.overall_status == ValidationStatus.FAIL:
            exit(1)
        elif summary.overall_status == ValidationStatus.WARNING:
            exit(2)
        else:
            exit(0)
            
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        exit(130)
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())