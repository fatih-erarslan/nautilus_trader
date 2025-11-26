#!/usr/bin/env python3
"""
Final Performance Validation Script for AI News Trading Platform.

This script orchestrates comprehensive performance validation and generates
final reports with optimization recommendations for production deployment.

Performance Targets Validated:
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
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from validation import PerformanceValidator, ValidationStatus
from reports import ReportGenerator, ReportConfiguration, PerformanceSummaryGenerator, OptimizationRecommendationEngine, BenchmarkComparisonAnalyzer


class FinalValidationOrchestrator:
    """Orchestrates final performance validation and reporting"""
    
    def __init__(self, config: Optional[Dict] = None, output_dir: str = None):
        """Initialize final validation orchestrator
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for results
        """
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.performance_validator = PerformanceValidator(config, str(self.output_dir))
        self.report_generator = ReportGenerator(output_dir=str(self.output_dir))
        self.summary_generator = PerformanceSummaryGenerator()
        self.recommendation_engine = OptimizationRecommendationEngine()
        self.comparison_analyzer = BenchmarkComparisonAnalyzer()
        
        # Validation results
        self.validation_results = None
        self.validation_summary = None
        self.performance_summary = None
        self.recommendations = None
        self.historical_comparison = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        # Create logs directory
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_file = logs_dir / f"final_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger('FinalValidation')
        logger.info(f"Final validation session started. Logs: {log_file}")
        return logger
    
    async def run_complete_validation(self, quick_mode: bool = False, 
                                    categories: Optional[List[str]] = None,
                                    include_historical: bool = True) -> Dict[str, Any]:
        """Run complete performance validation with reporting
        
        Args:
            quick_mode: Run only critical tests for faster validation
            categories: List of categories to validate
            include_historical: Whether to include historical comparison
            
        Returns:
            Dictionary with all validation results and reports
        """
        self.logger.info("="*80)
        self.logger.info("AI NEWS TRADING PLATFORM - FINAL PERFORMANCE VALIDATION")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Phase 1: Run Performance Validation
            self.logger.info("Phase 1: Running comprehensive performance validation...")
            self.validation_summary = await self.performance_validator.validate_all(
                quick_mode=quick_mode,
                categories=categories
            )
            
            # Get detailed results
            validation_data = await self._compile_validation_data()
            
            # Phase 2: Generate Performance Summary
            self.logger.info("Phase 2: Generating performance summary and analysis...")
            self.performance_summary = self.summary_generator.generate_summary(validation_data)
            
            # Phase 3: Generate Optimization Recommendations
            self.logger.info("Phase 3: Generating optimization recommendations...")
            self.recommendations = self.recommendation_engine.generate_recommendations(validation_data)
            
            # Phase 4: Historical Comparison (if requested)
            if include_historical:
                self.logger.info("Phase 4: Analyzing historical performance trends...")
                historical_data = self._load_historical_data()
                if historical_data:
                    self.historical_comparison = self.comparison_analyzer.compare_with_historical(
                        validation_data, historical_data
                    )
                else:
                    self.logger.warning("No historical data found for comparison")
            
            # Phase 5: Generate Comprehensive Reports
            self.logger.info("Phase 5: Generating comprehensive validation reports...")
            report_paths = await self._generate_reports(validation_data)
            
            # Phase 6: Store Progress in Memory
            await self._store_progress_in_memory()
            
            # Compile final results
            final_results = self._compile_final_results(validation_data, report_paths, start_time)
            
            # Log completion
            total_time = time.time() - start_time
            self.logger.info(f"Final validation completed in {total_time:.2f} seconds")
            self.logger.info(f"Overall Status: {self.validation_summary.overall_status.value}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Final validation failed: {str(e)}")
            raise
    
    async def _compile_validation_data(self) -> Dict[str, Any]:
        """Compile validation data from validator results"""
        # Convert ValidationSummary and results to dictionary format
        validation_data = {
            'summary': {
                'total_tests': self.validation_summary.total_tests,
                'passed_tests': self.validation_summary.passed_tests,
                'failed_tests': self.validation_summary.failed_tests,
                'warning_tests': self.validation_summary.warning_tests,
                'error_tests': self.validation_summary.error_tests,
                'skipped_tests': self.validation_summary.skipped_tests,
                'critical_failures': self.validation_summary.critical_failures,
                'overall_status': self.validation_summary.overall_status.value,
                'total_duration_seconds': self.validation_summary.total_duration_seconds,
                'validation_timestamp': self.validation_summary.validation_timestamp.isoformat(),
                'platform_version': self.validation_summary.platform_version
            },
            'results': [result.to_dict() for result in self.performance_validator.results],
            'targets': {
                key: {
                    'name': target.name,
                    'description': target.description,
                    'target_value': target.target_value,
                    'unit': target.unit,
                    'comparison_operator': target.comparison_operator,
                    'critical': target.critical,
                    'category': target.category
                }
                for key, target in self.performance_validator.targets.items()
            }
        }
        
        return validation_data
    
    def _load_historical_data(self) -> Optional[List[Dict]]:
        """Load historical validation data for comparison"""
        historical_files = []
        
        # Look for historical results in results directory
        results_pattern = self.output_dir / "validation_results_*.json"
        historical_files.extend(self.output_dir.glob("validation_results_*.json"))
        
        # Also check for backup files in memory directory
        memory_dir = Path("/workspaces/ai-news-trader/memory/backups")
        if memory_dir.exists():
            historical_files.extend(memory_dir.glob("backup-*.json"))
        
        # Load and parse historical data
        historical_data = []
        for file_path in sorted(historical_files)[-10:]:  # Last 10 files
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Check if it's validation data
                    if 'summary' in data and 'results' in data:
                        historical_data.append(data)
                    elif 'validation_results' in data:
                        historical_data.append(data['validation_results'])
            except Exception as e:
                self.logger.debug(f"Could not load historical data from {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(historical_data)} historical data points for comparison")
        return historical_data if historical_data else None
    
    async def _generate_reports(self, validation_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive validation reports"""
        # Configure report generation
        report_config = ReportConfiguration(
            title="AI News Trading Platform - Final Performance Validation Report",
            subtitle="Production Readiness Assessment and Optimization Recommendations",
            include_executive_summary=True,
            include_detailed_metrics=True,
            include_visual_charts=True,
            include_recommendations=True,
            include_historical_comparison=bool(self.historical_comparison),
            output_format="all"  # Generate all formats
        )
        
        # Update report generator configuration
        self.report_generator.config = report_config
        
        # Generate comprehensive report
        report_paths = self.report_generator.generate_comprehensive_report(
            validation_data=validation_data,
            historical_data=self._get_historical_data_for_report()
        )
        
        return report_paths
    
    def _get_historical_data_for_report(self) -> Optional[List[Dict]]:
        """Get historical data formatted for report generation"""
        if self.historical_comparison:
            # Convert historical comparison back to raw data format
            return []  # Placeholder - in real implementation would reconstruct
        return None
    
    async def _store_progress_in_memory(self):
        """Store validation progress in memory system"""
        memory_dir = Path("/workspaces/ai-news-trader/memory/data")
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        progress_data = {
            'validation_completed': True,
            'completion_timestamp': datetime.now().isoformat(),
            'overall_status': self.validation_summary.overall_status.value,
            'summary': {
                'total_tests': self.validation_summary.total_tests,
                'passed_tests': self.validation_summary.passed_tests,
                'failed_tests': self.validation_summary.failed_tests,
                'critical_failures': self.validation_summary.critical_failures,
                'duration_seconds': self.validation_summary.total_duration_seconds
            },
            'key_findings': self._extract_key_findings(),
            'recommendations_summary': self._extract_recommendations_summary(),
            'production_readiness': self._assess_production_readiness()
        }
        
        # Store progress
        progress_file = memory_dir / "swarm-benchmark-validation-progress.json"
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        self.logger.info(f"Validation progress stored in memory: {progress_file}")
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from validation results"""
        findings = []
        
        # Overall status finding
        if self.validation_summary.overall_status == ValidationStatus.PASS:
            findings.append("System meets all critical performance targets for production deployment")
        elif self.validation_summary.overall_status == ValidationStatus.FAIL:
            findings.append("System has critical performance issues that must be addressed before production")
        else:
            findings.append("System has some performance concerns but may be suitable for limited production use")
        
        # Critical failures
        if self.validation_summary.critical_failures:
            findings.append(f"Critical attention required: {len(self.validation_summary.critical_failures)} critical test failures")
        
        # Performance insights from summary
        if self.performance_summary and 'insights' in self.performance_summary:
            findings.extend(self.performance_summary['insights'][:3])  # Top 3 insights
        
        return findings
    
    def _extract_recommendations_summary(self) -> Dict[str, int]:
        """Extract summary of recommendations by priority"""
        if not self.recommendations:
            return {}
        
        priority_counts = {}
        for category, recs in self.recommendations.items():
            for rec in recs:
                priority = rec.get('priority', 'UNKNOWN')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return priority_counts
    
    def _assess_production_readiness(self) -> Dict[str, Any]:
        """Assess production readiness based on validation results"""
        readiness = {
            'ready_for_production': False,
            'confidence_level': 'LOW',
            'blocking_issues': [],
            'recommendations_before_deployment': [],
            'estimated_time_to_production': 'UNKNOWN'
        }
        
        # Check critical failures
        if not self.validation_summary.critical_failures:
            if self.validation_summary.overall_status == ValidationStatus.PASS:
                readiness['ready_for_production'] = True
                readiness['confidence_level'] = 'HIGH'
                readiness['estimated_time_to_production'] = 'READY'
            elif self.validation_summary.passed_tests / self.validation_summary.total_tests >= 0.8:
                readiness['ready_for_production'] = True
                readiness['confidence_level'] = 'MEDIUM'
                readiness['estimated_time_to_production'] = '1-2 weeks'
        else:
            readiness['blocking_issues'] = self.validation_summary.critical_failures
            readiness['estimated_time_to_production'] = '4-8 weeks'
        
        # Add high-priority recommendations
        if self.recommendations:
            for category, recs in self.recommendations.items():
                high_priority = [r for r in recs if r.get('priority') in ['CRITICAL', 'HIGH']]
                for rec in high_priority[:3]:  # Top 3 high priority
                    readiness['recommendations_before_deployment'].append(rec.get('title', ''))
        
        return readiness
    
    def _compile_final_results(self, validation_data: Dict[str, Any], 
                             report_paths: Dict[str, str], start_time: float) -> Dict[str, Any]:
        """Compile final results summary"""
        return {
            'validation_summary': {
                'overall_status': self.validation_summary.overall_status.value,
                'total_tests': self.validation_summary.total_tests,
                'passed_tests': self.validation_summary.passed_tests,
                'failed_tests': self.validation_summary.failed_tests,
                'critical_failures': self.validation_summary.critical_failures,
                'validation_duration': self.validation_summary.total_duration_seconds,
                'completion_time': datetime.now().isoformat()
            },
            'performance_analysis': {
                'key_metrics': self.performance_summary.get('key_metrics', []) if self.performance_summary else [],
                'bottlenecks': self.performance_summary.get('bottlenecks', []) if self.performance_summary else [],
                'scores': self.performance_summary.get('scores', {}) if self.performance_summary else {}
            },
            'recommendations': {
                'total_recommendations': sum(len(recs) for recs in self.recommendations.values()) if self.recommendations else 0,
                'high_priority_count': sum(
                    len([r for r in recs if r.get('priority') in ['CRITICAL', 'HIGH']])
                    for recs in self.recommendations.values()
                ) if self.recommendations else 0,
                'categories': list(self.recommendations.keys()) if self.recommendations else []
            },
            'historical_comparison': {
                'available': bool(self.historical_comparison),
                'trend_summary': self.historical_comparison.summary if self.historical_comparison else None
            },
            'production_readiness': self._assess_production_readiness(),
            'reports_generated': report_paths,
            'total_execution_time': time.time() - start_time
        }
    
    def print_executive_summary(self):
        """Print executive summary to console"""
        if not self.validation_summary:
            print("No validation results available")
            return
        
        print("\n" + "="*80)
        print("AI NEWS TRADING PLATFORM - VALIDATION EXECUTIVE SUMMARY")
        print("="*80)
        
        # Overall status
        status_icon = {
            ValidationStatus.PASS: "✅",
            ValidationStatus.FAIL: "❌",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.ERROR: "🔴"
        }.get(self.validation_summary.overall_status, "❓")
        
        print(f"Overall Status: {status_icon} {self.validation_summary.overall_status.value}")
        print(f"Validation Completed: {self.validation_summary.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {self.validation_summary.total_duration_seconds:.2f} seconds")
        print()
        
        # Test results
        print("TEST RESULTS:")
        print(f"  Total Tests: {self.validation_summary.total_tests}")
        print(f"  ✅ Passed: {self.validation_summary.passed_tests}")
        print(f"  ❌ Failed: {self.validation_summary.failed_tests}")
        print(f"  ⚠️  Warnings: {self.validation_summary.warning_tests}")
        print(f"  🔴 Errors: {self.validation_summary.error_tests}")
        print()
        
        # Critical failures
        if self.validation_summary.critical_failures:
            print("CRITICAL FAILURES:")
            for failure in self.validation_summary.critical_failures:
                print(f"  ❌ {failure}")
            print()
        
        # Production readiness
        readiness = self._assess_production_readiness()
        ready_icon = "✅" if readiness['ready_for_production'] else "❌"
        print(f"Production Ready: {ready_icon} {readiness['ready_for_production']}")
        print(f"Confidence Level: {readiness['confidence_level']}")
        print(f"Estimated Time to Production: {readiness['estimated_time_to_production']}")
        
        if readiness['blocking_issues']:
            print("\nBlocking Issues:")
            for issue in readiness['blocking_issues']:
                print(f"  - {issue}")
        
        print("="*80)


async def main():
    """Main entry point for final validation"""
    parser = argparse.ArgumentParser(
        description="AI News Trading Platform - Final Performance Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python final_validation.py --full                    # Complete validation
  python final_validation.py --quick                   # Quick validation
  python final_validation.py --categories latency throughput  # Specific categories
  python final_validation.py --output-dir /path/to/results    # Custom output
        """
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick validation (critical tests only)')
    parser.add_argument('--full', action='store_true',
                       help='Run full validation (default)')
    parser.add_argument('--categories', nargs='+', 
                       choices=['latency', 'throughput', 'resource', 'strategy'],
                       help='Specific categories to validate')
    parser.add_argument('--output-dir', 
                       help='Output directory for results and reports')
    parser.add_argument('--no-historical', action='store_true',
                       help='Skip historical comparison')
    parser.add_argument('--config', 
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    try:
        # Initialize orchestrator
        orchestrator = FinalValidationOrchestrator(
            config=config,
            output_dir=args.output_dir
        )
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Run validation
        print("Starting AI News Trading Platform Final Performance Validation...")
        print("This may take several minutes depending on the scope of validation.")
        
        results = await orchestrator.run_complete_validation(
            quick_mode=args.quick,
            categories=args.categories,
            include_historical=not args.no_historical
        )
        
        # Print executive summary
        orchestrator.print_executive_summary()
        
        # Print report locations
        if results['reports_generated']:
            print("\nREPORTS GENERATED:")
            for format_type, path in results['reports_generated'].items():
                print(f"  {format_type.upper()}: {path}")
        
        # Print recommendations summary
        rec_summary = results['recommendations']
        if rec_summary['total_recommendations'] > 0:
            print(f"\nOPTIMIZATION RECOMMENDATIONS:")
            print(f"  Total: {rec_summary['total_recommendations']}")
            print(f"  High Priority: {rec_summary['high_priority_count']}")
            print(f"  Categories: {', '.join(rec_summary['categories'])}")
        
        print(f"\nValidation completed in {results['total_execution_time']:.2f} seconds")
        
        # Exit with appropriate code
        if results['validation_summary']['overall_status'] == 'FAIL':
            print("\n❌ VALIDATION FAILED - Critical issues must be addressed before production")
            sys.exit(1)
        elif results['validation_summary']['failed_tests'] > 0:
            print("\n⚠️  VALIDATION PASSED WITH WARNINGS - Review failed tests before production")
            sys.exit(2)
        else:
            print("\n✅ VALIDATION PASSED - System ready for production deployment")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nValidation failed with error: {str(e)}")
        logging.error(f"Final validation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())