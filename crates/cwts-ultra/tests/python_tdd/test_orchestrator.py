"""
Test Orchestrator - Comprehensive TDD Test Runner
Coordinates all test suites with Complex Adaptive Systems principles
"""

import pytest
import asyncio
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestSuiteResult:
    """Result container for test suite execution"""
    suite_name: str
    passed: int
    failed: int
    skipped: int
    errors: int
    warnings: int
    execution_time: float
    coverage_percent: float
    exit_code: int
    stdout: str
    stderr: str
    
    @property
    def success_rate(self) -> float:
        total = self.passed + self.failed + self.errors
        return self.passed / total if total > 0 else 0.0
    
    @property
    def total_tests(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors

class TestOrchestrator:
    """Orchestrates comprehensive test execution with adaptive strategies"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results: Dict[str, TestSuiteResult] = {}
        self.global_metrics: Dict[str, Any] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite with orchestration"""
        logger.info("Starting comprehensive TDD test suite execution")
        
        start_time = time.time()
        
        # Test suites in execution order
        test_suites = [
            {
                'name': 'Unit Tests - Financial Calculations',
                'command': ['python', '-m', 'pytest', 
                           'tests/python_tdd/test_financial_calculations.py', 
                           '-v', '--tb=short', '--cov=.', '--cov-report=term-missing'],
                'critical': True,
                'timeout': 300
            },
            {
                'name': 'Unit Tests - Strategy Integration',
                'command': ['python', '-m', 'pytest', 
                           'tests/python_tdd/test_strategy_integration.py',
                           '-v', '--tb=short', '--cov-append'],
                'critical': True,
                'timeout': 600
            },
            {
                'name': 'Integration Tests - System Components',
                'command': ['python', '-m', 'pytest', 
                           'tests/integration/',
                           '-v', '--tb=short', '--cov-append'],
                'critical': False,
                'timeout': 900
            },
            {
                'name': 'Playwright E2E Tests',
                'command': ['python', '-m', 'pytest', 
                           'tests/playwright_e2e/',
                           '-v', '--tb=short', '-x'],
                'critical': False,
                'timeout': 1800
            },
            {
                'name': 'Performance Tests',
                'command': ['python', '-m', 'pytest', 
                           'tests/', '-m', 'performance',
                           '-v', '--tb=short'],
                'critical': False,
                'timeout': 1200
            },
            {
                'name': 'Security Tests',
                'command': ['python', '-m', 'pytest', 
                           'tests/', '-m', 'security',
                           '-v', '--tb=short'],
                'critical': True,
                'timeout': 600
            }
        ]
        
        # Execute test suites
        results = await self._execute_test_suites(test_suites)
        
        # Calculate global metrics
        total_time = time.time() - start_time
        self.global_metrics = self._calculate_global_metrics(results, total_time)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Adapt for future runs
        self._adapt_test_strategy()
        
        logger.info(f"Test suite completed in {total_time:.2f}s")
        return report
    
    async def _execute_test_suites(self, test_suites: List[Dict[str, Any]]) -> Dict[str, TestSuiteResult]:
        """Execute test suites with parallel capabilities"""
        results = {}
        
        # Separate critical and non-critical tests
        critical_tests = [t for t in test_suites if t.get('critical', False)]
        non_critical_tests = [t for t in test_suites if not t.get('critical', False)]
        
        # Execute critical tests sequentially
        for test_suite in critical_tests:
            logger.info(f"Executing critical test suite: {test_suite['name']}")
            result = await self._execute_single_test_suite(test_suite)
            results[test_suite['name']] = result
            
            # Stop if critical test fails
            if result.exit_code != 0:
                logger.error(f"Critical test suite failed: {test_suite['name']}")
                # Continue with other tests but mark as critical failure
        
        # Execute non-critical tests in parallel
        if non_critical_tests:
            logger.info("Executing non-critical tests in parallel")
            tasks = []
            for test_suite in non_critical_tests:
                task = self._execute_single_test_suite(test_suite)
                tasks.append((test_suite['name'], task))
            
            for name, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=1800)  # 30 min max
                    results[name] = result
                except asyncio.TimeoutError:
                    logger.error(f"Test suite timed out: {name}")
                    results[name] = TestSuiteResult(
                        suite_name=name,
                        passed=0, failed=0, skipped=0, errors=1, warnings=0,
                        execution_time=1800, coverage_percent=0.0,
                        exit_code=1, stdout="", stderr="Test timed out"
                    )
        
        return results
    
    async def _execute_single_test_suite(self, test_suite: Dict[str, Any]) -> TestSuiteResult:
        """Execute a single test suite"""
        suite_name = test_suite['name']
        command = test_suite['command']
        timeout = test_suite.get('timeout', 600)
        
        logger.info(f"Starting test suite: {suite_name}")
        start_time = time.time()
        
        try:
            # Execute test command
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': str(self.project_root)}
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse test results
            result = self._parse_test_output(
                suite_name, stdout.decode(), stderr.decode(),
                process.returncode, execution_time
            )
            
            logger.info(f"Completed {suite_name}: "
                       f"{result.passed}P/{result.failed}F/{result.skipped}S "
                       f"in {execution_time:.2f}s")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Test suite {suite_name} timed out after {timeout}s")
            return TestSuiteResult(
                suite_name=suite_name,
                passed=0, failed=0, skipped=0, errors=1, warnings=0,
                execution_time=timeout, coverage_percent=0.0,
                exit_code=1, stdout="", stderr="Test execution timed out"
            )
        except Exception as e:
            logger.error(f"Error executing {suite_name}: {e}")
            return TestSuiteResult(
                suite_name=suite_name,
                passed=0, failed=0, skipped=0, errors=1, warnings=0,
                execution_time=time.time() - start_time, coverage_percent=0.0,
                exit_code=1, stdout="", stderr=str(e)
            )
    
    def _parse_test_output(self, suite_name: str, stdout: str, stderr: str, 
                          exit_code: int, execution_time: float) -> TestSuiteResult:
        """Parse pytest output to extract test results"""
        
        # Initialize default values
        passed = failed = skipped = errors = warnings = 0
        coverage_percent = 0.0
        
        # Parse pytest summary line
        lines = stdout.split('\n')
        for line in lines:
            # Look for test results summary
            if 'passed' in line or 'failed' in line or 'error' in line:
                # Extract numbers from pytest summary
                import re
                
                passed_match = re.search(r'(\d+) passed', line)
                if passed_match:
                    passed = int(passed_match.group(1))
                
                failed_match = re.search(r'(\d+) failed', line)
                if failed_match:
                    failed = int(failed_match.group(1))
                
                skipped_match = re.search(r'(\d+) skipped', line)
                if skipped_match:
                    skipped = int(skipped_match.group(1))
                
                error_match = re.search(r'(\d+) error', line)
                if error_match:
                    errors = int(error_match.group(1))
            
            # Look for coverage percentage
            coverage_match = re.search(r'TOTAL.*?(\d+)%', line)
            if coverage_match:
                coverage_percent = float(coverage_match.group(1))
            
            # Count warnings
            if 'warning' in line.lower():
                warnings += 1
        
        return TestSuiteResult(
            suite_name=suite_name,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            warnings=warnings,
            execution_time=execution_time,
            coverage_percent=coverage_percent,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr
        )
    
    def _calculate_global_metrics(self, results: Dict[str, TestSuiteResult], 
                                 total_time: float) -> Dict[str, Any]:
        """Calculate global test metrics"""
        
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        total_skipped = sum(r.skipped for r in results.values())
        total_errors = sum(r.errors for r in results.values())
        total_warnings = sum(r.warnings for r in results.values())
        total_tests = total_passed + total_failed + total_skipped + total_errors
        
        # Calculate success rates
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        critical_success_rate = np.mean([r.success_rate for r in results.values() 
                                       if 'Critical' in r.suite_name or 'Unit' in r.suite_name])
        
        # Calculate coverage
        coverage_scores = [r.coverage_percent for r in results.values() if r.coverage_percent > 0]
        average_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
        
        # Calculate performance metrics
        avg_execution_time = np.mean([r.execution_time for r in results.values()])
        
        return {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_skipped': total_skipped,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'overall_success_rate': overall_success_rate,
            'critical_success_rate': critical_success_rate,
            'average_coverage': average_coverage,
            'total_execution_time': total_time,
            'average_execution_time': avg_execution_time,
            'test_suites_count': len(results),
            'failed_suites': [name for name, result in results.items() if result.exit_code != 0]
        }
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'project_root': str(self.project_root),
            'global_metrics': self.global_metrics,
            'test_results': {name: asdict(result) for name, result in self.test_results.items()},
            'summary': {
                'overall_status': 'PASSED' if self.global_metrics.get('overall_success_rate', 0) >= 0.95 else 'FAILED',
                'coverage_status': 'PASSED' if self.global_metrics.get('average_coverage', 0) >= 95.0 else 'NEEDS_IMPROVEMENT',
                'performance_status': 'OPTIMAL' if self.global_metrics.get('total_execution_time', 0) < 1800 else 'SLOW',
            },
            'recommendations': self._generate_recommendations(),
            'next_steps': self._generate_next_steps()
        }
        
        # Save report to file
        report_path = self.project_root / 'tests' / 'reports' / f'test_report_{int(time.time())}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if self.global_metrics.get('average_coverage', 0) < 95.0:
            recommendations.append("Increase test coverage to meet 95% minimum threshold")
        
        if self.global_metrics.get('total_failed', 0) > 0:
            recommendations.append("Address failing tests before proceeding to production")
        
        if self.global_metrics.get('total_execution_time', 0) > 1800:
            recommendations.append("Optimize test execution time - consider parallelization")
        
        if self.global_metrics.get('critical_success_rate', 0) < 1.0:
            recommendations.append("Critical test failures must be resolved immediately")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on test results"""
        next_steps = []
        
        if self.global_metrics.get('overall_success_rate', 0) >= 0.95:
            next_steps.append("Proceed with integration testing")
            next_steps.append("Schedule production deployment")
        else:
            next_steps.append("Fix failing tests")
            next_steps.append("Re-run test suite")
        
        next_steps.append("Update documentation based on test results")
        next_steps.append("Review performance metrics and optimize if needed")
        
        return next_steps
    
    def _adapt_test_strategy(self):
        """Adapt test strategy based on results using Complex Adaptive Systems principles"""
        
        # Calculate system fitness
        fitness_score = (
            self.global_metrics.get('overall_success_rate', 0) * 0.4 +
            (self.global_metrics.get('average_coverage', 0) / 100) * 0.3 +
            (1 - min(self.global_metrics.get('total_execution_time', 0) / 3600, 1)) * 0.3
        )
        
        # Store adaptation record
        adaptation = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'fitness_score': fitness_score,
            'metrics': self.global_metrics.copy(),
            'adaptations_applied': []
        }
        
        # Apply adaptations based on fitness
        if fitness_score < 0.7:
            adaptation['adaptations_applied'].append('Increase test timeout')
            adaptation['adaptations_applied'].append('Add more parallel execution')
        
        if self.global_metrics.get('average_coverage', 0) < 95.0:
            adaptation['adaptations_applied'].append('Add missing test cases')
            adaptation['adaptations_applied'].append('Review uncovered code paths')
        
        self.adaptation_history.append(adaptation)
        
        # Save adaptation history
        history_path = self.project_root / 'tests' / 'reports' / 'adaptation_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.adaptation_history, f, indent=2, default=str)

# Main orchestrator function
async def run_orchestrated_tests(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run orchestrated test suite"""
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    orchestrator = TestOrchestrator(project_root)
    return await orchestrator.run_comprehensive_test_suite()

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CWTS Ultra Test Orchestrator")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help='Project root directory')
    parser.add_argument('--output', type=Path, 
                       help='Output file for test report')
    
    args = parser.parse_args()
    
    # Run orchestrated tests
    results = asyncio.run(run_orchestrated_tests(args.project_root))
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUITE RESULTS")
    print("="*60)
    print(f"Total Tests: {results['global_metrics']['total_tests']}")
    print(f"Passed: {results['global_metrics']['total_passed']}")
    print(f"Failed: {results['global_metrics']['total_failed']}")
    print(f"Success Rate: {results['global_metrics']['overall_success_rate']:.1%}")
    print(f"Coverage: {results['global_metrics']['average_coverage']:.1f}%")
    print(f"Execution Time: {results['global_metrics']['total_execution_time']:.1f}s")
    print(f"Overall Status: {results['summary']['overall_status']}")
    print("="*60)
    
    if results['recommendations']:
        print("\nRECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"• {rec}")
    
    if results['next_steps']:
        print("\nNEXT STEPS:")
        for step in results['next_steps']:
            print(f"• {step}")
    
    print("\n")
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed report saved to: {args.output}")