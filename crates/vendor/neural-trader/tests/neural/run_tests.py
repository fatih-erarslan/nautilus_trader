#!/usr/bin/env python3
"""
Neural Forecasting Test Runner

Comprehensive test runner for neural forecasting components.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


class NeuralTestRunner:
    """Test runner for neural forecasting tests."""
    
    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.results = {}
        
    def run_unit_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run unit tests."""
        print("🧪 Running Neural Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest", 
            str(self.test_dir / "unit"),
            "-v" if verbose else "",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "-m", "unit"
        ]
        
        cmd = [c for c in cmd if c]  # Remove empty strings
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        self.results['unit_tests'] = {
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print(f"✅ Unit tests passed ({duration:.1f}s)")
        else:
            print(f"❌ Unit tests failed ({duration:.1f}s)")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results['unit_tests']
    
    def run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running Neural Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "integration"),
            "-v" if verbose else "",
            "--tb=short",
            "-m", "integration"
        ]
        
        cmd = [c for c in cmd if c]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        self.results['integration_tests'] = {
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print(f"✅ Integration tests passed ({duration:.1f}s)")
        else:
            print(f"❌ Integration tests failed ({duration:.1f}s)")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results['integration_tests']
    
    def run_performance_tests(self, verbose: bool = True, quick: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        print("⚡ Running Neural Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "performance"),
            "-v" if verbose else "",
            "--tb=short",
            "-m", "performance"
        ]
        
        if quick:
            cmd.extend(["-k", "not slow"])
        
        cmd = [c for c in cmd if c]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        self.results['performance_tests'] = {
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print(f"✅ Performance tests passed ({duration:.1f}s)")
        else:
            print(f"❌ Performance tests failed ({duration:.1f}s)")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results['performance_tests']
    
    def run_gpu_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run GPU-specific tests."""
        print("🖥️  Running Neural GPU Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-v" if verbose else "",
            "--tb=short",
            "-m", "gpu"
        ]
        
        cmd = [c for c in cmd if c]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        self.results['gpu_tests'] = {
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print(f"✅ GPU tests passed ({duration:.1f}s)")
        else:
            print(f"❌ GPU tests failed ({duration:.1f}s)")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results['gpu_tests']
    
    def run_all_tests(self, 
                     include_gpu: bool = True,
                     include_performance: bool = True,
                     quick: bool = False,
                     verbose: bool = True) -> Dict[str, Any]:
        """Run all neural forecasting tests."""
        print("🚀 Running Complete Neural Forecasting Test Suite...")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # Run unit tests
        self.run_unit_tests(verbose)
        
        # Run integration tests
        self.run_integration_tests(verbose)
        
        # Run performance tests if requested
        if include_performance:
            self.run_performance_tests(verbose, quick)
        
        # Run GPU tests if requested and available
        if include_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.run_gpu_tests(verbose)
                else:
                    print("⚠️  Skipping GPU tests (no GPU available)")
            except ImportError:
                print("⚠️  Skipping GPU tests (PyTorch not available)")
        
        total_duration = time.time() - total_start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Suite Summary:")
        print("=" * 60)
        
        passed_count = 0
        failed_count = 0
        
        for test_type, result in self.results.items():
            status = "✅ PASSED" if result['return_code'] == 0 else "❌ FAILED"
            duration = result['duration']
            print(f"{test_type:20} {status:10} ({duration:6.1f}s)")
            
            if result['return_code'] == 0:
                passed_count += 1
            else:
                failed_count += 1
        
        print("-" * 60)
        print(f"Total Tests: {passed_count + failed_count}")
        print(f"Passed:      {passed_count}")
        print(f"Failed:      {failed_count}")
        print(f"Duration:    {total_duration:.1f}s")
        print("=" * 60)
        
        # Generate results summary
        summary = {
            'total_duration': total_duration,
            'passed_count': passed_count,
            'failed_count': failed_count,
            'results': self.results,
            'timestamp': time.time()
        }
        
        return summary
    
    def run_coverage_report(self) -> Dict[str, Any]:
        """Generate coverage report."""
        print("📈 Generating Coverage Report...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--cov=src",
            "--cov=plans.neuralforecast",
            "--cov-report=html:htmlcov/neural",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage-neural.xml",
            "--tb=no",
            "-q"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Coverage report generated")
            print("📁 HTML report: htmlcov/neural/index.html")
            print("📄 XML report: coverage-neural.xml")
        else:
            print("❌ Coverage report generation failed")
            print(result.stderr)
        
        return {
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    
    def save_results(self, output_file: Path):
        """Save test results to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Results saved to: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Neural Forecasting Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Run all tests
  %(prog)s --unit                   # Run only unit tests
  %(prog)s --integration            # Run only integration tests
  %(prog)s --performance            # Run only performance tests
  %(prog)s --gpu                    # Run only GPU tests
  %(prog)s --all --quick             # Run all tests (quick mode)
  %(prog)s --coverage               # Generate coverage report
        """
    )
    
    # Test type selection
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--gpu', action='store_true', help='Run GPU tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    
    # Test options
    parser.add_argument('--quick', action='store_true', help='Quick mode (skip slow tests)')
    parser.add_argument('--quiet', action='store_true', help='Quiet output')
    parser.add_argument('--no-gpu', action='store_true', help='Skip GPU tests')
    parser.add_argument('--no-performance', action='store_true', help='Skip performance tests')
    
    # Output options
    parser.add_argument('--output', type=Path, help='Save results to file')
    parser.add_argument('--test-dir', type=Path, help='Test directory path')
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific test type selected
    if not any([args.all, args.unit, args.integration, args.performance, args.gpu, args.coverage]):
        args.all = True
    
    # Create test runner
    runner = NeuralTestRunner(args.test_dir)
    
    try:
        if args.coverage:
            runner.run_coverage_report()
        elif args.all:
            results = runner.run_all_tests(
                include_gpu=not args.no_gpu,
                include_performance=not args.no_performance,
                quick=args.quick,
                verbose=not args.quiet
            )
            
            # Exit with error code if any tests failed
            if results['failed_count'] > 0:
                sys.exit(1)
                
        else:
            # Run specific test types
            if args.unit:
                runner.run_unit_tests(not args.quiet)
            
            if args.integration:
                runner.run_integration_tests(not args.quiet)
            
            if args.performance:
                runner.run_performance_tests(not args.quiet, args.quick)
            
            if args.gpu:
                runner.run_gpu_tests(not args.quiet)
        
        # Save results if requested
        if args.output:
            runner.save_results(args.output)
    
    except KeyboardInterrupt:
        print("\n⚠️  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test run failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()