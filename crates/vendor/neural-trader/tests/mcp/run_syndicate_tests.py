#!/usr/bin/env python3
"""
Test Runner for Syndicate MCP Tools

This script provides an easy way to run different test suites
for the syndicate MCP integration with various options.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time
from datetime import datetime


def run_tests(test_type="all", verbose=True, coverage=False, markers=None, failfast=False):
    """Run syndicate tests with specified options"""
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add test files based on type
    if test_type == "all":
        cmd.extend([
            "tests/mcp/test_syndicate_tools.py",
            "tests/mcp/test_syndicate_integration.py"
        ])
    elif test_type == "unit":
        cmd.append("tests/mcp/test_syndicate_tools.py")
        if not markers:
            markers = ["unit"]
    elif test_type == "integration":
        cmd.append("tests/mcp/test_syndicate_integration.py")
        if not markers:
            markers = ["integration"]
    elif test_type == "security":
        cmd.extend([
            "tests/mcp/test_syndicate_tools.py",
            "tests/mcp/test_syndicate_integration.py"
        ])
        if not markers:
            markers = ["security"]
    elif test_type == "performance":
        cmd.extend([
            "tests/mcp/test_syndicate_tools.py",
            "tests/mcp/test_syndicate_integration.py"
        ])
        if not markers:
            markers = ["performance"]
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src.sports_betting.syndicate",
            "--cov=src.mcp.handlers",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add fail fast
    if failfast:
        cmd.extend(["--maxfail=1", "-x"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes"  # Colored output
    ])
    
    # Print command
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    duration = time.time() - start_time
    
    print("=" * 60)
    print(f"Tests completed in {duration:.2f} seconds")
    
    return result.returncode


def run_specific_test(test_name, verbose=True):
    """Run a specific test by name"""
    cmd = [
        "pytest",
        "-k", test_name,
        "--tb=short",
        "-s"
    ]
    
    if verbose:
        cmd.append("-v")
    
    print(f"Running test: {test_name}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode


def run_stress_tests(duration_seconds=60, concurrent_users=10):
    """Run stress tests to simulate heavy load"""
    print(f"Running stress tests for {duration_seconds}s with {concurrent_users} concurrent users")
    print("=" * 60)
    
    cmd = [
        "pytest",
        "tests/mcp/test_syndicate_tools.py::TestSyndicatePerformance",
        "-v",
        "-s",
        "--tb=short",
        f"--stress-duration={duration_seconds}",
        f"--concurrent-users={concurrent_users}"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode


def generate_test_report():
    """Generate HTML test report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"syndicate_test_report_{timestamp}.html"
    
    cmd = [
        "pytest",
        "tests/mcp/test_syndicate_tools.py",
        "tests/mcp/test_syndicate_integration.py",
        "--html=" + report_name,
        "--self-contained-html",
        "-v"
    ]
    
    print(f"Generating test report: {report_name}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    
    if result.returncode == 0:
        print(f"Report generated successfully: {report_name}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Syndicate MCP Tests")
    
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "security", "performance", "stress", "report"],
        default="all",
        nargs="?",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "-m", "--markers",
        nargs="+",
        help="Additional pytest markers to use"
    )
    
    parser.add_argument(
        "-f", "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "-k", "--test-name",
        help="Run specific test by name pattern"
    )
    
    parser.add_argument(
        "--stress-duration",
        type=int,
        default=60,
        help="Duration for stress tests in seconds"
    )
    
    parser.add_argument(
        "--concurrent-users",
        type=int,
        default=10,
        help="Number of concurrent users for stress tests"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("SYNDICATE MCP TOOLS TEST RUNNER")
    print("=" * 60)
    print(f"Test Type: {args.test_type}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Run specific test if requested
    if args.test_name:
        return_code = run_specific_test(args.test_name, args.verbose)
    elif args.test_type == "stress":
        return_code = run_stress_tests(args.stress_duration, args.concurrent_users)
    elif args.test_type == "report":
        return_code = generate_test_report()
    else:
        return_code = run_tests(
            test_type=args.test_type,
            verbose=args.verbose,
            coverage=args.coverage,
            markers=args.markers,
            failfast=args.failfast
        )
    
    # Print summary
    print("\n" + "=" * 60)
    if return_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60 + "\n")
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()