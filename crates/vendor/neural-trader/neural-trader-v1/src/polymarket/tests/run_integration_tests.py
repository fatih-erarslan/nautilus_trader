#!/usr/bin/env python3
"""
Integration Test Runner for Polymarket

This script runs all integration tests with coverage reporting and generates
comprehensive test reports including performance benchmarks and GPU validation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class IntegrationTestRunner:
    """Manages integration test execution and reporting."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.test_path = self.src_path / "polymarket" / "tests"
        self.integration_path = self.test_path / "integration"
        self.results_path = project_root / "test-results"
        self.coverage_path = project_root / "htmlcov"
        
        # Ensure directories exist
        self.results_path.mkdir(exist_ok=True)
    
    def run_tests(self, 
                  category: Optional[str] = None,
                  markers: Optional[List[str]] = None,
                  parallel: bool = True,
                  gpu: bool = True,
                  benchmark: bool = False) -> Dict[str, Any]:
        """Run integration tests with specified options."""
        
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.src_path)
        
        # Build pytest command
        cmd = ["pytest"]
        
        # Test path
        if category:
            test_file = self.integration_path / f"test_{category}.py"
            if test_file.exists():
                cmd.append(str(test_file))
            else:
                print(f"Warning: Test file {test_file} not found")
                return {"error": f"Test category '{category}' not found"}
        else:
            cmd.append(str(self.integration_path))
        
        # Markers
        if markers:
            marker_expr = " or ".join(markers)
            cmd.extend(["-m", marker_expr])
        elif not gpu:
            cmd.extend(["-m", "not gpu"])
        
        # Coverage
        cmd.extend([
            f"--cov={self.src_path / 'polymarket'}",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=json",
            "--cov-report=xml"
        ])
        
        # Output formats
        cmd.extend([
            f"--junit-xml={self.results_path / 'junit.xml'}",
            "--tb=short",
            "-v"
        ])
        
        # Parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])
        
        # Benchmarking
        if benchmark:
            cmd.extend([
                "--benchmark-only",
                f"--benchmark-json={self.results_path / 'benchmark.json'}",
                "--benchmark-verbose"
            ])
        
        # Run tests
        print(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        # Parse results
        return self._parse_results(result, duration)
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Analyze test coverage and generate reports."""
        coverage_file = self.project_root / "coverage.json"
        
        if not coverage_file.exists():
            return {"error": "No coverage data found. Run tests first."}
        
        with open(coverage_file) as f:
            coverage_data = json.load(f)
        
        # Calculate metrics
        total_lines = 0
        covered_lines = 0
        file_coverage = {}
        
        for file_path, file_data in coverage_data.get("files", {}).items():
            if "/tests/" not in file_path:  # Exclude test files
                lines = file_data.get("summary", {})
                file_lines = lines.get("num_statements", 0)
                file_covered = lines.get("covered_lines", 0)
                
                total_lines += file_lines
                covered_lines += file_covered
                
                if file_lines > 0:
                    coverage_percent = (file_covered / file_lines) * 100
                    file_coverage[file_path] = {
                        "lines": file_lines,
                        "covered": file_covered,
                        "percent": round(coverage_percent, 2),
                        "missing": lines.get("missing_lines", [])
                    }
        
        overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        return {
            "overall_coverage": round(overall_coverage, 2),
            "total_lines": total_lines,
            "covered_lines": covered_lines,
            "files": file_coverage,
            "report_path": str(self.coverage_path / "index.html")
        }
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance test results."""
        benchmark_file = self.results_path / "benchmark.json"
        
        if not benchmark_file.exists():
            return {"error": "No benchmark data found. Run with --benchmark flag."}
        
        with open(benchmark_file) as f:
            benchmark_data = json.load(f)
        
        # Extract key metrics
        benchmarks = benchmark_data.get("benchmarks", [])
        
        performance_summary = {
            "total_benchmarks": len(benchmarks),
            "metrics": {}
        }
        
        for bench in benchmarks:
            name = bench.get("name", "unknown")
            stats = bench.get("stats", {})
            
            performance_summary["metrics"][name] = {
                "mean_ms": round(stats.get("mean", 0) * 1000, 2),
                "min_ms": round(stats.get("min", 0) * 1000, 2),
                "max_ms": round(stats.get("max", 0) * 1000, 2),
                "stddev_ms": round(stats.get("stddev", 0) * 1000, 2),
                "iterations": stats.get("iterations", 0)
            }
        
        return performance_summary
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report_path = self.results_path / f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Get coverage data
        coverage = self.run_coverage_analysis()
        
        # Get performance data if available
        performance = self.run_performance_analysis()
        
        report_content = [
            "# Polymarket Integration Test Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Test Execution Summary",
            f"- Total Duration: {test_results.get('duration', 0):.2f} seconds",
            f"- Exit Code: {test_results.get('exit_code', 'N/A')}",
            f"- Tests Passed: {test_results.get('passed', 0)}",
            f"- Tests Failed: {test_results.get('failed', 0)}",
            f"- Tests Skipped: {test_results.get('skipped', 0)}"
        ]
        
        # Add coverage section
        if not coverage.get("error"):
            report_content.extend([
                "\n## Code Coverage",
                f"- Overall Coverage: {coverage['overall_coverage']}%",
                f"- Total Lines: {coverage['total_lines']}",
                f"- Covered Lines: {coverage['covered_lines']}",
                f"- [View HTML Report](file://{coverage['report_path']})",
                "\n### File Coverage"
            ])
            
            # Sort files by coverage percentage
            sorted_files = sorted(
                coverage["files"].items(),
                key=lambda x: x[1]["percent"]
            )
            
            for file_path, file_cov in sorted_files[:10]:  # Top 10 least covered
                short_path = file_path.replace(str(self.project_root), "")
                report_content.append(
                    f"- {short_path}: {file_cov['percent']}% "
                    f"({file_cov['covered']}/{file_cov['lines']})"
                )
        
        # Add performance section
        if not performance.get("error"):
            report_content.extend([
                "\n## Performance Benchmarks",
                f"- Total Benchmarks: {performance['total_benchmarks']}"
            ])
            
            for bench_name, metrics in performance["metrics"].items():
                report_content.extend([
                    f"\n### {bench_name}",
                    f"- Mean: {metrics['mean_ms']}ms",
                    f"- Min: {metrics['min_ms']}ms",
                    f"- Max: {metrics['max_ms']}ms",
                    f"- StdDev: {metrics['stddev_ms']}ms"
                ])
        
        # Add test output
        if test_results.get("stdout"):
            report_content.extend([
                "\n## Test Output",
                "```",
                test_results["stdout"][-5000:],  # Last 5000 chars
                "```"
            ])
        
        # Add errors if any
        if test_results.get("stderr"):
            report_content.extend([
                "\n## Errors",
                "```",
                test_results["stderr"][-5000:],  # Last 5000 chars
                "```"
            ])
        
        # Write report
        report_text = "\n".join(report_content)
        report_path.write_text(report_text)
        
        print(f"\nTest report generated: {report_path}")
        return str(report_path)
    
    def _parse_results(self, result: subprocess.CompletedProcess, duration: float) -> Dict[str, Any]:
        """Parse test execution results."""
        results = {
            "exit_code": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
        
        # Parse pytest output
        for line in result.stdout.split("\n"):
            if " passed" in line and " failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        results["passed"] = int(parts[i-1])
                    elif part == "failed":
                        results["failed"] = int(parts[i-1])
                    elif part == "skipped":
                        results["skipped"] = int(parts[i-1])
        
        return results
    
    def check_coverage_threshold(self, threshold: float = 100.0) -> bool:
        """Check if coverage meets threshold."""
        coverage = self.run_coverage_analysis()
        
        if coverage.get("error"):
            print(f"Error checking coverage: {coverage['error']}")
            return False
        
        actual_coverage = coverage["overall_coverage"]
        meets_threshold = actual_coverage >= threshold
        
        if meets_threshold:
            print(f"✅ Coverage {actual_coverage}% meets threshold of {threshold}%")
        else:
            print(f"❌ Coverage {actual_coverage}% below threshold of {threshold}%")
            
            # Show files with lowest coverage
            print("\nFiles with lowest coverage:")
            sorted_files = sorted(
                coverage["files"].items(),
                key=lambda x: x[1]["percent"]
            )
            
            for file_path, file_cov in sorted_files[:5]:
                short_path = file_path.replace(str(self.project_root), "")
                print(f"  - {short_path}: {file_cov['percent']}%")
                if file_cov["missing"]:
                    print(f"    Missing lines: {file_cov['missing'][:10]}...")
        
        return meets_threshold


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Polymarket integration tests")
    
    parser.add_argument(
        "--category",
        choices=["api_integration", "strategy_integration", "mcp_integration", 
                 "performance", "gpu_acceleration"],
        help="Test category to run"
    )
    
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Pytest markers to filter tests"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel test execution"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU tests"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=100.0,
        help="Coverage threshold percentage (default: 100)"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed test report"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    
    # Create runner
    runner = IntegrationTestRunner(project_root)
    
    # Run tests
    print("🚀 Starting Polymarket integration tests...")
    test_results = runner.run_tests(
        category=args.category,
        markers=args.markers,
        parallel=not args.no_parallel,
        gpu=not args.no_gpu,
        benchmark=args.benchmark
    )
    
    # Generate report if requested
    if args.report or test_results.get("exit_code", 1) != 0:
        runner.generate_report(test_results)
    
    # Check coverage threshold
    if not args.benchmark:  # Coverage not available in benchmark mode
        coverage_ok = runner.check_coverage_threshold(args.coverage_threshold)
        
        if not coverage_ok and test_results.get("exit_code", 0) == 0:
            test_results["exit_code"] = 1
    
    # Exit with test result code
    sys.exit(test_results.get("exit_code", 1))


if __name__ == "__main__":
    main()