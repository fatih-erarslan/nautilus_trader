"""MCP Test Runner.

Comprehensive test runner for all MCP tests with reporting.
"""

import pytest
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPTestRunner:
    """Runner for MCP test suite."""
    
    def __init__(self, results_dir: str = "test_results"):
        """Initialize test runner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.test_modules = [
            "test_mcp_protocol",
            "test_mcp_transport", 
            "test_mcp_integration",
            "test_mcp_performance"
        ]
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all MCP tests."""
        logger.info("Starting MCP test suite...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0
            }
        }
        
        start_time = time.time()
        
        # Run each test module
        for module in self.test_modules:
            logger.info(f"\nRunning {module}...")
            module_results = self.run_test_module(module, verbose)
            results["test_results"][module] = module_results
            
            # Update summary
            results["summary"]["total_tests"] += module_results["total"]
            results["summary"]["passed"] += module_results["passed"]
            results["summary"]["failed"] += module_results["failed"]
            results["summary"]["skipped"] += module_results["skipped"]
            results["summary"]["errors"] += module_results["errors"]
        
        results["total_time"] = time.time() - start_time
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def run_test_module(self, module_name: str, verbose: bool = True) -> Dict[str, Any]:
        """Run a specific test module."""
        pytest_args = [
            f"tests/mcp/{module_name}.py",
            "-v" if verbose else "-q",
            "--tb=short",
            f"--json-report-file={self.results_dir}/{module_name}_report.json"
        ]
        
        # Run pytest
        exit_code = pytest.main(pytest_args)
        
        # Parse results
        report_file = self.results_dir / f"{module_name}_report.json"
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            
            return {
                "module": module_name,
                "total": report["summary"]["total"],
                "passed": report["summary"]["passed"],
                "failed": report["summary"]["failed"],
                "skipped": report["summary"]["skipped"],
                "errors": report["summary"]["error"],
                "duration": report["duration"],
                "exit_code": exit_code
            }
        else:
            # Fallback if report not generated
            return {
                "module": module_name,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0,
                "exit_code": exit_code
            }
    
    def run_specific_tests(self, test_pattern: str, verbose: bool = True) -> Dict[str, Any]:
        """Run tests matching a specific pattern."""
        logger.info(f"Running tests matching pattern: {test_pattern}")
        
        pytest_args = [
            "tests/mcp/",
            "-k", test_pattern,
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        duration = time.time() - start_time
        
        return {
            "pattern": test_pattern,
            "exit_code": exit_code,
            "duration": duration
        }
    
    def run_performance_tests(self, stress_tests: bool = False) -> Dict[str, Any]:
        """Run performance tests specifically."""
        logger.info("Running MCP performance tests...")
        
        if stress_tests:
            # Include stress tests
            pytest_args = [
                "tests/mcp/test_mcp_performance.py",
                "-v",
                "--tb=short",
                "-m", "not skip"  # Run even skipped stress tests
            ]
        else:
            # Normal performance tests only
            pytest_args = [
                "tests/mcp/test_mcp_performance.py",
                "-v",
                "--tb=short"
            ]
        
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        duration = time.time() - start_time
        
        return {
            "test_type": "performance",
            "stress_tests_included": stress_tests,
            "exit_code": exit_code,
            "duration": duration
        }
    
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"mcp_test_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        summary = results["summary"]
        
        print("\n" + "="*60)
        print("MCP TEST SUITE SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ({summary['passed']/summary['total_tests']*100:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Total Time: {results['total_time']:.2f}s")
        print("="*60)
        
        if summary['failed'] > 0 or summary['errors'] > 0:
            print("\nFAILED TESTS:")
            for module, module_results in results["test_results"].items():
                if module_results["failed"] > 0 or module_results["errors"] > 0:
                    print(f"  - {module}: {module_results['failed']} failed, {module_results['errors']} errors")
    
    def generate_html_report(self, results: Dict[str, Any]):
        """Generate HTML test report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .module { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .passed { color: green; }
        .failed { color: red; }
        .skipped { color: orange; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>MCP Test Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Total Time: {total_time:.2f}s</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Total Tests</th>
                <th class="passed">Passed</th>
                <th class="failed">Failed</th>
                <th class="skipped">Skipped</th>
                <th class="failed">Errors</th>
            </tr>
            <tr>
                <td>{total_tests}</td>
                <td class="passed">{passed}</td>
                <td class="failed">{failed}</td>
                <td class="skipped">{skipped}</td>
                <td class="failed">{errors}</td>
            </tr>
        </table>
    </div>
    
    <div class="modules">
        <h2>Test Modules</h2>
        {modules_html}
    </div>
</body>
</html>
"""
        
        modules_html = ""
        for module, module_results in results["test_results"].items():
            modules_html += f"""
            <div class="module">
                <h3>{module}</h3>
                <p>Total: {module_results['total']} | 
                   <span class="passed">Passed: {module_results['passed']}</span> | 
                   <span class="failed">Failed: {module_results['failed']}</span> | 
                   <span class="skipped">Skipped: {module_results['skipped']}</span> | 
                   Duration: {module_results['duration']:.2f}s</p>
            </div>
            """
        
        html_content = html_template.format(
            timestamp=results["timestamp"],
            total_time=results["total_time"],
            total_tests=results["summary"]["total_tests"],
            passed=results["summary"]["passed"],
            failed=results["summary"]["failed"],
            skipped=results["summary"]["skipped"],
            errors=results["summary"]["errors"],
            modules_html=modules_html
        )
        
        report_file = self.results_dir / "mcp_test_report.html"
        with open(report_file, "w") as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_file}")


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Test Suite Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--module", type=str, help="Run specific test module")
    parser.add_argument("--pattern", type=str, help="Run tests matching pattern")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--stress", action="store_true", help="Include stress tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    
    args = parser.parse_args()
    
    runner = MCPTestRunner()
    
    if args.all:
        results = runner.run_all_tests(verbose=args.verbose)
        if args.html:
            runner.generate_html_report(results)
    elif args.module:
        results = runner.run_test_module(args.module, verbose=args.verbose)
        print(f"\nModule {args.module} results: {results}")
    elif args.pattern:
        results = runner.run_specific_tests(args.pattern, verbose=args.verbose)
        print(f"\nPattern '{args.pattern}' results: {results}")
    elif args.performance:
        results = runner.run_performance_tests(stress_tests=args.stress)
        print(f"\nPerformance test results: {results}")
    else:
        # Default: run all tests
        results = runner.run_all_tests(verbose=args.verbose)
        if args.html:
            runner.generate_html_report(results)


if __name__ == "__main__":
    main()