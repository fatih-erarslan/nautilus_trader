"""
Comprehensive Test Runner for Fantasy Collective System
Orchestrates all test suites with detailed reporting and coverage analysis
"""

import sys
import os
import subprocess
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRunner:
    """Comprehensive test runner with reporting capabilities"""
    
    def __init__(self, verbose: bool = True, parallel: bool = False):
        self.verbose = verbose
        self.parallel = parallel
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_suites": {},
            "summary": {},
            "coverage": {},
            "performance_metrics": {},
            "security_findings": {}
        }
        self.base_dir = Path(__file__).parent
        self.src_dir = self.base_dir.parent / "src"
        
    def run_test_suite(self, suite_name: str, test_file: str, 
                      markers: Optional[List[str]] = None,
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run a specific test suite"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.base_dir / test_file),
            "-v",
            "--tb=short",
            "--strict-markers",
            f"--cov={self.src_dir}",
            "--cov-report=term-missing",
            "--cov-report=json",
            "--cov-append",
            "--json-report",
            f"--json-report-file={self.base_dir}/reports/{suite_name}_report.json"
        ]
        
        # Add markers if specified
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add parallel execution if enabled
        if self.parallel:
            cmd.extend(["-n", "auto"])
        
        start_time = time.time()
        
        try:
            # Create reports directory if it doesn't exist
            reports_dir = self.base_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Run the test
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Parse results
            suite_result = {
                "suite_name": suite_name,
                "duration": duration,
                "return_code": result.returncode,
                "passed": result.returncode == 0,
                "stdout": result.stdout if self.verbose else result.stdout[:1000],
                "stderr": result.stderr if self.verbose else result.stderr[:1000],
                "test_count": self._extract_test_count(result.stdout),
                "coverage": self._extract_coverage(result.stdout)
            }
            
            # Load detailed JSON report if available
            json_report_file = reports_dir / f"{suite_name}_report.json"
            if json_report_file.exists():
                try:
                    with open(json_report_file, 'r') as f:
                        json_report = json.load(f)
                        suite_result["detailed_results"] = json_report
                except json.JSONDecodeError:
                    pass
            
            self.results["test_suites"][suite_name] = suite_result
            
            if self.verbose:
                self._print_suite_summary(suite_result)
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            suite_result = {
                "suite_name": suite_name,
                "duration": duration,
                "return_code": -1,
                "passed": False,
                "error": f"Test suite timed out after {timeout}s",
                "test_count": {"failed": 1, "total": 1}
            }
            self.results["test_suites"][suite_name] = suite_result
            return suite_result
            
        except Exception as e:
            duration = time.time() - start_time
            suite_result = {
                "suite_name": suite_name,
                "duration": duration,
                "return_code": -1,
                "passed": False,
                "error": str(e),
                "test_count": {"failed": 1, "total": 1}
            }
            self.results["test_suites"][suite_name] = suite_result
            return suite_result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites in the comprehensive test plan"""
        print("Starting Comprehensive Fantasy Collective Test Suite")
        print(f"Test execution started at: {self.results['start_time']}")
        
        # Define test suites
        test_suites = [
            {
                "name": "Unit Tests - Core Functionality",
                "file": "test_fantasy_collective.py",
                "markers": ["unit"],
                "timeout": 600  # 10 minutes
            },
            {
                "name": "Integration Tests - System Integration",
                "file": "test_fantasy_collective.py", 
                "markers": ["integration"],
                "timeout": 900  # 15 minutes
            },
            {
                "name": "Performance Tests - Concurrent Operations",
                "file": "performance/test_concurrent_performance.py",
                "markers": ["slow"],
                "timeout": 1800  # 30 minutes
            },
            {
                "name": "Security Tests - Vulnerability Assessment",
                "file": "security/test_security_validation.py",
                "markers": None,
                "timeout": 1200  # 20 minutes
            }
        ]
        
        # Run each test suite
        total_duration = 0
        for suite_config in test_suites:
            suite_result = self.run_test_suite(
                suite_config["name"],
                suite_config["file"],
                suite_config.get("markers"),
                suite_config.get("timeout")
            )
            total_duration += suite_result["duration"]
        
        # Generate final summary
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_duration"] = total_duration
        self._generate_summary()
        self._save_results()
        
        return self.results
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run quick subset of tests for rapid feedback"""
        print("Running Quick Test Suite")
        
        # Quick test configuration
        quick_suites = [
            {
                "name": "Quick Unit Tests",
                "file": "test_fantasy_collective.py",
                "markers": ["unit", "not slow"],
                "timeout": 300
            },
            {
                "name": "Quick Security Check",
                "file": "security/test_security_validation.py",
                "markers": ["not slow"] if self.parallel else None,
                "timeout": 300
            }
        ]
        
        total_duration = 0
        for suite_config in quick_suites:
            suite_result = self.run_test_suite(
                suite_config["name"],
                suite_config["file"],
                suite_config.get("markers"),
                suite_config.get("timeout")
            )
            total_duration += suite_result["duration"]
        
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_duration"] = total_duration
        self._generate_summary()
        
        return self.results
    
    def run_security_focused(self) -> Dict[str, Any]:
        """Run security-focused test suite"""
        print("Running Security-Focused Test Suite")
        
        security_suites = [
            {
                "name": "Input Validation Security",
                "file": "security/test_security_validation.py",
                "markers": None,
                "timeout": 1200
            },
            {
                "name": "Concurrency Security",
                "file": "performance/test_concurrent_performance.py",
                "markers": ["slow"],
                "timeout": 600
            }
        ]
        
        total_duration = 0
        for suite_config in security_suites:
            suite_result = self.run_test_suite(
                suite_config["name"],
                suite_config["file"],
                suite_config.get("markers"),
                suite_config.get("timeout")
            )
            total_duration += suite_result["duration"]
        
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_duration"] = total_duration
        self._generate_summary()
        self._analyze_security_findings()
        
        return self.results
    
    def run_performance_focused(self) -> Dict[str, Any]:
        """Run performance-focused test suite"""
        print("Running Performance-Focused Test Suite")
        
        performance_suites = [
            {
                "name": "Concurrent Performance Tests",
                "file": "performance/test_concurrent_performance.py",
                "markers": ["slow"],
                "timeout": 1800
            },
            {
                "name": "Load Testing",
                "file": "test_fantasy_collective.py",
                "markers": ["slow"],
                "timeout": 900
            }
        ]
        
        total_duration = 0
        for suite_config in performance_suites:
            suite_result = self.run_test_suite(
                suite_config["name"],
                suite_config["file"],
                suite_config.get("markers"),
                suite_config.get("timeout")
            )
            total_duration += suite_result["duration"]
        
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_duration"] = total_duration
        self._generate_summary()
        self._analyze_performance_metrics()
        
        return self.results
    
    def _extract_test_count(self, output: str) -> Dict[str, int]:
        """Extract test count from pytest output"""
        lines = output.split('\n')
        summary_line = None
        
        for line in lines:
            if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                summary_line = line
                break
            elif line.strip().endswith('passed'):
                summary_line = line
                break
        
        if not summary_line:
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        
        # Parse the summary line
        test_counts = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        
        import re
        
        # Match patterns like "10 passed", "2 failed", etc.
        patterns = {
            "passed": r'(\d+)\s+passed',
            "failed": r'(\d+)\s+failed', 
            "errors": r'(\d+)\s+error',
            "skipped": r'(\d+)\s+skipped'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, summary_line)
            if match:
                test_counts[key] = int(match.group(1))
        
        test_counts["total"] = sum(test_counts[k] for k in ["passed", "failed", "errors", "skipped"])
        
        return test_counts
    
    def _extract_coverage(self, output: str) -> Dict[str, Any]:
        """Extract coverage information from pytest output"""
        lines = output.split('\n')
        coverage_data = {"total_coverage": 0, "files": {}}
        
        in_coverage_section = False
        for line in lines:
            if "TOTAL" in line and "%" in line:
                # Extract total coverage percentage
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            coverage_data["total_coverage"] = int(part.replace('%', ''))
                        except ValueError:
                            pass
                        break
            
            # Extract individual file coverage if detailed report is available
            if line.startswith("src/") and "%" in line:
                parts = line.split()
                if len(parts) >= 4:
                    filename = parts[0]
                    coverage_percent = parts[-1]
                    if coverage_percent.endswith('%'):
                        try:
                            coverage_data["files"][filename] = int(coverage_percent.replace('%', ''))
                        except ValueError:
                            pass
        
        return coverage_data
    
    def _print_suite_summary(self, suite_result: Dict[str, Any]):
        """Print summary for a test suite"""
        name = suite_result["suite_name"]
        duration = suite_result["duration"]
        passed = suite_result["passed"]
        test_count = suite_result.get("test_count", {})
        coverage = suite_result.get("coverage", {})
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n{status} {name}")
        print(f"Duration: {duration:.1f}s")
        
        if test_count.get("total", 0) > 0:
            print(f"Tests: {test_count.get('passed', 0)} passed, "
                  f"{test_count.get('failed', 0)} failed, "
                  f"{test_count.get('skipped', 0)} skipped, "
                  f"{test_count.get('total', 0)} total")
        
        if coverage.get("total_coverage", 0) > 0:
            print(f"Coverage: {coverage['total_coverage']}%")
    
    def _generate_summary(self):
        """Generate overall test summary"""
        suites = self.results["test_suites"]
        
        total_tests = sum(s.get("test_count", {}).get("total", 0) for s in suites.values())
        passed_tests = sum(s.get("test_count", {}).get("passed", 0) for s in suites.values())
        failed_tests = sum(s.get("test_count", {}).get("failed", 0) for s in suites.values())
        skipped_tests = sum(s.get("test_count", {}).get("skipped", 0) for s in suites.values())
        
        passed_suites = sum(1 for s in suites.values() if s.get("passed", False))
        total_suites = len(suites)
        
        # Calculate overall coverage
        coverages = [s.get("coverage", {}).get("total_coverage", 0) 
                    for s in suites.values() if s.get("coverage", {}).get("total_coverage", 0) > 0]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0
        
        self.results["summary"] = {
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": total_suites - passed_suites,
            "suite_success_rate": (passed_suites / total_suites) * 100 if total_suites > 0 else 0,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "test_success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "average_coverage": avg_coverage,
            "overall_status": "PASSED" if passed_suites == total_suites else "FAILED"
        }
    
    def _analyze_security_findings(self):
        """Analyze security test results for findings"""
        security_findings = {
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "findings": []
        }
        
        # Look for security-related failures in test results
        for suite_name, suite_result in self.results["test_suites"].items():
            if "security" in suite_name.lower():
                if not suite_result.get("passed", False):
                    # Parse stderr for security-related failures
                    stderr = suite_result.get("stderr", "")
                    if "injection" in stderr.lower():
                        security_findings["critical_issues"] += 1
                        security_findings["findings"].append({
                            "severity": "critical",
                            "category": "injection",
                            "description": "Potential injection vulnerability detected"
                        })
                    elif "xss" in stderr.lower():
                        security_findings["high_issues"] += 1
                        security_findings["findings"].append({
                            "severity": "high", 
                            "category": "xss",
                            "description": "Potential XSS vulnerability detected"
                        })
                    elif "access control" in stderr.lower():
                        security_findings["high_issues"] += 1
                        security_findings["findings"].append({
                            "severity": "high",
                            "category": "access_control", 
                            "description": "Access control issue detected"
                        })
        
        self.results["security_findings"] = security_findings
    
    def _analyze_performance_metrics(self):
        """Analyze performance test results for metrics"""
        performance_metrics = {
            "throughput": {},
            "response_times": {},
            "resource_usage": {},
            "scalability": {}
        }
        
        for suite_name, suite_result in self.results["test_suites"].items():
            if "performance" in suite_name.lower():
                stdout = suite_result.get("stdout", "")
                
                # Extract performance metrics from output
                import re
                
                # Look for throughput metrics
                throughput_match = re.search(r'(\d+\.?\d*)\s+ops?/sec', stdout)
                if throughput_match:
                    performance_metrics["throughput"][suite_name] = float(throughput_match.group(1))
                
                # Look for response time metrics
                response_time_match = re.search(r'(\d+\.?\d*)\s*s.*response', stdout)
                if response_time_match:
                    performance_metrics["response_times"][suite_name] = float(response_time_match.group(1))
                
                # Look for memory usage metrics
                memory_match = re.search(r'(\d+\.?\d*)\s*MB.*memory', stdout)
                if memory_match:
                    performance_metrics["resource_usage"][suite_name] = float(memory_match.group(1))
        
        self.results["performance_metrics"] = performance_metrics
    
    def _save_results(self):
        """Save detailed results to file"""
        reports_dir = self.base_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"comprehensive_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {report_file}")
    
    def print_final_summary(self):
        """Print final test summary"""
        summary = self.results.get("summary", {})
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(f"Total Duration: {self.results.get('total_duration', 0):.1f}s")
        
        print(f"\nTest Suites:")
        print(f"  ✅ Passed: {summary.get('passed_suites', 0)}/{summary.get('total_suites', 0)} "
              f"({summary.get('suite_success_rate', 0):.1f}%)")
        
        print(f"\nIndividual Tests:")
        print(f"  ✅ Passed: {summary.get('passed_tests', 0)}")
        print(f"  ❌ Failed: {summary.get('failed_tests', 0)}")
        print(f"  ⚠️  Skipped: {summary.get('skipped_tests', 0)}")
        print(f"  📊 Total: {summary.get('total_tests', 0)}")
        print(f"  📈 Success Rate: {summary.get('test_success_rate', 0):.1f}%")
        
        print(f"\nCode Coverage:")
        print(f"  📊 Average Coverage: {summary.get('average_coverage', 0):.1f}%")
        
        # Security findings
        security_findings = self.results.get("security_findings", {})
        if security_findings:
            total_issues = (security_findings.get("critical_issues", 0) + 
                           security_findings.get("high_issues", 0) + 
                           security_findings.get("medium_issues", 0) + 
                           security_findings.get("low_issues", 0))
            
            if total_issues > 0:
                print(f"\n🔒 Security Findings:")
                print(f"  🚨 Critical: {security_findings.get('critical_issues', 0)}")
                print(f"  ⚠️  High: {security_findings.get('high_issues', 0)}")
                print(f"  📝 Medium: {security_findings.get('medium_issues', 0)}")
                print(f"  ℹ️  Low: {security_findings.get('low_issues', 0)}")
            else:
                print(f"\n🔒 Security: No issues detected")
        
        # Performance metrics
        performance_metrics = self.results.get("performance_metrics", {})
        if performance_metrics.get("throughput"):
            avg_throughput = sum(performance_metrics["throughput"].values()) / len(performance_metrics["throughput"])
            print(f"\n⚡ Performance:")
            print(f"  📊 Average Throughput: {avg_throughput:.1f} ops/sec")
        
        print(f"\n{'='*80}")


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="Fantasy Collective Comprehensive Test Runner")
    
    parser.add_argument("--mode", choices=["all", "quick", "security", "performance"], 
                       default="all", help="Test mode to run")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true",
                       help="Run tests in parallel")
    parser.add_argument("--output", "-o", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(verbose=args.verbose, parallel=args.parallel)
    
    # Run appropriate test suite
    if args.mode == "all":
        results = runner.run_all_tests()
    elif args.mode == "quick":
        results = runner.run_quick_tests()
    elif args.mode == "security":
        results = runner.run_security_focused()
    elif args.mode == "performance":
        results = runner.run_performance_focused()
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    # Print summary
    runner.print_final_summary()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    
    # Return appropriate exit code
    summary = results.get("summary", {})
    overall_status = summary.get("overall_status", "FAILED")
    
    return 0 if overall_status == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())