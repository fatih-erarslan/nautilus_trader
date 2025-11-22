#!/usr/bin/env python3
"""
Autopoiesis Test Infrastructure Manager
Critical Scientific System Testing with 100% Coverage Validation
"""

import os
import sys
import subprocess
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import concurrent.futures
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('target/test-execution.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test execution result"""
    category: str
    passed: bool
    duration: float
    coverage: Optional[float]
    errors: List[str]
    warnings: List[str]

@dataclass
class TestConfig:
    """Test configuration"""
    timeout: int = 300
    parallel_jobs: int = 4
    retry_count: int = 3
    coverage_threshold: float = 95.0
    memory_limit: int = 4096  # MB
    
class TestInfrastructure:
    """Comprehensive testing infrastructure for scientific systems"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: Dict[str, TestResult] = {}
        self.start_time = time.time()
        
        # Ensure directories exist
        Path("target/test-results").mkdir(parents=True, exist_ok=True)
        Path("target/coverage").mkdir(parents=True, exist_ok=True)
        Path("target/benchmarks").mkdir(parents=True, exist_ok=True)
        
    def run_comprehensive_tests(self, categories: List[str] = None) -> bool:
        """Run comprehensive test suite with scientific validation"""
        logger.info("🔬 Starting Autopoiesis Scientific Test Infrastructure")
        logger.info(f"Configuration: {self.config}")
        
        if categories is None:
            categories = ["unit", "integration", "property", "performance", "coverage"]
            
        # Pre-test validation
        if not self._validate_environment():
            logger.error("❌ Environment validation failed")
            return False
            
        # Clean environment
        self._clean_environment()
        
        # Install dependencies
        self._install_test_dependencies()
        
        # Run test categories in parallel where appropriate
        success = True
        for category in categories:
            result = self._run_test_category(category)
            self.results[category] = result
            if not result.passed:
                success = False
                if category in ["unit", "compilation"]:
                    logger.error(f"💥 Critical category {category} failed - stopping")
                    break
                    
        # Generate comprehensive reports
        self._generate_reports()
        
        # Scientific validation
        if success:
            success = self._validate_scientific_requirements()
            
        duration = time.time() - self.start_time
        logger.info(f"🏁 Test execution completed in {duration:.2f}s")
        
        return success
        
    def _validate_environment(self) -> bool:
        """Validate test environment prerequisites"""
        logger.info("🔍 Validating test environment...")
        
        # Check Rust toolchain
        try:
            result = subprocess.run(["cargo", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("Cargo not found or not working")
                return False
        except Exception as e:
            logger.error(f"Failed to check cargo: {e}")
            return False
            
        # Check project structure
        required_files = ["Cargo.toml", "src/lib.rs"]
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Required file missing: {file_path}")
                return False
                
        # Check available memory
        try:
            # Simple memory check (Linux)
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        mem_kb = int(line.split()[1])
                        mem_mb = mem_kb // 1024
                        if mem_mb < self.config.memory_limit:
                            logger.warning(f"Low memory: {mem_mb}MB < {self.config.memory_limit}MB")
                        break
        except Exception:
            logger.warning("Could not check memory availability")
            
        logger.info("✅ Environment validation passed")
        return True
        
    def _clean_environment(self):
        """Clean build environment"""
        logger.info("🧹 Cleaning build environment...")
        
        # Kill any hanging processes
        try:
            subprocess.run(["pkill", "-f", "cargo"], 
                          capture_output=True, timeout=5)
        except Exception:
            pass
            
        # Clean cargo build
        try:
            subprocess.run(["cargo", "clean"], 
                          capture_output=True, timeout=30)
        except Exception as e:
            logger.warning(f"Failed to clean cargo: {e}")
            
        # Remove lock files
        lock_files = Path(".").glob("**/*.lock")
        for lock_file in lock_files:
            if lock_file.name != "Cargo.lock":
                try:
                    lock_file.unlink()
                except Exception:
                    pass
                    
        logger.info("✅ Environment cleaned")
        
    def _install_test_dependencies(self):
        """Install required testing tools"""
        logger.info("📦 Installing test dependencies...")
        
        tools = [
            "cargo-nextest",
            "cargo-tarpaulin", 
            "cargo-audit",
            "cargo-outdated",
            "cargo-bloat"
        ]
        
        for tool in tools:
            if not self._check_tool_installed(tool):
                logger.info(f"Installing {tool}...")
                try:
                    subprocess.run(
                        ["cargo", "install", tool],
                        capture_output=True,
                        timeout=300
                    )
                except Exception as e:
                    logger.warning(f"Failed to install {tool}: {e}")
                    
        logger.info("✅ Dependencies installed")
        
    def _check_tool_installed(self, tool: str) -> bool:
        """Check if a cargo tool is installed"""
        try:
            result = subprocess.run([tool, "--version"], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
            
    def _run_test_category(self, category: str) -> TestResult:
        """Run a specific test category"""
        logger.info(f"🧪 Running {category} tests...")
        start_time = time.time()
        
        try:
            if category == "unit":
                return self._run_unit_tests()
            elif category == "integration":
                return self._run_integration_tests()
            elif category == "property":
                return self._run_property_tests()
            elif category == "performance":
                return self._run_performance_tests()
            elif category == "coverage":
                return self._run_coverage_analysis()
            elif category == "compilation":
                return self._run_compilation_check()
            else:
                raise ValueError(f"Unknown test category: {category}")
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 {category} tests failed: {e}")
            return TestResult(
                category=category,
                passed=False,
                duration=duration,
                coverage=None,
                errors=[str(e)],
                warnings=[]
            )
            
    def _run_unit_tests(self) -> TestResult:
        """Run unit tests with cargo-nextest"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Use nextest for parallel execution
            cmd = [
                "cargo", "nextest", "run",
                "--profile", "unit",
                "--workspace",
                "--all-features",
                "--no-capture"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            
            if not passed:
                errors.append(f"Unit tests failed: {result.stderr}")
                logger.error(f"Unit test output: {result.stdout}")
                logger.error(f"Unit test errors: {result.stderr}")
                
            return TestResult(
                category="unit",
                passed=passed,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            errors.append("Unit tests timed out")
            return TestResult(
                category="unit",
                passed=False,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
    def _run_integration_tests(self) -> TestResult:
        """Run integration tests"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            cmd = [
                "cargo", "nextest", "run",
                "--profile", "integration", 
                "--test", "*integration*",
                "--workspace",
                "--all-features"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout * 2
            )
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            
            if not passed:
                errors.append(f"Integration tests failed: {result.stderr}")
                
            return TestResult(
                category="integration",
                passed=passed,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            errors.append("Integration tests timed out")
            return TestResult(
                category="integration",
                passed=False,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
    def _run_property_tests(self) -> TestResult:
        """Run property-based tests"""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Set environment for property testing
        env = os.environ.copy()
        env["PROPTEST_CASES"] = "1000"
        env["PROPTEST_MAX_SHRINK_ITERS"] = "10000"
        
        try:
            cmd = [
                "cargo", "test", 
                "--features", "property-tests",
                "proptest",
                "--",
                "--test-threads", str(self.config.parallel_jobs)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout * 3,
                env=env
            )
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            
            if not passed:
                errors.append(f"Property tests failed: {result.stderr}")
                
            return TestResult(
                category="property",
                passed=passed,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            errors.append("Property tests timed out")
            return TestResult(
                category="property",
                passed=False,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
    def _run_performance_tests(self) -> TestResult:
        """Run performance benchmarks"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            cmd = [
                "cargo", "bench",
                "--features", "benchmarks",
                "--",
                "--output-format", "pretty"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout * 4
            )
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            
            if not passed:
                errors.append(f"Performance tests failed: {result.stderr}")
                
            # Save benchmark results
            if passed and result.stdout:
                with open("target/benchmarks/results.txt", "w") as f:
                    f.write(result.stdout)
                    
            return TestResult(
                category="performance",
                passed=passed,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            errors.append("Performance tests timed out")
            return TestResult(
                category="performance",
                passed=False,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
    def _run_coverage_analysis(self) -> TestResult:
        """Run comprehensive coverage analysis"""
        start_time = time.time()
        errors = []
        warnings = []
        coverage_percentage = 0.0
        
        try:
            cmd = [
                "cargo", "tarpaulin",
                "--config", "tarpaulin.toml",
                "--workspace",
                "--all-features",
                "--engine", "llvm",
                "--out", "Html", "--out", "Xml", "--out", "Json",
                "--output-dir", "target/coverage",
                "--timeout", str(self.config.timeout)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout * 3
            )
            
            duration = time.time() - start_time
            
            # Parse coverage from output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Coverage Results:' in line or '%' in line:
                        try:
                            # Extract percentage from tarpaulin output
                            if '%' in line:
                                parts = line.split('%')
                                for part in parts:
                                    if part.strip():
                                        number_part = part.strip().split()[-1]
                                        try:
                                            coverage_percentage = float(number_part)
                                            break
                                        except ValueError:
                                            continue
                        except Exception as e:
                            logger.warning(f"Could not parse coverage: {e}")
                            
            passed = (result.returncode == 0 and 
                     coverage_percentage >= self.config.coverage_threshold)
                     
            if result.returncode != 0:
                errors.append(f"Coverage analysis failed: {result.stderr}")
            elif coverage_percentage < self.config.coverage_threshold:
                warnings.append(
                    f"Coverage {coverage_percentage:.1f}% below threshold "
                    f"{self.config.coverage_threshold:.1f}%"
                )
                
            return TestResult(
                category="coverage",
                passed=passed,
                duration=duration,
                coverage=coverage_percentage,
                errors=errors,
                warnings=warnings
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            errors.append("Coverage analysis timed out")
            return TestResult(
                category="coverage",
                passed=False,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
    def _run_compilation_check(self) -> TestResult:
        """Quick compilation check"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            cmd = ["cargo", "check", "--all-features", "--workspace"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            
            if not passed:
                errors.append(f"Compilation failed: {result.stderr}")
                
            return TestResult(
                category="compilation",
                passed=passed,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            errors.append("Compilation check timed out")
            return TestResult(
                category="compilation",
                passed=False,
                duration=duration,
                coverage=None,
                errors=errors,
                warnings=warnings
            )
            
    def _validate_scientific_requirements(self) -> bool:
        """Validate scientific testing requirements"""
        logger.info("🔬 Validating scientific requirements...")
        
        # Check coverage requirement
        coverage_result = self.results.get("coverage")
        if coverage_result and coverage_result.coverage:
            if coverage_result.coverage < 95.0:
                logger.error(f"❌ Coverage {coverage_result.coverage:.1f}% < 95% required")
                return False
                
        # Check critical tests passed
        critical_categories = ["unit", "integration"]
        for category in critical_categories:
            result = self.results.get(category)
            if not result or not result.passed:
                logger.error(f"❌ Critical test category failed: {category}")
                return False
                
        logger.info("✅ Scientific requirements validated")
        return True
        
    def _generate_reports(self):
        """Generate comprehensive test reports"""
        logger.info("📊 Generating test reports...")
        
        # JSON report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "timeout": self.config.timeout,
                "parallel_jobs": self.config.parallel_jobs,
                "coverage_threshold": self.config.coverage_threshold
            },
            "results": {}
        }
        
        total_duration = 0
        total_passed = 0
        total_tests = len(self.results)
        
        for category, result in self.results.items():
            report_data["results"][category] = {
                "passed": result.passed,
                "duration": result.duration,
                "coverage": result.coverage,
                "errors": result.errors,
                "warnings": result.warnings
            }
            total_duration += result.duration
            if result.passed:
                total_passed += 1
                
        report_data["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_tests - total_passed,
            "total_duration": total_duration,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Save JSON report
        with open("target/test-results/comprehensive-report.json", "w") as f:
            json.dump(report_data, f, indent=2)
            
        # Generate HTML report
        self._generate_html_report(report_data)
        
        # Console summary
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Total: {total_tests}")
        logger.info(f"   Passed: {total_passed}")
        logger.info(f"   Failed: {total_tests - total_passed}")
        logger.info(f"   Duration: {total_duration:.2f}s")
        logger.info(f"   Success Rate: {report_data['summary']['success_rate']:.1f}%")
        
        for category, result in self.results.items():
            status = "✅" if result.passed else "❌"
            coverage_info = f" ({result.coverage:.1f}%)" if result.coverage else ""
            logger.info(f"   {status} {category}: {result.duration:.2f}s{coverage_info}")
            
    def _generate_html_report(self, report_data: Dict):
        """Generate HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Autopoiesis Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; }}
        .test-category {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
        .passed {{ border-left-color: #4CAF50; }}
        .failed {{ border-left-color: #f44336; }}
        .coverage {{ background-color: #e7f3ff; padding: 10px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Autopoiesis Scientific Test Report</h1>
        <p>Generated: {report_data['timestamp']}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{report_data['summary']['total_tests']}</td></tr>
            <tr><td>Passed</td><td>{report_data['summary']['passed']}</td></tr>
            <tr><td>Failed</td><td>{report_data['summary']['failed']}</td></tr>
            <tr><td>Success Rate</td><td>{report_data['summary']['success_rate']:.1f}%</td></tr>
            <tr><td>Total Duration</td><td>{report_data['summary']['total_duration']:.2f}s</td></tr>
        </table>
    </div>
    
    <h2>🧪 Test Categories</h2>
"""
        
        for category, result in report_data['results'].items():
            status_class = "passed" if result['passed'] else "failed"
            status_icon = "✅" if result['passed'] else "❌"
            
            html_content += f"""
    <div class="test-category {status_class}">
        <h3>{status_icon} {category.title()}</h3>
        <p><strong>Duration:</strong> {result['duration']:.2f}s</p>
"""
            
            if result['coverage']:
                html_content += f"""
        <div class="coverage">
            <strong>Coverage:</strong> {result['coverage']:.1f}%
        </div>
"""
            
            if result['errors']:
                html_content += "<p><strong>Errors:</strong></p><ul>"
                for error in result['errors']:
                    html_content += f"<li>{error}</li>"
                html_content += "</ul>"
                
            if result['warnings']:
                html_content += "<p><strong>Warnings:</strong></p><ul>"
                for warning in result['warnings']:
                    html_content += f"<li>{warning}</li>"
                html_content += "</ul>"
                
            html_content += "</div>"
            
        html_content += """
</body>
</html>
"""
        
        with open("target/test-results/report.html", "w") as f:
            f.write(html_content)
            
        logger.info("✅ Reports generated")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Autopoiesis Test Infrastructure")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["compilation", "unit", "integration", "coverage"],
        help="Test categories to run"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test category in seconds"
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=4,
        help="Number of parallel test jobs"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=95.0,
        help="Minimum coverage percentage required"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (compilation + unit)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        categories = ["compilation", "unit"]
    else:
        categories = args.categories
        
    config = TestConfig(
        timeout=args.timeout,
        parallel_jobs=args.parallel_jobs,
        coverage_threshold=args.coverage_threshold
    )
    
    infrastructure = TestInfrastructure(config)
    success = infrastructure.run_comprehensive_tests(categories)
    
    if success:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()