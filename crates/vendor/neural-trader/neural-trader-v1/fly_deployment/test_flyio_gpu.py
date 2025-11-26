#!/usr/bin/env python3
"""
Test script for Fly.io GPU optimization validation
Validates GPU setup, performance, and deployment readiness
"""

import os
import sys
import time
import json
import asyncio
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append('/workspaces/ai-news-trader')

from fly_deployment.flyio_performance_optimizer import FlyIOPerformanceOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlyIOGPUTester:
    """Test suite for Fly.io GPU optimization validation."""
    
    def __init__(self):
        self.optimizer = FlyIOPerformanceOptimizer()
        self.test_results = {}
    
    def test_gpu_detection(self) -> Dict[str, Any]:
        """Test GPU detection and configuration."""
        logger.info("🔍 Testing GPU detection...")
        
        gpu_info = self.optimizer.gpu_info
        
        result = {
            "test": "gpu_detection",
            "passed": False,
            "details": gpu_info
        }
        
        if gpu_info["available"]:
            if gpu_info["device_count"] > 0:
                device = gpu_info["devices"][0]
                
                # Check for A100 or suitable GPU
                suitable_gpus = ["A100", "V100", "RTX"]
                is_suitable = any(gpu_name in device["name"] for gpu_name in suitable_gpus)
                
                result["passed"] = is_suitable
                result["gpu_name"] = device["name"]
                result["memory_gb"] = device["total_memory_gb"]
                result["tensor_cores"] = device["tensor_cores"]
                
                if is_suitable:
                    logger.info(f"✅ Suitable GPU detected: {device['name']}")
                else:
                    logger.warning(f"⚠️  GPU detected but may not be optimal: {device['name']}")
            else:
                logger.error("❌ No GPU devices found")
        else:
            logger.error("❌ CUDA not available")
        
        return result
    
    def test_profile_selection(self) -> Dict[str, Any]:
        """Test optimization profile selection."""
        logger.info("🎛️  Testing profile selection...")
        
        test_workloads = ["real_time", "batch", "balanced", "memory_intensive"]
        results = {}
        
        for workload in test_workloads:
            try:
                profile = self.optimizer.select_optimal_profile(workload)
                results[workload] = {
                    "profile_name": profile.name,
                    "batch_size": profile.batch_size,
                    "precision": profile.precision,
                    "tensorrt_enabled": profile.tensorrt_enabled,
                    "gpu_memory_fraction": profile.gpu_memory_fraction
                }
                logger.info(f"✅ Profile selected for {workload}: {profile.name}")
            except Exception as e:
                results[workload] = {"error": str(e)}
                logger.error(f"❌ Failed to select profile for {workload}: {e}")
        
        return {
            "test": "profile_selection", 
            "passed": len([r for r in results.values() if "error" not in r]) == len(test_workloads),
            "details": results
        }
    
    def test_optimization_application(self) -> Dict[str, Any]:
        """Test applying optimizations."""
        logger.info("⚙️  Testing optimization application...")
        
        try:
            profile = self.optimizer.select_optimal_profile("balanced")
            
            # Apply optimizations
            self.optimizer.apply_gpu_optimizations(profile)
            self.optimizer.apply_system_optimizations(profile)
            
            # Check environment variables are set
            env_checks = {
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
                "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF")
            }
            
            passed = all(value is not None for value in env_checks.values())
            
            if passed:
                logger.info("✅ Optimizations applied successfully")
            else:
                logger.error("❌ Some optimizations failed to apply")
            
            return {
                "test": "optimization_application",
                "passed": passed,
                "details": {
                    "profile_used": profile.name,
                    "environment_variables": env_checks
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Optimization application failed: {e}")
            return {
                "test": "optimization_application",
                "passed": False,
                "error": str(e)
            }
    
    def test_performance_benchmark(self) -> Dict[str, Any]:
        """Test performance benchmarking."""
        logger.info("📊 Testing performance benchmarking...")
        
        try:
            profile = self.optimizer.select_optimal_profile("balanced")
            benchmark_results = self.optimizer.benchmark_configuration(profile)
            
            # Check if we got meaningful results
            required_metrics = ["cpu_matmul_1000_ms", "memory_total_gb"]
            if self.optimizer.gpu_info["available"]:
                required_metrics.extend(["gpu_matmul_512_ms", "gpu_utilization"])
            
            has_required_metrics = all(metric in benchmark_results for metric in required_metrics)
            
            if has_required_metrics:
                logger.info("✅ Performance benchmarking successful")
                
                # Log key metrics
                if "cpu_matmul_1000_ms" in benchmark_results:
                    logger.info(f"   CPU performance: {benchmark_results['cpu_matmul_1000_ms']:.2f}ms")
                if "gpu_matmul_512_ms" in benchmark_results:
                    logger.info(f"   GPU performance: {benchmark_results['gpu_matmul_512_ms']:.2f}ms")
                
            else:
                logger.error("❌ Benchmark results incomplete")
            
            return {
                "test": "performance_benchmark",
                "passed": has_required_metrics,
                "details": benchmark_results
            }
            
        except Exception as e:
            logger.error(f"❌ Performance benchmarking failed: {e}")
            return {
                "test": "performance_benchmark",
                "passed": False,
                "error": str(e)
            }
    
    def test_pytorch_integration(self) -> Dict[str, Any]:
        """Test PyTorch GPU integration."""
        logger.info("🔥 Testing PyTorch GPU integration...")
        
        try:
            import torch
            
            # Basic PyTorch tests
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            result = {
                "test": "pytorch_integration",
                "passed": cuda_available,
                "details": {
                    "cuda_available": cuda_available,
                    "device_count": device_count,
                    "pytorch_version": torch.__version__
                }
            }
            
            if cuda_available:
                # Test basic tensor operations
                device = torch.device("cuda:0")
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                
                start_time = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                operation_time = (time.time() - start_time) * 1000
                
                result["details"]["operation_time_ms"] = operation_time
                result["details"]["gpu_name"] = torch.cuda.get_device_name(0)
                result["details"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"✅ PyTorch GPU test passed ({operation_time:.2f}ms)")
            else:
                logger.warning("⚠️  PyTorch GPU not available, CPU mode will be used")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ PyTorch integration test failed: {e}")
            return {
                "test": "pytorch_integration",
                "passed": False,
                "error": str(e)
            }
    
    def test_neural_forecast_imports(self) -> Dict[str, Any]:
        """Test NeuralForecast library imports."""
        logger.info("🧠 Testing NeuralForecast imports...")
        
        try:
            # Test core imports
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NHITS
            
            # Test model creation
            model = NHITS(
                h=24,
                input_size=168,
                max_steps=1,
                enable_progress_bar=False
            )
            
            logger.info("✅ NeuralForecast imports and model creation successful")
            
            return {
                "test": "neural_forecast_imports",
                "passed": True,
                "details": {
                    "model_type": "NHITS",
                    "horizon": 24,
                    "input_size": 168
                }
            }
            
        except Exception as e:
            logger.error(f"❌ NeuralForecast import test failed: {e}")
            return {
                "test": "neural_forecast_imports", 
                "passed": False,
                "error": str(e)
            }
    
    def test_fly_io_environment(self) -> Dict[str, Any]:
        """Test Fly.io specific environment setup."""
        logger.info("🛩️  Testing Fly.io environment...")
        
        # Check for Fly.io specific environment variables
        fly_env_vars = {
            "FLY_APP_NAME": os.getenv("FLY_APP_NAME"),
            "FLY_REGION": os.getenv("FLY_REGION"),
            "FLY_ALLOC_ID": os.getenv("FLY_ALLOC_ID")
        }
        
        # Check neural forecast specific variables
        neural_env_vars = {
            "NEURAL_FORECAST_GPU_ENABLED": os.getenv("NEURAL_FORECAST_GPU_ENABLED"),
            "FLYIO_GPU_TYPE": os.getenv("FLYIO_GPU_TYPE"),
            "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF")
        }
        
        is_fly_env = any(var is not None for var in fly_env_vars.values())
        has_neural_config = any(var is not None for var in neural_env_vars.values())
        
        if is_fly_env:
            logger.info("✅ Running in Fly.io environment")
        else:
            logger.info("ℹ️  Not running in Fly.io (local testing)")
        
        return {
            "test": "fly_io_environment",
            "passed": True,  # Always pass, just informational
            "details": {
                "is_fly_environment": is_fly_env,
                "has_neural_config": has_neural_config,
                "fly_env_vars": fly_env_vars,
                "neural_env_vars": neural_env_vars
            }
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        logger.info("🚀 Starting Fly.io GPU optimization test suite...")
        
        tests = [
            self.test_gpu_detection,
            self.test_pytorch_integration,
            self.test_neural_forecast_imports,
            self.test_profile_selection,
            self.test_optimization_application,
            self.test_performance_benchmark,
            self.test_fly_io_environment
        ]
        
        results = []
        passed_tests = 0
        
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                if result.get("passed", False):
                    passed_tests += 1
            except Exception as e:
                results.append({
                    "test": test_func.__name__,
                    "passed": False,
                    "error": str(e)
                })
        
        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            "test_suite": "fly_io_gpu_optimization",
            "timestamp": time.time(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 80 else "FAIL",
            "results": results
        }
        
        # Log summary
        logger.info(f"📊 Test Suite Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {total_tests - passed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Overall Status: {summary['overall_status']}")
        
        if success_rate >= 80:
            logger.info("✅ Fly.io GPU optimization test suite PASSED")
        else:
            logger.error("❌ Fly.io GPU optimization test suite FAILED")
        
        return summary

def main():
    """Main entry point for GPU testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fly.io GPU Optimization Tester")
    parser.add_argument("--test", help="Run specific test")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    tester = FlyIOGPUTester()
    
    if args.test:
        # Run specific test
        test_method = getattr(tester, f"test_{args.test}", None)
        if test_method:
            result = test_method()
            print(json.dumps(result, indent=2))
        else:
            print(f"Test '{args.test}' not found")
            sys.exit(1)
    else:
        # Run all tests
        results = tester.run_all_tests()
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()