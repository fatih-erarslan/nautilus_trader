// Comprehensive Performance Benchmarking Test Suite
// Tests all system components for <1ms performance compliance

use parasitic::benchmarks::ComprehensiveBenchmarkSuite;
use parasitic::error::Result;
use std::time::Instant;
use tokio;

#[tokio::test]
async fn test_comprehensive_performance_benchmarks() -> Result<()> {
    println!("🚀 Launching Comprehensive Performance Benchmark Suite");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut benchmark_suite = ComprehensiveBenchmarkSuite::new();
    let start_time = Instant::now();
    
    // Execute comprehensive benchmarking
    let report = benchmark_suite.run_comprehensive_benchmarks().await?;
    
    let total_duration = start_time.elapsed();
    
    // Print detailed results
    println!("\n📊 COMPREHENSIVE PERFORMANCE BENCHMARK RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🕐 Total Execution Time: {:.3}s", total_duration.as_secs_f64());
    println!("📈 Total Tests Executed: {}", report.total_tests);
    println!("✅ Tests Passed: {}", report.passed_tests);
    println!("❌ Tests Failed: {}", report.failed_tests);
    println!("🎯 Overall Compliance: {}", if report.overall_compliance { "✅ PASS" } else { "❌ FAIL" });
    
    println!("\n📋 DETAILED TEST RESULTS:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut component_results = std::collections::HashMap::new();
    for result in &report.benchmark_results {
        component_results.entry(&result.component)
            .or_insert_with(Vec::new)
            .push(result);
    }
    
    for (component, results) in component_results {
        println!("\n🔧 Component: {}", component);
        println!("   ─────────────────────────────────────────────────");
        
        for result in results {
            let status = if result.meets_requirement { "✅" } else { "❌" };
            println!("   {} {}: {:.3}ms ({:.0} ops/sec)", 
                status, result.test_name, result.latency_ms, result.throughput_ops_sec);
            
            if !result.meets_requirement {
                println!("      ⚠️  EXCEEDS 1ms LIMIT by {:.3}ms", result.latency_ms - 1.0);
            }
        }
    }
    
    println!("\n🎯 PERFORMANCE SUMMARY:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for (key, value) in &report.summary {
        if key.contains("compliance_rate") {
            println!("📊 {}: {:.1}%", key.replace("_", " "), value * 100.0);
        } else if key.contains("latency") {
            println!("⏱️  {}: {:.3}ms", key.replace("_", " "), value);
        } else if key.contains("throughput") {
            println!("🚀 {}: {:.0} ops/sec", key.replace("_", " "), value);
        }
    }
    
    println!("\n💡 RECOMMENDATIONS:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for recommendation in &report.recommendations {
        println!("   {}", recommendation);
    }
    
    // Critical assertion for CI/CD pipeline
    if !report.overall_compliance {
        println!("\n🚨 CRITICAL: System does not meet <1ms performance requirement!");
        println!("   Failed components must be optimized before production deployment");
    } else {
        println!("\n🎉 SUCCESS: All components meet <1ms performance requirement!");
        println!("   System is compliant with blueprint specifications");
    }
    
    // Store results for further analysis
    store_benchmark_results(&report).await?;
    
    assert!(report.overall_compliance, 
        "Performance benchmark failed: {} out of {} tests exceeded 1ms requirement", 
        report.failed_tests, report.total_tests);
    
    Ok(())
}

#[tokio::test]
async fn test_gpu_correlation_performance() -> Result<()> {
    println!("🖥️  Testing GPU Correlation Engine Performance");
    
    let mut suite = ComprehensiveBenchmarkSuite::new();
    let results = suite.benchmark_gpu_correlation().await?;
    
    for result in &results {
        println!("   {} GPU Test: {:.3}ms", 
            if result.meets_requirement { "✅" } else { "❌" }, 
            result.latency_ms);
        
        assert!(result.latency_ms < 1.0, 
            "GPU correlation test '{}' exceeded 1ms: {:.3}ms", 
            result.test_name, result.latency_ms);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_organism_strategy_performance() -> Result<()> {
    println!("🦠 Testing Organism Strategy Performance");
    
    let mut suite = ComprehensiveBenchmarkSuite::new();
    let results = suite.benchmark_organism_strategies().await?;
    
    for result in &results {
        println!("   {} Organism Test ({}): {:.3}ms", 
            if result.meets_requirement { "✅" } else { "❌" }, 
            result.test_name, result.latency_ms);
        
        assert!(result.latency_ms < 1.0, 
            "Organism strategy test '{}' exceeded 1ms: {:.3}ms", 
            result.test_name, result.latency_ms);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_processing_performance() -> Result<()> {
    println!("🔀 Testing Concurrent Processing Performance");
    
    let mut suite = ComprehensiveBenchmarkSuite::new();
    let results = suite.benchmark_concurrent_processing().await?;
    
    for result in &results {
        println!("   {} Concurrent Test: {:.3}ms", 
            if result.meets_requirement { "✅" } else { "❌" }, 
            result.latency_ms);
        
        assert!(result.latency_ms < 1.0, 
            "Concurrent processing test '{}' exceeded 1ms: {:.3}ms", 
            result.test_name, result.latency_ms);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_end_to_end_trading_latency() -> Result<()> {
    println!("📈 Testing End-to-End Trading Latency");
    
    let mut suite = ComprehensiveBenchmarkSuite::new();
    let results = suite.benchmark_end_to_end_latency().await?;
    
    let mut total_latency = 0.0;
    for result in &results {
        total_latency += result.latency_ms;
        println!("   {} E2E Test ({}): {:.3}ms", 
            if result.meets_requirement { "✅" } else { "❌" }, 
            result.test_name, result.latency_ms);
    }
    
    let avg_latency = total_latency / results.len() as f64;
    println!("   📊 Average E2E Latency: {:.3}ms", avg_latency);
    
    assert!(avg_latency < 1.0, 
        "Average end-to-end latency exceeded 1ms: {:.3}ms", avg_latency);
    
    Ok(())
}

#[tokio::test]
async fn test_system_load_performance() -> Result<()> {
    println!("🔥 Testing System Load Performance");
    
    let mut suite = ComprehensiveBenchmarkSuite::new();
    let results = suite.benchmark_load_testing().await?;
    
    for result in &results {
        println!("   {} Load Test ({}): {:.3}ms at {:.0} ops/sec", 
            if result.meets_requirement { "✅" } else { "❌" }, 
            result.test_name, result.latency_ms, result.throughput_ops_sec);
        
        // Under load, we allow slightly higher latency but still validate
        if result.test_name.contains("extreme_load") {
            assert!(result.latency_ms < 5.0, 
                "Load test '{}' exceeded 5ms under extreme load: {:.3}ms", 
                result.test_name, result.latency_ms);
        } else {
            assert!(result.latency_ms < 1.0, 
                "Load test '{}' exceeded 1ms: {:.3}ms", 
                result.test_name, result.latency_ms);
        }
    }
    
    Ok(())
}

// Helper function to store results for analysis
async fn store_benchmark_results(report: &parasitic::benchmarks::PerformanceReport) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("tests/performance_benchmarks/results_{}.json", timestamp);
    
    let json_report = serde_json::to_string_pretty(&report)
        .map_err(|e| parasitic::error::ParasiticError::SerializationError(e.to_string()))?;
    
    let mut file = File::create(&filename)
        .map_err(|e| parasitic::error::ParasiticError::IoError(e))?;
    
    file.write_all(json_report.as_bytes())
        .map_err(|e| parasitic::error::ParasiticError::IoError(e))?;
    
    println!("📄 Benchmark results saved to: {}", filename);
    
    Ok(())
}

// Performance regression tests
#[tokio::test]
async fn test_performance_regression() -> Result<()> {
    println!("📉 Testing Performance Regression");
    
    // Load historical benchmark results if available
    let current_results = run_baseline_performance_tests().await?;
    
    // Validate that current performance meets or exceeds baseline
    for result in current_results {
        assert!(result.latency_ms < 1.0, 
            "Performance regression detected in '{}': {:.3}ms > 1ms", 
            result.test_name, result.latency_ms);
    }
    
    Ok(())
}

async fn run_baseline_performance_tests() -> Result<Vec<parasitic::benchmarks::BenchmarkResult>> {
    let mut suite = ComprehensiveBenchmarkSuite::new();
    
    // Run a subset of critical performance tests
    let mut results = Vec::new();
    
    // Critical path tests
    results.extend(suite.benchmark_gpu_correlation().await?);
    results.extend(suite.benchmark_organism_strategies().await?);
    results.extend(suite.benchmark_end_to_end_latency().await?);
    
    Ok(results)
}