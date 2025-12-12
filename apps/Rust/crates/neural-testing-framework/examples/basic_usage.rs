//! Basic Usage Example for Neural Testing Framework
//! 
//! This example demonstrates how to use the neural testing framework
//! to validate neural network implementations with real data.

use neural_testing_framework::{
    NeuralTestRunner, NeuralTestConfig, TestDataConfig, PerformanceThresholds,
    HardwareTestConfig, SimulationConfig, RiskConfig, MarketRegime,
    nhits_tests::{NHITSTestSuite, NHITSConfig, StackType, ActivationType},
    cdfa_tests::{CDFATestSuite, CDFAConfig},
    gpu_tests::{GPUTestSuite, GPUTestConfig},
    real_time_simulation::RealTimeSimulationSuite,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Testing Framework - Basic Usage Example");
    
    // Example 1: Run all tests with default configuration
    println!("\n📊 Example 1: Running all tests with default configuration");
    run_all_tests_example().await?;
    
    // Example 2: Run specific NHITS tests
    println!("\n🔍 Example 2: Running NHITS-specific tests");
    run_nhits_tests_example().await?;
    
    // Example 3: Run CDFA algorithm tests
    println!("\n🔬 Example 3: Running CDFA algorithm tests");
    run_cdfa_tests_example().await?;
    
    // Example 4: Run GPU acceleration tests (if GPU available)
    println!("\n🚀 Example 4: Running GPU acceleration tests");
    if let Err(e) = run_gpu_tests_example().await {
        println!("⚠️  GPU tests skipped: {}", e);
    }
    
    // Example 5: Run real-time trading simulation
    println!("\n📈 Example 5: Running real-time trading simulation");
    run_simulation_example().await?;
    
    // Example 6: Custom configuration example
    println!("\n⚙️  Example 6: Custom configuration example");
    run_custom_config_example().await?;
    
    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Example 1: Run all tests with default configuration
async fn run_all_tests_example() -> Result<(), Box<dyn std::error::Error>> {
    let config = NeuralTestConfig::default();
    let mut runner = NeuralTestRunner::new(config);
    
    println!("🔄 Running comprehensive neural network test suite...");
    runner.run_all_tests().await?;
    
    println!("✅ All tests completed");
    Ok(())
}

/// Example 2: NHITS-specific tests
async fn run_nhits_tests_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create NHITS configuration for testing
    let nhits_config = NHITSConfig {
        input_size: 24,           // 24 hours of input data
        output_size: 12,          // 12 hour forecast
        num_stacks: 3,            // 3 hierarchical stacks
        stack_hidden_sizes: vec![128, 256, 128],
        stack_types: vec![
            StackType::Trend,                // Trend component
            StackType::Seasonality(24),      // Daily seasonality
            StackType::Residual,             // Residual component
        ],
        pooling_kernels: vec![1, 2, 4],
        num_blocks: vec![2, 2, 2],
        activation: ActivationType::ReLU,
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 50,
        dropout_rate: 0.1,
    };
    
    let mut test_suite = NHITSTestSuite::new(nhits_config);
    
    println!("🔄 Running NHITS neural network tests...");
    let results = test_suite.run_comprehensive_tests().await?;
    
    // Print results summary
    let successful_tests = results.iter().filter(|r| r.success).count();
    println!("📊 NHITS Test Results: {}/{} tests passed", successful_tests, results.len());
    
    for result in &results {
        let status = if result.success { "✅" } else { "❌" };
        println!("  {} {}: {:.2}μs latency, {:.3} accuracy", 
                status, result.test_name, 
                result.metrics.inference_latency_us, 
                result.metrics.accuracy_metrics.r2);
    }
    
    Ok(())
}

/// Example 3: CDFA algorithm tests
async fn run_cdfa_tests_example() -> Result<(), Box<dyn std::error::Error>> {
    let cdfa_config = CDFAConfig {
        num_sources: 5,                    // Test with 5 prediction sources
        diversity_threshold: 0.5,          // 50% diversity threshold
        score_weight: 0.7,                // 70% weight on accuracy vs diversity
        adaptive_fusion_enabled: true,     // Enable adaptive fusion
        real_time_mode: true,              // Test real-time capabilities
        window_size: 100,                  // 100-sample sliding window
        update_frequency_ms: 10,           // Update every 10ms
    };
    
    let mut test_suite = CDFATestSuite::new(cdfa_config);
    
    println!("🔄 Running CDFA algorithm tests...");
    let results = test_suite.run_comprehensive_tests().await?;
    
    // Print results summary
    let successful_tests = results.iter().filter(|r| r.success).count();
    println!("📊 CDFA Test Results: {}/{} tests passed", successful_tests, results.len());
    
    // Find the best performing fusion method
    if let Some(best_result) = results.iter().max_by(|a, b| 
        a.metrics.accuracy_metrics.hit_rate.unwrap_or(0.0)
            .partial_cmp(&b.metrics.accuracy_metrics.hit_rate.unwrap_or(0.0))
            .unwrap()
    ) {
        println!("🏆 Best performing test: {} with {:.1}% hit rate", 
                best_result.test_name, 
                best_result.metrics.accuracy_metrics.hit_rate.unwrap_or(0.0) * 100.0);
    }
    
    Ok(())
}

/// Example 4: GPU acceleration tests
async fn run_gpu_tests_example() -> Result<(), Box<dyn std::error::Error>> {
    let gpu_config = GPUTestConfig {
        test_cpu_baseline: true,           // Compare against CPU
        test_cuda: true,                   // Test CUDA kernels
        test_memory_transfers: true,       // Test memory bandwidth
        test_concurrent_streams: true,     // Test parallel execution
        memory_stress_levels: vec![512, 1024], // Test memory stress
        batch_sizes: vec![1, 16, 32, 64], // Different batch sizes
    };
    
    let mut test_suite = GPUTestSuite::new(gpu_config)?;
    
    println!("🔄 Running GPU acceleration tests...");
    let results = test_suite.run_comprehensive_tests().await?;
    
    // Print GPU-specific metrics
    let successful_tests = results.iter().filter(|r| r.success).count();
    println!("📊 GPU Test Results: {}/{} tests passed", successful_tests, results.len());
    
    for result in &results {
        if let Some(gpu_util) = result.hardware_utilization.gpu_utilization {
            println!("  🚀 {}: {:.1}% GPU utilization, {:.2}μs latency", 
                    result.test_name, gpu_util, result.metrics.inference_latency_us);
        }
    }
    
    Ok(())
}

/// Example 5: Real-time trading simulation
async fn run_simulation_example() -> Result<(), Box<dyn std::error::Error>> {
    let simulation_config = SimulationConfig {
        simulation_duration_s: 60,         // 1-minute simulation
        update_frequency_ms: 100,          // 100ms updates (10Hz)
        num_strategies: 3,                 // Test 3 strategies
        risk_config: RiskConfig {
            max_position_size: 0.1,        // 10% max position
            stop_loss_pct: 0.02,           // 2% stop loss
            max_drawdown_pct: 0.05,        // 5% max drawdown
            volatility_scaling: 1.0,       // No volatility scaling
        },
    };
    
    let mut test_suite = RealTimeSimulationSuite::new(simulation_config);
    
    println!("🔄 Running real-time trading simulation...");
    let results = test_suite.run_comprehensive_tests().await?;
    
    // Print trading simulation metrics
    let successful_tests = results.iter().filter(|r| r.success).count();
    println!("📊 Simulation Test Results: {}/{} tests passed", successful_tests, results.len());
    
    for result in &results {
        if let Some(sharpe) = result.metrics.accuracy_metrics.sharpe_ratio {
            println!("  📈 {}: Sharpe {:.2}, Max DD {:.1}%", 
                    result.test_name, sharpe, 
                    result.metrics.accuracy_metrics.max_drawdown.unwrap_or(0.0) * 100.0);
        }
    }
    
    Ok(())
}

/// Example 6: Custom configuration
async fn run_custom_config_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create custom configuration for high-frequency trading scenario
    let custom_config = NeuralTestConfig {
        data_config: TestDataConfig {
            num_assets: 20,                // More assets
            sequence_length: 60,           // 1 minute of data at 1-second intervals
            num_features: 8,               // OHLCV + 3 technical indicators
            forecast_horizon: 5,           // 5-second forecast
            market_regimes: vec![
                MarketRegime::HighVolatility,
                MarketRegime::Crisis,
            ],
            noise_levels: vec![0.001, 0.005], // Lower noise for HFT
        },
        performance_thresholds: PerformanceThresholds {
            max_inference_time_us: 50.0,   // Stricter latency for HFT
            max_memory_usage_mb: 512.0,    // Lower memory limit
            min_accuracy: 0.75,            // Acceptable accuracy for speed
            max_training_time_s: 120.0,    // Faster training
            min_gpu_utilization: 0.8,      // Higher GPU utilization
        },
        hardware_config: HardwareTestConfig {
            test_cpu: true,
            test_gpu: true,
            test_quantum: false,
            test_distributed: false,
            memory_stress_levels: vec![256, 512], // Lower memory stress
        },
        simulation_config: SimulationConfig {
            simulation_duration_s: 30,     // Shorter simulation
            update_frequency_ms: 10,       // 100Hz updates
            num_strategies: 5,             // More strategies
            risk_config: RiskConfig {
                max_position_size: 0.05,    // Smaller positions
                stop_loss_pct: 0.01,        // Tighter stop loss
                max_drawdown_pct: 0.02,     // Lower drawdown tolerance
                volatility_scaling: 0.5,    // Conservative scaling
            },
        },
    };
    
    let mut runner = NeuralTestRunner::new(custom_config);
    
    println!("🔄 Running tests with custom HFT configuration...");
    runner.run_all_tests().await?;
    
    println!("✅ Custom configuration tests completed");
    Ok(())
}

/// Utility function to demonstrate test result analysis
fn analyze_test_results(results: &[neural_testing_framework::NeuralTestResults]) {
    if results.is_empty() {
        println!("No test results to analyze");
        return;
    }
    
    println!("\n📊 Test Results Analysis:");
    
    // Overall statistics
    let total_tests = results.len();
    let successful_tests = results.iter().filter(|r| r.success).count();
    let success_rate = (successful_tests as f64 / total_tests as f64) * 100.0;
    
    println!("  📈 Success Rate: {:.1}% ({}/{})", success_rate, successful_tests, total_tests);
    
    // Performance statistics
    let latencies: Vec<f64> = results.iter().map(|r| r.metrics.inference_latency_us).collect();
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let max_latency = latencies.iter().fold(0.0, |a, &b| a.max(b));
    let min_latency = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    println!("  ⚡ Latency Stats:");
    println!("    - Average: {:.2}μs", avg_latency);
    println!("    - Min: {:.2}μs", min_latency);
    println!("    - Max: {:.2}μs", max_latency);
    
    // Accuracy statistics
    let accuracies: Vec<f64> = results.iter().map(|r| r.metrics.accuracy_metrics.r2).collect();
    let avg_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
    let max_accuracy = accuracies.iter().fold(0.0, |a, &b| a.max(b));
    let min_accuracy = accuracies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    println!("  🎯 Accuracy Stats (R²):");
    println!("    - Average: {:.3}", avg_accuracy);
    println!("    - Min: {:.3}", min_accuracy);
    println!("    - Max: {:.3}", max_accuracy);
    
    // Failed tests
    let failed_tests: Vec<&neural_testing_framework::NeuralTestResults> = results.iter().filter(|r| !r.success).collect();
    if !failed_tests.is_empty() {
        println!("  ❌ Failed Tests:");
        for test in failed_tests {
            println!("    - {}: {}", test.test_name, 
                    test.errors.first().unwrap_or(&"Unknown error".to_string()));
        }
    }
}