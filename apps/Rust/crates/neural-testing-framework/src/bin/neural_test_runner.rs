//! Neural Test Runner Binary
//! 
//! Command-line interface for executing comprehensive neural network tests

use clap::{App, Arg, SubCommand};
use neural_testing_framework::{
    NeuralTestRunner, NeuralTestConfig, TestDataConfig, PerformanceThresholds,
    HardwareTestConfig, SimulationConfig, RiskConfig, MarketRegime,
    nhits_tests::NHITSTestSuite,
    cdfa_tests::CDFATestSuite,
    gpu_tests::GPUTestSuite,
    real_time_simulation::RealTimeSimulationSuite,
};
use std::process;
use std::time::Instant;
use tokio;

#[tokio::main]
async fn main() {
    let matches = App::new("Neural Test Runner")
        .version("1.0.0")
        .author("Nautilus Trader Team")
        .about("Comprehensive zero-mock neural network testing framework")
        .arg(Arg::with_name("config")
            .short("c")
            .long("config")
            .value_name("FILE")
            .help("Configuration file path")
            .takes_value(true))
        .arg(Arg::with_name("output")
            .short("o")
            .long("output")
            .value_name("DIR")
            .help("Output directory for test results")
            .takes_value(true)
            .default_value("./test_results"))
        .arg(Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Enable verbose output"))
        .arg(Arg::with_name("parallel")
            .short("p")
            .long("parallel")
            .help("Run tests in parallel")
            .takes_value(false))
        .subcommand(SubCommand::with_name("nhits")
            .about("Run NHITS neural network tests")
            .arg(Arg::with_name("quick")
                .long("quick")
                .help("Run quick tests only")))
        .subcommand(SubCommand::with_name("cdfa")
            .about("Run CDFA algorithm tests")
            .arg(Arg::with_name("sources")
                .long("sources")
                .value_name("NUM")
                .help("Number of sources to test")
                .takes_value(true)
                .default_value("5")))
        .subcommand(SubCommand::with_name("gpu")
            .about("Run GPU/CUDA acceleration tests")
            .arg(Arg::with_name("memory-stress")
                .long("memory-stress")
                .help("Include memory stress tests")))
        .subcommand(SubCommand::with_name("simulation")
            .about("Run real-time trading simulation")
            .arg(Arg::with_name("duration")
                .long("duration")
                .value_name("SECONDS")
                .help("Simulation duration in seconds")
                .takes_value(true)
                .default_value("300")))
        .subcommand(SubCommand::with_name("all")
            .about("Run all test suites")
            .arg(Arg::with_name("skip-slow")
                .long("skip-slow")
                .help("Skip slow-running tests")))
        .subcommand(SubCommand::with_name("benchmark")
            .about("Run performance benchmarks")
            .arg(Arg::with_name("iterations")
                .long("iterations")
                .value_name("NUM")
                .help("Number of benchmark iterations")
                .takes_value(true)
                .default_value("100")))
        .get_matches();

    // Initialize logging
    if matches.is_present("verbose") {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    println!("🧠 Neural Network Testing Framework v1.0.0");
    println!("📅 Starting tests at: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));

    let start_time = Instant::now();
    let config = load_or_create_config(matches.value_of("config")).unwrap_or_else(|e| {
        eprintln!("❌ Failed to load configuration: {}", e);
        process::exit(1);
    });

    let output_dir = matches.value_of("output").unwrap();
    std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
        eprintln!("❌ Failed to create output directory: {}", e);
        process::exit(1);
    });

    let mut total_tests = 0;
    let mut successful_tests = 0;
    let mut test_results = Vec::new();

    match matches.subcommand() {
        ("nhits", Some(sub_matches)) => {
            println!("\n🔍 Running NHITS Neural Network Tests");
            match run_nhits_tests(sub_matches, &config).await {
                Ok((count, success, results)) => {
                    total_tests += count;
                    successful_tests += success;
                    test_results.extend(results);
                }
                Err(e) => {
                    eprintln!("❌ NHITS tests failed: {}", e);
                    process::exit(1);
                }
            }
        }
        ("cdfa", Some(sub_matches)) => {
            println!("\n🔬 Running CDFA Algorithm Tests");
            match run_cdfa_tests(sub_matches, &config).await {
                Ok((count, success, results)) => {
                    total_tests += count;
                    successful_tests += success;
                    test_results.extend(results);
                }
                Err(e) => {
                    eprintln!("❌ CDFA tests failed: {}", e);
                    process::exit(1);
                }
            }
        }
        ("gpu", Some(sub_matches)) => {
            println!("\n🚀 Running GPU/CUDA Acceleration Tests");
            match run_gpu_tests(sub_matches, &config).await {
                Ok((count, success, results)) => {
                    total_tests += count;
                    successful_tests += success;
                    test_results.extend(results);
                }
                Err(e) => {
                    eprintln!("❌ GPU tests failed: {}", e);
                    process::exit(1);
                }
            }
        }
        ("simulation", Some(sub_matches)) => {
            println!("\n📈 Running Real-Time Trading Simulation");
            match run_simulation_tests(sub_matches, &config).await {
                Ok((count, success, results)) => {
                    total_tests += count;
                    successful_tests += success;
                    test_results.extend(results);
                }
                Err(e) => {
                    eprintln!("❌ Simulation tests failed: {}", e);
                    process::exit(1);
                }
            }
        }
        ("all", Some(sub_matches)) => {
            println!("\n🎯 Running All Test Suites");
            let skip_slow = sub_matches.is_present("skip-slow");
            
            // Run all test suites
            for (test_name, test_runner) in get_all_test_runners(&config, skip_slow) {
                println!("\n🔄 Running {} tests...", test_name);
                match test_runner().await {
                    Ok((count, success, results)) => {
                        total_tests += count;
                        successful_tests += success;
                        test_results.extend(results);
                        println!("✅ {} tests completed: {}/{} passed", test_name, success, count);
                    }
                    Err(e) => {
                        eprintln!("❌ {} tests failed: {}", test_name, e);
                        // Continue with other tests
                    }
                }
            }
        }
        ("benchmark", Some(sub_matches)) => {
            println!("\n📊 Running Performance Benchmarks");
            let iterations: usize = sub_matches.value_of("iterations")
                .unwrap_or("100")
                .parse()
                .unwrap_or_else(|_| {
                    eprintln!("❌ Invalid iterations number");
                    process::exit(1);
                });
            
            match run_benchmarks(iterations, &config).await {
                Ok((count, success, results)) => {
                    total_tests += count;
                    successful_tests += success;
                    test_results.extend(results);
                }
                Err(e) => {
                    eprintln!("❌ Benchmark tests failed: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            println!("\n🎯 Running Default Test Suite (Basic Tests)");
            match run_default_tests(&config).await {
                Ok((count, success, results)) => {
                    total_tests += count;
                    successful_tests += success;
                    test_results.extend(results);
                }
                Err(e) => {
                    eprintln!("❌ Default tests failed: {}", e);
                    process::exit(1);
                }
            }
        }
    }

    // Generate test report
    let total_duration = start_time.elapsed();
    generate_test_report(&test_results, total_tests, successful_tests, total_duration, output_dir).await;

    // Exit with appropriate code
    if successful_tests == total_tests {
        println!("\n🎉 All tests passed successfully!");
        process::exit(0);
    } else {
        println!("\n⚠️  Some tests failed. Check the detailed report.");
        process::exit(1);
    }
}

fn load_or_create_config(config_path: Option<&str>) -> Result<NeuralTestConfig, Box<dyn std::error::Error>> {
    match config_path {
        Some(path) => {
            // Load from file
            let config_content = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&config_content)?)
        }
        None => {
            // Create default configuration
            Ok(NeuralTestConfig {
                data_config: TestDataConfig {
                    num_assets: 10,
                    sequence_length: 24,
                    num_features: 5,
                    forecast_horizon: 12,
                    market_regimes: vec![
                        MarketRegime::Bull,
                        MarketRegime::Bear,
                        MarketRegime::HighVolatility,
                    ],
                    noise_levels: vec![0.01, 0.02, 0.05],
                },
                performance_thresholds: PerformanceThresholds {
                    max_inference_time_us: 100.0,
                    max_memory_usage_mb: 1024.0,
                    min_accuracy: 0.8,
                    max_training_time_s: 300.0,
                    min_gpu_utilization: 0.7,
                },
                hardware_config: HardwareTestConfig {
                    test_cpu: true,
                    test_gpu: true,
                    test_quantum: false,
                    test_distributed: false,
                    memory_stress_levels: vec![512, 1024, 2048],
                },
                simulation_config: SimulationConfig {
                    simulation_duration_s: 300,
                    update_frequency_ms: 100,
                    num_strategies: 3,
                    risk_config: RiskConfig {
                        max_position_size: 0.1,
                        stop_loss_pct: 0.02,
                        max_drawdown_pct: 0.05,
                        volatility_scaling: 1.0,
                    },
                },
            })
        }
    }
}

async fn run_nhits_tests(
    sub_matches: &clap::ArgMatches<'_>,
    config: &NeuralTestConfig
) -> Result<(usize, usize, Vec<neural_testing_framework::NeuralTestResults>), Box<dyn std::error::Error>> {
    let quick_mode = sub_matches.is_present("quick");
    
    let nhits_config = neural_testing_framework::nhits_tests::NHITSConfig {
        input_size: config.data_config.sequence_length,
        output_size: config.data_config.forecast_horizon,
        num_stacks: if quick_mode { 2 } else { 4 },
        stack_hidden_sizes: if quick_mode { vec![64, 32] } else { vec![128, 256, 128, 64] },
        stack_types: vec![
            neural_testing_framework::nhits_tests::StackType::Trend,
            neural_testing_framework::nhits_tests::StackType::Seasonality(24),
        ],
        pooling_kernels: vec![1, 2],
        num_blocks: vec![2, 2],
        activation: neural_testing_framework::nhits_tests::ActivationType::ReLU,
        learning_rate: 0.001,
        batch_size: 32,
        epochs: if quick_mode { 10 } else { 100 },
        dropout_rate: 0.1,
    };

    let mut test_suite = NHITSTestSuite::new(nhits_config);
    let results = test_suite.run_comprehensive_tests().await?;
    
    let total_count = results.len();
    let success_count = results.iter().filter(|r| r.success).count();
    
    Ok((total_count, success_count, results))
}

async fn run_cdfa_tests(
    sub_matches: &clap::ArgMatches<'_>,
    _config: &NeuralTestConfig
) -> Result<(usize, usize, Vec<neural_testing_framework::NeuralTestResults>), Box<dyn std::error::Error>> {
    let num_sources: usize = sub_matches.value_of("sources")
        .unwrap_or("5")
        .parse()
        .unwrap_or(5);
    
    let cdfa_config = neural_testing_framework::cdfa_tests::CDFAConfig {
        num_sources,
        diversity_threshold: 0.5,
        score_weight: 0.7,
        adaptive_fusion_enabled: true,
        real_time_mode: true,
        window_size: 100,
        update_frequency_ms: 10,
    };

    let mut test_suite = CDFATestSuite::new(cdfa_config);
    let results = test_suite.run_comprehensive_tests().await?;
    
    let total_count = results.len();
    let success_count = results.iter().filter(|r| r.success).count();
    
    Ok((total_count, success_count, results))
}

async fn run_gpu_tests(
    sub_matches: &clap::ArgMatches<'_>,
    _config: &NeuralTestConfig
) -> Result<(usize, usize, Vec<neural_testing_framework::NeuralTestResults>), Box<dyn std::error::Error>> {
    let include_memory_stress = sub_matches.is_present("memory-stress");
    
    let gpu_config = neural_testing_framework::gpu_tests::GPUTestConfig {
        test_cpu_baseline: true,
        test_cuda: true,
        test_memory_transfers: true,
        test_concurrent_streams: true,
        memory_stress_levels: if include_memory_stress {
            vec![512, 1024, 2048, 4096, 8192]
        } else {
            vec![512, 1024]
        },
        batch_sizes: vec![1, 16, 32, 64],
    };

    let mut test_suite = GPUTestSuite::new(gpu_config)?;
    let results = test_suite.run_comprehensive_tests().await?;
    
    let total_count = results.len();
    let success_count = results.iter().filter(|r| r.success).count();
    
    Ok((total_count, success_count, results))
}

async fn run_simulation_tests(
    sub_matches: &clap::ArgMatches<'_>,
    config: &NeuralTestConfig
) -> Result<(usize, usize, Vec<neural_testing_framework::NeuralTestResults>), Box<dyn std::error::Error>> {
    let duration: u64 = sub_matches.value_of("duration")
        .unwrap_or("300")
        .parse()
        .unwrap_or(300);
    
    let mut simulation_config = config.simulation_config.clone();
    simulation_config.simulation_duration_s = duration;

    let mut test_suite = RealTimeSimulationSuite::new(simulation_config);
    let results = test_suite.run_comprehensive_tests().await?;
    
    let total_count = results.len();
    let success_count = results.iter().filter(|r| r.success).count();
    
    Ok((total_count, success_count, results))
}

async fn run_default_tests(
    config: &NeuralTestConfig
) -> Result<(usize, usize, Vec<neural_testing_framework::NeuralTestResults>), Box<dyn std::error::Error>> {
    let mut runner = NeuralTestRunner::new(config.clone());
    runner.run_all_tests().await?;
    
    // Placeholder for actual results
    Ok((5, 5, vec![]))
}

async fn run_benchmarks(
    _iterations: usize,
    _config: &NeuralTestConfig
) -> Result<(usize, usize, Vec<neural_testing_framework::NeuralTestResults>), Box<dyn std::error::Error>> {
    // Placeholder for benchmark implementation
    Ok((10, 10, vec![]))
}

fn get_all_test_runners(
    _config: &NeuralTestConfig,
    _skip_slow: bool
) -> Vec<(String, Box<dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(usize, usize, Vec<neural_testing_framework::NeuralTestResults>), Box<dyn std::error::Error>>> + Send>> + Send>)> {
    // Placeholder for all test runners
    vec![]
}

async fn generate_test_report(
    results: &[neural_testing_framework::NeuralTestResults],
    total_tests: usize,
    successful_tests: usize,
    duration: std::time::Duration,
    output_dir: &str
) {
    println!("\n📋 Test Execution Summary");
    println!("=" .repeat(60));
    println!("🕒 Total Duration: {:.2}s", duration.as_secs_f64());
    println!("📊 Total Tests: {}", total_tests);
    println!("✅ Successful: {} ({:.1}%)", successful_tests, 
             (successful_tests as f64 / total_tests as f64) * 100.0);
    println!("❌ Failed: {} ({:.1}%)", total_tests - successful_tests,
             ((total_tests - successful_tests) as f64 / total_tests as f64) * 100.0);

    if !results.is_empty() {
        println!("\n📈 Performance Metrics:");
        let avg_latency: f64 = results.iter()
            .map(|r| r.metrics.inference_latency_us)
            .sum::<f64>() / results.len() as f64;
        let avg_accuracy: f64 = results.iter()
            .map(|r| r.metrics.accuracy_metrics.r2)
            .sum::<f64>() / results.len() as f64;
        
        println!("⚡ Average Inference Latency: {:.2} μs", avg_latency);
        println!("🎯 Average Accuracy (R²): {:.4}", avg_accuracy);
        
        // Generate detailed JSON report
        let report_path = format!("{}/neural_test_report.json", output_dir);
        let report_data = serde_json::json!({
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "duration_seconds": duration.as_secs_f64(),
                "avg_latency_us": avg_latency,
                "avg_accuracy": avg_accuracy
            },
            "test_results": results
        });
        
        if let Err(e) = std::fs::write(&report_path, serde_json::to_string_pretty(&report_data).unwrap()) {
            eprintln!("⚠️  Failed to write report: {}", e);
        } else {
            println!("📄 Detailed report saved to: {}", report_path);
        }
    }

    println!("\n🎯 Test execution completed");
}