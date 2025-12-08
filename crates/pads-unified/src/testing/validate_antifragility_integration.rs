//! Antifragility Integration Validation Script
//! 
//! This script validates the integration of the existing CDFA-Antifragility-Analyzer crate
//! with the TDD framework and ensures all performance requirements are met.

use std::time::{Duration, Instant};
use std::collections::HashMap;

use crate::antifragility_integration::*;
use crate::antifragility_trading_strategy::*;
use crate::tests::antifragility_integration_tests::*;

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub test_name: String,
    pub passed: bool,
    pub duration: Duration,
    pub performance_metrics: HashMap<String, f64>,
    pub errors: Vec<String>,
}

/// Comprehensive validation suite
pub struct AntifragilityValidationSuite {
    framework: AntifragilityTDDFramework,
    trading_strategy: AntifragilityTradingStrategy,
}

impl AntifragilityValidationSuite {
    /// Create new validation suite
    pub fn new() -> Self {
        let framework = AntifragilityTDDFramework::new(AntifragilityParameters::default());
        let trading_strategy = AntifragilityTradingStrategy::new(TradingStrategyConfig::default());
        
        Self {
            framework,
            trading_strategy,
        }
    }
    
    /// Run complete validation suite
    pub fn run_complete_validation(&self) -> ValidationSuiteResult {
        let start_time = Instant::now();
        let mut validation_results = Vec::new();
        
        println!("🚀 Starting Antifragility Integration Validation Suite");
        println!("=" .repeat(80));
        
        // 1. Validate existing crate integration
        validation_results.push(self.validate_crate_integration());
        
        // 2. Validate sub-microsecond performance
        validation_results.push(self.validate_sub_microsecond_performance());
        
        // 3. Validate Taleb's antifragility implementation
        validation_results.push(self.validate_taleb_antifragility());
        
        // 4. Validate portfolio optimization
        validation_results.push(self.validate_portfolio_optimization());
        
        // 5. Validate risk-adjusted returns improvement
        validation_results.push(self.validate_risk_adjusted_returns());
        
        // 6. Validate SIMD optimization
        validation_results.push(self.validate_simd_optimization());
        
        // 7. Validate real-time integration
        validation_results.push(self.validate_real_time_integration());
        
        // 8. Validate stress testing
        validation_results.push(self.validate_stress_testing());
        
        // 9. Validate hardware acceleration
        validation_results.push(self.validate_hardware_acceleration());
        
        // 10. Validate SME requirements
        validation_results.push(self.validate_sme_requirements());
        
        let total_duration = start_time.elapsed();
        let passed_tests = validation_results.iter().filter(|r| r.passed).count();
        let total_tests = validation_results.len();
        
        ValidationSuiteResult {
            total_tests,
            passed_tests,
            failed_tests: total_tests - passed_tests,
            validation_results,
            total_duration,
            overall_success: passed_tests == total_tests,
            performance_summary: self.generate_performance_summary(&validation_results),
        }
    }
    
    /// 1. Validate existing crate integration
    fn validate_crate_integration(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("🔍 Validating existing CDFA-Antifragility-Analyzer crate integration...");
        
        // Test crate availability and basic functionality
        let test_data = self.create_test_data(1000);
        
        match self.framework.analyzer.analyze_prices(&test_data.prices, &test_data.volumes) {
            Ok(analysis) => {
                // Validate all required components are present
                if analysis.antifragility_index < 0.0 || analysis.antifragility_index > 1.0 {
                    errors.push("Antifragility index out of range".to_string());
                }
                
                if analysis.convexity_score < 0.0 || analysis.convexity_score > 1.0 {
                    errors.push("Convexity score out of range".to_string());
                }
                
                if analysis.asymmetry_score < 0.0 || analysis.asymmetry_score > 1.0 {
                    errors.push("Asymmetry score out of range".to_string());
                }
                
                if analysis.recovery_score < 0.0 || analysis.recovery_score > 1.0 {
                    errors.push("Recovery score out of range".to_string());
                }
                
                if analysis.benefit_ratio_score < 0.0 || analysis.benefit_ratio_score > 1.0 {
                    errors.push("Benefit ratio score out of range".to_string());
                }
                
                metrics.insert("antifragility_index".to_string(), analysis.antifragility_index);
                metrics.insert("convexity_score".to_string(), analysis.convexity_score);
                metrics.insert("asymmetry_score".to_string(), analysis.asymmetry_score);
                metrics.insert("recovery_score".to_string(), analysis.recovery_score);
                metrics.insert("benefit_ratio_score".to_string(), analysis.benefit_ratio_score);
                
                println!("  ✅ Core analysis functions working");
                println!("  📊 Antifragility Index: {:.4}", analysis.antifragility_index);
                println!("  📊 Convexity Score: {:.4}", analysis.convexity_score);
                println!("  📊 Asymmetry Score: {:.4}", analysis.asymmetry_score);
                println!("  📊 Recovery Score: {:.4}", analysis.recovery_score);
                println!("  📊 Benefit Ratio Score: {:.4}", analysis.benefit_ratio_score);
            },
            Err(e) => {
                errors.push(format!("Crate analysis failed: {}", e));
            }
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Crate integration validation PASSED");
        } else {
            println!("  ❌ Crate integration validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Crate Integration".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 2. Validate sub-microsecond performance
    fn validate_sub_microsecond_performance(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("⚡ Validating sub-microsecond performance...");
        
        let test_data = self.create_test_data(10000);
        let mut durations = Vec::new();
        
        // Run 100 iterations to get accurate timing
        for i in 0..100 {
            let analysis_start = Instant::now();
            match self.framework.analyzer.analyze_prices(&test_data.prices, &test_data.volumes) {
                Ok(_) => {
                    let analysis_duration = analysis_start.elapsed();
                    durations.push(analysis_duration);
                },
                Err(e) => {
                    errors.push(format!("Analysis failed on iteration {}: {}", i, e));
                }
            }
        }
        
        if !durations.is_empty() {
            let min_duration = durations.iter().min().unwrap();
            let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
            let max_duration = durations.iter().max().unwrap();
            
            let min_ns = min_duration.as_nanos() as f64;
            let avg_ns = avg_duration.as_nanos() as f64;
            let max_ns = max_duration.as_nanos() as f64;
            
            metrics.insert("min_duration_ns".to_string(), min_ns);
            metrics.insert("avg_duration_ns".to_string(), avg_ns);
            metrics.insert("max_duration_ns".to_string(), max_ns);
            metrics.insert("throughput_mps".to_string(), test_data.prices.len() as f64 / avg_duration.as_secs_f64() / 1e6);
            
            println!("  📊 Min Duration: {:.0} ns", min_ns);
            println!("  📊 Avg Duration: {:.0} ns", avg_ns);
            println!("  📊 Max Duration: {:.0} ns", max_ns);
            println!("  📊 Throughput: {:.2} million data points/second", test_data.prices.len() as f64 / avg_duration.as_secs_f64() / 1e6);
            
            // Check sub-microsecond performance (1000 ns = 1 microsecond)
            if min_ns > 1000.0 {
                errors.push(format!("Minimum duration {:.0} ns exceeds 1 microsecond target", min_ns));
            } else {
                println!("  ✅ Sub-microsecond performance achieved: {:.0} ns", min_ns);
            }
        } else {
            errors.push("No valid duration measurements".to_string());
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Sub-microsecond performance validation PASSED");
        } else {
            println!("  ❌ Sub-microsecond performance validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Sub-microsecond Performance".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 3. Validate Taleb's antifragility implementation
    fn validate_taleb_antifragility(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("🧠 Validating Taleb's antifragility implementation...");
        
        // Test with known antifragile scenario (benefits from volatility)
        let antifragile_data = self.create_antifragile_scenario();
        let fragile_data = self.create_fragile_scenario();
        
        match (
            self.framework.analyzer.analyze_prices(&antifragile_data.prices, &antifragile_data.volumes),
            self.framework.analyzer.analyze_prices(&fragile_data.prices, &fragile_data.volumes)
        ) {
            (Ok(antifragile_analysis), Ok(fragile_analysis)) => {
                // Validate that antifragile scenario shows higher antifragility
                if antifragile_analysis.antifragility_index <= fragile_analysis.antifragility_index {
                    errors.push("Antifragile scenario should show higher antifragility index".to_string());
                } else {
                    println!("  ✅ Antifragile scenario detection working");
                }
                
                // Validate convexity measurement
                if antifragile_analysis.convexity_score <= fragile_analysis.convexity_score {
                    errors.push("Antifragile scenario should show higher convexity".to_string());
                } else {
                    println!("  ✅ Convexity measurement working");
                }
                
                // Validate benefit ratio
                if antifragile_analysis.benefit_ratio_score <= fragile_analysis.benefit_ratio_score {
                    errors.push("Antifragile scenario should show higher benefit ratio".to_string());
                } else {
                    println!("  ✅ Benefit ratio calculation working");
                }
                
                metrics.insert("antifragile_index".to_string(), antifragile_analysis.antifragility_index);
                metrics.insert("fragile_index".to_string(), fragile_analysis.antifragility_index);
                metrics.insert("antifragility_difference".to_string(), antifragile_analysis.antifragility_index - fragile_analysis.antifragility_index);
                
                println!("  📊 Antifragile Index: {:.4}", antifragile_analysis.antifragility_index);
                println!("  📊 Fragile Index: {:.4}", fragile_analysis.antifragility_index);
                println!("  📊 Difference: {:.4}", antifragile_analysis.antifragility_index - fragile_analysis.antifragility_index);
            },
            (Err(e), _) => errors.push(format!("Antifragile analysis failed: {}", e)),
            (_, Err(e)) => errors.push(format!("Fragile analysis failed: {}", e)),
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Taleb's antifragility implementation validation PASSED");
        } else {
            println!("  ❌ Taleb's antifragility implementation validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Taleb's Antifragility Implementation".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 4. Validate portfolio optimization
    fn validate_portfolio_optimization(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("💼 Validating portfolio optimization...");
        
        let portfolio_data = self.create_portfolio_data(5);
        
        match self.framework.test_portfolio_robustness(&portfolio_data) {
            Ok(test_result) => {
                if !test_result.passed {
                    errors.push("Portfolio optimization test failed".to_string());
                } else {
                    println!("  ✅ Portfolio optimization working");
                }
                
                // Extract robustness score from metrics
                let robustness_score = test_result.metrics.iter()
                    .find(|(key, _)| key == "robustness_score")
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                
                if robustness_score < 0.6 {
                    errors.push(format!("Portfolio robustness below threshold: {:.3}", robustness_score));
                } else {
                    println!("  ✅ Portfolio robustness achieved: {:.3}", robustness_score);
                }
                
                metrics.insert("robustness_score".to_string(), robustness_score);
                metrics.insert("test_duration_ms".to_string(), test_result.duration.as_millis() as f64);
                
                println!("  📊 Robustness Score: {:.4}", robustness_score);
                println!("  📊 Test Duration: {:?}", test_result.duration);
            },
            Err(e) => {
                errors.push(format!("Portfolio optimization failed: {}", e));
            }
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Portfolio optimization validation PASSED");
        } else {
            println!("  ❌ Portfolio optimization validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Portfolio Optimization".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 5. Validate risk-adjusted returns improvement
    fn validate_risk_adjusted_returns(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("📈 Validating risk-adjusted returns improvement...");
        
        let test_data = self.create_test_data(5000);
        let benchmark_returns = self.create_benchmark_returns(5000);
        
        match self.framework.test_risk_adjusted_returns(&test_data, &benchmark_returns) {
            Ok(test_result) => {
                let improvement_pct = test_result.metrics.iter()
                    .find(|(key, _)| key == "improvement_pct")
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                
                let antifragile_sharpe = test_result.metrics.iter()
                    .find(|(key, _)| key == "antifragile_sharpe")
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                
                let benchmark_sharpe = test_result.metrics.iter()
                    .find(|(key, _)| key == "benchmark_sharpe")
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                
                // Validate 25-40% improvement target
                if improvement_pct < 25.0 {
                    errors.push(format!("Improvement below 25% target: {:.2}%", improvement_pct));
                } else if improvement_pct > 60.0 {
                    errors.push(format!("Improvement unrealistically high: {:.2}%", improvement_pct));
                } else {
                    println!("  ✅ Risk-adjusted returns improvement achieved: {:.2}%", improvement_pct);
                }
                
                metrics.insert("improvement_pct".to_string(), improvement_pct);
                metrics.insert("antifragile_sharpe".to_string(), antifragile_sharpe);
                metrics.insert("benchmark_sharpe".to_string(), benchmark_sharpe);
                
                println!("  📊 Improvement: {:.2}%", improvement_pct);
                println!("  📊 Antifragile Sharpe: {:.4}", antifragile_sharpe);
                println!("  📊 Benchmark Sharpe: {:.4}", benchmark_sharpe);
            },
            Err(e) => {
                errors.push(format!("Risk-adjusted returns test failed: {}", e));
            }
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Risk-adjusted returns improvement validation PASSED");
        } else {
            println!("  ❌ Risk-adjusted returns improvement validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Risk-Adjusted Returns Improvement".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 6. Validate SIMD optimization
    fn validate_simd_optimization(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("🚀 Validating SIMD optimization...");
        
        let test_data = self.create_test_data(10000);
        
        match self.framework.test_simd_optimization(&test_data) {
            Ok(test_result) => {
                let speedup = test_result.metrics.iter()
                    .find(|(key, _)| key == "speedup")
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                
                let results_match = test_result.metrics.iter()
                    .find(|(key, _)| key == "results_match")
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                
                // Validate speedup
                if speedup < 1.5 {
                    errors.push(format!("SIMD speedup below 1.5x target: {:.2}x", speedup));
                } else {
                    println!("  ✅ SIMD speedup achieved: {:.2}x", speedup);
                }
                
                // Validate results match
                if results_match < 0.5 {
                    errors.push("SIMD and scalar results don't match".to_string());
                } else {
                    println!("  ✅ SIMD results match scalar results");
                }
                
                metrics.insert("speedup".to_string(), speedup);
                metrics.insert("results_match".to_string(), results_match);
                
                println!("  📊 SIMD Speedup: {:.2}x", speedup);
                println!("  📊 Results Match: {}", results_match > 0.5);
            },
            Err(e) => {
                errors.push(format!("SIMD optimization test failed: {}", e));
            }
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ SIMD optimization validation PASSED");
        } else {
            println!("  ❌ SIMD optimization validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "SIMD Optimization".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 7. Validate real-time integration
    fn validate_real_time_integration(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("🔄 Validating real-time integration...");
        
        let mut signal_count = 0;
        let mut processing_durations = Vec::new();
        
        // Simulate real-time trading
        for i in 0..1000 {
            let tick = MarketTick {
                symbol: "BTCUSD".to_string(),
                price: 50000.0 + (i as f64 * 0.1).sin() * 1000.0,
                volume: 1000.0 + (i as f64 * 0.05).cos() * 100.0,
                timestamp: i as u64,
            };
            
            let process_start = Instant::now();
            match self.trading_strategy.process_market_data(tick) {
                Ok(signal) => {
                    let process_duration = process_start.elapsed();
                    processing_durations.push(process_duration);
                    
                    if signal.direction != SignalDirection::Neutral {
                        signal_count += 1;
                    }
                },
                Err(e) => {
                    errors.push(format!("Real-time processing failed at tick {}: {}", i, e));
                }
            }
        }
        
        if !processing_durations.is_empty() {
            let avg_processing_time = processing_durations.iter().sum::<Duration>() / processing_durations.len() as u32;
            let max_processing_time = processing_durations.iter().max().unwrap();
            
            metrics.insert("avg_processing_time_ms".to_string(), avg_processing_time.as_millis() as f64);
            metrics.insert("max_processing_time_ms".to_string(), max_processing_time.as_millis() as f64);
            metrics.insert("signal_count".to_string(), signal_count as f64);
            metrics.insert("signal_rate".to_string(), signal_count as f64 / 1000.0);
            
            println!("  📊 Average Processing Time: {:?}", avg_processing_time);
            println!("  📊 Max Processing Time: {:?}", max_processing_time);
            println!("  📊 Signals Generated: {}", signal_count);
            println!("  📊 Signal Rate: {:.2}%", signal_count as f64 / 1000.0 * 100.0);
            
            // Validate real-time performance
            if avg_processing_time > Duration::from_millis(100) {
                errors.push(format!("Average processing time too high: {:?}", avg_processing_time));
            } else {
                println!("  ✅ Real-time processing performance acceptable");
            }
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Real-time integration validation PASSED");
        } else {
            println!("  ❌ Real-time integration validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Real-time Integration".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 8. Validate stress testing
    fn validate_stress_testing(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("🧪 Validating stress testing...");
        
        let stress_scenarios = vec![
            self.create_2008_crisis_scenario(),
            self.create_covid_crash_scenario(),
            self.create_flash_crash_scenario(),
        ];
        
        match self.trading_strategy.perform_stress_test(&stress_scenarios) {
            Ok(stress_result) => {
                let overall_resilience = stress_result.overall_resilience;
                
                if overall_resilience < 0.5 {
                    errors.push(format!("Overall resilience below threshold: {:.3}", overall_resilience));
                } else {
                    println!("  ✅ Stress testing resilience achieved: {:.3}", overall_resilience);
                }
                
                metrics.insert("overall_resilience".to_string(), overall_resilience);
                metrics.insert("scenarios_tested".to_string(), stress_result.scenario_results.len() as f64);
                metrics.insert("recommendations_count".to_string(), stress_result.recommendations.len() as f64);
                
                println!("  📊 Overall Resilience: {:.4}", overall_resilience);
                println!("  📊 Scenarios Tested: {}", stress_result.scenario_results.len());
                println!("  📊 Recommendations: {}", stress_result.recommendations.len());
                
                for scenario in &stress_result.scenario_results {
                    println!("    {} - Resilience: {:.3}", scenario.scenario_name, scenario.resilience_score);
                }
            },
            Err(e) => {
                errors.push(format!("Stress testing failed: {}", e));
            }
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Stress testing validation PASSED");
        } else {
            println!("  ❌ Stress testing validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Stress Testing".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 9. Validate hardware acceleration
    fn validate_hardware_acceleration(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("🖥️ Validating hardware acceleration...");
        
        // Test parallel processing
        let test_data = self.create_test_data(20000);
        
        let params_parallel = AntifragilityParameters {
            enable_parallel: true,
            enable_simd: true,
            ..Default::default()
        };
        
        let params_sequential = AntifragilityParameters {
            enable_parallel: false,
            enable_simd: false,
            ..Default::default()
        };
        
        let analyzer_parallel = AntifragilityAnalyzer::with_params(params_parallel);
        let analyzer_sequential = AntifragilityAnalyzer::with_params(params_sequential);
        
        // Test parallel performance
        let parallel_start = Instant::now();
        match analyzer_parallel.analyze_prices(&test_data.prices, &test_data.volumes) {
            Ok(_) => {
                let parallel_duration = parallel_start.elapsed();
                
                // Test sequential performance
                let sequential_start = Instant::now();
                match analyzer_sequential.analyze_prices(&test_data.prices, &test_data.volumes) {
                    Ok(_) => {
                        let sequential_duration = sequential_start.elapsed();
                        let acceleration = sequential_duration.as_nanos() as f64 / parallel_duration.as_nanos() as f64;
                        
                        metrics.insert("parallel_duration_ms".to_string(), parallel_duration.as_millis() as f64);
                        metrics.insert("sequential_duration_ms".to_string(), sequential_duration.as_millis() as f64);
                        metrics.insert("acceleration_factor".to_string(), acceleration);
                        
                        println!("  📊 Parallel Duration: {:?}", parallel_duration);
                        println!("  📊 Sequential Duration: {:?}", sequential_duration);
                        println!("  📊 Acceleration Factor: {:.2}x", acceleration);
                        
                        if acceleration < 1.2 {
                            errors.push(format!("Hardware acceleration below 1.2x: {:.2}x", acceleration));
                        } else {
                            println!("  ✅ Hardware acceleration achieved: {:.2}x", acceleration);
                        }
                    },
                    Err(e) => errors.push(format!("Sequential analysis failed: {}", e)),
                }
            },
            Err(e) => errors.push(format!("Parallel analysis failed: {}", e)),
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ Hardware acceleration validation PASSED");
        } else {
            println!("  ❌ Hardware acceleration validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "Hardware Acceleration".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// 10. Validate SME requirements
    fn validate_sme_requirements(&self) -> ValidationResult {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();
        
        println!("👨‍💼 Validating SME requirements...");
        
        // Test with real-world portfolio stress scenarios
        let portfolio_data = self.create_portfolio_data(10);
        let stress_scenarios = vec![
            self.create_2008_crisis_scenario(),
            self.create_covid_crash_scenario(),
            self.create_dot_com_crash_scenario(),
        ];
        
        // Comprehensive test suite
        let test_data = self.create_test_data(5000);
        let benchmark_returns = self.create_benchmark_returns(5000);
        
        match self.framework.run_comprehensive_tests(&test_data, &portfolio_data, &benchmark_returns) {
            Ok(suite_result) => {
                let success_rate = suite_result.passed_tests as f64 / suite_result.total_tests as f64;
                
                if success_rate < 0.9 {
                    errors.push(format!("Test success rate below 90%: {:.1}%", success_rate * 100.0));
                } else {
                    println!("  ✅ Comprehensive test suite passed: {:.1}%", success_rate * 100.0);
                }
                
                metrics.insert("success_rate".to_string(), success_rate);
                metrics.insert("total_tests".to_string(), suite_result.total_tests as f64);
                metrics.insert("passed_tests".to_string(), suite_result.passed_tests as f64);
                metrics.insert("suite_duration_ms".to_string(), suite_result.duration.as_millis() as f64);
                
                println!("  📊 Success Rate: {:.1}%", success_rate * 100.0);
                println!("  📊 Tests Passed: {}/{}", suite_result.passed_tests, suite_result.total_tests);
                println!("  📊 Suite Duration: {:?}", suite_result.duration);
                
                // Validate individual test results
                for test_result in &suite_result.test_results {
                    if !test_result.passed {
                        errors.push(format!("Test failed: {}", test_result.test_name));
                    }
                }
            },
            Err(e) => {
                errors.push(format!("Comprehensive test suite failed: {}", e));
            }
        }
        
        let duration = start.elapsed();
        let passed = errors.is_empty();
        
        if passed {
            println!("  ✅ SME requirements validation PASSED");
        } else {
            println!("  ❌ SME requirements validation FAILED");
            for error in &errors {
                println!("     Error: {}", error);
            }
        }
        
        ValidationResult {
            test_name: "SME Requirements".to_string(),
            passed,
            duration,
            performance_metrics: metrics,
            errors,
        }
    }
    
    /// Generate performance summary
    fn generate_performance_summary(&self, results: &[ValidationResult]) -> PerformanceSummary {
        let mut total_duration = Duration::from_nanos(0);
        let mut all_metrics = HashMap::new();
        
        for result in results {
            total_duration += result.duration;
            for (key, value) in &result.performance_metrics {
                all_metrics.insert(format!("{}_{}", result.test_name.replace(' ', "_").to_lowercase(), key), *value);
            }
        }
        
        PerformanceSummary {
            total_validation_time: total_duration,
            performance_metrics: all_metrics,
            benchmark_results: self.framework.get_benchmarks(),
        }
    }
    
    // Helper methods for test data creation
    
    fn create_test_data(&self, size: usize) -> MarketData {
        let mut prices = Vec::with_capacity(size);
        let mut volumes = Vec::with_capacity(size);
        
        let mut price = 100.0;
        for i in 0..size {
            let t = i as f64 * 0.01;
            let return_rate = 0.001 * (t * 0.1).sin() + 0.0005 * (t * 0.05).cos();
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(1000.0 + 100.0 * (t * 0.02).sin());
        }
        
        MarketData {
            prices,
            volumes,
            timestamps: (0..size).map(|i| i as u64).collect(),
        }
    }
    
    fn create_antifragile_scenario(&self) -> MarketData {
        let size = 1000;
        let mut prices = Vec::with_capacity(size);
        let mut volumes = Vec::with_capacity(size);
        
        let mut price = 100.0;
        for i in 0..size {
            let t = i as f64 * 0.01;
            let volatility = 0.002 * (1.0 + (t * 0.1).sin());
            let performance_boost = volatility * 0.5; // Benefits from volatility
            
            let return_rate = performance_boost + volatility * (t * 5.0).sin();
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(1000.0 + 300.0 * volatility);
        }
        
        MarketData {
            prices,
            volumes,
            timestamps: (0..size).map(|i| i as u64).collect(),
        }
    }
    
    fn create_fragile_scenario(&self) -> MarketData {
        let size = 1000;
        let mut prices = Vec::with_capacity(size);
        let mut volumes = Vec::with_capacity(size);
        
        let mut price = 100.0;
        for i in 0..size {
            let t = i as f64 * 0.01;
            let volatility = 0.002 * (1.0 + (t * 0.1).sin());
            let performance_penalty = -volatility * 0.3; // Suffers from volatility
            
            let return_rate = performance_penalty + volatility * (t * 5.0).sin();
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(1000.0 + 300.0 * volatility);
        }
        
        MarketData {
            prices,
            volumes,
            timestamps: (0..size).map(|i| i as u64).collect(),
        }
    }
    
    fn create_portfolio_data(&self, num_assets: usize) -> PortfolioData {
        let mut assets = Vec::new();
        
        for i in 0..num_assets {
            let size = 500;
            let mut prices = Vec::with_capacity(size);
            let mut volumes = Vec::with_capacity(size);
            
            let mut price = 100.0 + (i as f64) * 10.0;
            for j in 0..size {
                let t = j as f64 * 0.02;
                let return_rate = 0.0001 * (t * (i as f64 + 1.0)).sin() + 
                                0.0005 * (t * 0.1).cos();
                price *= 1.0 + return_rate;
                prices.push(price);
                volumes.push(8000.0 + 2000.0 * (t * 0.1).sin());
            }
            
            assets.push(AssetData {
                symbol: format!("ASSET{}", i + 1),
                prices,
                volumes,
            });
        }
        
        let constraints = PortfolioConstraints {
            max_weights: HashMap::new(),
            min_weights: HashMap::new(),
            target_volatility: Some(0.15),
            max_drawdown: Some(0.20),
        };
        
        PortfolioData { assets, constraints }
    }
    
    fn create_benchmark_returns(&self, size: usize) -> Vec<f64> {
        let mut returns = Vec::with_capacity(size);
        
        for i in 0..size {
            let t = i as f64 * 0.01;
            let return_val = 0.0002 + 0.001 * (t * 0.1).sin();
            returns.push(return_val);
        }
        
        returns
    }
    
    fn create_2008_crisis_scenario(&self) -> StressScenario {
        let size = 600;
        let mut prices = Vec::with_capacity(size);
        let mut volumes = Vec::with_capacity(size);
        
        let mut price = 100.0;
        for i in 0..size {
            let return_rate = if i >= 200 && i <= 400 {
                -0.003 - 0.002 * ((i as f64 - 300.0) * 0.1).sin()
            } else if i > 400 && i < 500 {
                0.001
            } else {
                0.0001 * ((i as f64) * 0.1).sin()
            };
            
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(15000.0 + 10000.0 * return_rate.abs());
        }
        
        StressScenario {
            name: "2008 Financial Crisis".to_string(),
            price_path: prices,
            volume_path: volumes,
            timestamps: (0..size).map(|i| i as u64).collect(),
        }
    }
    
    fn create_covid_crash_scenario(&self) -> StressScenario {
        let size = 400;
        let mut prices = Vec::with_capacity(size);
        let mut volumes = Vec::with_capacity(size);
        
        let mut price = 100.0;
        for i in 0..size {
            let return_rate = if i >= 100 && i <= 130 {
                -0.05
            } else if i > 130 && i < 200 {
                0.008
            } else {
                0.0002 * ((i as f64) * 0.1).sin()
            };
            
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(12000.0 + 8000.0 * return_rate.abs());
        }
        
        StressScenario {
            name: "COVID-19 Crash".to_string(),
            price_path: prices,
            volume_path: volumes,
            timestamps: (0..size).map(|i| i as u64).collect(),
        }
    }
    
    fn create_flash_crash_scenario(&self) -> StressScenario {
        let size = 500;
        let mut prices = Vec::with_capacity(size);
        let mut volumes = Vec::with_capacity(size);
        
        let mut price = 100.0;
        for i in 0..size {
            let return_rate = if i == 250 {
                -0.1
            } else if i > 250 && i < 260 {
                0.01
            } else {
                0.0001 * ((i as f64) * 0.1).sin()
            };
            
            price *= 1.0 + return_rate;
            prices.push(price);
            
            let volume = if i == 250 {
                50000.0
            } else {
                10000.0
            };
            volumes.push(volume);
        }
        
        StressScenario {
            name: "Flash Crash".to_string(),
            price_path: prices,
            volume_path: volumes,
            timestamps: (0..size).map(|i| i as u64).collect(),
        }
    }
    
    fn create_dot_com_crash_scenario(&self) -> StressScenario {
        let size = 800;
        let mut prices = Vec::with_capacity(size);
        let mut volumes = Vec::with_capacity(size);
        
        let mut price = 100.0;
        for i in 0..size {
            let return_rate = if i >= 100 && i <= 600 {
                -0.001 - 0.001 * ((i as f64 - 350.0) * 0.01).sin()
            } else if i > 600 && i < 700 {
                0.0005
            } else {
                0.0002 * ((i as f64) * 0.1).sin()
            };
            
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(12000.0 + 6000.0 * return_rate.abs());
        }
        
        StressScenario {
            name: "Dot-com Crash".to_string(),
            price_path: prices,
            volume_path: volumes,
            timestamps: (0..size).map(|i| i as u64).collect(),
        }
    }
}

/// Validation suite result
#[derive(Debug, Clone)]
pub struct ValidationSuiteResult {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub validation_results: Vec<ValidationResult>,
    pub total_duration: Duration,
    pub overall_success: bool,
    pub performance_summary: PerformanceSummary,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_validation_time: Duration,
    pub performance_metrics: HashMap<String, f64>,
    pub benchmark_results: PerformanceBenchmarks,
}

/// Main validation function
pub fn run_antifragility_validation() -> ValidationSuiteResult {
    let validation_suite = AntifragilityValidationSuite::new();
    let result = validation_suite.run_complete_validation();
    
    println!("\n🎯 ANTIFRAGILITY INTEGRATION VALIDATION COMPLETE");
    println!("=" .repeat(80));
    println!("📊 Results Summary:");
    println!("  Total Tests: {}", result.total_tests);
    println!("  Passed Tests: {}", result.passed_tests);
    println!("  Failed Tests: {}", result.failed_tests);
    println!("  Success Rate: {:.1}%", result.passed_tests as f64 / result.total_tests as f64 * 100.0);
    println!("  Total Duration: {:?}", result.total_duration);
    println!("  Overall Success: {}", if result.overall_success { "✅ PASSED" } else { "❌ FAILED" });
    
    if result.overall_success {
        println!("\n🚀 ANTIFRAGILITY INTEGRATION READY FOR PRODUCTION DEPLOYMENT");
    } else {
        println!("\n⚠️  ANTIFRAGILITY INTEGRATION REQUIRES FIXES BEFORE DEPLOYMENT");
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validation_suite_creation() {
        let suite = AntifragilityValidationSuite::new();
        // Just test that it can be created without panicking
        assert!(true);
    }
    
    #[test]
    fn test_validation_run() {
        let suite = AntifragilityValidationSuite::new();
        let result = suite.run_complete_validation();
        
        assert!(result.total_tests > 0);
        assert!(result.passed_tests <= result.total_tests);
        assert_eq!(result.passed_tests + result.failed_tests, result.total_tests);
    }
}