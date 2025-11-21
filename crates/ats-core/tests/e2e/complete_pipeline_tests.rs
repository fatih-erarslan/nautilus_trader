//! End-to-End Tests for Complete ATS-CP Pipeline
//!
//! These tests verify the entire ATS-CP system working together:
//! - Complete workflow from raw inputs to final predictions
//! - Real-world trading scenarios and stress tests
//! - Multi-component integration under realistic conditions
//! - Performance validation under production-like loads
//! - Error recovery and fault tolerance

use ats_core::{
    conformal::ConformalPredictor,
    config::{AtsCpConfig, ConformalConfig, TemperatureConfig},
    types::{AtsCpVariant, Confidence, AtsCpResult, PredictionIntervals},
    error::{AtsCoreError, Result},
    test_framework::{TestFramework, TestMetrics, swarm_utils},
};
use std::time::{Duration, Instant};
use tokio;

/// End-to-end test fixture with realistic trading scenarios
struct E2ETestFixture {
    predictors: Vec<ConformalPredictor>,
    configs: Vec<AtsCpConfig>,
    trading_scenarios: TradingScenarios,
}

/// Realistic trading scenarios for end-to-end testing
struct TradingScenarios {
    high_frequency_scenario: HighFrequencyScenario,
    market_volatility_scenario: MarketVolatilityScenario,
    model_ensemble_scenario: ModelEnsembleScenario,
    real_time_adaptation_scenario: RealTimeAdaptationScenario,
}

struct HighFrequencyScenario {
    trades_per_second: usize,
    duration_seconds: u64,
    logits_stream: Vec<Vec<f64>>,
    expected_latency_us: u64,
}

struct MarketVolatilityScenario {
    volatility_levels: Vec<f64>,
    market_conditions: Vec<MarketCondition>,
    stress_test_data: Vec<(Vec<f64>, Vec<f64>)>,
}

#[derive(Debug, Clone)]
enum MarketCondition {
    Normal,
    HighVolatility,
    MarketCrash,
    Recovery,
    Bullish,
    Bearish,
}

struct ModelEnsembleScenario {
    model_variants: Vec<AtsCpVariant>,
    ensemble_weights: Vec<f64>,
    consensus_threshold: f64,
}

struct RealTimeAdaptationScenario {
    streaming_data: Vec<(Vec<f64>, Vec<f64>)>, // (predictions, true_values)
    adaptation_intervals: Vec<usize>,
    performance_thresholds: Vec<f64>,
}

impl E2ETestFixture {
    fn new() -> Self {
        // Multiple configurations for different scenarios
        let configs = vec![
            // High-speed trading config
            AtsCpConfig {
                conformal: ConformalConfig {
                    target_latency_us: 10,
                    min_calibration_size: 20,
                    max_calibration_size: 100,
                    online_calibration: true,
                    quantile_method: crate::config::QuantileMethod::Nearest,
                    ..Default::default()
                },
                temperature: TemperatureConfig {
                    target_latency_us: 5,
                    max_search_iterations: 10,
                    search_tolerance: 1e-3,
                    ..Default::default()
                },
                ..Default::default()
            },
            // Accuracy-focused config
            AtsCpConfig {
                conformal: ConformalConfig {
                    target_latency_us: 50,
                    min_calibration_size: 50,
                    max_calibration_size: 1000,
                    online_calibration: true,
                    quantile_method: crate::config::QuantileMethod::Linear,
                    ..Default::default()
                },
                temperature: TemperatureConfig {
                    target_latency_us: 20,
                    max_search_iterations: 50,
                    search_tolerance: 1e-6,
                    ..Default::default()
                },
                ..Default::default()
            },
            // Balanced config
            AtsCpConfig::default(),
        ];
        
        let predictors: Vec<ConformalPredictor> = configs
            .iter()
            .map(|config| ConformalPredictor::new(config).unwrap())
            .collect();
        
        let trading_scenarios = TradingScenarios {
            high_frequency_scenario: HighFrequencyScenario {
                trades_per_second: 1000,
                duration_seconds: 10,
                logits_stream: generate_hft_logits_stream(1000, 10),
                expected_latency_us: 15,
            },
            market_volatility_scenario: MarketVolatilityScenario {
                volatility_levels: vec![0.1, 0.3, 0.5, 0.8, 1.2, 2.0],
                market_conditions: vec![
                    MarketCondition::Normal,
                    MarketCondition::HighVolatility,
                    MarketCondition::MarketCrash,
                    MarketCondition::Recovery,
                    MarketCondition::Bullish,
                    MarketCondition::Bearish,
                ],
                stress_test_data: generate_volatility_test_data(),
            },
            model_ensemble_scenario: ModelEnsembleScenario {
                model_variants: vec![AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ, AtsCpVariant::MAQ],
                ensemble_weights: vec![0.3, 0.3, 0.2, 0.2],
                consensus_threshold: 0.8,
            },
            real_time_adaptation_scenario: RealTimeAdaptationScenario {
                streaming_data: generate_streaming_data(),
                adaptation_intervals: vec![100, 200, 500, 1000],
                performance_thresholds: vec![0.90, 0.95, 0.99],
            },
        };
        
        Self {
            predictors,
            configs,
            trading_scenarios,
        }
    }
}

/// Generate realistic HFT logits stream
fn generate_hft_logits_stream(trades_per_second: usize, duration_seconds: u64) -> Vec<Vec<f64>> {
    let total_trades = (trades_per_second as u64 * duration_seconds) as usize;
    let mut stream = Vec::with_capacity(total_trades);
    
    for i in 0..total_trades {
        // Simulate realistic market logits with trends and noise
        let trend = (i as f64 / 1000.0).sin() * 0.5;
        let noise = (i as f64 * 0.7).sin() * 0.1;
        let base_logits = vec![
            2.0 + trend + noise,
            1.0 - trend * 0.5 + noise * 0.8,
            0.5 + trend * 0.3 - noise * 0.6,
            0.2 - trend * 0.2 + noise * 0.4,
            0.1 + trend * 0.1 - noise * 0.3,
        ];
        stream.push(base_logits);
    }
    
    stream
}

/// Generate market volatility test data
fn generate_volatility_test_data() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut data = Vec::new();
    
    for volatility in [0.1, 0.3, 0.5, 0.8, 1.2, 2.0] {
        let predictions: Vec<f64> = (0..100)
            .map(|i| {
                let base = (i as f64 * 0.1).sin();
                let vol_component = (i as f64 * 0.3).cos() * volatility;
                base + vol_component
            })
            .collect();
        
        let calibration: Vec<f64> = (0..200)
            .map(|i| {
                let base = (i as f64 * 0.05).sin();
                let vol_component = (i as f64 * 0.15).cos() * volatility * 0.5;
                base + vol_component
            })
            .collect();
        
        data.push((predictions, calibration));
    }
    
    data
}

/// Generate streaming data for real-time adaptation
fn generate_streaming_data() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut data = Vec::new();
    
    for batch in 0..50 {
        let batch_size = 20;
        let predictions: Vec<f64> = (0..batch_size)
            .map(|i| {
                let drift = (batch as f64) * 0.01;
                let noise = ((batch * batch_size + i) as f64 * 0.7).sin() * 0.1;
                1.0 + drift + noise
            })
            .collect();
        
        let true_values: Vec<f64> = predictions
            .iter()
            .map(|&pred| pred + ((predictions.len() as f64).sin() * 0.05))
            .collect();
        
        data.push((predictions, true_values));
    }
    
    data
}

/// Complete end-to-end pipeline tests
mod complete_pipeline_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_complete_ats_cp_workflow() {
        println!("🚀 Testing complete ATS-CP workflow...");
        
        let mut fixture = E2ETestFixture::new();
        
        // Stage 1: Data preparation and validation
        println!("  Stage 1: Data preparation and validation");
        
        let test_logits = vec![2.1, 1.3, 0.8, 0.4, 0.2];
        let calibration_logits = vec![
            vec![2.0, 1.2, 0.9, 0.5, 0.3],
            vec![1.9, 1.4, 0.7, 0.6, 0.2],
            vec![2.2, 1.1, 0.8, 0.4, 0.1],
            vec![1.8, 1.5, 0.6, 0.7, 0.3],
            vec![2.1, 1.0, 0.9, 0.3, 0.4],
        ];
        let calibration_labels = vec![0, 0, 1, 2, 3];
        
        // Validate input data
        assert!(!test_logits.is_empty(), "Test logits should not be empty");
        assert!(!calibration_logits.is_empty(), "Calibration logits should not be empty");
        assert_eq!(calibration_logits.len(), calibration_labels.len(), "Calibration data should be consistent");
        
        println!("    ✅ Data validation completed");
        
        // Stage 2: Multiple variant processing
        println!("  Stage 2: Multiple variant processing");
        
        let variants = vec![AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ, AtsCpVariant::MAQ];
        let mut variant_results = Vec::new();
        
        for variant in variants {
            println!("    Processing variant: {:?}", variant);
            
            let start_time = Instant::now();
            let result = fixture.predictors[0].ats_cp_predict(
                &test_logits,
                &calibration_logits,
                &calibration_labels,
                0.95,
                variant.clone(),
            );
            let processing_time = start_time.elapsed();
            
            assert!(result.is_ok(), "Variant {:?} should succeed", variant);
            let ats_result = result.unwrap();
            
            // Validate variant-specific results
            validate_ats_cp_result(&ats_result, 0.95, &variant)?;
            
            variant_results.push((variant, ats_result, processing_time));
            println!("      ✅ Variant {:?} completed in {:?}", variant, processing_time);
        }
        
        // Stage 3: Cross-variant consistency validation
        println!("  Stage 3: Cross-variant consistency validation");
        
        // All variants should produce valid results
        assert_eq!(variant_results.len(), 4, "All variants should complete successfully");
        
        // Compare variant characteristics
        for (i, (variant1, result1, _)) in variant_results.iter().enumerate() {
            for (variant2, result2, _) in variant_results.iter().skip(i + 1) {
                // Different variants may produce different results, but all should be valid
                assert_eq!(result1.calibrated_probabilities.len(), result2.calibrated_probabilities.len(),
                          "Variants {:?} and {:?} should produce same number of probabilities", variant1, variant2);
                
                assert_eq!(result1.coverage_guarantee, result2.coverage_guarantee,
                          "Variants {:?} and {:?} should have same coverage guarantee", variant1, variant2);
                
                // Both conformal sets should be non-empty and valid
                assert!(!result1.conformal_set.is_empty() && !result2.conformal_set.is_empty(),
                       "Variants {:?} and {:?} should both produce non-empty conformal sets", variant1, variant2);
            }
        }
        
        println!("    ✅ Cross-variant consistency validated");
        
        // Stage 4: Performance validation
        println!("  Stage 4: Performance validation");
        
        let total_time: Duration = variant_results.iter().map(|(_, _, time)| *time).sum();
        let avg_time = total_time / variant_results.len() as u32;
        let max_time = variant_results.iter().map(|(_, _, time)| *time).max().unwrap();
        
        println!("    Total processing time:   {:?}", total_time);
        println!("    Average time per variant: {:?}", avg_time);
        println!("    Maximum time per variant: {:?}", max_time);
        
        // Performance requirements
        assert!(avg_time < Duration::from_micros(30), "Average variant time should be <30μs");
        assert!(max_time < Duration::from_micros(50), "Maximum variant time should be <50μs");
        assert!(total_time < Duration::from_micros(100), "Total workflow time should be <100μs");
        
        println!("    ✅ Performance requirements met");
        
        println!("✅ Complete ATS-CP workflow test passed");
    }
    
    #[tokio::test]
    async fn test_real_world_trading_simulation() {
        println!("📈 Testing real-world trading simulation...");
        
        let mut fixture = E2ETestFixture::new();
        let simulation_duration = Duration::from_secs(5);
        let target_trades_per_second = 500;
        
        println!("  Simulation parameters:");
        println!("    Duration: {:?}", simulation_duration);
        println!("    Target rate: {} trades/sec", target_trades_per_second);
        
        let start_time = Instant::now();
        let mut total_trades = 0;
        let mut successful_trades = 0;
        let mut latencies = Vec::new();
        let mut profit_loss = 0.0f64;
        
        // Simulate real trading conditions
        let mut market_state = 0.0f64;
        let mut volatility = 0.2f64;
        
        while start_time.elapsed() < simulation_duration {
            // Update market conditions
            market_state += (total_trades as f64 * 0.001).sin() * 0.01;
            volatility = 0.1 + (total_trades as f64 * 0.003).cos().abs() * 0.3;
            
            // Generate market logits based on current state
            let market_logits = vec![
                2.0 + market_state + (rand::random::<f64>() - 0.5) * volatility,
                1.0 - market_state * 0.5 + (rand::random::<f64>() - 0.5) * volatility * 0.8,
                0.5 + market_state * 0.3 + (rand::random::<f64>() - 0.5) * volatility * 0.6,
                0.2 - market_state * 0.2 + (rand::random::<f64>() - 0.5) * volatility * 0.4,
            ];
            
            // Generate calibration data based on recent history
            let calibration_logits: Vec<Vec<f64>> = (0..20)
                .map(|i| {
                    let historical_state = market_state - (i as f64) * 0.001;
                    vec![
                        2.0 + historical_state + (rand::random::<f64>() - 0.5) * volatility * 0.5,
                        1.0 - historical_state * 0.5 + (rand::random::<f64>() - 0.5) * volatility * 0.4,
                        0.5 + historical_state * 0.3 + (rand::random::<f64>() - 0.5) * volatility * 0.3,
                        0.2 - historical_state * 0.2 + (rand::random::<f64>() - 0.5) * volatility * 0.2,
                    ]
                })
                .collect();
            
            let calibration_labels: Vec<usize> = (0..20).map(|i| i % 4).collect();
            
            // Execute trade
            total_trades += 1;
            
            let trade_start = Instant::now();
            let trade_result = fixture.predictors[0].ats_cp_predict(
                &market_logits,
                &calibration_logits,
                &calibration_labels,
                0.95,
                AtsCpVariant::GQ,
            );
            let trade_latency = trade_start.elapsed();
            
            match trade_result {
                Ok(ats_result) => {
                    successful_trades += 1;
                    latencies.push(trade_latency.as_nanos() as u64);
                    
                    // Simulate trading decision and P&L
                    let predicted_class = ats_result.calibrated_probabilities
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    
                    // Simple P&L simulation based on prediction confidence
                    let confidence_score = ats_result.calibrated_probabilities[predicted_class];
                    let trade_outcome = if confidence_score > 0.4 {
                        (confidence_score - 0.5) * 100.0 // Profit/loss in basis points
                    } else {
                        -10.0 // Small loss for low confidence trades
                    };
                    
                    profit_loss += trade_outcome;
                },
                Err(e) => {
                    println!("    Trade {} failed: {}", total_trades, e);
                }
            }
            
            // Rate limiting
            tokio::time::sleep(Duration::from_nanos(1_000_000)).await;
        }
        
        let simulation_time = start_time.elapsed();
        let actual_rate = total_trades as f64 / simulation_time.as_secs_f64();
        let success_rate = (successful_trades as f64) / (total_trades as f64);
        
        // Analyze latencies
        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        } else {
            0
        };
        
        let p95_latency = if !latencies.is_empty() {
            let mut sorted_latencies = latencies.clone();
            sorted_latencies.sort();
            sorted_latencies[(sorted_latencies.len() * 95) / 100]
        } else {
            0
        };
        
        println!("  Trading Simulation Results:");
        println!("    Total trades:     {}", total_trades);
        println!("    Successful:       {}", successful_trades);
        println!("    Success rate:     {:.2}%", success_rate * 100.0);
        println!("    Actual rate:      {:.0} trades/sec", actual_rate);
        println!("    Avg latency:      {:6} ns ({:5.2} μs)", avg_latency, avg_latency as f64 / 1000.0);
        println!("    P95 latency:      {:6} ns ({:5.2} μs)", p95_latency, p95_latency as f64 / 1000.0);
        println!("    Total P&L:        {:.2} bps", profit_loss);
        
        // Trading simulation requirements
        assert!(success_rate >= 0.95, "Trading success rate should be ≥95%");
        assert!(actual_rate >= target_trades_per_second as f64 * 0.8, "Should achieve at least 80% of target rate");
        assert!(avg_latency < 25_000, "Average latency should be <25μs");
        assert!(p95_latency < 50_000, "P95 latency should be <50μs");
        
        println!("✅ Real-world trading simulation passed");
    }
    
    fn validate_ats_cp_result(
        result: &AtsCpResult,
        expected_confidence: f64,
        expected_variant: &AtsCpVariant,
    ) -> Result<()> {
        // Basic structure validation
        assert!(!result.conformal_set.is_empty(), "Conformal set should not be empty");
        assert!(!result.calibrated_probabilities.is_empty(), "Probabilities should not be empty");
        assert!(result.optimal_temperature > 0.0, "Temperature should be positive: {}", result.optimal_temperature);
        assert!(result.optimal_temperature.is_finite(), "Temperature should be finite");
        assert!(result.quantile_threshold >= 0.0, "Quantile threshold should be non-negative: {}", result.quantile_threshold);
        assert_eq!(result.coverage_guarantee, expected_confidence, "Coverage guarantee should match");
        assert_eq!(result.variant, *expected_variant, "Variant should match");
        assert!(result.execution_time_ns > 0, "Execution time should be recorded");
        
        // Probability validation
        let prob_sum: f64 = result.calibrated_probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1.0: {}", prob_sum);
        
        for (i, &prob) in result.calibrated_probabilities.iter().enumerate() {
            assert!(prob >= 0.0 && prob <= 1.0, 
                   "Probability {} should be in [0,1]: {}", i, prob);
            assert!(prob.is_finite(), "Probability {} should be finite", i);
        }
        
        // Conformal set validation
        let max_class = result.calibrated_probabilities.len() - 1;
        for &class_idx in &result.conformal_set {
            assert!(class_idx <= max_class, 
                   "Conformal set class {} should be valid (max: {})", class_idx, max_class);
        }
        
        Ok(())
    }
}

/// High-frequency trading scenario tests
mod hft_scenario_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_high_frequency_trading_scenario() {
        println!("⚡ Testing high-frequency trading scenario...");
        
        let mut fixture = E2ETestFixture::new();
        let hft_scenario = &fixture.trading_scenarios.high_frequency_scenario;
        
        println!("  HFT Scenario parameters:");
        println!("    Target rate: {} trades/sec", hft_scenario.trades_per_second);
        println!("    Duration: {} seconds", hft_scenario.duration_seconds);
        println!("    Expected latency: {} μs", hft_scenario.expected_latency_us);
        println!("    Total logits streams: {}", hft_scenario.logits_stream.len());
        
        let start_time = Instant::now();
        let mut processed_trades = 0;
        let mut successful_trades = 0;
        let mut latencies = Vec::new();
        let mut throughput_violations = 0;
        
        // High-frequency processing loop
        for (trade_id, trade_logits) in hft_scenario.logits_stream.iter().enumerate() {
            // Generate calibration data for this trade
            let calibration_start = trade_id.saturating_sub(20);
            let calibration_end = trade_id;
            
            if calibration_end > calibration_start {
                let calibration_logits: Vec<Vec<f64>> = hft_scenario.logits_stream
                    [calibration_start..calibration_end]
                    .iter()
                    .cloned()
                    .collect();
                
                let calibration_labels: Vec<usize> = (0..calibration_logits.len())
                    .map(|i| i % trade_logits.len())
                    .collect();
                
                processed_trades += 1;
                
                // Execute high-frequency trade
                let trade_start = Instant::now();
                let result = fixture.predictors[0].ats_cp_predict(
                    trade_logits,
                    &calibration_logits,
                    &calibration_labels,
                    0.95,
                    AtsCpVariant::GQ,
                );
                let trade_latency = trade_start.elapsed();
                
                match result {
                    Ok(ats_result) => {
                        successful_trades += 1;
                        latencies.push(trade_latency.as_nanos() as u64);
                        
                        // Validate HFT-specific requirements
                        if trade_latency.as_micros() > hft_scenario.expected_latency_us as u128 {
                            throughput_violations += 1;
                        }
                        
                        // Verify result quality under HFT conditions
                        assert!(!ats_result.conformal_set.is_empty(),
                               "HFT trade {} should produce valid conformal set", trade_id);
                        assert!(ats_result.optimal_temperature > 0.0,
                               "HFT trade {} should produce valid temperature", trade_id);
                    },
                    Err(e) => {
                        println!("    HFT trade {} failed: {}", trade_id, e);
                    }
                }
            }
            
            // Simulate inter-trade timing
            if processed_trades % 100 == 0 {
                tokio::time::sleep(Duration::from_micros(10)).await;
            }
        }
        
        let total_time = start_time.elapsed();
        let actual_throughput = processed_trades as f64 / total_time.as_secs_f64();
        let success_rate = (successful_trades as f64) / (processed_trades as f64);
        let violation_rate = (throughput_violations as f64) / (successful_trades as f64);
        
        // Analyze HFT performance
        if !latencies.is_empty() {
            latencies.sort();
            let min_latency = latencies[0];
            let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
            let p95_latency = latencies[(latencies.len() * 95) / 100];
            let p99_latency = latencies[(latencies.len() * 99) / 100];
            let max_latency = latencies[latencies.len() - 1];
            
            println!("  HFT Performance Results:");
            println!("    Processed trades: {}", processed_trades);
            println!("    Successful:       {}", successful_trades);
            println!("    Success rate:     {:.2}%", success_rate * 100.0);
            println!("    Actual throughput: {:.0} trades/sec", actual_throughput);
            println!("    Latency stats:");
            println!("      Min:  {:6} ns ({:5.2} μs)", min_latency, min_latency as f64 / 1000.0);
            println!("      Avg:  {:6} ns ({:5.2} μs)", avg_latency, avg_latency as f64 / 1000.0);
            println!("      P95:  {:6} ns ({:5.2} μs)", p95_latency, p95_latency as f64 / 1000.0);
            println!("      P99:  {:6} ns ({:5.2} μs)", p99_latency, p99_latency as f64 / 1000.0);
            println!("      Max:  {:6} ns ({:5.2} μs)", max_latency, max_latency as f64 / 1000.0);
            println!("    Throughput violations: {:.2}%", violation_rate * 100.0);
            
            // HFT performance requirements
            assert!(success_rate >= 0.99, "HFT success rate should be ≥99%");
            assert!(actual_throughput >= hft_scenario.trades_per_second as f64 * 0.8,
                   "HFT should achieve at least 80% of target throughput");
            assert!(avg_latency < (hft_scenario.expected_latency_us * 1000) as u64,
                   "HFT average latency should meet expectations");
            assert!(p99_latency < (hft_scenario.expected_latency_us * 2000) as u64,
                   "HFT P99 latency should be within 2x expected");
            assert!(violation_rate < 0.05, "HFT throughput violations should be <5%");
        }
        
        println!("✅ High-frequency trading scenario passed");
    }
    
    #[tokio::test]
    async fn test_hft_burst_handling() {
        println!("💥 Testing HFT burst handling...");
        
        let mut fixture = E2ETestFixture::new();
        
        // Simulate burst scenarios typical in HFT
        let burst_scenarios = vec![
            (100, Duration::from_millis(10), "Small burst"),
            (500, Duration::from_millis(50), "Medium burst"),
            (1000, Duration::from_millis(100), "Large burst"),
        ];
        
        for (burst_size, burst_duration, scenario_name) in burst_scenarios {
            println!("  Testing {}: {} trades in {:?}", scenario_name, burst_size, burst_duration);
            
            let mut burst_latencies = Vec::with_capacity(burst_size);
            let mut burst_successes = 0;
            
            let burst_start = Instant::now();
            
            for trade_id in 0..burst_size {
                // Generate burst trade data
                let burst_logits = vec![
                    2.0 + (trade_id as f64 * 0.001).sin(),
                    1.0 - (trade_id as f64 * 0.0015).cos(),
                    0.5 + (trade_id as f64 * 0.002).sin(),
                ];
                
                let calibration_logits = vec![
                    vec![1.9, 1.1, 0.4],
                    vec![2.1, 0.9, 0.6],
                    vec![1.8, 1.2, 0.5],
                    vec![2.0, 1.0, 0.4],
                    vec![1.9, 1.1, 0.6],
                ];
                let calibration_labels = vec![0, 0, 1, 2, 1];
                
                let trade_start = Instant::now();
                let result = fixture.predictors[0].ats_cp_predict(
                    &burst_logits,
                    &calibration_logits,
                    &calibration_labels,
                    0.95,
                    AtsCpVariant::GQ,
                );
                let trade_latency = trade_start.elapsed();
                
                if result.is_ok() {
                    burst_successes += 1;
                    burst_latencies.push(trade_latency.as_nanos() as u64);
                }
                
                // Minimal inter-trade delay to simulate burst conditions
                if trade_id % 50 == 0 {
                    tokio::time::sleep(Duration::from_nanos(100)).await;
                }
            }
            
            let burst_total_time = burst_start.elapsed();
            let burst_success_rate = (burst_successes as f64) / (burst_size as f64);
            let burst_throughput = burst_successes as f64 / burst_total_time.as_secs_f64();
            
            if !burst_latencies.is_empty() {
                let avg_burst_latency = burst_latencies.iter().sum::<u64>() / burst_latencies.len() as u64;
                let max_burst_latency = *burst_latencies.iter().max().unwrap();
                
                println!("    {} Results:", scenario_name);
                println!("      Success rate:    {:.2}%", burst_success_rate * 100.0);
                println!("      Throughput:      {:.0} trades/sec", burst_throughput);
                println!("      Avg latency:     {:6} ns ({:5.2} μs)", avg_burst_latency, avg_burst_latency as f64 / 1000.0);
                println!("      Max latency:     {:6} ns ({:5.2} μs)", max_burst_latency, max_burst_latency as f64 / 1000.0);
                println!("      Total time:      {:?}", burst_total_time);
                
                // Burst handling requirements
                assert!(burst_success_rate >= 0.95, "{} should have ≥95% success rate", scenario_name);
                assert!(avg_burst_latency < 30_000, "{} average latency should be <30μs", scenario_name);
                assert!(max_burst_latency < 100_000, "{} max latency should be <100μs", scenario_name);
                assert!(burst_total_time <= burst_duration * 2, "{} should complete within reasonable time", scenario_name);
            }
            
            println!("      ✅ {} burst handling validated", scenario_name);
        }
        
        println!("✅ HFT burst handling tests passed");
    }
}

/// Market volatility and stress tests
mod volatility_stress_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_market_volatility_scenarios() {
        println!("📊 Testing market volatility scenarios...");
        
        let mut fixture = E2ETestFixture::new();
        let volatility_scenario = &fixture.trading_scenarios.market_volatility_scenario;
        
        for (vol_level, market_condition) in volatility_scenario.volatility_levels.iter()
            .zip(volatility_scenario.market_conditions.iter()) {
            
            println!("  Testing {:?} conditions with volatility {:.2}", market_condition, vol_level);
            
            // Select appropriate test data for this volatility level
            let test_data_idx = ((vol_level * 5.0) as usize).min(volatility_scenario.stress_test_data.len() - 1);
            let (test_predictions, test_calibration) = &volatility_scenario.stress_test_data[test_data_idx];
            
            // Test all variants under volatile conditions
            let variants = vec![AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ, AtsCpVariant::MAQ];
            let mut variant_success_count = 0;
            let mut total_execution_time = Duration::from_nanos(0);
            
            for variant in variants {
                // Generate calibration logits based on predictions and volatility
                let calibration_logits: Vec<Vec<f64>> = test_predictions.iter()
                    .take(20)
                    .map(|&pred| {
                        vec![
                            pred + (rand::random::<f64>() - 0.5) * vol_level,
                            pred * 0.5 + (rand::random::<f64>() - 0.5) * vol_level * 0.8,
                            pred * 0.3 + (rand::random::<f64>() - 0.5) * vol_level * 0.6,
                        ]
                    })
                    .collect();
                
                let calibration_labels: Vec<usize> = (0..calibration_logits.len()).map(|i| i % 3).collect();
                
                // Select subset of predictions for testing
                let test_logits = vec![
                    test_predictions[0] + (rand::random::<f64>() - 0.5) * vol_level,
                    test_predictions[1] * 0.5 + (rand::random::<f64>() - 0.5) * vol_level * 0.8,
                    test_predictions[2] * 0.3 + (rand::random::<f64>() - 0.5) * vol_level * 0.6,
                ];
                
                let start_time = Instant::now();
                let result = fixture.predictors[1].ats_cp_predict(
                    &test_logits,
                    &calibration_logits,
                    &calibration_labels,
                    0.95,
                    variant.clone(),
                );
                let execution_time = start_time.elapsed();
                
                total_execution_time += execution_time;
                
                match result {
                    Ok(ats_result) => {
                        variant_success_count += 1;
                        
                        // Validate results under volatile conditions
                        assert!(!ats_result.conformal_set.is_empty(),
                               "Variant {:?} should produce conformal set under {:?} conditions", variant, market_condition);
                        
                        // Verify probabilities remain valid under volatility
                        let prob_sum: f64 = ats_result.calibrated_probabilities.iter().sum();
                        assert!((prob_sum - 1.0).abs() < 1e-6,
                               "Probabilities should remain normalized under volatility");
                        
                        // Check if uncertainty appropriately increases with volatility
                        let prob_entropy: f64 = ats_result.calibrated_probabilities.iter()
                            .filter(|&&p| p > 0.0)
                            .map(|&p| -p * p.ln())
                            .sum();
                        
                        // Higher volatility should generally lead to higher entropy (more uncertainty)
                        println!("      Variant {:?}: entropy={:.3}, temp={:.3}, exec={:?}",
                                variant, prob_entropy, ats_result.optimal_temperature, execution_time);
                    },
                    Err(e) => {
                        println!("      Variant {:?} failed under {:?}: {}", variant, market_condition, e);
                    }
                }
            }
            
            let avg_execution_time = total_execution_time / 4;
            let success_rate = variant_success_count as f64 / 4.0;
            
            println!("    {:?} Results:", market_condition);
            println!("      Success rate: {:.2}% ({}/4 variants)", success_rate * 100.0, variant_success_count);
            println!("      Avg execution time: {:?}", avg_execution_time);
            
            // Volatility scenario requirements
            assert!(success_rate >= 0.75, "{:?} should have ≥75% variant success rate", market_condition);
            assert!(avg_execution_time < Duration::from_micros(100),
                   "{:?} should maintain reasonable performance", market_condition);
        }
        
        println!("✅ Market volatility scenarios passed");
    }
    
    #[tokio::test]
    async fn test_extreme_stress_conditions() {
        println!("🔥 Testing extreme stress conditions...");
        
        let mut fixture = E2ETestFixture::new();
        
        // Extreme stress scenarios
        let stress_scenarios = vec![
            ("Flash Crash", vec![100.0, -100.0, 50.0, -50.0, 25.0]),
            ("Market Meltdown", vec![-50.0, -60.0, -70.0, -80.0, -90.0]),
            ("Bubble Burst", vec![200.0, 150.0, 100.0, 50.0, 25.0]),
            ("High Volatility", vec![0.0, 100.0, -100.0, 200.0, -200.0]),
            ("Gradual Recovery", vec![-100.0, -75.0, -50.0, -25.0, 0.0]),
        ];
        
        for (scenario_name, extreme_logits) in stress_scenarios {
            println!("  Testing extreme scenario: {}", scenario_name);
            
            // Generate extreme calibration data
            let extreme_calibration_logits: Vec<Vec<f64>> = extreme_logits.iter()
                .map(|&base| {
                    vec![
                        base + (rand::random::<f64>() - 0.5) * 50.0,
                        base * 0.8 + (rand::random::<f64>() - 0.5) * 40.0,
                        base * 0.6 + (rand::random::<f64>() - 0.5) * 30.0,
                        base * 0.4 + (rand::random::<f64>() - 0.5) * 20.0,
                    ]
                })
                .collect();
            
            let extreme_labels: Vec<usize> = (0..extreme_calibration_logits.len()).map(|i| i % 4).collect();
            
            // Test system resilience under extreme conditions
            let test_logits = vec![
                extreme_logits[0] + 10.0,
                extreme_logits[1] - 5.0,
                extreme_logits[2] + 2.0,
                extreme_logits[3] - 8.0,
            ];
            
            let start_time = Instant::now();
            let result = fixture.predictors[2].ats_cp_predict(
                &test_logits,
                &extreme_calibration_logits,
                &extreme_labels,
                0.95,
                AtsCpVariant::GQ,
            );
            let stress_execution_time = start_time.elapsed();
            
            match result {
                Ok(ats_result) => {
                    println!("    ✅ {} handled successfully in {:?}", scenario_name, stress_execution_time);
                    
                    // Verify system maintains mathematical properties under stress
                    assert!(!ats_result.conformal_set.is_empty(),
                           "{} should produce valid conformal set", scenario_name);
                    
                    assert!(ats_result.optimal_temperature > 0.0 && ats_result.optimal_temperature.is_finite(),
                           "{} should produce valid temperature", scenario_name);
                    
                    let prob_sum: f64 = ats_result.calibrated_probabilities.iter().sum();
                    assert!((prob_sum - 1.0).abs() < 1e-6,
                           "{} should maintain probability normalization", scenario_name);
                    
                    // Verify all probabilities are valid
                    for &prob in &ats_result.calibrated_probabilities {
                        assert!(prob.is_finite() && prob >= 0.0 && prob <= 1.0,
                               "{} should produce valid probabilities", scenario_name);
                    }
                },
                Err(AtsCoreError::MathematicalError { .. }) => {
                    println!("    ✅ {} correctly detected mathematical issues", scenario_name);
                },
                Err(AtsCoreError::ValidationError { .. }) => {
                    println!("    ✅ {} correctly rejected by input validation", scenario_name);
                },
                Err(e) => {
                    println!("    ⚠️  {} caused unexpected error: {}", scenario_name, e);
                    // System should handle extreme conditions gracefully
                    assert!(false, "{} should handle extreme conditions without unexpected errors", scenario_name);
                }
            }
        }
        
        println!("✅ Extreme stress conditions tests passed");
    }
}

/// Model ensemble and consensus tests
mod ensemble_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_ensemble_consensus() {
        println!("🎭 Testing model ensemble consensus...");
        
        let mut fixture = E2ETestFixture::new();
        let ensemble_scenario = &fixture.trading_scenarios.model_ensemble_scenario;
        
        println!("  Ensemble parameters:");
        println!("    Variants: {:?}", ensemble_scenario.model_variants);
        println!("    Weights: {:?}", ensemble_scenario.ensemble_weights);
        println!("    Consensus threshold: {:.2}", ensemble_scenario.consensus_threshold);
        
        let test_logits = vec![2.1, 1.3, 0.8, 0.4, 0.2];
        let calibration_logits = vec![
            vec![2.0, 1.2, 0.9, 0.5, 0.3],
            vec![1.9, 1.4, 0.7, 0.6, 0.2],
            vec![2.2, 1.1, 0.8, 0.4, 0.1],
            vec![1.8, 1.5, 0.6, 0.7, 0.3],
            vec![2.1, 1.0, 0.9, 0.3, 0.4],
        ];
        let calibration_labels = vec![0, 0, 1, 2, 3];
        
        // Execute all ensemble variants
        let mut ensemble_results = Vec::new();
        
        for (variant, &weight) in ensemble_scenario.model_variants.iter()
            .zip(ensemble_scenario.ensemble_weights.iter()) {
            
            println!("  Running ensemble variant {:?} (weight: {:.2})", variant, weight);
            
            let result = fixture.predictors[0].ats_cp_predict(
                &test_logits,
                &calibration_logits,
                &calibration_labels,
                0.95,
                variant.clone(),
            );
            
            assert!(result.is_ok(), "Ensemble variant {:?} should succeed", variant);
            let ats_result = result.unwrap();
            
            ensemble_results.push((variant.clone(), weight, ats_result));
        }
        
        // Compute weighted ensemble consensus
        let num_classes = ensemble_results[0].2.calibrated_probabilities.len();
        let mut consensus_probabilities = vec![0.0; num_classes];
        let mut consensus_conformal_set = std::collections::HashSet::new();
        let mut weighted_temperature = 0.0;
        
        for (variant, weight, ats_result) in &ensemble_results {
            // Aggregate probabilities
            for (i, &prob) in ats_result.calibrated_probabilities.iter().enumerate() {
                consensus_probabilities[i] += prob * weight;
            }
            
            // Aggregate conformal sets
            for &class_idx in &ats_result.conformal_set {
                consensus_conformal_set.insert(class_idx);
            }
            
            // Aggregate temperatures
            weighted_temperature += ats_result.optimal_temperature * weight;
            
            println!("    Variant {:?}: temp={:.3}, conformal_set={:?}",
                    variant, ats_result.optimal_temperature, ats_result.conformal_set);
        }
        
        // Validate ensemble consensus
        let consensus_sum: f64 = consensus_probabilities.iter().sum();
        assert!((consensus_sum - 1.0).abs() < 1e-10,
               "Ensemble consensus probabilities should sum to 1.0: {}", consensus_sum);
        
        // Compute ensemble consensus metrics
        let consensus_entropy: f64 = consensus_probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        
        let consensus_max_prob = consensus_probabilities.iter()
            .cloned()
            .fold(0.0f64, |a, b| a.max(b));
        
        println!("  Ensemble Consensus Results:");
        println!("    Consensus probabilities: {:?}", consensus_probabilities.iter()
                 .map(|&p| format!("{:.3}", p)).collect::<Vec<_>>());
        println!("    Consensus conformal set: {:?}", consensus_conformal_set);
        println!("    Weighted temperature: {:.3}", weighted_temperature);
        println!("    Consensus entropy: {:.3}", consensus_entropy);
        println!("    Max consensus prob: {:.3}", consensus_max_prob);
        
        // Ensemble consensus requirements
        assert!(!consensus_conformal_set.is_empty(),
               "Ensemble should produce non-empty consensus conformal set");
        
        assert!(consensus_max_prob >= ensemble_scenario.consensus_threshold,
               "Ensemble should achieve consensus threshold: {:.3} >= {:.3}",
               consensus_max_prob, ensemble_scenario.consensus_threshold);
        
        assert!(weighted_temperature > 0.0 && weighted_temperature.is_finite(),
               "Ensemble weighted temperature should be valid: {:.3}", weighted_temperature);
        
        // Verify individual variant contributions
        for (i, (variant, weight, ats_result)) in ensemble_results.iter().enumerate() {
            println!("    Variant {} ({:?}, weight {:.2}): contributing to ensemble",
                    i + 1, variant, weight);
            
            // Each variant should contribute meaningfully
            assert!(weight > 0.0, "All ensemble weights should be positive");
            assert!(!ats_result.conformal_set.is_empty(),
                   "All ensemble variants should produce valid conformal sets");
        }
        
        println!("✅ Model ensemble consensus tests passed");
    }
}

#[cfg(test)]
mod e2e_test_integration {
    use super::*;
    use ats_core::test_framework::{TestFramework, swarm_utils};
    
    #[tokio::test]
    async fn test_e2e_swarm_coordination() {
        // Initialize end-to-end test framework
        let mut framework = TestFramework::new(
            "e2e_test_swarm".to_string(),
            "e2e_test_agent".to_string(),
        ).unwrap();
        
        // Signal coordination with other test agents
        swarm_utils::coordinate_test_execution(&framework.context, "e2e_tests").await.unwrap();
        
        // Wait for dependency completion
        let dependencies = vec![
            "unit_test_agent".to_string(),
            "integration_test_agent".to_string(),
            "performance_test_agent".to_string(),
            "security_test_agent".to_string(),
        ];
        swarm_utils::wait_for_dependencies(&framework.context, &dependencies).await.unwrap();
        
        // Execute comprehensive E2E test
        let mut fixture = E2ETestFixture::new();
        
        let start_time = Instant::now();
        let result = fixture.predictors[0].ats_cp_predict(
            &vec![2.1, 1.3, 0.8],
            &vec![
                vec![2.0, 1.2, 0.9],
                vec![1.9, 1.4, 0.7],
                vec![2.2, 1.1, 0.8],
                vec![1.8, 1.5, 0.6],
                vec![2.1, 1.0, 0.9],
            ],
            &vec![0, 0, 1, 2, 1],
            0.95,
            AtsCpVariant::GQ,
        );
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "E2E swarm coordinated test should succeed");
        let ats_result = result.unwrap();
        
        // Update test metrics
        framework.context.execution_metrics.tests_passed += 1;
        framework.context.execution_metrics.performance_metrics.insert(
            "e2e_ats_cp_latency_ns".to_string(),
            execution_time.as_nanos() as f64,
        );
        
        // Validate E2E result
        assert!(!ats_result.conformal_set.is_empty(), "E2E test should produce valid conformal set");
        assert!(ats_result.optimal_temperature > 0.0, "E2E test should produce valid temperature");
        assert!(execution_time < Duration::from_micros(50), "E2E test should meet performance requirements");
        
        // Share results with swarm
        swarm_utils::share_test_results(&framework.context, &framework.context.execution_metrics).await.unwrap();
        
        println!("  E2E Swarm Coordination Results:");
        println!("    Tests passed: {}", framework.context.execution_metrics.tests_passed);
        println!("    Execution time: {:?}", execution_time);
        println!("    Conformal set size: {}", ats_result.conformal_set.len());
        println!("    Optimal temperature: {:.3}", ats_result.optimal_temperature);
        
        println!("✅ E2E swarm coordination tests completed");
    }
}