//! Benchmarking Framework for Quantitative Analysis Performance
//!
//! This module provides comprehensive benchmarks for neural trading strategy
//! validation, backtesting performance, and statistical analysis speed.

use std::time::{Instant, Duration};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{
    quantitative_analysis::*,
    statistical_validation::*,
    factor_attribution::*,
    risk_metrics::*,
};

/// Comprehensive benchmarking suite for quantitative analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantitativeBenchmarks {
    /// Backtesting performance benchmarks
    pub backtest_benchmarks: BacktestBenchmarks,
    /// Statistical validation benchmarks
    pub validation_benchmarks: ValidationBenchmarks,
    /// Factor analysis benchmarks
    pub factor_benchmarks: FactorBenchmarks,
    /// Risk metrics benchmarks
    pub risk_benchmarks: RiskBenchmarks,
    /// End-to-end benchmarks
    pub e2e_benchmarks: EndToEndBenchmarks,
}

/// Backtesting performance benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestBenchmarks {
    /// Time to process N years of data
    pub data_processing_speed: HashMap<String, Duration>,
    /// Memory usage for different dataset sizes
    pub memory_usage: HashMap<String, usize>,
    /// Transaction cost calculation speed
    pub transaction_cost_speed: Duration,
    /// Portfolio update speed
    pub portfolio_update_speed: Duration,
    /// Performance metrics calculation speed
    pub metrics_calculation_speed: Duration,
}

/// Statistical validation benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationBenchmarks {
    /// Prediction accuracy test speed
    pub accuracy_test_speed: Duration,
    /// Statistical significance test speed
    pub significance_test_speed: Duration,
    /// Distribution test speed
    pub distribution_test_speed: Duration,
    /// Bootstrap analysis speed
    pub bootstrap_speed: Duration,
    /// Cross-validation speed
    pub cross_validation_speed: Duration,
}

/// Factor analysis benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorBenchmarks {
    /// Factor exposure calculation speed
    pub exposure_calculation_speed: Duration,
    /// Style analysis speed
    pub style_analysis_speed: Duration,
    /// Performance attribution speed
    pub attribution_speed: Duration,
    /// Risk decomposition speed
    pub risk_decomposition_speed: Duration,
    /// Regime analysis speed
    pub regime_analysis_speed: Duration,
}

/// Risk metrics benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskBenchmarks {
    /// VaR calculation speed by method
    pub var_calculation_speed: HashMap<String, Duration>,
    /// Stress testing speed
    pub stress_testing_speed: Duration,
    /// Monte Carlo simulation speed
    pub monte_carlo_speed: Duration,
    /// Extreme value analysis speed
    pub extreme_value_speed: Duration,
    /// Liquidity risk calculation speed
    pub liquidity_risk_speed: Duration,
}

/// End-to-end benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndToEndBenchmarks {
    /// Full analysis pipeline speed
    pub full_pipeline_speed: Duration,
    /// Memory peak usage
    pub peak_memory_usage: usize,
    /// Throughput (analyses per second)
    pub throughput: f64,
    /// Latency percentiles
    pub latency_percentiles: HashMap<String, Duration>,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of data points for testing
    pub data_points: usize,
    /// Number of assets for multi-asset tests
    pub num_assets: usize,
    /// Number of factors for factor analysis
    pub num_factors: usize,
    /// Number of Monte Carlo simulations
    pub mc_simulations: usize,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Whether to include memory profiling
    pub profile_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            data_points: 10000,   // ~40 years of daily data
            num_assets: 100,      // 100 assets
            num_factors: 20,      // 20 factors
            mc_simulations: 50000, // 50k simulations
            iterations: 10,       // 10 benchmark runs
            profile_memory: true,
        }
    }
}

/// Benchmark result for individual test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub duration: Duration,
    pub memory_usage: Option<usize>,
    pub throughput: Option<f64>,
    pub success: bool,
    pub error_message: Option<String>,
}

impl QuantitativeBenchmarks {
    /// Run comprehensive benchmarks
    pub fn run_benchmarks(config: BenchmarkConfig) -> Self {
        println!("🚀 Starting Quantitative Analysis Benchmarks");
        println!("   Data points: {}", config.data_points);
        println!("   Assets: {}", config.num_assets);
        println!("   Factors: {}", config.num_factors);
        println!("   MC Simulations: {}", config.mc_simulations);
        println!("   Iterations: {}", config.iterations);
        
        let mut benchmarks = Self {
            backtest_benchmarks: BacktestBenchmarks::default(),
            validation_benchmarks: ValidationBenchmarks::default(),
            factor_benchmarks: FactorBenchmarks::default(),
            risk_benchmarks: RiskBenchmarks::default(),
            e2e_benchmarks: EndToEndBenchmarks::default(),
        };

        // Generate test data
        let test_data = Self::generate_test_data(&config);
        
        // Run backtesting benchmarks
        benchmarks.backtest_benchmarks = Self::benchmark_backtesting(&config, &test_data);
        
        // Run validation benchmarks
        benchmarks.validation_benchmarks = Self::benchmark_validation(&config, &test_data);
        
        // Run factor analysis benchmarks
        benchmarks.factor_benchmarks = Self::benchmark_factor_analysis(&config, &test_data);
        
        // Run risk metrics benchmarks
        benchmarks.risk_benchmarks = Self::benchmark_risk_metrics(&config, &test_data);
        
        // Run end-to-end benchmarks
        benchmarks.e2e_benchmarks = Self::benchmark_end_to_end(&config, &test_data);
        
        benchmarks
    }

    /// Generate synthetic test data
    fn generate_test_data(config: &BenchmarkConfig) -> TestData {
        use rand::{Rng, SeedableRng};
        use rand_distr::{Distribution, Normal};
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.01).unwrap();
        
        // Generate returns
        let returns: Vec<f64> = (0..config.data_points)
            .map(|_| normal.sample(&mut rng))
            .collect();
        
        // Generate prices from returns
        let mut prices = vec![100.0]; // Starting price
        for &ret in &returns {
            let new_price = prices.last().unwrap() * (1.0 + ret);
            prices.push(new_price);
        }
        
        // Generate benchmark returns
        let benchmark_returns: Vec<f64> = (0..config.data_points)
            .map(|_| normal.sample(&mut rng) * 0.8) // Lower volatility benchmark
            .collect();
        
        // Generate factor returns
        let mut factor_returns = HashMap::new();
        for i in 0..config.num_factors {
            let factor_name = format!("factor_{}", i);
            let factor_data: Vec<f64> = (0..config.data_points)
                .map(|_| normal.sample(&mut rng) * 0.5)
                .collect();
            factor_returns.insert(factor_name, factor_data);
        }
        
        // Generate trading signals
        let signals: Vec<TradingSignal> = (0..config.data_points)
            .map(|i| TradingSignal {
                timestamp: chrono::Utc::now() + chrono::Duration::seconds(i as i64),
                symbol: "TEST".to_string(),
                signal_strength: rng.gen_range(-1.0..1.0),
                confidence: rng.gen_range(0.5..1.0),
                prediction_horizon: chrono::Duration::hours(1),
                expected_return: normal.sample(&mut rng),
                predicted_volatility: rng.gen_range(0.01..0.05),
            })
            .collect();
        
        TestData {
            returns,
            prices,
            benchmark_returns,
            factor_returns,
            signals,
            confidence: (0..config.data_points).map(|_| rng.gen_range(0.5..1.0)).collect(),
        }
    }

    /// Benchmark backtesting performance
    fn benchmark_backtesting(config: &BenchmarkConfig, data: &TestData) -> BacktestBenchmarks {
        println!("📊 Benchmarking Backtesting Performance...");
        
        let mut benchmarks = BacktestBenchmarks::default();
        
        // Benchmark data processing speed
        let mut data_processing_times = HashMap::new();
        for &size in &[1000, 5000, 10000, 25000] {
            if size <= data.returns.len() {
                let start = Instant::now();
                let subset = &data.returns[..size];
                let _processed: Vec<f64> = subset.iter().map(|r| r * 252.0).collect(); // Annualized
                let duration = start.elapsed();
                data_processing_times.insert(format!("{}_points", size), duration);
            }
        }
        benchmarks.data_processing_speed = data_processing_times;
        
        // Benchmark transaction cost calculation
        let start = Instant::now();
        let cost_model = TransactionCostModel {
            spread_cost: 0.001,
            market_impact: MarketImpact {
                linear_coeff: 0.1,
                sqrt_coeff: 0.5,
                decay_rate: 0.1,
            },
            commission: 0.005,
            borrow_cost: 0.02,
            slippage: SlippageModel {
                base_slippage: 0.0001,
                volume_impact: 0.001,
                volatility_multiplier: 1.5,
            },
        };
        
        let mut backtest_engine = BacktestEngine::new(cost_model);
        for signal in &data.signals[..100] { // Test with first 100 signals
            let _cost = backtest_engine.calculate_transaction_costs(
                1000.0, 
                &PricePoint {
                    timestamp: signal.timestamp,
                    open: 100.0,
                    high: 101.0,
                    low: 99.0,
                    close: 100.5,
                    volume: 10000.0,
                    bid: 100.0,
                    ask: 100.1,
                },
                signal
            );
        }
        benchmarks.transaction_cost_speed = start.elapsed();
        
        // Benchmark portfolio updates
        let start = Instant::now();
        for _ in 0..1000 {
            backtest_engine.update_portfolio_value();
        }
        benchmarks.portfolio_update_speed = start.elapsed();
        
        // Benchmark metrics calculation
        let start = Instant::now();
        backtest_engine.calculate_metrics();
        benchmarks.metrics_calculation_speed = start.elapsed();
        
        benchmarks
    }

    /// Benchmark statistical validation
    fn benchmark_validation(config: &BenchmarkConfig, data: &TestData) -> ValidationBenchmarks {
        println!("📈 Benchmarking Statistical Validation...");
        
        let mut benchmarks = ValidationBenchmarks::default();
        let mut validator = StatisticalValidator::new();
        
        // Benchmark accuracy tests
        let start = Instant::now();
        let _results = validator.validate_predictions(
            &data.returns[..1000],
            &data.benchmark_returns[..1000],
            &data.confidence[..1000]
        );
        benchmarks.accuracy_test_speed = start.elapsed();
        
        // Benchmark significance tests
        let start = Instant::now();
        validator.run_significance_tests(&data.returns[..1000], &data.benchmark_returns[..1000]);
        benchmarks.significance_test_speed = start.elapsed();
        
        // Benchmark distribution tests
        let start = Instant::now();
        validator.run_distribution_tests(&data.returns[..1000], &data.benchmark_returns[..1000]);
        benchmarks.distribution_test_speed = start.elapsed();
        
        // Benchmark bootstrap analysis (simplified)
        let start = Instant::now();
        for _ in 0..100 { // 100 bootstrap samples
            let _sample: Vec<f64> = data.returns[..100].iter().cloned().collect();
        }
        benchmarks.bootstrap_speed = start.elapsed();
        
        // Benchmark cross-validation (simplified)
        let start = Instant::now();
        let fold_size = data.returns.len() / 5;
        for i in 0..5 {
            let start_idx = i * fold_size;
            let end_idx = (start_idx + fold_size).min(data.returns.len());
            let _fold = &data.returns[start_idx..end_idx];
        }
        benchmarks.cross_validation_speed = start.elapsed();
        
        benchmarks
    }

    /// Benchmark factor analysis
    fn benchmark_factor_analysis(config: &BenchmarkConfig, data: &TestData) -> FactorBenchmarks {
        println!("⚖️ Benchmarking Factor Analysis...");
        
        let mut benchmarks = FactorBenchmarks::default();
        let mut factor_engine = FactorAttributionEngine::new();
        
        // Benchmark factor exposure calculation
        let start = Instant::now();
        factor_engine.calculate_factor_exposures(&data.returns[..1000], &data.factor_returns);
        benchmarks.exposure_calculation_speed = start.elapsed();
        
        // Benchmark style analysis
        let start = Instant::now();
        factor_engine.perform_style_analysis(
            &data.returns[..1000],
            &data.benchmark_returns[..1000],
            &data.factor_returns
        );
        benchmarks.style_analysis_speed = start.elapsed();
        
        // Benchmark performance attribution
        let start = Instant::now();
        factor_engine.calculate_performance_attribution(
            &data.returns[..1000],
            &data.benchmark_returns[..1000],
            &data.factor_returns
        );
        benchmarks.attribution_speed = start.elapsed();
        
        // Benchmark risk decomposition
        let start = Instant::now();
        factor_engine.calculate_risk_decomposition(&data.factor_returns);
        benchmarks.risk_decomposition_speed = start.elapsed();
        
        // Benchmark regime analysis
        let start = Instant::now();
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..1000)
            .map(|i| chrono::Utc::now() + chrono::Duration::seconds(i))
            .collect();
        factor_engine.analyze_regime_performance(
            &data.returns[..1000],
            &data.benchmark_returns[..1000],
            &timestamps
        );
        benchmarks.regime_analysis_speed = start.elapsed();
        
        benchmarks
    }

    /// Benchmark risk metrics
    fn benchmark_risk_metrics(config: &BenchmarkConfig, data: &TestData) -> RiskBenchmarks {
        println!("🎯 Benchmarking Risk Metrics...");
        
        let mut benchmarks = RiskBenchmarks::default();
        let mut risk_engine = RiskMetricsEngine::new();
        
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..data.returns.len())
            .map(|i| chrono::Utc::now() + chrono::Duration::seconds(i as i64))
            .collect();
        
        // Benchmark VaR calculations
        let mut var_speeds = HashMap::new();
        
        // Historical VaR
        let start = Instant::now();
        risk_engine.calculate_historical_var(&data.returns[..1000]);
        var_speeds.insert("historical".to_string(), start.elapsed());
        
        // Parametric VaR
        let start = Instant::now();
        risk_engine.calculate_parametric_var(&data.returns[..1000]);
        var_speeds.insert("parametric".to_string(), start.elapsed());
        
        // Monte Carlo VaR
        let start = Instant::now();
        risk_engine.calculate_monte_carlo_var(&data.returns[..1000]);
        var_speeds.insert("monte_carlo".to_string(), start.elapsed());
        
        benchmarks.var_calculation_speed = var_speeds;
        
        // Benchmark stress testing
        let start = Instant::now();
        risk_engine.perform_stress_tests(&data.returns[..1000], &data.prices[..1000]);
        benchmarks.stress_testing_speed = start.elapsed();
        
        // Benchmark Monte Carlo simulation (simplified)
        let start = Instant::now();
        let _simulations: Vec<f64> = (0..config.mc_simulations)
            .map(|_| data.returns[0] * (1.0 + rand::random::<f64>() * 0.1))
            .collect();
        benchmarks.monte_carlo_speed = start.elapsed();
        
        // Benchmark extreme value analysis
        let start = Instant::now();
        risk_engine.analyze_extreme_values(&data.returns[..1000]);
        benchmarks.extreme_value_speed = start.elapsed();
        
        // Benchmark liquidity risk
        let start = Instant::now();
        risk_engine.calculate_liquidity_risk(&data.prices[..1000]);
        benchmarks.liquidity_risk_speed = start.elapsed();
        
        benchmarks
    }

    /// Benchmark end-to-end pipeline
    fn benchmark_end_to_end(config: &BenchmarkConfig, data: &TestData) -> EndToEndBenchmarks {
        println!("🔗 Benchmarking End-to-End Pipeline...");
        
        let mut latency_measurements = Vec::new();
        
        for _ in 0..config.iterations {
            let start = Instant::now();
            
            // Full analysis pipeline
            let mut backtest_engine = BacktestEngine::new(TransactionCostModel {
                spread_cost: 0.001,
                market_impact: MarketImpact {
                    linear_coeff: 0.1,
                    sqrt_coeff: 0.5,
                    decay_rate: 0.1,
                },
                commission: 0.005,
                borrow_cost: 0.02,
                slippage: SlippageModel {
                    base_slippage: 0.0001,
                    volume_impact: 0.001,
                    volatility_multiplier: 1.5,
                },
            });
            
            let mut validator = StatisticalValidator::new();
            let mut factor_engine = FactorAttributionEngine::new();
            let mut risk_engine = RiskMetricsEngine::new();
            
            // Run backtesting
            let _backtest_results = backtest_engine.run_backtest(1000000.0);
            
            // Run validation
            let _validation_results = validator.validate_predictions(
                &data.returns[..500],
                &data.benchmark_returns[..500],
                &data.confidence[..500]
            );
            
            // Run factor analysis
            let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..500)
                .map(|i| chrono::Utc::now() + chrono::Duration::seconds(i))
                .collect();
            let _factor_results = factor_engine.analyze_factors(
                &data.returns[..500],
                &data.benchmark_returns[..500],
                &data.factor_returns,
                &timestamps
            );
            
            // Run risk metrics
            let _risk_results = risk_engine.calculate_risk_metrics(
                &data.returns[..500],
                &data.prices[..500],
                &timestamps
            );
            
            let duration = start.elapsed();
            latency_measurements.push(duration);
        }
        
        // Calculate statistics
        latency_measurements.sort();
        let mut latency_percentiles = HashMap::new();
        latency_percentiles.insert("p50".to_string(), latency_measurements[latency_measurements.len() / 2]);
        latency_percentiles.insert("p90".to_string(), latency_measurements[latency_measurements.len() * 9 / 10]);
        latency_percentiles.insert("p95".to_string(), latency_measurements[latency_measurements.len() * 95 / 100]);
        latency_percentiles.insert("p99".to_string(), latency_measurements[latency_measurements.len() * 99 / 100]);
        
        let total_time: Duration = latency_measurements.iter().sum();
        let avg_time = total_time / config.iterations as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();
        
        EndToEndBenchmarks {
            full_pipeline_speed: avg_time,
            peak_memory_usage: 0, // Would need memory profiling
            throughput,
            latency_percentiles,
        }
    }

    /// Print benchmark results
    pub fn print_results(&self) {
        println!("\n🎯 QUANTITATIVE ANALYSIS BENCHMARK RESULTS");
        println!("==========================================");
        
        // Backtesting results
        println!("\n📊 Backtesting Performance:");
        for (test, duration) in &self.backtest_benchmarks.data_processing_speed {
            println!("   {} processing: {:?}", test, duration);
        }
        println!("   Transaction cost calculation: {:?}", self.backtest_benchmarks.transaction_cost_speed);
        println!("   Portfolio updates: {:?}", self.backtest_benchmarks.portfolio_update_speed);
        println!("   Metrics calculation: {:?}", self.backtest_benchmarks.metrics_calculation_speed);
        
        // Validation results
        println!("\n📈 Statistical Validation Performance:");
        println!("   Accuracy tests: {:?}", self.validation_benchmarks.accuracy_test_speed);
        println!("   Significance tests: {:?}", self.validation_benchmarks.significance_test_speed);
        println!("   Distribution tests: {:?}", self.validation_benchmarks.distribution_test_speed);
        println!("   Bootstrap analysis: {:?}", self.validation_benchmarks.bootstrap_speed);
        println!("   Cross-validation: {:?}", self.validation_benchmarks.cross_validation_speed);
        
        // Factor analysis results
        println!("\n⚖️ Factor Analysis Performance:");
        println!("   Factor exposures: {:?}", self.factor_benchmarks.exposure_calculation_speed);
        println!("   Style analysis: {:?}", self.factor_benchmarks.style_analysis_speed);
        println!("   Performance attribution: {:?}", self.factor_benchmarks.attribution_speed);
        println!("   Risk decomposition: {:?}", self.factor_benchmarks.risk_decomposition_speed);
        println!("   Regime analysis: {:?}", self.factor_benchmarks.regime_analysis_speed);
        
        // Risk metrics results
        println!("\n🎯 Risk Metrics Performance:");
        for (method, duration) in &self.risk_benchmarks.var_calculation_speed {
            println!("   {} VaR: {:?}", method, duration);
        }
        println!("   Stress testing: {:?}", self.risk_benchmarks.stress_testing_speed);
        println!("   Monte Carlo: {:?}", self.risk_benchmarks.monte_carlo_speed);
        println!("   Extreme value analysis: {:?}", self.risk_benchmarks.extreme_value_speed);
        println!("   Liquidity risk: {:?}", self.risk_benchmarks.liquidity_risk_speed);
        
        // End-to-end results
        println!("\n🔗 End-to-End Performance:");
        println!("   Full pipeline: {:?}", self.e2e_benchmarks.full_pipeline_speed);
        println!("   Throughput: {:.2} analyses/sec", self.e2e_benchmarks.throughput);
        for (percentile, duration) in &self.e2e_benchmarks.latency_percentiles {
            println!("   Latency {}: {:?}", percentile, duration);
        }
        
        println!("\n✅ Benchmark completed successfully!");
    }

    /// Save results to JSON
    pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(filename, json)?;
        println!("📄 Benchmark results saved to {}", filename);
        Ok(())
    }
}

/// Test data structure
#[derive(Debug)]
struct TestData {
    returns: Vec<f64>,
    prices: Vec<f64>,
    benchmark_returns: Vec<f64>,
    factor_returns: HashMap<String, Vec<f64>>,
    signals: Vec<TradingSignal>,
    confidence: Vec<f64>,
}

// Default implementations
impl Default for BacktestBenchmarks {
    fn default() -> Self {
        Self {
            data_processing_speed: HashMap::new(),
            memory_usage: HashMap::new(),
            transaction_cost_speed: Duration::ZERO,
            portfolio_update_speed: Duration::ZERO,
            metrics_calculation_speed: Duration::ZERO,
        }
    }
}

impl Default for ValidationBenchmarks {
    fn default() -> Self {
        Self {
            accuracy_test_speed: Duration::ZERO,
            significance_test_speed: Duration::ZERO,
            distribution_test_speed: Duration::ZERO,
            bootstrap_speed: Duration::ZERO,
            cross_validation_speed: Duration::ZERO,
        }
    }
}

impl Default for FactorBenchmarks {
    fn default() -> Self {
        Self {
            exposure_calculation_speed: Duration::ZERO,
            style_analysis_speed: Duration::ZERO,
            attribution_speed: Duration::ZERO,
            risk_decomposition_speed: Duration::ZERO,
            regime_analysis_speed: Duration::ZERO,
        }
    }
}

impl Default for RiskBenchmarks {
    fn default() -> Self {
        Self {
            var_calculation_speed: HashMap::new(),
            stress_testing_speed: Duration::ZERO,
            monte_carlo_speed: Duration::ZERO,
            extreme_value_speed: Duration::ZERO,
            liquidity_risk_speed: Duration::ZERO,
        }
    }
}

impl Default for EndToEndBenchmarks {
    fn default() -> Self {
        Self {
            full_pipeline_speed: Duration::ZERO,
            peak_memory_usage: 0,
            throughput: 0.0,
            latency_percentiles: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite() {
        let config = BenchmarkConfig {
            data_points: 1000,
            num_assets: 10,
            num_factors: 5,
            mc_simulations: 1000,
            iterations: 3,
            profile_memory: false,
        };
        
        let benchmarks = QuantitativeBenchmarks::run_benchmarks(config);
        
        // Verify some benchmarks ran
        assert!(benchmarks.backtest_benchmarks.transaction_cost_speed > Duration::ZERO);
        assert!(benchmarks.validation_benchmarks.accuracy_test_speed > Duration::ZERO);
        assert!(benchmarks.factor_benchmarks.exposure_calculation_speed > Duration::ZERO);
        assert!(benchmarks.risk_benchmarks.stress_testing_speed > Duration::ZERO);
        assert!(benchmarks.e2e_benchmarks.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.data_points, 10000);
        assert_eq!(config.num_assets, 100);
        assert_eq!(config.iterations, 10);
    }
}