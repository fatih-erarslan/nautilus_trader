//! Statistical Validation Framework for Neural Trading Predictions
//!
//! This module provides comprehensive statistical tests for validating
//! neural network predictions and strategy significance.

use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Normal, StudentsT, ChiSquared, ContinuousCDF};
use statrs::statistics::{Statistics, OrderStatistics};

/// Comprehensive statistical validation suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalValidator {
    /// Prediction accuracy tests
    pub accuracy_tests: AccuracyTestSuite,
    /// Significance tests
    pub significance_tests: SignificanceTestSuite,
    /// Distribution tests
    pub distribution_tests: DistributionTestSuite,
    /// Correlation analysis
    pub correlation_analysis: CorrelationAnalysis,
    /// Model validation tests
    pub model_validation: ModelValidationSuite,
}

/// Prediction accuracy testing framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyTestSuite {
    /// Hit rate analysis
    pub hit_rate: HitRateAnalysis,
    /// Directional accuracy
    pub directional_accuracy: DirectionalAccuracy,
    /// Magnitude accuracy
    pub magnitude_accuracy: MagnitudeAccuracy,
    /// Time-based accuracy
    pub temporal_accuracy: TemporalAccuracy,
    /// Confidence-weighted accuracy
    pub confidence_weighted: ConfidenceWeightedAccuracy,
}

/// Statistical significance testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceTestSuite {
    /// T-tests for returns
    pub t_tests: TTestResults,
    /// Chi-square tests
    pub chi_square_tests: ChiSquareResults,
    /// Kolmogorov-Smirnov tests
    pub ks_tests: KSTestResults,
    /// Wilcoxon signed-rank tests
    pub wilcoxon_tests: WilcoxonResults,
    /// Bootstrap confidence intervals
    pub bootstrap_tests: BootstrapResults,
}

/// Distribution analysis and testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionTestSuite {
    /// Normality tests
    pub normality_tests: NormalityTests,
    /// Stationarity tests
    pub stationarity_tests: StationarityTests,
    /// Autocorrelation analysis
    pub autocorrelation: AutocorrelationAnalysis,
    /// Heteroskedasticity tests
    pub heteroskedasticity: HeteroskedasticityTests,
}

/// Correlation and dependency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    /// Linear correlations
    pub linear_correlations: HashMap<String, f64>,
    /// Rank correlations (Spearman)
    pub rank_correlations: HashMap<String, f64>,
    /// Mutual information
    pub mutual_information: HashMap<String, f64>,
    /// Copula analysis
    pub copula_analysis: CopulaAnalysis,
    /// Time-varying correlations
    pub rolling_correlations: RollingCorrelations,
}

/// Model validation framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationSuite {
    /// Cross-validation results
    pub cross_validation: CrossValidationResults,
    /// Out-of-sample tests
    pub out_of_sample: OutOfSampleResults,
    /// Walk-forward analysis
    pub walk_forward: WalkForwardResults,
    /// Regime analysis
    pub regime_analysis: RegimeAnalysis,
    /// Robustness tests
    pub robustness_tests: RobustnessTests,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitRateAnalysis {
    pub overall_hit_rate: f64,
    pub hit_rate_by_confidence: HashMap<String, f64>,
    pub hit_rate_by_timeframe: HashMap<String, f64>,
    pub hit_rate_trend: Vec<f64>,
    pub statistical_significance: f64,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalAccuracy {
    pub up_predictions: DirectionalStats,
    pub down_predictions: DirectionalStats,
    pub overall_accuracy: f64,
    pub bias_analysis: BiasAnalysis,
    pub confusion_matrix: ConfusionMatrix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalStats {
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasAnalysis {
    pub bullish_bias: f64,
    pub bearish_bias: f64,
    pub magnitude_bias: f64,
    pub temporal_bias: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub true_positive: usize,
    pub true_negative: usize,
    pub false_positive: usize,
    pub false_negative: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagnitudeAccuracy {
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub mean_absolute_percentage_error: f64,
    pub r_squared: f64,
    pub correlation_coefficient: f64,
    pub error_distribution: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAccuracy {
    pub accuracy_by_horizon: HashMap<String, f64>,
    pub accuracy_decay_rate: f64,
    pub optimal_horizon: String,
    pub time_series_metrics: TimeSeriesMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesMetrics {
    pub theil_u_statistic: f64,
    pub directional_change_accuracy: f64,
    pub persistence_accuracy: f64,
    pub trend_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceWeightedAccuracy {
    pub weighted_accuracy: f64,
    pub confidence_calibration: CalibrationMetrics,
    pub reliability_diagram: Vec<CalibrationBin>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub brier_score: f64,
    pub calibration_error: f64,
    pub sharpness: f64,
    pub reliability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub confidence_range: (f64, f64),
    pub predicted_probability: f64,
    pub observed_frequency: f64,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResults {
    pub single_sample_tests: HashMap<String, TTestResult>,
    pub paired_tests: HashMap<String, TTestResult>,
    pub two_sample_tests: HashMap<String, TTestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,
    pub degrees_freedom: f64,
    pub confidence_interval: (f64, f64),
    pub effect_size: f64,
    pub power: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquareResults {
    pub goodness_of_fit: HashMap<String, ChiSquareTest>,
    pub independence_tests: HashMap<String, ChiSquareTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquareTest {
    pub chi_square_statistic: f64,
    pub p_value: f64,
    pub degrees_freedom: usize,
    pub expected_frequencies: Vec<f64>,
    pub observed_frequencies: Vec<f64>,
    pub residuals: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KSTestResults {
    pub one_sample_tests: HashMap<String, KSTest>,
    pub two_sample_tests: HashMap<String, KSTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KSTest {
    pub ks_statistic: f64,
    pub p_value: f64,
    pub critical_value: f64,
    pub sample_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilcoxonResults {
    pub signed_rank_tests: HashMap<String, WilcoxonTest>,
    pub rank_sum_tests: HashMap<String, WilcoxonTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilcoxonTest {
    pub w_statistic: f64,
    pub p_value: f64,
    pub effect_size: f64,
    pub sample_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResults {
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    pub bootstrap_distribution: HashMap<String, Vec<f64>>,
    pub bias_estimates: HashMap<String, f64>,
    pub standard_errors: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityTests {
    pub shapiro_wilk: ShapiroWilkTest,
    pub anderson_darling: AndersonDarlingTest,
    pub jarque_bera: JarqueBeraTest,
    pub lilliefors: LillieforsTest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapiroWilkTest {
    pub w_statistic: f64,
    pub p_value: f64,
    pub sample_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndersonDarlingTest {
    pub ad_statistic: f64,
    pub p_value: f64,
    pub critical_values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JarqueBeraTest {
    pub jb_statistic: f64,
    pub p_value: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LillieforsTest {
    pub l_statistic: f64,
    pub p_value: f64,
    pub critical_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTests {
    pub augmented_dickey_fuller: ADFTest,
    pub phillips_perron: PPTest,
    pub kpss_test: KPSSTest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ADFTest {
    pub adf_statistic: f64,
    pub p_value: f64,
    pub critical_values: HashMap<String, f64>,
    pub used_lag: usize,
    pub n_observations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPTest {
    pub pp_statistic: f64,
    pub p_value: f64,
    pub critical_values: HashMap<String, f64>,
    pub lags: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPSSTest {
    pub kpss_statistic: f64,
    pub p_value: f64,
    pub critical_values: HashMap<String, f64>,
    pub lags: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationAnalysis {
    pub autocorrelations: Vec<f64>,
    pub partial_autocorrelations: Vec<f64>,
    pub ljung_box_test: LjungBoxTest,
    pub durbin_watson: f64,
    pub breusch_godfrey: BreuschGodfreyTest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LjungBoxTest {
    pub lb_statistic: f64,
    pub p_value: f64,
    pub degrees_freedom: usize,
    pub lags: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreuschGodfreyTest {
    pub bg_statistic: f64,
    pub p_value: f64,
    pub lags: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroskedasticityTests {
    pub breusch_pagan: BreuschPaganTest,
    pub white_test: WhiteTest,
    pub arch_test: ARCHTest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreuschPaganTest {
    pub bp_statistic: f64,
    pub p_value: f64,
    pub degrees_freedom: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteTest {
    pub white_statistic: f64,
    pub p_value: f64,
    pub degrees_freedom: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARCHTest {
    pub arch_statistic: f64,
    pub p_value: f64,
    pub lags: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopulaAnalysis {
    pub copula_type: String,
    pub parameters: Vec<f64>,
    pub goodness_of_fit: f64,
    pub tail_dependence: TailDependence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailDependence {
    pub upper_tail: f64,
    pub lower_tail: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingCorrelations {
    pub window_size: usize,
    pub correlations: Vec<f64>,
    pub volatility_regimes: Vec<VolatilityRegime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityRegime {
    pub start_date: String,
    pub end_date: String,
    pub regime_type: String,
    pub correlation: f64,
    pub volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    pub cv_scores: Vec<f64>,
    pub mean_score: f64,
    pub std_score: f64,
    pub fold_results: Vec<FoldResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    pub fold_number: usize,
    pub train_score: f64,
    pub validation_score: f64,
    pub test_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfSampleResults {
    pub oos_periods: Vec<OOSPeriod>,
    pub aggregate_metrics: HashMap<String, f64>,
    pub performance_stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OOSPeriod {
    pub start_date: String,
    pub end_date: String,
    pub metrics: HashMap<String, f64>,
    pub significance_tests: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResults {
    pub optimization_periods: Vec<OptimizationPeriod>,
    pub forward_test_results: Vec<ForwardTestResult>,
    pub parameter_stability: ParameterStability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPeriod {
    pub start_date: String,
    pub end_date: String,
    pub optimal_parameters: HashMap<String, f64>,
    pub in_sample_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardTestResult {
    pub start_date: String,
    pub end_date: String,
    pub used_parameters: HashMap<String, f64>,
    pub performance_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterStability {
    pub parameter_correlations: HashMap<String, f64>,
    pub stability_metrics: HashMap<String, f64>,
    pub regime_sensitivity: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAnalysis {
    pub identified_regimes: Vec<MarketRegime>,
    pub regime_transition_matrix: DMatrix<f64>,
    pub performance_by_regime: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegime {
    pub regime_id: String,
    pub start_date: String,
    pub end_date: String,
    pub characteristics: HashMap<String, f64>,
    pub regime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessTests {
    pub parameter_sensitivity: HashMap<String, SensitivityAnalysis>,
    pub stress_tests: Vec<StressTestResult>,
    pub monte_carlo_results: MonteCarloResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    pub parameter_name: String,
    pub base_value: f64,
    pub sensitivity_range: Vec<f64>,
    pub performance_impact: Vec<f64>,
    pub elasticity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    pub scenario_name: String,
    pub scenario_description: String,
    pub performance_metrics: HashMap<String, f64>,
    pub risk_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResults {
    pub num_simulations: usize,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    pub distribution_statistics: HashMap<String, DistributionStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: HashMap<String, f64>,
}

impl StatisticalValidator {
    /// Create new statistical validator
    pub fn new() -> Self {
        Self {
            accuracy_tests: AccuracyTestSuite::new(),
            significance_tests: SignificanceTestSuite::new(),
            distribution_tests: DistributionTestSuite::new(),
            correlation_analysis: CorrelationAnalysis::new(),
            model_validation: ModelValidationSuite::new(),
        }
    }

    /// Validate prediction accuracy
    pub fn validate_predictions(&mut self, predictions: &[f64], actuals: &[f64], confidence: &[f64]) -> ValidationResults {
        // Hit rate analysis
        self.accuracy_tests.hit_rate = self.calculate_hit_rate(predictions, actuals, confidence);
        
        // Directional accuracy
        self.accuracy_tests.directional_accuracy = self.calculate_directional_accuracy(predictions, actuals);
        
        // Magnitude accuracy
        self.accuracy_tests.magnitude_accuracy = self.calculate_magnitude_accuracy(predictions, actuals);
        
        // Statistical significance tests
        self.run_significance_tests(predictions, actuals);
        
        // Distribution tests
        self.run_distribution_tests(predictions, actuals);
        
        ValidationResults {
            overall_accuracy: self.accuracy_tests.hit_rate.overall_hit_rate,
            statistical_significance: self.significance_tests.t_tests.single_sample_tests
                .get("prediction_accuracy")
                .map(|t| t.p_value)
                .unwrap_or(1.0),
            confidence_intervals: self.calculate_confidence_intervals(predictions, actuals),
            recommendation: self.generate_recommendation(),
        }
    }

    /// Calculate hit rate analysis
    fn calculate_hit_rate(&self, predictions: &[f64], actuals: &[f64], confidence: &[f64]) -> HitRateAnalysis {
        let n = predictions.len().min(actuals.len());
        if n == 0 {
            return HitRateAnalysis::default();
        }

        let mut hits = 0;
        let mut confidence_buckets: HashMap<String, (usize, usize)> = HashMap::new();

        for i in 0..n {
            // Define hit as prediction and actual having same sign
            let prediction_direction = if predictions[i] > 0.0 { 1 } else { -1 };
            let actual_direction = if actuals[i] > 0.0 { 1 } else { -1 };
            
            if prediction_direction == actual_direction {
                hits += 1;
            }

            // Bucket by confidence
            let conf_bucket = if confidence[i] >= 0.8 {
                "high"
            } else if confidence[i] >= 0.6 {
                "medium"
            } else {
                "low"
            };

            let entry = confidence_buckets.entry(conf_bucket.to_string()).or_insert((0, 0));
            entry.1 += 1; // total
            if prediction_direction == actual_direction {
                entry.0 += 1; // hits
            }
        }

        let overall_hit_rate = hits as f64 / n as f64;

        // Calculate confidence-bucketed hit rates
        let mut hit_rate_by_confidence = HashMap::new();
        for (bucket, (hits, total)) in confidence_buckets {
            if total > 0 {
                hit_rate_by_confidence.insert(bucket, hits as f64 / total as f64);
            }
        }

        // Statistical significance test
        let p_null = 0.5; // null hypothesis: random guessing
        let z_score = (overall_hit_rate - p_null) / (p_null * (1.0 - p_null) / n as f64).sqrt();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

        // Confidence interval for hit rate
        let margin_error = 1.96 * (overall_hit_rate * (1.0 - overall_hit_rate) / n as f64).sqrt();
        let confidence_interval = (
            (overall_hit_rate - margin_error).max(0.0),
            (overall_hit_rate + margin_error).min(1.0)
        );

        HitRateAnalysis {
            overall_hit_rate,
            hit_rate_by_confidence,
            hit_rate_by_timeframe: HashMap::new(), // Would need timestamps to implement
            hit_rate_trend: vec![overall_hit_rate], // Simplified
            statistical_significance: p_value,
            confidence_interval,
        }
    }

    /// Calculate directional accuracy metrics
    fn calculate_directional_accuracy(&self, predictions: &[f64], actuals: &[f64]) -> DirectionalAccuracy {
        let n = predictions.len().min(actuals.len());
        if n == 0 {
            return DirectionalAccuracy::default();
        }

        let mut confusion_matrix = ConfusionMatrix {
            true_positive: 0,
            true_negative: 0,
            false_positive: 0,
            false_negative: 0,
        };

        let mut up_correct = 0;
        let mut up_total = 0;
        let mut down_correct = 0;
        let mut down_total = 0;

        for i in 0..n {
            let pred_up = predictions[i] > 0.0;
            let actual_up = actuals[i] > 0.0;

            match (pred_up, actual_up) {
                (true, true) => {
                    confusion_matrix.true_positive += 1;
                    up_correct += 1;
                }
                (false, false) => {
                    confusion_matrix.true_negative += 1;
                    down_correct += 1;
                }
                (true, false) => confusion_matrix.false_positive += 1,
                (false, true) => confusion_matrix.false_negative += 1,
            }

            if pred_up {
                up_total += 1;
                if actual_up {
                    up_correct += 1;
                }
            } else {
                down_total += 1;
                if !actual_up {
                    down_correct += 1;
                }
            }
        }

        let up_accuracy = if up_total > 0 { up_correct as f64 / up_total as f64 } else { 0.0 };
        let down_accuracy = if down_total > 0 { down_correct as f64 / down_total as f64 } else { 0.0 };

        let precision = if confusion_matrix.true_positive + confusion_matrix.false_positive > 0 {
            confusion_matrix.true_positive as f64 / (confusion_matrix.true_positive + confusion_matrix.false_positive) as f64
        } else { 0.0 };

        let recall = if confusion_matrix.true_positive + confusion_matrix.false_negative > 0 {
            confusion_matrix.true_positive as f64 / (confusion_matrix.true_positive + confusion_matrix.false_negative) as f64
        } else { 0.0 };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else { 0.0 };

        let overall_accuracy = (confusion_matrix.true_positive + confusion_matrix.true_negative) as f64 / n as f64;

        DirectionalAccuracy {
            up_predictions: DirectionalStats {
                total_predictions: up_total,
                correct_predictions: up_correct,
                accuracy: up_accuracy,
                precision,
                recall,
                f1_score,
            },
            down_predictions: DirectionalStats {
                total_predictions: down_total,
                correct_predictions: down_correct,
                accuracy: down_accuracy,
                precision: 1.0 - precision, // For down predictions
                recall: 1.0 - recall,       // For down predictions
                f1_score: 0.0, // Would need to calculate separately for down predictions
            },
            overall_accuracy,
            bias_analysis: self.calculate_bias_analysis(predictions, actuals),
            confusion_matrix,
        }
    }

    /// Calculate bias analysis
    fn calculate_bias_analysis(&self, predictions: &[f64], actuals: &[f64]) -> BiasAnalysis {
        let n = predictions.len().min(actuals.len());
        if n == 0 {
            return BiasAnalysis::default();
        }

        let pred_up_count = predictions.iter().filter(|&&p| p > 0.0).count();
        let actual_up_count = actuals.iter().filter(|&&a| a > 0.0).count();

        let bullish_bias = pred_up_count as f64 / n as f64 - actual_up_count as f64 / n as f64;
        let bearish_bias = -bullish_bias;

        // Magnitude bias: tendency to over or under-predict magnitude
        let mut magnitude_errors: Vec<f64> = Vec::new();
        for i in 0..n {
            magnitude_errors.push(predictions[i].abs() - actuals[i].abs());
        }
        let magnitude_bias: f64 = magnitude_errors.iter().sum::<f64>() / n as f64;

        BiasAnalysis {
            bullish_bias,
            bearish_bias,
            magnitude_bias,
            temporal_bias: HashMap::new(), // Would need timestamps to implement
        }
    }

    /// Calculate magnitude accuracy
    fn calculate_magnitude_accuracy(&self, predictions: &[f64], actuals: &[f64]) -> MagnitudeAccuracy {
        let n = predictions.len().min(actuals.len());
        if n == 0 {
            return MagnitudeAccuracy::default();
        }

        let mut errors: Vec<f64> = Vec::new();
        let mut absolute_errors: Vec<f64> = Vec::new();
        let mut percentage_errors: Vec<f64> = Vec::new();
        let mut squared_errors: Vec<f64> = Vec::new();

        let actual_mean: f64 = actuals.iter().sum::<f64>() / n as f64;

        for i in 0..n {
            let error = predictions[i] - actuals[i];
            let abs_error = error.abs();
            let sq_error = error * error;

            errors.push(error);
            absolute_errors.push(abs_error);
            squared_errors.push(sq_error);

            if actuals[i] != 0.0 {
                percentage_errors.push(abs_error / actuals[i].abs() * 100.0);
            }
        }

        let mae = absolute_errors.iter().sum::<f64>() / n as f64;
        let rmse = (squared_errors.iter().sum::<f64>() / n as f64).sqrt();
        let mape = if !percentage_errors.is_empty() {
            percentage_errors.iter().sum::<f64>() / percentage_errors.len() as f64
        } else { 0.0 };

        // Calculate R-squared
        let ss_res: f64 = squared_errors.iter().sum();
        let ss_tot: f64 = actuals.iter().map(|&a| (a - actual_mean).powi(2)).sum();
        let r_squared = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

        // Calculate correlation coefficient
        let pred_mean: f64 = predictions.iter().sum::<f64>() / n as f64;
        let numerator: f64 = (0..n).map(|i| (predictions[i] - pred_mean) * (actuals[i] - actual_mean)).sum();
        let pred_var: f64 = (0..n).map(|i| (predictions[i] - pred_mean).powi(2)).sum();
        let actual_var: f64 = (0..n).map(|i| (actuals[i] - actual_mean).powi(2)).sum();
        let correlation = if pred_var > 0.0 && actual_var > 0.0 {
            numerator / (pred_var * actual_var).sqrt()
        } else { 0.0 };

        MagnitudeAccuracy {
            mean_absolute_error: mae,
            root_mean_square_error: rmse,
            mean_absolute_percentage_error: mape,
            r_squared,
            correlation_coefficient: correlation,
            error_distribution: errors,
        }
    }

    /// Run statistical significance tests
    fn run_significance_tests(&mut self, predictions: &[f64], actuals: &[f64]) {
        let errors: Vec<f64> = predictions.iter().zip(actuals.iter())
            .map(|(p, a)| p - a)
            .collect();

        if errors.is_empty() {
            return;
        }

        // One-sample t-test: test if mean error is significantly different from 0
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let error_variance = errors.iter()
            .map(|e| (e - mean_error).powi(2))
            .sum::<f64>() / (errors.len() - 1) as f64;
        let std_error = (error_variance / errors.len() as f64).sqrt();

        if std_error > 0.0 {
            let t_stat = mean_error / std_error;
            let df = errors.len() - 1;
            let t_dist = StudentsT::new(0.0, 1.0, df as f64).unwrap();
            let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

            let t_critical = t_dist.inverse_cdf(0.975); // 95% confidence
            let margin_error = t_critical * std_error;
            let confidence_interval = (mean_error - margin_error, mean_error + margin_error);

            let effect_size = mean_error / error_variance.sqrt(); // Cohen's d

            let t_test_result = TTestResult {
                t_statistic: t_stat,
                p_value,
                degrees_freedom: df as f64,
                confidence_interval,
                effect_size,
                power: 0.8, // Would need more complex calculation
            };

            self.significance_tests.t_tests.single_sample_tests
                .insert("mean_error".to_string(), t_test_result);
        }
    }

    /// Run distribution tests
    fn run_distribution_tests(&mut self, predictions: &[f64], actuals: &[f64]) {
        let errors: Vec<f64> = predictions.iter().zip(actuals.iter())
            .map(|(p, a)| p - a)
            .collect();

        if errors.len() < 3 {
            return;
        }

        // Normality test - simplified Jarque-Bera test
        let n = errors.len() as f64;
        let mean = errors.iter().sum::<f64>() / n;
        let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            let skewness = errors.iter()
                .map(|e| ((e - mean) / std_dev).powi(3))
                .sum::<f64>() / n;

            let kurtosis = errors.iter()
                .map(|e| ((e - mean) / std_dev).powi(4))
                .sum::<f64>() / n - 3.0; // Excess kurtosis

            let jb_statistic = n / 6.0 * (skewness.powi(2) + kurtosis.powi(2) / 4.0);
            let chi_sq = ChiSquared::new(2.0).unwrap();
            let p_value = 1.0 - chi_sq.cdf(jb_statistic);

            self.distribution_tests.normality_tests.jarque_bera = JarqueBeraTest {
                jb_statistic,
                p_value,
                skewness,
                kurtosis,
            };
        }
    }

    /// Calculate confidence intervals
    fn calculate_confidence_intervals(&self, predictions: &[f64], actuals: &[f64]) -> HashMap<String, (f64, f64)> {
        let mut intervals = HashMap::new();

        let errors: Vec<f64> = predictions.iter().zip(actuals.iter())
            .map(|(p, a)| p - a)
            .collect();

        if !errors.is_empty() {
            let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
            let std_error = self.calculate_standard_error(&errors);

            if std_error > 0.0 {
                let margin = 1.96 * std_error; // 95% confidence
                intervals.insert("mean_error".to_string(), (mean_error - margin, mean_error + margin));
            }
        }

        intervals
    }

    /// Calculate standard error
    fn calculate_standard_error(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;

        (variance / data.len() as f64).sqrt()
    }

    /// Generate validation recommendation
    fn generate_recommendation(&self) -> String {
        let hit_rate = self.accuracy_tests.hit_rate.overall_hit_rate;
        let significance = self.accuracy_tests.hit_rate.statistical_significance;

        if hit_rate > 0.6 && significance < 0.05 {
            "Model shows strong predictive performance with statistical significance. Recommend for production deployment.".to_string()
        } else if hit_rate > 0.55 && significance < 0.1 {
            "Model shows moderate predictive performance. Consider additional validation and risk controls.".to_string()
        } else if hit_rate > 0.5 {
            "Model shows weak predictive performance. Requires significant improvement before deployment.".to_string()
        } else {
            "Model performance is not better than random. Not recommended for deployment.".to_string()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub overall_accuracy: f64,
    pub statistical_significance: f64,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    pub recommendation: String,
}

// Default implementations
impl Default for HitRateAnalysis {
    fn default() -> Self {
        Self {
            overall_hit_rate: 0.0,
            hit_rate_by_confidence: HashMap::new(),
            hit_rate_by_timeframe: HashMap::new(),
            hit_rate_trend: Vec::new(),
            statistical_significance: 1.0,
            confidence_interval: (0.0, 0.0),
        }
    }
}

impl Default for DirectionalAccuracy {
    fn default() -> Self {
        Self {
            up_predictions: DirectionalStats::default(),
            down_predictions: DirectionalStats::default(),
            overall_accuracy: 0.0,
            bias_analysis: BiasAnalysis::default(),
            confusion_matrix: ConfusionMatrix::default(),
        }
    }
}

impl Default for DirectionalStats {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            correct_predictions: 0,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
        }
    }
}

impl Default for BiasAnalysis {
    fn default() -> Self {
        Self {
            bullish_bias: 0.0,
            bearish_bias: 0.0,
            magnitude_bias: 0.0,
            temporal_bias: HashMap::new(),
        }
    }
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self {
            true_positive: 0,
            true_negative: 0,
            false_positive: 0,
            false_negative: 0,
        }
    }
}

impl Default for MagnitudeAccuracy {
    fn default() -> Self {
        Self {
            mean_absolute_error: 0.0,
            root_mean_square_error: 0.0,
            mean_absolute_percentage_error: 0.0,
            r_squared: 0.0,
            correlation_coefficient: 0.0,
            error_distribution: Vec::new(),
        }
    }
}

impl AccuracyTestSuite {
    fn new() -> Self {
        Self {
            hit_rate: HitRateAnalysis::default(),
            directional_accuracy: DirectionalAccuracy::default(),
            magnitude_accuracy: MagnitudeAccuracy::default(),
            temporal_accuracy: TemporalAccuracy::default(),
            confidence_weighted: ConfidenceWeightedAccuracy::default(),
        }
    }
}

impl SignificanceTestSuite {
    fn new() -> Self {
        Self {
            t_tests: TTestResults {
                single_sample_tests: HashMap::new(),
                paired_tests: HashMap::new(),
                two_sample_tests: HashMap::new(),
            },
            chi_square_tests: ChiSquareResults {
                goodness_of_fit: HashMap::new(),
                independence_tests: HashMap::new(),
            },
            ks_tests: KSTestResults {
                one_sample_tests: HashMap::new(),
                two_sample_tests: HashMap::new(),
            },
            wilcoxon_tests: WilcoxonResults {
                signed_rank_tests: HashMap::new(),
                rank_sum_tests: HashMap::new(),
            },
            bootstrap_tests: BootstrapResults {
                confidence_intervals: HashMap::new(),
                bootstrap_distribution: HashMap::new(),
                bias_estimates: HashMap::new(),
                standard_errors: HashMap::new(),
            },
        }
    }
}

impl DistributionTestSuite {
    fn new() -> Self {
        Self {
            normality_tests: NormalityTests::default(),
            stationarity_tests: StationarityTests::default(),
            autocorrelation: AutocorrelationAnalysis::default(),
            heteroskedasticity: HeteroskedasticityTests::default(),
        }
    }
}

impl CorrelationAnalysis {
    fn new() -> Self {
        Self {
            linear_correlations: HashMap::new(),
            rank_correlations: HashMap::new(),
            mutual_information: HashMap::new(),
            copula_analysis: CopulaAnalysis::default(),
            rolling_correlations: RollingCorrelations::default(),
        }
    }
}

impl ModelValidationSuite {
    fn new() -> Self {
        Self {
            cross_validation: CrossValidationResults::default(),
            out_of_sample: OutOfSampleResults::default(),
            walk_forward: WalkForwardResults::default(),
            regime_analysis: RegimeAnalysis::default(),
            robustness_tests: RobustnessTests::default(),
        }
    }
}

// Additional default implementations for complex types
impl Default for TemporalAccuracy {
    fn default() -> Self {
        Self {
            accuracy_by_horizon: HashMap::new(),
            accuracy_decay_rate: 0.0,
            optimal_horizon: String::new(),
            time_series_metrics: TimeSeriesMetrics::default(),
        }
    }
}

impl Default for TimeSeriesMetrics {
    fn default() -> Self {
        Self {
            theil_u_statistic: 0.0,
            directional_change_accuracy: 0.0,
            persistence_accuracy: 0.0,
            trend_accuracy: 0.0,
        }
    }
}

impl Default for ConfidenceWeightedAccuracy {
    fn default() -> Self {
        Self {
            weighted_accuracy: 0.0,
            confidence_calibration: CalibrationMetrics::default(),
            reliability_diagram: Vec::new(),
        }
    }
}

impl Default for CalibrationMetrics {
    fn default() -> Self {
        Self {
            brier_score: 0.0,
            calibration_error: 0.0,
            sharpness: 0.0,
            reliability: 0.0,
        }
    }
}

impl Default for NormalityTests {
    fn default() -> Self {
        Self {
            shapiro_wilk: ShapiroWilkTest::default(),
            anderson_darling: AndersonDarlingTest::default(),
            jarque_bera: JarqueBeraTest::default(),
            lilliefors: LillieforsTest::default(),
        }
    }
}

impl Default for ShapiroWilkTest {
    fn default() -> Self {
        Self {
            w_statistic: 0.0,
            p_value: 1.0,
            sample_size: 0,
        }
    }
}

impl Default for AndersonDarlingTest {
    fn default() -> Self {
        Self {
            ad_statistic: 0.0,
            p_value: 1.0,
            critical_values: Vec::new(),
        }
    }
}

impl Default for JarqueBeraTest {
    fn default() -> Self {
        Self {
            jb_statistic: 0.0,
            p_value: 1.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}

impl Default for LillieforsTest {
    fn default() -> Self {
        Self {
            l_statistic: 0.0,
            p_value: 1.0,
            critical_value: 0.0,
        }
    }
}

impl Default for StationarityTests {
    fn default() -> Self {
        Self {
            augmented_dickey_fuller: ADFTest::default(),
            phillips_perron: PPTest::default(),
            kpss_test: KPSSTest::default(),
        }
    }
}

impl Default for ADFTest {
    fn default() -> Self {
        Self {
            adf_statistic: 0.0,
            p_value: 1.0,
            critical_values: HashMap::new(),
            used_lag: 0,
            n_observations: 0,
        }
    }
}

impl Default for PPTest {
    fn default() -> Self {
        Self {
            pp_statistic: 0.0,
            p_value: 1.0,
            critical_values: HashMap::new(),
            lags: 0,
        }
    }
}

impl Default for KPSSTest {
    fn default() -> Self {
        Self {
            kpss_statistic: 0.0,
            p_value: 1.0,
            critical_values: HashMap::new(),
            lags: 0,
        }
    }
}

impl Default for AutocorrelationAnalysis {
    fn default() -> Self {
        Self {
            autocorrelations: Vec::new(),
            partial_autocorrelations: Vec::new(),
            ljung_box_test: LjungBoxTest::default(),
            durbin_watson: 0.0,
            breusch_godfrey: BreuschGodfreyTest::default(),
        }
    }
}

impl Default for LjungBoxTest {
    fn default() -> Self {
        Self {
            lb_statistic: 0.0,
            p_value: 1.0,
            degrees_freedom: 0,
            lags: 0,
        }
    }
}

impl Default for BreuschGodfreyTest {
    fn default() -> Self {
        Self {
            bg_statistic: 0.0,
            p_value: 1.0,
            lags: 0,
        }
    }
}

impl Default for HeteroskedasticityTests {
    fn default() -> Self {
        Self {
            breusch_pagan: BreuschPaganTest::default(),
            white_test: WhiteTest::default(),
            arch_test: ARCHTest::default(),
        }
    }
}

impl Default for BreuschPaganTest {
    fn default() -> Self {
        Self {
            bp_statistic: 0.0,
            p_value: 1.0,
            degrees_freedom: 0,
        }
    }
}

impl Default for WhiteTest {
    fn default() -> Self {
        Self {
            white_statistic: 0.0,
            p_value: 1.0,
            degrees_freedom: 0,
        }
    }
}

impl Default for ARCHTest {
    fn default() -> Self {
        Self {
            arch_statistic: 0.0,
            p_value: 1.0,
            lags: 0,
        }
    }
}

impl Default for CopulaAnalysis {
    fn default() -> Self {
        Self {
            copula_type: String::new(),
            parameters: Vec::new(),
            goodness_of_fit: 0.0,
            tail_dependence: TailDependence::default(),
        }
    }
}

impl Default for TailDependence {
    fn default() -> Self {
        Self {
            upper_tail: 0.0,
            lower_tail: 0.0,
        }
    }
}

impl Default for RollingCorrelations {
    fn default() -> Self {
        Self {
            window_size: 0,
            correlations: Vec::new(),
            volatility_regimes: Vec::new(),
        }
    }
}

impl Default for CrossValidationResults {
    fn default() -> Self {
        Self {
            cv_scores: Vec::new(),
            mean_score: 0.0,
            std_score: 0.0,
            fold_results: Vec::new(),
        }
    }
}

impl Default for OutOfSampleResults {
    fn default() -> Self {
        Self {
            oos_periods: Vec::new(),
            aggregate_metrics: HashMap::new(),
            performance_stability: 0.0,
        }
    }
}

impl Default for WalkForwardResults {
    fn default() -> Self {
        Self {
            optimization_periods: Vec::new(),
            forward_test_results: Vec::new(),
            parameter_stability: ParameterStability::default(),
        }
    }
}

impl Default for ParameterStability {
    fn default() -> Self {
        Self {
            parameter_correlations: HashMap::new(),
            stability_metrics: HashMap::new(),
            regime_sensitivity: HashMap::new(),
        }
    }
}

impl Default for RegimeAnalysis {
    fn default() -> Self {
        Self {
            identified_regimes: Vec::new(),
            regime_transition_matrix: DMatrix::zeros(0, 0),
            performance_by_regime: HashMap::new(),
        }
    }
}

impl Default for RobustnessTests {
    fn default() -> Self {
        Self {
            parameter_sensitivity: HashMap::new(),
            stress_tests: Vec::new(),
            monte_carlo_results: MonteCarloResults::default(),
        }
    }
}

impl Default for MonteCarloResults {
    fn default() -> Self {
        Self {
            num_simulations: 0,
            confidence_intervals: HashMap::new(),
            distribution_statistics: HashMap::new(),
        }
    }
}