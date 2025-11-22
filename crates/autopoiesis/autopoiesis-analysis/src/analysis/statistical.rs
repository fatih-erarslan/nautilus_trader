//! Statistical analysis implementation

use crate::prelude::*;
use crate::models::MarketData;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::VecDeque;

/// Statistical analysis engine for market data
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// Configuration for statistical calculations
    config: StatisticalAnalysisConfig,
    
    /// Historical data buffer
    data_buffer: VecDeque<DataPoint>,
    
    /// Calculated statistics cache
    stats_cache: StatisticsCache,
}

#[derive(Debug, Clone)]
pub struct StatisticalAnalysisConfig {
    /// Maximum number of data points to retain
    pub max_buffer_size: usize,
    
    /// Confidence levels for statistical tests
    pub confidence_levels: Vec<f64>,
    
    /// Lookback periods for various calculations
    pub lookback_periods: LookbackPeriods,
    
    /// Correlation analysis settings
    pub correlation_config: CorrelationConfig,
    
    /// Distribution analysis settings
    pub distribution_config: DistributionConfig,
    
    /// Volatility calculation settings
    pub volatility_config: VolatilityConfig,
}

#[derive(Debug, Clone)]
pub struct LookbackPeriods {
    pub short_term: usize,
    pub medium_term: usize,
    pub long_term: usize,
}

#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Minimum number of observations for correlation
    pub min_observations: usize,
    
    /// Correlation significance threshold
    pub significance_threshold: f64,
    
    /// Rolling correlation window
    pub rolling_window: usize,
}

#[derive(Debug, Clone)]
pub struct DistributionConfig {
    /// Number of bins for histogram analysis
    pub histogram_bins: usize,
    
    /// Minimum sample size for distribution tests
    pub min_sample_size: usize,
    
    /// Significance level for normality tests
    pub normality_test_alpha: f64,
}

#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    /// Method for volatility calculation
    pub method: VolatilityMethod,
    
    /// Annualization factor
    pub annualization_factor: f64,
    
    /// GARCH model parameters
    pub garch_params: GarchParams,
}

#[derive(Debug, Clone)]
pub enum VolatilityMethod {
    SimpleStdDev,
    ExponentialWeighted,
    Garch,
    Parkinson,
}

#[derive(Debug, Clone)]
pub struct GarchParams {
    pub alpha: f64,
    pub beta: f64,
    pub omega: f64,
}

#[derive(Debug, Clone)]
struct DataPoint {
    timestamp: DateTime<Utc>,
    price: f64,
    returns: Option<f64>,
    log_returns: Option<f64>,
    volume: f64,
}

#[derive(Debug, Clone, Default)]
struct StatisticsCache {
    descriptive_stats: Option<DescriptiveStatistics>,
    volatility_measures: Option<VolatilityMeasures>,
    distribution_analysis: Option<DistributionAnalysis>,
    correlation_matrix: Option<CorrelationMatrix>,
    last_update: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct StatisticalResults {
    pub timestamp: DateTime<Utc>,
    pub descriptive_stats: DescriptiveStatistics,
    pub volatility_measures: VolatilityMeasures,
    pub distribution_analysis: DistributionAnalysis,
    pub time_series_analysis: TimeSeriesAnalysis,
    pub risk_metrics: RiskMetrics,
}

#[derive(Debug, Clone)]
pub struct DescriptiveStatistics {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub std_dev: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub percentiles: Percentiles,
}

#[derive(Debug, Clone)]
pub struct Percentiles {
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone)]
pub struct VolatilityMeasures {
    pub historical_volatility: f64,
    pub realized_volatility: f64,
    pub exponential_volatility: f64,
    pub garch_volatility: Option<f64>,
    pub parkinson_volatility: Option<f64>,
    pub volatility_of_volatility: f64,
    pub volatility_clustering: f64,
}

#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    pub normality_test: NormalityTest,
    pub histogram: Histogram,
    pub moments: Moments,
    pub tail_analysis: TailAnalysis,
}

#[derive(Debug, Clone)]
pub struct NormalityTest {
    pub jarque_bera_statistic: f64,
    pub jarque_bera_p_value: f64,
    pub shapiro_wilk_statistic: Option<f64>,
    pub shapiro_wilk_p_value: Option<f64>,
    pub is_normal: bool,
}

#[derive(Debug, Clone)]
pub struct Histogram {
    pub bins: Vec<f64>,
    pub frequencies: Vec<usize>,
    pub bin_edges: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Moments {
    pub first_moment: f64,  // Mean
    pub second_moment: f64, // Variance
    pub third_moment: f64,  // Skewness
    pub fourth_moment: f64, // Kurtosis
    pub excess_kurtosis: f64,
}

#[derive(Debug, Clone)]
pub struct TailAnalysis {
    pub left_tail_probability: f64,
    pub right_tail_probability: f64,
    pub tail_ratio: f64,
    pub extreme_value_statistics: ExtremeValueStats,
}

#[derive(Debug, Clone)]
pub struct ExtremeValueStats {
    pub max_drawdown: f64,
    pub max_runup: f64,
    pub var_95: f64,
    pub var_99: f64,
    pub cvar_95: f64,
    pub cvar_99: f64,
}

#[derive(Debug, Clone)]
pub struct TimeSeriesAnalysis {
    pub autocorrelation: Vec<f64>,
    pub partial_autocorrelation: Vec<f64>,
    pub ljung_box_test: LjungBoxTest,
    pub adf_test: AdfTest,
    pub trend_analysis: TrendAnalysis,
}

#[derive(Debug, Clone)]
pub struct LjungBoxTest {
    pub statistic: f64,
    pub p_value: f64,
    pub is_white_noise: bool,
}

#[derive(Debug, Clone)]
pub struct AdfTest {
    pub statistic: f64,
    pub p_value: f64,
    pub critical_values: std::collections::HashMap<String, f64>,
    pub is_stationary: bool,
}

#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_duration: Duration,
    pub regime_changes: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Upward,
    Downward,
    Sideways,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub value_at_risk: ValueAtRisk,
    pub expected_shortfall: ExpectedShortfall,
    pub maximum_drawdown: f64,
    pub calmar_ratio: f64,
    pub sortino_ratio: f64,
    pub information_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct ValueAtRisk {
    pub var_95: f64,
    pub var_99: f64,
    pub var_99_9: f64,
}

#[derive(Debug, Clone)]
pub struct ExpectedShortfall {
    pub es_95: f64,
    pub es_99: f64,
    pub es_99_9: f64,
}

#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub symbols: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
}

impl Default for StatisticalAnalysisConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10000,
            confidence_levels: vec![0.90, 0.95, 0.99, 0.999],
            lookback_periods: LookbackPeriods {
                short_term: 30,
                medium_term: 90,
                long_term: 252,
            },
            correlation_config: CorrelationConfig {
                min_observations: 30,
                significance_threshold: 0.05,
                rolling_window: 60,
            },
            distribution_config: DistributionConfig {
                histogram_bins: 50,
                min_sample_size: 30,
                normality_test_alpha: 0.05,
            },
            volatility_config: VolatilityConfig {
                method: VolatilityMethod::ExponentialWeighted,
                annualization_factor: 252.0,
                garch_params: GarchParams {
                    alpha: 0.1,
                    beta: 0.85,
                    omega: 0.05,
                },
            },
        }
    }
}

impl StatisticalAnalysis {
    /// Create a new statistical analysis engine
    pub fn new(config: StatisticalAnalysisConfig) -> Self {
        Self {
            config,
            data_buffer: VecDeque::new(),
            stats_cache: StatisticsCache::default(),
        }
    }

    /// Add market data point for analysis
    pub async fn add_data(&mut self, market_data: &MarketData) -> Result<()> {
        let price = market_data.mid.to_f64().unwrap_or(0.0);
        let volume = market_data.volume_24h.to_f64().unwrap_or(0.0);

        // Calculate returns if we have previous data
        let (returns, log_returns) = if let Some(prev_price) = self.get_previous_price() {
            let ret = (price - prev_price) / prev_price;
            let log_ret = (price / prev_price).ln();
            (Some(ret), Some(log_ret))
        } else {
            (None, None)
        };

        let data_point = DataPoint {
            timestamp: market_data.timestamp,
            price,
            returns,
            log_returns,
            volume,
        };

        self.data_buffer.push_back(data_point);

        // Maintain buffer size
        while self.data_buffer.len() > self.config.max_buffer_size {
            self.data_buffer.pop_front();
        }

        Ok(())
    }

    /// Perform comprehensive statistical analysis
    pub async fn analyze(&mut self) -> Result<StatisticalResults> {
        if self.data_buffer.len() < 10 {
            return Err(Error::Analysis("Insufficient data for statistical analysis".to_string()));
        }

        let timestamp = Utc::now();

        // Calculate descriptive statistics
        let descriptive_stats = self.calculate_descriptive_statistics()?;

        // Calculate volatility measures
        let volatility_measures = self.calculate_volatility_measures()?;

        // Perform distribution analysis
        let distribution_analysis = self.calculate_distribution_analysis()?;

        // Perform time series analysis
        let time_series_analysis = self.calculate_time_series_analysis()?;

        // Calculate risk metrics
        let risk_metrics = self.calculate_risk_metrics()?;

        // Update cache
        self.stats_cache = StatisticsCache {
            descriptive_stats: Some(descriptive_stats.clone()),
            volatility_measures: Some(volatility_measures.clone()),
            distribution_analysis: Some(distribution_analysis.clone()),
            correlation_matrix: None, // Would be calculated with multiple symbols
            last_update: Some(timestamp),
        };

        Ok(StatisticalResults {
            timestamp,
            descriptive_stats,
            volatility_measures,
            distribution_analysis,
            time_series_analysis,
            risk_metrics,
        })
    }

    /// Get cached statistical results
    pub fn get_cached_results(&self) -> Option<StatisticalResults> {
        if let (Some(desc), Some(vol), Some(dist), Some(last_update)) = (
            &self.stats_cache.descriptive_stats,
            &self.stats_cache.volatility_measures,
            &self.stats_cache.distribution_analysis,
            self.stats_cache.last_update,
        ) {
            Some(StatisticalResults {
                timestamp: last_update,
                descriptive_stats: desc.clone(),
                volatility_measures: vol.clone(),
                distribution_analysis: dist.clone(),
                time_series_analysis: TimeSeriesAnalysis {
                    autocorrelation: vec![],
                    partial_autocorrelation: vec![],
                    ljung_box_test: LjungBoxTest {
                        statistic: 0.0,
                        p_value: 1.0,
                        is_white_noise: true,
                    },
                    adf_test: AdfTest {
                        statistic: 0.0,
                        p_value: 1.0,
                        critical_values: std::collections::HashMap::new(),
                        is_stationary: false,
                    },
                    trend_analysis: TrendAnalysis {
                        trend_direction: TrendDirection::Sideways,
                        trend_strength: 0.0,
                        trend_duration: Duration::zero(),
                        regime_changes: vec![],
                    },
                },
                risk_metrics: RiskMetrics {
                    value_at_risk: ValueAtRisk {
                        var_95: 0.0,
                        var_99: 0.0,
                        var_99_9: 0.0,
                    },
                    expected_shortfall: ExpectedShortfall {
                        es_95: 0.0,
                        es_99: 0.0,
                        es_99_9: 0.0,
                    },
                    maximum_drawdown: 0.0,
                    calmar_ratio: 0.0,
                    sortino_ratio: 0.0,
                    information_ratio: 0.0,
                },
            })
        } else {
            None
        }
    }

    /// Perform hypothesis testing
    pub async fn hypothesis_test(&self, test_type: HypothesisTest) -> Result<TestResult> {
        match test_type {
            HypothesisTest::Normality => self.test_normality().await,
            HypothesisTest::Stationarity => self.test_stationarity().await,
            HypothesisTest::Independence => self.test_independence().await,
            HypothesisTest::MeanReversion => self.test_mean_reversion().await,
        }
    }

    fn get_previous_price(&self) -> Option<f64> {
        self.data_buffer.back().map(|dp| dp.price)
    }

    fn calculate_descriptive_statistics(&self) -> Result<DescriptiveStatistics> {
        let prices: Vec<f64> = self.data_buffer.iter().map(|dp| dp.price).collect();
        let returns: Vec<f64> = self.data_buffer.iter()
            .filter_map(|dp| dp.returns)
            .collect();

        if prices.is_empty() {
            return Err(Error::Analysis("No price data available".to_string()));
        }

        let count = prices.len();
        let mean = prices.iter().sum::<f64>() / count as f64;
        
        // Calculate median
        let mut sorted_prices = prices.clone();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if count % 2 == 0 {
            (sorted_prices[count / 2 - 1] + sorted_prices[count / 2]) / 2.0
        } else {
            sorted_prices[count / 2]
        };

        // Calculate variance and standard deviation
        let variance = prices.iter()
            .map(|price| (price - mean).powi(2))
            .sum::<f64>() / (count - 1) as f64;
        let std_dev = variance.sqrt();

        // Calculate skewness and kurtosis
        let skewness = self.calculate_skewness(&prices, mean, std_dev);
        let kurtosis = self.calculate_kurtosis(&prices, mean, std_dev);

        let min = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        // Calculate percentiles
        let percentiles = self.calculate_percentiles(&sorted_prices);

        Ok(DescriptiveStatistics {
            count,
            mean,
            median,
            mode: None, // Mode calculation for continuous data is complex
            std_dev,
            variance,
            skewness,
            kurtosis,
            min,
            max,
            range,
            percentiles,
        })
    }

    fn calculate_skewness(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 {
            return 0.0;
        }

        let n = data.len() as f64;
        let sum_cubed_deviations = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>();

        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed_deviations
    }

    fn calculate_kurtosis(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 {
            return 0.0;
        }

        let n = data.len() as f64;
        let sum_fourth_deviations = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>();

        let numerator = n * (n + 1.0) * sum_fourth_deviations;
        let denominator = (n - 1.0) * (n - 2.0) * (n - 3.0);
        let correction = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));

        (numerator / denominator) - correction
    }

    fn calculate_percentiles(&self, sorted_data: &[f64]) -> Percentiles {
        let percentile = |p: f64| -> f64 {
            let n = sorted_data.len();
            let index = (p / 100.0) * (n - 1) as f64;
            let lower = index.floor() as usize;
            let upper = index.ceil() as usize;
            let weight = index - lower as f64;

            if lower == upper {
                sorted_data[lower]
            } else {
                sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
            }
        };

        Percentiles {
            p5: percentile(5.0),
            p10: percentile(10.0),
            p25: percentile(25.0),
            p50: percentile(50.0),
            p75: percentile(75.0),
            p90: percentile(90.0),
            p95: percentile(95.0),
            p99: percentile(99.0),
        }
    }

    fn calculate_volatility_measures(&self) -> Result<VolatilityMeasures> {
        let returns: Vec<f64> = self.data_buffer.iter()
            .filter_map(|dp| dp.returns)
            .collect();

        if returns.len() < 10 {
            return Err(Error::Analysis("Insufficient returns data for volatility calculation".to_string()));
        }

        // Historical volatility (standard deviation of returns)
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        let historical_volatility = variance.sqrt() * self.config.volatility_config.annualization_factor.sqrt();

        // Realized volatility (sum of squared returns)
        let realized_volatility = returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>().sqrt() * self.config.volatility_config.annualization_factor.sqrt();

        // Exponentially weighted volatility
        let exponential_volatility = self.calculate_exponential_volatility(&returns)?;

        // GARCH volatility (simplified)
        let garch_volatility = if returns.len() > 50 {
            Some(self.calculate_garch_volatility(&returns)?)
        } else {
            None
        };

        // Parkinson volatility (requires OHLC data - simplified here)
        let parkinson_volatility = Some(historical_volatility * 1.67); // Approximation

        // Volatility of volatility
        let rolling_vols: Vec<f64> = returns.windows(10)
            .map(|window| {
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let var = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64;
                var.sqrt()
            })
            .collect();

        let vol_of_vol = if rolling_vols.len() > 1 {
            let vol_mean = rolling_vols.iter().sum::<f64>() / rolling_vols.len() as f64;
            let vol_var = rolling_vols.iter()
                .map(|v| (v - vol_mean).powi(2))
                .sum::<f64>() / (rolling_vols.len() - 1) as f64;
            vol_var.sqrt()
        } else {
            0.0
        };

        // Volatility clustering measure
        let volatility_clustering = self.calculate_volatility_clustering(&returns);

        Ok(VolatilityMeasures {
            historical_volatility,
            realized_volatility,
            exponential_volatility,
            garch_volatility,
            parkinson_volatility,
            volatility_of_volatility: vol_of_vol,
            volatility_clustering,
        })
    }

    fn calculate_exponential_volatility(&self, returns: &[f64]) -> Result<f64> {
        let lambda: f64 = 0.94; // Decay factor
        let mut weights = Vec::new();
        let mut weight_sum = 0.0;

        for i in 0..returns.len() {
            let weight = lambda.powi(i as i32);
            weights.push(weight);
            weight_sum += weight;
        }

        // Normalize weights
        for weight in &mut weights {
            *weight /= weight_sum;
        }

        // Calculate weighted mean
        let weighted_mean = returns.iter()
            .rev()
            .zip(weights.iter())
            .map(|(r, w)| r * w)
            .sum::<f64>();

        // Calculate weighted variance
        let weighted_variance = returns.iter()
            .rev()
            .zip(weights.iter())
            .map(|(r, w)| (r - weighted_mean).powi(2) * w)
            .sum::<f64>();

        Ok(weighted_variance.sqrt() * self.config.volatility_config.annualization_factor.sqrt())
    }

    fn calculate_garch_volatility(&self, returns: &[f64]) -> Result<f64> {
        // Simplified GARCH(1,1) implementation
        let params = &self.config.volatility_config.garch_params;
        let mut sigma_squared = returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;

        // Iterate GARCH equation
        for return_val in returns.iter().rev().take(10) {
            sigma_squared = params.omega + params.alpha * return_val.powi(2) + params.beta * sigma_squared;
        }

        Ok(sigma_squared.sqrt() * self.config.volatility_config.annualization_factor.sqrt())
    }

    fn calculate_volatility_clustering(&self, returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return 0.0;
        }

        // Calculate autocorrelation of squared returns (measure of clustering)
        let squared_returns: Vec<f64> = returns.iter().map(|r| r.powi(2)).collect();
        let mean_sq = squared_returns.iter().sum::<f64>() / squared_returns.len() as f64;

        let lag1_corr = squared_returns.windows(2)
            .map(|pair| (pair[0] - mean_sq) * (pair[1] - mean_sq))
            .sum::<f64>() / squared_returns.len() as f64;

        let variance_sq = squared_returns.iter()
            .map(|sq| (sq - mean_sq).powi(2))
            .sum::<f64>() / squared_returns.len() as f64;

        if variance_sq > 0.0 {
            lag1_corr / variance_sq
        } else {
            0.0
        }
    }

    fn calculate_distribution_analysis(&self) -> Result<DistributionAnalysis> {
        let returns: Vec<f64> = self.data_buffer.iter()
            .filter_map(|dp| dp.returns)
            .collect();

        if returns.len() < self.config.distribution_config.min_sample_size {
            return Err(Error::Analysis("Insufficient data for distribution analysis".to_string()));
        }

        // Normality test
        let normality_test = self.jarque_bera_test(&returns)?;

        // Histogram
        let histogram = self.create_histogram(&returns)?;

        // Moments
        let moments = self.calculate_moments(&returns)?;

        // Tail analysis
        let tail_analysis = self.calculate_tail_analysis(&returns)?;

        Ok(DistributionAnalysis {
            normality_test,
            histogram,
            moments,
            tail_analysis,
        })
    }

    fn jarque_bera_test(&self, returns: &[f64]) -> Result<NormalityTest> {
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let skewness = self.calculate_skewness(returns, mean, std_dev);
        let kurtosis = self.calculate_kurtosis(returns, mean, std_dev);

        // Jarque-Bera statistic
        let jb_statistic = (n / 6.0) * (skewness.powi(2) + (kurtosis.powi(2) / 4.0));
        
        // Approximate p-value (chi-square distribution with 2 degrees of freedom)
        let p_value = if jb_statistic > 5.991 { 0.01 } else if jb_statistic > 4.605 { 0.05 } else { 0.1 };
        
        let is_normal = p_value > self.config.distribution_config.normality_test_alpha;

        Ok(NormalityTest {
            jarque_bera_statistic: jb_statistic,
            jarque_bera_p_value: p_value,
            shapiro_wilk_statistic: None, // Would require more complex implementation
            shapiro_wilk_p_value: None,
            is_normal,
        })
    }

    fn create_histogram(&self, returns: &[f64]) -> Result<Histogram> {
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_val = sorted_returns[0];
        let max_val = sorted_returns[sorted_returns.len() - 1];
        let range = max_val - min_val;
        let bin_width = range / self.config.distribution_config.histogram_bins as f64;

        let mut bin_edges = Vec::new();
        let mut frequencies = vec![0; self.config.distribution_config.histogram_bins];

        // Create bin edges
        for i in 0..=self.config.distribution_config.histogram_bins {
            bin_edges.push(min_val + i as f64 * bin_width);
        }

        // Count frequencies
        for &value in returns {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(self.config.distribution_config.histogram_bins - 1);
            frequencies[bin_index] += 1;
        }

        // Create bin centers
        let bins: Vec<f64> = (0..self.config.distribution_config.histogram_bins)
            .map(|i| min_val + (i as f64 + 0.5) * bin_width)
            .collect();

        Ok(Histogram {
            bins,
            frequencies,
            bin_edges,
        })
    }

    fn calculate_moments(&self, returns: &[f64]) -> Result<Moments> {
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;

        let second_moment = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let third_moment = returns.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / n;
        let fourth_moment = returns.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / n;

        let std_dev = second_moment.sqrt();
        let skewness = if std_dev > 0.0 { third_moment / std_dev.powi(3) } else { 0.0 };
        let kurtosis = if std_dev > 0.0 { fourth_moment / std_dev.powi(4) } else { 0.0 };
        let excess_kurtosis = kurtosis - 3.0;

        Ok(Moments {
            first_moment: mean,
            second_moment,
            third_moment,
            fourth_moment,
            excess_kurtosis,
        })
    }

    fn calculate_tail_analysis(&self, returns: &[f64]) -> Result<TailAnalysis> {
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_returns.len();
        let left_tail_5pct = sorted_returns[n / 20]; // 5th percentile
        let right_tail_5pct = sorted_returns[n * 19 / 20]; // 95th percentile

        let left_tail_count = returns.iter().filter(|&&r| r <= left_tail_5pct).count();
        let right_tail_count = returns.iter().filter(|&&r| r >= right_tail_5pct).count();

        let left_tail_probability = left_tail_count as f64 / n as f64;
        let right_tail_probability = right_tail_count as f64 / n as f64;
        let tail_ratio = right_tail_probability / left_tail_probability.max(1e-10);

        // Calculate VaR and CVaR
        let var_95 = -sorted_returns[(n as f64 * 0.05) as usize];
        let var_99 = -sorted_returns[(n as f64 * 0.01) as usize];

        let tail_5pct: Vec<f64> = sorted_returns.iter().take(n / 20).cloned().collect();
        let tail_1pct: Vec<f64> = sorted_returns.iter().take(n / 100).cloned().collect();

        let cvar_95 = if !tail_5pct.is_empty() {
            -tail_5pct.iter().sum::<f64>() / tail_5pct.len() as f64
        } else {
            var_95
        };

        let cvar_99 = if !tail_1pct.is_empty() {
            -tail_1pct.iter().sum::<f64>() / tail_1pct.len() as f64
        } else {
            var_99
        };

        // Calculate max drawdown and runup
        let cumulative_returns: Vec<f64> = returns.iter()
            .scan(0.0, |acc, &r| {
                *acc += r;
                Some(*acc)
            })
            .collect();

        let mut max_drawdown: f64 = 0.0;
        let mut max_runup: f64 = 0.0;
        let mut peak = cumulative_returns[0];
        let mut trough = cumulative_returns[0];

        for &cum_ret in &cumulative_returns {
            if cum_ret > peak {
                peak = cum_ret;
                let runup = peak - trough;
                max_runup = max_runup.max(runup);
            }
            if cum_ret < trough {
                trough = cum_ret;
            }
            let drawdown = peak - cum_ret;
            max_drawdown = max_drawdown.max(drawdown);
        }

        let extreme_value_statistics = ExtremeValueStats {
            max_drawdown,
            max_runup,
            var_95,
            var_99,
            cvar_95,
            cvar_99,
        };

        Ok(TailAnalysis {
            left_tail_probability,
            right_tail_probability,
            tail_ratio,
            extreme_value_statistics,
        })
    }

    fn calculate_time_series_analysis(&self) -> Result<TimeSeriesAnalysis> {
        let returns: Vec<f64> = self.data_buffer.iter()
            .filter_map(|dp| dp.returns)
            .collect();

        if returns.len() < 50 {
            return Ok(TimeSeriesAnalysis {
                autocorrelation: vec![],
                partial_autocorrelation: vec![],
                ljung_box_test: LjungBoxTest {
                    statistic: 0.0,
                    p_value: 1.0,
                    is_white_noise: true,
                },
                adf_test: AdfTest {
                    statistic: 0.0,
                    p_value: 1.0,
                    critical_values: std::collections::HashMap::new(),
                    is_stationary: false,
                },
                trend_analysis: TrendAnalysis {
                    trend_direction: TrendDirection::Sideways,
                    trend_strength: 0.0,
                    trend_duration: Duration::zero(),
                    regime_changes: vec![],
                },
            });
        }

        // Calculate autocorrelation function
        let autocorrelation = self.calculate_autocorrelation(&returns, 20);

        // Ljung-Box test for serial correlation
        let ljung_box_test = self.ljung_box_test(&returns, 10)?;

        // Simplified ADF test
        let adf_test = self.adf_test(&returns)?;

        // Trend analysis
        let trend_analysis = self.analyze_trend(&returns)?;

        Ok(TimeSeriesAnalysis {
            autocorrelation,
            partial_autocorrelation: vec![], // Would require more complex calculation
            ljung_box_test,
            adf_test,
            trend_analysis,
        })
    }

    fn calculate_autocorrelation(&self, data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        let mut autocorr = Vec::new();
        
        for lag in 0..=max_lag {
            if lag >= n {
                break;
            }

            let covariance = (0..(n - lag))
                .map(|i| (data[i] - mean) * (data[i + lag] - mean))
                .sum::<f64>() / (n - lag) as f64;

            let correlation = if variance > 0.0 { covariance / variance } else { 0.0 };
            autocorr.push(correlation);
        }

        autocorr
    }

    fn ljung_box_test(&self, returns: &[f64], lags: usize) -> Result<LjungBoxTest> {
        let autocorr = self.calculate_autocorrelation(returns, lags);
        let n = returns.len() as f64;

        let mut lb_statistic = 0.0;
        for (k, &rho_k) in autocorr.iter().enumerate().skip(1) {
            lb_statistic += rho_k.powi(2) / (n - k as f64);
        }
        lb_statistic *= n * (n + 2.0);

        // Simplified p-value calculation (chi-square with `lags` degrees of freedom)
        let p_value = if lb_statistic > 18.307 { 0.01 } else if lb_statistic > 12.592 { 0.05 } else { 0.1 };
        let is_white_noise = p_value > 0.05;

        Ok(LjungBoxTest {
            statistic: lb_statistic,
            p_value,
            is_white_noise,
        })
    }

    fn adf_test(&self, data: &[f64]) -> Result<AdfTest> {
        // Simplified ADF test - in practice would use proper regression
        let n = data.len();
        let differences: Vec<f64> = data.windows(2).map(|pair| pair[1] - pair[0]).collect();
        
        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance_diff = differences.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / differences.len() as f64;
        let std_dev_diff = variance_diff.sqrt();

        // Simplified test statistic
        let t_statistic = mean_diff / (std_dev_diff / (n as f64).sqrt());

        let mut critical_values = std::collections::HashMap::new();
        critical_values.insert("1%".to_string(), -3.43);
        critical_values.insert("5%".to_string(), -2.86);
        critical_values.insert("10%".to_string(), -2.57);

        let p_value = if t_statistic < -3.43 { 0.01 } else if t_statistic < -2.86 { 0.05 } else { 0.1 };
        let is_stationary = p_value < 0.05;

        Ok(AdfTest {
            statistic: t_statistic,
            p_value,
            critical_values,
            is_stationary,
        })
    }

    fn analyze_trend(&self, returns: &[f64]) -> Result<TrendAnalysis> {
        let prices: Vec<f64> = self.data_buffer.iter().map(|dp| dp.price).collect();
        
        if prices.len() < 10 {
            return Ok(TrendAnalysis {
                trend_direction: TrendDirection::Sideways,
                trend_strength: 0.0,
                trend_duration: Duration::zero(),
                regime_changes: vec![],
            });
        }

        // Simple linear regression to determine trend
        let n = prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = prices.iter().sum::<f64>() / n;

        let numerator = prices.iter().enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum::<f64>();

        let denominator = (0..prices.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum::<f64>();

        let slope = if denominator > 0.0 { numerator / denominator } else { 0.0 };
        
        let trend_direction = if slope > 0.001 {
            TrendDirection::Upward
        } else if slope < -0.001 {
            TrendDirection::Downward
        } else {
            TrendDirection::Sideways
        };

        let trend_strength = slope.abs();

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength,
            trend_duration: Duration::seconds(prices.len() as i64 * 60), // Assuming 1-minute intervals
            regime_changes: vec![], // Would require change point detection
        })
    }

    fn calculate_risk_metrics(&self) -> Result<RiskMetrics> {
        let returns: Vec<f64> = self.data_buffer.iter()
            .filter_map(|dp| dp.returns)
            .collect();

        if returns.is_empty() {
            return Ok(RiskMetrics {
                value_at_risk: ValueAtRisk {
                    var_95: 0.0,
                    var_99: 0.0,
                    var_99_9: 0.0,
                },
                expected_shortfall: ExpectedShortfall {
                    es_95: 0.0,
                    es_99: 0.0,
                    es_99_9: 0.0,
                },
                maximum_drawdown: 0.0,
                calmar_ratio: 0.0,
                sortino_ratio: 0.0,
                information_ratio: 0.0,
            });
        }

        // Calculate VaR and Expected Shortfall
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_returns.len();
        let var_95 = -sorted_returns[(n as f64 * 0.05) as usize];
        let var_99 = -sorted_returns[(n as f64 * 0.01) as usize];
        let var_99_9 = -sorted_returns[((n as f64 * 0.001) as usize).max(0)];

        let tail_5pct: Vec<f64> = sorted_returns.iter().take(n / 20).cloned().collect();
        let tail_1pct: Vec<f64> = sorted_returns.iter().take(n / 100).cloned().collect();
        let tail_0_1pct: Vec<f64> = sorted_returns.iter().take((n as f64 * 0.001) as usize).cloned().collect();

        let es_95 = if !tail_5pct.is_empty() {
            -tail_5pct.iter().sum::<f64>() / tail_5pct.len() as f64
        } else {
            var_95
        };

        let es_99 = if !tail_1pct.is_empty() {
            -tail_1pct.iter().sum::<f64>() / tail_1pct.len() as f64
        } else {
            var_99
        };

        let es_99_9 = if !tail_0_1pct.is_empty() {
            -tail_0_1pct.iter().sum::<f64>() / tail_0_1pct.len() as f64
        } else {
            var_99_9
        };

        // Calculate maximum drawdown
        let cumulative_returns: Vec<f64> = returns.iter()
            .scan(1.0, |acc, &r| {
                *acc *= 1.0 + r;
                Some(*acc)
            })
            .collect();

        let mut max_drawdown: f64 = 0.0;
        let mut peak = cumulative_returns[0];

        for &cum_ret in &cumulative_returns {
            if cum_ret > peak {
                peak = cum_ret;
            }
            let drawdown = (peak - cum_ret) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Calculate risk-adjusted ratios
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = {
            let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        };

        let calmar_ratio = if max_drawdown > 0.0 {
            mean_return * 252.0 / max_drawdown // Annualized
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_deviation = if !downside_returns.is_empty() {
            let downside_variance = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
            downside_variance.sqrt()
        } else {
            std_dev
        };

        let sortino_ratio = if downside_deviation > 0.0 {
            mean_return / downside_deviation * 252.0f64.sqrt()
        } else {
            0.0
        };

        let information_ratio = if std_dev > 0.0 {
            mean_return / std_dev * 252.0f64.sqrt()
        } else {
            0.0
        };

        Ok(RiskMetrics {
            value_at_risk: ValueAtRisk {
                var_95,
                var_99,
                var_99_9,
            },
            expected_shortfall: ExpectedShortfall {
                es_95,
                es_99,
                es_99_9,
            },
            maximum_drawdown: max_drawdown,
            calmar_ratio,
            sortino_ratio,
            information_ratio,
        })
    }

    async fn test_normality(&self) -> Result<TestResult> {
        let returns: Vec<f64> = self.data_buffer.iter()
            .filter_map(|dp| dp.returns)
            .collect();

        let normality_test = self.jarque_bera_test(&returns)?;
        
        Ok(TestResult {
            test_name: "Jarque-Bera Normality Test".to_string(),
            statistic: normality_test.jarque_bera_statistic,
            p_value: normality_test.jarque_bera_p_value,
            critical_value: 5.991, // Chi-square critical value at 5% significance
            conclusion: if normality_test.is_normal {
                "Fail to reject null hypothesis: Data appears normally distributed".to_string()
            } else {
                "Reject null hypothesis: Data does not appear normally distributed".to_string()
            },
        })
    }

    async fn test_stationarity(&self) -> Result<TestResult> {
        let prices: Vec<f64> = self.data_buffer.iter().map(|dp| dp.price).collect();
        let adf_test = self.adf_test(&prices)?;
        
        Ok(TestResult {
            test_name: "Augmented Dickey-Fuller Test".to_string(),
            statistic: adf_test.statistic,
            p_value: adf_test.p_value,
            critical_value: -2.86,
            conclusion: if adf_test.is_stationary {
                "Reject null hypothesis: Series is stationary".to_string()
            } else {
                "Fail to reject null hypothesis: Series has unit root (non-stationary)".to_string()
            },
        })
    }

    async fn test_independence(&self) -> Result<TestResult> {
        let returns: Vec<f64> = self.data_buffer.iter()
            .filter_map(|dp| dp.returns)
            .collect();

        let ljung_box = self.ljung_box_test(&returns, 10)?;
        
        Ok(TestResult {
            test_name: "Ljung-Box Test for Independence".to_string(),
            statistic: ljung_box.statistic,
            p_value: ljung_box.p_value,
            critical_value: 18.307, // Chi-square critical value for 10 lags at 5%
            conclusion: if ljung_box.is_white_noise {
                "Fail to reject null hypothesis: Series appears to be white noise (independent)".to_string()
            } else {
                "Reject null hypothesis: Series shows serial correlation (not independent)".to_string()
            },
        })
    }

    async fn test_mean_reversion(&self) -> Result<TestResult> {
        // Simplified mean reversion test using Hurst exponent approximation
        let prices: Vec<f64> = self.data_buffer.iter().map(|dp| dp.price).collect();
        
        if prices.len() < 50 {
            return Err(Error::Analysis("Insufficient data for mean reversion test".to_string()));
        }

        // Calculate log returns
        let log_returns: Vec<f64> = prices.windows(2)
            .map(|pair| (pair[1] / pair[0]).ln())
            .collect();

        // Simple variance ratio test
        let q = 10; // Look at 10-period variance
        let var_1 = log_returns.windows(2)
            .map(|pair| (pair[1] - pair[0]).powi(2))
            .sum::<f64>() / (log_returns.len() - 1) as f64;

        let q_returns: Vec<f64> = log_returns.chunks(q)
            .map(|chunk| chunk.iter().sum::<f64>())
            .collect();

        let var_q = if q_returns.len() > 1 {
            let mean_q = q_returns.iter().sum::<f64>() / q_returns.len() as f64;
            q_returns.iter().map(|r| (r - mean_q).powi(2)).sum::<f64>() / (q_returns.len() - 1) as f64
        } else {
            var_1
        };

        let variance_ratio = var_q / (q as f64 * var_1);
        let mean_reversion_statistic = (variance_ratio - 1.0).abs();

        let is_mean_reverting = variance_ratio < 0.8; // Simplified threshold
        let p_value = if is_mean_reverting { 0.02 } else { 0.2 };

        Ok(TestResult {
            test_name: "Variance Ratio Test for Mean Reversion".to_string(),
            statistic: mean_reversion_statistic,
            p_value,
            critical_value: 0.2,
            conclusion: if is_mean_reverting {
                "Evidence of mean reversion detected".to_string()
            } else {
                "No evidence of mean reversion detected".to_string()
            },
        })
    }
}

#[derive(Debug, Clone)]
pub enum HypothesisTest {
    Normality,
    Stationarity,
    Independence,
    MeanReversion,
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub critical_value: f64,
    pub conclusion: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_statistical_analysis_creation() {
        let config = StatisticalAnalysisConfig::default();
        let sa = StatisticalAnalysis::new(config);
        
        assert_eq!(sa.data_buffer.len(), 0);
    }

    #[tokio::test]
    async fn test_add_data() {
        let config = StatisticalAnalysisConfig::default();
        let mut sa = StatisticalAnalysis::new(config);
        
        let market_data = MarketData {
            symbol: "BTC/USD".to_string(),
            timestamp: Utc::now(),
            bid: dec!(50000),
            ask: dec!(50001),
            mid: dec!(50000.5),
            last: dec!(50000),
            volume_24h: dec!(1000),
            bid_size: dec!(10),
            ask_size: dec!(10),
        };

        let result = sa.add_data(&market_data).await;
        assert!(result.is_ok());
        assert_eq!(sa.data_buffer.len(), 1);
    }
}