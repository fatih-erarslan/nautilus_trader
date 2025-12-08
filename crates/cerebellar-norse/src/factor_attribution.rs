//! Factor Analysis and Performance Attribution Framework
//!
//! This module provides comprehensive factor analysis, style attribution,
//! and performance decomposition for neural trading strategies.

use std::collections::HashMap;
use nalgebra::{DMatrix, DVector, SVD};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};

/// Comprehensive factor analysis and attribution framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAttributionEngine {
    /// Risk factor models
    pub risk_factors: RiskFactorModel,
    /// Style analysis
    pub style_analysis: StyleAnalysis,
    /// Performance attribution
    pub performance_attribution: PerformanceAttribution,
    /// Factor exposures
    pub factor_exposures: FactorExposures,
    /// Regime-based attribution
    pub regime_attribution: RegimeAttribution,
}

/// Multi-factor risk model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactorModel {
    /// Market factors (beta, momentum, etc.)
    pub market_factors: MarketFactors,
    /// Macroeconomic factors
    pub macro_factors: MacroFactors,
    /// Fundamental factors
    pub fundamental_factors: FundamentalFactors,
    /// Statistical factors (PCA)
    pub statistical_factors: StatisticalFactors,
    /// Custom neural factors
    pub neural_factors: NeuralFactors,
}

/// Market-based risk factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFactors {
    /// Market beta
    pub market_beta: f64,
    /// Size factor (SMB - Small Minus Big)
    pub size_factor: f64,
    /// Value factor (HML - High Minus Low)
    pub value_factor: f64,
    /// Momentum factor
    pub momentum_factor: f64,
    /// Quality factor
    pub quality_factor: f64,
    /// Low volatility factor
    pub low_vol_factor: f64,
    /// Profitability factor
    pub profitability_factor: f64,
    /// Investment factor
    pub investment_factor: f64,
}

/// Macroeconomic factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroFactors {
    /// Interest rate sensitivity
    pub interest_rate_factor: f64,
    /// Inflation sensitivity
    pub inflation_factor: f64,
    /// Currency factor
    pub currency_factor: f64,
    /// Credit spread factor
    pub credit_spread_factor: f64,
    /// Volatility factor (VIX)
    pub volatility_factor: f64,
    /// Oil price factor
    pub oil_factor: f64,
    /// GDP growth factor
    pub gdp_factor: f64,
}

/// Fundamental factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundamentalFactors {
    /// Earnings growth
    pub earnings_growth: f64,
    /// Revenue growth
    pub revenue_growth: f64,
    /// Return on equity
    pub roe_factor: f64,
    /// Debt-to-equity
    pub leverage_factor: f64,
    /// Price-to-earnings
    pub pe_factor: f64,
    /// Price-to-book
    pub pb_factor: f64,
    /// Dividend yield
    pub dividend_yield: f64,
}

/// Principal component analysis factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFactors {
    /// Principal components
    pub principal_components: Vec<PrincipalComponent>,
    /// Factor loadings
    pub factor_loadings: DMatrix<f64>,
    /// Explained variance ratios
    pub explained_variance: Vec<f64>,
    /// Cumulative explained variance
    pub cumulative_variance: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalComponent {
    /// Component number
    pub component_id: usize,
    /// Eigenvalue
    pub eigenvalue: f64,
    /// Eigenvector
    pub eigenvector: Vec<f64>,
    /// Variance explained
    pub variance_explained: f64,
    /// Interpretation
    pub interpretation: String,
}

/// Neural network derived factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralFactors {
    /// Sentiment factor from news
    pub sentiment_factor: f64,
    /// Technical pattern factor
    pub technical_pattern_factor: f64,
    /// Volatility regime factor
    pub volatility_regime_factor: f64,
    /// Cross-asset correlation factor
    pub correlation_factor: f64,
    /// Market microstructure factor
    pub microstructure_factor: f64,
    /// Alternative data factor
    pub alternative_data_factor: f64,
}

/// Style analysis framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAnalysis {
    /// Returns-based style analysis
    pub returns_based: ReturnsBasedStyleAnalysis,
    /// Holdings-based style analysis
    pub holdings_based: HoldingsBasedStyleAnalysis,
    /// Time-varying style analysis
    pub time_varying: TimeVaryingStyleAnalysis,
    /// Style consistency metrics
    pub consistency_metrics: StyleConsistencyMetrics,
}

/// Returns-based style analysis (Sharpe method)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnsBasedStyleAnalysis {
    /// Style weights (sum to 1)
    pub style_weights: HashMap<String, f64>,
    /// R-squared of style regression
    pub r_squared: f64,
    /// Selection skill (alpha)
    pub selection_skill: f64,
    /// Tracking error
    pub tracking_error: f64,
    /// Active share
    pub active_share: f64,
    /// Style drift measure
    pub style_drift: f64,
}

/// Holdings-based style analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoldingsBasedStyleAnalysis {
    /// Portfolio characteristics
    pub portfolio_characteristics: PortfolioCharacteristics,
    /// Sector allocations
    pub sector_allocations: HashMap<String, f64>,
    /// Geographic allocations
    pub geographic_allocations: HashMap<String, f64>,
    /// Market cap distribution
    pub market_cap_distribution: MarketCapDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioCharacteristics {
    /// Weighted average market cap
    pub avg_market_cap: f64,
    /// Weighted average P/E ratio
    pub avg_pe_ratio: f64,
    /// Weighted average P/B ratio
    pub avg_pb_ratio: f64,
    /// Weighted average dividend yield
    pub avg_dividend_yield: f64,
    /// Weighted average beta
    pub avg_beta: f64,
    /// Portfolio turnover
    pub turnover: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketCapDistribution {
    /// Large cap allocation (>$10B)
    pub large_cap: f64,
    /// Mid cap allocation ($2B-$10B)
    pub mid_cap: f64,
    /// Small cap allocation (<$2B)
    pub small_cap: f64,
}

/// Time-varying style analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeVaryingStyleAnalysis {
    /// Rolling style exposures
    pub rolling_exposures: Vec<RollingStyleExposure>,
    /// Style timing ability
    pub style_timing: StyleTimingMetrics,
    /// Conditional style analysis
    pub conditional_analysis: ConditionalStyleAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingStyleExposure {
    /// Time period
    pub period: (DateTime<Utc>, DateTime<Utc>),
    /// Style exposures for this period
    pub exposures: HashMap<String, f64>,
    /// R-squared for this period
    pub r_squared: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTimingMetrics {
    /// Style timing coefficient
    pub timing_coefficient: f64,
    /// Timing p-value
    pub timing_p_value: f64,
    /// Market timing ability
    pub market_timing: f64,
    /// Volatility timing ability
    pub volatility_timing: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalStyleAnalysis {
    /// Style exposures conditional on market state
    pub market_state_exposures: HashMap<String, HashMap<String, f64>>,
    /// Style exposures conditional on volatility regime
    pub volatility_regime_exposures: HashMap<String, HashMap<String, f64>>,
}

/// Style consistency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConsistencyMetrics {
    /// Style consistency score (0-1)
    pub consistency_score: f64,
    /// Maximum style drift
    pub max_style_drift: f64,
    /// Average style drift
    pub avg_style_drift: f64,
    /// Style volatility
    pub style_volatility: HashMap<String, f64>,
}

/// Performance attribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAttribution {
    /// Brinson attribution
    pub brinson_attribution: BrinsonAttribution,
    /// Factor-based attribution
    pub factor_attribution: FactorBasedAttribution,
    /// Security selection attribution
    pub security_selection: SecuritySelectionAttribution,
    /// Timing attribution
    pub timing_attribution: TimingAttribution,
}

/// Brinson performance attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrinsonAttribution {
    /// Asset allocation effect
    pub asset_allocation: f64,
    /// Security selection effect
    pub security_selection: f64,
    /// Interaction effect
    pub interaction_effect: f64,
    /// Total attribution
    pub total_attribution: f64,
    /// Sector-level attribution
    pub sector_attribution: HashMap<String, SectorAttribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorAttribution {
    /// Sector name
    pub sector: String,
    /// Asset allocation contribution
    pub allocation_contribution: f64,
    /// Selection contribution
    pub selection_contribution: f64,
    /// Interaction contribution
    pub interaction_contribution: f64,
    /// Total contribution
    pub total_contribution: f64,
}

/// Factor-based performance attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorBasedAttribution {
    /// Market factor contribution
    pub market_contribution: f64,
    /// Style factor contributions
    pub style_contributions: HashMap<String, f64>,
    /// Sector factor contributions
    pub sector_contributions: HashMap<String, f64>,
    /// Specific return (alpha)
    pub specific_return: f64,
    /// Factor timing contributions
    pub timing_contributions: HashMap<String, f64>,
}

/// Security selection attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySelectionAttribution {
    /// Pure security selection
    pub pure_selection: f64,
    /// Within-sector selection
    pub within_sector_selection: HashMap<String, f64>,
    /// Cross-sector selection
    pub cross_sector_selection: f64,
    /// Security-specific alpha
    pub security_alpha: HashMap<String, f64>,
}

/// Timing attribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingAttribution {
    /// Market timing contribution
    pub market_timing: f64,
    /// Sector timing contribution
    pub sector_timing: HashMap<String, f64>,
    /// Factor timing contribution
    pub factor_timing: HashMap<String, f64>,
    /// Volatility timing contribution
    pub volatility_timing: f64,
}

/// Factor exposures analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorExposures {
    /// Current factor exposures
    pub current_exposures: HashMap<String, f64>,
    /// Historical exposures
    pub historical_exposures: Vec<HistoricalExposure>,
    /// Exposure limits
    pub exposure_limits: HashMap<String, ExposureLimit>,
    /// Risk decomposition
    pub risk_decomposition: RiskDecomposition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalExposure {
    /// Date
    pub date: DateTime<Utc>,
    /// Factor exposures on this date
    pub exposures: HashMap<String, f64>,
    /// Risk contribution
    pub risk_contribution: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureLimit {
    /// Minimum exposure
    pub min_exposure: f64,
    /// Maximum exposure
    pub max_exposure: f64,
    /// Target exposure
    pub target_exposure: f64,
    /// Current exposure
    pub current_exposure: f64,
    /// Limit breach indicator
    pub limit_breached: bool,
}

/// Risk decomposition by factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDecomposition {
    /// Factor risk contributions
    pub factor_contributions: HashMap<String, f64>,
    /// Specific risk
    pub specific_risk: f64,
    /// Total risk
    pub total_risk: f64,
    /// Risk percentages
    pub risk_percentages: HashMap<String, f64>,
    /// Correlation matrix
    pub correlation_matrix: DMatrix<f64>,
}

/// Regime-based attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAttribution {
    /// Performance by market regime
    pub regime_performance: HashMap<String, RegimePerformance>,
    /// Regime transition analysis
    pub regime_transitions: RegimeTransitionAnalysis,
    /// Factor sensitivity by regime
    pub regime_factor_sensitivity: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePerformance {
    /// Regime identifier
    pub regime_id: String,
    /// Performance in this regime
    pub performance: f64,
    /// Benchmark performance
    pub benchmark_performance: f64,
    /// Excess return
    pub excess_return: f64,
    /// Time in regime
    pub time_in_regime: Duration,
    /// Hit rate in regime
    pub hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTransitionAnalysis {
    /// Transition performance
    pub transition_performance: HashMap<String, f64>,
    /// Regime prediction accuracy
    pub regime_prediction_accuracy: f64,
    /// Transition timing
    pub transition_timing: HashMap<String, TimingMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Average time to detect transition
    pub detection_time: Duration,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
    /// Transition alpha
    pub transition_alpha: f64,
}

impl FactorAttributionEngine {
    /// Create new factor attribution engine
    pub fn new() -> Self {
        Self {
            risk_factors: RiskFactorModel::new(),
            style_analysis: StyleAnalysis::new(),
            performance_attribution: PerformanceAttribution::new(),
            factor_exposures: FactorExposures::new(),
            regime_attribution: RegimeAttribution::new(),
        }
    }

    /// Perform comprehensive factor analysis
    pub fn analyze_factors(
        &mut self,
        returns: &[f64],
        benchmark_returns: &[f64],
        factor_returns: &HashMap<String, Vec<f64>>,
        timestamps: &[DateTime<Utc>],
    ) -> FactorAnalysisResults {
        // Calculate factor exposures
        self.calculate_factor_exposures(returns, factor_returns);
        
        // Perform style analysis
        self.perform_style_analysis(returns, benchmark_returns, factor_returns);
        
        // Calculate performance attribution
        self.calculate_performance_attribution(returns, benchmark_returns, factor_returns);
        
        // Analyze regime-based performance
        self.analyze_regime_performance(returns, benchmark_returns, timestamps);
        
        // Calculate risk decomposition
        self.calculate_risk_decomposition(factor_returns);

        FactorAnalysisResults {
            factor_exposures: self.factor_exposures.current_exposures.clone(),
            style_weights: self.style_analysis.returns_based.style_weights.clone(),
            performance_attribution: self.performance_attribution.clone(),
            risk_decomposition: self.factor_exposures.risk_decomposition.clone(),
            alpha: self.style_analysis.returns_based.selection_skill,
            tracking_error: self.style_analysis.returns_based.tracking_error,
            information_ratio: if self.style_analysis.returns_based.tracking_error > 0.0 {
                self.style_analysis.returns_based.selection_skill / self.style_analysis.returns_based.tracking_error
            } else { 0.0 },
        }
    }

    /// Calculate factor exposures using regression analysis
    fn calculate_factor_exposures(&mut self, returns: &[f64], factor_returns: &HashMap<String, Vec<f64>>) {
        let n = returns.len();
        if n < 2 {
            return;
        }

        // Prepare factor matrix
        let num_factors = factor_returns.len();
        let mut factor_matrix = DMatrix::zeros(n, num_factors + 1); // +1 for intercept
        let mut factor_names = vec!["alpha".to_string()];
        
        // Add intercept column
        for i in 0..n {
            factor_matrix[(i, 0)] = 1.0;
        }

        // Add factor data
        let mut col_idx = 1;
        for (factor_name, factor_data) in factor_returns {
            factor_names.push(factor_name.clone());
            for (i, &value) in factor_data.iter().enumerate().take(n) {
                if i < n {
                    factor_matrix[(i, col_idx)] = value;
                }
            }
            col_idx += 1;
        }

        // Prepare return vector
        let return_vector = DVector::from_row_slice(returns);

        // Perform multiple regression using normal equations
        let xt = factor_matrix.transpose();
        let xtx = &xt * &factor_matrix;
        let xty = &xt * &return_vector;

        if let Some(xtx_inv) = xtx.try_inverse() {
            let coefficients = xtx_inv * xty;
            
            // Store factor exposures
            for (i, factor_name) in factor_names.iter().enumerate() {
                if i < coefficients.len() {
                    self.factor_exposures.current_exposures.insert(
                        factor_name.clone(),
                        coefficients[i]
                    );
                }
            }

            // Calculate R-squared
            let predicted = &factor_matrix * &coefficients;
            let residuals = &return_vector - &predicted;
            let ss_res = residuals.dot(&residuals);
            let mean_return = return_vector.mean();
            let total_deviations = return_vector.map(|r| r - mean_return);
            let ss_tot = total_deviations.dot(&total_deviations);
            
            let r_squared = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };
            
            // Store R-squared in style analysis
            self.style_analysis.returns_based.r_squared = r_squared;
            
            // Store alpha (intercept)
            if !coefficients.is_empty() {
                self.style_analysis.returns_based.selection_skill = coefficients[0];
            }
        }
    }

    /// Perform returns-based style analysis
    fn perform_style_analysis(
        &mut self,
        returns: &[f64],
        benchmark_returns: &[f64],
        factor_returns: &HashMap<String, Vec<f64>>
    ) {
        let n = returns.len().min(benchmark_returns.len());
        if n < 2 {
            return;
        }

        // Calculate tracking error
        let mut tracking_errors = Vec::new();
        for i in 0..n {
            tracking_errors.push(returns[i] - benchmark_returns[i]);
        }
        
        let mean_tracking_error = tracking_errors.iter().sum::<f64>() / n as f64;
        let tracking_error_variance = tracking_errors.iter()
            .map(|e| (e - mean_tracking_error).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        
        self.style_analysis.returns_based.tracking_error = tracking_error_variance.sqrt() * (252.0_f64).sqrt();

        // Calculate active share (simplified)
        self.style_analysis.returns_based.active_share = 0.5; // Would need holdings data for accurate calculation

        // Calculate style consistency
        self.calculate_style_consistency();
    }

    /// Calculate style consistency metrics
    fn calculate_style_consistency(&mut self) {
        // Simplified style consistency calculation
        // In practice, would analyze rolling style exposures
        let num_exposures = self.factor_exposures.current_exposures.len();
        if num_exposures > 0 {
            let exposure_variance: f64 = self.factor_exposures.current_exposures
                .values()
                .map(|&exp| exp.powi(2))
                .sum::<f64>() / num_exposures as f64;
            
            self.style_analysis.consistency_metrics.consistency_score = 
                1.0 / (1.0 + exposure_variance); // Higher variance = lower consistency
        }
    }

    /// Calculate performance attribution
    fn calculate_performance_attribution(
        &mut self,
        returns: &[f64],
        benchmark_returns: &[f64],
        factor_returns: &HashMap<String, Vec<f64>>
    ) {
        let n = returns.len().min(benchmark_returns.len());
        if n < 2 {
            return;
        }

        // Calculate total excess return
        let total_excess: f64 = returns.iter().zip(benchmark_returns.iter())
            .map(|(r, b)| r - b)
            .sum();

        // Factor-based attribution
        let mut factor_contributions = HashMap::new();
        let mut total_factor_contribution = 0.0;

        for (factor_name, factor_data) in factor_returns {
            if let Some(&exposure) = self.factor_exposures.current_exposures.get(factor_name) {
                let factor_return: f64 = factor_data.iter().take(n).sum();
                let contribution = exposure * factor_return;
                factor_contributions.insert(factor_name.clone(), contribution);
                total_factor_contribution += contribution;
            }
        }

        // Specific return (alpha)
        let specific_return = total_excess - total_factor_contribution;

        self.performance_attribution.factor_attribution = FactorBasedAttribution {
            market_contribution: factor_contributions.get("market").cloned().unwrap_or(0.0),
            style_contributions: factor_contributions.clone(),
            sector_contributions: HashMap::new(), // Would need sector data
            specific_return,
            timing_contributions: HashMap::new(), // Would need timing analysis
        };
    }

    /// Analyze regime-based performance
    fn analyze_regime_performance(
        &mut self,
        returns: &[f64],
        benchmark_returns: &[f64],
        timestamps: &[DateTime<Utc>]
    ) {
        // Simplified regime analysis
        // In practice, would use more sophisticated regime detection
        let n = returns.len().min(benchmark_returns.len()).min(timestamps.len());
        if n < 10 {
            return;
        }

        // Simple volatility-based regime classification
        let mut volatility_regimes = HashMap::new();
        let window_size = 20; // 20-day rolling window

        for i in window_size..n {
            let window_returns = &returns[i-window_size..i];
            let mean_return = window_returns.iter().sum::<f64>() / window_size as f64;
            let volatility = (window_returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / (window_size - 1) as f64).sqrt();

            let regime = if volatility > 0.02 { "high_vol" } else { "low_vol" };
            
            let entry = volatility_regimes.entry(regime.to_string()).or_insert(vec![]);
            entry.push((returns[i], benchmark_returns[i]));
        }

        // Calculate performance by regime
        for (regime, regime_returns) in volatility_regimes {
            let strategy_performance: f64 = regime_returns.iter().map(|(r, _)| r).sum();
            let benchmark_performance: f64 = regime_returns.iter().map(|(_, b)| b).sum();
            let excess_return = strategy_performance - benchmark_performance;

            let regime_perf = RegimePerformance {
                regime_id: regime.clone(),
                performance: strategy_performance,
                benchmark_performance,
                excess_return,
                time_in_regime: Duration::days(regime_returns.len() as i64),
                hit_rate: regime_returns.iter()
                    .filter(|(r, b)| r > b)
                    .count() as f64 / regime_returns.len() as f64,
            };

            self.regime_attribution.regime_performance.insert(regime, regime_perf);
        }
    }

    /// Calculate risk decomposition
    fn calculate_risk_decomposition(&mut self, factor_returns: &HashMap<String, Vec<f64>>) {
        let mut factor_contributions = HashMap::new();
        let mut total_factor_risk = 0.0;

        // Calculate factor risk contributions
        for (factor_name, factor_data) in factor_returns {
            if let Some(&exposure) = self.factor_exposures.current_exposures.get(factor_name) {
                if !factor_data.is_empty() {
                    let factor_volatility = self.calculate_volatility(factor_data);
                    let factor_risk = exposure.abs() * factor_volatility;
                    factor_contributions.insert(factor_name.clone(), factor_risk);
                    total_factor_risk += factor_risk.powi(2);
                }
            }
        }

        let total_factor_risk = total_factor_risk.sqrt();
        
        // Estimate specific risk (simplified)
        let specific_risk = total_factor_risk * 0.3; // Assume 30% specific risk
        let total_risk = (total_factor_risk.powi(2) + specific_risk.powi(2)).sqrt();

        // Calculate risk percentages
        let mut risk_percentages = HashMap::new();
        for (factor_name, &factor_risk) in &factor_contributions {
            if total_risk > 0.0 {
                risk_percentages.insert(factor_name.clone(), factor_risk / total_risk * 100.0);
            }
        }

        self.factor_exposures.risk_decomposition = RiskDecomposition {
            factor_contributions,
            specific_risk,
            total_risk,
            risk_percentages,
            correlation_matrix: DMatrix::identity(0, 0), // Would need factor correlation data
        };
    }

    /// Calculate volatility of a time series
    fn calculate_volatility(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;

        variance.sqrt() * (252.0_f64).sqrt() // Annualized volatility
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysisResults {
    pub factor_exposures: HashMap<String, f64>,
    pub style_weights: HashMap<String, f64>,
    pub performance_attribution: PerformanceAttribution,
    pub risk_decomposition: RiskDecomposition,
    pub alpha: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
}

// Default implementations
impl RiskFactorModel {
    fn new() -> Self {
        Self {
            market_factors: MarketFactors::default(),
            macro_factors: MacroFactors::default(),
            fundamental_factors: FundamentalFactors::default(),
            statistical_factors: StatisticalFactors::default(),
            neural_factors: NeuralFactors::default(),
        }
    }
}

impl StyleAnalysis {
    fn new() -> Self {
        Self {
            returns_based: ReturnsBasedStyleAnalysis::default(),
            holdings_based: HoldingsBasedStyleAnalysis::default(),
            time_varying: TimeVaryingStyleAnalysis::default(),
            consistency_metrics: StyleConsistencyMetrics::default(),
        }
    }
}

impl PerformanceAttribution {
    fn new() -> Self {
        Self {
            brinson_attribution: BrinsonAttribution::default(),
            factor_attribution: FactorBasedAttribution::default(),
            security_selection: SecuritySelectionAttribution::default(),
            timing_attribution: TimingAttribution::default(),
        }
    }
}

impl FactorExposures {
    fn new() -> Self {
        Self {
            current_exposures: HashMap::new(),
            historical_exposures: Vec::new(),
            exposure_limits: HashMap::new(),
            risk_decomposition: RiskDecomposition::default(),
        }
    }
}

impl RegimeAttribution {
    fn new() -> Self {
        Self {
            regime_performance: HashMap::new(),
            regime_transitions: RegimeTransitionAnalysis::default(),
            regime_factor_sensitivity: HashMap::new(),
        }
    }
}

// Default trait implementations
impl Default for MarketFactors {
    fn default() -> Self {
        Self {
            market_beta: 0.0,
            size_factor: 0.0,
            value_factor: 0.0,
            momentum_factor: 0.0,
            quality_factor: 0.0,
            low_vol_factor: 0.0,
            profitability_factor: 0.0,
            investment_factor: 0.0,
        }
    }
}

impl Default for MacroFactors {
    fn default() -> Self {
        Self {
            interest_rate_factor: 0.0,
            inflation_factor: 0.0,
            currency_factor: 0.0,
            credit_spread_factor: 0.0,
            volatility_factor: 0.0,
            oil_factor: 0.0,
            gdp_factor: 0.0,
        }
    }
}

impl Default for FundamentalFactors {
    fn default() -> Self {
        Self {
            earnings_growth: 0.0,
            revenue_growth: 0.0,
            roe_factor: 0.0,
            leverage_factor: 0.0,
            pe_factor: 0.0,
            pb_factor: 0.0,
            dividend_yield: 0.0,
        }
    }
}

impl Default for StatisticalFactors {
    fn default() -> Self {
        Self {
            principal_components: Vec::new(),
            factor_loadings: DMatrix::zeros(0, 0),
            explained_variance: Vec::new(),
            cumulative_variance: Vec::new(),
        }
    }
}

impl Default for NeuralFactors {
    fn default() -> Self {
        Self {
            sentiment_factor: 0.0,
            technical_pattern_factor: 0.0,
            volatility_regime_factor: 0.0,
            correlation_factor: 0.0,
            microstructure_factor: 0.0,
            alternative_data_factor: 0.0,
        }
    }
}

impl Default for ReturnsBasedStyleAnalysis {
    fn default() -> Self {
        Self {
            style_weights: HashMap::new(),
            r_squared: 0.0,
            selection_skill: 0.0,
            tracking_error: 0.0,
            active_share: 0.0,
            style_drift: 0.0,
        }
    }
}

impl Default for HoldingsBasedStyleAnalysis {
    fn default() -> Self {
        Self {
            portfolio_characteristics: PortfolioCharacteristics::default(),
            sector_allocations: HashMap::new(),
            geographic_allocations: HashMap::new(),
            market_cap_distribution: MarketCapDistribution::default(),
        }
    }
}

impl Default for PortfolioCharacteristics {
    fn default() -> Self {
        Self {
            avg_market_cap: 0.0,
            avg_pe_ratio: 0.0,
            avg_pb_ratio: 0.0,
            avg_dividend_yield: 0.0,
            avg_beta: 0.0,
            turnover: 0.0,
        }
    }
}

impl Default for MarketCapDistribution {
    fn default() -> Self {
        Self {
            large_cap: 0.0,
            mid_cap: 0.0,
            small_cap: 0.0,
        }
    }
}

impl Default for TimeVaryingStyleAnalysis {
    fn default() -> Self {
        Self {
            rolling_exposures: Vec::new(),
            style_timing: StyleTimingMetrics::default(),
            conditional_analysis: ConditionalStyleAnalysis::default(),
        }
    }
}

impl Default for StyleTimingMetrics {
    fn default() -> Self {
        Self {
            timing_coefficient: 0.0,
            timing_p_value: 1.0,
            market_timing: 0.0,
            volatility_timing: 0.0,
        }
    }
}

impl Default for ConditionalStyleAnalysis {
    fn default() -> Self {
        Self {
            market_state_exposures: HashMap::new(),
            volatility_regime_exposures: HashMap::new(),
        }
    }
}

impl Default for StyleConsistencyMetrics {
    fn default() -> Self {
        Self {
            consistency_score: 0.0,
            max_style_drift: 0.0,
            avg_style_drift: 0.0,
            style_volatility: HashMap::new(),
        }
    }
}

impl Default for BrinsonAttribution {
    fn default() -> Self {
        Self {
            asset_allocation: 0.0,
            security_selection: 0.0,
            interaction_effect: 0.0,
            total_attribution: 0.0,
            sector_attribution: HashMap::new(),
        }
    }
}

impl Default for FactorBasedAttribution {
    fn default() -> Self {
        Self {
            market_contribution: 0.0,
            style_contributions: HashMap::new(),
            sector_contributions: HashMap::new(),
            specific_return: 0.0,
            timing_contributions: HashMap::new(),
        }
    }
}

impl Default for SecuritySelectionAttribution {
    fn default() -> Self {
        Self {
            pure_selection: 0.0,
            within_sector_selection: HashMap::new(),
            cross_sector_selection: 0.0,
            security_alpha: HashMap::new(),
        }
    }
}

impl Default for TimingAttribution {
    fn default() -> Self {
        Self {
            market_timing: 0.0,
            sector_timing: HashMap::new(),
            factor_timing: HashMap::new(),
            volatility_timing: 0.0,
        }
    }
}

impl Default for RiskDecomposition {
    fn default() -> Self {
        Self {
            factor_contributions: HashMap::new(),
            specific_risk: 0.0,
            total_risk: 0.0,
            risk_percentages: HashMap::new(),
            correlation_matrix: DMatrix::zeros(0, 0),
        }
    }
}

impl Default for RegimeTransitionAnalysis {
    fn default() -> Self {
        Self {
            transition_performance: HashMap::new(),
            regime_prediction_accuracy: 0.0,
            transition_timing: HashMap::new(),
        }
    }
}