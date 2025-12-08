//! # Market Regime Detection
//!
//! Advanced market regime detection using SOC (Self-Organized Criticality) index,
//! black swan risk assessment, and multi-scale phase analysis.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use anyhow::Result;

use crate::panarchy::{MarketRegime, CyclePhase, ScaleLevel, MarketData};

/// Market regime detector with SOC and black swan analysis
#[derive(Debug)]
pub struct MarketRegimeDetector {
    /// SOC index calculator
    soc_calculator: SOCIndexCalculator,
    
    /// Black swan risk assessor
    black_swan_assessor: BlackSwanRiskAssessor,
    
    /// Regime classifier
    regime_classifier: RegimeClassifier,
    
    /// Volatility regime tracker
    volatility_tracker: VolatilityRegimeTracker,
    
    /// Correlation regime tracker
    correlation_tracker: CorrelationRegimeTracker,
    
    /// Multi-scale regime analyzer
    multi_scale_analyzer: MultiScaleRegimeAnalyzer,
    
    /// Regime history
    regime_history: VecDeque<RegimeState>,
    
    /// Configuration
    config: RegimeDetectionConfig,
}

/// SOC (Self-Organized Criticality) index calculator
#[derive(Debug, Clone)]
pub struct SOCIndexCalculator {
    /// Power law exponent tracker
    power_law_exponent: f64,
    
    /// Avalanche size distribution
    avalanche_distribution: VecDeque<f64>,
    
    /// Criticality indicators
    criticality_indicators: CriticalityIndicators,
    
    /// SOC history
    soc_history: VecDeque<SOCMeasurement>,
    
    /// Configuration
    config: SOCConfig,
}

/// Black swan risk assessor
#[derive(Debug, Clone)]
pub struct BlackSwanRiskAssessor {
    /// Tail risk calculator
    tail_risk_calculator: TailRiskCalculator,
    
    /// Extreme event detector
    extreme_event_detector: ExtremeEventDetector,
    
    /// Fat tail analyzer
    fat_tail_analyzer: FatTailAnalyzer,
    
    /// Antifragility metrics
    antifragility_metrics: AntifragilityMetrics,
    
    /// Black swan history
    black_swan_history: VecDeque<BlackSwanEvent>,
    
    /// Configuration
    config: BlackSwanConfig,
}

/// Regime classifier for market state identification
#[derive(Debug, Clone)]
pub struct RegimeClassifier {
    /// Classification models
    models: HashMap<String, ClassificationModel>,
    
    /// Feature extractors
    feature_extractors: HashMap<String, FeatureExtractor>,
    
    /// Classification rules
    classification_rules: Vec<ClassificationRule>,
    
    /// Regime probabilities
    regime_probabilities: HashMap<MarketRegime, f64>,
    
    /// Configuration
    config: ClassifierConfig,
}

/// Volatility regime tracker
#[derive(Debug, Clone)]
pub struct VolatilityRegimeTracker {
    /// Current volatility regime
    current_regime: VolatilityRegime,
    
    /// Volatility clustering detector
    clustering_detector: VolatilityClusteringDetector,
    
    /// GARCH model
    garch_model: GARCHModel,
    
    /// Volatility forecaster
    volatility_forecaster: VolatilityForecaster,
    
    /// Regime history
    regime_history: VecDeque<VolatilityRegimeState>,
    
    /// Configuration
    config: VolatilityConfig,
}

/// Correlation regime tracker
#[derive(Debug, Clone)]
pub struct CorrelationRegimeTracker {
    /// Current correlation regime
    current_regime: CorrelationRegime,
    
    /// Correlation matrix
    correlation_matrix: CorrelationMatrix,
    
    /// Correlation breakdown detector
    breakdown_detector: CorrelationBreakdownDetector,
    
    /// Dynamic correlation model
    dynamic_correlation_model: DynamicCorrelationModel,
    
    /// Regime history
    regime_history: VecDeque<CorrelationRegimeState>,
    
    /// Configuration
    config: CorrelationConfig,
}

/// Multi-scale regime analyzer
#[derive(Debug, Clone)]
pub struct MultiScaleRegimeAnalyzer {
    /// Scale-specific analyzers
    scale_analyzers: HashMap<ScaleLevel, ScaleRegimeAnalyzer>,
    
    /// Cross-scale regime interactions
    cross_scale_interactions: HashMap<(ScaleLevel, ScaleLevel), InteractionStrength>,
    
    /// Multi-scale coherence
    coherence_calculator: CoherenceCalculator,
    
    /// Configuration
    config: MultiScaleConfig,
}

/// Current regime state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeState {
    /// Primary market regime
    pub primary_regime: MarketRegime,
    
    /// Secondary regime (if transitioning)
    pub secondary_regime: Option<MarketRegime>,
    
    /// Regime confidence
    pub confidence: f64,
    
    /// Regime stability
    pub stability: f64,
    
    /// Transition probability
    pub transition_probability: f64,
    
    /// SOC index
    pub soc_index: f64,
    
    /// Black swan risk
    pub black_swan_risk: f64,
    
    /// Volatility regime
    pub volatility_regime: VolatilityRegime,
    
    /// Correlation regime
    pub correlation_regime: CorrelationRegime,
    
    /// Multi-scale coherence
    pub multi_scale_coherence: f64,
    
    /// Timestamp
    pub timestamp: Instant,
}

/// Volatility regimes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VolatilityRegime {
    /// Low volatility regime
    Low,
    /// Moderate volatility regime
    Moderate,
    /// High volatility regime
    High,
    /// Extreme volatility regime
    Extreme,
    /// Volatility clustering
    Clustering,
}

/// Correlation regimes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorrelationRegime {
    /// Low correlation regime
    Low,
    /// Moderate correlation regime
    Moderate,
    /// High correlation regime
    High,
    /// Correlation breakdown
    Breakdown,
    /// Flight to quality
    FlightToQuality,
}

/// SOC measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCMeasurement {
    /// SOC index value
    pub index: f64,
    
    /// Power law exponent
    pub power_law_exponent: f64,
    
    /// Criticality level
    pub criticality_level: f64,
    
    /// Avalanche frequency
    pub avalanche_frequency: f64,
    
    /// System stability
    pub system_stability: f64,
    
    /// Measurement timestamp
    pub timestamp: Instant,
}

/// Criticality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalityIndicators {
    /// Power law coefficient
    pub power_law_coefficient: f64,
    
    /// Fractal dimension
    pub fractal_dimension: f64,
    
    /// Scaling exponent
    pub scaling_exponent: f64,
    
    /// Correlation length
    pub correlation_length: f64,
    
    /// Critical slowing down
    pub critical_slowing_down: f64,
}

/// Black swan event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanEvent {
    /// Event identifier
    pub id: String,
    
    /// Event type
    pub event_type: BlackSwanType,
    
    /// Event magnitude
    pub magnitude: f64,
    
    /// Event probability
    pub probability: f64,
    
    /// Event impact
    pub impact: f64,
    
    /// Event duration
    pub duration: Duration,
    
    /// Event timestamp
    pub timestamp: Instant,
}

/// Black swan event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlackSwanType {
    /// Market crash
    MarketCrash,
    
    /// Flash crash
    FlashCrash,
    
    /// Liquidity crisis
    LiquidityCrisis,
    
    /// Correlation breakdown
    CorrelationBreakdown,
    
    /// External shock
    ExternalShock,
    
    /// Systemic risk
    SystemicRisk,
}

/// Tail risk calculator
#[derive(Debug, Clone)]
pub struct TailRiskCalculator {
    /// VaR calculator
    var_calculator: VaRCalculator,
    
    /// Expected shortfall calculator
    es_calculator: ESCalculator,
    
    /// Extreme value theory model
    evt_model: EVTModel,
    
    /// Tail dependence estimator
    tail_dependence_estimator: TailDependenceEstimator,
    
    /// Configuration
    config: TailRiskConfig,
}

/// Extreme event detector
#[derive(Debug, Clone)]
pub struct ExtremeEventDetector {
    /// Event detection models
    detection_models: HashMap<String, DetectionModel>,
    
    /// Event classifiers
    event_classifiers: HashMap<String, EventClassifier>,
    
    /// Event history
    event_history: VecDeque<ExtremeEvent>,
    
    /// Configuration
    config: ExtremeEventConfig,
}

/// Fat tail analyzer
#[derive(Debug, Clone)]
pub struct FatTailAnalyzer {
    /// Distribution fitter
    distribution_fitter: DistributionFitter,
    
    /// Tail index estimator
    tail_index_estimator: TailIndexEstimator,
    
    /// Kurtosis calculator
    kurtosis_calculator: KurtosisCalculator,
    
    /// Configuration
    config: FatTailConfig,
}

/// Antifragility metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilityMetrics {
    /// Antifragility index
    pub antifragility_index: f64,
    
    /// Stress benefit ratio
    pub stress_benefit_ratio: f64,
    
    /// Volatility benefit
    pub volatility_benefit: f64,
    
    /// Disorder adaptation
    pub disorder_adaptation: f64,
    
    /// Resilience factor
    pub resilience_factor: f64,
}

/// Implementation of market regime detector
impl MarketRegimeDetector {
    /// Create new market regime detector
    pub fn new(config: RegimeDetectionConfig) -> Self {
        let soc_calculator = SOCIndexCalculator::new(config.soc_config.clone());
        let black_swan_assessor = BlackSwanRiskAssessor::new(config.black_swan_config.clone());
        let regime_classifier = RegimeClassifier::new(config.classifier_config.clone());
        let volatility_tracker = VolatilityRegimeTracker::new(config.volatility_config.clone());
        let correlation_tracker = CorrelationRegimeTracker::new(config.correlation_config.clone());
        let multi_scale_analyzer = MultiScaleRegimeAnalyzer::new(config.multi_scale_config.clone());
        let regime_history = VecDeque::with_capacity(config.history_size);
        
        Self {
            soc_calculator,
            black_swan_assessor,
            regime_classifier,
            volatility_tracker,
            correlation_tracker,
            multi_scale_analyzer,
            regime_history,
            config,
        }
    }
    
    /// Detect current market regime
    pub async fn detect_regime(&mut self, market_data: &MarketData) -> Result<MarketRegime> {
        // Calculate SOC index
        let soc_index = self.soc_calculator.calculate_soc_index(market_data).await?;
        
        // Assess black swan risk
        let black_swan_risk = self.black_swan_assessor.assess_risk(market_data).await?;
        
        // Classify regime
        let regime_classification = self.regime_classifier.classify_regime(market_data).await?;
        
        // Track volatility regime
        let volatility_regime = self.volatility_tracker.track_regime(market_data).await?;
        
        // Track correlation regime
        let correlation_regime = self.correlation_tracker.track_regime(market_data).await?;
        
        // Analyze multi-scale coherence
        let multi_scale_coherence = self.multi_scale_analyzer.analyze_coherence(market_data).await?;
        
        // Combine all factors to determine primary regime
        let primary_regime = self.determine_primary_regime(
            &regime_classification,
            &volatility_regime,
            &correlation_regime,
            soc_index,
            black_swan_risk,
        ).await?;
        
        // Create regime state
        let regime_state = RegimeState {
            primary_regime,
            secondary_regime: None,
            confidence: regime_classification.confidence,
            stability: self.calculate_regime_stability(&regime_classification),
            transition_probability: self.calculate_transition_probability(&regime_classification),
            soc_index,
            black_swan_risk,
            volatility_regime,
            correlation_regime,
            multi_scale_coherence,
            timestamp: Instant::now(),
        };
        
        // Update history
        self.regime_history.push_back(regime_state);
        if self.regime_history.len() > self.config.history_size {
            self.regime_history.pop_front();
        }
        
        Ok(primary_regime)
    }
    
    /// Get current regime state
    pub fn get_current_regime_state(&self) -> Option<&RegimeState> {
        self.regime_history.back()
    }
    
    /// Get SOC index
    pub fn get_soc_index(&self) -> f64 {
        self.soc_calculator.get_current_soc_index()
    }
    
    /// Get black swan risk
    pub fn get_black_swan_risk(&self) -> f64 {
        self.black_swan_assessor.get_current_risk()
    }
    
    /// Get volatility regime
    pub fn get_volatility_regime(&self) -> VolatilityRegime {
        self.volatility_tracker.get_current_regime()
    }
    
    /// Get correlation regime
    pub fn get_correlation_regime(&self) -> CorrelationRegime {
        self.correlation_tracker.get_current_regime()
    }
    
    /// Determine primary regime from all factors
    async fn determine_primary_regime(
        &self,
        classification: &RegimeClassification,
        volatility_regime: &VolatilityRegime,
        correlation_regime: &CorrelationRegime,
        soc_index: f64,
        black_swan_risk: f64,
    ) -> Result<MarketRegime> {
        // Weight factors based on current conditions
        let mut regime_scores = HashMap::new();
        
        // Base classification score
        for (regime, probability) in &classification.regime_probabilities {
            regime_scores.insert(*regime, *probability * 0.4);
        }
        
        // Volatility regime influence
        match volatility_regime {
            VolatilityRegime::Low => {
                *regime_scores.entry(MarketRegime::LowVolatility).or_insert(0.0) += 0.2;
            }
            VolatilityRegime::High | VolatilityRegime::Extreme => {
                *regime_scores.entry(MarketRegime::HighVolatility).or_insert(0.0) += 0.2;
            }
            VolatilityRegime::Clustering => {
                *regime_scores.entry(MarketRegime::Crisis).or_insert(0.0) += 0.1;
            }
            _ => {}
        }
        
        // Correlation regime influence
        match correlation_regime {
            CorrelationRegime::Breakdown => {
                *regime_scores.entry(MarketRegime::Crisis).or_insert(0.0) += 0.15;
            }
            CorrelationRegime::FlightToQuality => {
                *regime_scores.entry(MarketRegime::Bear).or_insert(0.0) += 0.15;
            }
            _ => {}
        }
        
        // SOC index influence
        if soc_index > 0.8 {
            *regime_scores.entry(MarketRegime::Crisis).or_insert(0.0) += 0.1;
        }
        
        // Black swan risk influence
        if black_swan_risk > 0.7 {
            *regime_scores.entry(MarketRegime::Crisis).or_insert(0.0) += 0.15;
        }
        
        // Find regime with highest score
        let primary_regime = regime_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(regime, _)| *regime)
            .unwrap_or(MarketRegime::Sideways);
        
        Ok(primary_regime)
    }
    
    /// Calculate regime stability
    fn calculate_regime_stability(&self, classification: &RegimeClassification) -> f64 {
        // Stability based on classification confidence and history
        let confidence_stability = classification.confidence;
        let history_stability = self.calculate_history_stability();
        
        (confidence_stability * 0.6 + history_stability * 0.4).clamp(0.0, 1.0)
    }
    
    /// Calculate transition probability
    fn calculate_transition_probability(&self, classification: &RegimeClassification) -> f64 {
        // Transition probability based on classification uncertainty
        1.0 - classification.confidence
    }
    
    /// Calculate stability from history
    fn calculate_history_stability(&self) -> f64 {
        if self.regime_history.len() < 10 {
            return 0.5;
        }
        
        let recent_regimes: Vec<_> = self.regime_history.iter()
            .rev()
            .take(10)
            .map(|s| s.primary_regime)
            .collect();
        
        let unique_regimes: std::collections::HashSet<_> = recent_regimes.iter().collect();
        let stability = 1.0 - (unique_regimes.len() as f64 / recent_regimes.len() as f64);
        
        stability.clamp(0.0, 1.0)
    }
}

/// Implementation of SOC index calculator
impl SOCIndexCalculator {
    /// Create new SOC calculator
    pub fn new(config: SOCConfig) -> Self {
        Self {
            power_law_exponent: 0.0,
            avalanche_distribution: VecDeque::with_capacity(config.history_size),
            criticality_indicators: CriticalityIndicators::default(),
            soc_history: VecDeque::with_capacity(config.history_size),
            config,
        }
    }
    
    /// Calculate SOC index
    pub async fn calculate_soc_index(&mut self, market_data: &MarketData) -> Result<f64> {
        // Calculate power law exponent
        self.power_law_exponent = self.calculate_power_law_exponent(market_data)?;
        
        // Detect avalanches
        let avalanches = self.detect_avalanches(market_data)?;
        
        // Update avalanche distribution
        for avalanche in avalanches {
            self.avalanche_distribution.push_back(avalanche);
        }
        
        if self.avalanche_distribution.len() > self.config.history_size {
            self.avalanche_distribution.pop_front();
        }
        
        // Calculate criticality indicators
        self.criticality_indicators = self.calculate_criticality_indicators(market_data)?;
        
        // Calculate SOC index
        let soc_index = self.calculate_soc_from_indicators()?;
        
        // Record measurement
        let measurement = SOCMeasurement {
            index: soc_index,
            power_law_exponent: self.power_law_exponent,
            criticality_level: self.criticality_indicators.power_law_coefficient,
            avalanche_frequency: self.avalanche_distribution.len() as f64,
            system_stability: 1.0 - soc_index,
            timestamp: Instant::now(),
        };
        
        self.soc_history.push_back(measurement);
        if self.soc_history.len() > self.config.history_size {
            self.soc_history.pop_front();
        }
        
        Ok(soc_index)
    }
    
    /// Get current SOC index
    pub fn get_current_soc_index(&self) -> f64 {
        self.soc_history.back().map(|m| m.index).unwrap_or(0.0)
    }
    
    /// Calculate power law exponent
    fn calculate_power_law_exponent(&self, market_data: &MarketData) -> Result<f64> {
        if market_data.prices.len() < 100 {
            return Ok(0.0);
        }
        
        // Calculate returns
        let returns: Vec<f64> = market_data.prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        // Calculate absolute returns
        let abs_returns: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
        
        // Sort returns for power law fitting
        let mut sorted_returns = abs_returns.clone();
        sorted_returns.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        // Fit power law using rank-frequency method
        let n = sorted_returns.len() as f64;
        let mut sum_log_rank = 0.0;
        let mut sum_log_freq = 0.0;
        let mut sum_log_rank_sq = 0.0;
        let mut sum_log_rank_freq = 0.0;
        
        for (i, &value) in sorted_returns.iter().enumerate() {
            if value > 0.0 {
                let rank = (i + 1) as f64;
                let log_rank = rank.ln();
                let log_freq = value.ln();
                
                sum_log_rank += log_rank;
                sum_log_freq += log_freq;
                sum_log_rank_sq += log_rank * log_rank;
                sum_log_rank_freq += log_rank * log_freq;
            }
        }
        
        let denominator = n * sum_log_rank_sq - sum_log_rank * sum_log_rank;
        if denominator.abs() < 1e-10 {
            return Ok(0.0);
        }
        
        let exponent = (n * sum_log_rank_freq - sum_log_rank * sum_log_freq) / denominator;
        
        Ok(exponent.abs().clamp(0.0, 3.0))
    }
    
    /// Detect avalanches in market data
    fn detect_avalanches(&self, market_data: &MarketData) -> Result<Vec<f64>> {
        let mut avalanches = Vec::new();
        
        if market_data.prices.len() < 10 {
            return Ok(avalanches);
        }
        
        // Calculate returns
        let returns: Vec<f64> = market_data.prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        // Calculate rolling volatility
        let volatility = self.calculate_rolling_volatility(&returns, 20);
        
        // Detect avalanches as periods of high volatility
        let threshold = volatility * 2.0;
        let mut in_avalanche = false;
        let mut avalanche_magnitude = 0.0;
        
        for (i, &ret) in returns.iter().enumerate() {
            if ret.abs() > threshold {
                if !in_avalanche {
                    in_avalanche = true;
                    avalanche_magnitude = ret.abs();
                } else {
                    avalanche_magnitude += ret.abs();
                }
            } else if in_avalanche {
                avalanches.push(avalanche_magnitude);
                in_avalanche = false;
                avalanche_magnitude = 0.0;
            }
        }
        
        Ok(avalanches)
    }
    
    /// Calculate rolling volatility
    fn calculate_rolling_volatility(&self, returns: &[f64], window: usize) -> f64 {
        if returns.len() < window {
            return 0.0;
        }
        
        let recent_returns = &returns[returns.len() - window..];
        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (recent_returns.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    /// Calculate criticality indicators
    fn calculate_criticality_indicators(&self, market_data: &MarketData) -> Result<CriticalityIndicators> {
        Ok(CriticalityIndicators {
            power_law_coefficient: self.power_law_exponent,
            fractal_dimension: self.calculate_fractal_dimension(market_data)?,
            scaling_exponent: self.calculate_scaling_exponent(market_data)?,
            correlation_length: self.calculate_correlation_length(market_data)?,
            critical_slowing_down: self.calculate_critical_slowing_down(market_data)?,
        })
    }
    
    /// Calculate fractal dimension
    fn calculate_fractal_dimension(&self, market_data: &MarketData) -> Result<f64> {
        // Simplified fractal dimension calculation using box counting
        if market_data.prices.len() < 100 {
            return Ok(1.5);
        }
        
        let prices = &market_data.prices;
        let n = prices.len();
        let mut scales = Vec::new();
        let mut box_counts = Vec::new();
        
        for scale in (1..=10).map(|i| i * 10) {
            if scale < n {
                let boxes = self.count_boxes(prices, scale);
                scales.push((scale as f64).ln());
                box_counts.push((boxes as f64).ln());
            }
        }
        
        // Linear regression to find fractal dimension
        if scales.len() < 2 {
            return Ok(1.5);
        }
        
        let n_points = scales.len() as f64;
        let sum_x = scales.iter().sum::<f64>();
        let sum_y = box_counts.iter().sum::<f64>();
        let sum_xy = scales.iter().zip(box_counts.iter()).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = scales.iter().map(|x| x * x).sum::<f64>();
        
        let denominator = n_points * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return Ok(1.5);
        }
        
        let slope = (n_points * sum_xy - sum_x * sum_y) / denominator;
        let fractal_dimension = -slope;
        
        Ok(fractal_dimension.clamp(1.0, 2.0))
    }
    
    /// Count boxes for fractal dimension calculation
    fn count_boxes(&self, prices: &[f64], scale: usize) -> usize {
        let mut boxes = std::collections::HashSet::new();
        
        for i in 0..prices.len() {
            let box_x = i / scale;
            let box_y = (prices[i] * 1000.0) as i32 / scale as i32;
            boxes.insert((box_x, box_y));
        }
        
        boxes.len()
    }
    
    /// Calculate scaling exponent
    fn calculate_scaling_exponent(&self, market_data: &MarketData) -> Result<f64> {
        // Simplified scaling exponent calculation
        Ok(self.power_law_exponent * 0.5)
    }
    
    /// Calculate correlation length
    fn calculate_correlation_length(&self, market_data: &MarketData) -> Result<f64> {
        // Simplified correlation length calculation
        let correlation = market_data.calculate_correlation();
        Ok(correlation * 100.0)
    }
    
    /// Calculate critical slowing down
    fn calculate_critical_slowing_down(&self, market_data: &MarketData) -> Result<f64> {
        // Simplified critical slowing down calculation
        let volatility = market_data.calculate_volatility();
        Ok(volatility * 10.0)
    }
    
    /// Calculate SOC index from indicators
    fn calculate_soc_from_indicators(&self) -> Result<f64> {
        let indicators = &self.criticality_indicators;
        
        // Combine indicators to create SOC index
        let power_law_factor = if indicators.power_law_coefficient > 1.0 {
            (indicators.power_law_coefficient - 1.0) / 2.0
        } else {
            0.0
        };
        
        let fractal_factor = (indicators.fractal_dimension - 1.0).abs();
        let scaling_factor = indicators.scaling_exponent;
        let correlation_factor = indicators.correlation_length / 100.0;
        let slowing_factor = indicators.critical_slowing_down / 10.0;
        
        let soc_index = (power_law_factor * 0.3 + 
                        fractal_factor * 0.2 + 
                        scaling_factor * 0.2 + 
                        correlation_factor * 0.15 + 
                        slowing_factor * 0.15).clamp(0.0, 1.0);
        
        Ok(soc_index)
    }
}

/// Implementation of black swan risk assessor
impl BlackSwanRiskAssessor {
    /// Create new black swan assessor
    pub fn new(config: BlackSwanConfig) -> Self {
        Self {
            tail_risk_calculator: TailRiskCalculator::new(config.tail_risk_config.clone()),
            extreme_event_detector: ExtremeEventDetector::new(config.extreme_event_config.clone()),
            fat_tail_analyzer: FatTailAnalyzer::new(config.fat_tail_config.clone()),
            antifragility_metrics: AntifragilityMetrics::default(),
            black_swan_history: VecDeque::with_capacity(config.history_size),
            config,
        }
    }
    
    /// Assess black swan risk
    pub async fn assess_risk(&mut self, market_data: &MarketData) -> Result<f64> {
        // Calculate tail risk
        let tail_risk = self.tail_risk_calculator.calculate_tail_risk(market_data).await?;
        
        // Detect extreme events
        let extreme_events = self.extreme_event_detector.detect_events(market_data).await?;
        
        // Analyze fat tails
        let fat_tail_metrics = self.fat_tail_analyzer.analyze_fat_tails(market_data).await?;
        
        // Update antifragility metrics
        self.antifragility_metrics = self.calculate_antifragility_metrics(market_data).await?;
        
        // Combine factors for overall black swan risk
        let event_risk = extreme_events.len() as f64 * 0.1;
        let tail_risk_factor = tail_risk * 0.4;
        let fat_tail_factor = fat_tail_metrics.kurtosis_excess * 0.3;
        let antifragility_factor = (1.0 - self.antifragility_metrics.antifragility_index) * 0.2;
        
        let black_swan_risk = (event_risk + tail_risk_factor + fat_tail_factor + antifragility_factor).clamp(0.0, 1.0);
        
        // Record black swan events
        for event in extreme_events {
            if event.magnitude > self.config.event_threshold {
                let black_swan_event = BlackSwanEvent {
                    id: format!("bs-{}", uuid::Uuid::new_v4()),
                    event_type: self.classify_black_swan_type(&event),
                    magnitude: event.magnitude,
                    probability: event.probability,
                    impact: event.impact,
                    duration: event.duration,
                    timestamp: Instant::now(),
                };
                
                self.black_swan_history.push_back(black_swan_event);
            }
        }
        
        if self.black_swan_history.len() > self.config.history_size {
            self.black_swan_history.pop_front();
        }
        
        Ok(black_swan_risk)
    }
    
    /// Get current black swan risk
    pub fn get_current_risk(&self) -> f64 {
        // Calculate risk based on recent events
        let recent_events = self.black_swan_history.iter()
            .filter(|e| e.timestamp.elapsed() < Duration::from_secs(3600))
            .count();
        
        (recent_events as f64 * 0.2).clamp(0.0, 1.0)
    }
    
    /// Classify black swan event type
    fn classify_black_swan_type(&self, event: &ExtremeEvent) -> BlackSwanType {
        match event.event_type.as_str() {
            "market_crash" => BlackSwanType::MarketCrash,
            "flash_crash" => BlackSwanType::FlashCrash,
            "liquidity_crisis" => BlackSwanType::LiquidityCrisis,
            "correlation_breakdown" => BlackSwanType::CorrelationBreakdown,
            "external_shock" => BlackSwanType::ExternalShock,
            _ => BlackSwanType::SystemicRisk,
        }
    }
    
    /// Calculate antifragility metrics
    async fn calculate_antifragility_metrics(&self, market_data: &MarketData) -> Result<AntifragilityMetrics> {
        let volatility = market_data.calculate_volatility();
        let returns = self.calculate_returns(market_data);
        
        // Calculate stress benefit ratio
        let stress_benefit_ratio = self.calculate_stress_benefit_ratio(&returns, volatility);
        
        // Calculate volatility benefit
        let volatility_benefit = self.calculate_volatility_benefit(&returns, volatility);
        
        // Calculate disorder adaptation
        let disorder_adaptation = self.calculate_disorder_adaptation(&returns);
        
        // Calculate resilience factor
        let resilience_factor = self.calculate_resilience_factor(market_data);
        
        // Calculate overall antifragility index
        let antifragility_index = (stress_benefit_ratio * 0.3 + 
                                 volatility_benefit * 0.3 + 
                                 disorder_adaptation * 0.2 + 
                                 resilience_factor * 0.2).clamp(0.0, 1.0);
        
        Ok(AntifragilityMetrics {
            antifragility_index,
            stress_benefit_ratio,
            volatility_benefit,
            disorder_adaptation,
            resilience_factor,
        })
    }
    
    /// Calculate returns from market data
    fn calculate_returns(&self, market_data: &MarketData) -> Vec<f64> {
        market_data.prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
    
    /// Calculate stress benefit ratio
    fn calculate_stress_benefit_ratio(&self, returns: &[f64], volatility: f64) -> f64 {
        if returns.is_empty() || volatility == 0.0 {
            return 0.0;
        }
        
        // Calculate performance during high volatility periods
        let threshold = volatility * 2.0;
        let high_vol_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r.abs() > threshold)
            .cloned()
            .collect();
        
        if high_vol_returns.is_empty() {
            return 0.0;
        }
        
        let high_vol_mean = high_vol_returns.iter().sum::<f64>() / high_vol_returns.len() as f64;
        let overall_mean = returns.iter().sum::<f64>() / returns.len() as f64;
        
        if overall_mean != 0.0 {
            (high_vol_mean / overall_mean).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate volatility benefit
    fn calculate_volatility_benefit(&self, returns: &[f64], volatility: f64) -> f64 {
        if returns.is_empty() || volatility == 0.0 {
            return 0.0;
        }
        
        // Calculate convexity of returns
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let convexity = returns.iter()
            .map(|&r| (r - mean_return).powi(3))
            .sum::<f64>() / returns.len() as f64;
        
        (convexity / volatility.powi(3)).clamp(-1.0, 1.0)
    }
    
    /// Calculate disorder adaptation
    fn calculate_disorder_adaptation(&self, returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return 0.0;
        }
        
        // Calculate adaptation to changing conditions
        let mut adaptation_scores = Vec::new();
        
        for window in returns.windows(20) {
            let volatility = self.calculate_window_volatility(window);
            let mean_return = window.iter().sum::<f64>() / window.len() as f64;
            
            if volatility > 0.0 {
                adaptation_scores.push(mean_return / volatility);
            }
        }
        
        if adaptation_scores.is_empty() {
            return 0.0;
        }
        
        let mean_adaptation = adaptation_scores.iter().sum::<f64>() / adaptation_scores.len() as f64;
        mean_adaptation.clamp(-1.0, 1.0)
    }
    
    /// Calculate window volatility
    fn calculate_window_volatility(&self, window: &[f64]) -> f64 {
        if window.len() < 2 {
            return 0.0;
        }
        
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (window.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    /// Calculate resilience factor
    fn calculate_resilience_factor(&self, market_data: &MarketData) -> f64 {
        // Simplified resilience calculation
        let liquidity = market_data.calculate_liquidity();
        let volatility = market_data.calculate_volatility();
        
        (liquidity * 0.6 + (1.0 - volatility) * 0.4).clamp(0.0, 1.0)
    }
}

// Configuration structures and other type definitions...

/// Configuration structures
#[derive(Debug, Clone)]
pub struct RegimeDetectionConfig {
    pub soc_config: SOCConfig,
    pub black_swan_config: BlackSwanConfig,
    pub classifier_config: ClassifierConfig,
    pub volatility_config: VolatilityConfig,
    pub correlation_config: CorrelationConfig,
    pub multi_scale_config: MultiScaleConfig,
    pub history_size: usize,
}

#[derive(Debug, Clone)]
pub struct SOCConfig {
    pub history_size: usize,
    pub avalanche_threshold: f64,
    pub power_law_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct BlackSwanConfig {
    pub tail_risk_config: TailRiskConfig,
    pub extreme_event_config: ExtremeEventConfig,
    pub fat_tail_config: FatTailConfig,
    pub event_threshold: f64,
    pub history_size: usize,
}

#[derive(Debug, Clone)]
pub struct TailRiskConfig {
    pub confidence_level: f64,
    pub estimation_method: String,
}

#[derive(Debug, Clone)]
pub struct ExtremeEventConfig {
    pub detection_threshold: f64,
    pub classification_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct FatTailConfig {
    pub distribution_type: String,
    pub estimation_method: String,
}

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    pub model_type: String,
    pub feature_selection: Vec<String>,
    pub training_window: usize,
}

#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    pub model_type: String,
    pub forecast_horizon: usize,
    pub clustering_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    pub estimation_method: String,
    pub window_size: usize,
    pub breakdown_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct MultiScaleConfig {
    pub scales: Vec<ScaleLevel>,
    pub coherence_threshold: f64,
}

// Additional type definitions and implementations would continue...
// This provides a comprehensive foundation for regime detection with SOC and black swan analysis

/// Placeholder types for compilation
#[derive(Debug, Clone)]
pub struct RegimeClassification {
    pub regime_probabilities: HashMap<MarketRegime, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ExtremeEvent {
    pub event_type: String,
    pub magnitude: f64,
    pub probability: f64,
    pub impact: f64,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct FatTailMetrics {
    pub kurtosis_excess: f64,
    pub tail_index: f64,
}

/// Additional placeholder implementations
impl Default for CriticalityIndicators {
    fn default() -> Self {
        Self {
            power_law_coefficient: 0.0,
            fractal_dimension: 1.5,
            scaling_exponent: 0.0,
            correlation_length: 50.0,
            critical_slowing_down: 0.0,
        }
    }
}

impl Default for AntifragilityMetrics {
    fn default() -> Self {
        Self {
            antifragility_index: 0.5,
            stress_benefit_ratio: 0.0,
            volatility_benefit: 0.0,
            disorder_adaptation: 0.0,
            resilience_factor: 0.5,
        }
    }
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            soc_config: SOCConfig::default(),
            black_swan_config: BlackSwanConfig::default(),
            classifier_config: ClassifierConfig::default(),
            volatility_config: VolatilityConfig::default(),
            correlation_config: CorrelationConfig::default(),
            multi_scale_config: MultiScaleConfig::default(),
            history_size: 1000,
        }
    }
}

impl Default for SOCConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            avalanche_threshold: 0.02,
            power_law_threshold: 1.5,
        }
    }
}

impl Default for BlackSwanConfig {
    fn default() -> Self {
        Self {
            tail_risk_config: TailRiskConfig::default(),
            extreme_event_config: ExtremeEventConfig::default(),
            fat_tail_config: FatTailConfig::default(),
            event_threshold: 0.05,
            history_size: 1000,
        }
    }
}

impl Default for TailRiskConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            estimation_method: "historical".to_string(),
        }
    }
}

impl Default for ExtremeEventConfig {
    fn default() -> Self {
        Self {
            detection_threshold: 0.03,
            classification_threshold: 0.05,
        }
    }
}

impl Default for FatTailConfig {
    fn default() -> Self {
        Self {
            distribution_type: "student_t".to_string(),
            estimation_method: "mle".to_string(),
        }
    }
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            model_type: "random_forest".to_string(),
            feature_selection: vec!["volatility".to_string(), "correlation".to_string()],
            training_window: 1000,
        }
    }
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            model_type: "garch".to_string(),
            forecast_horizon: 10,
            clustering_threshold: 0.02,
        }
    }
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            estimation_method: "pearson".to_string(),
            window_size: 50,
            breakdown_threshold: 0.3,
        }
    }
}

impl Default for MultiScaleConfig {
    fn default() -> Self {
        Self {
            scales: vec![ScaleLevel::Micro, ScaleLevel::Meso, ScaleLevel::Macro],
            coherence_threshold: 0.6,
        }
    }
}

// Additional stub implementations for compilation
impl TailRiskCalculator {
    pub fn new(config: TailRiskConfig) -> Self {
        Self {
            var_calculator: VaRCalculator::new(),
            es_calculator: ESCalculator::new(),
            evt_model: EVTModel::new(),
            tail_dependence_estimator: TailDependenceEstimator::new(),
            config,
        }
    }
    
    pub async fn calculate_tail_risk(&self, market_data: &MarketData) -> Result<f64> {
        Ok(market_data.calculate_volatility() * 0.5)
    }
}

// Additional stub types for compilation
#[derive(Debug, Clone)]
pub struct VaRCalculator;
#[derive(Debug, Clone)]
pub struct ESCalculator;
#[derive(Debug, Clone)]
pub struct EVTModel;
#[derive(Debug, Clone)]
pub struct TailDependenceEstimator;
#[derive(Debug, Clone)]
pub struct DetectionModel;
#[derive(Debug, Clone)]
pub struct EventClassifier;
#[derive(Debug, Clone)]
pub struct DistributionFitter;
#[derive(Debug, Clone)]
pub struct TailIndexEstimator;
#[derive(Debug, Clone)]
pub struct KurtosisCalculator;
#[derive(Debug, Clone)]
pub struct ClassificationModel;
#[derive(Debug, Clone)]
pub struct FeatureExtractor;
#[derive(Debug, Clone)]
pub struct VolatilityClusteringDetector;
#[derive(Debug, Clone)]
pub struct GARCHModel;
#[derive(Debug, Clone)]
pub struct VolatilityForecaster;
#[derive(Debug, Clone)]
pub struct VolatilityRegimeState;
#[derive(Debug, Clone)]
pub struct CorrelationMatrix;
#[derive(Debug, Clone)]
pub struct CorrelationBreakdownDetector;
#[derive(Debug, Clone)]
pub struct DynamicCorrelationModel;
#[derive(Debug, Clone)]
pub struct CorrelationRegimeState;
#[derive(Debug, Clone)]
pub struct ScaleRegimeAnalyzer;
#[derive(Debug, Clone)]
pub struct InteractionStrength;
#[derive(Debug, Clone)]
pub struct CoherenceCalculator;

impl VaRCalculator {
    pub fn new() -> Self { Self }
}

impl ESCalculator {
    pub fn new() -> Self { Self }
}

impl EVTModel {
    pub fn new() -> Self { Self }
}

impl TailDependenceEstimator {
    pub fn new() -> Self { Self }
}

impl ExtremeEventDetector {
    pub fn new(config: ExtremeEventConfig) -> Self {
        Self {
            detection_models: HashMap::new(),
            event_classifiers: HashMap::new(),
            event_history: VecDeque::new(),
            config,
        }
    }
    
    pub async fn detect_events(&self, market_data: &MarketData) -> Result<Vec<ExtremeEvent>> {
        Ok(vec![])
    }
}

impl FatTailAnalyzer {
    pub fn new(config: FatTailConfig) -> Self {
        Self {
            distribution_fitter: DistributionFitter,
            tail_index_estimator: TailIndexEstimator,
            kurtosis_calculator: KurtosisCalculator,
            config,
        }
    }
    
    pub async fn analyze_fat_tails(&self, market_data: &MarketData) -> Result<FatTailMetrics> {
        Ok(FatTailMetrics {
            kurtosis_excess: 0.0,
            tail_index: 0.0,
        })
    }
}

impl RegimeClassifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self {
            models: HashMap::new(),
            feature_extractors: HashMap::new(),
            classification_rules: vec![],
            regime_probabilities: HashMap::new(),
            config,
        }
    }
    
    pub async fn classify_regime(&self, market_data: &MarketData) -> Result<RegimeClassification> {
        let mut probabilities = HashMap::new();
        probabilities.insert(MarketRegime::Sideways, 0.4);
        probabilities.insert(MarketRegime::Bull, 0.3);
        probabilities.insert(MarketRegime::Bear, 0.3);
        
        Ok(RegimeClassification {
            regime_probabilities: probabilities,
            confidence: 0.7,
        })
    }
}

impl VolatilityRegimeTracker {
    pub fn new(config: VolatilityConfig) -> Self {
        Self {
            current_regime: VolatilityRegime::Moderate,
            clustering_detector: VolatilityClusteringDetector,
            garch_model: GARCHModel,
            volatility_forecaster: VolatilityForecaster,
            regime_history: VecDeque::new(),
            config,
        }
    }
    
    pub async fn track_regime(&self, market_data: &MarketData) -> Result<VolatilityRegime> {
        let volatility = market_data.calculate_volatility();
        
        if volatility < 0.01 {
            Ok(VolatilityRegime::Low)
        } else if volatility < 0.03 {
            Ok(VolatilityRegime::Moderate)
        } else if volatility < 0.05 {
            Ok(VolatilityRegime::High)
        } else {
            Ok(VolatilityRegime::Extreme)
        }
    }
    
    pub fn get_current_regime(&self) -> VolatilityRegime {
        self.current_regime
    }
}

impl CorrelationRegimeTracker {
    pub fn new(config: CorrelationConfig) -> Self {
        Self {
            current_regime: CorrelationRegime::Moderate,
            correlation_matrix: CorrelationMatrix,
            breakdown_detector: CorrelationBreakdownDetector,
            dynamic_correlation_model: DynamicCorrelationModel,
            regime_history: VecDeque::new(),
            config,
        }
    }
    
    pub async fn track_regime(&self, market_data: &MarketData) -> Result<CorrelationRegime> {
        let correlation = market_data.calculate_correlation();
        
        if correlation < 0.3 {
            Ok(CorrelationRegime::Low)
        } else if correlation < 0.7 {
            Ok(CorrelationRegime::Moderate)
        } else {
            Ok(CorrelationRegime::High)
        }
    }
    
    pub fn get_current_regime(&self) -> CorrelationRegime {
        self.current_regime
    }
}

impl MultiScaleRegimeAnalyzer {
    pub fn new(config: MultiScaleConfig) -> Self {
        Self {
            scale_analyzers: HashMap::new(),
            cross_scale_interactions: HashMap::new(),
            coherence_calculator: CoherenceCalculator,
            config,
        }
    }
    
    pub async fn analyze_coherence(&self, market_data: &MarketData) -> Result<f64> {
        Ok(0.7)
    }
}

// Add the missing uuid dependency for compilation
use uuid::Uuid;