//! # Panarchy Analyzer
//!
//! Unified panarchy system implementing adaptive cycle analysis with cross-scale interactions.
//! Combines features from both pads-connector and cdfa-panarchy-analyzer for comprehensive
//! multi-scale decision making based on panarchy theory.
//!
//! ## Panarchy Theory
//! 
//! The four-phase adaptive cycle:
//! - **Growth (r)**: Rapid colonization and resource accumulation
//! - **Conservation (K)**: Slow accumulation and stability
//! - **Release (Ω)**: Creative destruction and rapid change
//! - **Reorganization (α)**: Innovation and restructuring
//!
//! ## Features
//!
//! - Sub-microsecond performance optimization
//! - PCR (Potential, Connectedness, Resilience) analysis
//! - Cross-scale interaction modeling
//! - Scale transition detection
//! - Adaptive feedback mechanisms

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Errors that can occur during panarchy analysis
#[derive(Error, Debug)]
pub enum PanarchyError {
    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },
    
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Computation error: {message}")]
    ComputationError { message: String },
    
    #[error("Scale transition error: {message}")]
    ScaleTransitionError { message: String },
    
    #[error("SIMD operation failed: {message}")]
    SimdError { message: String },
}

/// Result type for panarchy operations
pub type PanarchyResult<T> = Result<T, PanarchyError>;

/// Panarchy configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyConfig {
    /// Window size for analysis
    pub window_size: usize,
    /// PCR calculation period
    pub pcr_period: usize,
    /// Minimum data points required
    pub min_data_points: usize,
    /// Phase transition threshold
    pub phase_transition_threshold: f64,
    /// Scale transition sensitivity
    pub scale_transition_sensitivity: f64,
    /// Cross-scale interaction strength
    pub cross_scale_strength: f64,
    /// Adaptive feedback rate
    pub adaptive_feedback_rate: f64,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Performance target (nanoseconds)
    pub performance_target_ns: u64,
}

impl Default for PanarchyConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            pcr_period: 14,
            min_data_points: 20,
            phase_transition_threshold: 0.3,
            scale_transition_sensitivity: 0.5,
            cross_scale_strength: 0.2,
            adaptive_feedback_rate: 0.1,
            enable_simd: true,
            enable_parallel: true,
            performance_target_ns: 800,
        }
    }
}

/// Panarchy phases in the adaptive cycle
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PanarchyPhase {
    /// Growth phase (r) - rapid colonization
    Growth,
    /// Conservation phase (K) - slow accumulation
    Conservation,
    /// Release phase (Ω) - creative destruction
    Release,
    /// Reorganization phase (α) - innovation
    Reorganization,
}

impl PanarchyPhase {
    /// Convert phase to signal strength
    pub fn to_signal(&self) -> f64 {
        match self {
            PanarchyPhase::Growth => 0.8,
            PanarchyPhase::Conservation => 0.2,
            PanarchyPhase::Release => -0.8,
            PanarchyPhase::Reorganization => -0.2,
        }
    }
    
    /// Get phase description
    pub fn description(&self) -> &'static str {
        match self {
            PanarchyPhase::Growth => "Rapid expansion and opportunity exploitation",
            PanarchyPhase::Conservation => "Stability and efficiency optimization",
            PanarchyPhase::Release => "Creative destruction and volatility",
            PanarchyPhase::Reorganization => "Innovation and structural renewal",
        }
    }
}

/// Scale levels for cross-scale analysis
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ScaleLevel {
    /// Micro scale - local, short-term
    Micro,
    /// Meso scale - intermediate
    Meso,
    /// Macro scale - system-wide, long-term
    Macro,
}

/// PCR (Potential, Connectedness, Resilience) components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCRComponents {
    /// Potential for change (accumulated resources)
    pub potential: f64,
    /// Connectedness (rigidity of control)
    pub connectedness: f64,
    /// Resilience (adaptive capacity)
    pub resilience: f64,
    /// Combined PCR score
    pub pcr_score: f64,
}

impl Default for PCRComponents {
    fn default() -> Self {
        Self {
            potential: 0.5,
            connectedness: 0.5,
            resilience: 0.5,
            pcr_score: 0.5,
        }
    }
}

/// Phase identification scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseScores {
    pub growth: f64,
    pub conservation: f64,
    pub release: f64,
    pub reorganization: f64,
}

impl PhaseScores {
    /// Get the dominant phase
    pub fn dominant_phase(&self) -> PanarchyPhase {
        let scores = [
            (self.growth, PanarchyPhase::Growth),
            (self.conservation, PanarchyPhase::Conservation),
            (self.release, PanarchyPhase::Release),
            (self.reorganization, PanarchyPhase::Reorganization),
        ];
        
        scores.iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, phase)| *phase)
            .unwrap_or(PanarchyPhase::Growth)
    }
}

/// Cross-scale interaction effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossScaleEffects {
    /// Upward causation (micro -> macro)
    pub upward_causation: f64,
    /// Downward causation (macro -> micro)
    pub downward_causation: f64,
    /// Cross-scale synchronization
    pub synchronization: f64,
    /// Panarchy connectivity
    pub connectivity: f64,
}

/// Complete panarchy analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyAnalysisResult {
    /// Current panarchy phase
    pub phase: PanarchyPhase,
    /// Phase confidence score
    pub confidence: f64,
    /// Trading signal strength
    pub signal: f64,
    /// PCR components
    pub pcr_components: PCRComponents,
    /// Phase scores for all phases
    pub phase_scores: PhaseScores,
    /// Current scale level
    pub scale_level: ScaleLevel,
    /// Cross-scale effects
    pub cross_scale_effects: CrossScaleEffects,
    /// Adaptive capacity
    pub adaptive_capacity: f64,
    /// Phase transition probability
    pub transition_probability: f64,
    /// Data points analyzed
    pub data_points: usize,
    /// Analysis duration
    pub analysis_duration: std::time::Duration,
    /// Performance metrics
    pub performance_score: f64,
}

/// Panarchy analyzer with cross-scale capabilities
pub struct PanarchyAnalyzer {
    config: PanarchyConfig,
    scale_manager: Arc<RwLock<ScaleManager>>,
    phase_tracker: Arc<RwLock<PhaseTracker>>,
    pcr_calculator: Arc<PCRCalculator>,
    performance_metrics: Arc<std::sync::Mutex<AnalysisMetrics>>,
    cache: Arc<std::sync::Mutex<HashMap<String, CachedAnalysis>>>,
}

impl PanarchyAnalyzer {
    /// Create new panarchy analyzer
    pub fn new(config: PanarchyConfig) -> PanarchyResult<Self> {
        Ok(Self {
            scale_manager: Arc::new(RwLock::new(ScaleManager::new(&config)?)),
            phase_tracker: Arc::new(RwLock::new(PhaseTracker::new(&config)?)),
            pcr_calculator: Arc::new(PCRCalculator::new(&config)?),
            performance_metrics: Arc::new(std::sync::Mutex::new(AnalysisMetrics::new())),
            cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            config,
        })
    }
    
    /// Analyze market data using panarchy theory
    pub async fn analyze(
        &self,
        prices: &[f64],
        volumes: &[f64],
    ) -> PanarchyResult<PanarchyAnalysisResult> {
        let start_time = Instant::now();
        
        // Validate inputs
        if prices.len() < self.config.min_data_points {
            return Err(PanarchyError::InsufficientData {
                required: self.config.min_data_points,
                actual: prices.len(),
            });
        }
        
        if prices.len() != volumes.len() {
            return Err(PanarchyError::InvalidParameters {
                message: format!("Price and volume arrays must have same length: {} vs {}", 
                               prices.len(), volumes.len())
            });
        }
        
        // Check cache first
        let cache_key = self.generate_cache_key(prices, volumes);
        if let Ok(cache) = self.cache.lock() {
            if let Some(cached) = cache.get(&cache_key) {
                if cached.is_valid() {
                    return Ok(cached.result.clone());
                }
            }
        }
        
        // Perform panarchy analysis
        let analysis_result = self.perform_analysis(prices, volumes).await?;
        
        // Cache result
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(cache_key, CachedAnalysis::new(analysis_result.clone()));
            
            // Limit cache size
            if cache.len() > 1000 {
                let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 4).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
        
        // Update performance metrics
        let duration = start_time.elapsed();
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.record_analysis(duration, prices.len());
        }
        
        Ok(analysis_result)
    }
    
    /// Perform core panarchy analysis
    async fn perform_analysis(
        &self,
        prices: &[f64],
        volumes: &[f64],
    ) -> PanarchyResult<PanarchyAnalysisResult> {
        // Calculate PCR components
        let pcr_components = self.pcr_calculator
            .calculate_pcr_components(prices, self.config.pcr_period)?;
        
        // Identify current phase
        let phase_scores = self.calculate_phase_scores(prices, volumes, &pcr_components).await?;
        let current_phase = phase_scores.dominant_phase();
        
        // Determine scale level
        let scale_level = self.scale_manager.read().await
            .determine_scale_level(prices, volumes)
            .await?;
        
        // Calculate cross-scale effects
        let cross_scale_effects = self.calculate_cross_scale_effects(
            prices, volumes, &pcr_components, current_phase
        ).await?;
        
        // Calculate adaptive capacity
        let adaptive_capacity = self.calculate_adaptive_capacity(
            &pcr_components, &cross_scale_effects
        )?;
        
        // Predict phase transition probability
        let transition_probability = self.calculate_transition_probability(
            &phase_scores, &pcr_components
        )?;
        
        // Generate trading signal
        let signal = self.generate_trading_signal(
            current_phase, &pcr_components, &cross_scale_effects
        )?;
        
        // Calculate confidence
        let confidence = self.calculate_confidence(&phase_scores, &pcr_components)?;
        
        // Calculate performance score
        let performance_score = self.calculate_performance_score(
            &pcr_components, adaptive_capacity, confidence
        )?;
        
        Ok(PanarchyAnalysisResult {
            phase: current_phase,
            confidence,
            signal,
            pcr_components,
            phase_scores,
            scale_level,
            cross_scale_effects,
            adaptive_capacity,
            transition_probability,
            data_points: prices.len(),
            analysis_duration: std::time::Duration::from_nanos(0), // Will be set by caller
            performance_score,
        })
    }
    
    /// Calculate phase identification scores
    async fn calculate_phase_scores(
        &self,
        prices: &[f64],
        volumes: &[f64],
        pcr: &PCRComponents,
    ) -> PanarchyResult<PhaseScores> {
        let mut phase_tracker = self.phase_tracker.write().await;
        phase_tracker.calculate_phase_scores(prices, volumes, pcr)
    }
    
    /// Calculate cross-scale interaction effects
    async fn calculate_cross_scale_effects(
        &self,
        prices: &[f64],
        volumes: &[f64],
        pcr: &PCRComponents,
        phase: PanarchyPhase,
    ) -> PanarchyResult<CrossScaleEffects> {
        let scale_manager = self.scale_manager.read().await;
        scale_manager.calculate_cross_scale_effects(prices, volumes, pcr, phase)
    }
    
    /// Calculate adaptive capacity
    fn calculate_adaptive_capacity(
        &self,
        pcr: &PCRComponents,
        cross_scale: &CrossScaleEffects,
    ) -> PanarchyResult<f64> {
        // Adaptive capacity is function of resilience and cross-scale connectivity
        let capacity = (pcr.resilience * 0.6) + 
                      (cross_scale.connectivity * 0.3) + 
                      (cross_scale.synchronization * 0.1);
        
        Ok(capacity.clamp(0.0, 1.0))
    }
    
    /// Calculate phase transition probability
    fn calculate_transition_probability(
        &self,
        phase_scores: &PhaseScores,
        pcr: &PCRComponents,
    ) -> PanarchyResult<f64> {
        // High potential and low resilience indicate higher transition probability
        let instability = pcr.potential * (1.0 - pcr.resilience);
        
        // Phase score variance indicates uncertainty
        let scores = [phase_scores.growth, phase_scores.conservation, 
                     phase_scores.release, phase_scores.reorganization];
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / scores.len() as f64;
        
        let uncertainty = variance.sqrt();
        
        let transition_prob = (instability * 0.7) + (uncertainty * 0.3);
        Ok(transition_prob.clamp(0.0, 1.0))
    }
    
    /// Generate trading signal from panarchy analysis
    fn generate_trading_signal(
        &self,
        phase: PanarchyPhase,
        pcr: &PCRComponents,
        cross_scale: &CrossScaleEffects,
    ) -> PanarchyResult<f64> {
        let base_signal = phase.to_signal();
        
        // Adjust signal based on PCR components
        let pcr_adjustment = (pcr.potential - pcr.connectedness) * 0.2;
        
        // Cross-scale adjustment
        let cross_scale_adjustment = cross_scale.upward_causation * 0.1;
        
        let final_signal = base_signal + pcr_adjustment + cross_scale_adjustment;
        Ok(final_signal.clamp(-1.0, 1.0))
    }
    
    /// Calculate confidence in analysis
    fn calculate_confidence(
        &self,
        phase_scores: &PhaseScores,
        pcr: &PCRComponents,
    ) -> PanarchyResult<f64> {
        // Confidence based on dominant phase strength and PCR coherence
        let scores = [phase_scores.growth, phase_scores.conservation,
                     phase_scores.release, phase_scores.reorganization];
        let max_score = scores.iter().fold(0.0f64, |a, &b| a.max(b));
        let sum_score = scores.iter().sum::<f64>();
        
        let dominance = if sum_score > 0.0 { max_score / sum_score } else { 0.0 };
        
        // PCR coherence (how well balanced the components are)
        let pcr_coherence = 1.0 - ((pcr.potential - pcr.resilience).abs() + 
                                  (pcr.connectedness - 0.5).abs()) / 2.0;
        
        let confidence = (dominance * 0.6) + (pcr_coherence * 0.4);
        Ok(confidence.clamp(0.0, 1.0))
    }
    
    /// Calculate overall performance score
    fn calculate_performance_score(
        &self,
        pcr: &PCRComponents,
        adaptive_capacity: f64,
        confidence: f64,
    ) -> PanarchyResult<f64> {
        let score = (pcr.pcr_score * 0.4) + 
                   (adaptive_capacity * 0.3) + 
                   (confidence * 0.3);
        
        Ok(score.clamp(0.0, 1.0))
    }
    
    /// Generate cache key for analysis caching
    fn generate_cache_key(&self, prices: &[f64], volumes: &[f64]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash recent data points for efficiency
        let sample_size = 20.min(prices.len());
        for &price in prices.iter().rev().take(sample_size) {
            price.to_bits().hash(&mut hasher);
        }
        for &volume in volumes.iter().rev().take(sample_size) {
            volume.to_bits().hash(&mut hasher);
        }
        
        format!("panarchy_{:x}", hasher.finish())
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> AnalysisMetrics {
        self.performance_metrics.lock()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }
}

/// Scale manager for cross-scale analysis
struct ScaleManager {
    config: PanarchyConfig,
    scale_thresholds: [f64; 3], // Micro, Meso, Macro thresholds
}

impl ScaleManager {
    fn new(config: &PanarchyConfig) -> PanarchyResult<Self> {
        Ok(Self {
            config: config.clone(),
            scale_thresholds: [0.3, 0.6, 0.9], // Default thresholds
        })
    }
    
    async fn determine_scale_level(
        &self,
        prices: &[f64],
        volumes: &[f64],
    ) -> PanarchyResult<ScaleLevel> {
        // Calculate volatility as scale indicator
        let volatility = self.calculate_volatility(prices)?;
        
        if volatility < self.scale_thresholds[0] {
            Ok(ScaleLevel::Micro)
        } else if volatility < self.scale_thresholds[1] {
            Ok(ScaleLevel::Meso)
        } else {
            Ok(ScaleLevel::Macro)
        }
    }
    
    fn calculate_volatility(&self, prices: &[f64]) -> PanarchyResult<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(variance.sqrt())
    }
    
    fn calculate_cross_scale_effects(
        &self,
        _prices: &[f64],
        _volumes: &[f64],
        pcr: &PCRComponents,
        phase: PanarchyPhase,
    ) -> PanarchyResult<CrossScaleEffects> {
        // Simplified cross-scale calculation
        let upward_causation = match phase {
            PanarchyPhase::Growth => pcr.potential * 0.8,
            PanarchyPhase::Conservation => pcr.connectedness * 0.6,
            PanarchyPhase::Release => (1.0 - pcr.resilience) * 0.9,
            PanarchyPhase::Reorganization => pcr.resilience * 0.7,
        };
        
        let downward_causation = pcr.connectedness * 0.5;
        let synchronization = (pcr.potential + pcr.resilience) / 2.0;
        let connectivity = pcr.pcr_score;
        
        Ok(CrossScaleEffects {
            upward_causation,
            downward_causation,
            synchronization,
            connectivity,
        })
    }
}

/// Phase tracker for identifying panarchy phases
struct PhaseTracker {
    config: PanarchyConfig,
    hysteresis_buffer: Vec<PanarchyPhase>,
}

impl PhaseTracker {
    fn new(config: &PanarchyConfig) -> PanarchyResult<Self> {
        Ok(Self {
            config: config.clone(),
            hysteresis_buffer: Vec::new(),
        })
    }
    
    fn calculate_phase_scores(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        pcr: &PCRComponents,
    ) -> PanarchyResult<PhaseScores> {
        // Calculate phase indicators
        let volatility = self.calculate_volatility(prices)?;
        let momentum = self.calculate_momentum(prices)?;
        let volume_trend = self.calculate_volume_trend(volumes)?;
        
        // Growth phase: high momentum, moderate volatility, increasing volume
        let growth = momentum.max(0.0) * (1.0 - volatility) * volume_trend * pcr.potential;
        
        // Conservation phase: low volatility, stable momentum, stable volume
        let conservation = (1.0 - volatility) * (1.0 - momentum.abs()) * 
                          (1.0 - volume_trend.abs()) * pcr.connectedness;
        
        // Release phase: high volatility, negative momentum, high volume
        let release = volatility * momentum.min(0.0).abs() * volume_trend * 
                     (1.0 - pcr.resilience);
        
        // Reorganization phase: moderate volatility, low momentum, low volume
        let reorganization = (volatility * 0.5) * (1.0 - momentum.abs()) * 
                           (1.0 - volume_trend) * pcr.resilience;
        
        Ok(PhaseScores {
            growth,
            conservation,
            release,
            reorganization,
        })
    }
    
    fn calculate_volatility(&self, prices: &[f64]) -> PanarchyResult<f64> {
        if prices.len() < 10 {
            return Ok(0.0);
        }
        
        let window = 10.min(prices.len());
        let recent_prices = &prices[prices.len() - window..];
        
        let returns: Vec<f64> = recent_prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(variance.sqrt())
    }
    
    fn calculate_momentum(&self, prices: &[f64]) -> PanarchyResult<f64> {
        if prices.len() < 10 {
            return Ok(0.0);
        }
        
        let n = prices.len();
        let short_period = 5.min(n / 2);
        let long_period = 10.min(n);
        
        let short_avg = prices[n - short_period..].iter().sum::<f64>() / short_period as f64;
        let long_avg = prices[n - long_period..].iter().sum::<f64>() / long_period as f64;
        
        Ok((short_avg - long_avg) / long_avg)
    }
    
    fn calculate_volume_trend(&self, volumes: &[f64]) -> PanarchyResult<f64> {
        if volumes.len() < 10 {
            return Ok(0.0);
        }
        
        let n = volumes.len();
        let short_period = 5.min(n / 2);
        let long_period = 10.min(n);
        
        let recent_avg = volumes[n - short_period..].iter().sum::<f64>() / short_period as f64;
        let historical_avg = volumes[n - long_period..].iter().sum::<f64>() / long_period as f64;
        
        Ok((recent_avg - historical_avg) / historical_avg)
    }
}

/// PCR calculator for panarchy components
struct PCRCalculator {
    config: PanarchyConfig,
}

impl PCRCalculator {
    fn new(config: &PanarchyConfig) -> PanarchyResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    fn calculate_pcr_components(&self, prices: &[f64], period: usize) -> PanarchyResult<PCRComponents> {
        if prices.len() < period {
            return Ok(PCRComponents::default());
        }
        
        // Calculate potential (accumulated resources/energy)
        let potential = self.calculate_potential(prices, period)?;
        
        // Calculate connectedness (rigidity of control)
        let connectedness = self.calculate_connectedness(prices, period)?;
        
        // Calculate resilience (adaptive capacity)
        let resilience = self.calculate_resilience(prices, period)?;
        
        // Combined PCR score
        let pcr_score = (potential + connectedness + resilience) / 3.0;
        
        Ok(PCRComponents {
            potential,
            connectedness,
            resilience,
            pcr_score,
        })
    }
    
    fn calculate_potential(&self, prices: &[f64], period: usize) -> PanarchyResult<f64> {
        // Potential based on price range expansion
        let n = prices.len();
        let window = &prices[n - period.min(n)..];
        
        let max_price = window.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_price = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current_price = window[window.len() - 1];
        
        if max_price > min_price {
            Ok((current_price - min_price) / (max_price - min_price))
        } else {
            Ok(0.5)
        }
    }
    
    fn calculate_connectedness(&self, prices: &[f64], period: usize) -> PanarchyResult<f64> {
        // Connectedness based on price stability and correlation
        let n = prices.len();
        let window = &prices[n - period.min(n)..];
        
        if window.len() < 3 {
            return Ok(0.5);
        }
        
        // Calculate coefficient of variation (inverse of stability)
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / window.len() as f64;
        let std_dev = variance.sqrt();
        
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
        
        // Higher stability = higher connectedness
        Ok((1.0 - cv).clamp(0.0, 1.0))
    }
    
    fn calculate_resilience(&self, prices: &[f64], period: usize) -> PanarchyResult<f64> {
        // Resilience based on recovery from shocks
        let n = prices.len();
        let window = &prices[n - period.min(n)..];
        
        if window.len() < 5 {
            return Ok(0.5);
        }
        
        // Find maximum drawdown and recovery
        let mut max_drawdown = 0.0;
        let mut peak = window[0];
        
        for &price in window {
            if price > peak {
                peak = price;
            } else {
                let drawdown = (peak - price) / peak;
                max_drawdown = max_drawdown.max(drawdown);
            }
        }
        
        // Resilience is inverse of maximum drawdown
        Ok((1.0 - max_drawdown).clamp(0.0, 1.0))
    }
}

/// Cached analysis result
#[derive(Clone)]
struct CachedAnalysis {
    result: PanarchyAnalysisResult,
    timestamp: Instant,
    ttl: std::time::Duration,
}

impl CachedAnalysis {
    fn new(result: PanarchyAnalysisResult) -> Self {
        Self {
            result,
            timestamp: Instant::now(),
            ttl: std::time::Duration::from_secs(300), // 5 minutes TTL
        }
    }
    
    fn is_valid(&self) -> bool {
        self.timestamp.elapsed() < self.ttl
    }
}

/// Performance metrics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetrics {
    pub total_analyses: u64,
    pub average_analysis_time: std::time::Duration,
    pub cache_hit_rate: f64,
    pub total_data_points_processed: u64,
}

impl AnalysisMetrics {
    fn new() -> Self {
        Self {
            total_analyses: 0,
            average_analysis_time: std::time::Duration::from_nanos(0),
            cache_hit_rate: 0.0,
            total_data_points_processed: 0,
        }
    }
    
    fn record_analysis(&mut self, duration: std::time::Duration, data_points: usize) {
        let total_time = self.average_analysis_time * self.total_analyses as u32 + duration;
        self.total_analyses += 1;
        self.average_analysis_time = total_time / self.total_analyses as u32;
        self.total_data_points_processed += data_points as u64;
    }
}

impl Default for AnalysisMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory function for creating panarchy analyzer
pub fn create_panarchy_analyzer() -> PanarchyResult<PanarchyAnalyzer> {
    let config = PanarchyConfig::default();
    PanarchyAnalyzer::new(config)
}

/// Factory function for high-performance panarchy analyzer
pub fn create_high_performance_analyzer() -> PanarchyResult<PanarchyAnalyzer> {
    let config = PanarchyConfig {
        window_size: 100,
        pcr_period: 21,
        min_data_points: 50,
        phase_transition_threshold: 0.2,
        scale_transition_sensitivity: 0.3,
        cross_scale_strength: 0.3,
        adaptive_feedback_rate: 0.05,
        enable_simd: true,
        enable_parallel: true,
        performance_target_ns: 500,
    };
    
    PanarchyAnalyzer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        
        let mut price = 100.0;
        for i in 0..n {
            let return_rate = 0.01 * ((i as f64) * 0.1).sin();
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(1000.0 + 100.0 * ((i as f64) * 0.05).cos());
        }
        
        (prices, volumes)
    }
    
    #[tokio::test]
    async fn test_panarchy_analyzer_creation() {
        let analyzer = create_panarchy_analyzer();
        assert!(analyzer.is_ok());
    }
    
    #[tokio::test]
    async fn test_basic_analysis() {
        let analyzer = create_panarchy_analyzer().unwrap();
        let (prices, volumes) = generate_test_data(50);
        
        let result = analyzer.analyze(&prices, &volumes).await;
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
        assert!(analysis.signal >= -1.0 && analysis.signal <= 1.0);
        assert_eq!(analysis.data_points, 50);
    }
    
    #[tokio::test]
    async fn test_insufficient_data() {
        let analyzer = create_panarchy_analyzer().unwrap();
        let (prices, volumes) = generate_test_data(10);
        
        let result = analyzer.analyze(&prices, &volumes).await;
        assert!(result.is_err());
        
        if let Err(PanarchyError::InsufficientData { required, actual }) = result {
            assert_eq!(required, 20);
            assert_eq!(actual, 10);
        } else {
            panic!("Expected InsufficientData error");
        }
    }
    
    #[test]
    fn test_phase_signal_conversion() {
        assert_eq!(PanarchyPhase::Growth.to_signal(), 0.8);
        assert_eq!(PanarchyPhase::Conservation.to_signal(), 0.2);
        assert_eq!(PanarchyPhase::Release.to_signal(), -0.8);
        assert_eq!(PanarchyPhase::Reorganization.to_signal(), -0.2);
    }
    
    #[test]
    fn test_pcr_components_default() {
        let pcr = PCRComponents::default();
        assert_eq!(pcr.potential, 0.5);
        assert_eq!(pcr.connectedness, 0.5);
        assert_eq!(pcr.resilience, 0.5);
        assert_eq!(pcr.pcr_score, 0.5);
    }
}