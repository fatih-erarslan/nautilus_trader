//! Quantum pattern integration for Tengri trading strategy

use crate::Result;
use crate::types::*;

use quantum_enhanced::{QuantumPatternEngine, ClassicalInterface, QuantumConfig};
use quantum_enhanced::types::{QuantumSignal, MarketData as QuantumMarketData, ClassicalTradingSignal};

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};

/// Quantum pattern integration for Tengri strategy
pub struct TengriQuantumIntegration {
    /// Quantum pattern engine
    quantum_engine: Arc<QuantumPatternEngine>,
    /// Classical interface for signal conversion
    classical_interface: Arc<ClassicalInterface>,
    /// Configuration
    config: TengriQuantumConfig,
    /// Signal cache for efficiency
    signal_cache: Arc<RwLock<HashMap<String, CachedQuantumSignal>>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<QuantumIntegrationMetrics>>,
}

/// Configuration for quantum integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriQuantumConfig {
    /// Quantum pattern detection settings
    pub quantum_config: QuantumConfig,
    /// Signal processing settings
    pub signal_processing: SignalProcessingConfig,
    /// Performance optimization settings
    pub performance: QuantumPerformanceConfig,
    /// Risk management integration
    pub risk_integration: RiskIntegrationConfig,
}

/// Signal processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalProcessingConfig {
    /// Minimum signal confidence for trading
    pub min_signal_confidence: f64,
    /// Maximum signals to process per interval
    pub max_signals_per_interval: usize,
    /// Signal aggregation window (seconds)
    pub signal_aggregation_window_seconds: u64,
    /// Enable ensemble signal processing
    pub enable_ensemble_processing: bool,
    /// Claude Flow coordination settings
    pub claude_flow_coordination: bool,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPerformanceConfig {
    /// Target detection latency (microseconds)
    pub target_detection_latency_us: u64,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Memory pool size (MB)
    pub memory_pool_size_mb: usize,
    /// Cache size for signals
    pub signal_cache_size: usize,
}

/// Risk integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskIntegrationConfig {
    /// Quantum signal weight in risk calculations
    pub quantum_signal_weight: f64,
    /// Maximum position size from quantum signals
    pub max_quantum_position_size: f64,
    /// Enable quantum-based stop losses
    pub enable_quantum_stop_losses: bool,
    /// Coherence-based position sizing
    pub coherence_based_sizing: bool,
}

/// Cached quantum signal with metadata
#[derive(Debug, Clone)]
struct CachedQuantumSignal {
    signal: QuantumSignal,
    classical_signal: Option<ClassicalTradingSignal>,
    cache_time: DateTime<Utc>,
    access_count: u32,
}

/// Performance metrics for quantum integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumIntegrationMetrics {
    /// Total quantum patterns detected
    pub total_patterns_detected: u64,
    /// Average detection latency (microseconds)
    pub avg_detection_latency_us: f64,
    /// Trading signals generated
    pub trading_signals_generated: u64,
    /// Signal conversion success rate
    pub signal_conversion_rate: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Quantum signal accuracy (when backtested)
    pub quantum_signal_accuracy: f64,
}

impl TengriQuantumIntegration {
    /// Create new quantum integration
    pub async fn new(config: TengriQuantumConfig) -> Result<Self> {
        info!("Initializing Tengri Quantum Integration");

        // Initialize quantum pattern engine
        let quantum_engine = Arc::new(
            QuantumPatternEngine::new(config.quantum_config.clone()).await
                .map_err(|e| crate::TengriError::Strategy(format!("Quantum engine init failed: {}", e)))?
        );

        // Initialize classical interface
        let classical_interface = Arc::new(
            ClassicalInterface::new(config.quantum_config.clone()).await
                .map_err(|e| crate::TengriError::Strategy(format!("Classical interface init failed: {}", e)))?
        );

        let signal_cache = Arc::new(RwLock::new(HashMap::new()));
        let performance_metrics = Arc::new(RwLock::new(QuantumIntegrationMetrics::default()));

        info!("Tengri Quantum Integration initialized successfully");

        Ok(Self {
            quantum_engine,
            classical_interface,
            config,
            signal_cache,
            performance_metrics,
        })
    }

    /// Process market data and generate quantum trading signals
    pub async fn process_market_data(
        &self,
        market_data: &MarketData,
        instruments: &[String],
    ) -> Result<Vec<ClassicalTradingSignal>> {
        let start_time = std::time::Instant::now();
        
        debug!("Processing market data with quantum patterns for {} instruments", instruments.len());

        // Convert Tengri market data to quantum format
        let quantum_market_data = self.convert_to_quantum_market_data(market_data, instruments).await?;

        // Generate signals based on configuration
        let quantum_signals = if self.config.signal_processing.enable_ensemble_processing {
            self.quantum_engine.detect_ensemble_patterns(&quantum_market_data).await
                .map_err(|e| crate::TengriError::Strategy(format!("Ensemble detection failed: {}", e)))?
        } else {
            vec![self.quantum_engine.detect_quantum_patterns(&quantum_market_data).await
                .map_err(|e| crate::TengriError::Strategy(format!("Pattern detection failed: {}", e)))?]
        };

        // Filter signals by confidence
        let filtered_signals: Vec<QuantumSignal> = quantum_signals.into_iter()
            .filter(|signal| signal.confidence >= self.config.signal_processing.min_signal_confidence)
            .take(self.config.signal_processing.max_signals_per_interval)
            .collect();

        debug!("Filtered to {} high-confidence quantum signals", filtered_signals.len());

        // Convert to classical trading signals
        let mut classical_signals = Vec::new();
        for quantum_signal in &filtered_signals {
            // Check cache first
            if let Some(cached_signal) = self.get_cached_signal(&quantum_signal.id.to_string()).await? {
                classical_signals.push(cached_signal);
                continue;
            }

            // Convert quantum signal to classical
            if let Some(classical_signal) = self.classical_interface
                .convert_to_trading_signal(quantum_signal, &quantum_market_data.classical_data).await
                .map_err(|e| crate::TengriError::Strategy(format!("Signal conversion failed: {}", e)))?
            {
                // Cache the result
                self.cache_signal(quantum_signal.clone(), Some(classical_signal.clone())).await?;
                classical_signals.push(classical_signal);
            }
        }

        // Update performance metrics
        let detection_latency = start_time.elapsed().as_micros() as u64;
        self.update_performance_metrics(filtered_signals.len(), classical_signals.len(), detection_latency).await?;

        info!("Generated {} classical trading signals from quantum patterns in {}μs", 
              classical_signals.len(), detection_latency);

        Ok(classical_signals)
    }

    /// Apply quantum-enhanced risk management
    pub async fn apply_quantum_risk_management(
        &self,
        signal: &ClassicalTradingSignal,
        current_position: Option<f64>,
        account_balance: f64,
    ) -> Result<QuantumRiskAdjustment> {
        
        debug!("Applying quantum risk management for signal {}", signal.id);

        // Base position size from signal
        let base_position_size = signal.position_size_multiplier * account_balance * 0.01; // 1% base

        // Quantum adjustments
        let coherence_adjustment = if self.config.risk_integration.coherence_based_sizing {
            signal.quantum_metadata.coherence
        } else {
            1.0
        };

        let confidence_adjustment = signal.confidence;
        let entanglement_adjustment = 1.0 + signal.quantum_metadata.entanglement_strength * 0.2;

        // Calculate final position size
        let quantum_position_size = base_position_size * 
                                   coherence_adjustment * 
                                   confidence_adjustment * 
                                   entanglement_adjustment;

        let final_position_size = quantum_position_size
            .min(self.config.risk_integration.max_quantum_position_size * account_balance)
            .max(account_balance * 0.001); // Minimum 0.1%

        // Quantum-enhanced stop loss
        let quantum_stop_loss = if self.config.risk_integration.enable_quantum_stop_losses {
            // Adjust stop loss based on quantum coherence
            let base_stop_loss = signal.risk_parameters.stop_loss_pct;
            let coherence_factor = 1.0 - signal.quantum_metadata.coherence * 0.3;
            Some(base_stop_loss * coherence_factor)
        } else {
            Some(signal.risk_parameters.stop_loss_pct)
        };

        // Risk score calculation
        let risk_score = self.calculate_quantum_risk_score(signal, current_position, account_balance).await?;

        Ok(QuantumRiskAdjustment {
            original_position_size: base_position_size,
            quantum_adjusted_position_size: final_position_size,
            quantum_stop_loss,
            risk_score,
            coherence_factor: coherence_adjustment,
            confidence_factor: confidence_adjustment,
            entanglement_factor: entanglement_adjustment,
            recommendations: self.generate_risk_recommendations(signal, risk_score).await?,
        })
    }

    /// Validate quantum signal quality
    pub async fn validate_signal_quality(&self, signal: &ClassicalTradingSignal) -> Result<SignalQualityAssessment> {
        debug!("Validating quantum signal quality for {}", signal.id);

        let mut quality_score = 1.0;
        let mut quality_issues = Vec::new();

        // Check quantum metadata quality
        if signal.quantum_metadata.coherence < 0.5 {
            quality_score *= 0.7;
            quality_issues.push("Low quantum coherence detected".to_string());
        }

        if signal.quantum_metadata.validation_score < 0.6 {
            quality_score *= 0.8;
            quality_issues.push("Low quantum validation score".to_string());
        }

        if signal.quantum_metadata.entanglement_strength < 0.3 {
            quality_score *= 0.9;
            quality_issues.push("Weak entanglement strength".to_string());
        }

        // Check signal consistency
        if signal.confidence < 0.7 && signal.strength < 0.5 {
            quality_score *= 0.6;
            quality_issues.push("Low confidence and strength combination".to_string());
        }

        // Check pattern type appropriateness
        let pattern_quality = self.assess_pattern_type_quality(&signal.quantum_metadata.pattern_type).await?;
        quality_score *= pattern_quality;

        if pattern_quality < 0.8 {
            quality_issues.push("Pattern type may not be suitable for current market conditions".to_string());
        }

        Ok(SignalQualityAssessment {
            overall_quality_score: quality_score,
            is_high_quality: quality_score > 0.7,
            quality_issues,
            recommendations: self.generate_quality_recommendations(signal, quality_score).await?,
        })
    }

    /// Get current quantum integration performance metrics
    pub async fn get_performance_metrics(&self) -> Result<QuantumIntegrationMetrics> {
        let metrics = self.performance_metrics.read().await;
        Ok(metrics.clone())
    }

    /// Clear signal cache (for memory management)
    pub async fn clear_signal_cache(&self) -> Result<()> {
        let mut cache = self.signal_cache.write().await;
        cache.clear();
        info!("Quantum signal cache cleared");
        Ok(())
    }

    // Private helper methods

    async fn convert_to_quantum_market_data(
        &self,
        market_data: &MarketData,
        instruments: &[String],
    ) -> Result<QuantumMarketData> {
        
        // Convert Tengri market data format to quantum format
        let mut price_history = HashMap::new();
        let mut volume_data = HashMap::new();

        for instrument in instruments {
            if let Some(prices) = market_data.get_price_history(instrument) {
                price_history.insert(instrument.clone(), prices.clone());
            }
            if let Some(volumes) = market_data.get_volume_data(instrument) {
                volume_data.insert(instrument.clone(), volumes.clone());
            }
        }

        let quantum_market_data = quantum_enhanced::types::MarketData {
            price_history,
            volume_data,
            timestamps: market_data.get_timestamps().unwrap_or_default(),
            features: market_data.get_features().unwrap_or_else(|| ndarray::Array2::zeros((0, 0))),
            regime_indicators: market_data.get_regime_indicators().unwrap_or_else(|| ndarray::Array1::zeros(0)),
        };

        Ok(quantum_market_data)
    }

    async fn get_cached_signal(&self, signal_id: &str) -> Result<Option<ClassicalTradingSignal>> {
        let cache = self.signal_cache.read().await;
        
        if let Some(cached) = cache.get(signal_id) {
            // Check if cache is still valid (within aggregation window)
            let cache_age = Utc::now().signed_duration_since(cached.cache_time);
            if cache_age.num_seconds() < self.config.signal_processing.signal_aggregation_window_seconds as i64 {
                return Ok(cached.classical_signal.clone());
            }
        }
        
        Ok(None)
    }

    async fn cache_signal(
        &self,
        quantum_signal: QuantumSignal,
        classical_signal: Option<ClassicalTradingSignal>,
    ) -> Result<()> {
        let mut cache = self.signal_cache.write().await;
        
        let cached_signal = CachedQuantumSignal {
            signal: quantum_signal.clone(),
            classical_signal,
            cache_time: Utc::now(),
            access_count: 1,
        };
        
        cache.insert(quantum_signal.id.to_string(), cached_signal);
        
        // Limit cache size
        if cache.len() > self.config.performance.signal_cache_size {
            // Remove oldest entries
            let mut entries: Vec<_> = cache.iter().collect();
            entries.sort_by_key(|(_, cached)| cached.cache_time);
            
            for (key, _) in entries.iter().take(cache.len() - self.config.performance.signal_cache_size) {
                cache.remove(*key);
            }
        }
        
        Ok(())
    }

    async fn update_performance_metrics(
        &self,
        quantum_signals: usize,
        classical_signals: usize,
        detection_latency: u64,
    ) -> Result<()> {
        let mut metrics = self.performance_metrics.write().await;
        
        metrics.total_patterns_detected += quantum_signals as u64;
        metrics.trading_signals_generated += classical_signals as u64;
        
        // Update average latency
        let alpha = 0.1; // Exponential moving average factor
        metrics.avg_detection_latency_us = alpha * detection_latency as f64 + 
                                          (1.0 - alpha) * metrics.avg_detection_latency_us;
        
        // Update conversion rate
        if quantum_signals > 0 {
            let current_conversion_rate = classical_signals as f64 / quantum_signals as f64;
            metrics.signal_conversion_rate = alpha * current_conversion_rate + 
                                           (1.0 - alpha) * metrics.signal_conversion_rate;
        }
        
        Ok(())
    }

    async fn calculate_quantum_risk_score(
        &self,
        signal: &ClassicalTradingSignal,
        _current_position: Option<f64>,
        _account_balance: f64,
    ) -> Result<f64> {
        
        // Calculate composite risk score from quantum metrics
        let coherence_risk = 1.0 - signal.quantum_metadata.coherence;
        let confidence_risk = 1.0 - signal.confidence;
        let validation_risk = 1.0 - signal.quantum_metadata.validation_score;
        let entanglement_risk = if signal.quantum_metadata.entanglement_strength < 0.3 { 0.3 } else { 0.0 };
        
        let composite_risk = (coherence_risk * 0.3 + 
                             confidence_risk * 0.3 + 
                             validation_risk * 0.2 + 
                             entanglement_risk * 0.2).min(1.0).max(0.0);
        
        Ok(composite_risk)
    }

    async fn assess_pattern_type_quality(&self, pattern_type: &str) -> Result<f64> {
        // Assess pattern type quality based on current market conditions
        // This would integrate with market regime detection in a full implementation
        
        match pattern_type {
            "Superposition Momentum" => Ok(0.9), // Generally high quality
            "Entangled Correlation" => Ok(0.85),
            "Quantum Tunneling" => Ok(0.8),
            "Coherent Oscillation" => Ok(0.75),
            "Quantum Interference" => Ok(0.7),
            "Quantum Resonance" => Ok(0.8),
            _ => Ok(0.6), // Unknown patterns get lower quality score
        }
    }

    async fn generate_risk_recommendations(
        &self,
        signal: &ClassicalTradingSignal,
        risk_score: f64,
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        if risk_score > 0.7 {
            recommendations.push("High risk signal - consider reducing position size".to_string());
        }
        
        if signal.quantum_metadata.coherence < 0.5 {
            recommendations.push("Low coherence - use tighter stop losses".to_string());
        }
        
        if signal.quantum_metadata.entanglement_strength < 0.3 {
            recommendations.push("Weak entanglement - monitor for signal degradation".to_string());
        }
        
        if signal.confidence < 0.7 {
            recommendations.push("Lower confidence - wait for confirmation".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Signal quality is good - proceed with standard risk management".to_string());
        }
        
        Ok(recommendations)
    }

    async fn generate_quality_recommendations(
        &self,
        signal: &ClassicalTradingSignal,
        quality_score: f64,
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        if quality_score < 0.5 {
            recommendations.push("Consider waiting for higher quality quantum signals".to_string());
        } else if quality_score < 0.7 {
            recommendations.push("Moderate quality signal - use reduced position sizing".to_string());
        } else {
            recommendations.push("High quality quantum signal - suitable for trading".to_string());
        }
        
        if signal.quantum_metadata.coherence < 0.6 {
            recommendations.push("Low coherence may indicate signal instability".to_string());
        }
        
        if signal.quantum_metadata.validation_score < 0.7 {
            recommendations.push("Consider ensemble approach for improved validation".to_string());
        }
        
        Ok(recommendations)
    }
}

/// Quantum risk adjustment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRiskAdjustment {
    pub original_position_size: f64,
    pub quantum_adjusted_position_size: f64,
    pub quantum_stop_loss: Option<f64>,
    pub risk_score: f64,
    pub coherence_factor: f64,
    pub confidence_factor: f64,
    pub entanglement_factor: f64,
    pub recommendations: Vec<String>,
}

/// Signal quality assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualityAssessment {
    pub overall_quality_score: f64,
    pub is_high_quality: bool,
    pub quality_issues: Vec<String>,
    pub recommendations: Vec<String>,
}

impl Default for QuantumIntegrationMetrics {
    fn default() -> Self {
        Self {
            total_patterns_detected: 0,
            avg_detection_latency_us: 0.0,
            trading_signals_generated: 0,
            signal_conversion_rate: 0.0,
            cache_hit_rate: 0.0,
            quantum_signal_accuracy: 0.0,
        }
    }
}

impl Default for TengriQuantumConfig {
    fn default() -> Self {
        Self {
            quantum_config: QuantumConfig::default(),
            signal_processing: SignalProcessingConfig {
                min_signal_confidence: 0.7,
                max_signals_per_interval: 5,
                signal_aggregation_window_seconds: 60,
                enable_ensemble_processing: true,
                claude_flow_coordination: true,
            },
            performance: QuantumPerformanceConfig {
                target_detection_latency_us: 100,
                enable_parallel_processing: true,
                memory_pool_size_mb: 512,
                signal_cache_size: 100,
            },
            risk_integration: RiskIntegrationConfig {
                quantum_signal_weight: 0.3,
                max_quantum_position_size: 0.05, // 5% max
                enable_quantum_stop_losses: true,
                coherence_based_sizing: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_integration_creation() {
        let config = TengriQuantumConfig::default();
        let integration = TengriQuantumIntegration::new(config).await;
        assert!(integration.is_ok());
    }

    #[tokio::test]
    async fn test_market_data_conversion() {
        let config = TengriQuantumConfig::default();
        let integration = TengriQuantumIntegration::new(config).await.unwrap();
        
        // Create mock market data
        let market_data = create_mock_market_data();
        let instruments = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        
        let quantum_data = integration.convert_to_quantum_market_data(&market_data, &instruments).await;
        assert!(quantum_data.is_ok());
    }

    fn create_mock_market_data() -> MarketData {
        // Mock implementation for testing
        MarketData::new() // Assuming MarketData has a new() method
    }
}