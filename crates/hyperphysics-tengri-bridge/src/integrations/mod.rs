//! Integration layer connecting adapters to provide unified trading intelligence
//!
//! Each integration provides a high-level interface for specific analysis capabilities.

use crate::adapters::{
    AutopoiesisAdapter, AutopoiesisMarketRegime, AutopoiesisState, ConsciousnessAdapter,
    IntegrationLevel, MarketStateVector, PbitAdapter, PhiMetrics, QuantumAdapter,
    QuantumPatternMetrics, RiskAdapter, RiskLevel, RiskMetrics, SyncState, SyntergicAdapter,
    SyntergicMetrics, ThermoAdapter, ThermoMetrics, ThermoRegime, UncertaintyMetrics,
};
use crate::error::{BridgeError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// Re-export configuration types for external use
pub use crate::adapters::{
    AutopoiesisConfig, ConsciousnessConfig, PbitConfig, QuantumConfig, RiskConfig, SyntergicConfig,
    ThermoConfig,
};

// ============================================================================
// Integration Trait
// ============================================================================

/// Trait for all HyperPhysics integrations
#[async_trait]
pub trait HyperPhysicsIntegration: Send + Sync {
    /// Integration name
    fn name(&self) -> &'static str;

    /// Process market data and return signal contribution
    async fn process(&mut self, prices: &[f64], volumes: &[f64]) -> Result<SignalContribution>;

    /// Get confidence in current analysis
    fn confidence(&self) -> f64;

    /// Reset internal state
    fn reset(&mut self);
}

/// Contribution to trading signal from an integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalContribution {
    /// Source integration name
    pub source: String,
    /// Signal direction (-1.0 to 1.0)
    pub direction: f64,
    /// Confidence in signal (0.0 to 1.0)
    pub confidence: f64,
    /// Risk adjustment factor (0.0 to 1.0)
    pub risk_adjustment: f64,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, f64>,
}

impl SignalContribution {
    /// Create new signal contribution
    pub fn new(source: &str, direction: f64, confidence: f64) -> Self {
        Self {
            source: source.to_string(),
            direction: direction.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            risk_adjustment: 1.0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// With risk adjustment
    pub fn with_risk_adjustment(mut self, adjustment: f64) -> Self {
        self.risk_adjustment = adjustment.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: f64) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }
}

// ============================================================================
// Autopoiesis Integration
// ============================================================================

/// Integration for autopoiesis-based market regime detection
pub struct AutopoiesisIntegration {
    adapter: AutopoiesisAdapter,
    last_state: Option<AutopoiesisState>,
    confidence: f64,
}

impl AutopoiesisIntegration {
    /// Create new autopoiesis integration
    pub fn new(config: AutopoiesisConfig) -> Self {
        Self {
            adapter: AutopoiesisAdapter::new(config),
            last_state: None,
            confidence: 0.5,
        }
    }

    /// Get current regime
    pub fn current_regime(&self) -> Option<AutopoiesisMarketRegime> {
        self.last_state.as_ref().map(|s| s.regime)
    }

    /// Get current health
    pub fn health(&self) -> f64 {
        self.last_state.as_ref().map(|s| s.health).unwrap_or(1.0)
    }

    /// Check if near bifurcation
    pub fn near_bifurcation(&self) -> bool {
        self.last_state
            .as_ref()
            .map(|s| s.near_bifurcation)
            .unwrap_or(false)
    }
}

#[async_trait]
impl HyperPhysicsIntegration for AutopoiesisIntegration {
    fn name(&self) -> &'static str {
        "autopoiesis"
    }

    async fn process(&mut self, prices: &[f64], volumes: &[f64]) -> Result<SignalContribution> {
        let state = self.adapter.update(prices, volumes)?;

        // Calculate signal direction based on regime and health
        let direction = match state.regime {
            AutopoiesisMarketRegime::Stable => 0.5 * state.coherence,
            AutopoiesisMarketRegime::Normal => 0.3 * state.coherence,
            AutopoiesisMarketRegime::Transitional => 0.0,
            AutopoiesisMarketRegime::Degraded => -0.3,
            AutopoiesisMarketRegime::Chaotic => -0.5,
        };

        // Calculate confidence based on health stability
        self.confidence = state.health * state.coherence;

        // Risk adjustment based on regime
        let risk_adj = match state.regime {
            AutopoiesisMarketRegime::Stable => 1.0,
            AutopoiesisMarketRegime::Normal => 0.8,
            AutopoiesisMarketRegime::Transitional => 0.5,
            AutopoiesisMarketRegime::Degraded => 0.3,
            AutopoiesisMarketRegime::Chaotic => 0.1,
        };

        self.last_state = Some(state.clone());

        Ok(SignalContribution::new("autopoiesis", direction, self.confidence)
            .with_risk_adjustment(risk_adj)
            .with_metadata("health", state.health)
            .with_metadata("entropy", state.entropy)
            .with_metadata("coherence", state.coherence))
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn reset(&mut self) {
        self.last_state = None;
        self.confidence = 0.5;
    }
}

// ============================================================================
// Consciousness Integration
// ============================================================================

/// Integration for IIT Φ-based market coherence analysis
pub struct ConsciousnessIntegration {
    adapter: ConsciousnessAdapter,
    last_metrics: Option<PhiMetrics>,
    confidence: f64,
}

impl ConsciousnessIntegration {
    /// Create new consciousness integration
    pub fn new(config: ConsciousnessConfig) -> Result<Self> {
        Ok(Self {
            adapter: ConsciousnessAdapter::new(config)?,
            last_metrics: None,
            confidence: 0.5,
        })
    }

    /// Get current Φ value
    pub fn phi(&self) -> f64 {
        self.adapter.current_phi()
    }

    /// Get integration level
    pub fn integration_level(&self) -> Option<IntegrationLevel> {
        self.last_metrics.as_ref().map(|m| m.integration_level)
    }
}

#[async_trait]
impl HyperPhysicsIntegration for ConsciousnessIntegration {
    fn name(&self) -> &'static str {
        "consciousness"
    }

    async fn process(&mut self, prices: &[f64], volumes: &[f64]) -> Result<SignalContribution> {
        let market_state = MarketStateVector::from_market_data(prices, volumes);
        let metrics = self.adapter.calculate_market_phi(&market_state)?;

        // High Φ indicates integrated, coherent market state
        let direction = match metrics.integration_level {
            IntegrationLevel::High => 0.4 * metrics.phi,
            IntegrationLevel::Medium => 0.2 * metrics.phi,
            IntegrationLevel::Low => -0.1,
        };

        self.confidence = metrics.broadcast_strength.min(1.0);

        // Risk adjustment based on integration
        let risk_adj = match metrics.integration_level {
            IntegrationLevel::High => 1.0,
            IntegrationLevel::Medium => 0.7,
            IntegrationLevel::Low => 0.4,
        };

        self.last_metrics = Some(metrics.clone());

        Ok(SignalContribution::new("consciousness", direction, self.confidence)
            .with_risk_adjustment(risk_adj)
            .with_metadata("phi", metrics.phi)
            .with_metadata("broadcast_strength", metrics.broadcast_strength)
            .with_metadata("effective_info", metrics.effective_info))
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn reset(&mut self) {
        self.last_metrics = None;
        self.confidence = 0.5;
    }
}

// ============================================================================
// Thermo Integration
// ============================================================================

/// Integration for thermodynamic entropy-based volatility analysis
pub struct ThermoIntegration {
    adapter: ThermoAdapter,
    last_metrics: Option<ThermoMetrics>,
    confidence: f64,
}

impl ThermoIntegration {
    /// Create new thermo integration
    pub fn new(config: ThermoConfig) -> Self {
        Self {
            adapter: ThermoAdapter::new(config),
            last_metrics: None,
            confidence: 0.5,
        }
    }

    /// Get current entropy
    pub fn entropy(&self) -> f64 {
        self.adapter.current_entropy()
    }

    /// Get current regime
    pub fn regime(&self) -> Option<ThermoRegime> {
        self.last_metrics.as_ref().map(|m| m.regime)
    }
}

#[async_trait]
impl HyperPhysicsIntegration for ThermoIntegration {
    fn name(&self) -> &'static str {
        "thermo"
    }

    async fn process(&mut self, prices: &[f64], volumes: &[f64]) -> Result<SignalContribution> {
        let metrics = self.adapter.calculate_thermo_metrics(prices, volumes)?;

        // Low entropy = more predictable = stronger signals
        let direction = match metrics.regime {
            ThermoRegime::Equilibrium => 0.3 * (1.0 - metrics.entropy_production.tanh()),
            ThermoRegime::NearEquilibrium => 0.2 * (1.0 - metrics.entropy_production.tanh()),
            ThermoRegime::Dissipative => 0.0, // Forming new structure - wait
            ThermoRegime::HighEnergy => -0.2, // High volatility - reduce exposure
        };

        // Confidence inversely related to entropy
        self.confidence = (1.0 - metrics.entropy_production.tanh()).max(0.1);

        // Risk adjustment
        let risk_adj = match metrics.regime {
            ThermoRegime::Equilibrium => 1.0,
            ThermoRegime::NearEquilibrium => 0.8,
            ThermoRegime::Dissipative => 0.5,
            ThermoRegime::HighEnergy => 0.2,
        };

        self.last_metrics = Some(metrics.clone());

        Ok(
            SignalContribution::new("thermo", direction, self.confidence)
                .with_risk_adjustment(risk_adj)
                .with_metadata("entropy_production", metrics.entropy_production)
                .with_metadata("entropy_trend", metrics.entropy_trend)
                .with_metadata("effective_temperature", metrics.effective_temperature),
        )
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn reset(&mut self) {
        self.last_metrics = None;
        self.confidence = 0.5;
    }
}

// ============================================================================
// Risk Integration
// ============================================================================

/// Integration for hyperphysics-risk codependent models
pub struct RiskIntegration {
    adapter: RiskAdapter,
    last_metrics: Option<RiskMetrics>,
    confidence: f64,
}

impl RiskIntegration {
    /// Create new risk integration
    pub fn new(config: RiskConfig) -> Result<Self> {
        Ok(Self {
            adapter: RiskAdapter::new(config)?,
            last_metrics: None,
            confidence: 0.5,
        })
    }

    /// Get Kelly fraction
    pub fn kelly_fraction(&self) -> f64 {
        self.last_metrics
            .as_ref()
            .map(|m| m.kelly_fraction)
            .unwrap_or(0.0)
    }

    /// Get risk level
    pub fn risk_level(&self) -> Option<RiskLevel> {
        self.last_metrics.as_ref().map(|m| m.level)
    }

    /// Get VaR
    pub fn var(&self) -> f64 {
        self.last_metrics.as_ref().map(|m| m.var).unwrap_or(0.0)
    }
}

#[async_trait]
impl HyperPhysicsIntegration for RiskIntegration {
    fn name(&self) -> &'static str {
        "risk"
    }

    async fn process(&mut self, prices: &[f64], _volumes: &[f64]) -> Result<SignalContribution> {
        // Calculate returns
        if prices.len() < 2 {
            return Err(BridgeError::InsufficientData {
                required: 2,
                available: prices.len(),
            });
        }

        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

        let metrics = self.adapter.calculate_risk_metrics(&returns, &[])?;

        // Risk integration provides position sizing, not direction
        // Kelly criterion provides optimal fraction
        let direction = metrics.kelly_fraction; // Already -1 to 1

        // Confidence based on risk stability
        self.confidence = match metrics.level {
            RiskLevel::Low => 0.9,
            RiskLevel::Medium => 0.7,
            RiskLevel::High => 0.4,
            RiskLevel::Extreme => 0.1,
        };

        // Risk adjustment from risk level
        let risk_adj = match metrics.level {
            RiskLevel::Low => 1.0,
            RiskLevel::Medium => 0.6,
            RiskLevel::High => 0.3,
            RiskLevel::Extreme => 0.05,
        };

        self.last_metrics = Some(metrics.clone());

        Ok(SignalContribution::new("risk", direction, self.confidence)
            .with_risk_adjustment(risk_adj)
            .with_metadata("var", metrics.var)
            .with_metadata("cvar", metrics.cvar)
            .with_metadata("max_drawdown", metrics.max_drawdown)
            .with_metadata("kelly_fraction", metrics.kelly_fraction))
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn reset(&mut self) {
        self.last_metrics = None;
        self.confidence = 0.5;
    }
}

// ============================================================================
// P-Bit Integration
// ============================================================================

/// Integration for probabilistic signal uncertainty quantification
pub struct PbitIntegration {
    adapter: PbitAdapter,
    last_metrics: Option<UncertaintyMetrics>,
    confidence: f64,
}

impl PbitIntegration {
    /// Create new P-bit integration
    pub fn new(config: PbitConfig) -> Result<Self> {
        Ok(Self {
            adapter: PbitAdapter::new(config)?,
            last_metrics: None,
            confidence: 0.5,
        })
    }

    /// Get uncertainty estimate
    pub fn uncertainty(&self) -> f64 {
        self.last_metrics
            .as_ref()
            .map(|m| m.uncertainty)
            .unwrap_or(1.0)
    }

    /// Quantify signal uncertainty
    pub fn quantify_signal(&mut self, signal: f64, confidence: f64) -> Result<UncertaintyMetrics> {
        self.adapter.quantify_uncertainty(signal, confidence)
    }
}

#[async_trait]
impl HyperPhysicsIntegration for PbitIntegration {
    fn name(&self) -> &'static str {
        "pbit"
    }

    async fn process(&mut self, prices: &[f64], _volumes: &[f64]) -> Result<SignalContribution> {
        if prices.len() < 2 {
            return Err(BridgeError::InsufficientData {
                required: 2,
                available: prices.len(),
            });
        }

        // Calculate simple momentum signal
        let momentum =
            (prices.last().unwrap() - prices.first().unwrap()) / prices.first().unwrap();
        let normalized_signal = momentum.tanh();

        // Quantify uncertainty in signal
        let base_confidence = 0.6;
        let metrics = self
            .adapter
            .quantify_uncertainty(normalized_signal, base_confidence)?;

        // Direction comes from mean estimate
        let direction = metrics.mean_estimate;

        // Confidence inversely related to uncertainty
        self.confidence = (1.0 - metrics.uncertainty.tanh()).max(0.1);

        // Risk adjustment based on entropy (higher entropy = more uncertain)
        let risk_adj = (1.0 - metrics.entropy / 5.0).clamp(0.1, 1.0);

        self.last_metrics = Some(metrics.clone());

        Ok(SignalContribution::new("pbit", direction, self.confidence)
            .with_risk_adjustment(risk_adj)
            .with_metadata("uncertainty", metrics.uncertainty)
            .with_metadata("entropy", metrics.entropy)
            .with_metadata("pattern_stability", metrics.pattern_stability))
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn reset(&mut self) {
        self.last_metrics = None;
        self.confidence = 0.5;
    }
}

// ============================================================================
// Quantum Integration
// ============================================================================

/// Integration for quantum pattern detection
pub struct QuantumIntegration {
    adapter: QuantumAdapter,
    last_metrics: Option<QuantumPatternMetrics>,
    confidence: f64,
}

impl QuantumIntegration {
    /// Create new quantum integration
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            adapter: QuantumAdapter::new(config)?,
            last_metrics: None,
            confidence: 0.5,
        })
    }

    /// Get pattern strength
    pub fn pattern_strength(&self) -> f64 {
        self.last_metrics
            .as_ref()
            .map(|m| m.pattern_strength)
            .unwrap_or(0.0)
    }

    /// Get quantum coherence
    pub fn coherence(&self) -> f64 {
        self.last_metrics
            .as_ref()
            .map(|m| m.coherence)
            .unwrap_or(0.0)
    }
}

#[async_trait]
impl HyperPhysicsIntegration for QuantumIntegration {
    fn name(&self) -> &'static str {
        "quantum"
    }

    async fn process(&mut self, prices: &[f64], _volumes: &[f64]) -> Result<SignalContribution> {
        if prices.len() < 4 {
            return Err(BridgeError::InsufficientData {
                required: 4,
                available: prices.len(),
            });
        }

        // Detect quantum patterns
        let metrics = self.adapter.detect_patterns(prices)?;

        // Direction from pattern strength (already -1 to 1)
        let direction = metrics.pattern_strength;

        // Confidence from coherence
        self.confidence = metrics.coherence.abs().min(1.0);

        // Risk adjustment from entanglement (higher = more correlated features)
        let risk_adj = (1.0 - metrics.entanglement.abs()).clamp(0.3, 1.0);

        self.last_metrics = Some(metrics.clone());

        Ok(
            SignalContribution::new("quantum", direction, self.confidence)
                .with_risk_adjustment(risk_adj)
                .with_metadata("pattern_strength", metrics.pattern_strength)
                .with_metadata("entanglement", metrics.entanglement)
                .with_metadata("coherence", metrics.coherence)
                .with_metadata("measurement_entropy", metrics.measurement_entropy),
        )
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn reset(&mut self) {
        self.last_metrics = None;
        self.confidence = 0.5;
    }
}

// ============================================================================
// Syntergic Integration
// ============================================================================

/// Integration for coherence field analysis
pub struct SyntergicIntegration {
    adapter: SyntergicAdapter,
    last_metrics: Option<SyntergicMetrics>,
    confidence: f64,
}

impl SyntergicIntegration {
    /// Create new syntergic integration
    pub fn new(config: SyntergicConfig) -> Result<Self> {
        Ok(Self {
            adapter: SyntergicAdapter::new(config)?,
            last_metrics: None,
            confidence: 0.5,
        })
    }

    /// Get current coherence
    pub fn coherence(&self) -> f64 {
        self.adapter.current_coherence()
    }

    /// Get sync state
    pub fn sync_state(&self) -> Option<SyncState> {
        self.last_metrics.as_ref().map(|m| m.sync_state)
    }

    /// Check if unity achieved
    pub fn unity_achieved(&self) -> bool {
        self.last_metrics
            .as_ref()
            .map(|m| m.unity_achieved)
            .unwrap_or(false)
    }
}

#[async_trait]
impl HyperPhysicsIntegration for SyntergicIntegration {
    fn name(&self) -> &'static str {
        "syntergic"
    }

    async fn process(&mut self, prices: &[f64], volumes: &[f64]) -> Result<SignalContribution> {
        let market_state = MarketStateVector::from_market_data(prices, volumes);
        let metrics = self.adapter.analyze_coherence(&market_state)?;

        // Direction based on sync state
        let direction = match metrics.sync_state {
            SyncState::Unity => 0.5,
            SyncState::Converging => 0.3 + metrics.coherence_trend,
            SyncState::Partial => 0.1,
            SyncState::Diverging => -0.2,
            SyncState::Chaotic => -0.4,
        };

        // Confidence from coherence
        self.confidence = metrics.coherence;

        // Risk adjustment from sync state
        let risk_adj = match metrics.sync_state {
            SyncState::Unity => 1.0,
            SyncState::Converging => 0.8,
            SyncState::Partial => 0.5,
            SyncState::Diverging => 0.3,
            SyncState::Chaotic => 0.1,
        };

        self.last_metrics = Some(metrics.clone());

        Ok(
            SignalContribution::new("syntergic", direction, self.confidence)
                .with_risk_adjustment(risk_adj)
                .with_metadata("coherence", metrics.coherence)
                .with_metadata("field_energy", metrics.field_energy)
                .with_metadata("coherence_trend", metrics.coherence_trend),
        )
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn reset(&mut self) {
        self.last_metrics = None;
        self.confidence = 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_autopoiesis_integration() {
        let config = AutopoiesisConfig::default();
        let mut integration = AutopoiesisIntegration::new(config);

        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let volumes: Vec<f64> = vec![1_000_000.0; 100];

        let signal = integration.process(&prices, &volumes).await.unwrap();
        assert!(signal.direction >= -1.0 && signal.direction <= 1.0);
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_thermo_integration() {
        let config = ThermoConfig::default();
        let mut integration = ThermoIntegration::new(config);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let volumes: Vec<f64> = vec![1_000_000.0; 50];

        let signal = integration.process(&prices, &volumes).await.unwrap();
        assert!(signal.direction >= -1.0 && signal.direction <= 1.0);
    }
}
