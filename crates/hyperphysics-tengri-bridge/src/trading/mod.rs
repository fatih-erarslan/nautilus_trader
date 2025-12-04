//! Unified trading system combining all HyperPhysics integrations
//!
//! Provides a single entry point for trading signal generation using
//! autopoiesis, consciousness, thermodynamics, risk, p-bits, quantum,
//! and syntergic field analysis.

use crate::adapters::MarketStateVector;
use crate::error::{BridgeError, Result};
use crate::integrations::{
    AutopoiesisConfig, AutopoiesisIntegration, ConsciousnessConfig, ConsciousnessIntegration,
    HyperPhysicsIntegration, PbitConfig, PbitIntegration, QuantumConfig, QuantumIntegration,
    RiskConfig, RiskIntegration, SignalContribution, SyntergicConfig, SyntergicIntegration,
    ThermoConfig, ThermoIntegration,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, info, warn};

// ============================================================================
// Market Data
// ============================================================================

/// Market data for trading analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Price history
    pub prices: Vec<f64>,
    /// Volume history
    pub volumes: Vec<f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Symbol (optional)
    pub symbol: Option<String>,
}

impl MarketData {
    /// Create new market data
    pub fn new(prices: Vec<f64>, volumes: Vec<f64>) -> Self {
        Self {
            prices,
            volumes,
            timestamp: Utc::now(),
            symbol: None,
        }
    }

    /// Create with symbol
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Calculate returns
    pub fn returns(&self) -> Vec<f64> {
        self.prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate volatility
    pub fn volatility(&self) -> f64 {
        let returns = self.returns();
        if returns.is_empty() {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    }

    /// Get market state vector for consciousness/syntergic analysis
    pub fn to_state_vector(&self) -> MarketStateVector {
        MarketStateVector::from_market_data(&self.prices, &self.volumes)
    }
}

// ============================================================================
// Trading Configuration
// ============================================================================

/// Configuration for HyperPhysics trading system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Autopoiesis configuration
    pub autopoiesis: AutopoiesisConfig,
    /// Consciousness configuration
    pub consciousness: ConsciousnessConfig,
    /// Thermo configuration
    pub thermo: ThermoConfig,
    /// Risk configuration
    pub risk: RiskConfig,
    /// P-bit configuration
    pub pbit: PbitConfig,
    /// Quantum configuration
    pub quantum: QuantumConfig,
    /// Syntergic configuration
    pub syntergic: SyntergicConfig,
    /// Integration weights
    pub weights: IntegrationWeights,
    /// Signal generation settings
    pub signal_settings: SignalSettings,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            autopoiesis: AutopoiesisConfig::default(),
            consciousness: ConsciousnessConfig::default(),
            thermo: ThermoConfig::default(),
            risk: RiskConfig::default(),
            pbit: PbitConfig::default(),
            quantum: QuantumConfig::default(),
            syntergic: SyntergicConfig::default(),
            weights: IntegrationWeights::default(),
            signal_settings: SignalSettings::default(),
        }
    }
}

/// Weights for combining integration signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationWeights {
    /// Autopoiesis weight
    pub autopoiesis: f64,
    /// Consciousness weight
    pub consciousness: f64,
    /// Thermo weight
    pub thermo: f64,
    /// Risk weight
    pub risk: f64,
    /// P-bit weight
    pub pbit: f64,
    /// Quantum weight
    pub quantum: f64,
    /// Syntergic weight
    pub syntergic: f64,
}

impl Default for IntegrationWeights {
    fn default() -> Self {
        Self {
            autopoiesis: 0.20,
            consciousness: 0.15,
            thermo: 0.15,
            risk: 0.20,
            pbit: 0.10,
            quantum: 0.10,
            syntergic: 0.10,
        }
    }
}

impl IntegrationWeights {
    /// Normalize weights to sum to 1
    pub fn normalized(&self) -> Self {
        let sum = self.autopoiesis
            + self.consciousness
            + self.thermo
            + self.risk
            + self.pbit
            + self.quantum
            + self.syntergic;

        if sum <= 0.0 {
            return Self::default();
        }

        Self {
            autopoiesis: self.autopoiesis / sum,
            consciousness: self.consciousness / sum,
            thermo: self.thermo / sum,
            risk: self.risk / sum,
            pbit: self.pbit / sum,
            quantum: self.quantum / sum,
            syntergic: self.syntergic / sum,
        }
    }

    /// Get weight for integration by name
    pub fn get(&self, name: &str) -> f64 {
        match name {
            "autopoiesis" => self.autopoiesis,
            "consciousness" => self.consciousness,
            "thermo" => self.thermo,
            "risk" => self.risk,
            "pbit" => self.pbit,
            "quantum" => self.quantum,
            "syntergic" => self.syntergic,
            _ => 0.0,
        }
    }
}

/// Signal generation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSettings {
    /// Minimum confidence for signal generation
    pub min_confidence: f64,
    /// Minimum absolute direction for signal
    pub min_direction: f64,
    /// Maximum position size (0-1)
    pub max_position_size: f64,
    /// History window for signal averaging
    pub history_window: usize,
    /// Enable emergency exit signals
    pub enable_emergency_exit: bool,
    /// Emergency exit threshold (risk level)
    pub emergency_exit_threshold: f64,
}

impl Default for SignalSettings {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            min_direction: 0.1,
            max_position_size: 1.0,
            history_window: 20,
            enable_emergency_exit: true,
            emergency_exit_threshold: 0.9,
        }
    }
}

// ============================================================================
// Trading Signals
// ============================================================================

/// Signal direction enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalDirection {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold / neutral
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
    /// Reduce risk / exit positions
    ReduceRisk,
    /// Emergency exit
    EmergencyExit,
}

impl SignalDirection {
    /// Convert to position multiplier (-1 to 1)
    pub fn to_position(&self) -> f64 {
        match self {
            SignalDirection::StrongBuy => 1.0,
            SignalDirection::Buy => 0.5,
            SignalDirection::Hold => 0.0,
            SignalDirection::Sell => -0.5,
            SignalDirection::StrongSell => -1.0,
            SignalDirection::ReduceRisk => 0.0,
            SignalDirection::EmergencyExit => 0.0,
        }
    }

    /// Check if signal requires position exit
    pub fn requires_exit(&self) -> bool {
        matches!(self, SignalDirection::ReduceRisk | SignalDirection::EmergencyExit)
    }
}

/// Market regime enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Trending market
    Trending,
    /// Mean-reverting market
    MeanReverting,
    /// Transitional / uncertain
    Transitional,
    /// High volatility / chaotic
    Chaotic,
    /// Low volatility / stable
    Stable,
}

/// HyperPhysics trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperPhysicsSignal {
    /// Signal direction
    pub direction: SignalDirection,
    /// Raw direction value (-1 to 1)
    pub direction_value: f64,
    /// Overall confidence (0-1)
    pub confidence: f64,
    /// Recommended position size (0-1)
    pub position_size: f64,
    /// Current market regime
    pub regime: MarketRegime,
    /// Risk level (0-1)
    pub risk_level: f64,
    /// Individual integration contributions
    pub contributions: HashMap<String, SignalContribution>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl HyperPhysicsSignal {
    /// Create hold signal
    pub fn hold() -> Self {
        Self {
            direction: SignalDirection::Hold,
            direction_value: 0.0,
            confidence: 0.0,
            position_size: 0.0,
            regime: MarketRegime::Transitional,
            risk_level: 0.5,
            contributions: HashMap::new(),
            timestamp: Utc::now(),
        }
    }
}

/// Trading result with full analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingResult {
    /// Generated signal
    pub signal: HyperPhysicsSignal,
    /// Autopoietic health (0-1)
    pub autopoietic_health: f64,
    /// Consciousness Φ value
    pub phi: f64,
    /// Entropy production
    pub entropy: f64,
    /// VaR
    pub var: f64,
    /// Uncertainty estimate
    pub uncertainty: f64,
    /// Quantum coherence
    pub quantum_coherence: f64,
    /// Syntergic coherence
    pub syntergic_coherence: f64,
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// HyperPhysics Trading System
// ============================================================================

/// Unified HyperPhysics trading system
pub struct HyperPhysicsTradingSystem {
    /// Configuration
    config: TradingConfig,
    /// Autopoiesis integration
    autopoiesis: AutopoiesisIntegration,
    /// Consciousness integration
    consciousness: ConsciousnessIntegration,
    /// Thermo integration
    thermo: ThermoIntegration,
    /// Risk integration
    risk: RiskIntegration,
    /// P-bit integration
    pbit: PbitIntegration,
    /// Quantum integration
    quantum: QuantumIntegration,
    /// Syntergic integration
    syntergic: SyntergicIntegration,
    /// Signal history
    signal_history: VecDeque<HyperPhysicsSignal>,
    /// Normalized weights
    weights: IntegrationWeights,
}

impl HyperPhysicsTradingSystem {
    /// Create new HyperPhysics trading system
    pub fn new(config: TradingConfig) -> Result<Self> {
        info!("Initializing HyperPhysics Trading System");

        let autopoiesis = AutopoiesisIntegration::new(config.autopoiesis.clone());
        let consciousness = ConsciousnessIntegration::new(config.consciousness.clone())?;
        let thermo = ThermoIntegration::new(config.thermo.clone());
        let risk = RiskIntegration::new(config.risk.clone())?;
        let pbit = PbitIntegration::new(config.pbit.clone())?;
        let quantum = QuantumIntegration::new(config.quantum.clone())?;
        let syntergic = SyntergicIntegration::new(config.syntergic.clone())?;

        let weights = config.weights.normalized();

        info!("HyperPhysics Trading System initialized with {} integrations", 7);

        Ok(Self {
            config,
            autopoiesis,
            consciousness,
            thermo,
            risk,
            pbit,
            quantum,
            syntergic,
            signal_history: VecDeque::with_capacity(100),
            weights,
        })
    }

    /// Process market data and generate trading signal
    pub async fn process_market_data(&mut self, market_data: &MarketData) -> Result<TradingResult> {
        debug!("Processing market data with {} prices", market_data.prices.len());

        if market_data.prices.len() < 10 {
            return Err(BridgeError::InsufficientData {
                required: 10,
                available: market_data.prices.len(),
            });
        }

        // Process all integrations
        let mut contributions = HashMap::new();

        // Autopoiesis
        match self
            .autopoiesis
            .process(&market_data.prices, &market_data.volumes)
            .await
        {
            Ok(contrib) => {
                contributions.insert("autopoiesis".to_string(), contrib);
            }
            Err(e) => warn!("Autopoiesis integration failed: {}", e),
        }

        // Consciousness
        match self
            .consciousness
            .process(&market_data.prices, &market_data.volumes)
            .await
        {
            Ok(contrib) => {
                contributions.insert("consciousness".to_string(), contrib);
            }
            Err(e) => warn!("Consciousness integration failed: {}", e),
        }

        // Thermo
        match self
            .thermo
            .process(&market_data.prices, &market_data.volumes)
            .await
        {
            Ok(contrib) => {
                contributions.insert("thermo".to_string(), contrib);
            }
            Err(e) => warn!("Thermo integration failed: {}", e),
        }

        // Risk
        match self
            .risk
            .process(&market_data.prices, &market_data.volumes)
            .await
        {
            Ok(contrib) => {
                contributions.insert("risk".to_string(), contrib);
            }
            Err(e) => warn!("Risk integration failed: {}", e),
        }

        // P-bit
        match self
            .pbit
            .process(&market_data.prices, &market_data.volumes)
            .await
        {
            Ok(contrib) => {
                contributions.insert("pbit".to_string(), contrib);
            }
            Err(e) => warn!("P-bit integration failed: {}", e),
        }

        // Quantum
        match self
            .quantum
            .process(&market_data.prices, &market_data.volumes)
            .await
        {
            Ok(contrib) => {
                contributions.insert("quantum".to_string(), contrib);
            }
            Err(e) => warn!("Quantum integration failed: {}", e),
        }

        // Syntergic
        match self
            .syntergic
            .process(&market_data.prices, &market_data.volumes)
            .await
        {
            Ok(contrib) => {
                contributions.insert("syntergic".to_string(), contrib);
            }
            Err(e) => warn!("Syntergic integration failed: {}", e),
        }

        // Combine signals
        let signal = self.combine_signals(&contributions)?;

        // Store in history
        self.signal_history.push_back(signal.clone());
        if self.signal_history.len() > self.config.signal_settings.history_window {
            self.signal_history.pop_front();
        }

        // Build result
        let result = TradingResult {
            signal,
            autopoietic_health: self.autopoiesis.health(),
            phi: self.consciousness.phi(),
            entropy: self.thermo.entropy(),
            var: self.risk.var(),
            uncertainty: self.pbit.uncertainty(),
            quantum_coherence: self.quantum.coherence(),
            syntergic_coherence: self.syntergic.coherence(),
            timestamp: Utc::now(),
        };

        debug!(
            "Generated signal: {:?}, confidence: {:.4}",
            result.signal.direction, result.signal.confidence
        );

        Ok(result)
    }

    /// Combine individual integration signals into unified signal
    fn combine_signals(
        &self,
        contributions: &HashMap<String, SignalContribution>,
    ) -> Result<HyperPhysicsSignal> {
        if contributions.is_empty() {
            return Ok(HyperPhysicsSignal::hold());
        }

        // Calculate weighted direction
        let mut weighted_direction: f64 = 0.0;
        let mut weighted_confidence: f64 = 0.0;
        let mut total_weight: f64 = 0.0;
        let mut min_risk_adjustment: f64 = 1.0;

        for (name, contrib) in contributions {
            let weight = self.weights.get(name);
            if weight > 0.0 {
                weighted_direction += contrib.direction * contrib.confidence * weight;
                weighted_confidence += contrib.confidence * weight;
                total_weight += weight;
                min_risk_adjustment = min_risk_adjustment.min(contrib.risk_adjustment);
            }
        }

        if total_weight <= 0.0 {
            return Ok(HyperPhysicsSignal::hold());
        }

        let direction_value = weighted_direction / total_weight;
        let confidence = weighted_confidence / total_weight;

        // Determine signal direction
        let direction = self.classify_direction(direction_value, confidence, min_risk_adjustment);

        // Calculate position size
        let position_size = self.calculate_position_size(direction_value, confidence, min_risk_adjustment);

        // Determine market regime
        let regime = self.determine_regime();

        // Calculate risk level
        let risk_level = 1.0 - min_risk_adjustment;

        Ok(HyperPhysicsSignal {
            direction,
            direction_value,
            confidence,
            position_size,
            regime,
            risk_level,
            contributions: contributions.clone(),
            timestamp: Utc::now(),
        })
    }

    /// Classify direction into signal enum
    fn classify_direction(
        &self,
        direction: f64,
        confidence: f64,
        risk_adj: f64,
    ) -> SignalDirection {
        let settings = &self.config.signal_settings;

        // Emergency exit check
        if settings.enable_emergency_exit && risk_adj < 1.0 - settings.emergency_exit_threshold {
            return SignalDirection::EmergencyExit;
        }

        // Reduce risk if adjustment is low
        if risk_adj < 0.3 {
            return SignalDirection::ReduceRisk;
        }

        // Check minimum thresholds
        if confidence < settings.min_confidence || direction.abs() < settings.min_direction {
            return SignalDirection::Hold;
        }

        // Classify based on direction
        if direction > 0.4 {
            SignalDirection::StrongBuy
        } else if direction > 0.15 {
            SignalDirection::Buy
        } else if direction < -0.4 {
            SignalDirection::StrongSell
        } else if direction < -0.15 {
            SignalDirection::Sell
        } else {
            SignalDirection::Hold
        }
    }

    /// Calculate recommended position size
    fn calculate_position_size(&self, direction: f64, confidence: f64, risk_adj: f64) -> f64 {
        let settings = &self.config.signal_settings;

        // Base position from direction and confidence
        let base = direction.abs() * confidence;

        // Apply risk adjustment
        let adjusted = base * risk_adj;

        // Apply Kelly fraction from risk integration
        let kelly = self.risk.kelly_fraction().abs();
        let kelly_adjusted = adjusted * (0.5 + 0.5 * kelly);

        // Clamp to max position size
        kelly_adjusted.clamp(0.0, settings.max_position_size)
    }

    /// Determine current market regime
    fn determine_regime(&self) -> MarketRegime {
        // Use autopoiesis regime as primary indicator
        if let Some(regime) = self.autopoiesis.current_regime() {
            return match regime {
                crate::adapters::AutopoiesisMarketRegime::Stable => MarketRegime::Stable,
                crate::adapters::AutopoiesisMarketRegime::Normal => MarketRegime::Trending,
                crate::adapters::AutopoiesisMarketRegime::Transitional => MarketRegime::Transitional,
                crate::adapters::AutopoiesisMarketRegime::Degraded => MarketRegime::MeanReverting,
                crate::adapters::AutopoiesisMarketRegime::Chaotic => MarketRegime::Chaotic,
            };
        }

        // Fall back to thermo regime
        if let Some(regime) = self.thermo.regime() {
            return match regime {
                crate::adapters::ThermoRegime::Equilibrium => MarketRegime::Stable,
                crate::adapters::ThermoRegime::NearEquilibrium => MarketRegime::MeanReverting,
                crate::adapters::ThermoRegime::Dissipative => MarketRegime::Transitional,
                crate::adapters::ThermoRegime::HighEnergy => MarketRegime::Chaotic,
            };
        }

        MarketRegime::Transitional
    }

    /// Get current autopoietic health
    pub fn autopoietic_health(&self) -> f64 {
        self.autopoiesis.health()
    }

    /// Get current Φ value
    pub fn phi(&self) -> f64 {
        self.consciousness.phi()
    }

    /// Get current entropy
    pub fn entropy(&self) -> f64 {
        self.thermo.entropy()
    }

    /// Get Kelly fraction
    pub fn kelly_fraction(&self) -> f64 {
        self.risk.kelly_fraction()
    }

    /// Check if near bifurcation
    pub fn near_bifurcation(&self) -> bool {
        self.autopoiesis.near_bifurcation()
    }

    /// Get syntergic unity status
    pub fn unity_achieved(&self) -> bool {
        self.syntergic.unity_achieved()
    }

    /// Get signal history
    pub fn signal_history(&self) -> &VecDeque<HyperPhysicsSignal> {
        &self.signal_history
    }

    /// Reset all integrations
    pub fn reset(&mut self) {
        self.autopoiesis.reset();
        self.consciousness.reset();
        self.thermo.reset();
        self.risk.reset();
        self.pbit.reset();
        self.quantum.reset();
        self.syntergic.reset();
        self.signal_history.clear();
        info!("HyperPhysics Trading System reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trading_system_creation() {
        let config = TradingConfig::default();
        let system = HyperPhysicsTradingSystem::new(config);
        assert!(system.is_ok());
    }

    #[tokio::test]
    async fn test_process_market_data() {
        let config = TradingConfig::default();
        let mut system = HyperPhysicsTradingSystem::new(config).unwrap();

        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let volumes: Vec<f64> = vec![1_000_000.0; 100];
        let market_data = MarketData::new(prices, volumes);

        let result = system.process_market_data(&market_data).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.signal.confidence >= 0.0 && result.signal.confidence <= 1.0);
        assert!(result.signal.position_size >= 0.0 && result.signal.position_size <= 1.0);
    }

    #[test]
    fn test_market_data_returns() {
        let prices = vec![100.0, 101.0, 102.0, 101.0];
        let market_data = MarketData::new(prices, vec![1000.0; 4]);

        let returns = market_data.returns();
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_integration_weights_normalization() {
        let weights = IntegrationWeights {
            autopoiesis: 2.0,
            consciousness: 1.0,
            thermo: 1.0,
            risk: 2.0,
            pbit: 1.0,
            quantum: 1.0,
            syntergic: 1.0,
        };

        let normalized = weights.normalized();
        let sum = normalized.autopoiesis
            + normalized.consciousness
            + normalized.thermo
            + normalized.risk
            + normalized.pbit
            + normalized.quantum
            + normalized.syntergic;

        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_signal_direction() {
        assert_eq!(SignalDirection::StrongBuy.to_position(), 1.0);
        assert_eq!(SignalDirection::StrongSell.to_position(), -1.0);
        assert_eq!(SignalDirection::Hold.to_position(), 0.0);
        assert!(SignalDirection::EmergencyExit.requires_exit());
    }
}
