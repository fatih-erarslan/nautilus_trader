//! CWTS Coordinator - Unified Complex Weighted Trading System Orchestration
//!
//! This module provides centralized coordination across all CWTS subsystems:
//! - Autopoiesis (self-organization, boundary maintenance, emergence)
//! - Game Theory (Nash equilibrium, multi-agent risk, strategic positioning)
//! - Physics Simulation (market-physics bridge, order flow dynamics)
//! - Nautilus (execution risk, backtest validation, live trading guards)
//! - Neural Forecasting (ensemble prediction, conformal uncertainty)
//!
//! ## Consensus Protocol
//!
//! The coordinator implements a weighted consensus mechanism where each
//! subsystem contributes risk assessments that are aggregated based on:
//! - Subsystem confidence (entropy-weighted)
//! - Recent accuracy (rolling validation)
//! - Market regime relevance
//!
//! ## Scientific References
//!
//! - Kahneman & Tversky (1979): "Prospect Theory" for decision weighting
//! - Arrow (1951): "Social Choice and Individual Values" for aggregation
//! - Condorcet (1785): Jury theorem for collective wisdom

use crate::core::{Portfolio, RiskLevel, RiskDecision, MarketRegime, Timestamp, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::bft_consensus::{
    BftConsensusConfig, BftConsensusEngine, BftConsensusResult,
    ConsensusProof, ProposalContext,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for CWTS coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CWTSConfig {
    /// Enable autopoiesis integration.
    #[cfg(feature = "cwts-autopoiesis")]
    pub enable_autopoiesis: bool,

    /// Enable game theory integration.
    #[cfg(feature = "cwts-game-theory")]
    pub enable_game_theory: bool,

    /// Enable physics simulation.
    #[cfg(feature = "cwts-physics")]
    pub enable_physics: bool,

    /// Enable nautilus execution bridge.
    #[cfg(feature = "cwts-nautilus")]
    pub enable_nautilus: bool,

    /// Enable neural forecasting.
    #[cfg(feature = "cwts-neural")]
    pub enable_neural: bool,

    /// Weight for autopoiesis risk contribution [0, 1].
    pub weight_autopoiesis: f64,

    /// Weight for game theory risk contribution [0, 1].
    pub weight_game_theory: f64,

    /// Weight for physics simulation contribution [0, 1].
    pub weight_physics: f64,

    /// Weight for nautilus execution risk [0, 1].
    pub weight_nautilus: f64,

    /// Weight for neural forecasting contribution [0, 1].
    pub weight_neural: f64,

    /// Consensus threshold for decision (0.5 = simple majority).
    pub consensus_threshold: f64,

    /// Maximum latency budget for full CWTS check (microseconds).
    pub max_latency_us: u64,

    /// Enable adaptive weight adjustment based on accuracy.
    pub adaptive_weights: bool,

    /// Window size for rolling accuracy calculation.
    pub accuracy_window: usize,

    /// Enable Byzantine Fault Tolerance consensus.
    pub enable_bft: bool,

    /// BFT configuration (used when enable_bft is true).
    pub bft_config: BftConsensusConfig,
}

impl Default for CWTSConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "cwts-autopoiesis")]
            enable_autopoiesis: true,
            #[cfg(feature = "cwts-game-theory")]
            enable_game_theory: true,
            #[cfg(feature = "cwts-physics")]
            enable_physics: true,
            #[cfg(feature = "cwts-nautilus")]
            enable_nautilus: true,
            #[cfg(feature = "cwts-neural")]
            enable_neural: true,
            weight_autopoiesis: 0.15,
            weight_game_theory: 0.20,
            weight_physics: 0.15,
            weight_nautilus: 0.25,
            weight_neural: 0.25,
            consensus_threshold: 0.6,
            max_latency_us: 500,
            adaptive_weights: true,
            accuracy_window: 100,
            enable_bft: false, // Disabled by default for backward compatibility
            bft_config: BftConsensusConfig::default(),
        }
    }
}

impl CWTSConfig {
    /// Create configuration with BFT enabled.
    pub fn with_bft() -> Self {
        Self {
            enable_bft: true,
            ..Default::default()
        }
    }

    /// Create configuration for production use.
    pub fn production() -> Self {
        Self {
            consensus_threshold: 0.65,
            max_latency_us: 250,
            ..Default::default()
        }
    }

    /// Create configuration for backtesting.
    pub fn backtest() -> Self {
        Self {
            max_latency_us: 10_000, // Relaxed for backtest
            ..Default::default()
        }
    }

    /// Validate configuration weights sum to 1.0.
    pub fn validate(&self) -> Result<(), &'static str> {
        let total = self.weight_autopoiesis
            + self.weight_game_theory
            + self.weight_physics
            + self.weight_nautilus
            + self.weight_neural;

        if (total - 1.0).abs() > 0.001 {
            return Err("CWTS weights must sum to 1.0");
        }

        if self.consensus_threshold < 0.0 || self.consensus_threshold > 1.0 {
            return Err("Consensus threshold must be in [0, 1]");
        }

        Ok(())
    }
}

// ============================================================================
// Subsystem Risk Assessment
// ============================================================================

/// Risk assessment from a single CWTS subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemRisk {
    /// Subsystem identifier.
    pub subsystem: SubsystemId,
    /// Risk level assessment.
    pub risk_level: RiskLevel,
    /// Confidence in assessment [0, 1].
    pub confidence: f64,
    /// Risk score [0, 1] where 1 is maximum risk.
    pub risk_score: f64,
    /// Recommended position adjustment factor.
    pub position_factor: f64,
    /// Human-readable reasoning.
    pub reasoning: String,
    /// Timestamp of assessment.
    pub timestamp: Timestamp,
    /// Latency of computation in nanoseconds.
    pub latency_ns: u64,
}

impl SubsystemRisk {
    /// Create a normal risk assessment.
    pub fn normal(subsystem: SubsystemId, confidence: f64, latency_ns: u64) -> Self {
        Self {
            subsystem,
            risk_level: RiskLevel::Normal,
            confidence,
            risk_score: 0.1,
            position_factor: 1.0,
            reasoning: "Normal conditions".to_string(),
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }

    /// Create an elevated risk assessment.
    pub fn elevated(
        subsystem: SubsystemId,
        confidence: f64,
        risk_score: f64,
        reason: impl Into<String>,
        latency_ns: u64,
    ) -> Self {
        Self {
            subsystem,
            risk_level: RiskLevel::Elevated,
            confidence,
            risk_score,
            position_factor: 0.8,
            reasoning: reason.into(),
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }

    /// Create a critical risk assessment.
    pub fn critical(
        subsystem: SubsystemId,
        confidence: f64,
        reason: impl Into<String>,
        latency_ns: u64,
    ) -> Self {
        Self {
            subsystem,
            risk_level: RiskLevel::Critical,
            confidence,
            risk_score: 0.9,
            position_factor: 0.1,
            reasoning: reason.into(),
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }
}

/// Identifier for CWTS subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubsystemId {
    /// Autopoiesis subsystem.
    Autopoiesis,
    /// Game theory subsystem.
    GameTheory,
    /// Physics simulation subsystem.
    Physics,
    /// Nautilus execution subsystem.
    Nautilus,
    /// Neural forecasting subsystem.
    Neural,
}

impl SubsystemId {
    /// Get short name for logging.
    pub const fn short_name(&self) -> &'static str {
        match self {
            Self::Autopoiesis => "AUTO",
            Self::GameTheory => "GAME",
            Self::Physics => "PHYS",
            Self::Nautilus => "NAUT",
            Self::Neural => "NEUR",
        }
    }
}

// ============================================================================
// Integrated Risk Metrics
// ============================================================================

/// Integrated risk metrics from all CWTS subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedRiskMetrics {
    /// Overall risk level (weighted consensus).
    pub overall_risk: RiskLevel,
    /// Overall risk score [0, 1].
    pub overall_score: f64,
    /// Consensus confidence [0, 1].
    pub consensus_confidence: f64,
    /// Whether consensus was reached.
    pub consensus_reached: bool,
    /// Individual subsystem assessments.
    pub subsystem_risks: Vec<SubsystemRisk>,
    /// Recommended position factor [0, 1].
    pub recommended_position_factor: f64,
    /// Total latency for all subsystems in nanoseconds.
    pub total_latency_ns: u64,
    /// Current market regime (if detected).
    pub market_regime: Option<MarketRegime>,
    /// Timestamp of integration.
    pub timestamp: Timestamp,
}

impl IntegratedRiskMetrics {
    /// Check if any subsystem flagged emergency.
    pub fn has_emergency(&self) -> bool {
        self.subsystem_risks
            .iter()
            .any(|r| r.risk_level == RiskLevel::Emergency)
    }

    /// Get highest risk level across subsystems.
    pub fn max_risk_level(&self) -> RiskLevel {
        self.subsystem_risks
            .iter()
            .map(|r| r.risk_level)
            .max()
            .unwrap_or(RiskLevel::Normal)
    }

    /// Get subsystem with highest risk.
    pub fn highest_risk_subsystem(&self) -> Option<&SubsystemRisk> {
        self.subsystem_risks
            .iter()
            .max_by(|a, b| a.risk_score.partial_cmp(&b.risk_score).unwrap())
    }
}

// ============================================================================
// CWTS Decision
// ============================================================================

/// Decision from CWTS coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CWTSDecision {
    /// Whether action is allowed.
    pub allowed: bool,
    /// Risk metrics that informed decision.
    pub metrics: IntegratedRiskMetrics,
    /// Primary reason for decision.
    pub reason: String,
    /// Recommended actions.
    pub recommendations: Vec<String>,
    /// Decision timestamp.
    pub timestamp: Timestamp,
}

impl CWTSDecision {
    /// Create approval decision.
    pub fn approve(metrics: IntegratedRiskMetrics) -> Self {
        Self {
            allowed: true,
            reason: "All CWTS subsystems approved".to_string(),
            recommendations: vec![],
            timestamp: Timestamp::now(),
            metrics,
        }
    }

    /// Create rejection decision.
    pub fn reject(
        metrics: IntegratedRiskMetrics,
        reason: impl Into<String>,
        recommendations: Vec<String>,
    ) -> Self {
        Self {
            allowed: false,
            reason: reason.into(),
            recommendations,
            timestamp: Timestamp::now(),
            metrics,
        }
    }

    /// Convert to core RiskDecision.
    pub fn to_risk_decision(&self) -> RiskDecision {
        RiskDecision {
            allowed: self.allowed,
            risk_level: self.metrics.overall_risk,
            reason: self.reason.clone(),
            size_adjustment: self.metrics.recommended_position_factor,
            timestamp: self.timestamp,
            latency_ns: self.metrics.total_latency_ns,
        }
    }
}

// ============================================================================
// CWTS Coordinator
// ============================================================================

/// Central coordinator for all CWTS subsystems.
pub struct CWTSCoordinator {
    /// Configuration.
    config: CWTSConfig,

    /// Autopoiesis adapter.
    #[cfg(feature = "cwts-autopoiesis")]
    autopoiesis: Option<super::autopoiesis_integration::AutopoiesisRiskAdapter>,

    /// Game theory adapter.
    #[cfg(feature = "cwts-game-theory")]
    game_theory: Option<super::game_theory_integration::GameTheoryRiskAdapter>,

    /// Physics adapter.
    #[cfg(feature = "cwts-physics")]
    physics: Option<super::physics_integration::PhysicsRiskAdapter>,

    /// Nautilus adapter.
    #[cfg(feature = "cwts-nautilus")]
    nautilus: Option<super::nautilus_integration::NautilusRiskAdapter>,

    /// Neural adapter.
    #[cfg(feature = "cwts-neural")]
    neural: Option<super::neural_integration::NeuralRiskAdapter>,

    /// Rolling accuracy history for adaptive weights.
    accuracy_history: HashMap<SubsystemId, Vec<bool>>,

    /// Adaptive weights (updated based on accuracy).
    adaptive_weights: HashMap<SubsystemId, f64>,

    /// BFT consensus engine (when enable_bft is true).
    bft_engine: Option<BftConsensusEngine>,

    /// Last BFT consensus result (for audit trail).
    last_bft_result: Option<BftConsensusResult>,
}

impl CWTSCoordinator {
    /// Create new CWTS coordinator with configuration.
    pub fn new(config: CWTSConfig) -> Result<Self, &'static str> {
        config.validate()?;

        let mut adaptive_weights = HashMap::new();
        adaptive_weights.insert(SubsystemId::Autopoiesis, config.weight_autopoiesis);
        adaptive_weights.insert(SubsystemId::GameTheory, config.weight_game_theory);
        adaptive_weights.insert(SubsystemId::Physics, config.weight_physics);
        adaptive_weights.insert(SubsystemId::Nautilus, config.weight_nautilus);
        adaptive_weights.insert(SubsystemId::Neural, config.weight_neural);

        // Initialize BFT engine if enabled
        let bft_engine = if config.enable_bft {
            Some(BftConsensusEngine::new(config.bft_config.clone()))
        } else {
            None
        };

        Ok(Self {
            config,
            #[cfg(feature = "cwts-autopoiesis")]
            autopoiesis: None,
            #[cfg(feature = "cwts-game-theory")]
            game_theory: None,
            #[cfg(feature = "cwts-physics")]
            physics: None,
            #[cfg(feature = "cwts-nautilus")]
            nautilus: None,
            #[cfg(feature = "cwts-neural")]
            neural: None,
            accuracy_history: HashMap::new(),
            adaptive_weights,
            bft_engine,
            last_bft_result: None,
        })
    }

    /// Create coordinator with default configuration.
    pub fn default_coordinator() -> Result<Self, &'static str> {
        Self::new(CWTSConfig::default())
    }

    /// Initialize autopoiesis subsystem.
    #[cfg(feature = "cwts-autopoiesis")]
    pub fn init_autopoiesis(
        &mut self,
        config: super::autopoiesis_integration::AutopoiesisConfig,
    ) -> Result<(), &'static str> {
        self.autopoiesis = Some(super::autopoiesis_integration::AutopoiesisRiskAdapter::new(config));
        Ok(())
    }

    /// Initialize game theory subsystem.
    #[cfg(feature = "cwts-game-theory")]
    pub fn init_game_theory(
        &mut self,
        config: super::game_theory_integration::GameTheoryConfig,
    ) -> Result<(), &'static str> {
        self.game_theory = Some(super::game_theory_integration::GameTheoryRiskAdapter::new(config));
        Ok(())
    }

    /// Initialize physics subsystem.
    #[cfg(feature = "cwts-physics")]
    pub fn init_physics(
        &mut self,
        config: super::physics_integration::PhysicsConfig,
    ) -> Result<(), &'static str> {
        self.physics = Some(super::physics_integration::PhysicsRiskAdapter::new(config));
        Ok(())
    }

    /// Initialize nautilus subsystem.
    #[cfg(feature = "cwts-nautilus")]
    pub fn init_nautilus(
        &mut self,
        config: super::nautilus_integration::NautilusConfig,
    ) -> Result<(), &'static str> {
        self.nautilus = Some(super::nautilus_integration::NautilusRiskAdapter::new(config));
        Ok(())
    }

    /// Initialize neural subsystem.
    #[cfg(feature = "cwts-neural")]
    pub fn init_neural(
        &mut self,
        config: super::neural_integration::NeuralConfig,
    ) -> Result<(), &'static str> {
        self.neural = Some(super::neural_integration::NeuralRiskAdapter::new(config));
        Ok(())
    }

    /// Perform full CWTS risk assessment.
    pub fn assess_risk(&self, portfolio: &Portfolio, symbol: Option<&Symbol>) -> IntegratedRiskMetrics {
        let start = Timestamp::now();
        let mut subsystem_risks = Vec::new();

        // Collect assessments from all enabled subsystems
        #[cfg(feature = "cwts-autopoiesis")]
        if self.config.enable_autopoiesis {
            if let Some(ref adapter) = self.autopoiesis {
                let risk = adapter.assess_portfolio_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-game-theory")]
        if self.config.enable_game_theory {
            if let Some(ref adapter) = self.game_theory {
                let risk = adapter.assess_strategic_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-physics")]
        if self.config.enable_physics {
            if let Some(ref adapter) = self.physics {
                let risk = adapter.assess_physics_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-nautilus")]
        if self.config.enable_nautilus {
            if let Some(ref adapter) = self.nautilus {
                let risk = adapter.assess_portfolio_execution_risk(portfolio, symbol);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-neural")]
        if self.config.enable_neural {
            if let Some(ref adapter) = self.neural {
                let risk = adapter.assess_forecast_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        // If no subsystems active, return normal
        if subsystem_risks.is_empty() {
            return IntegratedRiskMetrics {
                overall_risk: RiskLevel::Normal,
                overall_score: 0.0,
                consensus_confidence: 1.0,
                consensus_reached: true,
                subsystem_risks: vec![],
                recommended_position_factor: 1.0,
                total_latency_ns: 0,
                market_regime: None,
                timestamp: Timestamp::now(),
            };
        }

        // Aggregate using weighted consensus
        let (overall_score, consensus_confidence, consensus_reached) =
            self.weighted_consensus(&subsystem_risks);

        // Determine overall risk level from score
        let overall_risk = Self::score_to_risk_level(overall_score);

        // Calculate recommended position factor
        let recommended_position_factor = subsystem_risks
            .iter()
            .map(|r| {
                let weight = self.adaptive_weights
                    .get(&r.subsystem)
                    .copied()
                    .unwrap_or(0.2);
                r.position_factor * weight
            })
            .sum::<f64>()
            .clamp(0.0, 1.0);

        // Total latency
        let total_latency_ns = subsystem_risks.iter().map(|r| r.latency_ns).sum();

        // Detect market regime from subsystems
        let market_regime = self.detect_regime_from_subsystems(&subsystem_risks);

        IntegratedRiskMetrics {
            overall_risk,
            overall_score,
            consensus_confidence,
            consensus_reached,
            subsystem_risks,
            recommended_position_factor,
            total_latency_ns,
            market_regime,
            timestamp: Timestamp::now(),
        }
    }

    /// Make trading decision based on CWTS assessment.
    pub fn make_decision(&self, portfolio: &Portfolio, symbol: Option<&Symbol>) -> CWTSDecision {
        let metrics = self.assess_risk(portfolio, symbol);

        // Emergency override - any subsystem can veto
        if metrics.has_emergency() {
            let emergency_subsystem = metrics
                .subsystem_risks
                .iter()
                .find(|r| r.risk_level == RiskLevel::Emergency)
                .map(|r| r.subsystem.short_name())
                .unwrap_or("UNKNOWN");

            return CWTSDecision::reject(
                metrics,
                format!("Emergency halt triggered by {}", emergency_subsystem),
                vec![
                    "Halt all trading immediately".to_string(),
                    "Reduce exposure to zero".to_string(),
                    "Investigate root cause".to_string(),
                ],
            );
        }

        // Critical risk - reject unless overridden
        if metrics.overall_risk >= RiskLevel::Critical {
            return CWTSDecision::reject(
                metrics,
                "Critical risk level - trading restricted",
                vec![
                    "Reduce position sizes to 10%".to_string(),
                    "Close high-risk positions".to_string(),
                    "Wait for risk normalization".to_string(),
                ],
            );
        }

        // High risk - conditional approval with size reduction
        if metrics.overall_risk == RiskLevel::High {
            let mut decision = CWTSDecision::approve(metrics);
            decision.reason = "Approved with restrictions - high risk".to_string();
            decision.recommendations = vec![
                "Limit position sizes to 50%".to_string(),
                "Tighten stop-losses".to_string(),
            ];
            return decision;
        }

        // Consensus not reached - require human review
        if !metrics.consensus_reached {
            return CWTSDecision::reject(
                metrics,
                "Consensus not reached - subsystems disagree",
                vec![
                    "Review individual subsystem assessments".to_string(),
                    "Consider manual override if appropriate".to_string(),
                ],
            );
        }

        // Normal/Elevated - approve
        CWTSDecision::approve(metrics)
    }

    /// Weighted consensus aggregation.
    fn weighted_consensus(&self, risks: &[SubsystemRisk]) -> (f64, f64, bool) {
        if risks.is_empty() {
            return (0.0, 1.0, true);
        }

        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        let mut confidence_sum = 0.0;

        for risk in risks {
            let weight = self.adaptive_weights
                .get(&risk.subsystem)
                .copied()
                .unwrap_or(0.2);

            weighted_score += risk.risk_score * weight * risk.confidence;
            total_weight += weight;
            confidence_sum += risk.confidence * weight;
        }

        let overall_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };

        let consensus_confidence = if total_weight > 0.0 {
            confidence_sum / total_weight
        } else {
            0.0
        };

        // Check if consensus reached (subsystems sufficiently agree)
        let variance = risks
            .iter()
            .map(|r| (r.risk_score - overall_score).powi(2))
            .sum::<f64>()
            / risks.len() as f64;

        let consensus_reached = variance.sqrt() < (1.0 - self.config.consensus_threshold);

        (overall_score, consensus_confidence, consensus_reached)
    }

    /// Convert risk score to risk level.
    fn score_to_risk_level(score: f64) -> RiskLevel {
        match score {
            s if s >= 0.9 => RiskLevel::Emergency,
            s if s >= 0.75 => RiskLevel::Critical,
            s if s >= 0.5 => RiskLevel::High,
            s if s >= 0.25 => RiskLevel::Elevated,
            _ => RiskLevel::Normal,
        }
    }

    /// Detect market regime from subsystem signals.
    fn detect_regime_from_subsystems(&self, _risks: &[SubsystemRisk]) -> Option<MarketRegime> {
        // This would aggregate regime signals from various subsystems
        // For now, return None until regime detection is implemented
        None
    }

    /// Update accuracy history for adaptive weight adjustment.
    pub fn record_prediction_outcome(&mut self, subsystem: SubsystemId, correct: bool) {
        let history = self.accuracy_history
            .entry(subsystem)
            .or_insert_with(Vec::new);

        history.push(correct);

        // Trim to window size
        if history.len() > self.config.accuracy_window {
            history.remove(0);
        }

        // Update adaptive weights if enabled
        if self.config.adaptive_weights {
            self.update_adaptive_weights();
        }
    }

    /// Update adaptive weights based on recent accuracy.
    fn update_adaptive_weights(&mut self) {
        let mut accuracies: HashMap<SubsystemId, f64> = HashMap::new();

        for (subsystem, history) in &self.accuracy_history {
            if !history.is_empty() {
                let correct_count = history.iter().filter(|&&x| x).count();
                let accuracy = correct_count as f64 / history.len() as f64;
                accuracies.insert(*subsystem, accuracy);
            }
        }

        if accuracies.is_empty() {
            return;
        }

        // Calculate new weights proportional to accuracy
        let total_accuracy: f64 = accuracies.values().sum();
        if total_accuracy > 0.0 {
            for (subsystem, accuracy) in &accuracies {
                let new_weight = accuracy / total_accuracy;
                // Smooth update (70% old, 30% new)
                if let Some(weight) = self.adaptive_weights.get_mut(subsystem) {
                    *weight = 0.7 * *weight + 0.3 * new_weight;
                }
            }

            // Normalize weights
            let total_weight: f64 = self.adaptive_weights.values().sum();
            if total_weight > 0.0 {
                for weight in self.adaptive_weights.values_mut() {
                    *weight /= total_weight;
                }
            }
        }
    }

    /// Get current adaptive weights.
    pub fn get_adaptive_weights(&self) -> &HashMap<SubsystemId, f64> {
        &self.adaptive_weights
    }

    /// Get configuration.
    pub fn config(&self) -> &CWTSConfig {
        &self.config
    }

    // ========================================================================
    // BFT Consensus Methods
    // ========================================================================

    /// Perform BFT-enabled risk assessment with Byzantine fault tolerance.
    ///
    /// This method uses PBFT-style consensus to aggregate subsystem risk assessments
    /// while detecting and filtering out Byzantine (malfunctioning) subsystems.
    ///
    /// Returns both the standard IntegratedRiskMetrics and a BFT consensus result
    /// that includes Byzantine detection and consensus proof.
    pub fn assess_risk_bft(
        &mut self,
        portfolio: &Portfolio,
        symbol: Option<&Symbol>,
    ) -> (IntegratedRiskMetrics, Option<BftConsensusResult>) {
        // If BFT is disabled, fall back to standard assessment
        if !self.config.enable_bft || self.bft_engine.is_none() {
            return (self.assess_risk(portfolio, symbol), None);
        }

        let start = Timestamp::now();
        let mut subsystem_risks = Vec::new();

        // Collect assessments from all enabled subsystems (same as assess_risk)
        #[cfg(feature = "cwts-autopoiesis")]
        if self.config.enable_autopoiesis {
            if let Some(ref adapter) = self.autopoiesis {
                let risk = adapter.assess_portfolio_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-game-theory")]
        if self.config.enable_game_theory {
            if let Some(ref adapter) = self.game_theory {
                let risk = adapter.assess_strategic_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-physics")]
        if self.config.enable_physics {
            if let Some(ref adapter) = self.physics {
                let risk = adapter.assess_physics_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-nautilus")]
        if self.config.enable_nautilus {
            if let Some(ref adapter) = self.nautilus {
                let risk = adapter.assess_portfolio_execution_risk(portfolio, symbol);
                subsystem_risks.push(risk);
            }
        }

        #[cfg(feature = "cwts-neural")]
        if self.config.enable_neural {
            if let Some(ref adapter) = self.neural {
                let risk = adapter.assess_forecast_risk(portfolio);
                subsystem_risks.push(risk);
            }
        }

        // If no subsystems active, return normal metrics with no BFT result
        if subsystem_risks.is_empty() {
            return (IntegratedRiskMetrics {
                overall_risk: RiskLevel::Normal,
                overall_score: 0.0,
                consensus_confidence: 1.0,
                consensus_reached: true,
                subsystem_risks: vec![],
                recommended_position_factor: 1.0,
                total_latency_ns: 0,
                market_regime: None,
                timestamp: Timestamp::now(),
            }, None);
        }

        // Run BFT consensus
        let bft_engine = self.bft_engine.as_mut().unwrap();

        // Create proposal context from portfolio
        let context = ProposalContext {
            portfolio_size: portfolio.total_value(),
            volatility: portfolio.volatility_estimate(),
            drawdown: portfolio.current_drawdown(),
            regime_hint: None,
        };

        // Start consensus round
        let _proposal = bft_engine.start_consensus(context);

        // Submit all subsystem assessments
        for risk in &subsystem_risks {
            let _ = bft_engine.submit_assessment(risk.clone());
        }

        // Finalize consensus
        let bft_result = bft_engine.finalize_consensus().ok();

        // Build metrics using BFT result
        let (overall_risk, overall_score, consensus_reached, byzantine_filtered) =
            if let Some(ref result) = bft_result {
                (
                    result.agreed_risk_level,
                    result.aggregated_score,
                    result.consensus_achieved,
                    result.byzantine_suspects.clone(),
                )
            } else {
                // Fallback to weighted consensus
                let (score, _, reached) = self.weighted_consensus(&subsystem_risks);
                (Self::score_to_risk_level(score), score, reached, vec![])
            };

        // Calculate position factor, excluding Byzantine subsystems
        let valid_risks: Vec<&SubsystemRisk> = subsystem_risks
            .iter()
            .filter(|r| !byzantine_filtered.contains(&r.subsystem))
            .collect();

        let recommended_position_factor = if valid_risks.is_empty() {
            1.0
        } else {
            valid_risks
                .iter()
                .map(|r| {
                    let weight = self.adaptive_weights
                        .get(&r.subsystem)
                        .copied()
                        .unwrap_or(0.2);
                    r.position_factor * weight
                })
                .sum::<f64>()
                .clamp(0.0, 1.0)
        };

        // Total latency
        let total_latency_ns = subsystem_risks.iter().map(|r| r.latency_ns).sum();

        let metrics = IntegratedRiskMetrics {
            overall_risk,
            overall_score,
            consensus_confidence: bft_result
                .as_ref()
                .map(|r| r.agreement_count as f64 / r.participant_count as f64)
                .unwrap_or(1.0),
            consensus_reached,
            subsystem_risks,
            recommended_position_factor,
            total_latency_ns,
            market_regime: None,
            timestamp: Timestamp::now(),
        };

        // Store last BFT result for audit
        self.last_bft_result = bft_result.clone();

        (metrics, bft_result)
    }

    /// Make trading decision using BFT consensus.
    ///
    /// This extends make_decision with Byzantine fault detection.
    pub fn make_decision_bft(
        &mut self,
        portfolio: &Portfolio,
        symbol: Option<&Symbol>,
    ) -> (CWTSDecision, Option<BftConsensusResult>) {
        let (metrics, bft_result) = self.assess_risk_bft(portfolio, symbol);

        // Check if any Byzantine subsystems detected
        let byzantine_alert = bft_result
            .as_ref()
            .map(|r| !r.byzantine_suspects.is_empty())
            .unwrap_or(false);

        // Emergency override - any subsystem can veto
        if metrics.has_emergency() {
            let emergency_subsystem = metrics
                .subsystem_risks
                .iter()
                .find(|r| r.risk_level == RiskLevel::Emergency)
                .map(|r| r.subsystem.short_name())
                .unwrap_or("UNKNOWN");

            let mut recommendations = vec![
                "Halt all trading immediately".to_string(),
                "Reduce exposure to zero".to_string(),
                "Investigate root cause".to_string(),
            ];

            if byzantine_alert {
                recommendations.push(
                    "WARNING: Byzantine subsystem behavior detected".to_string()
                );
            }

            return (CWTSDecision::reject(
                metrics,
                format!("Emergency halt triggered by {}", emergency_subsystem),
                recommendations,
            ), bft_result);
        }

        // Critical risk - reject unless overridden
        if metrics.overall_risk >= RiskLevel::Critical {
            let mut recommendations = vec![
                "Reduce position sizes to 10%".to_string(),
                "Close high-risk positions".to_string(),
                "Wait for risk normalization".to_string(),
            ];

            if byzantine_alert {
                recommendations.push(
                    "Note: Some subsystems showed inconsistent behavior".to_string()
                );
            }

            return (CWTSDecision::reject(
                metrics,
                "Critical risk level - trading restricted",
                recommendations,
            ), bft_result);
        }

        // High risk - conditional approval
        if metrics.overall_risk == RiskLevel::High {
            let mut decision = CWTSDecision::approve(metrics);
            decision.reason = "Approved with restrictions - high risk".to_string();
            decision.recommendations = vec![
                "Limit position sizes to 50%".to_string(),
                "Tighten stop-losses".to_string(),
            ];
            return (decision, bft_result);
        }

        // BFT-specific: Check if consensus wasn't reached
        if !metrics.consensus_reached {
            return (CWTSDecision::reject(
                metrics,
                "BFT consensus not reached - subsystems disagree significantly",
                vec![
                    "Review individual subsystem assessments".to_string(),
                    "Check for Byzantine behavior in logs".to_string(),
                    "Consider manual override if appropriate".to_string(),
                ],
            ), bft_result);
        }

        // Normal/Elevated - approve
        (CWTSDecision::approve(metrics), bft_result)
    }

    /// Get the last BFT consensus result (for audit trail).
    pub fn last_bft_result(&self) -> Option<&BftConsensusResult> {
        self.last_bft_result.as_ref()
    }

    /// Get the last consensus proof (for cryptographic verification).
    pub fn last_consensus_proof(&self) -> Option<&ConsensusProof> {
        self.last_bft_result.as_ref().map(|r| &r.proof)
    }

    /// Check if BFT consensus is enabled.
    pub fn is_bft_enabled(&self) -> bool {
        self.config.enable_bft && self.bft_engine.is_some()
    }

    /// Get currently detected Byzantine subsystems.
    pub fn get_byzantine_subsystems(&self) -> Vec<SubsystemId> {
        self.bft_engine
            .as_ref()
            .map(|e| e.get_known_byzantine().to_vec())
            .unwrap_or_default()
    }

    /// Manually mark a subsystem as Byzantine (after external analysis).
    pub fn mark_subsystem_byzantine(&mut self, subsystem: SubsystemId) {
        if let Some(ref mut engine) = self.bft_engine {
            engine.mark_byzantine(subsystem);
        }
    }

    /// Clear Byzantine status for a subsystem (after recovery).
    pub fn clear_subsystem_byzantine(&mut self, subsystem: SubsystemId) {
        if let Some(ref mut engine) = self.bft_engine {
            engine.clear_byzantine(subsystem);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_weights_sum() {
        let config = CWTSConfig::default();
        let total = config.weight_autopoiesis
            + config.weight_game_theory
            + config.weight_physics
            + config.weight_nautilus
            + config.weight_neural;
        assert!((total - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    }

    #[test]
    fn test_config_validation() {
        let config = CWTSConfig::default();
        assert!(config.validate().is_ok());

        let mut bad_config = CWTSConfig::default();
        bad_config.weight_neural = 0.5; // Now weights won't sum to 1.0
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_risk_level_conversion() {
        assert_eq!(CWTSCoordinator::score_to_risk_level(0.0), RiskLevel::Normal);
        assert_eq!(CWTSCoordinator::score_to_risk_level(0.3), RiskLevel::Elevated);
        assert_eq!(CWTSCoordinator::score_to_risk_level(0.6), RiskLevel::High);
        assert_eq!(CWTSCoordinator::score_to_risk_level(0.8), RiskLevel::Critical);
        assert_eq!(CWTSCoordinator::score_to_risk_level(0.95), RiskLevel::Emergency);
    }

    #[test]
    fn test_subsystem_id_short_name() {
        assert_eq!(SubsystemId::Autopoiesis.short_name(), "AUTO");
        assert_eq!(SubsystemId::GameTheory.short_name(), "GAME");
        assert_eq!(SubsystemId::Physics.short_name(), "PHYS");
        assert_eq!(SubsystemId::Nautilus.short_name(), "NAUT");
        assert_eq!(SubsystemId::Neural.short_name(), "NEUR");
    }

    #[test]
    fn test_subsystem_risk_creation() {
        let risk = SubsystemRisk::normal(SubsystemId::Neural, 0.9, 1000);
        assert_eq!(risk.risk_level, RiskLevel::Normal);
        assert_eq!(risk.position_factor, 1.0);

        let critical = SubsystemRisk::critical(
            SubsystemId::GameTheory,
            0.95,
            "Nash equilibrium unstable",
            2000,
        );
        assert_eq!(critical.risk_level, RiskLevel::Critical);
        assert!(critical.risk_score > 0.8);
    }

    #[test]
    fn test_coordinator_creation() {
        let coordinator = CWTSCoordinator::default_coordinator();
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_weighted_consensus_empty() {
        let coordinator = CWTSCoordinator::default_coordinator().unwrap();
        let (score, confidence, reached) = coordinator.weighted_consensus(&[]);
        assert_eq!(score, 0.0);
        assert_eq!(confidence, 1.0);
        assert!(reached);
    }

    #[test]
    fn test_integrated_metrics_helpers() {
        let metrics = IntegratedRiskMetrics {
            overall_risk: RiskLevel::Elevated,
            overall_score: 0.3,
            consensus_confidence: 0.85,
            consensus_reached: true,
            subsystem_risks: vec![
                SubsystemRisk::normal(SubsystemId::Neural, 0.9, 1000),
                SubsystemRisk::elevated(
                    SubsystemId::GameTheory,
                    0.8,
                    0.4,
                    "Mild strategic concern",
                    1500,
                ),
            ],
            recommended_position_factor: 0.9,
            total_latency_ns: 2500,
            market_regime: None,
            timestamp: Timestamp::now(),
        };

        assert!(!metrics.has_emergency());
        assert_eq!(metrics.max_risk_level(), RiskLevel::Elevated);

        let highest = metrics.highest_risk_subsystem().unwrap();
        assert_eq!(highest.subsystem, SubsystemId::GameTheory);
    }

    #[test]
    fn test_decision_to_risk_decision() {
        let metrics = IntegratedRiskMetrics {
            overall_risk: RiskLevel::Normal,
            overall_score: 0.1,
            consensus_confidence: 0.95,
            consensus_reached: true,
            subsystem_risks: vec![],
            recommended_position_factor: 1.0,
            total_latency_ns: 500,
            market_regime: None,
            timestamp: Timestamp::now(),
        };

        let decision = CWTSDecision::approve(metrics);
        let risk_decision = decision.to_risk_decision();

        assert!(risk_decision.allowed);
        assert_eq!(risk_decision.risk_level, RiskLevel::Normal);
        assert_eq!(risk_decision.size_adjustment, 1.0);
    }

    #[test]
    fn test_adaptive_weight_recording() {
        let mut coordinator = CWTSCoordinator::default_coordinator().unwrap();

        // Record some outcomes
        for _ in 0..10 {
            coordinator.record_prediction_outcome(SubsystemId::Neural, true);
            coordinator.record_prediction_outcome(SubsystemId::GameTheory, false);
        }

        let weights = coordinator.get_adaptive_weights();
        // Neural should have higher weight due to better accuracy
        assert!(weights.get(&SubsystemId::Neural).unwrap()
            > weights.get(&SubsystemId::GameTheory).unwrap());
    }
}
