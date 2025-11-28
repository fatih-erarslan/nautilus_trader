//! Autopoiesis Integration for CWTS Risk Management
//!
//! Implements self-organizing risk systems based on Maturana & Varela's autopoiesis
//! theory and Prigogine's dissipative structures.
//!
//! ## Key Concepts
//!
//! - **Autopoiesis**: Self-maintaining systems that define their own boundaries
//! - **Operational Closure**: The system produces its own components
//! - **Structural Coupling**: Interaction with environment while maintaining identity
//! - **Dissipative Structures**: Systems maintained far from equilibrium
//!
//! ## Risk Applications
//!
//! - Boundary maintenance: Keep risk within operational closure
//! - Emergence detection: Identify novel risk patterns
//! - Self-organization: Adaptive risk management without central control

use crate::core::{MarketRegime, RiskLevel, Symbol};
use crate::sentinels::SentinelId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// Import from autopoiesis crate when available
// The autopoiesis crate provides self-organizing system primitives
// We wrap them in our risk-aware adapters
#[cfg(feature = "cwts-autopoiesis")]
use autopoiesis::prelude::*;

/// Health status of an autopoietic system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemHealth {
    /// System is self-maintaining normally
    Healthy,
    /// System is adapting to environmental pressure
    Adapting,
    /// System boundaries are under stress
    Stressed,
    /// System is undergoing reorganization
    Reorganizing,
    /// System integrity is compromised
    Critical,
    /// System has lost autopoietic closure
    Collapsed,
}

impl SystemHealth {
    /// Convert health status to risk level
    #[must_use]
    pub fn to_risk_level(&self) -> RiskLevel {
        match self {
            Self::Healthy => RiskLevel::Normal,
            Self::Adapting => RiskLevel::Elevated,
            Self::Stressed => RiskLevel::High,
            Self::Reorganizing => RiskLevel::High,
            Self::Critical => RiskLevel::Critical,
            Self::Collapsed => RiskLevel::Emergency,
        }
    }

    /// Get health score (0.0 = collapsed, 1.0 = healthy)
    #[must_use]
    pub fn score(&self) -> f64 {
        match self {
            Self::Healthy => 1.0,
            Self::Adapting => 0.8,
            Self::Stressed => 0.6,
            Self::Reorganizing => 0.4,
            Self::Critical => 0.2,
            Self::Collapsed => 0.0,
        }
    }
}

/// State of system boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryState {
    /// Current boundary integrity (0.0-1.0)
    pub integrity: f64,
    /// Rate of entropy production
    pub entropy_rate: f64,
    /// Energy flux through boundary
    pub energy_flux: f64,
    /// Number of boundary violations
    pub violations: u32,
    /// Time since last boundary adjustment
    pub time_since_adjustment_ms: u64,
    /// Active boundary conditions
    pub active_conditions: Vec<BoundaryCondition>,
}

/// A boundary condition in the autopoietic system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    /// Unique identifier
    pub id: String,
    /// Condition type
    pub condition_type: BoundaryConditionType,
    /// Current value
    pub current_value: f64,
    /// Threshold for violation
    pub threshold: f64,
    /// Whether condition is currently violated
    pub is_violated: bool,
}

/// Types of boundary conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryConditionType {
    /// Maximum risk exposure
    MaxRiskExposure,
    /// Minimum liquidity
    MinLiquidity,
    /// Maximum drawdown
    MaxDrawdown,
    /// Correlation threshold
    CorrelationLimit,
    /// Position concentration
    ConcentrationLimit,
    /// Volatility tolerance
    VolatilityTolerance,
    /// Counterparty exposure
    CounterpartyLimit,
}

/// Alert for emergence detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceAlert {
    /// Unique alert ID
    pub id: u64,
    /// Time of detection
    pub timestamp: DateTime<Utc>,
    /// Type of emergence
    pub emergence_type: EmergenceType,
    /// Magnitude of emergence (0.0-1.0)
    pub magnitude: f64,
    /// Affected symbols
    pub affected_symbols: Vec<Symbol>,
    /// Recommended action
    pub recommended_action: EmergenceAction,
    /// Confidence in detection
    pub confidence: f64,
}

/// Types of emergence patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceType {
    /// New correlation pattern forming
    CorrelationEmergence,
    /// Strategy network reorganization
    StrategyEvolution,
    /// Risk regime transition
    RegimeTransition,
    /// Market structure change
    StructuralChange,
    /// Collective behavior emergence
    CollectiveBehavior,
    /// Self-organized criticality
    CriticalityApproach,
    /// Dissipative structure formation
    DissipativeFormation,
}

/// Recommended action for emergence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceAction {
    /// Monitor the pattern
    Monitor,
    /// Adapt strategy allocation
    AdaptAllocation,
    /// Reduce risk exposure
    ReduceExposure,
    /// Hedge positions
    Hedge,
    /// Trigger circuit breaker
    CircuitBreaker,
    /// Full risk shutdown
    Shutdown,
}

/// Configuration for autopoiesis risk adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoiesisConfig {
    /// Adaptation rate (0.0-1.0)
    pub adaptation_rate: f64,
    /// Self-maintenance threshold
    pub self_maintenance_threshold: f64,
    /// Emergence detection sensitivity
    pub emergence_sensitivity: f64,
    /// Boundary integrity minimum
    pub min_boundary_integrity: f64,
    /// Maximum entropy rate before reorganization
    pub max_entropy_rate: f64,
    /// Enable dissipative structure detection
    pub enable_dissipative_detection: bool,
    /// Complexity threshold for emergence
    pub complexity_threshold: f64,
}

impl Default for AutopoiesisConfig {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.1,
            self_maintenance_threshold: 0.8,
            emergence_sensitivity: 0.7,
            min_boundary_integrity: 0.6,
            max_entropy_rate: 0.5,
            enable_dissipative_detection: true,
            complexity_threshold: 0.85,
        }
    }
}

/// Autopoiesis-based risk adapter
///
/// Integrates autopoietic systems theory into risk management:
/// - Maintains operational closure of risk boundaries
/// - Detects emergence of new risk patterns
/// - Enables self-organization of risk controls
pub struct AutopoiesisRiskAdapter {
    config: AutopoiesisConfig,
    health: RwLock<SystemHealth>,
    boundary_state: RwLock<BoundaryState>,
    emergence_alerts: RwLock<Vec<EmergenceAlert>>,
    alert_counter: AtomicU64,
    strategy_network: RwLock<StrategyNetwork>,
    complexity_history: RwLock<Vec<f64>>,
    entropy_history: RwLock<Vec<f64>>,
}

/// Internal strategy network representation
#[derive(Debug, Clone, Default)]
struct StrategyNetwork {
    nodes: HashMap<String, StrategyNode>,
    edges: Vec<(String, String, f64)>,
    total_complexity: f64,
}

#[derive(Debug, Clone)]
struct StrategyNode {
    id: String,
    performance_score: f64,
    risk_contribution: f64,
    is_active: bool,
}

impl AutopoiesisRiskAdapter {
    /// Create a new autopoiesis risk adapter
    #[must_use]
    pub fn new(config: AutopoiesisConfig) -> Self {
        Self {
            config,
            health: RwLock::new(SystemHealth::Healthy),
            boundary_state: RwLock::new(BoundaryState {
                integrity: 1.0,
                entropy_rate: 0.0,
                energy_flux: 0.0,
                violations: 0,
                time_since_adjustment_ms: 0,
                active_conditions: Vec::new(),
            }),
            emergence_alerts: RwLock::new(Vec::new()),
            alert_counter: AtomicU64::new(0),
            strategy_network: RwLock::new(StrategyNetwork::default()),
            complexity_history: RwLock::new(Vec::with_capacity(1000)),
            entropy_history: RwLock::new(Vec::with_capacity(1000)),
        }
    }

    /// Get current system health
    #[must_use]
    pub fn health(&self) -> SystemHealth {
        *self.health.read()
    }

    /// Get current boundary state
    #[must_use]
    pub fn boundary_state(&self) -> BoundaryState {
        self.boundary_state.read().clone()
    }

    /// Get recent emergence alerts
    #[must_use]
    pub fn recent_alerts(&self, limit: usize) -> Vec<EmergenceAlert> {
        let alerts = self.emergence_alerts.read();
        alerts.iter().rev().take(limit).cloned().collect()
    }

    /// Process market update and assess autopoietic health
    pub fn process_market_update(
        &self,
        regime: MarketRegime,
        volatility: f64,
        correlation_matrix: &[f64],
        position_risks: &[(Symbol, f64)],
    ) -> SystemHealth {
        // Calculate entropy rate from volatility and correlations
        let entropy_rate = self.calculate_entropy_rate(volatility, correlation_matrix);

        // Update boundary state
        self.update_boundary_state(entropy_rate, position_risks);

        // Check for emergence patterns
        self.detect_emergence(regime, correlation_matrix);

        // Calculate system health
        let new_health = self.calculate_health(entropy_rate);

        // Update health
        *self.health.write() = new_health;

        new_health
    }

    /// Calculate entropy production rate
    fn calculate_entropy_rate(&self, volatility: f64, correlation_matrix: &[f64]) -> f64 {
        // Entropy increases with volatility (use normalized scale for typical vol range 0-1)
        // Map volatility directly to entropy contribution in [0, 0.5] range
        let volatility_entropy = volatility.clamp(0.0, 1.0) * 0.5;

        // Entropy increases with correlation (less diversity)
        let n = (correlation_matrix.len() as f64).sqrt() as usize;
        let avg_correlation = if n > 1 {
            let sum: f64 = correlation_matrix.iter()
                .enumerate()
                .filter(|(i, _)| *i % (n + 1) != 0) // Skip diagonal
                .map(|(_, &c)| c.abs())
                .sum();
            sum / ((n * n - n) as f64).max(1.0)
        } else {
            0.0
        };
        let correlation_entropy = avg_correlation * 0.5;

        (volatility_entropy + correlation_entropy).clamp(0.0, 1.0)
    }

    /// Update boundary state
    fn update_boundary_state(&self, entropy_rate: f64, position_risks: &[(Symbol, f64)]) {
        let mut boundary = self.boundary_state.write();

        // Update entropy tracking
        boundary.entropy_rate = entropy_rate;

        // Calculate boundary integrity
        let max_risk: f64 = position_risks.iter().map(|(_, r)| *r).fold(0.0, f64::max);
        let total_risk: f64 = position_risks.iter().map(|(_, r)| *r).sum();

        // Integrity degrades with high concentration and entropy
        let concentration = if total_risk > 0.0 { max_risk / total_risk } else { 0.0 };
        let entropy_penalty = entropy_rate * 0.3;
        let concentration_penalty = concentration * 0.2;

        boundary.integrity = (1.0 - entropy_penalty - concentration_penalty).clamp(0.0, 1.0);

        // Check boundary conditions
        for condition in &mut boundary.active_conditions {
            match condition.condition_type {
                BoundaryConditionType::MaxRiskExposure => {
                    condition.current_value = total_risk;
                    condition.is_violated = total_risk > condition.threshold;
                }
                BoundaryConditionType::ConcentrationLimit => {
                    condition.current_value = concentration;
                    condition.is_violated = concentration > condition.threshold;
                }
                BoundaryConditionType::VolatilityTolerance => {
                    condition.current_value = entropy_rate;
                    condition.is_violated = entropy_rate > condition.threshold;
                }
                _ => {}
            }
        }

        // Count violations
        boundary.violations = boundary.active_conditions.iter()
            .filter(|c| c.is_violated)
            .count() as u32;

        // Track entropy history
        let mut entropy_history = self.entropy_history.write();
        entropy_history.push(entropy_rate);
        if entropy_history.len() > 1000 {
            entropy_history.remove(0);
        }
    }

    /// Detect emergence patterns
    fn detect_emergence(&self, regime: MarketRegime, correlation_matrix: &[f64]) {
        // Calculate complexity metric
        let complexity = self.calculate_complexity(correlation_matrix);

        // Track complexity history
        {
            let mut history = self.complexity_history.write();
            history.push(complexity);
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        // Detect criticality approach
        if complexity > self.config.complexity_threshold {
            self.emit_emergence_alert(
                EmergenceType::CriticalityApproach,
                complexity,
                vec![],
                EmergenceAction::Monitor,
            );
        }

        // Detect regime transitions
        let entropy_rate = self.boundary_state.read().entropy_rate;
        if entropy_rate > self.config.max_entropy_rate {
            let action = if entropy_rate > 0.8 {
                EmergenceAction::ReduceExposure
            } else {
                EmergenceAction::AdaptAllocation
            };

            self.emit_emergence_alert(
                EmergenceType::RegimeTransition,
                entropy_rate,
                vec![],
                action,
            );
        }

        // Detect dissipative structure formation
        if self.config.enable_dissipative_detection {
            self.detect_dissipative_structures(complexity, entropy_rate);
        }
    }

    /// Calculate system complexity
    fn calculate_complexity(&self, correlation_matrix: &[f64]) -> f64 {
        let n = (correlation_matrix.len() as f64).sqrt() as usize;
        if n < 2 {
            return 0.0;
        }

        // Complexity increases with non-trivial correlations
        let non_trivial: Vec<f64> = correlation_matrix.iter()
            .enumerate()
            .filter(|(i, _)| *i % (n + 1) != 0)
            .map(|(_, &c)| c.abs())
            .filter(|&c| c > 0.1 && c < 0.9)
            .collect();

        if non_trivial.is_empty() {
            return 0.0;
        }

        // Higher entropy in correlation distribution = higher complexity
        let mean = non_trivial.iter().sum::<f64>() / non_trivial.len() as f64;
        let variance = non_trivial.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f64>() / non_trivial.len() as f64;

        // Complexity is maximized when correlations are diverse
        (variance.sqrt() * 2.0).clamp(0.0, 1.0)
    }

    /// Detect dissipative structure formation
    fn detect_dissipative_structures(&self, complexity: f64, entropy_rate: f64) {
        // Dissipative structures form when:
        // 1. System is far from equilibrium (high entropy rate)
        // 2. Complexity is increasing
        // 3. New patterns are emerging

        let history = self.complexity_history.read();
        if history.len() < 10 {
            return;
        }

        // Check for complexity increase trend
        let recent = &history[history.len() - 10..];
        let trend: f64 = recent.windows(2)
            .map(|w| w[1] - w[0])
            .sum::<f64>() / 9.0;

        if trend > 0.01 && entropy_rate > 0.3 && complexity > 0.5 {
            self.emit_emergence_alert(
                EmergenceType::DissipativeFormation,
                complexity,
                vec![],
                EmergenceAction::Monitor,
            );
        }
    }

    /// Emit an emergence alert
    fn emit_emergence_alert(
        &self,
        emergence_type: EmergenceType,
        magnitude: f64,
        affected_symbols: Vec<Symbol>,
        recommended_action: EmergenceAction,
    ) {
        let alert = EmergenceAlert {
            id: self.alert_counter.fetch_add(1, Ordering::SeqCst),
            timestamp: Utc::now(),
            emergence_type,
            magnitude,
            affected_symbols,
            recommended_action,
            confidence: magnitude.min(1.0),
        };

        let mut alerts = self.emergence_alerts.write();
        alerts.push(alert);

        // Keep only recent alerts
        if alerts.len() > 100 {
            alerts.remove(0);
        }
    }

    /// Calculate system health based on metrics
    fn calculate_health(&self, entropy_rate: f64) -> SystemHealth {
        let boundary = self.boundary_state.read();

        // Determine health based on multiple factors
        if boundary.integrity < 0.2 || boundary.violations > 5 {
            return SystemHealth::Collapsed;
        }

        if boundary.integrity < 0.4 || entropy_rate > 0.8 {
            return SystemHealth::Critical;
        }

        if boundary.violations > 3 {
            return SystemHealth::Reorganizing;
        }

        if boundary.integrity < 0.6 || entropy_rate > 0.5 {
            return SystemHealth::Stressed;
        }

        if entropy_rate > 0.3 || boundary.violations > 0 {
            return SystemHealth::Adapting;
        }

        SystemHealth::Healthy
    }

    /// Add a boundary condition
    pub fn add_boundary_condition(&self, condition: BoundaryCondition) {
        let mut boundary = self.boundary_state.write();
        boundary.active_conditions.push(condition);
    }

    /// Trigger system reorganization
    pub fn trigger_reorganization(&self) {
        *self.health.write() = SystemHealth::Reorganizing;

        // Reset entropy history to start fresh
        let mut entropy_history = self.entropy_history.write();
        entropy_history.clear();

        // Reset boundary violations
        let mut boundary = self.boundary_state.write();
        boundary.violations = 0;
        boundary.time_since_adjustment_ms = 0;
    }

    /// Get risk level based on autopoietic state
    #[must_use]
    pub fn risk_level(&self) -> RiskLevel {
        self.health().to_risk_level()
    }

    /// Check if system should adapt
    #[must_use]
    pub fn should_adapt(&self) -> bool {
        let health = self.health();
        matches!(health, SystemHealth::Stressed | SystemHealth::Critical)
    }

    /// Check if shutdown is recommended
    #[must_use]
    pub fn should_shutdown(&self) -> bool {
        matches!(self.health(), SystemHealth::Collapsed)
    }

    /// Assess portfolio risk and return SubsystemRisk for coordinator.
    ///
    /// This is the primary entry point for the CWTS coordinator to get
    /// autopoiesis-based risk assessment.
    #[must_use]
    pub fn assess_portfolio_risk(&self, portfolio: &crate::core::Portfolio) -> super::coordinator::SubsystemRisk {
        use super::coordinator::{SubsystemRisk, SubsystemId};
        use crate::core::Timestamp;

        let start = std::time::Instant::now();

        // Calculate position risks from portfolio
        let position_risks: Vec<(Symbol, f64)> = portfolio.positions
            .iter()
            .map(|p| {
                // Risk as fraction of total portfolio
                let risk = if portfolio.total_value > 0.0 {
                    p.market_value().abs() / portfolio.total_value
                } else {
                    0.0
                };
                (p.symbol, risk)
            })
            .collect();

        // Calculate volatility estimate from positions (simplified)
        let volatility = portfolio.drawdown_pct() / 100.0 * 0.5 + 0.1;

        // Calculate correlation matrix (identity for simplicity - real impl would use history)
        let n = position_risks.len().max(1);
        let correlation_matrix: Vec<f64> = (0..n*n)
            .map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.3 })
            .collect();

        // Process market update with current regime (use Normal as default)
        let health = self.process_market_update(
            crate::core::MarketRegime::Unknown,
            volatility,
            &correlation_matrix,
            &position_risks,
        );

        let latency_ns = start.elapsed().as_nanos() as u64;
        let boundary = self.boundary_state.read();

        // Build risk assessment
        let risk_level = health.to_risk_level();
        let risk_score = 1.0 - health.score();
        let confidence = boundary.integrity;

        let position_factor = match health {
            SystemHealth::Healthy => 1.0,
            SystemHealth::Adapting => 0.9,
            SystemHealth::Stressed => 0.7,
            SystemHealth::Reorganizing => 0.5,
            SystemHealth::Critical => 0.2,
            SystemHealth::Collapsed => 0.0,
        };

        let reasoning = format!(
            "Autopoiesis: {} (integrity={:.2}, entropy={:.2}, violations={})",
            match health {
                SystemHealth::Healthy => "System healthy",
                SystemHealth::Adapting => "System adapting",
                SystemHealth::Stressed => "Boundaries stressed",
                SystemHealth::Reorganizing => "System reorganizing",
                SystemHealth::Critical => "Critical state",
                SystemHealth::Collapsed => "Operational closure lost",
            },
            boundary.integrity,
            boundary.entropy_rate,
            boundary.violations
        );

        SubsystemRisk {
            subsystem: SubsystemId::Autopoiesis,
            risk_level,
            confidence,
            risk_score,
            position_factor,
            reasoning,
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }
}

impl Default for AutopoiesisRiskAdapter {
    fn default() -> Self {
        Self::new(AutopoiesisConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_to_risk_level() {
        assert_eq!(SystemHealth::Healthy.to_risk_level(), RiskLevel::Normal);
        assert_eq!(SystemHealth::Critical.to_risk_level(), RiskLevel::Critical);
    }

    #[test]
    fn test_health_scores() {
        assert_eq!(SystemHealth::Healthy.score(), 1.0);
        assert_eq!(SystemHealth::Collapsed.score(), 0.0);
    }

    #[test]
    fn test_adapter_creation() {
        let adapter = AutopoiesisRiskAdapter::default();
        assert_eq!(adapter.health(), SystemHealth::Healthy);
    }

    #[test]
    fn test_boundary_condition() {
        let adapter = AutopoiesisRiskAdapter::default();

        adapter.add_boundary_condition(BoundaryCondition {
            id: "max_risk".to_string(),
            condition_type: BoundaryConditionType::MaxRiskExposure,
            current_value: 0.0,
            threshold: 0.1,
            is_violated: false,
        });

        let boundary = adapter.boundary_state();
        assert_eq!(boundary.active_conditions.len(), 1);
    }

    #[test]
    fn test_entropy_rate_calculation() {
        let adapter = AutopoiesisRiskAdapter::default();

        // 3x3 correlation matrix (identity)
        let corr_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // Low volatility + identity matrix = low entropy
        let entropy = adapter.calculate_entropy_rate(0.1, &corr_matrix);
        assert!(entropy < 0.5);

        // High volatility = higher entropy
        let entropy_high = adapter.calculate_entropy_rate(0.5, &corr_matrix);
        assert!(entropy_high > entropy);
    }

    #[test]
    fn test_emergence_detection() {
        let adapter = AutopoiesisRiskAdapter::default();

        // Simulate high complexity correlation matrix
        let corr_matrix = [
            1.0, 0.3, 0.5, 0.2,
            0.3, 1.0, 0.4, 0.6,
            0.5, 0.4, 1.0, 0.35,
            0.2, 0.6, 0.35, 1.0,
        ];

        let positions = vec![
            (Symbol::new("BTC"), 0.3),
            (Symbol::new("ETH"), 0.2),
        ];

        // Process update
        let health = adapter.process_market_update(
            MarketRegime::BullTrending,
            0.4,
            &corr_matrix,
            &positions,
        );

        assert!(matches!(health, SystemHealth::Healthy | SystemHealth::Adapting));
    }

    #[test]
    fn test_reorganization_trigger() {
        let adapter = AutopoiesisRiskAdapter::default();

        adapter.trigger_reorganization();

        assert_eq!(adapter.health(), SystemHealth::Reorganizing);
    }
}
