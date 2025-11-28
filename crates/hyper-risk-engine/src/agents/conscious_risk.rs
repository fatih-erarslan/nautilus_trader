//! Conscious Risk Agent - Active Inference with Thermodynamic Constraints.
//!
//! Implements the Free Energy Principle for risk-aware action selection.
//! Uses thermodynamic constraints (Landauer bound) for position sizing.
//!
//! ## Scientific Foundation
//!
//! Based on the Free Energy Principle (Friston, 2010):
//! - F = E_q[ln q(φ) - ln p(φ, y)] (Variational Free Energy)
//! - Agents act to minimize surprise (prediction error)
//! - Thermodynamic feasibility via Landauer bound (kT ln(2) per bit erased)
//!
//! ## References
//! - Friston (2010): "The free-energy principle: a unified brain theory?"
//! - Landauer (1961): "Irreversibility and Heat Generation in the Computing Process"
//! - Parr & Friston (2019): "Generalised free energy and active inference"

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;
use nalgebra as na;

#[cfg(feature = "active-inference")]
use active_inference_agent::{
    ActiveInferenceAgent, GenerativeModel, ThermodynamicState, ConsciousExperience,
    BOLTZMANN_K, LANDAUER_LIMIT_300K,
};

use crate::core::types::{MarketRegime, Portfolio, RiskDecision, RiskLevel, Symbol, Timestamp};
use crate::core::error::{Result, RiskError};
use super::base::{Agent, AgentId, AgentStatus, AgentConfig, AgentStats};

/// Configuration for the Conscious Risk Agent.
#[derive(Debug, Clone)]
pub struct ConsciousRiskConfig {
    /// Base agent config.
    pub base: AgentConfig,

    // === Free Energy Principle Parameters ===

    /// Belief state dimensions (number of market states).
    /// Higher dimensions capture more nuanced market beliefs.
    pub state_dim: usize,
    /// Observation dimensions (observed market features).
    pub obs_dim: usize,
    /// Precision (inverse temperature) for action selection.
    /// Higher precision = more deterministic action selection.
    pub precision: f64,
    /// Free energy threshold for position adjustment.
    /// Actions triggered when F > threshold.
    pub free_energy_threshold: f64,
    /// Maximum position adjustment per cycle (fraction of current).
    pub max_adjustment_per_cycle: f64,

    // === Thermodynamic Constraints ===

    /// System temperature in Kelvin.
    pub temperature: f64,
    /// Energy budget in Joules per decision cycle.
    /// Bounds computational complexity via Landauer principle.
    pub energy_budget: f64,
    /// Enable reversible computing mode (lower energy cost).
    pub reversible_mode: bool,
    /// Maximum bits erasure per cycle (computational budget).
    pub max_bits_per_cycle: u64,

    // === Position Sizing via Thermodynamic Efficiency ===

    /// Base position size factor.
    pub base_position_factor: f64,
    /// Minimum thermodynamic efficiency for full position.
    pub min_efficiency_full_position: f64,
    /// Scale position by entropy production.
    pub entropy_scaling: bool,
}

impl Default for ConsciousRiskConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "ConsciousRiskAgent".to_string(),
                max_latency_us: 800, // <800μs target
                ..Default::default()
            },
            // Free Energy Principle
            state_dim: 8,     // 8 market states (bull, bear, sideways, etc.)
            obs_dim: 6,       // price, volume, volatility, momentum, sentiment, correlation
            precision: 2.0,   // Moderate determinism
            free_energy_threshold: 0.5,
            max_adjustment_per_cycle: 0.10, // 10% max adjustment

            // Thermodynamic constraints
            temperature: 300.0,  // Room temperature (K)
            energy_budget: 1e-12, // 1 picojoule per cycle (sufficient for ~1000 bits)
            reversible_mode: false,
            max_bits_per_cycle: 1000,

            // Position sizing
            base_position_factor: 1.0,
            min_efficiency_full_position: 0.5,
            entropy_scaling: true,
        }
    }
}

/// Risk action selected by Free Energy minimization.
#[derive(Debug, Clone)]
pub enum RiskAction {
    /// Maintain current position.
    Hold,
    /// Reduce exposure by factor.
    ReduceExposure { factor: f64, reason: String },
    /// Increase exposure by factor.
    IncreaseExposure { factor: f64, reason: String },
    /// Hedge specific risk.
    Hedge { symbol: Symbol, hedge_ratio: f64 },
    /// Exit position entirely.
    Exit { symbol: Symbol, urgency: f64 },
}

/// Conscious experience from a risk processing cycle.
#[derive(Debug, Clone)]
pub struct RiskExperience {
    /// Belief distribution over market states.
    pub market_belief: Vec<f64>,
    /// Variational free energy (surprise).
    pub free_energy: f64,
    /// Selected risk action.
    pub selected_action: RiskAction,
    /// Temporal thickness (phenomenological depth).
    pub temporal_thickness: f64,
    /// Entropy rate (information flow).
    pub entropy_rate: f64,
    /// Energy consumed this cycle (Joules).
    pub energy_consumed: f64,
    /// Thermodynamic efficiency.
    pub efficiency: f64,
    /// Processing latency (ns).
    pub latency_ns: u64,
}

/// Market observation vector for active inference.
#[derive(Debug, Clone)]
pub struct MarketObservation {
    /// Price change (normalized).
    pub price_delta: f64,
    /// Volume (normalized).
    pub volume: f64,
    /// Realized volatility.
    pub volatility: f64,
    /// Momentum indicator.
    pub momentum: f64,
    /// Sentiment score.
    pub sentiment: f64,
    /// Correlation with benchmark.
    pub correlation: f64,
}

impl MarketObservation {
    /// Convert to nalgebra vector for inference.
    pub fn to_vector(&self) -> na::DVector<f64> {
        na::DVector::from_vec(vec![
            self.price_delta,
            self.volume,
            self.volatility,
            self.momentum,
            self.sentiment,
            self.correlation,
        ])
    }

    /// Create from portfolio and regime.
    pub fn from_portfolio_regime(portfolio: &Portfolio, regime: MarketRegime) -> Self {
        // Extract features from portfolio state
        let price_delta = if portfolio.total_value > 0.0 {
            portfolio.unrealized_pnl / portfolio.total_value
        } else {
            0.0
        };

        // Regime-based features
        let (volatility, momentum, sentiment) = match regime {
            MarketRegime::BullTrending => (0.18, 0.8, 0.7),
            MarketRegime::BearTrending => (0.25, -0.6, 0.3),
            MarketRegime::SidewaysLow => (0.08, 0.0, 0.5),
            MarketRegime::SidewaysHigh => (0.35, 0.1, 0.4),
            MarketRegime::Crisis => (0.50, -0.5, 0.1),
            MarketRegime::Recovery => (0.22, 0.4, 0.6),
            MarketRegime::Unknown => (0.15, 0.0, 0.5),
        };

        Self {
            price_delta,
            volume: 0.5, // Normalized baseline
            volatility,
            momentum,
            sentiment,
            correlation: 0.6, // Typical market correlation
        }
    }
}

/// Conscious Risk Agent implementing Free Energy Principle.
#[derive(Debug)]
pub struct ConsciousRiskAgent {
    /// Configuration.
    config: ConsciousRiskConfig,
    /// Current status.
    status: AtomicU8,
    /// Statistics.
    stats: AgentStats,

    // === Active Inference Components (feature-gated) ===

    /// Active inference agent.
    #[cfg(feature = "active-inference")]
    inference_agent: RwLock<Option<ActiveInferenceAgent>>,

    /// Last conscious experience.
    #[cfg(feature = "active-inference")]
    last_experience: RwLock<Option<RiskExperience>>,

    /// Accumulated free energy (for trend detection).
    accumulated_free_energy: RwLock<Vec<f64>>,

    /// Market state beliefs history.
    belief_history: RwLock<Vec<na::DVector<f64>>>,
}

impl ConsciousRiskAgent {
    /// Create new Conscious Risk Agent.
    pub fn new(config: ConsciousRiskConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            #[cfg(feature = "active-inference")]
            inference_agent: RwLock::new(None),
            #[cfg(feature = "active-inference")]
            last_experience: RwLock::new(None),
            accumulated_free_energy: RwLock::new(Vec::with_capacity(100)),
            belief_history: RwLock::new(Vec::with_capacity(100)),
        }
    }

    /// Initialize the active inference agent with market-specific generative model.
    #[cfg(feature = "active-inference")]
    pub fn init_inference(&self) -> Result<()> {
        let mut agent_lock = self.inference_agent.write();

        if agent_lock.is_some() {
            return Ok(()); // Already initialized
        }

        // Create generative model for market states
        let model = GenerativeModel::new(self.config.state_dim, self.config.obs_dim);

        // Set up informed prior preferences (prefer low-risk states)
        let mut preferences = na::DVector::from_element(self.config.state_dim, 0.0);
        // State 0-2: Low risk (preferred)
        // State 3-5: Medium risk
        // State 6-7: High risk (avoid)
        for i in 0..self.config.state_dim {
            preferences[i] = if i < 3 {
                0.4 / 3.0  // Prefer low-risk
            } else if i < 6 {
                0.4 / 3.0  // Neutral medium-risk
            } else {
                0.2 / 2.0  // Avoid high-risk
            };
        }

        // Initialize belief with maximum entropy (uniform)
        let initial_belief = na::DVector::from_element(
            self.config.state_dim,
            1.0 / self.config.state_dim as f64
        );

        // Create agent with thermodynamic constraints
        let mut agent = ActiveInferenceAgent::with_consciousness(
            model,
            initial_belief,
            self.config.temperature,
            self.config.energy_budget,
        );

        // Set precision for action selection
        agent.precision = self.config.precision;

        // Configure thermodynamic state
        if let Some(ref mut thermo) = agent.thermodynamics {
            thermo.set_reversible_mode(self.config.reversible_mode);
        }

        // Define action repertoire (risk management actions)
        // Actions encode expected state transitions
        agent.actions = self.create_action_repertoire();

        *agent_lock = Some(agent);
        Ok(())
    }

    /// Create action repertoire for risk management.
    #[cfg(feature = "active-inference")]
    fn create_action_repertoire(&self) -> Vec<na::DVector<f64>> {
        let dim = self.config.state_dim;
        let mut actions = Vec::new();

        // Action 0: Hold (no state change)
        actions.push(na::DVector::from_element(dim, 0.0));

        // Action 1: Risk reduction (shift toward low-risk states)
        let mut reduce = na::DVector::from_element(dim, 0.0);
        for i in 0..dim {
            if i < dim / 3 {
                reduce[i] = 0.1;  // Increase low-risk
            } else if i >= 2 * dim / 3 {
                reduce[i] = -0.1; // Decrease high-risk
            }
        }
        actions.push(reduce);

        // Action 2: Risk increase (shift toward opportunity states)
        let mut increase = na::DVector::from_element(dim, 0.0);
        for i in 0..dim {
            if i < dim / 3 {
                increase[i] = -0.05;
            } else if i < 2 * dim / 3 {
                increase[i] = 0.1;  // Increase medium-risk (opportunity)
            }
        }
        actions.push(increase);

        // Action 3: Defensive (strong shift to safety)
        let mut defensive = na::DVector::from_element(dim, 0.0);
        defensive[0] = 0.2;  // Maximize safety state
        for i in dim / 2..dim {
            defensive[i] = -0.1;
        }
        actions.push(defensive);

        // Action 4: Opportunistic (calculated risk-taking)
        let mut opportunistic = na::DVector::from_element(dim, 0.0);
        if dim > 4 {
            opportunistic[dim / 2] = 0.15;  // Medium-risk opportunity
            opportunistic[0] = -0.05;
        }
        actions.push(opportunistic);

        actions
    }

    /// Process market observation through active inference.
    #[cfg(feature = "active-inference")]
    pub fn process_observation(&self, observation: &MarketObservation) -> Result<RiskExperience> {
        let start = Instant::now();

        // Ensure initialized
        self.init_inference()?;

        let mut agent_lock = self.inference_agent.write();
        let agent = agent_lock.as_mut().ok_or_else(|| {
            RiskError::ConfigurationError("Active inference agent not initialized".to_string())
        })?;

        // Convert observation to vector
        let obs_vec = observation.to_vector();

        // Perform conscious processing cycle
        let experience = agent.conscious_cycle(&obs_vec).map_err(|e| {
            RiskError::ConfigurationError(format!("Conscious cycle failed: {}", e))
        })?;

        // Determine action from experience
        let action = self.experience_to_action(&experience, observation);

        // Get thermodynamic metrics
        let (energy_consumed, efficiency) = if let Some(ref thermo) = agent.thermodynamics {
            (thermo.energy_consumed, thermo.efficiency())
        } else {
            (0.0, 1.0)
        };

        let latency = start.elapsed().as_nanos() as u64;

        // Build risk experience
        let risk_exp = RiskExperience {
            market_belief: experience.belief.iter().cloned().collect(),
            free_energy: experience.free_energy,
            selected_action: action,
            temporal_thickness: experience.temporal_thickness,
            entropy_rate: experience.entropy_rate,
            energy_consumed,
            efficiency,
            latency_ns: latency,
        };

        // Update history
        {
            let mut fe_history = self.accumulated_free_energy.write();
            fe_history.push(experience.free_energy);
            if fe_history.len() > 100 {
                fe_history.remove(0);
            }
        }
        {
            let mut belief_hist = self.belief_history.write();
            belief_hist.push(experience.belief.clone());
            if belief_hist.len() > 100 {
                belief_hist.remove(0);
            }
        }

        // Store last experience
        *self.last_experience.write() = Some(risk_exp.clone());

        Ok(risk_exp)
    }

    /// Convert conscious experience to risk action.
    #[cfg(feature = "active-inference")]
    fn experience_to_action(&self, exp: &ConsciousExperience, obs: &MarketObservation) -> RiskAction {
        // Map selected action index to risk action
        if let Some(ref action_vec) = exp.selected_action {
            // Determine action type from action vector pattern
            let max_idx = action_vec.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let action_magnitude = action_vec.iter().map(|x| x.abs()).sum::<f64>();

            // High free energy + high volatility = defensive
            if exp.free_energy > self.config.free_energy_threshold && obs.volatility > 0.3 {
                return RiskAction::ReduceExposure {
                    factor: self.thermodynamic_position_factor(exp),
                    reason: format!(
                        "High free energy ({:.3}) with elevated volatility ({:.1}%)",
                        exp.free_energy, obs.volatility * 100.0
                    ),
                };
            }

            // Low free energy + positive momentum = opportunity
            if exp.free_energy < self.config.free_energy_threshold * 0.5 && obs.momentum > 0.6 {
                return RiskAction::IncreaseExposure {
                    factor: self.thermodynamic_position_factor(exp).min(self.config.max_adjustment_per_cycle),
                    reason: format!(
                        "Low free energy ({:.3}) with positive momentum ({:.1}%)",
                        exp.free_energy, obs.momentum * 100.0
                    ),
                };
            }

            // Check entropy rate for consciousness quality
            if exp.entropy_rate > 0.5 && max_idx >= self.config.state_dim * 2 / 3 {
                // High entropy rate + high-risk belief state
                return RiskAction::ReduceExposure {
                    factor: 0.5 * self.thermodynamic_position_factor(exp),
                    reason: format!(
                        "High entropy rate ({:.3}) indicates unstable market regime",
                        exp.entropy_rate
                    ),
                };
            }

            // Use action magnitude as signal
            if action_magnitude < 0.1 {
                return RiskAction::Hold;
            }
        }

        RiskAction::Hold
    }

    /// Calculate position factor based on thermodynamic efficiency.
    #[cfg(feature = "active-inference")]
    fn thermodynamic_position_factor(&self, exp: &ConsciousExperience) -> f64 {
        let agent_lock = self.inference_agent.read();

        if let Some(ref agent) = *agent_lock {
            if let Some(ref thermo) = agent.thermodynamics {
                let efficiency = thermo.efficiency();

                // Scale position by thermodynamic efficiency
                // High efficiency = more confident = larger position
                // Low efficiency = uncertain = smaller position
                if self.config.entropy_scaling {
                    let entropy_factor = (-exp.entropy_rate).exp().min(1.0);
                    return self.config.base_position_factor * efficiency * entropy_factor;
                }

                if efficiency >= self.config.min_efficiency_full_position {
                    return self.config.base_position_factor;
                } else {
                    return self.config.base_position_factor * (efficiency / self.config.min_efficiency_full_position);
                }
            }
        }

        self.config.base_position_factor * 0.5 // Conservative default
    }

    /// Get free energy trend (rising = increasing surprise/risk).
    pub fn free_energy_trend(&self) -> f64 {
        let history = self.accumulated_free_energy.read();
        if history.len() < 2 {
            return 0.0;
        }

        // Linear regression slope
        let n = history.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = history.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &y) in history.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean) * (x - x_mean);
        }

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Get last conscious experience.
    #[cfg(feature = "active-inference")]
    pub fn last_experience(&self) -> Option<RiskExperience> {
        self.last_experience.read().clone()
    }

    /// Get Landauer limit at configured temperature.
    #[cfg(feature = "active-inference")]
    pub fn landauer_limit(&self) -> f64 {
        BOLTZMANN_K * self.config.temperature * std::f64::consts::LN_2
    }

    /// Check if agent is initialized.
    #[cfg(feature = "active-inference")]
    pub fn is_initialized(&self) -> bool {
        self.inference_agent.read().is_some()
    }

    /// Get current belief distribution.
    #[cfg(feature = "active-inference")]
    pub fn current_belief(&self) -> Option<Vec<f64>> {
        self.inference_agent.read().as_ref().map(|a| a.belief.iter().cloned().collect())
    }

    fn status_from_u8(val: u8) -> AgentStatus {
        match val {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            _ => AgentStatus::ShuttingDown,
        }
    }
}

impl Agent for ConsciousRiskAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        #[cfg(feature = "active-inference")]
        {
            // Create market observation from portfolio and regime
            let observation = MarketObservation::from_portfolio_regime(portfolio, regime);

            // Process through active inference
            match self.process_observation(&observation) {
                Ok(experience) => {
                    let latency = start.elapsed().as_nanos() as u64;
                    self.stats.record_cycle(latency);
                    self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

                    // Convert risk action to risk decision
                    match &experience.selected_action {
                        RiskAction::Hold => Ok(None),
                        RiskAction::ReduceExposure { factor, reason } => {
                            Ok(Some(RiskDecision {
                                allowed: true,
                                risk_level: RiskLevel::Elevated,
                                reason: format!("[FEP] {}", reason),
                                size_adjustment: 1.0 - factor,
                                timestamp: Timestamp::now(),
                                latency_ns: latency,
                            }))
                        },
                        RiskAction::IncreaseExposure { factor, reason } => {
                            Ok(Some(RiskDecision {
                                allowed: true,
                                risk_level: RiskLevel::Normal,
                                reason: format!("[FEP] {}", reason),
                                size_adjustment: 1.0 + factor,
                                timestamp: Timestamp::now(),
                                latency_ns: latency,
                            }))
                        },
                        RiskAction::Hedge { symbol, hedge_ratio } => {
                            Ok(Some(RiskDecision {
                                allowed: true,
                                risk_level: RiskLevel::Elevated,
                                reason: format!("[FEP] Hedge {} at {:.1}%", symbol, hedge_ratio * 100.0),
                                size_adjustment: 1.0,
                                timestamp: Timestamp::now(),
                                latency_ns: latency,
                            }))
                        },
                        RiskAction::Exit { symbol, urgency } => {
                            let risk_level = if *urgency > 0.8 {
                                RiskLevel::Critical
                            } else {
                                RiskLevel::High
                            };
                            Ok(Some(RiskDecision {
                                allowed: false,
                                risk_level,
                                reason: format!("[FEP] Exit {} with urgency {:.1}%", symbol, urgency * 100.0),
                                size_adjustment: 0.0,
                                timestamp: Timestamp::now(),
                                latency_ns: latency,
                            }))
                        },
                    }
                },
                Err(e) => {
                    self.stats.record_error();
                    self.status.store(AgentStatus::Error as u8, Ordering::Relaxed);
                    tracing::warn!("Conscious risk processing failed: {}", e);
                    // Fall back to conservative decision
                    let latency = start.elapsed().as_nanos() as u64;
                    Ok(Some(RiskDecision {
                        allowed: true,
                        risk_level: RiskLevel::Elevated,
                        reason: "[FEP] Active inference unavailable, using conservative mode".to_string(),
                        size_adjustment: 0.8,  // 20% reduction
                        timestamp: Timestamp::now(),
                        latency_ns: latency,
                    }))
                }
            }
        }

        #[cfg(not(feature = "active-inference"))]
        {
            // Without active-inference feature, use regime-based heuristics
            let latency = start.elapsed().as_nanos() as u64;
            self.stats.record_cycle(latency);
            self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

            let adjustment = regime.risk_multiplier();
            if adjustment < 0.8 {
                Ok(Some(RiskDecision {
                    allowed: true,
                    risk_level: RiskLevel::Elevated,
                    reason: format!("Regime {} suggests reduced exposure", adjustment),
                    size_adjustment: adjustment,
                    timestamp: Timestamp::now(),
                    latency_ns: latency,
                }))
            } else {
                Ok(None)
            }
        }
    }

    fn start(&self) -> Result<()> {
        #[cfg(feature = "active-inference")]
        {
            self.init_inference()?;
        }
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conscious_risk_creation() {
        let config = ConsciousRiskConfig::default();
        let agent = ConsciousRiskAgent::new(config);
        assert_eq!(agent.status(), AgentStatus::Idle);
    }

    #[test]
    fn test_market_observation() {
        let portfolio = Portfolio::new(100_000.0);
        let obs = MarketObservation::from_portfolio_regime(&portfolio, MarketRegime::SidewaysLow);
        assert!(obs.volatility > 0.0);
        assert!(obs.volatility < 1.0);
    }

    #[test]
    fn test_free_energy_trend_empty() {
        let config = ConsciousRiskConfig::default();
        let agent = ConsciousRiskAgent::new(config);
        let trend = agent.free_energy_trend();
        assert_eq!(trend, 0.0); // No history yet
    }

    #[cfg(feature = "active-inference")]
    #[test]
    fn test_active_inference_init() {
        let config = ConsciousRiskConfig::default();
        let agent = ConsciousRiskAgent::new(config);

        let result = agent.init_inference();
        assert!(result.is_ok());
        assert!(agent.is_initialized());
    }

    #[cfg(feature = "active-inference")]
    #[test]
    fn test_landauer_limit() {
        let config = ConsciousRiskConfig::default();
        let agent = ConsciousRiskAgent::new(config);

        let limit = agent.landauer_limit();
        // At 300K, should be approximately 2.87e-21 J
        assert!((limit - LANDAUER_LIMIT_300K).abs() < 1e-23);
    }

    #[cfg(feature = "active-inference")]
    #[test]
    fn test_conscious_processing() {
        let config = ConsciousRiskConfig::default();
        let agent = ConsciousRiskAgent::new(config);

        agent.init_inference().unwrap();

        let obs = MarketObservation {
            price_delta: 0.02,
            volume: 0.6,
            volatility: 0.18,
            momentum: 0.55,
            sentiment: 0.6,
            correlation: 0.7,
        };

        let result = agent.process_observation(&obs);
        assert!(result.is_ok());

        let experience = result.unwrap();
        assert!(experience.free_energy.is_finite());
        assert!(experience.market_belief.len() == 8);
        assert!(experience.energy_consumed >= 0.0);
    }

    #[cfg(feature = "active-inference")]
    #[test]
    fn test_thermodynamic_constraints() {
        let mut config = ConsciousRiskConfig::default();
        config.energy_budget = 1e-18; // Very small budget
        config.max_bits_per_cycle = 10;

        let agent = ConsciousRiskAgent::new(config);
        agent.init_inference().unwrap();

        // Should still work but with limited precision
        let obs = MarketObservation {
            price_delta: 0.0,
            volume: 0.5,
            volatility: 0.15,
            momentum: 0.5,
            sentiment: 0.5,
            correlation: 0.5,
        };

        let result = agent.process_observation(&obs);
        assert!(result.is_ok());
    }
}
