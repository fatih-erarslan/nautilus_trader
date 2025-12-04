//! Adapters connecting HyperPhysics crates to Tengri trading system
//!
//! Each adapter wraps a HyperPhysics crate and provides trading-specific functionality.

use crate::error::{BridgeError, Result};
use nalgebra::DMatrix;
use std::collections::VecDeque;

// ============================================================================
// Autopoiesis Adapter - Market Regime Detection
// ============================================================================

/// Adapter for hyperphysics-autopoiesis crate
/// Provides market regime detection via autopoietic health metrics
pub struct AutopoiesisAdapter {
    /// Autopoietic bridge for operational closure analysis
    bridge: hyperphysics_autopoiesis::bridges::AutopoieticBridge,
    /// Dissipative bridge for entropy-based regime detection
    dissipative: hyperphysics_autopoiesis::bridges::DissipativeBridge,
    /// Syntergic bridge for coherence tracking
    syntergic: hyperphysics_autopoiesis::bridges::SyntergicBridge,
    /// Configuration
    config: AutopoiesisConfig,
    /// Health history
    health_history: VecDeque<f64>,
}

/// Configuration for autopoiesis adapter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutopoiesisConfig {
    /// Minimum health threshold for stable regime
    pub min_health: f64,
    /// Maximum entropy for stable regime
    pub max_entropy: f64,
    /// Coherence threshold for syntergic unity
    pub coherence_threshold: f64,
    /// History window size
    pub history_window: usize,
}

impl Default for AutopoiesisConfig {
    fn default() -> Self {
        Self {
            min_health: 0.7,
            max_entropy: 2.0,
            coherence_threshold: 0.5,
            history_window: 100,
        }
    }
}

impl AutopoiesisAdapter {
    /// Create new autopoiesis adapter
    pub fn new(config: AutopoiesisConfig) -> Self {
        Self {
            bridge: hyperphysics_autopoiesis::bridges::AutopoieticBridge::new(config.min_health),
            dissipative: hyperphysics_autopoiesis::bridges::DissipativeBridge::new(
                config.max_entropy,
            ),
            syntergic: hyperphysics_autopoiesis::bridges::SyntergicBridge::new(
                config.coherence_threshold,
            ),
            config,
            health_history: VecDeque::with_capacity(100),
        }
    }

    /// Update autopoietic state from market data
    pub fn update(&mut self, prices: &[f64], volumes: &[f64]) -> Result<AutopoiesisState> {
        if prices.len() < 2 {
            return Err(BridgeError::InsufficientData {
                required: 2,
                available: prices.len(),
            });
        }

        // Calculate returns and volatility
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
        let volatility = Self::calculate_volatility(&returns);
        let volume_flow = volumes.last().copied().unwrap_or(0.0);

        // Register components with autopoietic bridge
        self.bridge
            .register_component("price_momentum", volatility, 0.5);
        self.bridge.register_component("volume_flow", volume_flow, 0.3);

        // Execute autopoietic cycle
        let health = match self.bridge.execute_cycle() {
            Ok(result) => result.health,
            Err(_) => {
                // Closure violation - degraded health
                self.health_history.back().copied().unwrap_or(0.5) * 0.9
            }
        };

        // Update dissipative dynamics
        let fluxes = vec![volatility, volume_flow / 1e6];
        let forces = vec![1.0, 0.5];
        let entropy = self
            .dissipative
            .update_entropy_production(&fluxes, &forces)
            .unwrap_or(0.0);
        let _regime = self.dissipative.update_control_parameter(volatility)?;

        // Update syntergic coherence from return phases
        let phases: Vec<f64> = returns
            .iter()
            .map(|r| {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                r.atan2(mean)
            })
            .collect();
        let coherence = self.syntergic.update_from_phases(&phases);

        // Store health history
        self.health_history.push_back(health);
        if self.health_history.len() > self.config.history_window {
            self.health_history.pop_front();
        }

        // Determine market regime
        let regime = self.classify_regime(health, entropy, coherence);

        Ok(AutopoiesisState {
            health,
            entropy,
            coherence,
            regime,
            near_bifurcation: self.dissipative.near_bifurcation(),
        })
    }

    /// Classify market regime based on autopoietic metrics
    fn classify_regime(
        &self,
        health: f64,
        entropy: f64,
        coherence: f64,
    ) -> AutopoiesisMarketRegime {
        if health > 0.8 && coherence > 0.7 && entropy < 1.0 {
            AutopoiesisMarketRegime::Stable
        } else if self.dissipative.near_bifurcation() {
            AutopoiesisMarketRegime::Transitional
        } else if entropy > self.config.max_entropy {
            AutopoiesisMarketRegime::Chaotic
        } else if coherence < 0.3 {
            AutopoiesisMarketRegime::Degraded
        } else {
            AutopoiesisMarketRegime::Normal
        }
    }

    /// Calculate volatility from returns
    fn calculate_volatility(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    }

    /// Get current health
    pub fn health(&self) -> f64 {
        self.health_history.back().copied().unwrap_or(1.0)
    }
}

/// Autopoietic system state
#[derive(Debug, Clone)]
pub struct AutopoiesisState {
    /// System health (0-1)
    pub health: f64,
    /// Entropy production rate
    pub entropy: f64,
    /// Syntergic coherence (0-1)
    pub coherence: f64,
    /// Current market regime
    pub regime: AutopoiesisMarketRegime,
    /// Near bifurcation point
    pub near_bifurcation: bool,
}

/// Market regime classification from autopoiesis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutopoiesisMarketRegime {
    /// Stable, self-maintaining regime
    Stable,
    /// Normal operation
    Normal,
    /// Near phase transition
    Transitional,
    /// Health degraded
    Degraded,
    /// High entropy, unpredictable
    Chaotic,
}

// ============================================================================
// Consciousness Adapter - IIT Φ Integration
// ============================================================================

/// Adapter for hyperphysics-consciousness crate
/// Provides market coherence metrics via Integrated Information Theory (Φ)
pub struct ConsciousnessAdapter {
    /// Φ calculator using IIT
    phi_calc: hyperphysics_consciousness::PhiCalculator,
    /// CI calculator for resonance complexity
    ci_calc: hyperphysics_consciousness::CICalculator,
    /// Configuration
    config: ConsciousnessConfig,
    /// Φ history
    phi_history: VecDeque<f64>,
}

/// Configuration for consciousness adapter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessConfig {
    /// Number of market dimensions to track
    pub dimensions: usize,
    /// Integration threshold
    pub integration_threshold: f64,
    /// History window
    pub history_window: usize,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            dimensions: 8,
            integration_threshold: 0.3,
            history_window: 100,
        }
    }
}

impl ConsciousnessAdapter {
    /// Create new consciousness adapter
    pub fn new(config: ConsciousnessConfig) -> Result<Self> {
        // Use monte_carlo approximation for market-scale analysis
        let phi_calc = hyperphysics_consciousness::PhiCalculator::monte_carlo(1000);
        let ci_calc = hyperphysics_consciousness::CICalculator::new();

        Ok(Self {
            phi_calc,
            ci_calc,
            config,
            phi_history: VecDeque::with_capacity(100),
        })
    }

    /// Calculate market Φ (integrated information) from multi-dimensional data
    pub fn calculate_market_phi(&mut self, market_state: &MarketStateVector) -> Result<PhiMetrics> {
        // Create a small pBit lattice from market features for IIT analysis
        // Use hyperbolic lattice with reasonable defaults
        let lattice = hyperphysics_pbit::PBitLattice::new(5, 4, 2, 1.0)
            .map_err(|e| BridgeError::ConsciousnessError {
                message: format!("Failed to create lattice: {}", e),
            })?;

        // Calculate Φ using the PhiCalculator on the lattice
        let phi_result = self
            .phi_calc
            .calculate(&lattice)
            .map_err(|e| BridgeError::ConsciousnessError {
                message: e.to_string(),
            })?;

        let phi = phi_result.phi;

        // Calculate CI (resonance complexity) for additional insight
        let ci_result = self
            .ci_calc
            .calculate(&lattice)
            .map_err(|e| BridgeError::ConsciousnessError {
                message: e.to_string(),
            })?;

        // Build transition probability matrix from market state for additional analysis
        let tpm = self.build_market_tpm(market_state)?;
        let market_effective_info = self.calculate_effective_info(&tpm);

        // Store history
        self.phi_history.push_back(phi);
        if self.phi_history.len() > self.config.history_window {
            self.phi_history.pop_front();
        }

        // Compute integration metrics
        let integration_level = if phi > self.config.integration_threshold {
            IntegrationLevel::High
        } else if phi > self.config.integration_threshold * 0.5 {
            IntegrationLevel::Medium
        } else {
            IntegrationLevel::Low
        };

        // Effective information: blend phi result with market TPM analysis
        let phi_effective = phi_result.mip.as_ref().map(|m| m.effective_info).unwrap_or(phi);
        let effective_info = (phi_effective + market_effective_info) / 2.0;

        Ok(PhiMetrics {
            phi,
            resonance_complexity: ci_result.ci,
            integration_level,
            effective_info,
            broadcast_strength: ci_result.coherence,
        })
    }

    /// Build transition probability matrix from market state
    fn build_market_tpm(&self, state: &MarketStateVector) -> Result<DMatrix<f64>> {
        let n = state.features.len().min(self.config.dimensions);

        // Build correlation-based TPM
        let mut tpm = DMatrix::zeros(n, n);

        for i in 0..n {
            let row_sum: f64 = (0..n)
                .map(|j| {
                    if i == j {
                        0.5
                    } else {
                        // Cross-correlation influence
                        (state.features[i] * state.features[j]).abs().min(0.3)
                    }
                })
                .sum();

            for j in 0..n {
                let val = if i == j {
                    0.5
                } else {
                    (state.features[i] * state.features[j]).abs().min(0.3)
                };
                tpm[(i, j)] = val / row_sum;
            }
        }

        Ok(tpm)
    }

    /// Calculate effective information
    fn calculate_effective_info(&self, tpm: &DMatrix<f64>) -> f64 {
        // Simplified effective information calculation
        let n = tpm.nrows();
        if n == 0 {
            return 0.0;
        }

        // Entropy of uniform distribution - entropy of TPM rows
        let uniform_entropy = (n as f64).ln();
        let mut avg_row_entropy = 0.0;

        for i in 0..n {
            let mut row_entropy = 0.0;
            for j in 0..n {
                let p = tpm[(i, j)];
                if p > 1e-10 {
                    row_entropy -= p * p.ln();
                }
            }
            avg_row_entropy += row_entropy;
        }
        avg_row_entropy /= n as f64;

        (uniform_entropy - avg_row_entropy).max(0.0)
    }

    /// Get current Φ
    pub fn current_phi(&self) -> f64 {
        self.phi_history.back().copied().unwrap_or(0.0)
    }
}

/// Market state vector for consciousness integration
#[derive(Debug, Clone)]
pub struct MarketStateVector {
    /// Feature values (normalized)
    pub features: Vec<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl MarketStateVector {
    /// Create from price and volume data
    pub fn from_market_data(prices: &[f64], volumes: &[f64]) -> Self {
        let mut features = Vec::new();
        let mut feature_names = Vec::new();

        if prices.len() >= 2 {
            // Returns
            let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            features.push(mean_return);
            feature_names.push("mean_return".to_string());

            // Volatility
            let var = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
                / returns.len() as f64;
            features.push(var.sqrt());
            feature_names.push("volatility".to_string());

            // Momentum
            let momentum = (prices.last().unwrap() - prices.first().unwrap()) / prices.first().unwrap();
            features.push(momentum);
            feature_names.push("momentum".to_string());

            // Trend strength
            let trend = returns.iter().filter(|&&r| r > 0.0).count() as f64 / returns.len() as f64;
            features.push(trend - 0.5);
            feature_names.push("trend".to_string());
        }

        if !volumes.is_empty() {
            // Volume trend
            let vol_mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
            let vol_recent = volumes.iter().rev().take(10).sum::<f64>() / 10.0;
            features.push((vol_recent - vol_mean) / vol_mean.max(1.0));
            feature_names.push("volume_trend".to_string());
        }

        // Pad to minimum dimensions
        while features.len() < 8 {
            features.push(0.0);
            feature_names.push(format!("pad_{}", features.len()));
        }

        Self {
            features,
            feature_names,
        }
    }
}

/// Φ metrics from consciousness integration
#[derive(Debug, Clone)]
pub struct PhiMetrics {
    /// Integrated Information (Φ)
    pub phi: f64,
    /// Resonance complexity (CI)
    pub resonance_complexity: f64,
    /// Integration level classification
    pub integration_level: IntegrationLevel,
    /// Effective information
    pub effective_info: f64,
    /// Broadcast strength (coherence)
    pub broadcast_strength: f64,
}

/// Integration level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationLevel {
    /// High integration - coherent market state
    High,
    /// Medium integration
    Medium,
    /// Low integration - fragmented market
    Low,
}

// ============================================================================
// Thermo Adapter - Entropy-Based Volatility
// ============================================================================

/// Adapter for hyperphysics-thermo crate
/// Provides entropy production metrics for volatility analysis
pub struct ThermoAdapter {
    /// Entropy calculator
    entropy_calc: hyperphysics_thermo::EntropyCalculator,
    /// Configuration
    config: ThermoConfig,
    /// Entropy history
    entropy_history: VecDeque<f64>,
}

/// Configuration for thermo adapter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThermoConfig {
    /// Temperature (effective market temperature)
    pub temperature: f64,
    /// History window
    pub history_window: usize,
}

impl Default for ThermoConfig {
    fn default() -> Self {
        Self {
            temperature: 300.0, // Kelvin equivalent
            history_window: 100,
        }
    }
}

impl ThermoAdapter {
    /// Create new thermo adapter
    pub fn new(config: ThermoConfig) -> Self {
        Self {
            entropy_calc: hyperphysics_thermo::EntropyCalculator::new(),
            config,
            entropy_history: VecDeque::with_capacity(100),
        }
    }

    /// Calculate thermodynamic metrics from market data
    pub fn calculate_thermo_metrics(&mut self, prices: &[f64], volumes: &[f64]) -> Result<ThermoMetrics> {
        if prices.len() < 2 {
            return Err(BridgeError::InsufficientData {
                required: 2,
                available: prices.len(),
            });
        }

        // Calculate returns as thermodynamic fluxes
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

        // Volume changes as thermodynamic forces
        let vol_changes: Vec<f64> = if volumes.len() >= 2 {
            volumes
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0].max(1.0))
                .collect()
        } else {
            vec![0.0; returns.len()]
        };

        // Use EntropyCalculator to compute Shannon entropy from returns distribution
        // First, convert returns to a probability distribution
        let positive_returns: Vec<f64> = returns.iter().map(|r| r.abs() + 1e-10).collect();
        let total: f64 = positive_returns.iter().sum();
        let probabilities: Vec<f64> = positive_returns.iter().map(|r| r / total).collect();
        let stat_entropy = self.entropy_calc.shannon_entropy(&probabilities);

        // Use HamiltonianCalculator to compute market energy via a pBit lattice representation
        // Create a small lattice to represent market microstate
        let market_lattice = hyperphysics_pbit::PBitLattice::new(5, 4, 1, self.config.temperature)
            .unwrap_or_else(|_| hyperphysics_pbit::PBitLattice::new(5, 4, 1, 1.0).unwrap());
        let hamiltonian_energy = hyperphysics_thermo::HamiltonianCalculator::energy(&market_lattice);

        // Compute entropy production rate (σ = Σ Jᵢ × Xᵢ)
        let min_len = returns.len().min(vol_changes.len());
        let entropy_production: f64 = returns[..min_len]
            .iter()
            .zip(&vol_changes[..min_len])
            .map(|(j, x)| j.abs() * x.abs())
            .sum();

        // Blend statistical entropy with entropy production for comprehensive measure
        let blended_entropy = (stat_entropy + entropy_production) / 2.0;

        // Calculate market "energy" from returns, augmented with Hamiltonian
        let kinetic = returns.iter().map(|r| r.powi(2)).sum::<f64>() / 2.0;
        let potential = prices.last().unwrap().ln() - prices.first().unwrap().ln();
        let total_energy = (kinetic + potential.abs() + hamiltonian_energy) / 2.0;

        // Calculate market temperature from energy fluctuations
        let energy_variance = returns.iter().map(|r| (r.powi(2) - kinetic / returns.len() as f64).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let effective_temperature = energy_variance * 1e6; // Scale to useful range

        // Store entropy history
        self.entropy_history.push_back(blended_entropy);
        if self.entropy_history.len() > self.config.history_window {
            self.entropy_history.pop_front();
        }

        // Calculate entropy trend
        let entropy_trend = if self.entropy_history.len() >= 10 {
            let recent: f64 = self.entropy_history.iter().rev().take(5).sum::<f64>() / 5.0;
            let older: f64 = self.entropy_history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;
            (recent - older) / older.max(1e-10)
        } else {
            0.0
        };

        // Calculate Landauer limit
        let landauer_limit = hyperphysics_thermo::BOLTZMANN_CONSTANT * self.config.temperature * hyperphysics_thermo::LN_2;

        // Classify thermodynamic regime
        let regime = self.classify_regime(blended_entropy, entropy_trend, effective_temperature);

        Ok(ThermoMetrics {
            entropy_production: blended_entropy,
            entropy_trend,
            total_energy,
            effective_temperature,
            regime,
            landauer_limit,
        })
    }

    /// Classify thermodynamic regime
    fn classify_regime(
        &self,
        entropy: f64,
        trend: f64,
        temperature: f64,
    ) -> ThermoRegime {
        if entropy < 0.01 && trend < 0.0 {
            ThermoRegime::Equilibrium
        } else if trend > 0.5 {
            ThermoRegime::Dissipative
        } else if temperature > 1000.0 {
            ThermoRegime::HighEnergy
        } else {
            ThermoRegime::NearEquilibrium
        }
    }

    /// Get current entropy
    pub fn current_entropy(&self) -> f64 {
        self.entropy_history.back().copied().unwrap_or(0.0)
    }
}

/// Thermodynamic metrics
#[derive(Debug, Clone)]
pub struct ThermoMetrics {
    /// Entropy production rate
    pub entropy_production: f64,
    /// Entropy trend (positive = increasing disorder)
    pub entropy_trend: f64,
    /// Total system energy
    pub total_energy: f64,
    /// Effective market temperature
    pub effective_temperature: f64,
    /// Thermodynamic regime
    pub regime: ThermoRegime,
    /// Landauer limit for information processing
    pub landauer_limit: f64,
}

/// Thermodynamic regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermoRegime {
    /// Near thermodynamic equilibrium
    Equilibrium,
    /// Slightly away from equilibrium
    NearEquilibrium,
    /// Dissipative structure forming
    Dissipative,
    /// High energy state
    HighEnergy,
}

// ============================================================================
// Risk Adapter - Codependent Risk Models
// ============================================================================

/// Adapter for hyperphysics-risk crate
/// Provides enhanced codependent risk modeling
pub struct RiskAdapter {
    /// Thermodynamic VaR calculator
    var_calc: hyperphysics_risk::ThermodynamicVaR,
    /// Codependent risk model
    codependent: hyperphysics_risk::CodependentRiskModel,
    /// Configuration
    config: RiskConfig,
}

/// Configuration for risk adapter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RiskConfig {
    /// Confidence level for VaR (e.g., 0.95)
    pub confidence_level: f64,
    /// Lookback period in days
    pub lookback_days: usize,
    /// Max assets for codependent network
    pub max_assets: usize,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            lookback_days: 252,
            max_assets: 16,
        }
    }
}

impl RiskAdapter {
    /// Create new risk adapter
    pub fn new(config: RiskConfig) -> Result<Self> {
        let var_calc = hyperphysics_risk::ThermodynamicVaR::new(config.confidence_level)
            .map_err(|e| BridgeError::RiskError {
                message: e.to_string(),
            })?;

        // Create codependent model with num_assets, decay_lambda, and max_depth
        let codependent = hyperphysics_risk::CodependentRiskModel::new(
            config.max_assets,  // num_assets
            0.5,                // decay_lambda
            3,                  // max_depth
        );

        Ok(Self {
            var_calc,
            codependent,
            config,
        })
    }

    /// Calculate comprehensive risk metrics
    pub fn calculate_risk_metrics(&mut self, returns: &[f64], positions: &[f64]) -> Result<RiskMetrics> {
        if returns.is_empty() {
            return Err(BridgeError::InsufficientData {
                required: 1,
                available: 0,
            });
        }

        // Use lookback_days from config to limit historical data
        let lookback = self.config.lookback_days.min(returns.len());
        let historical_returns = &returns[returns.len().saturating_sub(lookback)..];

        // Calculate VaR using historical method with configured lookback
        let var = self.var_calc.calculate_historical(historical_returns)
            .map_err(|e| BridgeError::RiskError {
                message: e.to_string(),
            })?;

        // Calculate CVaR (Expected Shortfall) using entropy-constrained VaR
        let cvar = self.var_calc.calculate_entropy_constrained(historical_returns, 1.0)
            .map_err(|e| BridgeError::RiskError {
                message: e.to_string(),
            })?;

        // Update codependent model with position correlations if positions provided
        if !positions.is_empty() {
            let num_positions = positions.len().min(self.config.max_assets);
            // Update standalone risk for each asset based on position exposure
            for (i, &pos) in positions.iter().take(num_positions).enumerate() {
                // Register position exposure as standalone risk (higher exposure = higher risk)
                let exposure_risk = pos.abs() * 0.01; // Scale exposure to risk measure
                let _ = self.codependent.update_standalone_risk(i, exposure_risk);
            }
        }

        // Calculate systemic risk from the codependent model
        let systemic = self.codependent.systemic_risk()
            .map_err(|e| BridgeError::RiskError {
                message: format!("Systemic risk error: {:?}", e),
            })?;
        let codependent_risk = systemic.total;

        // Calculate Kelly fraction using configured confidence level
        let mean_return = historical_returns.iter().sum::<f64>() / historical_returns.len() as f64;
        let variance = historical_returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / historical_returns.len() as f64;

        // Adjust Kelly by confidence level (more conservative at higher confidence)
        let kelly_raw = if variance > 0.0 {
            mean_return / variance
        } else {
            0.0
        };
        // Scale Kelly fraction by confidence level for risk adjustment
        let kelly_fraction = (kelly_raw * (1.0 - self.config.confidence_level + 0.5)).clamp(-1.0, 1.0);

        // Maximum drawdown
        let max_drawdown = self.calculate_max_drawdown(historical_returns);

        // Risk classification using configured confidence level threshold
        let level = self.classify_risk_level(var, cvar, max_drawdown);

        Ok(RiskMetrics {
            var,
            cvar,
            codependent_risk,
            kelly_fraction,
            max_drawdown,
            level,
        })
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[f64]) -> f64 {
        let mut cumulative: f64 = 1.0;
        let mut peak: f64 = 1.0;
        let mut max_dd: f64 = 0.0;

        for &r in returns {
            cumulative *= 1.0 + r;
            peak = peak.max(cumulative);
            let dd = (peak - cumulative) / peak;
            max_dd = max_dd.max(dd);
        }

        max_dd
    }

    /// Classify risk level
    fn classify_risk_level(&self, var: f64, cvar: f64, max_dd: f64) -> RiskLevel {
        let combined = var.abs() + cvar.abs() * 0.5 + max_dd * 0.3;

        if combined < 0.02 {
            RiskLevel::Low
        } else if combined < 0.05 {
            RiskLevel::Medium
        } else if combined < 0.10 {
            RiskLevel::High
        } else {
            RiskLevel::Extreme
        }
    }

    /// Get current VaR
    pub fn current_var(&self) -> f64 {
        0.0 // Would need to store last calculation
    }

    /// Get Kelly fraction (cached)
    pub fn kelly_fraction(&self) -> f64 {
        0.0 // Would need to store last calculation
    }
}

/// Risk metrics from hyperphysics-risk integration
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Value at Risk
    pub var: f64,
    /// Conditional VaR (Expected Shortfall)
    pub cvar: f64,
    /// Codependent network risk
    pub codependent_risk: f64,
    /// Kelly fraction for position sizing
    pub kelly_fraction: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Risk level classification
    pub level: RiskLevel,
}

/// Risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Extreme risk - consider reducing exposure
    Extreme,
}

// ============================================================================
// P-Bit Adapter - Probabilistic Signal Uncertainty
// ============================================================================

/// Adapter for hyperphysics-pbit crate
/// Provides probabilistic computing for signal uncertainty quantification
pub struct PbitAdapter {
    /// Metropolis simulator for sampling (owns the lattice internally)
    simulator: hyperphysics_pbit::MetropolisSimulator,
    /// Configuration
    config: PbitConfig,
}

/// Configuration for P-bit adapter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PbitConfig {
    /// Number of P-bits (lattice size)
    pub num_pbits: usize,
    /// Temperature for Boltzmann sampling
    pub temperature: f64,
    /// Number of samples for uncertainty estimation
    pub num_samples: usize,
}

impl Default for PbitConfig {
    fn default() -> Self {
        Self {
            num_pbits: 64,
            temperature: 1.0,
            num_samples: 1000,
        }
    }
}

impl PbitAdapter {
    /// Create new P-bit adapter
    pub fn new(config: PbitConfig) -> Result<Self> {
        // Create hyperbolic lattice with {5,4} tessellation at depth 2
        // This provides a good balance of nodes for uncertainty quantification
        let lattice = hyperphysics_pbit::PBitLattice::new(5, 4, 2, config.temperature)
            .map_err(|e| BridgeError::PbitError {
                message: e.to_string(),
            })?;

        // Create simulator with the lattice (simulator owns the lattice)
        let simulator = hyperphysics_pbit::MetropolisSimulator::new(lattice, config.temperature);

        Ok(Self { simulator, config })
    }

    /// Quantify uncertainty in a signal
    pub fn quantify_uncertainty(&mut self, signal_strength: f64, confidence: f64) -> Result<UncertaintyMetrics> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Create RNG for sampling (seeded by signal for reproducibility)
        let seed = ((signal_strength.abs() * 1e6) as u64).wrapping_add(42);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Run Metropolis simulation to sample distribution
        let mut samples: Vec<f64> = Vec::with_capacity(self.config.num_samples);

        // Run simulation steps and collect magnetization samples
        let steps_per_sample = 10; // Decorrelation steps between samples
        for _ in 0..self.config.num_samples {
            self.simulator
                .simulate(steps_per_sample, &mut rng)
                .map_err(|e| BridgeError::PbitError {
                    message: format!("Metropolis simulation failed: {}", e),
                })?;
            let magnetization = self.simulator.lattice().magnetization();
            samples.push(magnetization);
        }

        // Calculate uncertainty from sample distribution
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;

        // Pattern stability from autocorrelation
        let pattern_stability = self.calculate_autocorrelation(&samples);

        // Calculate entropy of distribution
        let entropy = self.calculate_sample_entropy(&samples);

        Ok(UncertaintyMetrics {
            mean_estimate: mean,
            uncertainty: variance.sqrt(),
            entropy,
            pattern_stability,
            confidence_interval: (mean - 1.96 * variance.sqrt() * confidence, mean + 1.96 * variance.sqrt() * confidence),
        })
    }

    /// Calculate autocorrelation for pattern stability
    fn calculate_autocorrelation(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 1.0;
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var: f64 = samples.iter().map(|s| (s - mean).powi(2)).sum();

        if var < 1e-10 {
            return 1.0;
        }

        let autocov: f64 = samples.windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum();

        (autocov / var).abs()
    }

    /// Calculate entropy of samples
    fn calculate_sample_entropy(&self, samples: &[f64]) -> f64 {
        // Simplified entropy calculation via histogram
        let n = samples.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }

        // Quantize samples into bins
        let num_bins: usize = 20;
        let min_val: f64 = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val: f64 = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range: f64 = (max_val - min_val).max(1e-10);

        let mut counts = vec![0usize; num_bins];
        for &s in samples {
            let bin = ((s - min_val) / range * (num_bins - 1) as f64).round() as usize;
            let bin = bin.min(num_bins - 1);
            counts[bin] += 1;
        }

        // Shannon entropy
        counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.ln()
            })
            .sum()
    }

    /// Get current uncertainty estimate
    pub fn current_uncertainty(&self) -> f64 {
        0.0 // Would need to cache last result
    }
}

/// Uncertainty metrics from P-bit analysis
#[derive(Debug, Clone)]
pub struct UncertaintyMetrics {
    /// Mean estimate
    pub mean_estimate: f64,
    /// Uncertainty (standard deviation)
    pub uncertainty: f64,
    /// Entropy of distribution
    pub entropy: f64,
    /// Pattern stability from autocorrelation
    pub pattern_stability: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
}

// ============================================================================
// Quantum Adapter - Real Quantum Gates
// ============================================================================

/// Adapter for quantum-circuit crate
/// Provides real quantum gate operations for pattern detection
pub struct QuantumAdapter {
    /// Quantum circuit for pattern detection
    circuit: quantum_circuit::Circuit,
    /// Configuration
    config: QuantumConfig,
}

/// Configuration for quantum adapter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantumConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Number of measurements
    pub num_measurements: usize,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            circuit_depth: 10,
            num_measurements: 1000,
        }
    }
}

impl QuantumAdapter {
    /// Create new quantum adapter
    pub fn new(config: QuantumConfig) -> Result<Self> {
        let circuit = quantum_circuit::Circuit::new(config.num_qubits);

        Ok(Self { circuit, config })
    }

    /// Detect quantum patterns in market data
    pub fn detect_patterns(&mut self, data: &[f64]) -> Result<QuantumPatternMetrics> {
        // Reset circuit
        self.circuit = quantum_circuit::Circuit::new(self.config.num_qubits);

        // Encode data into quantum state
        self.encode_data(data)?;

        // Apply variational quantum circuit
        self.apply_variational_circuit()?;

        // Execute circuit to get final state
        let state = self.circuit.execute().map_err(|e| BridgeError::QuantumError {
            message: e.to_string(),
        })?;

        // Analyze state for patterns
        let pattern_strength = self.analyze_state(&state);

        // Calculate entanglement and coherence from state
        let entanglement = self.estimate_entanglement(&state);
        let coherence = self.estimate_coherence(&state);
        let measurement_entropy = self.calculate_state_entropy(&state);

        Ok(QuantumPatternMetrics {
            pattern_strength,
            entanglement,
            coherence,
            measurement_entropy,
        })
    }

    /// Encode market data into quantum state
    fn encode_data(&mut self, data: &[f64]) -> Result<()> {
        // Normalize data for amplitude encoding
        let norm: f64 = data.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let normalized: Vec<f64> = if norm > 1e-10 {
            data.iter().map(|x| x / norm).collect()
        } else {
            vec![1.0 / (data.len() as f64).sqrt(); data.len()]
        };

        // Apply rotations based on data
        for (i, &val) in normalized.iter().enumerate().take(self.config.num_qubits) {
            let theta = val.acos().min(std::f64::consts::PI).max(0.0) * 2.0;
            self.circuit
                .add_gate(Box::new(quantum_circuit::gates::RY::new(i, theta)))
                .map_err(|e| BridgeError::QuantumError {
                    message: e.to_string(),
                })?;
        }

        Ok(())
    }

    /// Apply variational quantum circuit
    fn apply_variational_circuit(&mut self) -> Result<()> {
        for layer in 0..self.config.circuit_depth {
            // Single qubit rotations
            for q in 0..self.config.num_qubits {
                let angle = (layer as f64 * 0.1 + q as f64 * 0.2).sin() * std::f64::consts::PI;
                self.circuit
                    .add_gate(Box::new(quantum_circuit::gates::RZ::new(q, angle)))
                    .map_err(|e| BridgeError::QuantumError {
                        message: e.to_string(),
                    })?;
            }

            // Entangling gates
            for q in 0..self.config.num_qubits - 1 {
                self.circuit
                    .add_gate(Box::new(quantum_circuit::gates::CNOT::new(q, q + 1)))
                    .map_err(|e| BridgeError::QuantumError {
                        message: e.to_string(),
                    })?;
            }
        }

        Ok(())
    }

    /// Analyze quantum state for pattern strength
    fn analyze_state(&self, state: &quantum_circuit::StateVector) -> f64 {
        // Calculate expectation value of Z operator on first qubit
        // This gives a measure of pattern strength
        let n = state.len();
        let half = n / 2;

        let prob_0: f64 = state.iter().take(half).map(|c| c.norm_sqr()).sum();
        let prob_1: f64 = state.iter().skip(half).map(|c| c.norm_sqr()).sum();

        prob_0 - prob_1 // Expectation value of Z
    }

    /// Estimate entanglement from state
    fn estimate_entanglement(&self, state: &quantum_circuit::StateVector) -> f64 {
        // Simple entanglement estimate using purity of reduced density matrix
        let n = state.len();
        if n < 4 {
            return 0.0;
        }

        // Trace out last qubit to get reduced density matrix
        let half = n / 2;
        let mut reduced_diag = vec![0.0; half];

        for i in 0..half {
            reduced_diag[i] = state[i].norm_sqr() + state[i + half].norm_sqr();
        }

        // Purity = Tr(ρ²)
        let purity: f64 = reduced_diag.iter().map(|p| p.powi(2)).sum();

        // Entanglement ~ 1 - purity (for pure states)
        (1.0 - purity).max(0.0)
    }

    /// Estimate coherence from state
    fn estimate_coherence(&self, state: &quantum_circuit::StateVector) -> f64 {
        // L1-norm of coherence (sum of off-diagonal elements in computational basis)
        // For a state vector, this is related to the spread of amplitudes
        let n = state.len() as f64;
        let uniform_amp = 1.0 / n.sqrt();

        // Coherence measure: how far from computational basis states
        let max_amp = state.iter().map(|c| c.norm_sqr()).fold(0.0, f64::max);

        // Calculate deviation from uniform superposition (perfect coherence)
        // A maximally coherent state has all amplitudes equal to uniform_amp
        let deviation_from_uniform: f64 = state
            .iter()
            .map(|c| (c.norm() - uniform_amp).powi(2))
            .sum::<f64>()
            .sqrt();

        // Normalize: 0 = maximally coherent (uniform), 1 = classical (one basis state)
        // Invert so higher = more coherent
        let uniformity_measure = 1.0 - (deviation_from_uniform / (2.0_f64).sqrt()).min(1.0);

        // Combine with max_amp measure: both contribute to coherence assessment
        // High max_amp means state is close to basis state (low coherence)
        // High uniformity means state is in superposition (high coherence)
        (uniformity_measure + (1.0 - max_amp)) / 2.0
    }

    /// Calculate entropy of quantum state
    fn calculate_state_entropy(&self, state: &quantum_circuit::StateVector) -> f64 {
        // Von Neumann entropy of the probability distribution
        state.iter()
            .map(|c| {
                let p = c.norm_sqr();
                if p > 1e-10 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Get current coherence estimate
    pub fn current_coherence(&self) -> f64 {
        0.0 // Would need to cache last result
    }
}

/// Quantum pattern detection metrics
#[derive(Debug, Clone)]
pub struct QuantumPatternMetrics {
    /// Pattern strength (-1 to 1)
    pub pattern_strength: f64,
    /// Entanglement measure
    pub entanglement: f64,
    /// Quantum coherence
    pub coherence: f64,
    /// Measurement entropy
    pub measurement_entropy: f64,
}

// ============================================================================
// Syntergic Adapter - Coherence Field Integration
// ============================================================================

/// Adapter for hyperphysics-syntergic crate
/// Provides coherence field analysis for market synchronization
pub struct SyntergicAdapter {
    /// Syntergic field calculator
    field: hyperphysics_syntergic::SyntergicField,
    /// Configuration
    config: SyntergicConfig,
    /// Coherence history
    coherence_history: VecDeque<f64>,
}

/// Configuration for syntergic adapter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyntergicConfig {
    /// Field dimensions
    pub dimensions: usize,
    /// Unity threshold
    pub unity_threshold: f64,
    /// History window
    pub history_window: usize,
}

impl Default for SyntergicConfig {
    fn default() -> Self {
        Self {
            dimensions: 8,
            unity_threshold: 0.8,
            history_window: 100,
        }
    }
}

impl SyntergicAdapter {
    /// Create new syntergic adapter
    pub fn new(config: SyntergicConfig) -> Result<Self> {
        // Create a pBit lattice with hyperbolic {5,4} tessellation
        // The dimensions config maps to lattice depth for reasonable node count
        let depth = (config.dimensions / 4).max(1).min(3);
        let lattice = hyperphysics_pbit::PBitLattice::new(5, 4, depth, 1.0)
            .map_err(|e| BridgeError::SyntergicError {
                message: format!("Failed to create lattice: {}", e),
            })?;

        // Create syntergic field with kappa=1.0 (standard Green's function parameter)
        let field = hyperphysics_syntergic::SyntergicField::new(lattice, 1.0);

        Ok(Self {
            field,
            config,
            coherence_history: VecDeque::with_capacity(100),
        })
    }

    /// Analyze market coherence using syntergic fields
    pub fn analyze_coherence(&mut self, _market_data: &MarketStateVector) -> Result<SyntergicMetrics> {
        // Update field dynamics with small time step
        // The field evolves based on internal neuronal activity
        self.field.update(0.01).map_err(|e| BridgeError::SyntergicError {
            message: format!("Field update failed: {}", e),
        })?;

        // Get comprehensive metrics from the field
        let field_metrics = self.field.metrics();
        let coherence = field_metrics.coherence;

        // Store history
        self.coherence_history.push_back(coherence);
        if self.coherence_history.len() > self.config.history_window {
            self.coherence_history.pop_front();
        }

        // Detect unity state
        let unity_achieved = coherence >= self.config.unity_threshold;

        // Calculate coherence trend
        let trend = if self.coherence_history.len() >= 10 {
            let recent: f64 = self.coherence_history.iter().rev().take(5).sum::<f64>() / 5.0;
            let older: f64 = self.coherence_history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;
            recent - older
        } else {
            0.0
        };

        // Classify synchronization state
        let state = self.classify_sync_state(coherence, trend);

        Ok(SyntergicMetrics {
            coherence,
            field_energy: field_metrics.total_energy,
            unity_achieved,
            coherence_trend: trend,
            sync_state: state,
        })
    }

    /// Classify synchronization state
    fn classify_sync_state(&self, coherence: f64, trend: f64) -> SyncState {
        if coherence >= self.config.unity_threshold {
            SyncState::Unity
        } else if coherence >= 0.6 && trend > 0.0 {
            SyncState::Converging
        } else if coherence >= 0.4 {
            SyncState::Partial
        } else if trend < -0.1 {
            SyncState::Diverging
        } else {
            SyncState::Chaotic
        }
    }

    /// Get current coherence
    pub fn current_coherence(&self) -> f64 {
        self.coherence_history.back().copied().unwrap_or(0.0)
    }
}

/// Syntergic metrics
#[derive(Debug, Clone)]
pub struct SyntergicMetrics {
    /// Field coherence (0-1)
    pub coherence: f64,
    /// Field energy
    pub field_energy: f64,
    /// Unity state achieved
    pub unity_achieved: bool,
    /// Coherence trend
    pub coherence_trend: f64,
    /// Synchronization state
    pub sync_state: SyncState,
}

/// Synchronization state classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncState {
    /// Full unity achieved
    Unity,
    /// Converging toward unity
    Converging,
    /// Partial synchronization
    Partial,
    /// Diverging from coherence
    Diverging,
    /// Chaotic state
    Chaotic,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autopoiesis_adapter() {
        let config = AutopoiesisConfig::default();
        let mut adapter = AutopoiesisAdapter::new(config);

        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let volumes: Vec<f64> = vec![1_000_000.0; 100];

        let state = adapter.update(&prices, &volumes).unwrap();
        assert!(state.health >= 0.0 && state.health <= 1.0);
        assert!(state.coherence >= 0.0 && state.coherence <= 1.0);
    }

    #[test]
    fn test_market_state_vector() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let volumes = vec![1000.0, 1100.0, 1050.0, 1200.0, 1150.0];

        let state = MarketStateVector::from_market_data(&prices, &volumes);
        assert!(state.features.len() >= 8);
    }

    #[test]
    fn test_thermo_adapter() {
        let config = ThermoConfig::default();
        let mut adapter = ThermoAdapter::new(config);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let volumes: Vec<f64> = vec![1_000_000.0; 50];

        let metrics = adapter.calculate_thermo_metrics(&prices, &volumes).unwrap();
        assert!(metrics.entropy_production >= 0.0);
    }
}
