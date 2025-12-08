//! Dynamical systems analysis for autopoietic phenomena
//!
//! This module provides tools for analyzing the dynamics of autopoietic systems,
//! including bifurcation detection, emergence monitoring, and stability analysis.
//!
//! ## Theoretical Foundations
//!
//! - **Bifurcation Theory**: Qualitative changes in system behavior as parameters vary
//! - **Self-Organized Criticality**: Systems naturally evolving to critical states
//! - **Emergence Detection**: Identifying higher-order patterns from component interactions
//!
//! ## SOC Integration (hyperphysics-geometry)
//!
//! Now integrates with `hyperphysics-geometry::SOCCoordinator` for criticality-driven
//! bifurcation detection. SOC provides:
//! - Branching ratio σ ≈ 1.0 at criticality
//! - Power-law avalanche distributions (τ ≈ 1.5)
//! - Automatic emergence detection at phase transitions
//!
//! ## References
//! - Strogatz (2014) "Nonlinear Dynamics and Chaos"
//! - Bak (1996) "How Nature Works: Self-Organized Criticality"
//! - Kauffman (1993) "The Origins of Order"

use std::collections::VecDeque;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::error::{AutopoiesisError, Result};
use hyperphysics_geometry::{SOCCoordinator, SOCModulation, SOCStats, LorentzVec4D};

/// Configuration for dynamical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsConfig {
    /// Window size for time series analysis
    pub window_size: usize,
    /// Threshold for bifurcation detection
    pub bifurcation_threshold: f64,
    /// Minimum eigenvalue gap for emergence detection
    pub emergence_eigenvalue_gap: f64,
    /// Sampling interval for dynamics
    pub sample_interval_ms: u64,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            bifurcation_threshold: 0.1,
            emergence_eigenvalue_gap: 0.5,
            sample_interval_ms: 100,
        }
    }
}

/// Core dynamics engine for autopoietic systems
///
/// Tracks system evolution, detects transitions, and monitors stability.
/// Now integrates with SOCCoordinator for criticality-driven bifurcation detection.
#[derive(Debug)]
pub struct AutopoieticDynamics {
    /// Configuration
    config: DynamicsConfig,
    /// Time series of state vectors
    state_history: VecDeque<DVector<f64>>,
    /// Time series of control parameters
    control_history: VecDeque<f64>,
    /// Detected bifurcation events
    bifurcations: Vec<BifurcationEvent>,
    /// Current Lyapunov exponent estimate
    lyapunov_estimate: f64,
    /// Dimension of state space
    state_dimension: usize,
    /// SOC coordinator for criticality-driven dynamics
    soc_coordinator: Option<SOCCoordinator>,
    /// History of SOC statistics for criticality tracking
    soc_history: VecDeque<SOCStats>,
}

impl AutopoieticDynamics {
    /// Create new dynamics engine with specified dimension
    pub fn new(config: DynamicsConfig, state_dimension: usize) -> Self {
        Self {
            config,
            state_history: VecDeque::with_capacity(1000),
            control_history: VecDeque::with_capacity(1000),
            bifurcations: Vec::new(),
            lyapunov_estimate: 0.0,
            state_dimension,
            soc_coordinator: None,
            soc_history: VecDeque::with_capacity(1000),
        }
    }

    /// Attach an SOC coordinator for criticality-driven dynamics
    ///
    /// When attached, bifurcation detection is enhanced with SOC metrics:
    /// - Bifurcations are more likely near σ = 1.0 (critical branching ratio)
    /// - Power-law exponent τ ≈ 1.5 indicates proximity to phase transition
    /// - Emergence events are detected when SOC avalanche distributions shift
    pub fn with_soc_coordinator(mut self, coordinator: SOCCoordinator) -> Self {
        self.soc_coordinator = Some(coordinator);
        self
    }

    /// Record SOC statistics and check for criticality-driven bifurcation
    ///
    /// SOC provides an alternative view of system dynamics:
    /// - Branching ratio σ → 1 indicates approach to criticality
    /// - Power-law exponent τ → 1.5 indicates optimal criticality
    /// - Large avalanches indicate potential phase transition
    pub fn record_soc_stats(&mut self, stats: SOCStats) -> Option<BifurcationEvent> {
        self.soc_history.push_back(stats.clone());

        // Maintain history size
        while self.soc_history.len() > self.config.window_size {
            self.soc_history.pop_front();
        }

        // Check for criticality-driven bifurcation
        self.detect_soc_bifurcation(&stats)
    }

    /// Detect bifurcation from SOC statistics
    ///
    /// A bifurcation is likely when:
    /// 1. σ crosses 1.0 (criticality threshold)
    /// 2. τ shifts significantly (power-law exponent change)
    /// 3. Avalanche size distribution changes shape
    fn detect_soc_bifurcation(&mut self, stats: &SOCStats) -> Option<BifurcationEvent> {
        if self.soc_history.len() < 2 {
            return None;
        }

        let prev_stats = &self.soc_history[self.soc_history.len() - 2];

        // Check for criticality crossing (σ crosses 1.0)
        let prev_subcritical = prev_stats.sigma_measured < 1.0;
        let curr_subcritical = stats.sigma_measured < 1.0;
        let criticality_crossed = prev_subcritical != curr_subcritical;

        // Check for power-law exponent shift
        let tau_shift = (stats.power_law_tau - prev_stats.power_law_tau).abs();
        let significant_tau_shift = tau_shift > 0.2;

        // Check for avalanche distribution change
        let avg_size_ratio = if prev_stats.avg_avalanche_size > 1e-10 {
            stats.avg_avalanche_size / prev_stats.avg_avalanche_size
        } else {
            1.0
        };
        let avalanche_shift = avg_size_ratio > 2.0 || avg_size_ratio < 0.5;

        if criticality_crossed || significant_tau_shift || avalanche_shift {
            let bifurcation_type = if criticality_crossed {
                if stats.sigma_measured > 1.0 {
                    BifurcationType::Hopf // Transition to oscillatory/chaotic
                } else {
                    BifurcationType::SaddleNode // Transition to stable attractor
                }
            } else if significant_tau_shift {
                BifurcationType::PeriodDoubling // Power-law change often precedes period doubling
            } else {
                BifurcationType::PitchforkSupercritical
            };

            let event = BifurcationEvent {
                timestamp: chrono::Utc::now(),
                control_parameter: stats.sigma_measured,
                bifurcation_type,
                variance_ratio: avg_size_ratio,
                eigenvalue_crossing: Some(stats.sigma_measured - 1.0),
            };

            self.bifurcations.push(event.clone());
            return Some(event);
        }

        None
    }

    /// Get current SOC modulation factor
    ///
    /// Returns a factor in [0, 1] indicating proximity to criticality.
    /// Used to modulate other system parameters based on SOC state.
    pub fn soc_modulation(&self) -> f64 {
        self.soc_history.back().map(|stats| {
            let sigma_dev = (stats.sigma_measured - 1.0).abs();
            let tau_dev = (stats.power_law_tau - 1.5).abs();

            // Gaussian modulation centered at criticality
            let sigma_factor = (-sigma_dev * sigma_dev / 0.02).exp();
            let tau_factor = (-tau_dev * tau_dev / 0.5).exp();

            sigma_factor * 0.7 + tau_factor * 0.3
        }).unwrap_or(0.5)
    }

    /// Check if system is at SOC criticality
    pub fn is_at_criticality(&self) -> bool {
        self.soc_history.back().map(|stats| stats.is_critical).unwrap_or(false)
    }

    /// Get SOC history for analysis
    pub fn soc_history(&self) -> &VecDeque<SOCStats> {
        &self.soc_history
    }

    /// Record a new state observation
    pub fn record_state(&mut self, state: DVector<f64>, control_param: f64) -> Result<()> {
        if state.len() != self.state_dimension {
            return Err(AutopoiesisError::NumericalError {
                operation: "record_state".to_string(),
                message: format!(
                    "State dimension mismatch: expected {}, got {}",
                    self.state_dimension,
                    state.len()
                ),
            });
        }

        self.state_history.push_back(state);
        self.control_history.push_back(control_param);

        // Maintain window size
        while self.state_history.len() > self.config.window_size {
            self.state_history.pop_front();
            self.control_history.pop_front();
        }

        // Update Lyapunov estimate if enough data
        if self.state_history.len() >= 10 {
            self.update_lyapunov_estimate();
        }

        Ok(())
    }

    /// Update Lyapunov exponent estimate from recent dynamics
    fn update_lyapunov_estimate(&mut self) {
        if self.state_history.len() < 2 {
            return;
        }

        // Estimate largest Lyapunov exponent from divergence of nearby trajectories
        let mut sum_log_divergence = 0.0;
        let mut count = 0;

        let states: Vec<_> = self.state_history.iter().collect();

        for i in 1..states.len() {
            let diff = states[i] - states[i - 1];
            let norm = diff.norm();

            if norm > 1e-10 {
                sum_log_divergence += norm.ln();
                count += 1;
            }
        }

        if count > 0 {
            self.lyapunov_estimate = sum_log_divergence / count as f64;
        }
    }

    /// Detect potential bifurcation from recent dynamics
    pub fn detect_bifurcation(&mut self) -> Option<BifurcationEvent> {
        if self.state_history.len() < self.config.window_size / 2 {
            return None;
        }

        // Compute variance in first and second half of window
        let half = self.state_history.len() / 2;
        let first_half: Vec<_> = self.state_history.iter().take(half).collect();
        let second_half: Vec<_> = self.state_history.iter().skip(half).collect();

        let var1 = self.compute_variance(&first_half);
        let var2 = self.compute_variance(&second_half);

        // Bifurcation indicated by significant variance change
        let variance_ratio = if var1 > 1e-10 { var2 / var1 } else { 1.0 };

        if (variance_ratio - 1.0).abs() > self.config.bifurcation_threshold {
            let control_param = self.control_history.back().copied().unwrap_or(0.0);

            let bifurcation_type = if variance_ratio > 1.0 {
                BifurcationType::PitchforkSupercritical
            } else {
                BifurcationType::SaddleNode
            };

            let event = BifurcationEvent {
                timestamp: chrono::Utc::now(),
                control_parameter: control_param,
                bifurcation_type,
                variance_ratio,
                eigenvalue_crossing: None,
            };

            self.bifurcations.push(event.clone());
            return Some(event);
        }

        None
    }

    /// Compute variance across state vectors
    fn compute_variance(&self, states: &[&DVector<f64>]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }

        let n = states.len() as f64;
        let dim = self.state_dimension;

        // Compute mean
        let mut mean = DVector::zeros(dim);
        for state in states {
            mean += *state;
        }
        mean /= n;

        // Compute variance
        let mut variance = 0.0;
        for state in states {
            let diff = *state - &mean;
            variance += diff.norm_squared();
        }
        variance /= n;

        variance
    }

    /// Get current Lyapunov exponent estimate
    pub fn lyapunov_exponent(&self) -> f64 {
        self.lyapunov_estimate
    }

    /// Check if system is in chaotic regime
    pub fn is_chaotic(&self) -> bool {
        self.lyapunov_estimate > 0.0
    }

    /// Get bifurcation history
    pub fn bifurcation_history(&self) -> &[BifurcationEvent] {
        &self.bifurcations
    }

    /// Get current state dimension
    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }
}

/// Types of bifurcations in dynamical systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BifurcationType {
    /// Saddle-node: fixed point appears/disappears
    SaddleNode,
    /// Transcritical: exchange of stability
    Transcritical,
    /// Pitchfork (supercritical): symmetric branching
    PitchforkSupercritical,
    /// Pitchfork (subcritical): unstable branching
    PitchforkSubcritical,
    /// Hopf: limit cycle emerges
    Hopf,
    /// Period doubling: route to chaos
    PeriodDoubling,
}

/// Record of a detected bifurcation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationEvent {
    /// When bifurcation was detected
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Control parameter value at bifurcation
    pub control_parameter: f64,
    /// Type of bifurcation
    pub bifurcation_type: BifurcationType,
    /// Variance ratio indicating bifurcation strength
    pub variance_ratio: f64,
    /// Eigenvalue that crossed zero (if computed)
    pub eigenvalue_crossing: Option<f64>,
}

/// Detector for bifurcation points in parameter space
///
/// Uses eigenvalue analysis and continuation methods to identify
/// qualitative changes in system behavior.
///
/// ## References
/// - Kuznetsov (2004) "Elements of Applied Bifurcation Theory"
#[derive(Debug)]
pub struct BifurcationDetector {
    /// Jacobian matrix evaluator
    jacobian_history: VecDeque<DMatrix<f64>>,
    /// Eigenvalue history
    eigenvalue_history: VecDeque<DVector<f64>>,
    /// Detection threshold for eigenvalue crossing
    crossing_threshold: f64,
    /// Maximum history length
    max_history: usize,
}

impl Default for BifurcationDetector {
    fn default() -> Self {
        Self::new(1e-6, 100)
    }
}

impl BifurcationDetector {
    /// Create new bifurcation detector
    pub fn new(crossing_threshold: f64, max_history: usize) -> Self {
        Self {
            jacobian_history: VecDeque::with_capacity(max_history),
            eigenvalue_history: VecDeque::with_capacity(max_history),
            crossing_threshold,
            max_history,
        }
    }

    /// Record Jacobian and compute eigenvalues
    pub fn record_jacobian(&mut self, jacobian: DMatrix<f64>) -> Vec<f64> {
        // Compute eigenvalues
        let eigenvalues: Vec<f64> = jacobian
            .clone()
            .symmetric_eigenvalues()
            .iter()
            .copied()
            .collect();

        self.jacobian_history.push_back(jacobian);
        self.eigenvalue_history
            .push_back(DVector::from_vec(eigenvalues.clone()));

        // Maintain history size
        while self.jacobian_history.len() > self.max_history {
            self.jacobian_history.pop_front();
            self.eigenvalue_history.pop_front();
        }

        eigenvalues
    }

    /// Detect eigenvalue zero crossing (bifurcation indicator)
    pub fn detect_zero_crossing(&self) -> Option<(usize, f64)> {
        if self.eigenvalue_history.len() < 2 {
            return None;
        }

        let prev = self.eigenvalue_history.iter().nth_back(1)?;
        let curr = self.eigenvalue_history.back()?;

        // Check each eigenvalue for sign change
        for i in 0..prev.len().min(curr.len()) {
            let prev_val = prev[i];
            let curr_val = curr[i];

            // Sign change through zero
            if prev_val * curr_val < 0.0 {
                // Linear interpolation to estimate crossing point
                let crossing_estimate = prev_val / (prev_val - curr_val);
                return Some((i, crossing_estimate));
            }

            // Very close to zero
            if curr_val.abs() < self.crossing_threshold {
                return Some((i, curr_val));
            }
        }

        None
    }

    /// Classify bifurcation type from eigenvalue structure
    pub fn classify_bifurcation(&self, eigenvalues: &[f64]) -> Option<BifurcationType> {
        let near_zero: Vec<_> = eigenvalues
            .iter()
            .filter(|&&e| e.abs() < self.crossing_threshold * 100.0)
            .collect();

        if near_zero.is_empty() {
            return None;
        }

        // Count positive and negative eigenvalues
        let positive = eigenvalues.iter().filter(|&&e| e > 0.0).count();
        let _negative = eigenvalues.iter().filter(|&&e| e < 0.0).count();

        // Simple classification heuristics
        if near_zero.len() == 1 {
            if positive == 0 {
                Some(BifurcationType::SaddleNode)
            } else {
                Some(BifurcationType::Transcritical)
            }
        } else if near_zero.len() == 2 {
            // Could be Hopf or pitchfork
            Some(BifurcationType::PitchforkSupercritical)
        } else {
            None
        }
    }

    /// Get stability assessment (all eigenvalues negative = stable)
    pub fn is_stable(&self) -> bool {
        self.eigenvalue_history
            .back()
            .map(|ev| ev.iter().all(|&e| e < 0.0))
            .unwrap_or(true)
    }
}

/// Monitor for emergent phenomena in complex systems
///
/// Detects emergence through information-theoretic and dynamical signatures:
/// - Eigenvalue gap emergence (collective modes)
/// - Information integration increase
/// - Order parameter formation
///
/// ## References
/// - Tononi (2008) "Consciousness as Integrated Information"
/// - Friston (2010) "The free-energy principle"
#[derive(Debug)]
pub struct EmergenceMonitor {
    /// History of covariance matrices
    covariance_history: VecDeque<DMatrix<f64>>,
    /// History of eigenvalue spectra
    spectrum_history: VecDeque<Vec<f64>>,
    /// Emergence events detected
    emergence_events: Vec<EmergenceEvent>,
    /// Configuration
    eigenvalue_gap_threshold: f64,
    /// Maximum history
    max_history: usize,
}

impl Default for EmergenceMonitor {
    fn default() -> Self {
        Self::new(0.5, 100)
    }
}

impl EmergenceMonitor {
    /// Create new emergence monitor
    pub fn new(eigenvalue_gap_threshold: f64, max_history: usize) -> Self {
        Self {
            covariance_history: VecDeque::with_capacity(max_history),
            spectrum_history: VecDeque::with_capacity(max_history),
            emergence_events: Vec::new(),
            eigenvalue_gap_threshold,
            max_history,
        }
    }

    /// Record system state and check for emergence
    pub fn record_and_check(&mut self, covariance: DMatrix<f64>) -> Option<EmergenceEvent> {
        // Compute eigenspectrum
        let mut eigenvalues: Vec<f64> = covariance
            .clone()
            .symmetric_eigenvalues()
            .iter()
            .copied()
            .collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let emergence = self.detect_emergence(&eigenvalues);

        self.covariance_history.push_back(covariance);
        self.spectrum_history.push_back(eigenvalues);

        // Maintain history
        while self.covariance_history.len() > self.max_history {
            self.covariance_history.pop_front();
            self.spectrum_history.pop_front();
        }

        if let Some(event) = &emergence {
            self.emergence_events.push(event.clone());
        }

        emergence
    }

    /// Detect emergence from eigenvalue structure
    fn detect_emergence(&self, eigenvalues: &[f64]) -> Option<EmergenceEvent> {
        if eigenvalues.len() < 2 {
            return None;
        }

        // Check for dominant mode emergence (large gap between largest eigenvalues)
        let total: f64 = eigenvalues.iter().sum();
        if total < 1e-10 {
            return None;
        }

        let normalized: Vec<f64> = eigenvalues.iter().map(|e| e / total).collect();

        // Participation ratio: measure of how many modes contribute
        let participation: f64 = 1.0 / normalized.iter().map(|p| p * p).sum::<f64>();

        // Gap between first and second eigenvalue
        let gap = if normalized.len() >= 2 {
            normalized[0] - normalized[1]
        } else {
            0.0
        };

        // Emergence indicated by:
        // 1. Large gap (collective mode dominates)
        // 2. Low participation ratio (few effective dimensions)
        if gap > self.eigenvalue_gap_threshold || participation < eigenvalues.len() as f64 / 2.0 {
            Some(EmergenceEvent {
                timestamp: chrono::Utc::now(),
                emergence_type: if gap > self.eigenvalue_gap_threshold {
                    EmergenceType::CollectiveMode
                } else {
                    EmergenceType::DimensionalReduction
                },
                eigenvalue_gap: gap,
                participation_ratio: participation,
                dominant_eigenvalue: normalized[0],
            })
        } else {
            None
        }
    }

    /// Get emergence history
    pub fn emergence_history(&self) -> &[EmergenceEvent] {
        &self.emergence_events
    }

    /// Compute current effective dimensionality
    pub fn effective_dimensionality(&self) -> f64 {
        self.spectrum_history
            .back()
            .map(|spec| {
                let total: f64 = spec.iter().sum();
                if total < 1e-10 {
                    return 0.0;
                }
                let normalized: Vec<f64> = spec.iter().map(|e| e / total).collect();
                1.0 / normalized.iter().map(|p| p * p).sum::<f64>()
            })
            .unwrap_or(0.0)
    }
}

/// Types of emergence phenomena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceType {
    /// Single collective mode dominates
    CollectiveMode,
    /// Dimensional reduction (information compression)
    DimensionalReduction,
    /// Synchronization across components
    Synchronization,
    /// Pattern formation
    PatternFormation,
}

/// Record of detected emergence event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceEvent {
    /// When emergence was detected
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Type of emergence
    pub emergence_type: EmergenceType,
    /// Eigenvalue gap that triggered detection
    pub eigenvalue_gap: f64,
    /// Participation ratio at detection
    pub participation_ratio: f64,
    /// Value of dominant eigenvalue
    pub dominant_eigenvalue: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_autopoietic_dynamics_recording() {
        let config = DynamicsConfig::default();
        let mut dynamics = AutopoieticDynamics::new(config, 3);

        for i in 0..50 {
            let state = DVector::from_vec(vec![i as f64, i as f64 * 0.5, i as f64 * 0.1]);
            dynamics.record_state(state, i as f64 * 0.01).unwrap();
        }

        assert!(dynamics.state_history.len() <= dynamics.config.window_size);
    }

    #[test]
    fn test_bifurcation_detector_stability() {
        let mut detector = BifurcationDetector::default();

        // Stable system: all negative eigenvalues
        let stable_jacobian = DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, -2.0, -3.0]));
        detector.record_jacobian(stable_jacobian);

        assert!(detector.is_stable());

        // Unstable system: one positive eigenvalue
        let unstable_jacobian = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, -2.0, -3.0]));
        detector.record_jacobian(unstable_jacobian);

        assert!(!detector.is_stable());
    }

    #[test]
    fn test_emergence_monitor_collective_mode() {
        let mut monitor = EmergenceMonitor::new(0.3, 100);

        // Covariance with dominant mode
        let mut cov = DMatrix::zeros(5, 5);
        cov[(0, 0)] = 10.0; // Dominant
        cov[(1, 1)] = 1.0;
        cov[(2, 2)] = 1.0;
        cov[(3, 3)] = 1.0;
        cov[(4, 4)] = 1.0;

        let emergence = monitor.record_and_check(cov);
        assert!(emergence.is_some());

        let event = emergence.unwrap();
        assert_eq!(event.emergence_type, EmergenceType::CollectiveMode);
    }

    #[test]
    fn test_effective_dimensionality() {
        let mut monitor = EmergenceMonitor::new(0.3, 100);

        // All equal eigenvalues: max dimensionality
        let uniform_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]));
        monitor.record_and_check(uniform_cov);

        let dim = monitor.effective_dimensionality();
        assert_relative_eq!(dim, 4.0, epsilon = 1e-10);

        // Single dominant eigenvalue: low dimensionality
        let mut dominant_cov = DMatrix::zeros(4, 4);
        dominant_cov[(0, 0)] = 100.0;
        dominant_cov[(1, 1)] = 0.01;
        dominant_cov[(2, 2)] = 0.01;
        dominant_cov[(3, 3)] = 0.01;

        monitor.record_and_check(dominant_cov);

        let dim = monitor.effective_dimensionality();
        assert!(dim < 2.0);
    }
}
