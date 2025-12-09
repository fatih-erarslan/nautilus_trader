//! pBit-Enhanced Panarchy Phase Analyzer
//!
//! Maps adaptive cycle phases to Ising model dynamics for probabilistic
//! phase transition detection.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! The Panarchy cycle phases map to Ising magnetization:
//! - **Growth (r)**: M > 0, aligned spins, momentum building
//! - **Conservation (K)**: M ≈ 1, fully magnetized, stable regime
//! - **Release (Ω)**: M decreasing, disorder increasing, critical transition
//! - **Reorganization (α)**: M ≈ 0, maximum disorder, new structure forming
//!
//! Phase transitions occur near critical temperature T_c = 2.269 (2D Ising)

use crate::{MarketPhase, PCRComponents, PanarchyError, Result};

/// Critical temperature for 2D Ising model (Onsager solution)
/// T_c = 2/ln(1 + √2) ≈ 2.269185314213022
pub const ISING_CRITICAL_TEMP: f64 = 2.269185314213022;

/// pBit Panarchy configuration
#[derive(Debug, Clone)]
pub struct PBitPanarchyConfig {
    /// Base temperature for Boltzmann dynamics
    pub temperature: f64,
    /// Coupling strength between market indicators
    pub coupling_j: f64,
    /// Phase transition threshold
    pub transition_threshold: f64,
    /// Number of Monte Carlo samples
    pub num_samples: usize,
}

impl Default for PBitPanarchyConfig {
    fn default() -> Self {
        Self {
            temperature: ISING_CRITICAL_TEMP,
            coupling_j: 1.0,
            transition_threshold: 0.5,
            num_samples: 1000,
        }
    }
}

/// pBit-enhanced phase detector using Ising dynamics
#[derive(Debug, Clone)]
pub struct PBitPhaseDetector {
    config: PBitPanarchyConfig,
    /// Historical magnetization for trend detection
    magnetization_history: Vec<f64>,
    /// Phase transition indicators
    transition_indicators: Vec<TransitionIndicator>,
}

/// Phase transition indicator
#[derive(Debug, Clone)]
pub struct TransitionIndicator {
    /// From phase
    pub from_phase: MarketPhase,
    /// To phase
    pub to_phase: MarketPhase,
    /// Transition probability
    pub probability: f64,
    /// Time until expected transition
    pub eta_bars: usize,
}

impl PBitPhaseDetector {
    /// Create new phase detector
    pub fn new(config: PBitPanarchyConfig) -> Self {
        Self {
            config,
            magnetization_history: Vec::with_capacity(1000),
            transition_indicators: Vec::new(),
        }
    }

    /// Map PCR components to Ising magnetization
    pub fn pcr_to_magnetization(&self, pcr: &PCRComponents) -> f64 {
        // Combine PCR into effective field
        let h_eff = pcr.potential * 0.4 + pcr.connectedness * 0.3 + pcr.resilience * 0.3;
        
        // Boltzmann probability for positive spin
        let p_up = self.sigmoid(h_eff / self.config.temperature);
        
        // Magnetization: M = 2*P(↑) - 1 ∈ [-1, 1]
        2.0 * p_up - 1.0
    }

    /// Sigmoid activation (pBit probability)
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Detect current phase from magnetization
    pub fn detect_phase(&self, magnetization: f64, susceptibility: f64) -> MarketPhase {
        // Map magnetization to Panarchy phases
        if magnetization > 0.7 {
            MarketPhase::Conservation // K phase - high order
        } else if magnetization > 0.3 {
            MarketPhase::Growth // r phase - building momentum
        } else if magnetization < -0.3 && susceptibility > 1.0 {
            MarketPhase::Release // Ω phase - critical transition
        } else {
            MarketPhase::Reorganization // α phase - disorder/renewal
        }
    }

    /// Calculate magnetic susceptibility (phase transition indicator)
    /// χ = dM/dH = β(1 - M²) in mean-field approximation
    pub fn susceptibility(&self, magnetization: f64) -> f64 {
        let beta = 1.0 / self.config.temperature;
        beta * (1.0 - magnetization.powi(2))
    }

    /// Analyze phase transitions using pBit dynamics
    pub fn analyze(&mut self, pcr_series: &[PCRComponents]) -> Result<PBitPhaseResult> {
        if pcr_series.is_empty() {
            return Err(PanarchyError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        // Calculate magnetization series
        let magnetizations: Vec<f64> = pcr_series
            .iter()
            .map(|pcr| self.pcr_to_magnetization(pcr))
            .collect();

        // Store history
        self.magnetization_history.extend(magnetizations.iter().cloned());
        if self.magnetization_history.len() > 10000 {
            self.magnetization_history.drain(0..5000);
        }

        // Current state
        let current_m = *magnetizations.last().unwrap_or(&0.0);
        let current_chi = self.susceptibility(current_m);
        let current_phase = self.detect_phase(current_m, current_chi);

        // Calculate order parameter (absolute magnetization)
        let order_parameter = current_m.abs();

        // Entropy from magnetization: S = -Σ p_i ln(p_i)
        let p_up = (current_m + 1.0) / 2.0;
        let entropy = if p_up > 0.0 && p_up < 1.0 {
            -p_up * p_up.ln() - (1.0 - p_up) * (1.0 - p_up).ln()
        } else {
            0.0
        };

        // Detect phase transitions
        let transitions = self.detect_transitions(&magnetizations);

        // Calculate correlation length (diverges at T_c)
        let temp_ratio = self.config.temperature / ISING_CRITICAL_TEMP;
        let correlation_length = if (temp_ratio - 1.0).abs() > 0.01 {
            1.0 / (temp_ratio - 1.0).abs()
        } else {
            100.0 // Cap near T_c
        };

        Ok(PBitPhaseResult {
            current_phase,
            magnetization: current_m,
            susceptibility: current_chi,
            order_parameter,
            entropy,
            correlation_length,
            transitions,
            phase_confidence: self.phase_confidence(current_m, current_chi),
        })
    }

    /// Detect phase transitions from magnetization changes
    fn detect_transitions(&self, magnetizations: &[f64]) -> Vec<TransitionIndicator> {
        let mut transitions = Vec::new();

        if magnetizations.len() < 10 {
            return transitions;
        }

        // Calculate rate of change
        let recent = &magnetizations[magnetizations.len() - 10..];
        let dm_dt = (recent.last().unwrap() - recent.first().unwrap()) / 10.0;

        let current_m = *magnetizations.last().unwrap();
        let current_chi = self.susceptibility(current_m);
        let current_phase = self.detect_phase(current_m, current_chi);

        // Predict transitions based on trajectory
        match current_phase {
            MarketPhase::Growth => {
                if dm_dt > 0.0 {
                    transitions.push(TransitionIndicator {
                        from_phase: MarketPhase::Growth,
                        to_phase: MarketPhase::Conservation,
                        probability: dm_dt.min(1.0),
                        eta_bars: ((0.7 - current_m) / dm_dt.max(0.01)) as usize,
                    });
                }
            }
            MarketPhase::Conservation => {
                // High susceptibility indicates impending release
                if current_chi > 1.5 {
                    transitions.push(TransitionIndicator {
                        from_phase: MarketPhase::Conservation,
                        to_phase: MarketPhase::Release,
                        probability: (current_chi - 1.0).min(1.0),
                        eta_bars: (10.0 / current_chi) as usize,
                    });
                }
            }
            MarketPhase::Release => {
                transitions.push(TransitionIndicator {
                    from_phase: MarketPhase::Release,
                    to_phase: MarketPhase::Reorganization,
                    probability: 0.8,
                    eta_bars: 5,
                });
            }
            MarketPhase::Reorganization => {
                if dm_dt > 0.05 {
                    transitions.push(TransitionIndicator {
                        from_phase: MarketPhase::Reorganization,
                        to_phase: MarketPhase::Growth,
                        probability: dm_dt.min(1.0),
                        eta_bars: 10,
                    });
                }
            }
            _ => {}
        }

        transitions
    }

    /// Calculate confidence in phase detection
    fn phase_confidence(&self, magnetization: f64, susceptibility: f64) -> f64 {
        // High magnetization magnitude = confident about order
        let m_confidence = magnetization.abs();

        // Low susceptibility = stable phase
        let chi_confidence = 1.0 / (1.0 + susceptibility);

        (m_confidence + chi_confidence) / 2.0
    }
}

/// Result of pBit phase analysis
#[derive(Debug, Clone)]
pub struct PBitPhaseResult {
    /// Current detected phase
    pub current_phase: MarketPhase,
    /// Magnetization (order parameter direction)
    pub magnetization: f64,
    /// Magnetic susceptibility (phase transition indicator)
    pub susceptibility: f64,
    /// Order parameter magnitude |M|
    pub order_parameter: f64,
    /// System entropy
    pub entropy: f64,
    /// Correlation length (diverges at transitions)
    pub correlation_length: f64,
    /// Predicted transitions
    pub transitions: Vec<TransitionIndicator>,
    /// Confidence in phase detection
    pub phase_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pcr(potential: f64, connectedness: f64, resilience: f64) -> PCRComponents {
        PCRComponents {
            potential,
            connectedness,
            resilience,
        }
    }

    #[test]
    fn test_magnetization_mapping() {
        let config = PBitPanarchyConfig::default();
        let detector = PBitPhaseDetector::new(config);

        // High PCR → positive magnetization
        let high_pcr = make_pcr(0.9, 0.8, 0.85);
        let m_high = detector.pcr_to_magnetization(&high_pcr);
        assert!(m_high > 0.0);

        // Low PCR → negative magnetization
        let low_pcr = make_pcr(0.1, 0.2, 0.15);
        let m_low = detector.pcr_to_magnetization(&low_pcr);
        assert!(m_low < 0.0);
    }

    #[test]
    fn test_phase_detection() {
        let config = PBitPanarchyConfig::default();
        let detector = PBitPhaseDetector::new(config);

        // High magnetization → Conservation phase
        let phase_k = detector.detect_phase(0.8, 0.5);
        assert!(matches!(phase_k, MarketPhase::Conservation));

        // Moderate magnetization → Growth phase
        let phase_r = detector.detect_phase(0.5, 0.5);
        assert!(matches!(phase_r, MarketPhase::Growth));

        // Negative with high susceptibility → Release
        let phase_omega = detector.detect_phase(-0.5, 1.5);
        assert!(matches!(phase_omega, MarketPhase::Release));
    }

    #[test]
    fn test_susceptibility_wolfram() {
        // Wolfram: β(1 - M²) for T=T_c, M=0.5
        // β = 1/T_c ≈ 0.4407
        // χ = 0.4407 * (1 - 0.25) = 0.4407 * 0.75 ≈ 0.330
        let config = PBitPanarchyConfig::default();
        let detector = PBitPhaseDetector::new(config);
        
        let chi = detector.susceptibility(0.5);
        assert!((chi - 0.330).abs() < 0.01);
    }

    #[test]
    fn test_critical_temperature() {
        // Wolfram: 2/ln(1 + √2) = 2.269185314213022
        assert!((ISING_CRITICAL_TEMP - 2.269185314213022).abs() < 1e-10);
    }

    #[test]
    fn test_full_analysis() {
        let config = PBitPanarchyConfig::default();
        let mut detector = PBitPhaseDetector::new(config);

        // Create PCR series showing growth phase
        let pcr_series: Vec<PCRComponents> = (0..20)
            .map(|i| make_pcr(0.3 + i as f64 * 0.03, 0.4, 0.5))
            .collect();

        let result = detector.analyze(&pcr_series);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.magnetization > 0.0);
        assert!(result.phase_confidence > 0.0);
    }
}
