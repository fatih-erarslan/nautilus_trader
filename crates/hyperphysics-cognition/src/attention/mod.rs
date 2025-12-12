//! Hyperbolic attention mechanism (H^11 Lorentz model)
//!
//! Implements attention as an intrinsic property of hyperbolic geometry:
//! - **Higher curvature (κ↑)** → Narrower focus, deeper processing
//! - **Lower curvature (κ↓)** → Broader awareness, parallel processing
//!
//! Key insight: Attention is NOT a separate module but emerges from the
//! geometry itself. The Lorentz distance metric naturally implements
//! attention through curvature modulation.
//!
//! ## Scientific Basis
//!
//! - Chami et al. (2020) "Hyperbolic Graph Neural Networks", NeurIPS
//! - Nickel & Kiela (2017) "Poincaré Embeddings for Learning Hierarchies"
//! - Sala et al. (2018) "Representation Tradeoffs for Hyperbolic Embeddings"
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │          HYPERBOLIC ATTENTION (H^11)                │
//! │                                                     │
//! │  Curvature κ ──► Attention Bandwidth BW = k/κ      │
//! │                                                     │
//! │  ┌─────────────┐        ┌─────────────┐            │
//! │  │   κ = 0.1   │        │   κ = 10.0  │            │
//! │  │   (Broad)   │        │  (Narrow)   │            │
//! │  │  BW = 100   │        │   BW = 1    │            │
//! │  └─────────────┘        └─────────────┘            │
//! │         │                      │                   │
//! │         ▼                      ▼                   │
//! │  ┌────────────────────────────────┐                │
//! │  │   Locus Coeruleus Modulation   │                │
//! │  │   (Norepinephrine)             │                │
//! │  │   gain_LC ∈ [0.5, 2.0]         │                │
//! │  └────────────────────────────────┘                │
//! └─────────────────────────────────────────────────────┘
//! ```

use crate::error::{CognitionError, Result};
use crate::types::{LorentzPoint, AttentionBandwidth, ArousalLevel};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, trace};

/// Attention state in hyperbolic space
#[derive(Debug, Clone)]
pub struct AttentionState {
    /// Current position in H^11 (Lorentz coordinates)
    pub lorentz_point: LorentzPoint,

    /// Curvature parameter κ ∈ [0.1, 10.0]
    /// Higher κ = narrower focus, lower κ = broader awareness
    pub curvature: f64,

    /// Attention bandwidth: BW = k / κ
    pub bandwidth: AttentionBandwidth,

    /// Locus Coeruleus gain modulation
    pub lc_gain: f64,

    /// Timestamp (milliseconds)
    pub timestamp: u64,
}

impl AttentionState {
    /// Create new attention state
    pub fn new(curvature: f64) -> Result<Self> {
        if !(crate::CURVATURE_RANGE.0..=crate::CURVATURE_RANGE.1).contains(&curvature) {
            return Err(CognitionError::InvalidCurvature(
                curvature,
                crate::CURVATURE_RANGE.0,
                crate::CURVATURE_RANGE.1,
            ));
        }

        Ok(Self {
            lorentz_point: lorentz_origin(),
            curvature,
            bandwidth: AttentionBandwidth::from_curvature(curvature, 10.0),
            lc_gain: 1.0, // Neutral gain
            timestamp: 0,
        })
    }

    /// Update curvature (modulates attention focus)
    pub fn set_curvature(&mut self, curvature: f64) -> Result<()> {
        if !(crate::CURVATURE_RANGE.0..=crate::CURVATURE_RANGE.1).contains(&curvature) {
            return Err(CognitionError::InvalidCurvature(
                curvature,
                crate::CURVATURE_RANGE.0,
                crate::CURVATURE_RANGE.1,
            ));
        }

        self.curvature = curvature;
        self.bandwidth = AttentionBandwidth::from_curvature(curvature, 10.0);
        trace!("Attention curvature updated: κ={:.2}, BW={:.2}", curvature, self.bandwidth.value);
        Ok(())
    }

    /// Update Locus Coeruleus gain (arousal modulation)
    pub fn set_lc_gain(&mut self, gain: f64) {
        self.lc_gain = gain.clamp(0.5, 2.0);
        trace!("LC gain updated: {:.2}", self.lc_gain);
    }

    /// Compute effective curvature (modulated by LC)
    pub fn effective_curvature(&self) -> f64 {
        self.curvature * self.lc_gain
    }
}

/// Attention configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Initial curvature
    pub initial_curvature: f64,

    /// Curvature adaptation rate
    pub adaptation_rate: f64,

    /// Enable LC modulation
    pub enable_lc_modulation: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            initial_curvature: crate::DEFAULT_CURVATURE,
            adaptation_rate: 0.1,
            enable_lc_modulation: true,
        }
    }
}

/// Hyperbolic attention mechanism
pub struct HyperbolicAttention {
    /// Current attention state
    state: Arc<RwLock<AttentionState>>,

    /// Configuration
    config: AttentionConfig,

    /// Curvature modulator (optional)
    modulator: Option<CurvatureModulator>,

    /// LC gain controller (optional)
    lc_controller: Option<LocusCoeruleusGain>,
}

impl HyperbolicAttention {
    /// Create new hyperbolic attention mechanism
    pub fn new(initial_curvature: f64) -> Result<Self> {
        let state = AttentionState::new(initial_curvature)?;
        let config = AttentionConfig::default();

        debug!("🎯 Hyperbolic attention initialized: κ={:.2}", initial_curvature);

        Ok(Self {
            state: Arc::new(RwLock::new(state)),
            config: config.clone(),
            modulator: Some(CurvatureModulator::new(config.adaptation_rate)),
            lc_controller: if config.enable_lc_modulation {
                Some(LocusCoeruleusGain::new())
            } else {
                None
            },
        })
    }

    /// Get current attention state
    pub fn state(&self) -> AttentionState {
        self.state.read().clone()
    }

    /// Set attention curvature
    pub fn set_curvature(&self, curvature: f64) -> Result<()> {
        let mut state = self.state.write();
        state.set_curvature(curvature)
    }

    /// Modulate attention based on arousal
    pub fn modulate_arousal(&self, arousal: ArousalLevel) {
        if let Some(ref lc) = self.lc_controller {
            let gain = lc.compute_gain(arousal);
            let mut state = self.state.write();
            state.set_lc_gain(gain);
        }
    }

    /// Adapt curvature based on cognitive load
    pub fn adapt_to_load(&self, load: f64) {
        if let Some(ref modulator) = self.modulator {
            let target_curvature = modulator.compute_target(load);
            let _ = self.set_curvature(target_curvature);
        }
    }

    /// Get current bandwidth
    pub fn bandwidth(&self) -> f64 {
        self.state.read().bandwidth.value
    }

    /// Get effective curvature (modulated by LC)
    pub fn effective_curvature(&self) -> f64 {
        self.state.read().effective_curvature()
    }
}

/// Curvature modulator (adapts attention based on load)
pub struct CurvatureModulator {
    /// Adaptation rate (learning rate for curvature updates)
    adaptation_rate: f64,
}

impl CurvatureModulator {
    /// Create new curvature modulator
    pub fn new(adaptation_rate: f64) -> Self {
        Self { adaptation_rate }
    }

    /// Compute target curvature based on cognitive load
    /// High load → higher curvature (narrow focus)
    /// Low load → lower curvature (broad awareness)
    pub fn compute_target(&self, load: f64) -> f64 {
        let (min_k, max_k) = crate::CURVATURE_RANGE;

        // Nonlinear mapping: load ∈ [0,1] → κ ∈ [0.1, 10.0]
        // Use exponential to create wider range at high loads
        let normalized = load.clamp(0.0, 1.0);
        min_k + (max_k - min_k) * normalized.powf(2.0)
    }
}

/// Locus Coeruleus gain modulation (norepinephrine)
pub struct LocusCoeruleusGain {
    /// Baseline gain
    baseline: f64,
}

impl LocusCoeruleusGain {
    /// Create new LC gain controller
    pub fn new() -> Self {
        Self { baseline: 1.0 }
    }

    /// Compute gain from arousal level
    /// Low arousal (sleep) → low gain (0.5)
    /// High arousal (alert) → high gain (2.0)
    pub fn compute_gain(&self, arousal: ArousalLevel) -> f64 {
        let a = arousal.value();
        // Linear mapping: arousal ∈ [0,1] → gain ∈ [0.5, 2.0]
        0.5 + 1.5 * a
    }
}

impl Default for LocusCoeruleusGain {
    fn default() -> Self {
        Self::new()
    }
}

/// Get Lorentz origin in H^11 (identity point on hyperboloid)
fn lorentz_origin() -> LorentzPoint {
    let mut point = [0.0; 12];
    point[0] = 1.0; // Time coordinate
    point
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_state_creation() {
        let state = AttentionState::new(1.0).unwrap();
        assert_eq!(state.curvature, 1.0);
        assert_eq!(state.lc_gain, 1.0);
    }

    #[test]
    fn test_curvature_validation() {
        // Valid
        assert!(AttentionState::new(1.0).is_ok());
        assert!(AttentionState::new(0.1).is_ok());
        assert!(AttentionState::new(10.0).is_ok());

        // Invalid
        assert!(AttentionState::new(0.05).is_err());
        assert!(AttentionState::new(15.0).is_err());
    }

    #[test]
    fn test_bandwidth_computation() {
        let state = AttentionState::new(1.0).unwrap();
        assert_eq!(state.bandwidth.value, 10.0);

        let mut state = AttentionState::new(5.0).unwrap();
        assert_eq!(state.bandwidth.value, 2.0);
        assert!(state.bandwidth.is_narrow_focus());

        state.set_curvature(0.2).unwrap();
        assert_eq!(state.bandwidth.value, 50.0);
        assert!(state.bandwidth.is_broad_awareness());
    }

    #[test]
    fn test_lc_modulation() {
        let mut state = AttentionState::new(1.0).unwrap();

        // Low arousal → low gain
        state.set_lc_gain(0.6);
        assert_eq!(state.effective_curvature(), 0.6);

        // High arousal → high gain
        state.set_lc_gain(1.8);
        assert_eq!(state.effective_curvature(), 1.8);
    }

    #[test]
    fn test_hyperbolic_attention() {
        let attention = HyperbolicAttention::new(1.0).unwrap();
        assert_eq!(attention.bandwidth(), 10.0);

        attention.set_curvature(5.0).unwrap();
        assert_eq!(attention.bandwidth(), 2.0);
    }

    #[test]
    fn test_curvature_modulator() {
        let modulator = CurvatureModulator::new(0.1);

        // Low load → low curvature
        let low = modulator.compute_target(0.2);
        assert!(low < 1.0);

        // High load → high curvature
        let high = modulator.compute_target(0.8);
        assert!(high > 5.0);
    }

    #[test]
    fn test_lc_gain() {
        let lc = LocusCoeruleusGain::new();

        // Sleep
        let sleep_gain = lc.compute_gain(ArousalLevel::new(0.0));
        assert_eq!(sleep_gain, 0.5);

        // Awake
        let awake_gain = lc.compute_gain(ArousalLevel::new(1.0));
        assert_eq!(awake_gain, 2.0);

        // Moderate
        let moderate_gain = lc.compute_gain(ArousalLevel::new(0.5));
        assert_eq!(moderate_gain, 1.25);
    }
}
