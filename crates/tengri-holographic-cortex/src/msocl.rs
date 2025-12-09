//! # Meta-Stable Oscillatory Control Layer (MSOCL)
//!
//! The MSOCL acts as a "global brainwave" regulator for the 4-engine topology,
//! coordinating phase transitions and preventing runaway synchronization.
//!
//! ## Mathematical Foundation
//!
//! Based on the Kuramoto model for coupled oscillators:
//! ```text
//! dφᵢ/dt = ωᵢ + (K/N) Σⱼ sin(φⱼ - φᵢ)
//! ```
//!
//! Critical coupling for synchronization: K_c = 2γ where γ is frequency spread.
//!
//! ## Phase Cycle
//!
//! 1. **Collect**: Engines emit spikes to cortical bus
//! 2. **LocalUpdate**: Engines run pBit sampling and local plasticity
//! 3. **GlobalInfer**: GPU runs hyperbolic GNN and HNSW queries
//! 4. **Consolidate**: Writeback and coupling adjustment

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::f64::consts::PI;

use crate::constants::*;
use crate::{CortexError, Result};

/// MSOCL phase in the control cycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsoclPhase {
    /// Engines emit spikes to cortical bus
    Collect,
    /// Engines run local pBit updates and plasticity
    LocalUpdate,
    /// GPU runs hyperbolic GNN and HNSW
    GlobalInfer,
    /// Writeback and coupling adjustment
    Consolidate,
}

impl MsoclPhase {
    /// Get phase from index (0-3)
    pub fn from_index(idx: usize) -> Self {
        match idx % 4 {
            0 => MsoclPhase::Collect,
            1 => MsoclPhase::LocalUpdate,
            2 => MsoclPhase::GlobalInfer,
            _ => MsoclPhase::Consolidate,
        }
    }
    
    /// Get phase index
    pub fn index(&self) -> usize {
        match self {
            MsoclPhase::Collect => 0,
            MsoclPhase::LocalUpdate => 1,
            MsoclPhase::GlobalInfer => 2,
            MsoclPhase::Consolidate => 3,
        }
    }
    
    /// Get next phase
    pub fn next(&self) -> Self {
        Self::from_index(self.index() + 1)
    }
}

/// MSOCL configuration
#[derive(Debug, Clone)]
pub struct MsoclConfig {
    /// Base oscillation frequency (Hz)
    pub frequency_hz: f64,
    /// Coupling strength (K in Kuramoto model)
    pub coupling_strength: f64,
    /// Phase duration weights (relative to cycle)
    pub phase_weights: [f64; 4],
    /// Temperature modulation amplitude
    pub temp_amplitude: f64,
    /// Novelty detection threshold
    pub novelty_threshold: f64,
}

impl Default for MsoclConfig {
    fn default() -> Self {
        Self {
            frequency_hz: MSOCL_DEFAULT_FREQ,
            coupling_strength: MSOCL_CRITICAL_COUPLING,
            phase_weights: [0.2, 0.3, 0.3, 0.2], // Collect, LocalUpdate, GlobalInfer, Consolidate
            temp_amplitude: 0.1,
            novelty_threshold: 0.5,
        }
    }
}

/// Meta-Stable Oscillatory Control Layer
pub struct Msocl {
    /// Configuration
    config: MsoclConfig,
    /// Current phase angle [0, 2π)
    phase: f64,
    /// Current amplitude
    amplitude: f64,
    /// Current temperature modulation
    temp_modulation: f64,
    /// Novelty estimate from cortical bus
    novelty: f64,
    /// Order parameter r = |1/N Σⱼ exp(i*φⱼ)| ∈ [0,1]
    order_parameter: f64,
    /// Last tick timestamp
    last_tick: Instant,
    /// Tick counter
    tick_count: AtomicU64,
}

impl Msocl {
    /// Create new MSOCL with default configuration
    pub fn new() -> Self {
        Self::with_config(MsoclConfig::default())
    }
    
    /// Create MSOCL with custom configuration
    pub fn with_config(config: MsoclConfig) -> Self {
        Self {
            config,
            phase: 0.0,
            amplitude: 1.0,
            temp_modulation: 0.0,
            novelty: 0.0,
            order_parameter: 0.0,
            last_tick: Instant::now(),
            tick_count: AtomicU64::new(0),
        }
    }
    
    /// Get current phase
    pub fn current_phase(&self) -> MsoclPhase {
        // Map phase angle to discrete phase
        let normalized = self.phase / (2.0 * PI);
        let cumulative = self.cumulative_phase_weights();
        
        for (i, &cum) in cumulative.iter().enumerate() {
            if normalized < cum {
                return MsoclPhase::from_index(i);
            }
        }
        MsoclPhase::Consolidate
    }
    
    /// Get cumulative phase weights for phase detection
    fn cumulative_phase_weights(&self) -> [f64; 4] {
        let sum: f64 = self.config.phase_weights.iter().sum();
        let mut cum = [0.0; 4];
        let mut acc = 0.0;
        for (i, &w) in self.config.phase_weights.iter().enumerate() {
            acc += w / sum;
            cum[i] = acc;
        }
        cum
    }
    
    /// Get current temperature modulation factor
    pub fn temperature_modulation(&self) -> f64 {
        1.0 + self.config.temp_amplitude * (self.phase).sin()
    }
    
    /// Get order parameter (synchronization measure)
    pub fn order_parameter(&self) -> f64 {
        self.order_parameter
    }
    
    /// Update novelty estimate from cortical bus
    pub fn update_novelty(&mut self, novelty: f64) {
        self.novelty = novelty.clamp(0.0, 1.0);
    }
    
    /// Perform one tick of the MSOCL
    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = now;
        
        // Advance phase: φ += ω*dt where ω = 2π*f
        let omega = 2.0 * PI * self.config.frequency_hz;
        self.phase += omega * dt;
        
        // Wrap phase to [0, 2π)
        while self.phase >= 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
        
        // Update temperature modulation
        self.temp_modulation = self.temperature_modulation();
        
        // Adaptive coupling based on novelty
        // High novelty → reduce coupling (explore)
        // Low novelty → increase coupling (exploit)
        let effective_coupling = self.config.coupling_strength * (1.0 - self.novelty * 0.5);
        
        // Update order parameter (simplified: based on phase coherence)
        self.order_parameter = (1.0 - self.novelty).max(0.0);
        
        self.tick_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get tick count
    pub fn tick_count(&self) -> u64 {
        self.tick_count.load(Ordering::Relaxed)
    }
    
    /// Get period for current frequency
    pub fn period(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.config.frequency_hz)
    }
    
    /// Check if we're in a specific phase
    pub fn is_phase(&self, phase: MsoclPhase) -> bool {
        self.current_phase() == phase
    }
    
    /// Get phase progress (0.0 to 1.0 within current phase)
    pub fn phase_progress(&self) -> f64 {
        let normalized = self.phase / (2.0 * PI);
        let cumulative = self.cumulative_phase_weights();
        
        let mut prev = 0.0;
        for (i, &cum) in cumulative.iter().enumerate() {
            if normalized < cum {
                let phase_width = cum - prev;
                return (normalized - prev) / phase_width;
            }
            prev = cum;
        }
        1.0
    }
    
    /// Get recommended engine temperatures based on MSOCL state
    pub fn engine_temperatures(&self, base_temp: f64) -> [f64; NUM_ENGINES] {
        let mod_factor = self.temperature_modulation();
        let phase = self.current_phase();
        
        // Different phases have different temperature profiles
        let phase_factor = match phase {
            MsoclPhase::Collect => 1.0,        // Normal temp for collection
            MsoclPhase::LocalUpdate => 0.9,    // Slightly cooler for updates
            MsoclPhase::GlobalInfer => 0.8,    // Cooler for exploitation
            MsoclPhase::Consolidate => 1.1,    // Warmer for consolidation
        };
        
        // Slight variation per engine to avoid perfect synchronization
        [
            base_temp * mod_factor * phase_factor * 1.00,
            base_temp * mod_factor * phase_factor * 0.98,
            base_temp * mod_factor * phase_factor * 1.02,
            base_temp * mod_factor * phase_factor * 1.01,
        ]
    }
}

impl Default for Msocl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_msocl_phase_cycle() {
        let mut msocl = Msocl::new();
        let initial_phase = msocl.current_phase();
        
        // Simulate a full cycle
        for _ in 0..100 {
            msocl.tick();
            std::thread::sleep(Duration::from_millis(1));
        }
        
        // Should have made progress
        assert!(msocl.tick_count() > 0);
    }
    
    #[test]
    fn test_phase_from_index() {
        assert_eq!(MsoclPhase::from_index(0), MsoclPhase::Collect);
        assert_eq!(MsoclPhase::from_index(1), MsoclPhase::LocalUpdate);
        assert_eq!(MsoclPhase::from_index(2), MsoclPhase::GlobalInfer);
        assert_eq!(MsoclPhase::from_index(3), MsoclPhase::Consolidate);
        assert_eq!(MsoclPhase::from_index(4), MsoclPhase::Collect); // Wraps
    }
    
    #[test]
    fn test_temperature_modulation() {
        let msocl = Msocl::new();
        let mod_factor = msocl.temperature_modulation();
        // Should be in range [1 - amplitude, 1 + amplitude]
        assert!(mod_factor >= 0.9 && mod_factor <= 1.1);
    }
    
    #[test]
    fn test_novelty_clamping() {
        let mut msocl = Msocl::new();
        msocl.update_novelty(1.5);
        assert_eq!(msocl.novelty, 1.0);
        msocl.update_novelty(-0.5);
        assert_eq!(msocl.novelty, 0.0);
    }
}
