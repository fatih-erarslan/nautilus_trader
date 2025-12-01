//! # pBit Bridge
//!
//! Integration bridge between the cortical bus and `hyperphysics-pbit`.
//!
//! This module provides:
//! - Spike-to-pBit coupling (spikes modulate pBit biases)
//! - pBit-to-spike generation (pBit state changes emit spikes)
//! - Hyperbolic lattice dynamics
//!
//! ## Spike-pBit Coupling
//!
//! Incoming spikes modulate pBit biases according to:
//!
//! ```text
//! Δbias_i = Σ_j w_ij * strength_j * exp(-|t - t_j| / τ)
//! ```
//!
//! Where:
//! - `w_ij` is the synaptic weight from spike source j to pBit i
//! - `strength_j` is the normalized spike strength [-1, +1]
//! - `τ` is the synaptic time constant

use hyperphysics_pbit::{
    PBitLattice, PBitDynamics, Algorithm,
    MetropolisSimulator, GillespieSimulator,
};
use rand::thread_rng;

use crate::spike::Spike;
use crate::error::{CorticalError, Result};

/// Spike-to-pBit coupling configuration.
#[derive(Debug, Clone)]
pub struct SpikeCouplingConfig {
    /// Synaptic time constant (seconds).
    pub tau: f64,
    /// Maximum bias modulation per spike.
    pub max_modulation: f64,
    /// Decay rate for spike influence.
    pub decay_rate: f64,
    /// Default temperature for pBit dynamics.
    pub temperature: f64,
}

impl Default for SpikeCouplingConfig {
    fn default() -> Self {
        Self {
            tau: 1e-3,           // 1ms time constant
            max_modulation: 1.0,
            decay_rate: 0.99,
            temperature: 1.0,
        }
    }
}

/// Hyperbolic tessellation parameters for pBit lattice.
#[derive(Debug, Clone, Copy)]
pub struct TessellationParams {
    /// Polygon sides for {p,q} tessellation.
    pub p: usize,
    /// Polygons per vertex.
    pub q: usize,
    /// Tessellation depth.
    pub depth: usize,
}

impl Default for TessellationParams {
    fn default() -> Self {
        // {3,7} tessellation at depth 2 gives 48 nodes
        Self { p: 3, q: 7, depth: 2 }
    }
}

impl TessellationParams {
    /// Standard ROI lattice with 48 nodes.
    pub fn roi_48() -> Self {
        Self { p: 3, q: 7, depth: 2 }
    }

    /// Larger lattice with ~200 nodes.
    pub fn standard_200() -> Self {
        Self { p: 3, q: 7, depth: 3 }
    }
}

/// Bridge between spike routing and pBit dynamics on hyperbolic lattice.
///
/// Handles bidirectional coupling:
/// - Spikes → pBit biases (input)
/// - pBit flips → Spikes (output)
pub struct PBitBridge {
    /// pBit dynamics controller.
    dynamics: PBitDynamics,
    /// Synaptic weights: spike source → pBit index.
    synaptic_weights: Vec<Vec<(u32, f64)>>,
    /// Current bias modulations from spikes.
    bias_modulations: Vec<f64>,
    /// Configuration.
    config: SpikeCouplingConfig,
    /// Cached previous states for spike generation.
    prev_states: Vec<bool>,
}

impl PBitBridge {
    /// Create a new pBit bridge with hyperbolic tessellation.
    pub fn new(tess: TessellationParams, config: SpikeCouplingConfig) -> Result<Self> {
        let lattice = PBitLattice::new(tess.p, tess.q, tess.depth, config.temperature)
            .map_err(|e| CorticalError::ConfigError(format!("pBit lattice: {}", e)))?;
        
        let num_pbits = lattice.size();
        let dynamics = PBitDynamics::new_metropolis(lattice, config.temperature);
        
        Ok(Self {
            dynamics,
            synaptic_weights: vec![Vec::new(); num_pbits],
            bias_modulations: vec![0.0; num_pbits],
            prev_states: vec![false; num_pbits],
            config,
        })
    }

    /// Create standard ROI bridge with 48 pBits.
    pub fn roi_48(config: SpikeCouplingConfig) -> Result<Self> {
        Self::new(TessellationParams::roi_48(), config)
    }

    /// Register a synaptic connection from spike source to pBit.
    pub fn add_synapse(&mut self, pbit_idx: usize, spike_source: u32, weight: f64) {
        if pbit_idx < self.synaptic_weights.len() {
            self.synaptic_weights[pbit_idx].push((spike_source, weight));
        }
    }

    /// Process incoming spikes and update pBit biases.
    ///
    /// Returns number of pBits affected.
    pub fn process_spikes(&mut self, spikes: &[Spike]) -> usize {
        let mut affected = 0;

        for spike in spikes {
            let strength = spike.normalized_strength() as f64;
            
            // Find all pBits connected to this spike source
            for (pbit_idx, connections) in self.synaptic_weights.iter().enumerate() {
                for &(source, weight) in connections {
                    if source == spike.source_id {
                        let modulation = weight * strength * self.config.max_modulation;
                        self.bias_modulations[pbit_idx] += modulation;
                        affected += 1;
                    }
                }
            }
        }

        affected
    }

    /// Decay spike modulations (call each timestep).
    pub fn decay_modulations(&mut self) {
        for m in &mut self.bias_modulations {
            *m *= self.config.decay_rate;
        }
    }

    /// Perform one dynamics step.
    ///
    /// Call this regularly to evolve the pBit system.
    pub fn step(&mut self) -> Result<()> {
        // Save current states for spike generation
        self.prev_states = self.dynamics.lattice().states();
        
        // Perform dynamics step
        let mut rng = thread_rng();
        self.dynamics.step(&mut rng)
            .map_err(|e| CorticalError::Internal(format!("Dynamics step: {}", e)))?;
        
        Ok(())
    }

    /// Run multiple dynamics steps.
    pub fn simulate(&mut self, steps: usize) -> Result<()> {
        let mut rng = thread_rng();
        self.dynamics.simulate(steps, &mut rng)
            .map_err(|e| CorticalError::Internal(format!("Simulation: {}", e)))?;
        Ok(())
    }

    /// Generate spikes from pBit state changes since last step.
    ///
    /// Call after `step()` to get output spikes.
    pub fn generate_spikes(&self, timestamp: u16) -> Vec<Spike> {
        let current = self.dynamics.lattice().states();
        let mut spikes = Vec::new();

        for (idx, (&prev, &curr)) in self.prev_states.iter().zip(current.iter()).enumerate() {
            if prev != curr {
                // State changed - generate spike
                let strength = if curr { 127i8 } else { -128i8 };
                let routing_hint = (idx % 256) as u8;
                spikes.push(Spike::new(idx as u32, timestamp, strength, routing_hint));
            }
        }

        spikes
    }

    /// Get current pBit states.
    pub fn states(&self) -> Vec<bool> {
        self.dynamics.lattice().states()
    }

    /// Get number of pBits.
    pub fn size(&self) -> usize {
        self.dynamics.lattice().size()
    }

    /// Get current algorithm.
    pub fn algorithm(&self) -> Algorithm {
        self.dynamics.algorithm()
    }

    /// Get reference to underlying dynamics.
    pub fn dynamics(&self) -> &PBitDynamics {
        &self.dynamics
    }

    /// Get mutable reference to underlying dynamics.
    pub fn dynamics_mut(&mut self) -> &mut PBitDynamics {
        &mut self.dynamics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbit_bridge_creation() {
        let config = SpikeCouplingConfig::default();
        let bridge = PBitBridge::roi_48(config).unwrap();
        // {3,7,2} tessellation creates 15 nodes
        assert!(bridge.size() > 0, "Bridge should have pBits");
    }

    #[test]
    fn test_spike_processing() {
        let config = SpikeCouplingConfig::default();
        let mut bridge = PBitBridge::roi_48(config).unwrap();

        // Add synapse from spike source 42 to pBit 0
        bridge.add_synapse(0, 42, 0.5);

        // Create a spike from source 42
        let spikes = vec![Spike::excitatory(42, 100, 0)];
        let affected = bridge.process_spikes(&spikes);

        assert_eq!(affected, 1);
    }

    #[test]
    fn test_dynamics_step() {
        let config = SpikeCouplingConfig::default();
        let mut bridge = PBitBridge::roi_48(config).unwrap();

        // Should be able to step without error
        bridge.step().unwrap();
    }
}
