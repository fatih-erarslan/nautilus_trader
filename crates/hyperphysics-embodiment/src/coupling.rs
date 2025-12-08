//! Neural-Body Coupling Configuration
//!
//! Defines how neural activity is translated to muscle forces and
//! how body state feeds back into the neural network.
//!
//! ## STDP Integration
//!
//! This module now integrates with `hyperphysics-stdp` for spike-timing-dependent
//! plasticity in sensorimotor loops. STDP enables:
//!
//! - **Sensory-Motor Learning**: Strengthen connections when motor outputs match
//!   predicted sensory consequences (forward model learning)
//! - **Proprioceptive Calibration**: Adapt neural-body mapping based on
//!   body state feedback timing
//! - **Three-Factor Learning**: Modulate plasticity with reward/neuromodulator signals
//!
//! ## References
//! - Wolpert & Kawato (1998) "Multiple paired forward and inverse models"
//! - Dayan & Abbott (2001) "Theoretical Neuroscience" Ch. 8

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use hyperphysics_stdp::{
    ClassicalStdp, ClassicalStdpParams, TripletStdp, TripletStdpParams,
    RewardModulatedStdp, RewardModulatedParams, RewardSignal,
    PlasticityController, PlasticityRule, WeightBounds, WeightUpdate,
};

/// Coupling configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CouplingConfig {
    /// Coupling mode
    pub mode: CouplingMode,

    /// Neural to physics time ratio
    /// e.g., 10 means 10 neural steps per physics step
    pub time_ratio: u32,

    /// Muscle force scaling factor
    pub force_scale: f32,

    /// Proprioceptive gain
    pub proprioceptive_gain: f32,

    /// Enable proprioceptive feedback
    pub proprioception_enabled: bool,

    /// Muscle activation smoothing time constant (ms)
    pub activation_tau: f32,

    /// Proprioceptive delay (ms)
    pub proprioceptive_delay: f32,

    /// Classical STDP parameters for sensorimotor plasticity
    #[cfg_attr(feature = "serde", serde(skip))]
    pub stdp_params: Option<ClassicalStdpParams>,

    /// Enable STDP-based sensorimotor learning
    pub stdp_enabled: bool,

    /// Reward signal gain for three-factor learning
    pub reward_gain: f32,
}

impl Default for CouplingConfig {
    fn default() -> Self {
        Self {
            mode: CouplingMode::Bidirectional,
            time_ratio: 10,
            force_scale: 1.0,
            proprioceptive_gain: 1.0,
            proprioception_enabled: true,
            activation_tau: 20.0,
            proprioceptive_delay: 5.0,
            stdp_params: None,
            stdp_enabled: false,
            reward_gain: 1.0,
        }
    }
}

impl CouplingConfig {
    /// Configuration for fast forward locomotion
    pub fn locomotion() -> Self {
        Self {
            mode: CouplingMode::Bidirectional,
            time_ratio: 20,
            force_scale: 1.5,
            proprioceptive_gain: 2.0,
            proprioception_enabled: true,
            activation_tau: 15.0,
            proprioceptive_delay: 3.0,
            stdp_params: None,
            stdp_enabled: false,
            reward_gain: 1.0,
        }
    }

    /// Open-loop (no proprioception)
    pub fn open_loop() -> Self {
        Self {
            mode: CouplingMode::NeuralToBody,
            time_ratio: 10,
            force_scale: 1.0,
            proprioceptive_gain: 0.0,
            proprioception_enabled: false,
            activation_tau: 20.0,
            proprioceptive_delay: 0.0,
            stdp_params: None,
            stdp_enabled: false,
            reward_gain: 0.0,
        }
    }

    /// High-fidelity simulation
    pub fn high_fidelity() -> Self {
        Self {
            mode: CouplingMode::Bidirectional,
            time_ratio: 40,
            force_scale: 1.0,
            proprioceptive_gain: 1.0,
            proprioception_enabled: true,
            activation_tau: 10.0,
            proprioceptive_delay: 2.0,
            stdp_params: None,
            stdp_enabled: false,
            reward_gain: 1.0,
        }
    }

    /// Configuration with STDP-based sensorimotor learning
    ///
    /// Enables spike-timing-dependent plasticity for:
    /// - Forward model learning (sensory prediction from motor commands)
    /// - Inverse model learning (motor commands from desired sensory states)
    /// - Proprioceptive calibration
    pub fn with_stdp_learning() -> Self {
        let stdp_params = ClassicalStdpParams {
            tau_plus: 20.0,   // LTP time constant (ms)
            tau_minus: 20.0,  // LTD time constant (ms)
            a_plus: 0.01,     // LTP amplitude
            a_minus: 0.012,   // LTD amplitude (slightly stronger for stability)
            ..Default::default()
        };

        Self {
            mode: CouplingMode::Bidirectional,
            time_ratio: 20,
            force_scale: 1.0,
            proprioceptive_gain: 1.5,
            proprioception_enabled: true,
            activation_tau: 15.0,
            proprioceptive_delay: 5.0,
            stdp_params: Some(stdp_params),
            stdp_enabled: true,
            reward_gain: 1.0,
        }
    }

    /// Enable STDP with custom configuration
    pub fn with_stdp_params(mut self, params: ClassicalStdpParams) -> Self {
        self.stdp_params = Some(params);
        self.stdp_enabled = true;
        self
    }

    /// Set reward gain for three-factor learning
    pub fn with_reward_gain(mut self, gain: f32) -> Self {
        self.reward_gain = gain;
        self
    }
}

/// Sensorimotor STDP coordinator
///
/// Manages spike-timing-dependent plasticity between motor commands
/// and sensory feedback for closed-loop sensorimotor learning.
///
/// Uses the hyperphysics-stdp crate's reward-modulated STDP for
/// three-factor learning in sensorimotor loops.
pub struct SensorimotorSTDP {
    /// Plasticity controller with reward-modulated STDP
    controller: PlasticityController,
    /// Number of motor neurons
    num_motor: usize,
    /// Number of sensory neurons
    num_sensory: usize,
    /// Current reward/neuromodulator signal
    reward_signal: f32,
    /// Weights for motor-sensory connections
    weights: Vec<f32>,
}

impl SensorimotorSTDP {
    /// Create new sensorimotor STDP coordinator
    pub fn new(params: ClassicalStdpParams, num_motor: usize, num_sensory: usize) -> Self {
        let num_synapses = num_motor * num_sensory;
        let mut controller = PlasticityController::new();

        // Add classical STDP rule
        controller.add_rule(Box::new(ClassicalStdp::new(num_synapses, params)));

        Self {
            controller,
            num_motor,
            num_sensory,
            reward_signal: 0.0,
            weights: vec![0.5; num_synapses], // Initialize at midpoint
        }
    }

    /// Create with reward-modulated STDP for three-factor learning
    pub fn with_reward_modulation(num_motor: usize, num_sensory: usize) -> Self {
        let num_synapses = num_motor * num_sensory;
        let controller = PlasticityController::with_reward_stdp(num_synapses);

        Self {
            controller,
            num_motor,
            num_sensory,
            reward_signal: 0.0,
            weights: vec![0.5; num_synapses],
        }
    }

    /// Record motor neuron spike (presynaptic for forward model)
    pub fn record_motor_spike(&mut self, neuron_id: usize, time: f64) {
        // Motor spikes are presynaptic in the forward model
        // (motor command → expected sensory consequence)
        for sensory_id in 0..self.num_sensory {
            let synapse_id = neuron_id * self.num_sensory + sensory_id;
            self.controller.on_pre_spike(synapse_id, time);
        }
    }

    /// Record sensory neuron spike (postsynaptic for forward model)
    pub fn record_sensory_spike(&mut self, neuron_id: usize, time: f64) {
        // Sensory spikes are postsynaptic in the forward model
        self.controller.on_post_spike(neuron_id, time);
    }

    /// Set reward signal for three-factor learning
    pub fn set_reward(&mut self, reward: f32) {
        self.reward_signal = reward;
        self.controller.set_learning_rate(reward.abs());
    }

    /// Apply STDP updates to weights
    pub fn apply_updates(&mut self) {
        self.controller.apply(&mut self.weights);
    }

    /// Get weight for motor→sensory connection
    pub fn get_weight(&self, motor_id: usize, sensory_id: usize) -> f32 {
        let idx = motor_id * self.num_sensory + sensory_id;
        self.weights.get(idx).copied().unwrap_or(0.0)
    }

    /// Get all weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Reset learning state
    pub fn reset(&mut self) {
        self.controller.reset();
    }

    /// Get plasticity statistics
    pub fn stats(&self) -> hyperphysics_stdp::PlasticityStats {
        self.controller.stats()
    }
}

/// Coupling mode between neural and body simulations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CouplingMode {
    /// One-way: neural → body (open-loop)
    NeuralToBody,

    /// One-way: body → neural (passive observation)
    BodyToNeural,

    /// Two-way: neural ↔ body (closed-loop)
    Bidirectional,

    /// Decoupled: both run independently
    Decoupled,
}

impl CouplingMode {
    /// Check if neural output affects body
    pub fn neural_drives_body(&self) -> bool {
        matches!(self, Self::NeuralToBody | Self::Bidirectional)
    }

    /// Check if body state affects neural
    pub fn body_drives_neural(&self) -> bool {
        matches!(self, Self::BodyToNeural | Self::Bidirectional)
    }
}

/// Muscle segment mapping
/// Maps the 96 muscles to body segments for force application
#[derive(Debug, Clone)]
pub struct SegmentMapping {
    /// Muscle indices for each body segment
    /// Each segment has 4 muscles (dorsal-right, ventral-right, ventral-left, dorsal-left)
    pub segment_muscles: [[usize; 4]; 24],

    /// Particle indices that belong to each segment
    pub segment_particles: Vec<Vec<usize>>,

    /// Segment centers (computed from particles)
    pub segment_centers: Vec<[f32; 3]>,
}

impl Default for SegmentMapping {
    fn default() -> Self {
        // Default mapping: muscle index = segment * 4 + quadrant
        let mut segment_muscles = [[0usize; 4]; 24];
        for seg in 0..24 {
            for quad in 0..4 {
                segment_muscles[seg][quad] = seg * 4 + quad;
            }
        }

        Self {
            segment_muscles,
            segment_particles: vec![Vec::new(); 24],
            segment_centers: vec![[0.0; 3]; 24],
        }
    }
}

impl SegmentMapping {
    /// Create mapping from particle positions
    /// Assumes particles are organized along the body axis (x-axis)
    pub fn from_particles(positions: &[[f32; 3]], num_segments: usize) -> Self {
        let mut mapping = Self::default();

        if positions.is_empty() {
            return mapping;
        }

        // Find body extent
        let x_min = positions.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let x_max = positions.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let segment_length = (x_max - x_min) / num_segments as f32;

        // Assign particles to segments
        mapping.segment_particles = vec![Vec::new(); num_segments];

        for (i, pos) in positions.iter().enumerate() {
            let segment = ((pos[0] - x_min) / segment_length) as usize;
            let segment = segment.min(num_segments - 1);
            mapping.segment_particles[segment].push(i);
        }

        // Compute segment centers
        mapping.segment_centers = vec![[0.0; 3]; num_segments];
        for (seg, particles) in mapping.segment_particles.iter().enumerate() {
            if particles.is_empty() {
                continue;
            }

            let mut center = [0.0_f32; 3];
            for &p in particles {
                center[0] += positions[p][0];
                center[1] += positions[p][1];
                center[2] += positions[p][2];
            }

            let n = particles.len() as f32;
            mapping.segment_centers[seg] = [center[0] / n, center[1] / n, center[2] / n];
        }

        mapping
    }

    /// Get muscle indices for a segment
    pub fn get_segment_muscles(&self, segment: usize) -> &[usize; 4] {
        &self.segment_muscles[segment]
    }

    /// Get particles in a segment
    pub fn get_segment_particles(&self, segment: usize) -> &[usize] {
        &self.segment_particles[segment]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_modes() {
        let mode = CouplingMode::Bidirectional;
        assert!(mode.neural_drives_body());
        assert!(mode.body_drives_neural());

        let mode = CouplingMode::NeuralToBody;
        assert!(mode.neural_drives_body());
        assert!(!mode.body_drives_neural());
    }

    #[test]
    fn test_segment_mapping() {
        // Create a simple line of particles
        let positions: Vec<[f32; 3]> = (0..48)
            .map(|i| [i as f32 * 0.1, 0.0, 0.0])
            .collect();

        let mapping = SegmentMapping::from_particles(&positions, 24);

        // Should have 2 particles per segment
        for seg in 0..24 {
            assert_eq!(mapping.segment_particles[seg].len(), 2);
        }
    }
}
