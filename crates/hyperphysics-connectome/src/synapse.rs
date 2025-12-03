//! Synaptic Connections and Dynamics
//!
//! Models chemical synapses and electrical gap junctions.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::neuron::NeuronId;

/// Synapse type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SynapseType {
    /// Chemical synapse (one-directional)
    Chemical,
    /// Electrical gap junction (bidirectional)
    GapJunction,
}

/// A synaptic connection between two neurons
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Synapse {
    /// Presynaptic neuron
    pub pre: NeuronId,
    /// Postsynaptic neuron
    pub post: NeuronId,
    /// Synapse type
    pub synapse_type: SynapseType,
    /// Synaptic weight (conductance in nS)
    pub weight: f32,
    /// Number of synaptic contacts (from electron microscopy)
    pub num_contacts: u32,
    /// Reversal potential for chemical synapses (mV)
    pub e_rev: f32,
}

impl Synapse {
    /// Create a new chemical synapse
    pub fn chemical(pre: NeuronId, post: NeuronId, weight: f32) -> Self {
        Self {
            pre,
            post,
            synapse_type: SynapseType::Chemical,
            weight,
            num_contacts: 1,
            e_rev: 0.0, // Excitatory default
        }
    }

    /// Create a new gap junction
    pub fn gap_junction(pre: NeuronId, post: NeuronId, weight: f32) -> Self {
        Self {
            pre,
            post,
            synapse_type: SynapseType::GapJunction,
            weight,
            num_contacts: 1,
            e_rev: 0.0, // Not used for gap junctions
        }
    }

    /// Set as excitatory (E_rev = 0 mV)
    pub fn excitatory(mut self) -> Self {
        self.e_rev = 0.0;
        self
    }

    /// Set as inhibitory (E_rev = -80 mV)
    pub fn inhibitory(mut self) -> Self {
        self.e_rev = -80.0;
        self
    }

    /// Set reversal potential
    pub fn with_reversal(mut self, e_rev: f32) -> Self {
        self.e_rev = e_rev;
        self
    }

    /// Set number of contacts
    pub fn with_contacts(mut self, num_contacts: u32) -> Self {
        self.num_contacts = num_contacts;
        self
    }

    /// Check if synapse is excitatory
    pub fn is_excitatory(&self) -> bool {
        self.e_rev > -40.0
    }

    /// Check if synapse is inhibitory
    pub fn is_inhibitory(&self) -> bool {
        self.e_rev <= -40.0
    }

    /// Check if this is a gap junction
    pub fn is_gap_junction(&self) -> bool {
        matches!(self.synapse_type, SynapseType::GapJunction)
    }
}

/// Dynamic state of a synapse during simulation
#[derive(Debug, Clone, Copy, Default)]
pub struct SynapticState {
    /// Synaptic conductance (fraction of max, 0-1)
    pub g: f32,
    /// Presynaptic calcium concentration (for plasticity)
    pub ca_pre: f32,
    /// Postsynaptic calcium concentration (for plasticity)
    pub ca_post: f32,
    /// Eligibility trace (for reward-modulated plasticity)
    pub eligibility: f32,
    /// Short-term plasticity: facilitation variable
    pub facilitation: f32,
    /// Short-term plasticity: depression variable
    pub depression: f32,
}

impl SynapticState {
    /// Create new synaptic state
    pub fn new() -> Self {
        Self {
            g: 0.0,
            ca_pre: 0.0,
            ca_post: 0.0,
            eligibility: 0.0,
            facilitation: 1.0,
            depression: 1.0,
        }
    }
}

/// Parameters for chemical synapse dynamics
#[derive(Debug, Clone, Copy)]
pub struct ChemicalSynapseParams {
    /// Rise time constant (ms)
    pub tau_rise: f32,
    /// Decay time constant (ms)
    pub tau_decay: f32,
    /// Maximum conductance (nS)
    pub g_max: f32,
    /// Delay (ms)
    pub delay: f32,
}

impl Default for ChemicalSynapseParams {
    fn default() -> Self {
        Self {
            tau_rise: 0.5,
            tau_decay: 5.0,
            g_max: 1.0,
            delay: 1.0,
        }
    }
}

impl ChemicalSynapseParams {
    /// AMPA-type (fast excitatory)
    pub fn ampa() -> Self {
        Self {
            tau_rise: 0.2,
            tau_decay: 2.0,
            g_max: 1.0,
            delay: 1.0,
        }
    }

    /// NMDA-type (slow excitatory)
    pub fn nmda() -> Self {
        Self {
            tau_rise: 2.0,
            tau_decay: 100.0,
            g_max: 0.5,
            delay: 1.0,
        }
    }

    /// GABA-A type (fast inhibitory)
    pub fn gaba_a() -> Self {
        Self {
            tau_rise: 0.5,
            tau_decay: 10.0,
            g_max: 1.0,
            delay: 1.0,
        }
    }

    /// Update conductance after presynaptic spike
    pub fn on_spike(&self, state: &mut SynapticState) {
        // Instantaneous rise approximation
        state.g += self.g_max * state.facilitation * state.depression;
    }

    /// Update conductance decay
    pub fn step(&self, state: &mut SynapticState, dt: f32) {
        // Exponential decay
        state.g *= (-dt / self.tau_decay).exp();

        // Short-term plasticity recovery
        state.facilitation += (1.0 - state.facilitation) * dt / 500.0;
        state.depression += (1.0 - state.depression) * dt / 200.0;
    }
}

/// Parameters for gap junction dynamics
#[derive(Debug, Clone, Copy)]
pub struct GapJunctionParams {
    /// Conductance (nS)
    pub g: f32,
    /// Rectification factor (1.0 = no rectification)
    pub rectification: f32,
}

impl Default for GapJunctionParams {
    fn default() -> Self {
        Self {
            g: 1.0,
            rectification: 1.0,
        }
    }
}

impl GapJunctionParams {
    /// Calculate current through gap junction
    ///
    /// I = g * (V_pre - V_post)
    pub fn current(&self, v_pre: f32, v_post: f32) -> f32 {
        let dv = v_pre - v_post;

        // Apply rectification if needed
        let g_eff = if self.rectification != 1.0 {
            if dv > 0.0 {
                self.g * self.rectification
            } else {
                self.g / self.rectification
            }
        } else {
            self.g
        };

        g_eff * dv
    }
}

/// Short-term plasticity parameters (Tsodyks-Markram model)
#[derive(Debug, Clone, Copy)]
pub struct ShortTermPlasticityParams {
    /// Use parameter (fraction released per spike)
    pub u: f32,
    /// Time constant for facilitation recovery (ms)
    pub tau_f: f32,
    /// Time constant for depression recovery (ms)
    pub tau_d: f32,
}

impl Default for ShortTermPlasticityParams {
    fn default() -> Self {
        Self {
            u: 0.5,
            tau_f: 500.0,
            tau_d: 200.0,
        }
    }
}

impl ShortTermPlasticityParams {
    /// Facilitating synapse
    pub fn facilitating() -> Self {
        Self {
            u: 0.1,
            tau_f: 1000.0,
            tau_d: 100.0,
        }
    }

    /// Depressing synapse
    pub fn depressing() -> Self {
        Self {
            u: 0.8,
            tau_f: 100.0,
            tau_d: 800.0,
        }
    }

    /// Update on presynaptic spike
    pub fn on_spike(&self, state: &mut SynapticState) {
        // Facilitation increases
        state.facilitation += self.u * (1.0 - state.facilitation);
        // Depression decreases
        state.depression *= 1.0 - state.facilitation;
    }

    /// Update recovery between spikes
    pub fn step(&self, state: &mut SynapticState, dt: f32) {
        state.facilitation += (1.0 - state.facilitation) * dt / self.tau_f;
        state.depression += (1.0 - state.depression) * dt / self.tau_d;
    }
}

// ============================================================================
// Graded Synapse Models (from c302)
// For non-spiking neurons that use analog voltage-dependent transmission
// ============================================================================

/// Graded synapse model (Level C1)
///
/// Implements continuous voltage-dependent synaptic transmission
/// suitable for non-spiking neurons in C. elegans.
///
/// Follows the c302 GradedSynapse model:
/// I_syn = g_max * s * (V_post - E_rev)
/// where s = 1 / (1 + exp(-k * (V_pre - V_th)))
#[derive(Debug, Clone, Copy)]
pub struct GradedSynapseParams {
    /// Maximum conductance (nS)
    pub g_max: f32,
    /// Reversal potential (mV)
    pub e_rev: f32,
    /// Activation threshold voltage (mV)
    pub v_th: f32,
    /// Steepness of sigmoid (1/mV)
    pub k: f32,
    /// Time constant for activation (ms)
    pub tau: f32,
}

impl Default for GradedSynapseParams {
    fn default() -> Self {
        Self {
            g_max: 0.2,    // 0.2 nS default
            e_rev: 0.0,    // Excitatory default
            v_th: -30.0,   // Activation threshold -30 mV
            k: 0.125,      // Sigmoid steepness
            tau: 10.0,     // 10 ms time constant
        }
    }
}

impl GradedSynapseParams {
    /// Create excitatory graded synapse
    pub fn excitatory() -> Self {
        Self {
            e_rev: 0.0,
            ..Default::default()
        }
    }

    /// Create inhibitory graded synapse
    pub fn inhibitory() -> Self {
        Self {
            e_rev: -80.0,
            ..Default::default()
        }
    }

    /// Calculate steady-state activation from presynaptic voltage
    #[inline]
    pub fn activation_inf(&self, v_pre: f32) -> f32 {
        1.0 / (1.0 + (-self.k * (v_pre - self.v_th)).exp())
    }

    /// Update synaptic state based on presynaptic voltage
    pub fn step(&self, state: &mut GradedSynapticState, v_pre: f32, dt: f32) {
        let s_inf = self.activation_inf(v_pre);
        // First-order kinetics: ds/dt = (s_inf - s) / tau
        state.s += (s_inf - state.s) * dt / self.tau;
        state.s = state.s.clamp(0.0, 1.0);
    }

    /// Calculate current flowing through synapse
    ///
    /// Returns current in nA (positive = inward)
    pub fn current(&self, state: &GradedSynapticState, v_post: f32) -> f32 {
        self.g_max * state.s * (self.e_rev - v_post)
    }
}

/// State for graded synapse
#[derive(Debug, Clone, Copy, Default)]
pub struct GradedSynapticState {
    /// Synaptic activation (0-1)
    pub s: f32,
}

impl GradedSynapticState {
    /// Create new graded synaptic state
    pub fn new() -> Self {
        Self { s: 0.0 }
    }
}

/// Advanced graded synapse model (Level D1)
///
/// Implements the GradedSynapse2 model from c302 with explicit
/// rise and decay kinetics, calcium dynamics, and facilitation/depression.
///
/// This model captures more detailed synaptic dynamics including:
/// - Separate rise and decay time constants
/// - Presynaptic calcium dynamics
/// - Short-term plasticity
#[derive(Debug, Clone, Copy)]
pub struct GradedSynapse2Params {
    /// Maximum conductance (nS)
    pub g_max: f32,
    /// Reversal potential (mV)
    pub e_rev: f32,
    /// Rise time constant (ms)
    pub tau_rise: f32,
    /// Decay time constant (ms)
    pub tau_decay: f32,
    /// Activation threshold (mV)
    pub v_th: f32,
    /// Calcium entry rate constant
    pub ca_rate: f32,
    /// Calcium decay time constant (ms)
    pub tau_ca: f32,
    /// Calcium threshold for release
    pub ca_th: f32,
    /// Hill coefficient for calcium cooperativity
    pub n_hill: f32,
}

impl Default for GradedSynapse2Params {
    fn default() -> Self {
        Self {
            g_max: 0.3,
            e_rev: 0.0,
            tau_rise: 1.0,
            tau_decay: 20.0,
            v_th: -35.0,
            ca_rate: 0.01,
            tau_ca: 50.0,
            ca_th: 0.5,
            n_hill: 4.0,
        }
    }
}

impl GradedSynapse2Params {
    /// Create excitatory graded synapse 2
    pub fn excitatory() -> Self {
        Self {
            e_rev: 0.0,
            ..Default::default()
        }
    }

    /// Create inhibitory graded synapse 2
    pub fn inhibitory() -> Self {
        Self {
            e_rev: -80.0,
            ..Default::default()
        }
    }

    /// Create fast graded synapse (shorter time constants)
    pub fn fast() -> Self {
        Self {
            tau_rise: 0.5,
            tau_decay: 5.0,
            tau_ca: 20.0,
            ..Default::default()
        }
    }

    /// Create slow graded synapse (longer time constants)
    pub fn slow() -> Self {
        Self {
            tau_rise: 5.0,
            tau_decay: 100.0,
            tau_ca: 200.0,
            ..Default::default()
        }
    }

    /// Update synaptic state
    pub fn step(&self, state: &mut GradedSynapse2State, v_pre: f32, dt: f32) {
        // Update presynaptic calcium
        let ca_entry = if v_pre > self.v_th {
            self.ca_rate * (v_pre - self.v_th)
        } else {
            0.0
        };
        let dca = ca_entry - state.ca / self.tau_ca;
        state.ca += dca * dt;
        state.ca = state.ca.max(0.0);

        // Calculate release probability using Hill equation
        let ca_term = state.ca.powf(self.n_hill);
        let ca_th_term = self.ca_th.powf(self.n_hill);
        let release_prob = ca_term / (ca_term + ca_th_term);

        // Update synaptic activation with separate rise/decay
        let ds = if release_prob > state.s {
            (release_prob - state.s) / self.tau_rise
        } else {
            (release_prob - state.s) / self.tau_decay
        };
        state.s += ds * dt;
        state.s = state.s.clamp(0.0, 1.0);
    }

    /// Calculate synaptic current
    pub fn current(&self, state: &GradedSynapse2State, v_post: f32) -> f32 {
        self.g_max * state.s * (self.e_rev - v_post)
    }
}

/// State for advanced graded synapse
#[derive(Debug, Clone, Copy, Default)]
pub struct GradedSynapse2State {
    /// Synaptic activation (0-1)
    pub s: f32,
    /// Presynaptic calcium concentration
    pub ca: f32,
}

impl GradedSynapse2State {
    /// Create new state
    pub fn new() -> Self {
        Self { s: 0.0, ca: 0.0 }
    }
}

/// Graded synapse connection with embedded parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GradedSynapse {
    /// Presynaptic neuron
    pub pre: NeuronId,
    /// Postsynaptic neuron
    pub post: NeuronId,
    /// Synapse parameters
    pub params: GradedSynapseType,
}

/// Type of graded synapse
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GradedSynapseType {
    /// Basic graded synapse (Level C1)
    Basic(GradedSynapseParamsStored),
    /// Advanced graded synapse (Level D1)
    Advanced(GradedSynapse2ParamsStored),
}

/// Storable version of GradedSynapseParams
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GradedSynapseParamsStored {
    pub g_max: f32,
    pub e_rev: f32,
    pub v_th: f32,
    pub k: f32,
    pub tau: f32,
}

impl From<GradedSynapseParams> for GradedSynapseParamsStored {
    fn from(p: GradedSynapseParams) -> Self {
        Self {
            g_max: p.g_max,
            e_rev: p.e_rev,
            v_th: p.v_th,
            k: p.k,
            tau: p.tau,
        }
    }
}

impl From<GradedSynapseParamsStored> for GradedSynapseParams {
    fn from(p: GradedSynapseParamsStored) -> Self {
        Self {
            g_max: p.g_max,
            e_rev: p.e_rev,
            v_th: p.v_th,
            k: p.k,
            tau: p.tau,
        }
    }
}

/// Storable version of GradedSynapse2Params
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GradedSynapse2ParamsStored {
    pub g_max: f32,
    pub e_rev: f32,
    pub tau_rise: f32,
    pub tau_decay: f32,
    pub v_th: f32,
    pub ca_rate: f32,
    pub tau_ca: f32,
    pub ca_th: f32,
    pub n_hill: f32,
}

impl From<GradedSynapse2Params> for GradedSynapse2ParamsStored {
    fn from(p: GradedSynapse2Params) -> Self {
        Self {
            g_max: p.g_max,
            e_rev: p.e_rev,
            tau_rise: p.tau_rise,
            tau_decay: p.tau_decay,
            v_th: p.v_th,
            ca_rate: p.ca_rate,
            tau_ca: p.tau_ca,
            ca_th: p.ca_th,
            n_hill: p.n_hill,
        }
    }
}

impl From<GradedSynapse2ParamsStored> for GradedSynapse2Params {
    fn from(p: GradedSynapse2ParamsStored) -> Self {
        Self {
            g_max: p.g_max,
            e_rev: p.e_rev,
            tau_rise: p.tau_rise,
            tau_decay: p.tau_decay,
            v_th: p.v_th,
            ca_rate: p.ca_rate,
            tau_ca: p.tau_ca,
            ca_th: p.ca_th,
            n_hill: p.n_hill,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_creation() {
        let syn = Synapse::chemical(0, 1, 1.5).excitatory();
        assert_eq!(syn.pre, 0);
        assert_eq!(syn.post, 1);
        assert!(syn.is_excitatory());

        let inh = Synapse::chemical(0, 1, 1.0).inhibitory();
        assert!(inh.is_inhibitory());

        let gap = Synapse::gap_junction(0, 1, 0.5);
        assert!(gap.is_gap_junction());
    }

    #[test]
    fn test_synaptic_dynamics() {
        let params = ChemicalSynapseParams::default();
        let mut state = SynapticState::new();

        // Trigger spike
        params.on_spike(&mut state);
        let g_peak = state.g;

        // Decay
        for _ in 0..100 {
            params.step(&mut state, 0.1);
        }

        assert!(state.g < g_peak, "Conductance should decay");
    }

    #[test]
    fn test_gap_junction_current() {
        let params = GapJunctionParams::default();

        // Current should flow from higher to lower potential
        let i = params.current(-50.0, -70.0);
        assert!(i > 0.0, "Current should be positive when V_pre > V_post");

        let i2 = params.current(-70.0, -50.0);
        assert!(i2 < 0.0, "Current should be negative when V_pre < V_post");
    }

    #[test]
    fn test_graded_synapse_activation() {
        let params = GradedSynapseParams::default();
        let mut state = GradedSynapticState::new();

        // Below threshold - low activation
        let s_low = params.activation_inf(-60.0);
        assert!(s_low < 0.2, "Activation should be low below threshold: got {}", s_low);

        // Above threshold - high activation
        let s_high = params.activation_inf(-10.0);
        assert!(s_high > 0.8, "Activation should be high above threshold: got {}", s_high);

        // Update state with depolarized presynaptic neuron
        // Need more iterations for tau=10ms
        for _ in 0..500 {
            params.step(&mut state, -10.0, 0.1);
        }
        assert!(state.s > 0.5, "State should reach high activation: got {}", state.s);
    }

    #[test]
    fn test_graded_synapse_current() {
        let params = GradedSynapseParams::excitatory();
        let state = GradedSynapticState { s: 1.0 };

        // Excitatory current should be positive (inward) when V_post < E_rev
        let i = params.current(&state, -60.0);
        assert!(i > 0.0, "Excitatory current should be positive");

        // Inhibitory synapse
        let inh_params = GradedSynapseParams::inhibitory();
        let i_inh = inh_params.current(&state, -60.0);
        assert!(i_inh < 0.0, "Inhibitory current should be negative");
    }

    #[test]
    fn test_graded_synapse2_calcium_dynamics() {
        let params = GradedSynapse2Params::default();
        let mut state = GradedSynapse2State::new();

        // Depolarize presynaptic neuron
        for _ in 0..200 {
            params.step(&mut state, 0.0, 0.1); // Above threshold
        }

        assert!(state.ca > 0.0, "Calcium should accumulate");
        assert!(state.s > 0.0, "Synapse should activate");
    }
}
