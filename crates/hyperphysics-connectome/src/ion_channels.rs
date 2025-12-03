//! Ion Channel Models
//!
//! Implements biophysically detailed ion channel models based on
//! electrophysiological studies of C. elegans neurons.
//!
//! ## References
//!
//! - Boyle & Cohen (2008): Caenorhabditis elegans body wall muscles are simple
//!   actuators. PLOS Computational Biology.
//! - Nicoletti et al. (2019): Biophysical modeling of C. elegans neurons
//! - Mellem et al. (2008): Action potentials in C. elegans

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Potassium Channels
// ============================================================================

/// Slow potassium channel (k_slow) parameters
///
/// This channel is responsible for the slow afterhyperpolarization
/// and contributes to spike frequency adaptation in C. elegans neurons.
///
/// Based on Boyle & Cohen (2008) muscle cell model.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KSlowParams {
    /// Maximum conductance density (mS/cm²)
    pub g_max: f32,
    /// Reversal potential (mV)
    pub e_k: f32,
    /// Activation time constant (ms)
    pub tau_act: f32,
    /// Half-activation voltage (mV)
    pub v_half: f32,
    /// Slope factor (mV)
    pub slope: f32,
}

impl Default for KSlowParams {
    fn default() -> Self {
        Self {
            g_max: 1.8,       // mS/cm²
            e_k: -80.0,       // mV
            tau_act: 100.0,   // ms - slow kinetics
            v_half: -10.0,    // mV
            slope: 15.0,      // mV
        }
    }
}

impl KSlowParams {
    /// Steady-state activation
    #[inline]
    pub fn n_inf(&self, v: f32) -> f32 {
        1.0 / (1.0 + ((self.v_half - v) / self.slope).exp())
    }

    /// Voltage-dependent time constant
    #[inline]
    pub fn tau_n(&self, v: f32) -> f32 {
        // Bell-shaped time constant curve
        self.tau_act / (1.0 + ((v - self.v_half).abs() / 30.0).exp())
    }

    /// Calculate current
    #[inline]
    pub fn current(&self, state: &KSlowState, v: f32) -> f32 {
        let n = state.n;
        self.g_max * n * n * n * n * (v - self.e_k)
    }

    /// Update channel state
    pub fn step(&self, state: &mut KSlowState, v: f32, dt: f32) {
        let n_inf = self.n_inf(v);
        let tau = self.tau_n(v);
        state.n += (n_inf - state.n) * dt / tau;
        state.n = state.n.clamp(0.0, 1.0);
    }
}

/// State for slow potassium channel
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KSlowState {
    /// Activation gating variable (0-1)
    pub n: f32,
}

impl KSlowState {
    /// Create at resting state
    pub fn resting(v: f32, params: &KSlowParams) -> Self {
        Self {
            n: params.n_inf(v),
        }
    }
}

/// Fast potassium channel (k_fast / Kv) parameters
///
/// A-type potassium current responsible for rapid repolarization
/// and action potential termination.
///
/// Based on Mellem et al. (2008) AWC neuron recordings.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KFastParams {
    /// Maximum conductance density (mS/cm²)
    pub g_max: f32,
    /// Reversal potential (mV)
    pub e_k: f32,
    /// Activation V_half (mV)
    pub v_half_act: f32,
    /// Inactivation V_half (mV)
    pub v_half_inact: f32,
    /// Activation slope (mV)
    pub slope_act: f32,
    /// Inactivation slope (mV)
    pub slope_inact: f32,
    /// Activation time constant (ms)
    pub tau_act: f32,
    /// Inactivation time constant (ms)
    pub tau_inact: f32,
}

impl Default for KFastParams {
    fn default() -> Self {
        Self {
            g_max: 36.0,       // mS/cm² - high conductance for fast repolarization
            e_k: -77.0,        // mV
            v_half_act: -20.0, // mV
            v_half_inact: -60.0, // mV
            slope_act: 15.0,   // mV
            slope_inact: -10.0, // mV (negative for inactivation)
            tau_act: 1.0,      // ms - fast activation
            tau_inact: 50.0,   // ms - slower inactivation
        }
    }
}

impl KFastParams {
    /// Activation steady-state
    #[inline]
    pub fn n_inf(&self, v: f32) -> f32 {
        1.0 / (1.0 + ((self.v_half_act - v) / self.slope_act).exp())
    }

    /// Inactivation steady-state
    #[inline]
    pub fn h_inf(&self, v: f32) -> f32 {
        1.0 / (1.0 + ((v - self.v_half_inact) / (-self.slope_inact)).exp())
    }

    /// Calculate current
    #[inline]
    pub fn current(&self, state: &KFastState, v: f32) -> f32 {
        let n = state.n;
        self.g_max * n * n * n * n * state.h * (v - self.e_k)
    }

    /// Update channel state
    pub fn step(&self, state: &mut KFastState, v: f32, dt: f32) {
        // Activation
        let n_inf = self.n_inf(v);
        state.n += (n_inf - state.n) * dt / self.tau_act;
        state.n = state.n.clamp(0.0, 1.0);

        // Inactivation
        let h_inf = self.h_inf(v);
        state.h += (h_inf - state.h) * dt / self.tau_inact;
        state.h = state.h.clamp(0.0, 1.0);
    }
}

/// State for fast potassium channel
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KFastState {
    /// Activation gating variable
    pub n: f32,
    /// Inactivation gating variable
    pub h: f32,
}

impl Default for KFastState {
    fn default() -> Self {
        Self { n: 0.0, h: 1.0 }
    }
}

impl KFastState {
    /// Create at resting state
    pub fn resting(v: f32, params: &KFastParams) -> Self {
        Self {
            n: params.n_inf(v),
            h: params.h_inf(v),
        }
    }
}

// ============================================================================
// Calcium Channels
// ============================================================================

/// Calcium channel (ca_boyle) parameters
///
/// L-type calcium channel based on Boyle & Cohen (2008) body wall muscle model.
/// This channel is critical for muscle contraction and graded potential generation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CaBoyleParams {
    /// Maximum conductance density (mS/cm²)
    pub g_max: f32,
    /// Reversal potential (mV) - calculated from Nernst equation
    pub e_ca: f32,
    /// Half-activation voltage (mV)
    pub v_half: f32,
    /// Activation slope (mV)
    pub slope: f32,
    /// Activation time constant (ms)
    pub tau_act: f32,
    /// Calcium removal rate constant (1/ms)
    pub k_ca: f32,
    /// Resting calcium concentration (mM)
    pub ca_rest: f32,
}

impl Default for CaBoyleParams {
    fn default() -> Self {
        Self {
            g_max: 4.0,        // mS/cm² (from Boyle & Cohen 2008)
            e_ca: 60.0,        // mV
            v_half: -24.0,     // mV (from Boyle & Cohen 2008)
            slope: 6.5,        // mV
            tau_act: 2.0,      // ms
            k_ca: 0.1,         // 1/ms - calcium removal rate
            ca_rest: 0.0001,   // 100 nM resting calcium
        }
    }
}

impl CaBoyleParams {
    /// Steady-state activation
    #[inline]
    pub fn m_inf(&self, v: f32) -> f32 {
        1.0 / (1.0 + ((self.v_half - v) / self.slope).exp())
    }

    /// Calculate current
    ///
    /// Uses GHK constant field equation approximation
    #[inline]
    pub fn current(&self, state: &CaBoyleState, v: f32) -> f32 {
        let m = state.m;
        self.g_max * m * m * (v - self.e_ca)
    }

    /// Calculate calcium influx (affects intracellular [Ca²⁺])
    #[inline]
    pub fn ca_influx(&self, state: &CaBoyleState, v: f32) -> f32 {
        // Calcium influx proportional to current magnitude
        let i_ca = -self.current(state, v); // Negative because current is inward
        if i_ca > 0.0 {
            i_ca * 0.001 // Scale factor for conversion to mM
        } else {
            0.0
        }
    }

    /// Update channel state and calcium dynamics
    pub fn step(&self, state: &mut CaBoyleState, v: f32, dt: f32) {
        // Activation kinetics
        let m_inf = self.m_inf(v);
        state.m += (m_inf - state.m) * dt / self.tau_act;
        state.m = state.m.clamp(0.0, 1.0);

        // Calcium dynamics
        let ca_influx = self.ca_influx(state, v);
        let ca_removal = self.k_ca * (state.ca - self.ca_rest);
        state.ca += (ca_influx - ca_removal) * dt;
        state.ca = state.ca.max(0.0);
    }
}

/// State for calcium channel
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CaBoyleState {
    /// Activation gating variable
    pub m: f32,
    /// Intracellular calcium concentration (mM)
    pub ca: f32,
}

impl Default for CaBoyleState {
    fn default() -> Self {
        Self {
            m: 0.0,
            ca: 0.0001, // 100 nM
        }
    }
}

impl CaBoyleState {
    /// Create at resting state
    pub fn resting(v: f32, params: &CaBoyleParams) -> Self {
        Self {
            m: params.m_inf(v),
            ca: params.ca_rest,
        }
    }
}

// ============================================================================
// Leak Channels
// ============================================================================

/// Leak channel parameters
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LeakParams {
    /// Leak conductance (mS/cm²)
    pub g_leak: f32,
    /// Leak reversal potential (mV)
    pub e_leak: f32,
}

impl Default for LeakParams {
    fn default() -> Self {
        Self {
            g_leak: 0.3,    // mS/cm²
            e_leak: -55.0,  // mV
        }
    }
}

impl LeakParams {
    /// Calculate leak current
    #[inline]
    pub fn current(&self, v: f32) -> f32 {
        self.g_leak * (v - self.e_leak)
    }
}

// ============================================================================
// Composite Ion Channel Set for C. elegans Neurons
// ============================================================================

/// Complete ion channel set for a C. elegans neuron
///
/// Combines multiple channel types to model realistic neuronal dynamics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IonChannelSet {
    /// Fast potassium channel
    pub k_fast: Option<KFastParams>,
    /// Slow potassium channel
    pub k_slow: Option<KSlowParams>,
    /// Calcium channel
    pub ca: Option<CaBoyleParams>,
    /// Leak channel
    pub leak: LeakParams,
}

impl Default for IonChannelSet {
    fn default() -> Self {
        Self {
            k_fast: Some(KFastParams::default()),
            k_slow: Some(KSlowParams::default()),
            ca: Some(CaBoyleParams::default()),
            leak: LeakParams::default(),
        }
    }
}

impl IonChannelSet {
    /// Create minimal channel set (leak only)
    pub fn minimal() -> Self {
        Self {
            k_fast: None,
            k_slow: None,
            ca: None,
            leak: LeakParams::default(),
        }
    }

    /// Create motor neuron channel set (Boyle & Cohen model)
    pub fn motor_neuron() -> Self {
        Self {
            k_fast: Some(KFastParams {
                g_max: 36.0,
                ..Default::default()
            }),
            k_slow: Some(KSlowParams {
                g_max: 1.8,
                ..Default::default()
            }),
            ca: Some(CaBoyleParams {
                g_max: 4.0,
                ..Default::default()
            }),
            leak: LeakParams {
                g_leak: 0.3,
                e_leak: -55.0,
            },
        }
    }

    /// Create sensory neuron channel set
    pub fn sensory_neuron() -> Self {
        Self {
            k_fast: Some(KFastParams {
                g_max: 20.0,
                ..Default::default()
            }),
            k_slow: Some(KSlowParams {
                g_max: 1.0,
                ..Default::default()
            }),
            ca: Some(CaBoyleParams {
                g_max: 2.0,
                ..Default::default()
            }),
            leak: LeakParams {
                g_leak: 0.5,
                e_leak: -60.0,
            },
        }
    }

    /// Create interneuron channel set
    pub fn interneuron() -> Self {
        Self {
            k_fast: Some(KFastParams {
                g_max: 30.0,
                ..Default::default()
            }),
            k_slow: Some(KSlowParams {
                g_max: 2.0,
                ..Default::default()
            }),
            ca: Some(CaBoyleParams {
                g_max: 3.0,
                ..Default::default()
            }),
            leak: LeakParams {
                g_leak: 0.4,
                e_leak: -58.0,
            },
        }
    }

    /// Calculate total ionic current
    pub fn total_current(&self, state: &IonChannelState, v: f32) -> f32 {
        let mut i_total = self.leak.current(v);

        if let Some(ref params) = self.k_fast {
            i_total += params.current(&state.k_fast, v);
        }
        if let Some(ref params) = self.k_slow {
            i_total += params.current(&state.k_slow, v);
        }
        if let Some(ref params) = self.ca {
            i_total += params.current(&state.ca, v);
        }

        i_total
    }

    /// Update all channel states
    pub fn step(&self, state: &mut IonChannelState, v: f32, dt: f32) {
        if let Some(ref params) = self.k_fast {
            params.step(&mut state.k_fast, v, dt);
        }
        if let Some(ref params) = self.k_slow {
            params.step(&mut state.k_slow, v, dt);
        }
        if let Some(ref params) = self.ca {
            params.step(&mut state.ca, v, dt);
        }
    }
}

/// Combined state for all ion channels
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IonChannelState {
    /// Fast potassium channel state
    pub k_fast: KFastState,
    /// Slow potassium channel state
    pub k_slow: KSlowState,
    /// Calcium channel state
    pub ca: CaBoyleState,
}

impl IonChannelState {
    /// Create at resting voltage
    pub fn resting(v: f32, channels: &IonChannelSet) -> Self {
        Self {
            k_fast: channels.k_fast.as_ref()
                .map(|p| KFastState::resting(v, p))
                .unwrap_or_default(),
            k_slow: channels.k_slow.as_ref()
                .map(|p| KSlowState::resting(v, p))
                .unwrap_or_default(),
            ca: channels.ca.as_ref()
                .map(|p| CaBoyleState::resting(v, p))
                .unwrap_or_default(),
        }
    }

    /// Get intracellular calcium concentration
    pub fn calcium(&self) -> f32 {
        self.ca.ca
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_slow_activation() {
        let params = KSlowParams::default();

        // Below threshold
        let n_low = params.n_inf(-60.0);
        assert!(n_low < 0.2, "k_slow should be mostly closed at rest");

        // Above threshold
        let n_high = params.n_inf(20.0);
        assert!(n_high > 0.8, "k_slow should be mostly open at +20mV");
    }

    #[test]
    fn test_k_fast_inactivation() {
        let params = KFastParams::default();
        let mut state = KFastState::default();

        // At depolarized potential, inactivation should develop
        for _ in 0..100 {
            params.step(&mut state, 0.0, 1.0);
        }
        assert!(state.h < 0.5, "k_fast should inactivate at 0mV");
    }

    #[test]
    fn test_ca_boyle_activation() {
        let params = CaBoyleParams::default();
        let mut state = CaBoyleState::default();

        // Depolarize to activate
        for _ in 0..20 {
            params.step(&mut state, 0.0, 0.5);
        }
        assert!(state.m > 0.5, "Ca channel should activate at 0mV");
        assert!(state.ca > 0.0001, "Calcium should accumulate");
    }

    #[test]
    fn test_ion_channel_set_motor() {
        let channels = IonChannelSet::motor_neuron();
        let mut state = IonChannelState::resting(-65.0, &channels);

        // At rest, current should be near zero
        let i_rest = channels.total_current(&state, -65.0);
        assert!(i_rest.abs() < 5.0, "Current should be small at rest: got {}", i_rest);

        // Depolarize and check currents
        for _ in 0..100 {
            channels.step(&mut state, 0.0, 0.1);
        }
        let i_depol = channels.total_current(&state, 0.0);
        // At 0mV: K+ produces outward (positive) current, Ca2+ produces inward (negative) current
        // Net depends on conductance balance; K+ dominates so net should be positive (outward)
        // Using Ohm's law: I = g*(V - E_rev)
        // K+: g_k*(0 - (-77)) > 0 (outward)
        // Ca2+: g_ca*(0 - 60) < 0 (inward)
        // With larger K+ conductance, net current should be significantly non-zero
        assert!(i_depol.abs() > 10.0, "Net current should be significant at 0mV: got {}", i_depol);
    }

    #[test]
    fn test_leak_current() {
        let leak = LeakParams::default();

        // At reversal potential, no current
        let i_rev = leak.current(leak.e_leak);
        assert!(i_rev.abs() < 1e-6, "No current at reversal potential");

        // Above reversal, outward current
        let i_out = leak.current(0.0);
        assert!(i_out > 0.0, "Outward current above E_leak");
    }
}
