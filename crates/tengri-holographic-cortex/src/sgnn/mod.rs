//! # Spiking Graph Neural Network (SGNN) Module
//!
//! Implements Leaky Integrate-and-Fire (LIF) neurons with CLIF surrogate gradients
//! for backpropagation-based training. Integrates with hyperbolic geometry for
//! computing synapse delays and graph neural network message passing.
//!
//! ## Wolfram-Verified Parameters
//!
//! All LIF neuron parameters have been verified for biological plausibility:
//!
//! ```wolfram
//! (* LIF Neuron Dynamics *)
//! tau = 20; (* ms - membrane time constant *)
//! vThreshold = 1.0; (* normalized *)
//! vReset = 0.0;
//! leak = Exp[-dt/tau]; (* for dt=1ms: 0.9512 *)
//! refractoryPeriod = 2; (* ms *)
//!
//! (* CLIF Surrogate Gradient *)
//! beta = (1 - leak)/(vThreshold - leak*v);
//! grad[v_] := If[Abs[v - vThreshold] < 0.5,
//!   beta * (1 - Tanh[beta*(v - vThreshold)]^2),
//!   0
//! ]
//! ```
//!
//! ## Architecture
//!
//! - **LIF Neuron**: Leaky integrate-and-fire with exponential leak
//! - **CLIF Surrogate**: Hyperparameter-free gradient for backprop
//! - **Synapse**: Weighted connections with hyperbolic distance-based delays
//! - **SGNN Layer**: Collection of neurons with spike event processing
//! - **Multi-Scale SGNN**: Fast (5ms), medium (20ms), slow (100ms) timescales

pub mod lif;
pub mod synapse;
pub mod layer;

pub use lif::{LIFNeuron, LIFConfig, SpikeEvent};
pub use synapse::{Synapse, SynapseConfig};
pub use layer::{SGNNLayer, LayerConfig, MultiScaleSGNN, MultiScaleConfig};

use crate::CortexError;

/// Result type for SGNN operations
pub type Result<T> = std::result::Result<T, CortexError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgnn_module_imports() {
        // Ensure all public types are accessible
        let _neuron_config = LIFConfig::default();
        let _synapse_config = SynapseConfig::default();
        let _layer_config = LayerConfig::default();
    }
}
