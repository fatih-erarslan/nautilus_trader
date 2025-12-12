//! # Quantum Knowledge System Cognitive API
//!
//! **The Super Pill of Wisdom** - A comprehensive 9-layer cognitive architecture
//! exposing thermodynamic computing, active inference, consciousness, meta-cognition, and quantum innovations.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                   QKS COGNITIVE API (9 Layers)                      │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  Layer 9: Quantum         │ Tensor networks, temporal reservoir     │
//! │  Layer 8: Integration     │ Full cognitive loop orchestration       │
//! │  Layer 7: MetaCognition   │ Self-model, introspection, meta-learning│
//! │  Layer 6: Consciousness   │ IIT Φ, global workspace, awareness      │
//! │  Layer 5: Collective      │ Swarm coordination, consensus           │
//! │  Layer 4: Learning        │ Three-factor STDP, reasoning bank       │
//! │  Layer 3: Decision        │ Active inference, Expected Free Energy  │
//! │  Layer 2: Cognitive       │ Attention, memory, pattern recognition  │
//! │  Layer 1: Thermodynamic   │ Energy management, Tc=2.269185          │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use qks_plugin::api::prelude::*;
//!
//! // Layer 1: Thermodynamic foundation
//! let energy = get_energy()?;
//! set_temperature(2.269185)?; // Critical temperature
//!
//! // Layer 3: Active inference decision
//! let action = select_action(&observations, &preferences)?;
//!
//! // Layer 6: Consciousness check
//! let phi = compute_phi(&network_state)?;
//! if is_conscious(phi) {
//!     println!("System is conscious: Φ = {}", phi);
//! }
//!
//! // Layer 8: Full cognitive cycle
//! let output = cognitive_cycle(&sensory_input)?;
//! ```
//!
//! ## Layer Descriptions
//!
//! - **Layer 1 (Thermodynamic)**: Energy-efficient computing with Ising model criticality
//! - **Layer 2 (Cognitive)**: Basic perception, attention allocation, and memory
//! - **Layer 3 (Decision)**: Active inference with Expected Free Energy minimization
//! - **Layer 4 (Learning)**: Synaptic plasticity with three-factor STDP
//! - **Layer 5 (Collective)**: Multi-agent swarm coordination and distributed consensus
//! - **Layer 6 (Consciousness)**: Integrated Information Theory (IIT) and global workspace
//! - **Layer 7 (MetaCognition)**: Self-modeling, introspection, and meta-learning
//! - **Layer 8 (Integration)**: Homeostatic orchestration of all cognitive layers
//! - **Layer 9 (Quantum)**: Tensor networks, temporal reservoir, compressed states, circuit knitting

pub mod thermodynamic;
pub mod cognitive;
pub mod decision;
pub mod learning;
pub mod collective;
pub mod consciousness;
pub mod metacognition;
pub mod integration;
pub mod quantum;
pub mod prelude;

// Re-export all public APIs
pub use thermodynamic::*;
pub use cognitive::*;
pub use decision::*;
pub use learning::*;
pub use collective::*;
pub use consciousness::*;
pub use metacognition::*;
pub use integration::*;
pub use quantum::*;
