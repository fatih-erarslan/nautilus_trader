//! # Tengri Holographic Cortex
//!
//! A unified pBit-based cognitive architecture combining Graph Neural Networks (GNN)
//! and Spiking Neural Networks (SNN) in an 11D Hyperbolic Lattice Spacetime Continuum.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ (E) MSOCL - Meta-Stable Oscillatory Control Layer               │
//! │     • Global phase coordination (Kuramoto model)                │
//! │     • Temperature modulation across engines                     │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ (A) 4-Engine pBit Topology (2×2 Square)                         │
//! │     ┌───────┐     ┌───────┐                                    │
//! │     │ Eng A │◀───▶│ Eng B │   • Local pBit dynamics (AVX2)     │
//! │     └───────┘     └───────┘   • Boltzmann sampling             │
//! │          ▲             ▲      • Cross-coupling tensors         │
//! │          │   K^αβ      │                                       │
//! │          ▼             ▼                                       │
//! │     ┌───────┐     ┌───────┐                                    │
//! │     │ Eng D │◀───▶│ Eng C │   • Möbius blending to H¹¹        │
//! │     └───────┘     └───────┘                                    │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ (C) Ultra-Fast Cortical Bus (UFCB)                              │
//! │     • Tier A: <50μs spikes (pinned hugepages)                   │
//! │     • Tier B: <1ms embeddings (GPU P2P)                         │
//! │     • Tier C: <10ms model shards (NVMe streaming)               │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ (B) 11D Hyperbolic Relational Holographic Substrate             │
//! │     • Lorentz model: ⟨x,y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ                 │
//! │     • exp/log maps for tangent space ↔ hyperboloid              │
//! │     • Gyrovector message passing                                │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ (D) HNSW + LSH Memory Fabric                                    │
//! │     • LSH: k=8 hash functions, L=32 tables                      │
//! │     • HNSW: M=16-32, efConstruction=200                         │
//! │     • Hyperbolic distance for similarity                        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Wolfram-Verified Mathematical Foundations
//!
//! All constants and formulas have been formally verified using Wolfram:
//!
//! ### Ising Model (2D Square Lattice)
//! - Critical temperature: T_c = 2/ln(1+√2) = 2.269185314213022
//!
//! ### pBit Sampling
//! - P(s=+1) = σ((h-bias)/T) = 1/(1 + exp(-(h-bias)/T))
//! - At h=0, bias=0, T=1: P = 0.5 (verified)
//!
//! ### Hyperbolic Geometry (Lorentz H¹¹)
//! - Constraint: -x₀² + x₁² + ... + x₁₁² = -1
//! - Lift from R¹¹: x₀ = √(1 + ||z||²)
//! - Distance: d(x,y) = acosh(-⟨x,y⟩_L)
//!
//! ### Möbius Addition (Poincaré Ball)
//! - x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
//!
//! ### STDP Learning
//! - LTP (Δt > 0): ΔW = A₊ × exp(-Δt/τ₊)
//! - LTD (Δt < 0): ΔW = -A₋ × exp(Δt/τ₋)
//! - At Δt=10ms, A₊=0.1, τ₊=20ms: ΔW = 0.0607 (verified)
//!
//! ### Annealing Schedule
//! - Optimal (convergence guarantee): T(t) = T₀/ln(1+t)
//! - Fast (suboptimal): T(t) = T₀ × α^t, α=0.99

pub mod constants;
pub mod msocl;
pub mod engine;
pub mod topology;
pub mod hyperbolic;
pub mod cortical_bus;
pub mod memory_fabric;
pub mod gpu;
pub mod simd;
pub mod eligibility;
pub mod csr;
pub mod ricci;
pub mod sgnn;

pub use constants::*;
pub use msocl::{Msocl, MsoclPhase, MsoclConfig};
pub use engine::{PBitEngine, EngineConfig};
pub use topology::{Cortex4, TopologyConfig, CouplingTensor, SmallWorldTopology64, SmallWorldConfig, TopologyStats};
pub use hyperbolic::{LorentzPoint11, MobiusBlend, HyperbolicOps};
pub use cortical_bus::{CorticalBus, BusTier, SpikePacket};
pub use memory_fabric::{MemoryFabric, MemoryConfig};
pub use gpu::{GpuConfig, GpuNodeData, GpuEdgeData, MemoryEstimate, PerfEstimate};
pub use eligibility::{SparseEligibilityTrace, TraceParams, TraceStats};
pub use csr::CSRGraph;
pub use ricci::{Regime, RegimeDetector, RicciGraph, forman_ricci};
pub use sgnn::{LIFNeuron, LIFConfig, SpikeEvent, Synapse, SynapseConfig, SGNNLayer, LayerConfig, MultiScaleSGNN, MultiScaleConfig};

use thiserror::Error;

/// Errors in the Tengri Holographic Cortex
#[derive(Error, Debug)]
pub enum CortexError {
    #[error("Engine error: {0}")]
    EngineError(String),
    
    #[error("Topology error: {0}")]
    TopologyError(String),
    
    #[error("Hyperbolic geometry error: {0}")]
    HyperbolicError(String),
    
    #[error("Cortical bus error: {0}")]
    BusError(String),
    
    #[error("Memory fabric error: {0}")]
    MemoryError(String),
    
    #[error("MSOCL phase error: {0}")]
    MsoclError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, CortexError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ising_critical_temperature() {
        let tc = ISING_CRITICAL_TEMP;
        let expected = 2.269185314213022;
        assert!((tc - expected).abs() < 1e-10, "T_c = {tc} vs expected {expected}");
    }
    
    #[test]
    fn test_pbit_probability_balanced() {
        // P(s=1) at h=0, bias=0, T=1 should be 0.5
        let p = pbit_probability(0.0, 0.0, 1.0);
        assert!((p - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_pbit_probability_high_field() {
        // P(s=1) at h=1, bias=0, T=0.1 should be near 1
        let p = pbit_probability(1.0, 0.0, 0.1);
        assert!(p > 0.9999);
    }
    
    #[test]
    fn test_boltzmann_normalization() {
        let energies = [0.0, 1.0, 2.0, 3.0];
        let probs = boltzmann_probabilities(&energies, 1.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
