//! # Fibonacci Pentagon Topology
//!
//! A 5-engine pBit topology based on golden ratio (φ) coupling for consciousness emergence.
//!
//! ## Architecture
//!
//! ```text
//!           Engine 0
//!              *
//!            /   \
//!          φ/2   \φ/2
//!          /       \
//!    Eng 4 *       * Eng 1
//!          \       /
//!       φ/2 \     / φ/2
//!            \   /
//!        φ⁻¹/2\ /φ⁻¹/2
//!             X
//!        φ⁻¹/2/ \φ⁻¹/2
//!            /   \
//!           /     \
//!    Eng 3 *-------* Eng 2
//!            φ/2
//! ```
//!
//! ## Mathematical Foundation
//!
//! - **Coupling Matrix**: Based on golden ratio φ = (1 + √5)/2
//!   - Adjacent engines: φ/2 coupling strength
//!   - Skip-one engines: φ⁻¹/2 coupling strength
//!
//! - **Eigenvalue Spectrum** (Wolfram-verified):
//!   - λ₀ = φ + φ⁻¹ = √5 ≈ 2.236 (largest eigenvalue)
//!   - Other eigenvalues follow golden ratio relationships
//!
//! - **Temperature Modulation**: Using golden angle 2π/φ²
//!
//! ## Modules
//!
//! - **constants**: Golden ratio, Fibonacci sequences, coupling matrices
//! - **pentagon**: 5-engine topology with MSOCL integration
//! - **mobius_blend**: Hyperbolic H^11 embeddings with Möbius addition
//! - **fibonacci_stdp**: Multi-scale STDP learning (τ = 13, 21, 34, 55, 89 ms)
//! - **fractal_gnn**: L-system based fractal GNN growth
//! - **emergence**: IIT Phi consciousness metrics and criticality analysis

pub mod constants;
pub mod pentagon;
pub mod mobius_blend;
pub mod fibonacci_stdp;
pub mod fractal_gnn;
pub mod emergence;

// Re-export core constants
pub use constants::*;

// Re-export pentagon topology
pub use pentagon::{
    FibonacciPentagon, PentagonConfig, FibonacciCoupling,
    PENTAGON_ENGINES, FIBONACCI_COUPLING_SCALE, PHASE_COHERENCE_THRESHOLD,
};

// Re-export Möbius hyperbolic blending
pub use mobius_blend::{
    MobiusBlender,
    mobius_add, lorentz_lift, hyperbolic_distance, poincare_to_lorentz,
    blend_pentagon_states,
    HYPERBOLIC_DIM, LORENTZ_DIM, HYPERBOLIC_CURVATURE,
};

// Re-export Fibonacci STDP learning
pub use fibonacci_stdp::{
    FibonacciSTDP, fibonacci_stdp_weight_change, compute_stdp_balance,
    effective_time_constant,
};

// Re-export fractal GNN
pub use fractal_gnn::{
    FibonacciLSystem, FractalGNNLayer, FractalPentagonGNN,
    GOLDEN_ANGLE_RAD, FRACTAL_DIM_GOLDEN, MAX_LSYSTEM_DEPTH,
};

// Re-export emergence metrics
pub use emergence::{
    PhiCalculator, PentagonEmergence, CriticalityAnalysis, Partition,
    shannon_entropy, conditional_entropy, mutual_information, transfer_entropy,
    MIN_ENTROPY, CRITICAL_BRANCHING_RATIO, CRITICAL_HURST, AVALANCHE_EXPONENT_CRITICAL,
    CRITICALITY_TOLERANCE,
};
