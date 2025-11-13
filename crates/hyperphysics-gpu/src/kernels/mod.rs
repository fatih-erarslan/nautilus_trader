//! GPU compute kernels for HyperPhysics simulation
//!
//! High-performance WGSL shaders for pBit lattice evolution.

/// WGSL shader for pBit state update using Gillespie algorithm
pub const PBIT_UPDATE_SHADER: &str = include_str!("pbit_update.wgsl");

/// WGSL shader for energy calculation (Ising Hamiltonian)
pub const ENERGY_SHADER: &str = include_str!("energy.wgsl");

/// WGSL shader for entropy calculation (Shannon entropy)
pub const ENTROPY_SHADER: &str = include_str!("entropy.wgsl");

/// WGSL shader for GPU random number generation (Xorshift128+)
pub const RNG_SHADER: &str = include_str!("rng_xorshift128.wgsl");

/// WGSL shader for hyperbolic distance calculation
pub const DISTANCE_SHADER: &str = include_str!("distance.wgsl");

/// WGSL shader for coupling network computation
pub const COUPLING_SHADER: &str = include_str!("coupling.wgsl");

/// WGSL shader for Integrated Information (Φ) approximation
pub const PHI_SHADER: &str = include_str!("phi.wgsl");
