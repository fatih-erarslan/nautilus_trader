//! Quantum-Enhanced Reservoir Computing (QERC)
//!
//! Hybrid classical-quantum reservoir computing for time series analysis.
//! Uses pBit-based quantum kernels for nonlinear temporal pattern recognition.
//!
//! ## Architecture
//!
//! ```text
//! Input → W_in → Reservoir (Echo State Network) → pBit Kernel → W_out → Output
//!                        ↓                            ↓
//!                 Leaking dynamics            Boltzmann sampling
//! ```
//!
//! ## Key Features
//!
//! - 500-node reservoir with configurable pBit quantum kernel
//! - Temporal processing with sliding windows
//! - pBit-based nonlinear transformations (replaces PennyLane)
//! - Efficient Boltzmann sampling for quantum-inspired computation

pub mod pbit_reservoir;

// Re-export main types
pub use pbit_reservoir::{PBitReservoir, PBitReservoirConfig, ReservoirOutput};

// Legacy support - deprecated alias
#[deprecated(note = "Use PBitReservoir instead - this was incorrectly named")]
pub type QuantumErrorCorrection = PBitReservoir;