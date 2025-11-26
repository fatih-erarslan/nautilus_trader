//! Quantum-inspired neural network components
//!
//! This module provides quantum-inspired classical implementations of
//! neural network components that leverage concepts from quantum computing
//! without requiring actual quantum hardware. These implementations use
//! biologically-plausible dynamics and complex-valued operations.
//!
//! # Features
//!
//! - **State Encoding**: Multiple encoding schemes (Amplitude, Angle, Phase)
//! - **Complex-Valued Operations**: Using complex numbers for richer state spaces
//! - **BioCognitive LSTM**: LSTM with biological neural dynamics
//! - **Quantum-Inspired Attention**: Attention using inner products in Hilbert space
//!
//! # Mathematical Foundation
//!
//! The quantum-inspired approach uses:
//! - Complex-valued state vectors: |ψ⟩ = Σᵢ αᵢ|i⟩
//! - Unitary-like transformations preserving norm
//! - Phase-aware operations for interference effects

mod types;
mod config;
mod encoding;
mod bio_cognitive;
mod complex_lstm;

pub use types::*;
pub use config::*;
pub use encoding::*;
pub use bio_cognitive::*;
pub use complex_lstm::*;
