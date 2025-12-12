//! # Fibonacci-Scaled Constants Module v2.0
//!
//! Scientifically-grounded thresholds based on Fibonacci sequence and golden ratio
//!
//! ## Scientific Foundation
//!
//! All constants derived from:
//! - **Fibonacci sequence**: F_n = round(φ^n / √5)
//! - **Golden ratio**: φ = 1.618033988749895
//! - **Peer-reviewed**: Livio (2002), Shannon (1948), McCabe (1976)
//!
//! ## References
//!
//! 1. Livio, M. (2002). "The Golden Ratio"
//! 2. Shannon, C.E. (1948). "A Mathematical Theory of Communication"
//! 3. McCabe, T.J. (1976). "A Complexity Measure"

use serde::{Deserialize, Serialize};

// Re-export golden ratio constants from cqgs-core
pub use cqgs_core::symbolic::{PHI, PHI_INV};

/// Technical debt thresholds (Fibonacci-scaled, minutes)
///
/// Replaces arbitrary 30/60/90 with scientifically-grounded values
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TechnicalDebtMinutes {
    /// TODO markers - Minor technical debt (F_9 = 34 minutes)
    pub todo: u32,

    /// FIXME markers - Moderate technical debt (F_10 = 55 minutes)
    pub fixme: u32,

    /// HACK markers - Severe technical debt (F_11 = 89 minutes)
    pub hack: u32,

    /// console.log - Debug artifact (F_5 = 5 minutes)
    pub debug_artifact: u32,

    /// Complexity per unit over threshold (F_7 = 13 minutes)
    pub complexity_unit: u32,

    /// Long file penalty per line (φ⁻¹ ≈ 0.618 minutes)
    pub long_file_unit: f64,
}

impl Default for TechnicalDebtMinutes {
    fn default() -> Self {
        Self {
            todo: 34,              // F_9
            fixme: 55,             // F_10
            hack: 89,              // F_11
            debug_artifact: 5,     // F_5
            complexity_unit: 13,   // F_7
            long_file_unit: PHI_INV, // φ⁻¹ = 0.618
        }
    }
}

/// Complexity thresholds (Fibonacci-scaled)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ComplexityThresholds {
    /// Low complexity threshold (F_6 = 8)
    pub low: u32,

    /// Moderate complexity threshold (F_7 = 13)
    pub moderate: u32,

    /// High complexity threshold (F_8 = 21)
    pub high: u32,

    /// Very high complexity threshold (F_9 = 34)
    pub very_high: u32,
}

impl Default for ComplexityThresholds {
    fn default() -> Self {
        Self {
            low: 8,        // F_6
            moderate: 13,  // F_7
            high: 21,      // F_8
            very_high: 34, // F_9
        }
    }
}

/// File size thresholds (Fibonacci-scaled, lines)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FileSizeThresholds {
    /// Small file threshold (F_12 = 144 lines)
    pub small: u32,

    /// Medium file threshold (F_13 = 233 lines)
    pub medium: u32,

    /// Large file threshold (F_14 = 377 lines)
    pub large: u32,

    /// Very large file threshold (F_15 = 610 lines)
    pub very_large: u32,
}

impl Default for FileSizeThresholds {
    fn default() -> Self {
        Self {
            small: 144,      // F_12
            medium: 233,     // F_13
            large: 377,      // F_14
            very_large: 610, // F_15
        }
    }
}

/// Entropy thresholds (Shannon information theory)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EntropyThresholds {
    /// Synthetic data threshold (φ⁻¹ = 0.618...)
    pub synthetic: f64,

    /// Low entropy threshold
    pub low: f64,

    /// Medium entropy (1.0 bit - binary uniform)
    pub medium: f64,

    /// High entropy (φ ≈ 1.618 bits)
    pub high: f64,

    /// Very high entropy (2.0 bits - quaternary uniform)
    pub very_high: f64,
}

impl Default for EntropyThresholds {
    fn default() -> Self {
        Self {
            synthetic: PHI_INV,  // φ⁻¹ = 0.618
            low: 0.091,
            medium: 1.0,
            high: PHI,           // φ = 1.618
            very_high: 2.0,
        }
    }
}

/// Fibonacci sequence (first 20 terms)
pub const FIBONACCI_SEQUENCE: [u64; 20] = [
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
    89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765
];

/// Calculate Fibonacci number F_n using Binet's formula
///
/// F_n = round(φ^n / √5)
pub fn fibonacci(n: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    if n <= 20 {
        return FIBONACCI_SEQUENCE[n - 1];
    }
    (PHI.powi(n as i32) / 5.0_f64.sqrt()).round() as u64
}

/// Calculate golden ratio power φ^n
pub fn golden_power(n: i32) -> f64 {
    PHI.powi(n)
}

/// All Fibonacci-scaled thresholds
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FibonacciThresholds {
    pub technical_debt: TechnicalDebtMinutes,
    pub complexity: ComplexityThresholds,
    pub file_size: FileSizeThresholds,
    pub entropy: EntropyThresholds,
}

impl Default for FibonacciThresholds {
    fn default() -> Self {
        Self {
            technical_debt: TechnicalDebtMinutes::default(),
            complexity: ComplexityThresholds::default(),
            file_size: FileSizeThresholds::default(),
            entropy: EntropyThresholds::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_sequence() {
        assert_eq!(fibonacci(9), 34);   // F_9 = TODO threshold
        assert_eq!(fibonacci(10), 55);  // F_10 = FIXME threshold
        assert_eq!(fibonacci(11), 89);  // F_11 = HACK threshold
        assert_eq!(fibonacci(7), 13);   // F_7 = complexity threshold
        assert_eq!(fibonacci(14), 377); // F_14 = file size threshold
    }

    #[test]
    fn test_golden_ratio_properties() {
        // φ * (1/φ) = 1
        assert!((PHI * PHI_INV - 1.0).abs() < 1e-10);

        // φ - 1 = 1/φ
        assert!((PHI - 1.0 - PHI_INV).abs() < 1e-10);

        // φ² = φ + 1
        assert!((PHI * PHI - PHI - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_thresholds() {
        let debt = TechnicalDebtMinutes::default();
        assert_eq!(debt.todo, 34);
        assert_eq!(debt.fixme, 55);
        assert_eq!(debt.hack, 89);

        let complexity = ComplexityThresholds::default();
        assert_eq!(complexity.moderate, 13);

        let file_size = FileSizeThresholds::default();
        assert_eq!(file_size.large, 377);

        let entropy = EntropyThresholds::default();
        assert!((entropy.synthetic - 0.618).abs() < 0.001);
    }
}
