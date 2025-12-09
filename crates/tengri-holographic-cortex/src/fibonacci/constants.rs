//! Fibonacci Pentagonal Cortex Constants
//!
//! This module provides fundamental mathematical constants for the pentagonal cortex topology,
//! grounded in the golden ratio φ, Fibonacci sequences, and critical phenomena.
//!
//! # Mathematical Foundations
//!
//! ## Golden Ratio Properties
//! - φ = (1 + √5) / 2 ≈ 1.618033988749895
//! - φ⁻¹ = φ - 1 ≈ 0.618033988749895
//! - φ² = φ + 1
//! - φ × φ⁻¹ = 1
//!
//! ## Ising Critical Temperature
//! - T_c = 2 / ln(1 + √2) ≈ 2.269185314213022 (Onsager solution, 2D square lattice)
//!
//! ## Pentagon Geometry
//! - Interior angle: 108°
//! - Angular phases: [0°, 72°, 144°, 216°, 288°]
//! - Golden angle: 360° / φ² ≈ 137.5077640500378°

use std::f64::consts::PI;

// ============================================================================
// Fundamental Constants
// ============================================================================

/// Golden ratio φ = (1 + √5) / 2
///
/// The golden ratio appears throughout nature and mathematics, including:
/// - Fibonacci sequence limiting ratio
/// - Pentagon geometry
/// - Phyllotaxis patterns
/// - Self-similar fractal structures
pub const PHI: f64 = 1.618033988749895;

/// Inverse golden ratio φ⁻¹ = 1/φ = φ - 1
///
/// Related to φ by: φ × φ⁻¹ = 1 and φ - φ⁻¹ = 1
pub const PHI_INV: f64 = 0.618033988749895;

/// Ising model critical temperature (2D square lattice)
///
/// T_c = 2 / ln(1 + √2) ≈ 2.269185314213022
///
/// This is Lars Onsager's exact solution (1944) for the 2D Ising model
/// critical temperature, representing the phase transition point between
/// ordered and disordered magnetic states.
///
/// Reference: Onsager, L. (1944). "Crystal Statistics. I. A Two-Dimensional
/// Model with an Order-Disorder Transition". Physical Review. 65 (3–4): 117–149.
pub const ISING_CRITICAL_TEMP: f64 = 2.269185314213022;

/// Golden angle in degrees: 360° / φ²
///
/// The golden angle ≈ 137.5077640500378° minimizes overlapping in
/// phyllotactic patterns and appears in optimal packing arrangements.
pub const GOLDEN_ANGLE: f64 = 137.5077640500378;

// ============================================================================
// STDP Time Constants (Fibonacci Sequence)
// ============================================================================

/// STDP time constants derived from Fibonacci sequence
///
/// Values: [13, 21, 34, 55, 89] milliseconds
///
/// These time constants are used in Spike-Timing Dependent Plasticity (STDP)
/// learning rules, with Fibonacci scaling providing multi-scale temporal processing:
/// - τ₁ = 13ms: Fast synaptic dynamics
/// - τ₂ = 21ms: Standard STDP window
/// - τ₃ = 34ms: Medium-term integration
/// - τ₄ = 55ms: Extended temporal correlations
/// - τ₅ = 89ms: Long-range temporal dependencies
pub const FIBONACCI_TAU: [f64; 5] = [13.0, 21.0, 34.0, 55.0, 89.0];

// ============================================================================
// Pentagon Phase Angles
// ============================================================================

/// Pentagonal phase angles in degrees: [0°, 72°, 144°, 216°, 288°]
///
/// These angles define the vertices of a regular pentagon, evenly distributed
/// around a circle with 72° = 360° / 5 spacing.
pub const PENTAGON_PHASES: [f64; 5] = [0.0, 72.0, 144.0, 216.0, 288.0];

// ============================================================================
// Fibonacci Coupling Matrix
// ============================================================================

/// 5×5 Fibonacci coupling matrix for pentagonal cortex
///
/// Structure:
/// - Diagonal: 1.0 (self-coupling)
/// - Adjacent vertices (i, i±1 mod 5): φ (strong coupling)
/// - Skip vertices (i, i±2 mod 5): φ⁻¹ (weak coupling)
///
/// This matrix encodes the topological structure of the pentagonal cortex
/// with golden ratio-based coupling strengths.
pub const FIBONACCI_COUPLING: [[f64; 5]; 5] = [
    // Node 0: couples to 1,4 (adjacent), 2,3 (skip)
    [1.0,     PHI,     PHI_INV, PHI_INV, PHI    ],
    // Node 1: couples to 0,2 (adjacent), 3,4 (skip)
    [PHI,     1.0,     PHI,     PHI_INV, PHI_INV],
    // Node 2: couples to 1,3 (adjacent), 0,4 (skip)
    [PHI_INV, PHI,     1.0,     PHI,     PHI_INV],
    // Node 3: couples to 2,4 (adjacent), 0,1 (skip)
    [PHI_INV, PHI_INV, PHI,     1.0,     PHI    ],
    // Node 4: couples to 0,3 (adjacent), 1,2 (skip)
    [PHI,     PHI_INV, PHI_INV, PHI,     1.0    ],
];

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate Fibonacci sequence up to n terms
///
/// # Arguments
/// * `n` - Number of Fibonacci terms to generate
///
/// # Returns
/// Vector containing the first n Fibonacci numbers
///
/// # Examples
/// ```
/// use tengri_holographic_cortex::fibonacci::constants::fibonacci_sequence;
///
/// let fib = fibonacci_sequence(8);
/// assert_eq!(fib, vec![1, 1, 2, 3, 5, 8, 13, 21]);
/// ```
pub fn fibonacci_sequence(n: usize) -> Vec<u64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1];
    }

    let mut seq = Vec::with_capacity(n);
    seq.push(1);
    seq.push(1);

    for i in 2..n {
        let next = seq[i - 1] + seq[i - 2];
        seq.push(next);
    }

    seq
}

/// Check if a ratio approximates the golden ratio within tolerance
///
/// # Arguments
/// * `ratio` - The ratio to test
/// * `tolerance` - Maximum allowed deviation from φ
///
/// # Returns
/// `true` if |ratio - φ| < tolerance
///
/// # Examples
/// ```
/// use tengri_holographic_cortex::fibonacci::constants::is_golden_ratio;
///
/// assert!(is_golden_ratio(1.618, 0.001));
/// assert!(is_golden_ratio(21.0 / 13.0, 0.01));
/// ```
pub fn is_golden_ratio(ratio: f64, tolerance: f64) -> bool {
    (ratio - PHI).abs() < tolerance
}

/// Compute the golden ratio from consecutive Fibonacci numbers
///
/// As n → ∞, F(n+1) / F(n) → φ
///
/// # Arguments
/// * `n` - Index of Fibonacci number (must be ≥ 2)
///
/// # Returns
/// Approximation of φ using F(n+1) / F(n)
///
/// # Examples
/// ```
/// use tengri_holographic_cortex::fibonacci::constants::golden_ratio_approximation;
///
/// let approx = golden_ratio_approximation(20);
/// assert!((approx - 1.618033988749895).abs() < 1e-6);
/// ```
pub fn golden_ratio_approximation(n: usize) -> f64 {
    assert!(n >= 2, "Fibonacci index must be at least 2");
    let seq = fibonacci_sequence(n + 1);
    seq[n] as f64 / seq[n - 1] as f64
}

/// Convert degrees to radians
#[inline]
pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Convert radians to degrees
#[inline]
pub fn rad_to_deg(radians: f64) -> f64 {
    radians * 180.0 / PI
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // ------------------------------------------------------------------------
    // Golden Ratio Identity Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_golden_ratio_multiplication() {
        // φ × φ⁻¹ = 1
        let product = PHI * PHI_INV;
        assert!(
            (product - 1.0).abs() < EPSILON,
            "φ × φ⁻¹ must equal 1, got {}",
            product
        );
    }

    #[test]
    fn test_golden_ratio_difference() {
        // φ - φ⁻¹ = 1
        let difference = PHI - PHI_INV;
        assert!(
            (difference - 1.0).abs() < EPSILON,
            "φ - φ⁻¹ must equal 1, got {}",
            difference
        );
    }

    #[test]
    fn test_golden_ratio_squared() {
        // φ² = φ + 1
        let phi_squared = PHI * PHI;
        let phi_plus_one = PHI + 1.0;
        assert!(
            (phi_squared - phi_plus_one).abs() < EPSILON,
            "φ² must equal φ + 1, got φ² = {}, φ + 1 = {}",
            phi_squared,
            phi_plus_one
        );
    }

    #[test]
    fn test_golden_ratio_from_sqrt5() {
        // φ = (1 + √5) / 2
        let computed_phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!(
            (PHI - computed_phi).abs() < EPSILON,
            "φ constant must match (1 + √5) / 2"
        );
    }

    #[test]
    fn test_phi_inverse_from_phi() {
        // φ⁻¹ = 1 / φ
        let computed_inv = 1.0 / PHI;
        assert!(
            (PHI_INV - computed_inv).abs() < EPSILON,
            "φ⁻¹ must equal 1 / φ"
        );
    }

    // ------------------------------------------------------------------------
    // Ising Critical Temperature Test
    // ------------------------------------------------------------------------

    #[test]
    fn test_ising_critical_temp_onsager() {
        // T_c = 2 / ln(1 + √2)
        let computed_tc = 2.0 / (1.0 + 2.0_f64.sqrt()).ln();
        assert!(
            (ISING_CRITICAL_TEMP - computed_tc).abs() < EPSILON,
            "Ising T_c must match Onsager solution"
        );
    }

    // ------------------------------------------------------------------------
    // Golden Angle Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_golden_angle_formula() {
        // Golden angle = 360° / φ²
        let computed_angle = 360.0 / (PHI * PHI);
        assert!(
            (GOLDEN_ANGLE - computed_angle).abs() < EPSILON,
            "Golden angle must equal 360° / φ²"
        );
    }

    // ------------------------------------------------------------------------
    // Fibonacci Sequence Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_fibonacci_tau_sequence() {
        // Verify FIBONACCI_TAU follows Fibonacci pattern
        for i in 2..FIBONACCI_TAU.len() {
            let sum = FIBONACCI_TAU[i - 1] + FIBONACCI_TAU[i - 2];
            assert!(
                (FIBONACCI_TAU[i] - sum).abs() < EPSILON,
                "FIBONACCI_TAU[{}] = {} must equal sum of previous two: {}",
                i,
                FIBONACCI_TAU[i],
                sum
            );
        }
    }

    #[test]
    fn test_fibonacci_generation() {
        let fib = fibonacci_sequence(10);
        let expected = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        assert_eq!(fib, expected, "Generated Fibonacci sequence incorrect");
    }

    #[test]
    fn test_fibonacci_empty() {
        let fib = fibonacci_sequence(0);
        assert_eq!(fib.len(), 0, "Empty Fibonacci sequence should have length 0");
    }

    #[test]
    fn test_fibonacci_convergence_to_phi() {
        // F(n+1) / F(n) → φ as n → ∞
        let ratio = golden_ratio_approximation(30);
        assert!(
            (ratio - PHI).abs() < 1e-9,
            "Fibonacci ratio should converge to φ"
        );
    }

    // ------------------------------------------------------------------------
    // Pentagon Phase Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_pentagon_phases_spacing() {
        // Each phase should be 72° apart
        for i in 0..PENTAGON_PHASES.len() - 1 {
            let spacing = PENTAGON_PHASES[i + 1] - PENTAGON_PHASES[i];
            assert!(
                (spacing - 72.0).abs() < EPSILON,
                "Pentagon phases must be 72° apart"
            );
        }
    }

    #[test]
    fn test_pentagon_phases_wraparound() {
        // Phase 0 + 360° should equal phase 5
        let wraparound = PENTAGON_PHASES[0] + 360.0;
        let expected_next = PENTAGON_PHASES[4] + 72.0;
        assert!(
            (wraparound - expected_next).abs() < EPSILON,
            "Pentagon phases must wrap around correctly"
        );
    }

    // ------------------------------------------------------------------------
    // Fibonacci Coupling Matrix Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_coupling_matrix_diagonal() {
        // Diagonal elements should be 1.0 (self-coupling)
        for i in 0..5 {
            assert!(
                (FIBONACCI_COUPLING[i][i] - 1.0).abs() < EPSILON,
                "Diagonal element [{}][{}] must be 1.0",
                i,
                i
            );
        }
    }

    #[test]
    fn test_coupling_matrix_symmetry() {
        // Matrix should be symmetric: M[i][j] = M[j][i]
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (FIBONACCI_COUPLING[i][j] - FIBONACCI_COUPLING[j][i]).abs() < EPSILON,
                    "Coupling matrix must be symmetric: M[{}][{}] ≠ M[{}][{}]",
                    i,
                    j,
                    j,
                    i
                );
            }
        }
    }

    #[test]
    fn test_coupling_adjacent_vertices() {
        // Adjacent vertices (i, i+1 mod 5) should have coupling φ
        let adjacent_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];

        for (i, j) in adjacent_pairs.iter() {
            assert!(
                (FIBONACCI_COUPLING[*i][*j] - PHI).abs() < EPSILON,
                "Adjacent coupling [{}][{}] must be φ",
                i,
                j
            );
        }
    }

    #[test]
    fn test_coupling_skip_vertices() {
        // Skip vertices (i, i+2 mod 5) should have coupling φ⁻¹
        let skip_pairs = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)];

        for (i, j) in skip_pairs.iter() {
            assert!(
                (FIBONACCI_COUPLING[*i][*j] - PHI_INV).abs() < EPSILON,
                "Skip coupling [{}][{}] must be φ⁻¹",
                i,
                j
            );
        }
    }

    #[test]
    fn test_coupling_matrix_eigenvalue_bounds() {
        // For a symmetric matrix with elements in [φ⁻¹, φ],
        // eigenvalues should be bounded

        // Compute trace (sum of eigenvalues)
        let trace: f64 = (0..5).map(|i| FIBONACCI_COUPLING[i][i]).sum();
        assert!(
            (trace - 5.0).abs() < EPSILON,
            "Trace of coupling matrix must equal 5"
        );

        // Verify all entries are within expected range
        for i in 0..5 {
            for j in 0..5 {
                let val = FIBONACCI_COUPLING[i][j];
                assert!(
                    val >= PHI_INV - EPSILON && val <= PHI + EPSILON,
                    "Coupling value [{}][{}] = {} out of range [φ⁻¹, φ]",
                    i,
                    j,
                    val
                );
            }
        }
    }

    #[test]
    fn test_coupling_matrix_row_sums() {
        // Row sums provide insight into vertex influence
        let expected_row_sum = 1.0 + 2.0 * PHI + 2.0 * PHI_INV;

        for i in 0..5 {
            let row_sum: f64 = FIBONACCI_COUPLING[i].iter().sum();
            assert!(
                (row_sum - expected_row_sum).abs() < EPSILON,
                "Row {} sum must equal 1 + 2φ + 2φ⁻¹",
                i
            );
        }
    }

    // ------------------------------------------------------------------------
    // Helper Function Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_is_golden_ratio() {
        assert!(is_golden_ratio(PHI, 0.001));
        assert!(is_golden_ratio(1.618, 0.001));
        assert!(is_golden_ratio(21.0 / 13.0, 0.01));
        assert!(!is_golden_ratio(1.5, 0.01));
    }

    #[test]
    fn test_deg_to_rad_conversion() {
        assert!((deg_to_rad(180.0) - PI).abs() < EPSILON);
        assert!((deg_to_rad(360.0) - 2.0 * PI).abs() < EPSILON);
        assert!((deg_to_rad(90.0) - PI / 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_rad_to_deg_conversion() {
        assert!((rad_to_deg(PI) - 180.0).abs() < EPSILON);
        assert!((rad_to_deg(2.0 * PI) - 360.0).abs() < EPSILON);
        assert!((rad_to_deg(PI / 2.0) - 90.0).abs() < EPSILON);
    }

    #[test]
    fn test_angle_conversion_roundtrip() {
        let angles = [0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0];
        for angle in angles.iter() {
            let roundtrip = rad_to_deg(deg_to_rad(*angle));
            assert!(
                (roundtrip - angle).abs() < EPSILON,
                "Angle conversion roundtrip failed for {}°",
                angle
            );
        }
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_fibonacci_coupling_with_phases() {
        // Verify that pentagonal phases align with coupling structure
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    continue;
                }

                let phase_diff = (PENTAGON_PHASES[j] - PENTAGON_PHASES[i]).abs();
                let normalized_diff = if phase_diff > 180.0 {
                    360.0 - phase_diff
                } else {
                    phase_diff
                };

                // 72° = adjacent (φ coupling)
                // 144° = skip (φ⁻¹ coupling)
                if (normalized_diff - 72.0).abs() < EPSILON {
                    assert!(
                        (FIBONACCI_COUPLING[i][j] - PHI).abs() < EPSILON,
                        "72° phase difference should have φ coupling"
                    );
                } else if (normalized_diff - 144.0).abs() < EPSILON {
                    assert!(
                        (FIBONACCI_COUPLING[i][j] - PHI_INV).abs() < EPSILON,
                        "144° phase difference should have φ⁻¹ coupling"
                    );
                }
            }
        }
    }

    #[test]
    fn test_stdp_tau_golden_ratios() {
        // Verify that consecutive STDP time constants approximate φ
        for i in 1..FIBONACCI_TAU.len() {
            let ratio = FIBONACCI_TAU[i] / FIBONACCI_TAU[i - 1];
            assert!(
                is_golden_ratio(ratio, 0.05),
                "STDP tau ratio τ{}/τ{} = {} should approximate φ",
                i + 1,
                i,
                ratio
            );
        }
    }
}
