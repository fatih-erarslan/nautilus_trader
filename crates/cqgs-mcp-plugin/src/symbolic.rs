//! # Symbolic Computation Module v2.0
//!
//! Symbolic computation with Shannon entropy - Native integration with cqgs-core
//!
//! ## Updates in v2.0
//! - ✅ Direct integration with cqgs-core native implementation
//! - ✅ Full symbolic differentiation and integration
//! - ✅ Expression evaluation and simplification
//! - ✅ Peer-reviewed Shannon entropy (Shannon 1948)
//! - ✅ Golden ratio constants (Livio 2002)

// Re-export from cqgs-core native implementation
pub use cqgs_core::symbolic::*;

use anyhow::Result;

/// Convenience wrapper for Shannon entropy computation (MCP compatibility)
pub fn compute_entropy(probabilities: &[f64]) -> Result<f64> {
    shannon_entropy(probabilities)
        .map_err(|e| anyhow::anyhow!("Entropy calculation failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_entropy_wrapper() {
        // Uniform distribution (2 outcomes) = 1 bit
        let probs = vec![0.5, 0.5];
        let entropy = compute_entropy(&probs).unwrap();
        assert!((entropy - 1.0).abs() < 1e-10, "Expected 1.0 bit, got {}", entropy);
    }

    #[test]
    fn test_golden_ratio_constants() {
        // Verify φ * (1/φ) = 1
        assert!((PHI * PHI_INV - 1.0).abs() < 1e-10);
        // Verify φ - 1 = 1/φ
        assert!((PHI - 1.0 - PHI_INV).abs() < 1e-10);
    }
}
