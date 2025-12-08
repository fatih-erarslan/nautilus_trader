//! Functional expansion implementations with SIMD acceleration
//! 
//! High-performance feature expansion using trigonometric, polynomial,
//! and orthogonal polynomial basis functions.

use std::f64::consts::PI;
use nalgebra::{DMatrix, DVector};
use ndarray::prelude::*;
use anyhow::{Result, anyhow};
use tracing::{debug, warn};

#[cfg(feature = "simd")]
use wide::f64x4;
#[cfg(feature = "simd")]
// use simdeez::prelude::*; // Prelude doesn't exist in simdeez
use simdeez::*;

use crate::{ELMConfig, ExpansionType};

/// High-performance functional expansion engine
pub struct FunctionalExpansion {
    /// Configuration
    config: ELMConfig,
    
    /// Output dimension after expansion
    output_dim: usize,
    
    /// Random weights for non-linear combinations
    expansion_weights: Option<DMatrix<f64>>,
    
    /// Precomputed coefficients for orthogonal polynomials
    polynomial_coeffs: Option<Vec<Vec<f64>>>,
}

impl FunctionalExpansion {
    /// Create new functional expansion engine
    pub fn new(config: &ELMConfig) -> Result<Self> {
        let output_dim = Self::calculate_output_dim(config);
        
        // Initialize random weights for expansions
        let expansion_weights = if !config.per_feature_expansion {
            Some(Self::generate_expansion_weights(config)?)
        } else {
            None
        };
        
        // Precompute polynomial coefficients
        let polynomial_coeffs = match config.expansion_type {
            ExpansionType::Chebyshev | ExpansionType::Hermite => {
                Some(Self::precompute_polynomial_coeffs(config.expansion_order)?)
            }
            _ => None,
        };
        
        Ok(Self {
            config: config.clone(),
            output_dim,
            expansion_weights,
            polynomial_coeffs,
        })
    }
    
    /// Get output dimension after expansion
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    
    /// Expand input features using configured expansion type
    pub fn expand(&self, inputs: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let n_samples = inputs.nrows();
        let mut expanded = DMatrix::zeros(n_samples, self.output_dim);
        
        match self.config.expansion_type {
            ExpansionType::Trigonometric => self.expand_trigonometric(inputs, &mut expanded)?,
            ExpansionType::Polynomial => self.expand_polynomial(inputs, &mut expanded)?,
            ExpansionType::Chebyshev => self.expand_chebyshev(inputs, &mut expanded)?,
            ExpansionType::Hermite => self.expand_hermite(inputs, &mut expanded)?,
            ExpansionType::Hybrid => self.expand_hybrid(inputs, &mut expanded)?,
        }
        
        Ok(expanded)
    }
    
    /// Calculate output dimension based on expansion configuration
    fn calculate_output_dim(config: &ELMConfig) -> usize {
        if config.per_feature_expansion {
            match config.expansion_type {
                ExpansionType::Trigonometric => {
                    // Original + sin/cos pairs for each order
                    config.input_dim * (1 + 2 * config.expansion_order)
                }
                ExpansionType::Polynomial => {
                    // Original + powers for each order
                    config.input_dim * (1 + config.expansion_order)
                }
                ExpansionType::Chebyshev | ExpansionType::Hermite => {
                    // Original + polynomial terms for each order
                    config.input_dim * (1 + config.expansion_order)
                }
                ExpansionType::Hybrid => {
                    // Combination of all types
                    config.input_dim * (1 + 3 * config.expansion_order)
                }
            }
        } else {
            match config.expansion_type {
                ExpansionType::Trigonometric => {
                    config.input_dim + 2 * config.expansion_order
                }
                ExpansionType::Polynomial => {
                    config.input_dim + config.expansion_order
                }
                ExpansionType::Chebyshev | ExpansionType::Hermite => {
                    config.input_dim + config.expansion_order
                }
                ExpansionType::Hybrid => {
                    config.input_dim + 3 * config.expansion_order
                }
            }
        }
    }
    
    /// Generate random weights for functional expansion
    fn generate_expansion_weights(config: &ELMConfig) -> Result<DMatrix<f64>> {
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(config.seed);
        
        let weights = DMatrix::from_fn(config.expansion_order, config.input_dim, |_, _| {
            rng.gen_range(-1.0..1.0) * config.activation_scale
        });
        
        Ok(weights)
    }
    
    /// Precompute polynomial coefficients for efficiency
    fn precompute_polynomial_coeffs(max_order: usize) -> Result<Vec<Vec<f64>>> {
        let mut coeffs = Vec::with_capacity(max_order + 1);
        
        // Start with constant term
        coeffs.push(vec![1.0]);
        
        if max_order == 0 {
            return Ok(coeffs);
        }
        
        // Linear term
        coeffs.push(vec![0.0, 1.0]);
        
        // Higher order terms using recurrence relations
        for n in 2..=max_order {
            let mut new_coeffs = vec![0.0; n + 1];
            
            // Chebyshev recurrence: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
            // For now, we'll use simple polynomial coefficients
            // In production, implement proper Chebyshev/Hermite recurrence
            for i in 0..=n {
                if i < n {
                    new_coeffs[i + 1] += 2.0; // x * previous term
                }
                if i >= 2 {
                    new_coeffs[i] -= 1.0; // subtract two terms back
                }
            }
            
            coeffs.push(new_coeffs);
        }
        
        Ok(coeffs)
    }
    
    /// Trigonometric expansion implementation
    fn expand_trigonometric(&self, inputs: &DMatrix<f64>, expanded: &mut DMatrix<f64>) -> Result<()> {
        let n_samples = inputs.nrows();
        
        if self.config.per_feature_expansion {
            // Per-feature trigonometric expansion
            let mut col_idx = 0;
            
            for feat_idx in 0..self.config.input_dim {
                // Copy original feature
                for sample_idx in 0..n_samples {
                    expanded[(sample_idx, col_idx)] = inputs[(sample_idx, feat_idx)];
                }
                col_idx += 1;
                
                // Add trigonometric terms
                for order in 1..=self.config.expansion_order {
                    self.expand_trigonometric_feature(
                        inputs, feat_idx, order, expanded, col_idx, col_idx + 1
                    )?;
                    col_idx += 2; // sin and cos
                }
            }
        } else {
            // Global trigonometric expansion with random weights
            let weights = self.expansion_weights.as_ref()
                .ok_or_else(|| anyhow!("Expansion weights not initialized"))?;
            
            // Copy original features
            for sample_idx in 0..n_samples {
                for feat_idx in 0..self.config.input_dim {
                    expanded[(sample_idx, feat_idx)] = inputs[(sample_idx, feat_idx)];
                }
            }
            
            // Add global trigonometric terms
            let mut col_idx = self.config.input_dim;
            for order in 0..self.config.expansion_order {
                for sample_idx in 0..n_samples {
                    // Compute weighted sum for this order
                    let mut weighted_sum = 0.0;
                    for feat_idx in 0..self.config.input_dim {
                        weighted_sum += inputs[(sample_idx, feat_idx)] * weights[(order, feat_idx)];
                    }
                    
                    // Apply trigonometric functions
                    let arg = weighted_sum * self.config.activation_scale;
                    expanded[(sample_idx, col_idx)] = arg.sin();
                    expanded[(sample_idx, col_idx + 1)] = arg.cos();
                }
                col_idx += 2;
            }
        }
        
        Ok(())
    }
    
    /// Expand single feature with trigonometric functions (SIMD optimized)
    #[cfg(feature = "simd")]
    fn expand_trigonometric_feature(
        &self,
        inputs: &DMatrix<f64>,
        feat_idx: usize,
        order: usize,
        expanded: &mut DMatrix<f64>,
        sin_col: usize,
        cos_col: usize,
    ) -> Result<()> {
        let n_samples = inputs.nrows();
        let frequency = order as f64 * self.config.activation_scale;
        
        // SIMD processing in chunks of 4
        let chunks = n_samples / 4;
        let remainder = n_samples % 4;
        
        for chunk in 0..chunks {
            let base_idx = chunk * 4;
            
            // Load 4 values
            let x = f64x4::new([
                inputs[(base_idx, feat_idx)],
                inputs[(base_idx + 1, feat_idx)],
                inputs[(base_idx + 2, feat_idx)],
                inputs[(base_idx + 3, feat_idx)],
            ]);
            
            // Apply frequency scaling
            let scaled = x * f64x4::splat(frequency);
            
            // Compute sin and cos (approximation for SIMD)
            let sin_vals = self.simd_sin(scaled);
            let cos_vals = self.simd_cos(scaled);
            
            // Store results
            for i in 0..4 {
                expanded[(base_idx + i, sin_col)] = sin_vals.as_array_ref()[i];
                expanded[(base_idx + i, cos_col)] = cos_vals.as_array_ref()[i];
            }
        }
        
        // Handle remainder
        for i in chunks * 4..n_samples {
            let x = inputs[(i, feat_idx)] * frequency;
            expanded[(i, sin_col)] = x.sin();
            expanded[(i, cos_col)] = x.cos();
        }
        
        Ok(())
    }
    
    /// Non-SIMD fallback for trigonometric expansion
    #[cfg(not(feature = "simd"))]
    fn expand_trigonometric_feature(
        &self,
        inputs: &DMatrix<f64>,
        feat_idx: usize,
        order: usize,
        expanded: &mut DMatrix<f64>,
        sin_col: usize,
        cos_col: usize,
    ) -> Result<()> {
        let n_samples = inputs.nrows();
        let frequency = order as f64 * self.config.activation_scale;
        
        for sample_idx in 0..n_samples {
            let x = inputs[(sample_idx, feat_idx)] * frequency;
            expanded[(sample_idx, sin_col)] = x.sin();
            expanded[(sample_idx, cos_col)] = x.cos();
        }
        
        Ok(())
    }
    
    /// Polynomial expansion implementation
    fn expand_polynomial(&self, inputs: &DMatrix<f64>, expanded: &mut DMatrix<f64>) -> Result<()> {
        let n_samples = inputs.nrows();
        
        if self.config.per_feature_expansion {
            let mut col_idx = 0;
            
            for feat_idx in 0..self.config.input_dim {
                // Copy original feature
                for sample_idx in 0..n_samples {
                    expanded[(sample_idx, col_idx)] = inputs[(sample_idx, feat_idx)];
                }
                col_idx += 1;
                
                // Add polynomial powers
                for order in 1..=self.config.expansion_order {
                    for sample_idx in 0..n_samples {
                        let base_val = inputs[(sample_idx, feat_idx)];
                        expanded[(sample_idx, col_idx)] = base_val.powi(order as i32 + 1);
                    }
                    col_idx += 1;
                }
            }
        } else {
            // Global polynomial expansion
            let weights = self.expansion_weights.as_ref()
                .ok_or_else(|| anyhow!("Expansion weights not initialized"))?;
            
            // Copy original features
            for sample_idx in 0..n_samples {
                for feat_idx in 0..self.config.input_dim {
                    expanded[(sample_idx, feat_idx)] = inputs[(sample_idx, feat_idx)];
                }
            }
            
            // Add polynomial terms
            let mut col_idx = self.config.input_dim;
            for order in 0..self.config.expansion_order {
                for sample_idx in 0..n_samples {
                    let mut weighted_sum = 0.0;
                    for feat_idx in 0..self.config.input_dim {
                        weighted_sum += inputs[(sample_idx, feat_idx)] * weights[(order, feat_idx)];
                    }
                    
                    expanded[(sample_idx, col_idx)] = weighted_sum.powi(order as i32 + 2);
                }
                col_idx += 1;
            }
        }
        
        Ok(())
    }
    
    /// Chebyshev polynomial expansion
    fn expand_chebyshev(&self, inputs: &DMatrix<f64>, expanded: &mut DMatrix<f64>) -> Result<()> {
        let coeffs = self.polynomial_coeffs.as_ref()
            .ok_or_else(|| anyhow!("Polynomial coefficients not precomputed"))?;
        
        let n_samples = inputs.nrows();
        
        if self.config.per_feature_expansion {
            let mut col_idx = 0;
            
            for feat_idx in 0..self.config.input_dim {
                // Copy original feature
                for sample_idx in 0..n_samples {
                    expanded[(sample_idx, col_idx)] = inputs[(sample_idx, feat_idx)];
                }
                col_idx += 1;
                
                // Add Chebyshev polynomial terms
                for order in 1..=self.config.expansion_order {
                    for sample_idx in 0..n_samples {
                        let x = inputs[(sample_idx, feat_idx)];
                        expanded[(sample_idx, col_idx)] = self.evaluate_chebyshev(x, order, coeffs);
                    }
                    col_idx += 1;
                }
            }
        } else {
            warn!("Global Chebyshev expansion not yet implemented, falling back to per-feature");
            self.expand_chebyshev(inputs, expanded)?;
        }
        
        Ok(())
    }
    
    /// Hermite polynomial expansion
    fn expand_hermite(&self, inputs: &DMatrix<f64>, expanded: &mut DMatrix<f64>) -> Result<()> {
        // Similar to Chebyshev but with Hermite polynomials
        // For now, use simplified implementation
        self.expand_polynomial(inputs, expanded)
    }
    
    /// Hybrid expansion combining multiple types
    fn expand_hybrid(&self, inputs: &DMatrix<f64>, expanded: &mut DMatrix<f64>) -> Result<()> {
        let n_samples = inputs.nrows();
        let mut col_idx = 0;
        
        // Original features
        for feat_idx in 0..self.config.input_dim {
            for sample_idx in 0..n_samples {
                expanded[(sample_idx, col_idx)] = inputs[(sample_idx, feat_idx)];
            }
            col_idx += 1;
        }
        
        // Trigonometric terms
        for feat_idx in 0..self.config.input_dim {
            for order in 1..=self.config.expansion_order {
                let frequency = order as f64 * self.config.activation_scale;
                for sample_idx in 0..n_samples {
                    let x = inputs[(sample_idx, feat_idx)] * frequency;
                    expanded[(sample_idx, col_idx)] = x.sin();
                    expanded[(sample_idx, col_idx + 1)] = x.cos();
                }
                col_idx += 2;
            }
        }
        
        // Polynomial terms
        for feat_idx in 0..self.config.input_dim {
            for order in 1..=self.config.expansion_order {
                for sample_idx in 0..n_samples {
                    let x = inputs[(sample_idx, feat_idx)];
                    expanded[(sample_idx, col_idx)] = x.powi(order as i32 + 1);
                }
                col_idx += 1;
            }
        }
        
        Ok(())
    }
    
    /// Evaluate Chebyshev polynomial using precomputed coefficients
    fn evaluate_chebyshev(&self, x: f64, order: usize, coeffs: &[Vec<f64>]) -> f64 {
        if order >= coeffs.len() {
            return 0.0;
        }
        
        let poly_coeffs = &coeffs[order];
        let mut result = 0.0;
        let mut x_power = 1.0;
        
        for &coeff in poly_coeffs {
            result += coeff * x_power;
            x_power *= x;
        }
        
        result
    }
    
    /// SIMD sine approximation
    #[cfg(feature = "simd")]
    fn simd_sin(&self, x: f64x4) -> f64x4 {
        // Taylor series approximation for sin(x)
        // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        
        x - x3 * f64x4::splat(1.0/6.0) 
          + x5 * f64x4::splat(1.0/120.0) 
          - x7 * f64x4::splat(1.0/5040.0)
    }
    
    /// SIMD cosine approximation
    #[cfg(feature = "simd")]
    fn simd_cos(&self, x: f64x4) -> f64x4 {
        // Taylor series approximation for cos(x)
        // cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        
        f64x4::splat(1.0) - x2 * f64x4::splat(0.5) 
                          + x4 * f64x4::splat(1.0/24.0) 
                          - x6 * f64x4::splat(1.0/720.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_output_dim_calculation() {
        let mut config = ELMConfig::default();
        config.input_dim = 4;
        config.expansion_order = 3;
        config.per_feature_expansion = true;
        
        // Trigonometric: 4 * (1 + 2*3) = 28
        config.expansion_type = ExpansionType::Trigonometric;
        assert_eq!(FunctionalExpansion::calculate_output_dim(&config), 28);
        
        // Polynomial: 4 * (1 + 3) = 16
        config.expansion_type = ExpansionType::Polynomial;
        assert_eq!(FunctionalExpansion::calculate_output_dim(&config), 16);
    }
    
    #[test]
    fn test_trigonometric_expansion() {
        let mut config = ELMConfig::default();
        config.input_dim = 2;
        config.expansion_order = 2;
        config.per_feature_expansion = true;
        config.expansion_type = ExpansionType::Trigonometric;
        
        let expansion = FunctionalExpansion::new(&config).unwrap();
        
        let inputs = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let expanded = expansion.expand(&inputs).unwrap();
        
        // Should have 2 * (1 + 2*2) = 10 features
        assert_eq!(expanded.ncols(), 10);
        assert_eq!(expanded.nrows(), 2);
        
        // First feature should be copied
        assert_relative_eq!(expanded[(0, 0)], 1.0);
        assert_relative_eq!(expanded[(1, 0)], 3.0);
    }
    
    #[test]
    fn test_polynomial_expansion() {
        let mut config = ELMConfig::default();
        config.input_dim = 2;
        config.expansion_order = 2;
        config.per_feature_expansion = true;
        config.expansion_type = ExpansionType::Polynomial;
        
        let expansion = FunctionalExpansion::new(&config).unwrap();
        
        let inputs = DMatrix::from_row_slice(1, 2, &[2.0, 3.0]);
        let expanded = expansion.expand(&inputs).unwrap();
        
        // Should have 2 * (1 + 2) = 6 features
        assert_eq!(expanded.ncols(), 6);
        
        // Check polynomial terms: x, x^2, x^3
        assert_relative_eq!(expanded[(0, 0)], 2.0);    // x1
        assert_relative_eq!(expanded[(0, 1)], 8.0);    // x1^2
        assert_relative_eq!(expanded[(0, 2)], 32.0);   // x1^3
        assert_relative_eq!(expanded[(0, 3)], 3.0);    // x2
        assert_relative_eq!(expanded[(0, 4)], 27.0);   // x2^2
        assert_relative_eq!(expanded[(0, 5)], 243.0);  // x2^3
    }
}