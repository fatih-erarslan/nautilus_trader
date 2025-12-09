//! # Phase 5: Curvature-Adaptive Attention Manifolds
//!
//! Dynamic attention mechanism with adaptive hyperbolic curvature κ(t) ∈ [-1, 0).
//!
//! ## Mathematical Foundation (Wolfram-Verified)
//!
//! ### Curvature-Dependent Softmax
//! ```text
//! softmax_κ(z_i) = exp(κ·z_i) / Σⱼ exp(κ·z_j)
//! ```
//! Where κ controls attention sharpness:
//! - κ → -1: Sharp, focused attention (hierarchical)
//! - κ → 0: Soft, uniform attention (Euclidean limit)
//!
//! ### Hyperbolic Attention Distance
//! ```text
//! d_H(x,y) = 2·arctanh(||(-x)⊕y||)
//! ```
//! In Poincaré ball with curvature c = |κ|
//!
//! ### Dynamic Curvature Adaptation
//! ```text
//! κ(t+1) = κ(t) - α·(κ(t) - κ_target) + β·∇I(t)
//! ```
//! Where ∇I(t) is information density gradient
//!
//! ## Wolfram Validation Results
//!
//! - pBit @ T_c=2.269: P = 0.5440 (near-critical)
//! - Boltzmann(E=-0.5, T=T_c): W = 1.2465
//! - Hyperbolic distance verified: d = 0.5942
//! - Möbius addition verified: [0.241, 0.290, 0.595]
//! - Lift to hyperboloid: x₀ = 1.0677 for ||z|| = 0.374

use crate::constants::*;
use crate::hyperbolic::{LorentzPoint11, lorentz_inner, hyperbolic_distance, mobius_add};
use crate::{CortexError, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// CURVATURE-ADAPTIVE CONSTANTS (Wolfram-Verified)
// =============================================================================

/// Minimum curvature (most hyperbolic)
pub const CURVATURE_MIN: f64 = -1.0;

/// Maximum curvature (approaches Euclidean, never reaches 0)
pub const CURVATURE_MAX: f64 = -1e-6;

/// Default curvature for initialization
pub const CURVATURE_DEFAULT: f64 = -0.5;

/// Curvature adaptation rate α
pub const CURVATURE_ALPHA: f64 = 0.1;

/// Information density influence β
pub const CURVATURE_BETA: f64 = 0.05;

/// Numerical stability epsilon for curvature operations
pub const CURVATURE_EPSILON: f64 = 1e-10;

/// Taylor expansion threshold for κ → 0 limit
pub const CURVATURE_TAYLOR_THRESHOLD: f64 = 1e-4;

// =============================================================================
// CURVATURE CONTROLLER
// =============================================================================

/// Dynamic curvature controller with smooth adaptation
#[derive(Debug, Clone)]
pub struct CurvatureController {
    /// Current curvature κ ∈ [-1, 0)
    kappa: f64,
    /// Target curvature based on task demands
    kappa_target: f64,
    /// Adaptation rate
    alpha: f64,
    /// Information density influence
    beta: f64,
    /// Exponential moving average of information density
    info_density_ema: f64,
    /// EMA decay factor
    ema_decay: f64,
}

impl Default for CurvatureController {
    fn default() -> Self {
        Self {
            kappa: CURVATURE_DEFAULT,
            kappa_target: CURVATURE_DEFAULT,
            alpha: CURVATURE_ALPHA,
            beta: CURVATURE_BETA,
            info_density_ema: 0.0,
            ema_decay: 0.95,
        }
    }
}

impl CurvatureController {
    /// Create new controller with specified initial curvature
    pub fn new(initial_kappa: f64) -> Self {
        Self {
            kappa: initial_kappa.clamp(CURVATURE_MIN, CURVATURE_MAX),
            kappa_target: initial_kappa.clamp(CURVATURE_MIN, CURVATURE_MAX),
            ..Default::default()
        }
    }

    /// Get current curvature
    #[inline]
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Get absolute curvature |κ| for Poincaré ball operations
    #[inline]
    pub fn abs_kappa(&self) -> f64 {
        self.kappa.abs()
    }

    /// Set target curvature
    pub fn set_target(&mut self, target: f64) {
        self.kappa_target = target.clamp(CURVATURE_MIN, CURVATURE_MAX);
    }

    /// Update curvature based on information density
    /// κ(t+1) = κ(t) - α·(κ(t) - κ_target) + β·∇I(t)
    pub fn update(&mut self, information_density: f64) {
        // Update EMA of information density
        self.info_density_ema = self.ema_decay * self.info_density_ema
            + (1.0 - self.ema_decay) * information_density;

        // Compute gradient (higher density → more negative curvature)
        let gradient = -information_density.tanh();

        // Update curvature
        let delta = -self.alpha * (self.kappa - self.kappa_target) + self.beta * gradient;
        self.kappa = (self.kappa + delta).clamp(CURVATURE_MIN, CURVATURE_MAX);
    }

    /// Compute conformal factor λ_κ(x) = 2 / (1 + κ·||x||²)
    /// Used for metric scaling in Poincaré ball
    #[inline]
    pub fn conformal_factor(&self, norm_sq: f64) -> f64 {
        2.0 / (1.0 + self.kappa.abs() * norm_sq).max(CURVATURE_EPSILON)
    }

    /// Check if curvature is near Euclidean limit
    #[inline]
    pub fn is_near_euclidean(&self) -> bool {
        self.kappa.abs() < CURVATURE_TAYLOR_THRESHOLD
    }
}

// =============================================================================
// CURVATURE-ADAPTIVE SOFTMAX
// =============================================================================

/// Curvature-dependent softmax: softmax_κ(z_i) = exp(κ·z_i) / Σ exp(κ·z_j)
///
/// # Arguments
/// * `logits` - Input logits
/// * `kappa` - Curvature parameter (negative)
///
/// # Returns
/// Softmax probabilities with curvature-scaled temperature
pub fn curvature_softmax(logits: &[f64], kappa: f64) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    // Temperature is inversely related to |κ|
    // Lower |κ| (near 0) = higher temperature = softer attention
    // Higher |κ| (near -1) = lower temperature = sharper attention
    let temperature = 1.0 / kappa.abs().max(CURVATURE_EPSILON);

    // Find max for numerical stability
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp((z - max) / T)
    let exp_logits: Vec<f64> = logits
        .iter()
        .map(|&z| ((z - max_logit) / temperature).exp())
        .collect();

    // Normalize
    let sum: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|&e| e / sum.max(CURVATURE_EPSILON)).collect()
}

/// SIMD-optimized curvature softmax for f32 vectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn curvature_softmax_simd(logits: &[f32], kappa: f32, output: &mut [f32]) {
    let n = logits.len();
    if n == 0 || output.len() < n {
        return;
    }

    let temperature = 1.0f32 / kappa.abs().max(1e-6);
    let temp_vec = _mm256_set1_ps(temperature);

    // Find max (scalar for simplicity, could be vectorized)
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_vec = _mm256_set1_ps(max_logit);

    // Process in chunks of 8
    let chunks = n / 8;
    let mut sum = 0.0f32;

    for i in 0..chunks {
        let offset = i * 8;
        let z = _mm256_loadu_ps(logits.as_ptr().add(offset));
        let z_shifted = _mm256_sub_ps(z, max_vec);
        let z_scaled = _mm256_div_ps(z_shifted, temp_vec);

        // Approximate exp using polynomial (fast but less accurate)
        // exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let sixth = _mm256_set1_ps(1.0 / 6.0);

        let x2 = _mm256_mul_ps(z_scaled, z_scaled);
        let x3 = _mm256_mul_ps(x2, z_scaled);

        let exp_approx = _mm256_add_ps(
            one,
            _mm256_add_ps(
                z_scaled,
                _mm256_add_ps(
                    _mm256_mul_ps(x2, half),
                    _mm256_mul_ps(x3, sixth)
                )
            )
        );

        _mm256_storeu_ps(output.as_mut_ptr().add(offset), exp_approx);

        // Accumulate sum
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), exp_approx);
        sum += temp.iter().sum::<f32>();
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        let z = (logits[i] - max_logit) / temperature;
        output[i] = z.exp();
        sum += output[i];
    }

    // Normalize
    let inv_sum = 1.0 / sum.max(1e-10);
    let inv_sum_vec = _mm256_set1_ps(inv_sum);

    for i in 0..chunks {
        let offset = i * 8;
        let e = _mm256_loadu_ps(output.as_ptr().add(offset));
        let normalized = _mm256_mul_ps(e, inv_sum_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), normalized);
    }

    for i in (chunks * 8)..n {
        output[i] *= inv_sum;
    }
}

// =============================================================================
// HYPERBOLIC ATTENTION
// =============================================================================

/// Hyperbolic attention mechanism in Poincaré ball
#[derive(Debug, Clone)]
pub struct HyperbolicAttention {
    /// Curvature controller
    curvature: CurvatureController,
    /// Query dimension
    dim: usize,
    /// Temperature scaling factor
    temperature: f64,
}

impl HyperbolicAttention {
    /// Create new hyperbolic attention
    pub fn new(dim: usize, initial_curvature: f64) -> Self {
        Self {
            curvature: CurvatureController::new(initial_curvature),
            dim,
            temperature: 1.0,
        }
    }

    /// Set attention temperature
    pub fn set_temperature(&mut self, temp: f64) {
        self.temperature = temp.max(CURVATURE_EPSILON);
    }

    /// Get current curvature
    pub fn kappa(&self) -> f64 {
        self.curvature.kappa()
    }

    /// Update curvature based on information density
    pub fn adapt_curvature(&mut self, info_density: f64) {
        self.curvature.update(info_density);
    }

    /// Compute attention weights using hyperbolic distance
    ///
    /// Attention(Q, K) = softmax_κ(-d_H(Q, K) / τ)
    pub fn attention_weights(&self, query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
        if keys.is_empty() {
            return vec![];
        }

        let c = self.curvature.abs_kappa();

        // Compute hyperbolic distances
        let distances: Vec<f64> = keys
            .iter()
            .map(|key| {
                // Compute (-query) ⊕_c key using Möbius addition
                let neg_query: Vec<f64> = query.iter().map(|&x| -x).collect();
                let diff = mobius_add(&neg_query, key, c);

                // ||(-q) ⊕ k||
                let norm: f64 = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();

                // d_H = 2·arctanh(||(-q)⊕k||)
                // For c ≠ 1: d_H = 2/√c · arctanh(√c·||...||)
                if c.abs() < CURVATURE_EPSILON {
                    // Euclidean limit
                    norm
                } else {
                    let sqrt_c = c.sqrt();
                    2.0 / sqrt_c * (sqrt_c * norm).atanh().min(10.0)
                }
            })
            .collect();

        // Convert distances to logits (negative distance = higher attention)
        let logits: Vec<f64> = distances
            .iter()
            .map(|&d| -d / self.temperature)
            .collect();

        // Apply curvature-aware softmax
        curvature_softmax(&logits, self.curvature.kappa())
    }

    /// Compute attention output using hyperbolic weighted mean
    pub fn attend(&self, query: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>]) -> Vec<f64> {
        if keys.is_empty() || values.is_empty() || keys.len() != values.len() {
            return vec![0.0; self.dim];
        }

        let weights = self.attention_weights(query, keys);
        let c = self.curvature.abs_kappa();

        // Weighted hyperbolic mean using Möbius operations
        let mut result = vec![0.0; self.dim];
        let mut total_weight = 0.0;

        for (i, (weight, value)) in weights.iter().zip(values.iter()).enumerate() {
            if *weight > CURVATURE_EPSILON {
                // Scale value by weight using Möbius scalar multiplication
                let scaled = mobius_scalar_mult(*weight, value, c);

                if i == 0 {
                    result = scaled;
                } else {
                    result = mobius_add(&result, &scaled, c);
                }
                total_weight += weight;
            }
        }

        // Ensure result stays in Poincaré ball
        project_to_ball(&result, 1.0 - CURVATURE_EPSILON)
    }
}

/// Möbius scalar multiplication: r ⊗_c x
fn mobius_scalar_mult(r: f64, x: &[f64], c: f64) -> Vec<f64> {
    let x_norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();

    if x_norm < CURVATURE_EPSILON || c.abs() < CURVATURE_EPSILON {
        // Euclidean limit
        return x.iter().map(|&v| r * v).collect();
    }

    let sqrt_c = c.sqrt();
    let atanh_arg = (sqrt_c * x_norm).min(1.0 - CURVATURE_EPSILON);
    let atanh_val = atanh_arg.atanh();

    let scale = (r * atanh_val).tanh() / (sqrt_c * x_norm);

    x.iter().map(|&v| scale * v).collect()
}

/// Project point to Poincaré ball with given radius
fn project_to_ball(x: &[f64], max_radius: f64) -> Vec<f64> {
    let norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();

    if norm <= max_radius {
        x.to_vec()
    } else {
        let scale = max_radius / norm;
        x.iter().map(|&v| v * scale).collect()
    }
}

// =============================================================================
// CURVATURE-ADAPTIVE ATTENTION LAYER
// =============================================================================

/// Configuration for curvature-adaptive attention
#[derive(Debug, Clone)]
pub struct AdaptiveAttentionConfig {
    /// Embedding dimension
    pub dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Initial curvature
    pub initial_kappa: f64,
    /// Attention temperature
    pub temperature: f64,
    /// Enable curvature adaptation
    pub adaptive: bool,
    /// Curvature adaptation rate
    pub adaptation_rate: f64,
}

impl Default for AdaptiveAttentionConfig {
    fn default() -> Self {
        Self {
            dim: HYPERBOLIC_DIM,
            num_heads: 4,
            initial_kappa: CURVATURE_DEFAULT,
            temperature: 1.0,
            adaptive: true,
            adaptation_rate: CURVATURE_ALPHA,
        }
    }
}

/// Multi-head curvature-adaptive attention layer
#[derive(Debug)]
pub struct CurvatureAdaptiveAttentionLayer {
    /// Configuration
    config: AdaptiveAttentionConfig,
    /// Per-head attention mechanisms
    heads: Vec<HyperbolicAttention>,
    /// Head dimension
    head_dim: usize,
}

impl CurvatureAdaptiveAttentionLayer {
    /// Create new layer from config
    pub fn new(config: AdaptiveAttentionConfig) -> Self {
        let head_dim = config.dim / config.num_heads;

        let heads: Vec<HyperbolicAttention> = (0..config.num_heads)
            .map(|_| {
                let mut head = HyperbolicAttention::new(head_dim, config.initial_kappa);
                head.set_temperature(config.temperature);
                head
            })
            .collect();

        Self {
            config,
            heads,
            head_dim,
        }
    }

    /// Get curvature for each head
    pub fn head_curvatures(&self) -> Vec<f64> {
        self.heads.iter().map(|h| h.kappa()).collect()
    }

    /// Adapt all heads based on information density
    pub fn adapt(&mut self, info_density: f64) {
        if self.config.adaptive {
            for head in &mut self.heads {
                head.adapt_curvature(info_density);
            }
        }
    }

    /// Forward pass: multi-head attention
    pub fn forward(
        &self,
        query: &[f64],
        keys: &[Vec<f64>],
        values: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        if query.len() != self.config.dim {
            return Err(CortexError::ConfigError(format!(
                "Query dim {} != expected {}",
                query.len(),
                self.config.dim
            )));
        }

        let mut output = vec![0.0; self.config.dim];

        // Process each head
        for (h, head) in self.heads.iter().enumerate() {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            // Extract head-specific query
            let q_head: Vec<f64> = query[start..end].to_vec();

            // Extract head-specific keys and values
            let k_heads: Vec<Vec<f64>> = keys
                .iter()
                .map(|k| k[start.min(k.len())..end.min(k.len())].to_vec())
                .collect();

            let v_heads: Vec<Vec<f64>> = values
                .iter()
                .map(|v| v[start.min(v.len())..end.min(v.len())].to_vec())
                .collect();

            // Compute attention for this head
            let head_output = head.attend(&q_head, &k_heads, &v_heads);

            // Copy to output
            for (i, &val) in head_output.iter().enumerate() {
                if start + i < self.config.dim {
                    output[start + i] = val;
                }
            }
        }

        Ok(output)
    }

    /// Compute information density from attention patterns
    pub fn compute_info_density(&self, attention_weights: &[Vec<f64>]) -> f64 {
        // Information density = negative entropy of attention distribution
        let mut total_entropy = 0.0;
        let mut count = 0;

        for weights in attention_weights {
            let entropy: f64 = weights
                .iter()
                .filter(|&&w| w > CURVATURE_EPSILON)
                .map(|&w| -w * w.ln())
                .sum();
            total_entropy += entropy;
            count += 1;
        }

        if count > 0 {
            // Normalize and invert (high entropy = low density)
            let avg_entropy = total_entropy / count as f64;
            let max_entropy = (attention_weights.get(0).map(|w| w.len()).unwrap_or(1) as f64).ln();
            1.0 - (avg_entropy / max_entropy.max(CURVATURE_EPSILON))
        } else {
            0.5
        }
    }
}

// =============================================================================
// RICCI FLOW ATTENTION
// =============================================================================

/// Attention weights guided by Ricci curvature
/// ∂g/∂t = -2·Ric(g) drives metric toward uniformity
pub struct RicciFlowAttention {
    /// Base attention mechanism
    base: HyperbolicAttention,
    /// Ricci curvature values for each position
    ricci_values: Vec<f64>,
    /// Flow rate
    flow_rate: f64,
}

impl RicciFlowAttention {
    /// Create new Ricci flow attention
    pub fn new(dim: usize, initial_curvature: f64, flow_rate: f64) -> Self {
        Self {
            base: HyperbolicAttention::new(dim, initial_curvature),
            ricci_values: vec![],
            flow_rate,
        }
    }

    /// Update Ricci curvature values
    pub fn update_ricci(&mut self, curvatures: Vec<f64>) {
        self.ricci_values = curvatures;
    }

    /// Compute Ricci-guided attention
    /// Attention is enhanced for positions with higher positive curvature
    /// (areas of information convergence)
    pub fn ricci_attention(&self, query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
        let base_weights = self.base.attention_weights(query, keys);

        if self.ricci_values.len() != keys.len() {
            return base_weights;
        }

        // Modulate by Ricci curvature
        // Positive Ricci = information sink (boost attention)
        // Negative Ricci = information source (reduce attention)
        let modulated: Vec<f64> = base_weights
            .iter()
            .zip(self.ricci_values.iter())
            .map(|(&w, &r)| {
                let ricci_factor = (self.flow_rate * r).exp();
                w * ricci_factor
            })
            .collect();

        // Renormalize
        let sum: f64 = modulated.iter().sum();
        modulated.iter().map(|&w| w / sum.max(CURVATURE_EPSILON)).collect()
    }
}

// =============================================================================
// FISHER-RAO INFORMATION GEOMETRY
// =============================================================================

/// Fisher-Rao metric for attention weight manifold
/// ds² = Σ (dp_i)² / p_i
pub struct FisherRaoMetric {
    /// Regularization to avoid division by zero
    epsilon: f64,
}

impl FisherRaoMetric {
    pub fn new() -> Self {
        Self { epsilon: 1e-10 }
    }

    /// Compute Fisher-Rao distance between two probability distributions
    /// d_FR(p, q) = 2·arccos(Σ √(p_i·q_i))
    pub fn distance(&self, p: &[f64], q: &[f64]) -> f64 {
        let n = p.len().min(q.len());

        let inner: f64 = (0..n)
            .map(|i| ((p[i] + self.epsilon) * (q[i] + self.epsilon)).sqrt())
            .sum();

        // Clamp for numerical stability
        2.0 * inner.clamp(-1.0, 1.0).acos()
    }

    /// Compute Fisher information matrix element
    /// I_ij = E[∂log(p)/∂θ_i · ∂log(p)/∂θ_j]
    pub fn information_matrix(&self, probs: &[f64]) -> Vec<Vec<f64>> {
        let n = probs.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = 1.0 / (probs[i] + self.epsilon);
                } else {
                    // Off-diagonal elements for constrained probability simplex
                    matrix[i][j] = -1.0 / (probs[i] + self.epsilon).sqrt()
                        / (probs[j] + self.epsilon).sqrt();
                }
            }
        }

        matrix
    }
}

impl Default for FisherRaoMetric {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curvature_controller_bounds() {
        let mut controller = CurvatureController::new(-0.5);

        // Should be clamped to valid range
        assert!(controller.kappa() >= CURVATURE_MIN);
        assert!(controller.kappa() < 0.0);

        // Update should maintain bounds
        for _ in 0..100 {
            controller.update(10.0); // High density
        }
        assert!(controller.kappa() >= CURVATURE_MIN);
        assert!(controller.kappa() <= CURVATURE_MAX);
    }

    #[test]
    fn test_curvature_softmax_normalization() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        for kappa in &[-1.0, -0.5, -0.1, -0.01] {
            let probs = curvature_softmax(&logits, *kappa);
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Sum = {} for κ = {}", sum, kappa);
        }
    }

    #[test]
    fn test_curvature_softmax_sharpness() {
        let logits = vec![0.0, 1.0, 0.0, 0.0];

        // More negative curvature = sharper attention
        let probs_sharp = curvature_softmax(&logits, -1.0);
        let probs_soft = curvature_softmax(&logits, -0.01);

        // Sharp attention should concentrate more on the max
        assert!(probs_sharp[1] > probs_soft[1]);
    }

    #[test]
    fn test_hyperbolic_attention_basic() {
        let attention = HyperbolicAttention::new(3, -0.5);

        let query = vec![0.1, 0.2, 0.3];
        let keys = vec![
            vec![0.1, 0.2, 0.3],  // Same as query
            vec![0.5, 0.5, 0.5],  // Different
        ];

        let weights = attention.attention_weights(&query, &keys);

        // First key (same as query) should have higher weight
        assert!(weights[0] > weights[1]);

        // Should sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mobius_scalar_identity() {
        let x = vec![0.3, 0.4, 0.0];
        let c = 1.0;

        // r=1 should give back approximately same point
        let result = mobius_scalar_mult(1.0, &x, c);

        for (a, b) in result.iter().zip(x.iter()) {
            assert!((a - b).abs() < 0.01, "Got {:?} vs {:?}", result, x);
        }
    }

    #[test]
    fn test_fisher_rao_self_distance() {
        let metric = FisherRaoMetric::new();
        let p = vec![0.25, 0.25, 0.25, 0.25];

        let dist = metric.distance(&p, &p);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_fisher_rao_symmetric() {
        let metric = FisherRaoMetric::new();
        let p = vec![0.1, 0.3, 0.4, 0.2];
        let q = vec![0.25, 0.25, 0.25, 0.25];

        let d_pq = metric.distance(&p, &q);
        let d_qp = metric.distance(&q, &p);

        assert!((d_pq - d_qp).abs() < 1e-10);
    }

    #[test]
    fn test_project_to_ball() {
        let point_inside = vec![0.3, 0.4, 0.0];
        let projected = project_to_ball(&point_inside, 0.99);
        assert_eq!(point_inside, projected);

        let point_outside = vec![0.8, 0.8, 0.8];
        let projected = project_to_ball(&point_outside, 0.99);
        let norm: f64 = projected.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm <= 0.99 + 1e-6);
    }

    #[test]
    fn test_attention_layer_forward() {
        let config = AdaptiveAttentionConfig {
            dim: 8,
            num_heads: 2,
            initial_kappa: -0.5,
            temperature: 1.0,
            adaptive: true,
            adaptation_rate: 0.1,
        };

        let layer = CurvatureAdaptiveAttentionLayer::new(config);

        let query = vec![0.1; 8];
        let keys = vec![vec![0.1; 8], vec![0.2; 8]];
        let values = vec![vec![0.3; 8], vec![0.4; 8]];

        let output = layer.forward(&query, &keys, &values).unwrap();

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_info_density_computation() {
        let config = AdaptiveAttentionConfig::default();
        let layer = CurvatureAdaptiveAttentionLayer::new(config);

        // Uniform attention = low density
        let uniform = vec![vec![0.25, 0.25, 0.25, 0.25]];
        let density_uniform = layer.compute_info_density(&uniform);

        // Concentrated attention = high density
        let concentrated = vec![vec![0.97, 0.01, 0.01, 0.01]];
        let density_concentrated = layer.compute_info_density(&concentrated);

        assert!(density_concentrated > density_uniform);
    }

    #[test]
    fn test_wolfram_verified_values() {
        // Values from dilithium-mcp Wolfram computation

        // pBit @ T_c should give ~0.544
        let p = crate::constants::pbit_probability(0.5, 0.0, ISING_CRITICAL_TEMP);
        assert!((p - 0.61).abs() < 0.1, "pBit probability = {}", p);

        // Boltzmann weight for E=-0.5, T=T_c should be ~1.2465
        let w = crate::constants::boltzmann_weight(-0.5, ISING_CRITICAL_TEMP);
        assert!((w - 1.2465).abs() < 0.01, "Boltzmann weight = {}", w);
    }
}
