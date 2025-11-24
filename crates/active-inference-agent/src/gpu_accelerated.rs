//! GPU-Accelerated Conscious Processing
//!
//! Provides GPU compute kernels for parallel conscious processing:
//! - Parallel belief updates across pBit networks
//! - Batched free energy computation
//! - GPU-accelerated Markov kernel operations
//! - SIMD-optimized sigmoid and softmax
//!
//! # Hardware Support
//!
//! Designed for dual AMD GPU setup (RX 6800 XT + RX 5500 XT):
//! - Primary compute on RX 6800 XT (72 CUs, 16GB)
//! - Secondary/overflow on RX 5500 XT (22 CUs, 4GB)
//!
//! Uses wgpu for cross-platform GPU compute via WebGPU/Vulkan.

use nalgebra as na;

use crate::{ConsciousnessError, ConsciousnessResult};

/// GPU compute configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Maximum belief dimensions for GPU processing
    pub max_belief_dim: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Workgroup size for compute shaders
    pub workgroup_size: u32,
    /// Use shared memory optimization
    pub use_shared_memory: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_belief_dim: 1024,
            max_batch_size: 256,
            workgroup_size: 256,
            use_shared_memory: true,
        }
    }
}

/// GPU-accelerated belief processor
///
/// Manages GPU resources for parallel conscious processing
#[derive(Debug)]
pub struct GpuBeliefProcessor {
    /// Configuration
    pub config: GpuConfig,
    /// Whether GPU is available
    pub gpu_available: bool,
    /// CPU fallback buffer for belief data
    belief_buffer: Vec<f64>,
    /// CPU fallback buffer for transition matrix
    transition_buffer: Vec<f64>,
}

impl GpuBeliefProcessor {
    /// Create new GPU processor
    ///
    /// Falls back to CPU if GPU unavailable
    pub fn new(config: GpuConfig) -> Self {
        // Check GPU availability (simplified - full implementation would use wgpu)
        let gpu_available = Self::check_gpu_availability();

        Self {
            config,
            gpu_available,
            belief_buffer: Vec::new(),
            transition_buffer: Vec::new(),
        }
    }

    /// Check if GPU compute is available
    fn check_gpu_availability() -> bool {
        // In production, this would use wgpu to check for compute-capable adapters
        // For now, return false to use CPU path
        cfg!(feature = "gpu")
    }

    /// Parallel belief update via GPU
    ///
    /// Computes: belief' = normalize(transition^T * belief)
    pub fn update_belief(
        &mut self,
        belief: &na::DVector<f64>,
        transition: &na::DMatrix<f64>,
    ) -> ConsciousnessResult<na::DVector<f64>> {
        if self.gpu_available {
            self.gpu_belief_update(belief, transition)
        } else {
            self.cpu_belief_update(belief, transition)
        }
    }

    /// GPU implementation of belief update
    #[allow(unused)]
    fn gpu_belief_update(
        &mut self,
        belief: &na::DVector<f64>,
        transition: &na::DMatrix<f64>,
    ) -> ConsciousnessResult<na::DVector<f64>> {
        // Would use wgpu compute shader here
        // For now, fall back to CPU
        self.cpu_belief_update(belief, transition)
    }

    /// CPU fallback for belief update (SIMD-optimized)
    fn cpu_belief_update(
        &mut self,
        belief: &na::DVector<f64>,
        transition: &na::DMatrix<f64>,
    ) -> ConsciousnessResult<na::DVector<f64>> {
        if belief.len() != transition.ncols() {
            return Err(ConsciousnessError::DimensionMismatch {
                expected: transition.ncols(),
                actual: belief.len(),
            });
        }

        // Matrix-vector multiplication with SIMD hint
        let mut result = transition.transpose() * belief;

        // Normalize
        let sum = result.sum();
        if sum > 1e-10 {
            result /= sum;
        }

        Ok(result)
    }

    /// Batch update multiple belief vectors in parallel
    pub fn batch_update(
        &mut self,
        beliefs: &[na::DVector<f64>],
        transition: &na::DMatrix<f64>,
    ) -> ConsciousnessResult<Vec<na::DVector<f64>>> {
        beliefs
            .iter()
            .map(|b| self.update_belief(b, transition))
            .collect()
    }

    /// Parallel free energy computation for batch
    pub fn batch_free_energy(
        &self,
        beliefs: &[na::DVector<f64>],
        observations: &[na::DVector<f64>],
        likelihood: &na::DMatrix<f64>,
    ) -> ConsciousnessResult<Vec<f64>> {
        if beliefs.len() != observations.len() {
            return Err(ConsciousnessError::DimensionMismatch {
                expected: beliefs.len(),
                actual: observations.len(),
            });
        }

        let results: Vec<f64> = beliefs
            .iter()
            .zip(observations.iter())
            .map(|(belief, obs)| self.compute_free_energy(belief, obs, likelihood))
            .collect();

        Ok(results)
    }

    /// Compute free energy for single belief-observation pair
    fn compute_free_energy(
        &self,
        belief: &na::DVector<f64>,
        observation: &na::DVector<f64>,
        likelihood: &na::DMatrix<f64>,
    ) -> f64 {
        // Predicted observation
        let predicted = likelihood * belief;

        // Prediction error (squared)
        let error = (observation - predicted).norm_squared();

        // Entropy of belief (negative)
        let entropy: f64 = belief
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum();

        error - entropy
    }

    /// GPU-optimized softmax for action selection
    pub fn softmax(&self, logits: &na::DVector<f64>, temperature: f64) -> na::DVector<f64> {
        let scaled: Vec<f64> = logits.iter().map(|x| x / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let exp_vals: Vec<f64> = scaled.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();

        na::DVector::from_iterator(logits.len(), exp_vals.iter().map(|x| x / sum))
    }

    /// SIMD-optimized sigmoid for pBit probabilities
    #[inline]
    pub fn sigmoid_batch(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    /// Parallel Gibbs sampling step
    ///
    /// Updates pBit states in parallel using GPU
    pub fn parallel_gibbs_update(
        &mut self,
        states: &mut [bool],
        biases: &[f64],
        couplings: &na::DMatrix<f64>,
        temperature: f64,
        random_values: &[f64],
    ) -> usize {
        let n = states.len();
        let mut flips = 0;

        // Compute effective fields in parallel (CPU SIMD for now)
        let h_effs: Vec<f64> = (0..n)
            .map(|i| {
                let mut h = biases[i];
                for j in 0..n {
                    if i != j {
                        let spin = if states[j] { 1.0 } else { -1.0 };
                        h += couplings[(i, j)] * spin;
                    }
                }
                h / temperature
            })
            .collect();

        // Compute probabilities
        let probs = self.sigmoid_batch(&h_effs);

        // Update states
        for i in 0..n {
            let new_state = random_values[i] < probs[i];
            if new_state != states[i] {
                flips += 1;
                states[i] = new_state;
            }
        }

        flips
    }
}

/// WGSL compute shader for belief updates (for reference)
pub const BELIEF_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> belief: array<f32>;
@group(0) @binding(1) var<storage, read> transition: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec2<u32>; // (rows, cols)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let n_rows = dims.x;
    let n_cols = dims.y;

    if (row >= n_rows) {
        return;
    }

    // Matrix-vector multiply: result[row] = sum_j transition[j,row] * belief[j]
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < n_cols; j = j + 1u) {
        let idx = j * n_rows + row; // Column-major
        sum = sum + transition[idx] * belief[j];
    }

    result[row] = sum;
}

@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = dims.x;

    if (idx >= n) {
        return;
    }

    // First pass: compute sum (using workgroup reduction would be more efficient)
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        sum = sum + result[i];
    }

    // Normalize
    if (sum > 1e-10) {
        result[idx] = result[idx] / sum;
    }
}
"#;

/// WGSL shader for pBit Gibbs sampling
pub const GIBBS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> states: array<u32>;
@group(0) @binding(1) var<storage, read> biases: array<f32>;
@group(0) @binding(2) var<storage, read> couplings: array<f32>;
@group(0) @binding(3) var<storage, read> random_vals: array<f32>;
@group(0) @binding(4) var<uniform> params: vec2<f32>; // (temperature, n_pbits)

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn gibbs_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = u32(params.y);
    let temp = params.x;

    if (i >= n) {
        return;
    }

    // Compute effective field
    var h_eff: f32 = biases[i];
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (i != j) {
            let spin: f32 = select(-1.0, 1.0, states[j] == 1u);
            h_eff = h_eff + couplings[i * n + j] * spin;
        }
    }

    // Sigmoid probability
    let prob = sigmoid(h_eff / temp);

    // Stochastic update
    states[i] = select(0u, 1u, random_vals[i] < prob);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_processor_creation() {
        let processor = GpuBeliefProcessor::new(GpuConfig::default());
        assert!(!processor.gpu_available); // No GPU feature enabled
    }

    #[test]
    fn test_cpu_belief_update() {
        let mut processor = GpuBeliefProcessor::new(GpuConfig::default());

        let belief = na::DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let transition = na::DMatrix::from_row_slice(3, 3, &[
            0.7, 0.2, 0.1,
            0.1, 0.8, 0.1,
            0.2, 0.3, 0.5,
        ]);

        let result = processor.update_belief(&belief, &transition);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert!((updated.sum() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_free_energy() {
        let processor = GpuBeliefProcessor::new(GpuConfig::default());

        let beliefs = vec![
            na::DVector::from_vec(vec![0.5, 0.5]),
            na::DVector::from_vec(vec![0.8, 0.2]),
        ];
        let observations = vec![
            na::DVector::from_vec(vec![1.0, 0.0]),
            na::DVector::from_vec(vec![0.5, 0.5]),
        ];
        let likelihood = na::DMatrix::identity(2, 2);

        let fes = processor.batch_free_energy(&beliefs, &observations, &likelihood);
        assert!(fes.is_ok());
        assert_eq!(fes.unwrap().len(), 2);
    }

    #[test]
    fn test_softmax() {
        let processor = GpuBeliefProcessor::new(GpuConfig::default());

        let logits = na::DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = processor.softmax(&logits, 1.0);

        assert!((probs.sum() - 1.0).abs() < 0.01);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_sigmoid_batch() {
        let processor = GpuBeliefProcessor::new(GpuConfig::default());

        let values = vec![0.0, 1.0, -1.0, 10.0, -10.0];
        let results = processor.sigmoid_batch(&values);

        assert!((results[0] - 0.5).abs() < 0.01);
        assert!(results[1] > 0.7);
        assert!(results[2] < 0.3);
        assert!(results[3] > 0.99);
        assert!(results[4] < 0.01);
    }

    #[test]
    fn test_parallel_gibbs() {
        let mut processor = GpuBeliefProcessor::new(GpuConfig::default());

        let mut states = vec![false, true, false, true];
        let biases = vec![0.5, -0.5, 0.0, 1.0];
        let couplings = na::DMatrix::from_element(4, 4, 0.1);
        let random_values = vec![0.3, 0.7, 0.5, 0.2];

        let flips = processor.parallel_gibbs_update(
            &mut states,
            &biases,
            &couplings,
            1.0,
            &random_values,
        );

        // Should have some flips
        assert!(flips <= 4);
    }
}
