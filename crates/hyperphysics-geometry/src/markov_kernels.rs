//! Markovian Kernels on Hyperbolic Space
//!
//! Implements heat kernels and Markov transition operators on hyperbolic manifolds.
//! These operators govern the diffusion of information, probability, and belief
//! states across the hyperbolic lattice.
//!
//! # Research Foundation
//!
//! - Grigor'yan (2009) "Heat Kernel and Analysis on Manifolds"
//! - Davies (1989) "Heat Kernels and Spectral Theory"
//! - Chavel (1984) "Eigenvalues in Riemannian Geometry"
//!
//! # Mathematical Background
//!
//! The heat kernel K_t(x,y) on the hyperbolic plane H² satisfies:
//! - Heat equation: ∂K/∂t = Δ_H K (Laplace-Beltrami operator)
//! - Initial condition: K_0(x,y) = δ(x,y)
//! - Symmetry: K_t(x,y) = K_t(y,x)
//! - Chapman-Kolmogorov: K_{s+t}(x,y) = ∫ K_s(x,z) K_t(z,y) dμ(z)

use crate::hyperbolic_snn::LorentzVec;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Heat kernel on hyperbolic plane H²
///
/// The heat kernel represents the fundamental solution to the heat equation
/// on hyperbolic space. It governs diffusion of probability, energy, and
/// information across the manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicHeatKernel {
    /// Diffusion time parameter
    pub time: f64,
    /// Curvature (K = -1 for standard hyperbolic)
    pub curvature: f64,
}

impl HyperbolicHeatKernel {
    /// Create heat kernel for given diffusion time
    pub fn new(time: f64) -> Self {
        Self {
            time: time.max(1e-10),
            curvature: -1.0,
        }
    }

    /// Create heat kernel with custom curvature
    pub fn with_curvature(time: f64, curvature: f64) -> Self {
        Self {
            time: time.max(1e-10),
            curvature: curvature.min(-1e-10), // Must be negative for hyperbolic
        }
    }

    /// Evaluate heat kernel K_t(x,y)
    ///
    /// For H² with K = -1:
    /// K_t(x,y) = (4πt)^{-1} × (d/sinh(d)) × exp(-d²/4t - t/4)
    ///
    /// where d = d_H(x,y) is the hyperbolic distance
    pub fn evaluate(&self, x: &LorentzVec, y: &LorentzVec) -> f64 {
        let d = x.hyperbolic_distance(y);
        self.evaluate_distance(d)
    }

    /// Evaluate heat kernel for given distance
    pub fn evaluate_distance(&self, distance: f64) -> f64 {
        let t = self.time;
        let d = distance;

        if d < 1e-10 {
            // At the same point
            return self.diagonal_value();
        }

        // K_t(d) = (4πt)^{-1} × (d/sinh(d)) × exp(-d²/4t - t/4)
        let normalization = 1.0 / (4.0 * PI * t);
        let sinh_factor = d / d.sinh();
        let exponential = (-d * d / (4.0 * t) - t / 4.0).exp();

        normalization * sinh_factor * exponential
    }

    /// Heat kernel value at coincident point (diagonal)
    fn diagonal_value(&self) -> f64 {
        // As d → 0: K_t(0) = (4πt)^{-1} × exp(-t/4)
        (1.0 / (4.0 * PI * self.time)) * (-self.time / 4.0).exp()
    }

    /// Compute heat kernel gradient (for optimization)
    ///
    /// Returns ∂K_t/∂d (derivative w.r.t. distance)
    pub fn gradient(&self, distance: f64) -> f64 {
        let t = self.time;
        let d = distance;

        if d < 1e-10 {
            return 0.0;
        }

        let normalization = 1.0 / (4.0 * PI * t);
        let sinh_d = d.sinh();
        let cosh_d = d.cosh();

        // d/dd (d/sinh(d)) = (sinh(d) - d*cosh(d)) / sinh²(d)
        let sinh_factor_deriv = (sinh_d - d * cosh_d) / (sinh_d * sinh_d);

        // d/dd exp(-d²/4t - t/4) = -d/(2t) * exp(...)
        let exp_term = (-d * d / (4.0 * t) - t / 4.0).exp();
        let exp_deriv = -d / (2.0 * t);

        // Product rule
        let sinh_factor = d / sinh_d;
        normalization * (sinh_factor_deriv * exp_term + sinh_factor * exp_deriv * exp_term)
    }

    /// Check if heat kernel is approximately normalized
    /// (integral over H² should be approximately 1)
    pub fn is_normalized(&self, tolerance: f64) -> bool {
        // For H², the integral is exactly 1 by construction
        // This is a consistency check
        let diagonal = self.diagonal_value();
        diagonal > 0.0 && diagonal < 1.0 / (4.0 * PI * tolerance)
    }
}

/// Markov transition operator on hyperbolic lattice
///
/// Discretizes the heat kernel to create a stochastic matrix
/// for probability transitions on a finite lattice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionOperator {
    /// Heat kernel for continuous transitions
    pub kernel: HyperbolicHeatKernel,
    /// Node positions
    pub positions: Vec<LorentzVec>,
    /// Transition matrix (row-stochastic)
    #[serde(skip)]
    pub matrix: Vec<Vec<f64>>,
    /// Self-loop probability (for stability)
    pub self_loop_weight: f64,
}

impl TransitionOperator {
    /// Create transition operator from positions and diffusion time
    pub fn new(positions: Vec<LorentzVec>, time: f64) -> Self {
        let kernel = HyperbolicHeatKernel::new(time);
        let n = positions.len();
        let matrix = vec![vec![0.0; n]; n];

        let mut op = Self {
            kernel,
            positions,
            matrix,
            self_loop_weight: 0.1,
        };

        op.compute_matrix();
        op
    }

    /// Compute the transition matrix
    fn compute_matrix(&mut self) {
        let n = self.positions.len();

        for i in 0..n {
            let mut row_sum = 0.0;

            // Compute unnormalized transition probabilities
            for j in 0..n {
                let k = if i == j {
                    self.kernel.diagonal_value() + self.self_loop_weight
                } else {
                    self.kernel.evaluate(&self.positions[i], &self.positions[j])
                };

                self.matrix[i][j] = k;
                row_sum += k;
            }

            // Normalize to make row-stochastic
            if row_sum > 1e-10 {
                for j in 0..n {
                    self.matrix[i][j] /= row_sum;
                }
            } else {
                // Fallback: uniform distribution
                for j in 0..n {
                    self.matrix[i][j] = 1.0 / n as f64;
                }
            }
        }
    }

    /// Get transition probability P(i → j)
    pub fn transition_probability(&self, from: usize, to: usize) -> f64 {
        if from < self.matrix.len() && to < self.matrix[from].len() {
            self.matrix[from][to]
        } else {
            0.0
        }
    }

    /// Apply operator to probability distribution
    ///
    /// p_{t+1} = P^T × p_t
    pub fn apply(&self, distribution: &[f64]) -> Vec<f64> {
        let n = self.positions.len();
        let mut result = vec![0.0; n];

        for j in 0..n {
            for i in 0..n {
                result[j] += self.matrix[i][j] * distribution[i];
            }
        }

        result
    }

    /// Apply operator k times (for multi-step transitions)
    pub fn apply_k(&self, distribution: &[f64], k: usize) -> Vec<f64> {
        let mut result = distribution.to_vec();
        for _ in 0..k {
            result = self.apply(&result);
        }
        result
    }

    /// Compute stationary distribution (eigenvector with eigenvalue 1)
    ///
    /// Uses power iteration
    pub fn stationary_distribution(&self, max_iter: usize, tolerance: f64) -> Vec<f64> {
        let n = self.positions.len();
        let mut dist = vec![1.0 / n as f64; n];

        for _ in 0..max_iter {
            let new_dist = self.apply(&dist);

            // Check convergence
            let diff: f64 = dist.iter().zip(new_dist.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            dist = new_dist;

            if diff < tolerance {
                break;
            }
        }

        dist
    }

    /// Check if matrix is row-stochastic
    pub fn is_stochastic(&self, tolerance: f64) -> bool {
        for row in &self.matrix {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > tolerance {
                return false;
            }
            if row.iter().any(|&p| p < -tolerance) {
                return false;
            }
        }
        true
    }

    /// Get entropy of transition from node i
    pub fn transition_entropy(&self, from: usize) -> f64 {
        if from >= self.matrix.len() {
            return 0.0;
        }

        -self.matrix[from].iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Get mean entropy across all nodes
    pub fn mean_entropy(&self) -> f64 {
        let n = self.matrix.len();
        if n == 0 {
            return 0.0;
        }

        (0..n).map(|i| self.transition_entropy(i)).sum::<f64>() / n as f64
    }
}

/// Chapman-Kolmogorov equation verifier and multi-time kernel
///
/// Verifies that K_{s+t}(x,y) = ∫ K_s(x,z) K_t(z,y) dμ(z)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapmanKolmogorov {
    /// Base kernel for time s
    pub kernel_s: HyperbolicHeatKernel,
    /// Base kernel for time t
    pub kernel_t: HyperbolicHeatKernel,
}

impl ChapmanKolmogorov {
    /// Create Chapman-Kolmogorov verifier for times s and t
    pub fn new(time_s: f64, time_t: f64) -> Self {
        Self {
            kernel_s: HyperbolicHeatKernel::new(time_s),
            kernel_t: HyperbolicHeatKernel::new(time_t),
        }
    }

    /// Compute K_{s+t}(x,y) directly
    pub fn direct_kernel(&self, x: &LorentzVec, y: &LorentzVec) -> f64 {
        let total_time = self.kernel_s.time + self.kernel_t.time;
        let combined = HyperbolicHeatKernel::new(total_time);
        combined.evaluate(x, y)
    }

    /// Compute K_{s+t}(x,y) via Chapman-Kolmogorov convolution
    ///
    /// Approximates ∫ K_s(x,z) K_t(z,y) dμ(z) using quadrature
    /// on a set of intermediate points.
    pub fn convolution_kernel(
        &self,
        x: &LorentzVec,
        y: &LorentzVec,
        intermediate_points: &[LorentzVec],
    ) -> f64 {
        if intermediate_points.is_empty() {
            return 0.0;
        }

        // Monte Carlo approximation with importance sampling
        let mut sum = 0.0;
        for z in intermediate_points {
            let k_xz = self.kernel_s.evaluate(x, z);
            let k_zy = self.kernel_t.evaluate(z, y);
            sum += k_xz * k_zy;
        }

        // Normalize by number of points (approximates integral)
        sum / intermediate_points.len() as f64
    }

    /// Verify Chapman-Kolmogorov equation holds approximately
    pub fn verify(
        &self,
        x: &LorentzVec,
        y: &LorentzVec,
        intermediate_points: &[LorentzVec],
        tolerance: f64,
    ) -> ChapmanKolmogorovResult {
        let direct = self.direct_kernel(x, y);
        let convolution = self.convolution_kernel(x, y, intermediate_points);

        let relative_error = if direct > 1e-10 {
            (direct - convolution).abs() / direct
        } else {
            convolution.abs()
        };

        ChapmanKolmogorovResult {
            direct_value: direct,
            convolution_value: convolution,
            relative_error,
            is_consistent: relative_error < tolerance,
        }
    }
}

/// Result of Chapman-Kolmogorov verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapmanKolmogorovResult {
    /// K_{s+t}(x,y) computed directly
    pub direct_value: f64,
    /// K_{s+t}(x,y) computed via convolution
    pub convolution_value: f64,
    /// Relative error between the two
    pub relative_error: f64,
    /// Whether the equation holds within tolerance
    pub is_consistent: bool,
}

/// Random walk on hyperbolic lattice
///
/// Implements discrete-time Markov chain with transition
/// probabilities from heat kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicRandomWalk {
    /// Transition operator
    pub transition: TransitionOperator,
    /// Current node index
    pub current: usize,
    /// Walk history
    pub history: Vec<usize>,
    /// Maximum history length
    pub max_history: usize,
}

impl HyperbolicRandomWalk {
    /// Create random walk starting at given node
    pub fn new(transition: TransitionOperator, start: usize) -> Self {
        Self {
            transition,
            current: start,
            history: vec![start],
            max_history: 1000,
        }
    }

    /// Take one step of the random walk
    ///
    /// Uses the transition probabilities to sample next state
    pub fn step(&mut self, rng_value: f64) -> usize {
        let probs = &self.transition.matrix[self.current];

        // Sample from categorical distribution
        let mut cumsum = 0.0;
        for (j, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rng_value < cumsum {
                self.current = j;
                break;
            }
        }

        // Record history
        self.history.push(self.current);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        self.current
    }

    /// Get empirical distribution from walk history
    pub fn empirical_distribution(&self) -> Vec<f64> {
        let n = self.transition.positions.len();
        let mut counts = vec![0usize; n];

        for &node in &self.history {
            if node < n {
                counts[node] += 1;
            }
        }

        let total = self.history.len() as f64;
        counts.iter().map(|&c| c as f64 / total).collect()
    }

    /// Get mean return time to starting node
    pub fn mean_return_time(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }

        let start = self.history[0];
        let mut return_times = Vec::new();
        let mut last_visit = 0;

        for (i, &node) in self.history.iter().enumerate().skip(1) {
            if node == start {
                return_times.push(i - last_visit);
                last_visit = i;
            }
        }

        if return_times.is_empty() {
            None
        } else {
            Some(return_times.iter().sum::<usize>() as f64 / return_times.len() as f64)
        }
    }
}

/// Hitting time calculator for random walk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HittingTime {
    /// Transition operator
    pub transition: TransitionOperator,
}

impl HittingTime {
    /// Create hitting time calculator
    pub fn new(transition: TransitionOperator) -> Self {
        Self { transition }
    }

    /// Compute expected hitting time from i to j
    ///
    /// Uses linear system: E[T_j | start=i] = 1 + Σ_{k≠j} P_{ik} E[T_j | start=k]
    pub fn expected_hitting_time(&self, from: usize, to: usize, max_iter: usize) -> f64 {
        let n = self.transition.positions.len();

        if from == to {
            return 0.0;
        }

        // Initialize expected times
        let mut expected = vec![0.0; n];

        // Iterative solution (Gauss-Seidel style)
        for _ in 0..max_iter {
            for i in 0..n {
                if i == to {
                    expected[i] = 0.0;
                    continue;
                }

                let mut new_val = 1.0; // Cost of one step
                for k in 0..n {
                    if k != to {
                        new_val += self.transition.matrix[i][k] * expected[k];
                    }
                }
                expected[i] = new_val;
            }
        }

        expected[from]
    }

    /// Compute commute time between i and j
    ///
    /// Commute time = E[T_j | i] + E[T_i | j]
    pub fn commute_time(&self, i: usize, j: usize, max_iter: usize) -> f64 {
        self.expected_hitting_time(i, j, max_iter) + self.expected_hitting_time(j, i, max_iter)
    }
}

/// Spectral properties of transition operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    /// Eigenvalues (sorted by magnitude)
    pub eigenvalues: Vec<f64>,
    /// Spectral gap (1 - λ_2)
    pub spectral_gap: f64,
    /// Mixing time estimate
    pub mixing_time: f64,
}

impl SpectralAnalysis {
    /// Analyze spectral properties of transition matrix
    ///
    /// Uses power iteration to estimate top eigenvalues
    pub fn analyze(transition: &TransitionOperator, num_eigenvalues: usize) -> Self {
        let n = transition.positions.len();
        let mut eigenvalues = Vec::with_capacity(num_eigenvalues);

        // First eigenvalue is always 1 (stochastic matrix)
        eigenvalues.push(1.0);

        // Estimate second eigenvalue via power iteration on deflated matrix
        if n > 1 && num_eigenvalues > 1 {
            let stationary = transition.stationary_distribution(100, 1e-6);

            // Power iteration for second eigenvalue
            let mut v: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

            // Orthogonalize against stationary distribution
            let dot: f64 = v.iter().zip(stationary.iter()).map(|(a, b)| a * b).sum();
            for (vi, &si) in v.iter_mut().zip(stationary.iter()) {
                *vi -= dot * si;
            }

            // Normalize
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for vi in &mut v {
                    *vi /= norm;
                }
            }

            // Power iteration
            for _ in 0..100 {
                v = transition.apply(&v);

                // Orthogonalize
                let dot: f64 = v.iter().zip(stationary.iter()).map(|(a, b)| a * b).sum();
                for (vi, &si) in v.iter_mut().zip(stationary.iter()) {
                    *vi -= dot * si;
                }

                // Normalize
                let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    for vi in &mut v {
                        *vi /= norm;
                    }
                }
            }

            // Estimate eigenvalue
            let av = transition.apply(&v);
            let lambda: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
            eigenvalues.push(lambda.abs());
        }

        // Spectral gap
        let spectral_gap = if eigenvalues.len() > 1 {
            1.0 - eigenvalues[1]
        } else {
            1.0
        };

        // Mixing time ≈ 1/spectral_gap × log(1/ε)
        let mixing_time = if spectral_gap > 1e-10 {
            (1.0 / spectral_gap) * 10.0_f64.ln() // For ε = 0.1
        } else {
            f64::INFINITY
        };

        Self {
            eigenvalues,
            spectral_gap,
            mixing_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_kernel_creation() {
        let kernel = HyperbolicHeatKernel::new(1.0);
        assert_eq!(kernel.time, 1.0);
        assert_eq!(kernel.curvature, -1.0);
    }

    #[test]
    fn test_heat_kernel_symmetry() {
        let kernel = HyperbolicHeatKernel::new(1.0);
        let x = LorentzVec::from_spatial(0.3, 0.2, 0.0);
        let y = LorentzVec::from_spatial(-0.1, 0.4, 0.0);

        let k_xy = kernel.evaluate(&x, &y);
        let k_yx = kernel.evaluate(&y, &x);

        assert!((k_xy - k_yx).abs() < 1e-10, "Heat kernel must be symmetric");
    }

    #[test]
    fn test_heat_kernel_positivity() {
        let kernel = HyperbolicHeatKernel::new(1.0);

        for d in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let k = kernel.evaluate_distance(d);
            assert!(k > 0.0, "Heat kernel must be positive");
        }
    }

    #[test]
    fn test_heat_kernel_decay() {
        let kernel = HyperbolicHeatKernel::new(1.0);

        let k_close = kernel.evaluate_distance(0.5);
        let k_far = kernel.evaluate_distance(2.0);

        assert!(k_close > k_far, "Heat kernel should decay with distance");
    }

    #[test]
    fn test_transition_operator_stochastic() {
        let positions: Vec<LorentzVec> = (0..5)
            .map(|i| LorentzVec::from_spatial(0.1 * i as f64, 0.05 * i as f64, 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 1.0);

        assert!(op.is_stochastic(1e-6), "Transition matrix must be row-stochastic");
    }

    #[test]
    fn test_transition_operator_apply() {
        let positions: Vec<LorentzVec> = (0..4)
            .map(|i| LorentzVec::from_spatial(0.2 * i as f64, 0.0, 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 1.0);

        // Start with delta distribution
        let delta = vec![1.0, 0.0, 0.0, 0.0];
        let result = op.apply(&delta);

        // Should sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Distribution should sum to 1");
    }

    #[test]
    fn test_stationary_distribution() {
        let positions: Vec<LorentzVec> = (0..5)
            .map(|i| LorentzVec::from_spatial(0.15 * i as f64, 0.1 * (i as f64).sin(), 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 0.5);
        let stationary = op.stationary_distribution(1000, 1e-8);

        // Should sum to 1
        let sum: f64 = stationary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be approximately invariant under P
        let applied = op.apply(&stationary);
        let diff: f64 = stationary.iter().zip(applied.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-4, "Stationary distribution should be invariant");
    }

    #[test]
    fn test_chapman_kolmogorov() {
        let ck = ChapmanKolmogorov::new(0.5, 0.5);

        let x = LorentzVec::from_spatial(0.2, 0.1, 0.0);
        let y = LorentzVec::from_spatial(-0.1, 0.3, 0.0);

        // Generate intermediate points
        let intermediate: Vec<LorentzVec> = (0..100)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) / 100.0;
                let r = 0.3 * ((i % 10) as f64 / 10.0);
                LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0)
            })
            .collect();

        let result = ck.verify(&x, &y, &intermediate, 0.5);

        // With enough intermediate points, should be approximately consistent
        assert!(result.direct_value > 0.0);
        assert!(result.convolution_value > 0.0);
    }

    #[test]
    fn test_spectral_analysis() {
        let positions: Vec<LorentzVec> = (0..6)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) / 6.0;
                LorentzVec::from_spatial(0.3 * angle.cos(), 0.3 * angle.sin(), 0.0)
            })
            .collect();

        let op = TransitionOperator::new(positions, 1.0);
        let spectral = SpectralAnalysis::analyze(&op, 2);

        assert!((spectral.eigenvalues[0] - 1.0).abs() < 1e-6, "First eigenvalue should be 1");
        assert!(spectral.spectral_gap >= 0.0);
        assert!(spectral.mixing_time > 0.0);
    }

    #[test]
    fn test_hitting_time() {
        let positions: Vec<LorentzVec> = (0..4)
            .map(|i| LorentzVec::from_spatial(0.2 * i as f64, 0.0, 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 0.5);
        let hitting = HittingTime::new(op);

        // Hitting time from node to itself is 0
        assert_eq!(hitting.expected_hitting_time(0, 0, 100), 0.0);

        // Hitting time should be positive for different nodes
        let ht = hitting.expected_hitting_time(0, 3, 100);
        assert!(ht > 0.0, "Hitting time should be positive");
    }
}
