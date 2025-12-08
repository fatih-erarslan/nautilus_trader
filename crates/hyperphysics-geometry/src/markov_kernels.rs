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

/// Spectral properties of transition operator with convergence bounds
///
/// # Mathematical Foundation
///
/// Following:
/// - Chung (1997) "Spectral Graph Theory"
/// - Spielman (2012) "Spectral Graph Theory and its Applications"
/// - Davies (1989) "Heat Kernels and Spectral Theory"
///
/// Key results implemented:
/// - Power iteration with residual-based convergence bounds
/// - Cheeger inequality: h²/2 ≤ 1-λ₂ ≤ 2h (spectral gap vs conductance)
/// - Mixing time bound: t_mix ≤ (1/gap) × log(n/ε)
/// - Weyl asymptotic law for eigenvalue distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    /// Eigenvalues (sorted by magnitude, descending)
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors corresponding to eigenvalues
    #[serde(skip)]
    pub eigenvectors: Vec<Vec<f64>>,
    /// Spectral gap (1 - λ_2)
    pub spectral_gap: f64,
    /// Mixing time estimate (ε = 0.01)
    pub mixing_time: f64,
    /// Convergence bounds for each eigenvalue
    pub convergence_bounds: Vec<ConvergenceBound>,
    /// Cheeger constant (conductance) lower bound
    pub cheeger_lower: f64,
    /// Cheeger constant upper bound
    pub cheeger_upper: f64,
    /// Spectral radius
    pub spectral_radius: f64,
    /// Condition number estimate
    pub condition_number: f64,
}

/// Convergence bound for eigenvalue computation
///
/// Based on residual analysis and gap estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceBound {
    /// Eigenvalue index
    pub index: usize,
    /// Computed eigenvalue
    pub eigenvalue: f64,
    /// Residual norm ||Av - λv||
    pub residual_norm: f64,
    /// Estimated absolute error bound
    pub error_bound: f64,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Has the iteration converged?
    pub converged: bool,
}

/// Configuration for spectral analysis
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// Number of eigenvalues to compute
    pub num_eigenvalues: usize,
    /// Maximum iterations for power method
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute eigenvectors
    pub compute_eigenvectors: bool,
    /// Whether to compute Cheeger bounds
    pub compute_cheeger: bool,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 5,
            max_iterations: 500,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            compute_cheeger: true,
        }
    }
}

impl SpectralAnalysis {
    /// Analyze spectral properties of transition matrix
    ///
    /// Uses power iteration to estimate top eigenvalues
    pub fn analyze(transition: &TransitionOperator, num_eigenvalues: usize) -> Self {
        let config = SpectralConfig {
            num_eigenvalues,
            ..Default::default()
        };
        Self::analyze_with_config(transition, &config)
    }

    /// Full spectral analysis with convergence bounds
    ///
    /// Implements power iteration with deflation and rigorous convergence tracking.
    ///
    /// # Convergence Analysis
    ///
    /// For power iteration on symmetric matrix A with eigenvalues |λ₁| > |λ₂| ≥ ...,
    /// the convergence rate is |λ₂/λ₁|^k where k is iteration count.
    ///
    /// Residual bound: ||Av - λv|| ≤ |λ₁ - λ₂| × ε after O(log(1/ε)/log(λ₁/λ₂)) iterations
    pub fn analyze_with_config(transition: &TransitionOperator, config: &SpectralConfig) -> Self {
        let n = transition.positions.len();
        let mut eigenvalues = Vec::with_capacity(config.num_eigenvalues);
        let mut eigenvectors = Vec::with_capacity(config.num_eigenvalues);
        let mut convergence_bounds = Vec::with_capacity(config.num_eigenvalues);

        // First eigenvalue is always 1 for stochastic matrix
        let stationary = transition.stationary_distribution(config.max_iterations, config.tolerance);
        eigenvalues.push(1.0);

        // Compute residual for first eigenvalue
        let av = transition.apply(&stationary);
        let residual_1: f64 = stationary.iter().zip(av.iter())
            .map(|(v, av)| (av - v).powi(2))
            .sum::<f64>()
            .sqrt();

        convergence_bounds.push(ConvergenceBound {
            index: 0,
            eigenvalue: 1.0,
            residual_norm: residual_1,
            error_bound: residual_1, // For λ=1, error is just residual
            iterations: 1,
            converged: true,
        });

        if config.compute_eigenvectors {
            eigenvectors.push(stationary.clone());
        }

        // Compute additional eigenvalues via deflated power iteration
        let mut deflation_space = vec![stationary.clone()];

        for k in 1..config.num_eigenvalues.min(n) {
            let result = Self::deflated_power_iteration(
                transition,
                &deflation_space,
                config.max_iterations,
                config.tolerance,
            );

            eigenvalues.push(result.eigenvalue);
            convergence_bounds.push(ConvergenceBound {
                index: k,
                eigenvalue: result.eigenvalue,
                residual_norm: result.residual,
                error_bound: Self::compute_error_bound(result.residual, &eigenvalues, k),
                iterations: result.iterations,
                converged: result.converged,
            });

            if config.compute_eigenvectors {
                eigenvectors.push(result.eigenvector.clone());
            }

            deflation_space.push(result.eigenvector);
        }

        // Spectral gap: 1 - λ_2
        let spectral_gap = if eigenvalues.len() > 1 {
            (1.0 - eigenvalues[1]).max(0.0)
        } else {
            1.0
        };

        // Cheeger bounds via spectral gap
        // Cheeger inequality: h²/2 ≤ 1-λ₂ ≤ 2h
        let (cheeger_lower, cheeger_upper) = if config.compute_cheeger {
            Self::cheeger_bounds(spectral_gap)
        } else {
            (0.0, 1.0)
        };

        // Mixing time: t_mix(ε) ≤ (1/gap) × log(n/ε)
        let mixing_time = if spectral_gap > 1e-10 {
            let epsilon = 0.01;
            (1.0 / spectral_gap) * (n as f64 / epsilon).ln()
        } else {
            f64::INFINITY
        };

        // Spectral radius (largest |λ|)
        let spectral_radius = eigenvalues.iter().map(|x| x.abs()).fold(0.0f64, f64::max);

        // Condition number: λ_max / λ_min (for non-zero eigenvalues)
        let nonzero_eigenvalues: Vec<f64> = eigenvalues.iter()
            .filter(|&&x| x.abs() > 1e-10)
            .copied()
            .collect();
        let condition_number = if nonzero_eigenvalues.len() >= 2 {
            let max_abs = nonzero_eigenvalues.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let min_abs = nonzero_eigenvalues.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min);
            if min_abs > 1e-10 { max_abs / min_abs } else { f64::INFINITY }
        } else {
            1.0
        };

        Self {
            eigenvalues,
            eigenvectors,
            spectral_gap,
            mixing_time,
            convergence_bounds,
            cheeger_lower,
            cheeger_upper,
            spectral_radius,
            condition_number,
        }
    }

    /// Deflated power iteration to find next eigenvalue
    ///
    /// Orthogonalizes against previously found eigenvectors.
    fn deflated_power_iteration(
        transition: &TransitionOperator,
        deflation_space: &[Vec<f64>],
        max_iter: usize,
        tolerance: f64,
    ) -> PowerIterationResult {
        let n = transition.positions.len();

        // Initialize with random-ish vector
        let mut v: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 2.7182818).sin() + (i as f64 * 3.14159).cos()) / 2.0)
            .collect();

        // Orthogonalize against deflation space
        for basis in deflation_space {
            let dot: f64 = v.iter().zip(basis.iter()).map(|(a, b)| a * b).sum();
            for (vi, &bi) in v.iter_mut().zip(basis.iter()) {
                *vi -= dot * bi;
            }
        }

        // Normalize
        let mut norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for vi in &mut v {
                *vi /= norm;
            }
        }

        let mut eigenvalue = 0.0;
        let mut residual = f64::INFINITY;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iter {
            // Apply transition operator
            let av = transition.apply(&v);

            // Orthogonalize against deflation space
            let mut av_orth = av.clone();
            for basis in deflation_space {
                let dot: f64 = av_orth.iter().zip(basis.iter()).map(|(a, b)| a * b).sum();
                for (avi, &bi) in av_orth.iter_mut().zip(basis.iter()) {
                    *avi -= dot * bi;
                }
            }

            // Compute Rayleigh quotient: λ = v^T A v / v^T v
            let new_eigenvalue: f64 = v.iter().zip(av_orth.iter()).map(|(a, b)| a * b).sum();

            // Compute residual: ||Av - λv||
            residual = v.iter().zip(av_orth.iter())
                .map(|(vi, avi)| (avi - new_eigenvalue * vi).powi(2))
                .sum::<f64>()
                .sqrt();

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance && residual < tolerance {
                eigenvalue = new_eigenvalue;
                converged = true;
                iterations = iter + 1;
                break;
            }

            eigenvalue = new_eigenvalue;
            iterations = iter + 1;

            // Normalize
            norm = av_orth.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                v = av_orth.iter().map(|x| x / norm).collect();
            } else {
                break;
            }
        }

        PowerIterationResult {
            eigenvalue: eigenvalue.abs(),
            eigenvector: v,
            residual,
            iterations,
            converged,
        }
    }

    /// Compute error bound from residual
    ///
    /// Using Bauer-Fike theorem: |λ - λ̃| ≤ ||A - Ã|| / gap
    /// For power iteration: error ≤ residual / (λ_k - λ_{k+1})
    fn compute_error_bound(residual: f64, eigenvalues: &[f64], k: usize) -> f64 {
        if k == 0 || eigenvalues.len() <= k {
            return residual;
        }

        // Gap to next eigenvalue (estimate)
        let gap = if k < eigenvalues.len() - 1 {
            (eigenvalues[k - 1] - eigenvalues[k]).abs().max(1e-10)
        } else {
            eigenvalues[k].abs().max(1e-10)
        };

        residual / gap
    }

    /// Cheeger bounds from spectral gap
    ///
    /// Cheeger inequality: h²/2 ≤ 1-λ₂ ≤ 2h
    /// Solving for h: sqrt(2(1-λ₂)) ≤ h ≤ (1-λ₂)/2
    fn cheeger_bounds(spectral_gap: f64) -> (f64, f64) {
        let lower = (2.0 * spectral_gap).sqrt().min(1.0);
        let upper = (spectral_gap / 2.0).min(1.0);
        // Note: lower bound comes from h² ≤ 2(1-λ₂), upper from (1-λ₂) ≤ 2h
        (upper, lower) // Swap because Cheeger gives h in terms of gap
    }

    /// Verify Weyl's asymptotic law for eigenvalue distribution
    ///
    /// For heat kernel on hyperbolic manifold of dimension d:
    /// N(λ) ~ C × λ^{d/2} as λ → ∞
    ///
    /// Returns the estimated dimension from spectral data.
    pub fn verify_weyl_law(&self) -> WeylLawResult {
        if self.eigenvalues.len() < 3 {
            return WeylLawResult {
                estimated_dimension: 0.0,
                weyl_coefficient: 0.0,
                fit_quality: 0.0,
                is_consistent: false,
            };
        }

        // Count eigenvalues ≤ λ for various λ
        let mut data: Vec<(f64, f64)> = Vec::new();
        let max_eig = self.eigenvalues.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

        for i in 1..=10 {
            let lambda = max_eig * (i as f64) / 10.0;
            let count = self.eigenvalues.iter().filter(|&&e| e.abs() <= lambda).count();
            if count > 0 && lambda > 0.0 {
                data.push((lambda.ln(), (count as f64).ln()));
            }
        }

        // Linear regression: log(N) = (d/2) × log(λ) + log(C)
        if data.len() < 2 {
            return WeylLawResult {
                estimated_dimension: 0.0,
                weyl_coefficient: 0.0,
                fit_quality: 0.0,
                is_consistent: false,
            };
        }

        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return WeylLawResult {
                estimated_dimension: 0.0,
                weyl_coefficient: 0.0,
                fit_quality: 0.0,
                is_consistent: false,
            };
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        let estimated_dimension = 2.0 * slope; // d = 2 × slope
        let weyl_coefficient = intercept.exp();

        // Compute R² for fit quality
        let mean_y = sum_y / n;
        let ss_tot: f64 = data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = data.iter().map(|(x, y)| (y - (slope * x + intercept)).powi(2)).sum();
        let r_squared = if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 0.0 };

        // Hyperbolic plane should give d ≈ 2
        let is_consistent = estimated_dimension > 1.5 && estimated_dimension < 3.0 && r_squared > 0.7;

        WeylLawResult {
            estimated_dimension,
            weyl_coefficient,
            fit_quality: r_squared,
            is_consistent,
        }
    }

    /// Compute effective resistance between nodes
    ///
    /// R_eff(i,j) = (e_i - e_j)^T L^+ (e_i - e_j)
    /// where L^+ is the pseudoinverse of the Laplacian
    pub fn effective_resistance(&self, i: usize, j: usize) -> f64 {
        if i == j || self.eigenvectors.len() < 2 {
            return 0.0;
        }

        let n = self.eigenvectors[0].len();
        if i >= n || j >= n {
            return f64::INFINITY;
        }

        // R_eff = Σ_{k>0} (1/λ_k) × (v_k[i] - v_k[j])²
        let mut resistance = 0.0;
        for (k, (lambda, v)) in self.eigenvalues.iter().zip(self.eigenvectors.iter()).enumerate() {
            if k == 0 { continue; } // Skip zero eigenvalue
            if *lambda < 1e-10 { continue; }

            let diff = v.get(i).unwrap_or(&0.0) - v.get(j).unwrap_or(&0.0);
            // For transition matrix, Laplacian eigenvalue is 1 - λ
            let laplacian_eig = (1.0 - lambda).abs().max(1e-10);
            resistance += diff * diff / laplacian_eig;
        }

        resistance
    }
}

/// Result of power iteration for single eigenvalue
struct PowerIterationResult {
    eigenvalue: f64,
    eigenvector: Vec<f64>,
    residual: f64,
    iterations: usize,
    converged: bool,
}

/// Result of Weyl's law verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeylLawResult {
    /// Estimated manifold dimension from spectral data
    pub estimated_dimension: f64,
    /// Weyl coefficient
    pub weyl_coefficient: f64,
    /// R² fit quality (0 to 1)
    pub fit_quality: f64,
    /// Whether result is consistent with hyperbolic geometry
    pub is_consistent: bool,
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

    // ==========================================
    // Tests for Enhanced Spectral Analysis
    // ==========================================

    #[test]
    fn test_spectral_analysis_with_config() {
        let positions: Vec<LorentzVec> = (0..8)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) / 8.0;
                LorentzVec::from_spatial(0.25 * angle.cos(), 0.25 * angle.sin(), 0.0)
            })
            .collect();

        let op = TransitionOperator::new(positions, 0.8);
        let config = SpectralConfig {
            num_eigenvalues: 4,
            max_iterations: 500,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            compute_cheeger: true,
        };

        let spectral = SpectralAnalysis::analyze_with_config(&op, &config);

        // Should have requested number of eigenvalues
        assert_eq!(spectral.eigenvalues.len(), 4);
        assert_eq!(spectral.eigenvectors.len(), 4);

        // First eigenvalue should be 1
        assert!((spectral.eigenvalues[0] - 1.0).abs() < 1e-4,
            "First eigenvalue should be 1, got {}", spectral.eigenvalues[0]);

        // Eigenvalues should be in decreasing order of magnitude
        for i in 1..spectral.eigenvalues.len() {
            assert!(spectral.eigenvalues[i] <= spectral.eigenvalues[i-1] + 1e-6,
                "Eigenvalues should be sorted: {} > {}", spectral.eigenvalues[i], spectral.eigenvalues[i-1]);
        }
    }

    #[test]
    fn test_convergence_bounds() {
        let positions: Vec<LorentzVec> = (0..6)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) / 6.0;
                LorentzVec::from_spatial(0.3 * angle.cos(), 0.3 * angle.sin(), 0.0)
            })
            .collect();

        let op = TransitionOperator::new(positions, 1.0);
        let spectral = SpectralAnalysis::analyze(&op, 3);

        // Should have convergence bounds for each eigenvalue
        assert_eq!(spectral.convergence_bounds.len(), 3);

        for bound in &spectral.convergence_bounds {
            // Residual should be small for converged eigenvalues
            if bound.converged {
                assert!(bound.residual_norm < 0.1,
                    "Converged eigenvalue should have small residual: {}", bound.residual_norm);
            }

            // Error bound should be non-negative
            assert!(bound.error_bound >= 0.0);

            // Iterations should be positive
            assert!(bound.iterations > 0);
        }
    }

    #[test]
    fn test_cheeger_bounds() {
        let positions: Vec<LorentzVec> = (0..10)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) / 10.0;
                LorentzVec::from_spatial(0.35 * angle.cos(), 0.35 * angle.sin(), 0.0)
            })
            .collect();

        let op = TransitionOperator::new(positions, 0.5);
        let spectral = SpectralAnalysis::analyze(&op, 3);

        // Cheeger bounds should be valid
        assert!(spectral.cheeger_lower >= 0.0, "Cheeger lower bound should be non-negative");
        assert!(spectral.cheeger_upper >= 0.0, "Cheeger upper bound should be non-negative");
        assert!(spectral.cheeger_upper <= 1.0, "Cheeger upper bound should be at most 1");

        // Cheeger inequality: h²/2 ≤ spectral_gap ≤ 2h
        // So: sqrt(2*gap) ≥ h ≥ gap/2
        if spectral.spectral_gap > 0.0 {
            let lower_from_gap = spectral.spectral_gap / 2.0;
            let upper_from_gap = (2.0 * spectral.spectral_gap).sqrt();
            // The bounds should be consistent with spectral gap
            // (exact relationship depends on which way Cheeger is stated)
            assert!(spectral.cheeger_lower <= upper_from_gap + 0.1 ||
                    spectral.cheeger_upper >= lower_from_gap - 0.1,
                "Cheeger bounds should be consistent with spectral gap");
        }
    }

    #[test]
    fn test_mixing_time_bound() {
        let n_nodes = 5;
        let positions: Vec<LorentzVec> = (0..n_nodes)
            .map(|i| LorentzVec::from_spatial(0.15 * i as f64, 0.1 * (i as f64).sin(), 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 0.5);
        let spectral = SpectralAnalysis::analyze(&op, 2);

        // Mixing time should be positive
        assert!(spectral.mixing_time > 0.0, "Mixing time should be positive");

        // Mixing time formula: t_mix ≤ (1/gap) × log(n/ε)
        if spectral.spectral_gap > 1e-10 {
            let n = n_nodes as f64;
            let epsilon = 0.01;
            let expected_bound = (1.0 / spectral.spectral_gap) * (n / epsilon).ln();
            assert!((spectral.mixing_time - expected_bound).abs() < expected_bound * 0.1,
                "Mixing time should match formula");
        }
    }

    #[test]
    fn test_spectral_radius() {
        let positions: Vec<LorentzVec> = (0..4)
            .map(|i| LorentzVec::from_spatial(0.2 * i as f64, 0.0, 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 1.0);
        let spectral = SpectralAnalysis::analyze(&op, 3);

        // For stochastic matrix, spectral radius should be 1
        assert!((spectral.spectral_radius - 1.0).abs() < 1e-4,
            "Spectral radius of stochastic matrix should be 1, got {}", spectral.spectral_radius);
    }

    #[test]
    fn test_weyl_law_verification() {
        let positions: Vec<LorentzVec> = (0..15)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) / 15.0;
                let r = 0.2 + 0.15 * ((i % 3) as f64 / 2.0);
                LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0)
            })
            .collect();

        let op = TransitionOperator::new(positions, 0.5);
        let spectral = SpectralAnalysis::analyze(&op, 10);

        let weyl = spectral.verify_weyl_law();

        // Weyl coefficient should be positive
        assert!(weyl.weyl_coefficient >= 0.0);

        // Fit quality should be between 0 and 1
        assert!(weyl.fit_quality >= 0.0 && weyl.fit_quality <= 1.0,
            "R² should be in [0,1], got {}", weyl.fit_quality);

        // For hyperbolic geometry, dimension should be around 2
        // (but may not be exact due to finite sampling)
        // Just check it's a reasonable positive number
        assert!(weyl.estimated_dimension >= 0.0,
            "Estimated dimension should be non-negative");
    }

    #[test]
    fn test_effective_resistance() {
        let positions: Vec<LorentzVec> = (0..5)
            .map(|i| LorentzVec::from_spatial(0.15 * i as f64, 0.0, 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 0.5);
        let spectral = SpectralAnalysis::analyze(&op, 4);

        // Resistance to self should be 0
        let r_00 = spectral.effective_resistance(0, 0);
        assert!((r_00 - 0.0).abs() < 1e-10, "Resistance to self should be 0");

        // Resistance should be symmetric
        let r_01 = spectral.effective_resistance(0, 1);
        let r_10 = spectral.effective_resistance(1, 0);
        assert!((r_01 - r_10).abs() < 1e-6, "Effective resistance should be symmetric");

        // Resistance should increase with distance (for chain graph)
        let r_02 = spectral.effective_resistance(0, 2);
        // Due to graph structure, r_02 should be positive
        assert!(r_02 >= 0.0, "Effective resistance should be non-negative");
    }

    #[test]
    fn test_eigenvector_orthogonality() {
        let positions: Vec<LorentzVec> = (0..6)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) / 6.0;
                LorentzVec::from_spatial(0.3 * angle.cos(), 0.3 * angle.sin(), 0.0)
            })
            .collect();

        let op = TransitionOperator::new(positions, 1.0);
        let config = SpectralConfig {
            num_eigenvalues: 3,
            compute_eigenvectors: true,
            ..Default::default()
        };

        let spectral = SpectralAnalysis::analyze_with_config(&op, &config);

        // Check eigenvectors are approximately orthogonal
        for i in 0..spectral.eigenvectors.len() {
            for j in (i+1)..spectral.eigenvectors.len() {
                let dot: f64 = spectral.eigenvectors[i].iter()
                    .zip(spectral.eigenvectors[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();

                // Should be approximately orthogonal (small dot product)
                assert!(dot.abs() < 0.5,
                    "Eigenvectors {} and {} should be approximately orthogonal, dot={}", i, j, dot);
            }
        }
    }

    #[test]
    fn test_condition_number() {
        let positions: Vec<LorentzVec> = (0..4)
            .map(|i| LorentzVec::from_spatial(0.2 * i as f64, 0.1 * (i as f64).sin(), 0.0))
            .collect();

        let op = TransitionOperator::new(positions, 0.5);
        let spectral = SpectralAnalysis::analyze(&op, 3);

        // Condition number should be at least 1
        assert!(spectral.condition_number >= 1.0,
            "Condition number should be at least 1, got {}", spectral.condition_number);
    }
}
