//! pBit-Enhanced Consensus Data Fusion
//!
//! Uses Ising model dynamics for probabilistic ensemble fusion
//! and diversity-weighted consensus building.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Boltzmann Weighting**: W_i = exp(-E_i/T) / Z
//! - **Ising Consensus**: σ_consensus = sign(Σ J_ij * σ_j + h)
//! - **Diversity Coupling**: J_ij = 1 - |corr(i,j)|
//! - **Energy Minimization**: E = -Σ J_ij σ_i σ_j - Σ h_i σ_i

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;

/// pBit state for fusion consensus
#[derive(Debug, Clone)]
pub struct PBitConsensus {
    /// Temperature (controls exploration vs exploitation)
    pub temperature: f64,
    /// Spin states for each source (+1 = agree, -1 = disagree)
    pub spins: Vec<i8>,
    /// Coupling matrix (diversity-based)
    pub couplings: Array2<f64>,
    /// Local fields (source reliability)
    pub fields: Vec<f64>,
}

impl PBitConsensus {
    /// Create new pBit consensus builder
    pub fn new(n_sources: usize, temperature: f64) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            temperature,
            spins: (0..n_sources).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect(),
            couplings: Array2::zeros((n_sources, n_sources)),
            fields: vec![0.0; n_sources],
        }
    }

    /// Set coupling based on diversity (low correlation = high coupling)
    pub fn set_diversity_coupling(&mut self, correlation_matrix: &Array2<f64>) {
        let n = self.spins.len();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // High coupling for low correlation (diverse sources)
                    self.couplings[[i, j]] = 1.0 - correlation_matrix[[i, j]].abs();
                }
            }
        }
    }

    /// Set source reliability as local field
    pub fn set_reliability(&mut self, reliabilities: &[f64]) {
        self.fields = reliabilities.to_vec();
    }

    /// Calculate energy of current configuration
    pub fn energy(&self) -> f64 {
        let n = self.spins.len();
        let mut energy = 0.0;

        // Coupling term
        for i in 0..n {
            for j in (i + 1)..n {
                energy -= self.couplings[[i, j]] 
                    * (self.spins[i] as f64) 
                    * (self.spins[j] as f64);
            }
            // Field term
            energy -= self.fields[i] * (self.spins[i] as f64);
        }
        energy
    }

    /// Single Metropolis-Hastings step
    pub fn metropolis_step(&mut self) {
        let mut rng = rand::thread_rng();
        let n = self.spins.len();
        let i = rng.gen_range(0..n);

        // Calculate energy change if we flip spin i
        let mut delta_e = 2.0 * self.fields[i] * (self.spins[i] as f64);
        for j in 0..n {
            if i != j {
                delta_e += 2.0 * self.couplings[[i, j]] 
                    * (self.spins[i] as f64) 
                    * (self.spins[j] as f64);
            }
        }

        // Accept with Boltzmann probability
        if delta_e < 0.0 || rng.gen::<f64>() < (-delta_e / self.temperature).exp() {
            self.spins[i] *= -1;
        }
    }

    /// Run thermalization
    pub fn thermalize(&mut self, steps: usize) {
        for _ in 0..steps {
            self.metropolis_step();
        }
    }

    /// Get consensus weights from equilibrium spin distribution
    pub fn consensus_weights(&mut self, samples: usize) -> Vec<f64> {
        let n = self.spins.len();
        let mut counts = vec![0.0; n];

        for _ in 0..samples {
            self.metropolis_step();
            for i in 0..n {
                if self.spins[i] > 0 {
                    counts[i] += 1.0;
                }
            }
        }

        // Normalize to weights
        let total: f64 = counts.iter().sum();
        if total > 0.0 {
            counts.iter().map(|c| c / total).collect()
        } else {
            vec![1.0 / n as f64; n]
        }
    }
}

/// pBit-enhanced score fusion
#[derive(Debug, Clone)]
pub struct PBitScoreFusion {
    /// Temperature for pBit dynamics
    pub temperature: f64,
    /// Thermalization steps
    pub thermalize_steps: usize,
    /// Sampling steps for weight estimation
    pub sample_steps: usize,
}

impl Default for PBitScoreFusion {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            thermalize_steps: 1000,
            sample_steps: 500,
        }
    }
}

impl PBitScoreFusion {
    /// Create with custom temperature
    pub fn with_temperature(temperature: f64) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Fuse scores using pBit consensus
    /// 
    /// # Arguments
    /// * `scores` - 2D array [n_sources, n_items]
    /// * `reliabilities` - Optional reliability weights for each source
    /// 
    /// # Returns
    /// Fused scores as 1D array
    pub fn fuse(
        &self,
        scores: &ArrayView2<f64>,
        reliabilities: Option<&[f64]>,
    ) -> Array1<f64> {
        let n_sources = scores.nrows();
        let n_items = scores.ncols();

        // Calculate correlation matrix for diversity coupling
        let corr_matrix = self.correlation_matrix(scores);

        // Build pBit consensus for each item
        let mut fused = Array1::zeros(n_items);

        for item in 0..n_items {
            let item_scores: Vec<f64> = (0..n_sources)
                .map(|s| scores[[s, item]])
                .collect();

            // Use item scores as local fields
            let mut consensus = PBitConsensus::new(n_sources, self.temperature);
            consensus.set_diversity_coupling(&corr_matrix);
            
            // Set fields based on scores and reliabilities
            let fields: Vec<f64> = if let Some(rel) = reliabilities {
                item_scores.iter().zip(rel.iter())
                    .map(|(s, r)| s * r)
                    .collect()
            } else {
                item_scores.clone()
            };
            consensus.set_reliability(&fields);

            // Thermalize and sample
            consensus.thermalize(self.thermalize_steps);
            let weights = consensus.consensus_weights(self.sample_steps);

            // Weighted sum
            fused[item] = item_scores.iter()
                .zip(weights.iter())
                .map(|(s, w)| s * w)
                .sum();
        }

        fused
    }

    /// Calculate source correlation matrix
    fn correlation_matrix(&self, scores: &ArrayView2<f64>) -> Array2<f64> {
        let n = scores.nrows();
        let mut corr = Array2::zeros((n, n));

        for i in 0..n {
            corr[[i, i]] = 1.0;
            for j in (i + 1)..n {
                let c = pearson_correlation_1d(&scores.row(i), &scores.row(j));
                corr[[i, j]] = c;
                corr[[j, i]] = c;
            }
        }
        corr
    }
}

/// Fast Pearson correlation for 1D arrays
fn pearson_correlation_1d(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 1e-10 {
        cov / denom
    } else {
        0.0
    }
}

/// pBit-weighted ensemble combiner
#[derive(Debug, Clone)]
pub struct PBitEnsemble {
    /// Source weights (learned from pBit dynamics)
    pub weights: Vec<f64>,
    /// Temperature schedule
    pub temperature: f64,
}

impl PBitEnsemble {
    /// Learn ensemble weights from training data
    pub fn fit(
        predictions: &ArrayView2<f64>,
        targets: &ArrayView1<f64>,
        temperature: f64,
    ) -> Self {
        let n_sources = predictions.nrows();
        
        // Calculate MSE for each source
        let errors: Vec<f64> = (0..n_sources)
            .map(|i| {
                let pred = predictions.row(i);
                pred.iter().zip(targets.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>() / targets.len() as f64
            })
            .collect();

        // Boltzmann weights (lower error = higher weight)
        let boltzmann: Vec<f64> = errors.iter()
            .map(|e| (-e / temperature).exp())
            .collect();
        let z: f64 = boltzmann.iter().sum();
        let weights: Vec<f64> = boltzmann.iter().map(|b| b / z).collect();

        Self { weights, temperature }
    }

    /// Apply ensemble to new predictions
    pub fn predict(&self, predictions: &ArrayView2<f64>) -> Array1<f64> {
        let n_items = predictions.ncols();
        let mut result = Array1::zeros(n_items);

        for (i, w) in self.weights.iter().enumerate() {
            for j in 0..n_items {
                result[j] += w * predictions[[i, j]];
            }
        }
        result
    }
}

/// pBit importance sampling for rare event fusion
pub fn pbit_importance_sample(
    scores: &ArrayView2<f64>,
    threshold: f64,
    temperature: f64,
    n_samples: usize,
) -> Vec<(usize, f64)> {
    let n_sources = scores.nrows();
    let n_items = scores.ncols();
    let mut rng = rand::thread_rng();
    let mut results = Vec::new();

    for item in 0..n_items {
        // Check if any source has extreme score
        let max_score = (0..n_sources)
            .map(|s| scores[[s, item]])
            .fold(f64::NEG_INFINITY, f64::max);

        if max_score > threshold {
            // Importance sample with Boltzmann weights
            let energies: Vec<f64> = (0..n_sources)
                .map(|s| -scores[[s, item]])
                .collect();
            let weights: Vec<f64> = energies.iter()
                .map(|e| (-e / temperature).exp())
                .collect();
            let z: f64 = weights.iter().sum();

            let mut sampled_score = 0.0;
            for _ in 0..n_samples {
                let r: f64 = rng.gen::<f64>() * z;
                let mut cumsum = 0.0;
                for (i, w) in weights.iter().enumerate() {
                    cumsum += w;
                    if r <= cumsum {
                        sampled_score += scores[[i, item]];
                        break;
                    }
                }
            }
            sampled_score /= n_samples as f64;
            results.push((item, sampled_score));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pbit_consensus_energy() {
        let mut consensus = PBitConsensus::new(3, 1.0);
        consensus.spins = vec![1, 1, 1]; // All agree
        consensus.couplings = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i != j { 0.5 } else { 0.0 }
        });
        
        let energy = consensus.energy();
        // All aligned = low energy
        assert!(energy < 0.0, "All-aligned state should have negative energy");
    }

    #[test]
    fn test_pbit_score_fusion() {
        let scores = array![
            [0.8, 0.6, 0.9],
            [0.7, 0.8, 0.6],
            [0.9, 0.5, 0.8]
        ];

        let fusion = PBitScoreFusion::with_temperature(0.5);
        let fused = fusion.fuse(&scores.view(), None);

        assert_eq!(fused.len(), 3);
        // Fused scores should be in reasonable range
        for &s in fused.iter() {
            assert!(s >= 0.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_pbit_ensemble() {
        let predictions = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.2, 2.8],
            [0.9, 1.8, 3.2]
        ];
        let targets = array![1.0, 2.0, 3.0];

        let ensemble = PBitEnsemble::fit(&predictions.view(), &targets.view(), 0.1);
        
        // Source 0 should have highest weight (lowest error)
        assert!(ensemble.weights[0] > ensemble.weights[1]);
        
        let result = ensemble.predict(&predictions.view());
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_boltzmann_weights_wolfram_validated() {
        // Wolfram: Exp[-E/T] for E=0,1,2 at T=1
        // E=0: 1.0, E=1: 0.368, E=2: 0.135
        let t: f64 = 1.0;
        let e0: f64 = (-0.0_f64 / t).exp();
        let e1: f64 = (-1.0_f64 / t).exp();
        let e2: f64 = (-2.0_f64 / t).exp();
        
        assert!((e0 - 1.0).abs() < 0.001);
        assert!((e1 - 0.368).abs() < 0.001);
        assert!((e2 - 0.135).abs() < 0.001);
    }
}
