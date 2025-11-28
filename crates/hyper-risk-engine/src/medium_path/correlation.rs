//! Dynamic correlation tracking.
//!
//! Implements exponentially weighted correlation updates
//! similar to DCC-GARCH but optimized for real-time use.

/// Correlation matrix with exponential updates.
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// Number of assets.
    pub n: usize,
    /// Correlation values (upper triangular, row-major).
    pub values: Vec<f64>,
    /// Asset identifiers.
    pub assets: Vec<String>,
}

impl CorrelationMatrix {
    /// Create new correlation matrix.
    pub fn new(assets: Vec<String>) -> Self {
        let n = assets.len();
        let size = n * (n - 1) / 2;
        Self {
            n,
            values: vec![0.0; size],
            assets,
        }
    }

    /// Get correlation between two assets.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 1.0;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let idx = self.index(i, j);
        self.values.get(idx).cloned().unwrap_or(0.0)
    }

    /// Set correlation between two assets.
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i == j {
            return;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let idx = self.index(i, j);
        if idx < self.values.len() {
            self.values[idx] = value.clamp(-1.0, 1.0);
        }
    }

    /// Convert (i, j) to linear index.
    fn index(&self, i: usize, j: usize) -> usize {
        // Upper triangular index
        i * self.n - i * (i + 1) / 2 + j - i - 1
    }

    /// Get as full matrix (n x n).
    pub fn to_full_matrix(&self) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..self.n {
                let corr = self.get(i, j);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }
        matrix
    }
}

/// Correlation tracker configuration.
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Decay factor for exponential weighting.
    pub decay: f64,
    /// Minimum observations before calculating.
    pub min_observations: usize,
    /// Shrinkage factor toward identity.
    pub shrinkage: f64,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            decay: 0.94,
            min_observations: 20,
            shrinkage: 0.1,
        }
    }
}

/// Dynamic correlation tracker.
#[derive(Debug)]
pub struct CorrelationTracker {
    /// Configuration.
    config: CorrelationConfig,
    /// Asset identifiers.
    assets: Vec<String>,
    /// Running covariance estimates.
    covariances: Vec<f64>,
    /// Running variance estimates.
    variances: Vec<f64>,
    /// Observation count.
    count: usize,
}

impl CorrelationTracker {
    /// Create new correlation tracker.
    pub fn new(assets: Vec<String>, config: CorrelationConfig) -> Self {
        let n = assets.len();
        let cov_size = n * (n - 1) / 2;
        Self {
            config,
            assets,
            covariances: vec![0.0; cov_size],
            variances: vec![0.01; n], // Initialize with small variance
            count: 0,
        }
    }

    /// Update with new return observations.
    ///
    /// # Arguments
    /// * `returns` - Vector of returns for each asset (same order as assets)
    pub fn update(&mut self, returns: &[f64]) {
        assert_eq!(returns.len(), self.assets.len());

        let decay = self.config.decay;
        let n = self.assets.len();

        // Update variances
        for (i, &r) in returns.iter().enumerate() {
            self.variances[i] = decay * self.variances[i] + (1.0 - decay) * r * r;
        }

        // Update covariances
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let cov = returns[i] * returns[j];
                self.covariances[idx] = decay * self.covariances[idx] + (1.0 - decay) * cov;
                idx += 1;
            }
        }

        self.count += 1;
    }

    /// Get current correlation matrix.
    pub fn get_correlations(&self) -> CorrelationMatrix {
        let mut matrix = CorrelationMatrix::new(self.assets.clone());

        if self.count < self.config.min_observations {
            // Return identity-like matrix if not enough data
            return matrix;
        }

        let n = self.assets.len();
        let shrinkage = self.config.shrinkage;

        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let var_i = self.variances[i].max(1e-10);
                let var_j = self.variances[j].max(1e-10);

                // Raw correlation
                let raw_corr = self.covariances[idx] / (var_i.sqrt() * var_j.sqrt());

                // Apply shrinkage toward zero
                let shrunk_corr = (1.0 - shrinkage) * raw_corr;

                matrix.set(i, j, shrunk_corr);
                idx += 1;
            }
        }

        matrix
    }

    /// Get volatilities.
    pub fn get_volatilities(&self) -> Vec<f64> {
        self.variances.iter().map(|v| v.sqrt() * (252.0_f64).sqrt()).collect()
    }

    /// Get observation count.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get covariance matrix (for portfolio optimization).
    pub fn get_covariance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.assets.len();
        let mut matrix = vec![vec![0.0; n]; n];

        // Diagonal (variances)
        for i in 0..n {
            matrix[i][i] = self.variances[i];
        }

        // Off-diagonal (covariances)
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                matrix[i][j] = self.covariances[idx];
                matrix[j][i] = self.covariances[idx];
                idx += 1;
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_matrix() {
        let assets = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut matrix = CorrelationMatrix::new(assets);

        // Diagonal should be 1
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 1), 1.0);

        // Set and get
        matrix.set(0, 1, 0.5);
        assert!((matrix.get(0, 1) - 0.5).abs() < 1e-10);
        assert!((matrix.get(1, 0) - 0.5).abs() < 1e-10); // Symmetric
    }

    #[test]
    fn test_correlation_tracker() {
        let assets = vec!["A".to_string(), "B".to_string()];
        let config = CorrelationConfig {
            min_observations: 5,
            shrinkage: 0.0, // No shrinkage for test to get raw correlation
            decay: 0.94,
        };
        let mut tracker = CorrelationTracker::new(assets, config);

        // Add perfectly positively correlated returns
        for _ in 0..50 {
            tracker.update(&[0.02, 0.02]);
            tracker.update(&[-0.02, -0.02]);
        }

        let corr = tracker.get_correlations();
        let rho = corr.get(0, 1);

        // Should be high positive correlation (perfect correlation = 1.0)
        assert!(rho > 0.5, "Expected high positive correlation, got {}", rho);
    }

    #[test]
    fn test_covariance_matrix() {
        let assets = vec!["A".to_string(), "B".to_string()];
        let config = CorrelationConfig::default();
        let mut tracker = CorrelationTracker::new(assets, config);

        for _ in 0..50 {
            tracker.update(&[0.01, 0.02]);
        }

        let cov = tracker.get_covariance_matrix();
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);

        // Diagonal should be positive
        assert!(cov[0][0] > 0.0);
        assert!(cov[1][1] > 0.0);
    }
}
