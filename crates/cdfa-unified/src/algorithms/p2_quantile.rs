use ndarray::{Array1, ArrayView1};

/// P² Quantile Estimation Algorithm
/// 
/// The P² algorithm provides an efficient way to estimate quantiles
/// of a data stream without storing all observations.
/// 
/// Reference: Jain, R. and Chlamtac, I. (1985). The P² algorithm for 
/// dynamic calculation of quantiles and histograms without storing observations.
pub struct P2QuantileEstimator {
    p: f64,           // Desired quantile (0 <= p <= 1)
    n: [i32; 5],      // Position markers
    ns: [f64; 5],     // Desired positions
    dns: [f64; 5],    // Increments in desired positions
    q: [f64; 5],      // Heights of markers (quantile estimates)
    count: usize,     // Number of observations processed
}

impl P2QuantileEstimator {
    /// Create a new P² quantile estimator for the given quantile
    pub fn new(p: f64) -> Result<Self, &'static str> {
        if p < 0.0 || p > 1.0 {
            return Err("Quantile p must be between 0 and 1");
        }
        
        Ok(Self {
            p,
            n: [0; 5],
            ns: [0.0; 5],
            dns: [0.0; 5],
            q: [0.0; 5],
            count: 0,
        })
    }
    
    /// Add a new observation and update the quantile estimate
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        
        if self.count <= 5 {
            // Initialization phase: store first 5 observations
            self.q[self.count - 1] = x;
            
            if self.count == 5 {
                // Sort the initial observations
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                // Initialize position markers
                for i in 0..5 {
                    self.n[i] = i as i32;
                    self.ns[i] = i as f64;
                }
                
                // Initialize desired positions
                self.ns[0] = 0.0;
                self.ns[1] = 2.0 * self.p;
                self.ns[2] = 4.0 * self.p;
                self.ns[3] = 2.0 + 2.0 * self.p;
                self.ns[4] = 4.0;
                
                // Initialize increments
                self.dns[0] = 0.0;
                self.dns[1] = self.p / 2.0;
                self.dns[2] = self.p;
                self.dns[3] = (1.0 + self.p) / 2.0;
                self.dns[4] = 1.0;
            }
            return;
        }
        
        // Find cell k such that q[k] <= x < q[k+1]
        let mut k = 0;
        if x < self.q[0] {
            self.q[0] = x;
            k = 0;
        } else if x < self.q[1] {
            k = 0;
        } else if x < self.q[2] {
            k = 1;
        } else if x < self.q[3] {
            k = 2;
        } else if x <= self.q[4] {
            k = 3;
        } else {
            self.q[4] = x;
            k = 3;
        }
        
        // Increment position markers
        for i in (k + 1)..5 {
            self.n[i] += 1;
        }
        
        // Update desired positions
        for i in 0..5 {
            self.ns[i] += self.dns[i];
        }
        
        // Adjust heights of markers 1, 2, 3 if necessary
        for i in 1..4 {
            let d = self.ns[i] - self.n[i] as f64;
            
            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1) || 
               (d <= -1.0 && self.n[i - 1] - self.n[i] < -1) {
                
                let d_sign = if d >= 0.0 { 1.0 } else { -1.0 };
                
                // Try parabolic formula
                let qi_new = self.parabolic_formula(i, d_sign);
                
                // Check if the new estimate is between neighboring estimates
                if self.q[i - 1] < qi_new && qi_new < self.q[i + 1] {
                    self.q[i] = qi_new;
                } else {
                    // Use linear formula
                    self.q[i] = self.linear_formula(i, d_sign);
                }
                
                self.n[i] += d_sign as i32;
            }
        }
    }
    
    /// Parabolic formula for updating quantile estimates
    fn parabolic_formula(&self, i: usize, d: f64) -> f64 {
        let ni_minus_1 = self.n[i - 1] as f64;
        let ni = self.n[i] as f64;
        let ni_plus_1 = self.n[i + 1] as f64;
        
        let qi_minus_1 = self.q[i - 1];
        let qi = self.q[i];
        let qi_plus_1 = self.q[i + 1];
        
        qi + d / (ni_plus_1 - ni_minus_1) * (
            (ni - ni_minus_1 + d) * (qi_plus_1 - qi) / (ni_plus_1 - ni) +
            (ni_plus_1 - ni - d) * (qi - qi_minus_1) / (ni - ni_minus_1)
        )
    }
    
    /// Linear formula for updating quantile estimates
    fn linear_formula(&self, i: usize, d: f64) -> f64 {
        let j = if d > 0.0 { i + 1 } else { i - 1 };
        self.q[i] + d * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i]) as f64
    }
    
    /// Get the current quantile estimate
    pub fn quantile(&self) -> Option<f64> {
        if self.count >= 5 {
            Some(self.q[2]) // Middle marker
        } else if self.count > 0 {
            // For less than 5 observations, compute exact quantile
            let mut sorted = self.q[..self.count].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let index = (self.p * (self.count - 1) as f64).round() as usize;
            Some(sorted[index.min(self.count - 1)])
        } else {
            None
        }
    }
    
    /// Get the number of observations processed
    pub fn count(&self) -> usize {
        self.count
    }
    
    /// Reset the estimator
    pub fn reset(&mut self) {
        self.n = [0; 5];
        self.ns = [0.0; 5];
        self.dns = [0.0; 5];
        self.q = [0.0; 5];
        self.count = 0;
    }
}

/// Multiple quantile estimator using P² algorithm
pub struct P2MultiQuantileEstimator {
    estimators: Vec<P2QuantileEstimator>,
    quantiles: Vec<f64>,
}

impl P2MultiQuantileEstimator {
    /// Create a new multi-quantile estimator
    pub fn new(quantiles: Vec<f64>) -> Result<Self, &'static str> {
        if quantiles.is_empty() {
            return Err("Must specify at least one quantile");
        }
        
        for &q in &quantiles {
            if q < 0.0 || q > 1.0 {
                return Err("All quantiles must be between 0 and 1");
            }
        }
        
        let mut estimators = Vec::new();
        for &q in &quantiles {
            estimators.push(P2QuantileEstimator::new(q)?);
        }
        
        Ok(Self {
            estimators,
            quantiles: quantiles,
        })
    }
    
    /// Update all estimators with a new observation
    pub fn update(&mut self, x: f64) {
        for estimator in &mut self.estimators {
            estimator.update(x);
        }
    }
    
    /// Get all current quantile estimates
    pub fn quantiles(&self) -> Vec<Option<f64>> {
        self.estimators.iter().map(|e| e.quantile()).collect()
    }
    
    /// Get a specific quantile estimate
    pub fn quantile(&self, index: usize) -> Option<f64> {
        self.estimators.get(index)?.quantile()
    }
    
    /// Get the number of observations processed
    pub fn count(&self) -> usize {
        self.estimators.first().map(|e| e.count()).unwrap_or(0)
    }
    
    /// Reset all estimators
    pub fn reset(&mut self) {
        for estimator in &mut self.estimators {
            estimator.reset();
        }
    }
}

/// Batch quantile computation using the P² algorithm
pub struct P2BatchQuantiles;

impl P2BatchQuantiles {
    /// Compute a single quantile from a batch of data
    pub fn quantile(data: &ArrayView1<f64>, p: f64) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let mut estimator = P2QuantileEstimator::new(p)?;
        
        for &x in data.iter() {
            estimator.update(x);
        }
        
        estimator.quantile().ok_or("Failed to compute quantile")
    }
    
    /// Compute multiple quantiles from a batch of data
    pub fn quantiles(data: &ArrayView1<f64>, quantiles: &[f64]) -> Result<Vec<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if quantiles.is_empty() {
            return Err("Must specify at least one quantile");
        }
        
        let mut estimator = P2MultiQuantileEstimator::new(quantiles.to_vec())?;
        
        for &x in data.iter() {
            estimator.update(x);
        }
        
        let results = estimator.quantiles();
        let mut output = Vec::new();
        
        for result in results {
            output.push(result.ok_or("Failed to compute quantile")?);
        }
        
        Ok(output)
    }
    
    /// Compute standard quantiles (quartiles and median)
    pub fn standard_quantiles(data: &ArrayView1<f64>) -> Result<StandardQuantiles, &'static str> {
        let quantiles = Self::quantiles(data, &[0.0, 0.25, 0.5, 0.75, 1.0])?;
        
        Ok(StandardQuantiles {
            min: quantiles[0],
            q1: quantiles[1],
            median: quantiles[2],
            q3: quantiles[3],
            max: quantiles[4],
        })
    }
    
    /// Compute rolling quantiles
    pub fn rolling_quantile(
        data: &ArrayView1<f64>, 
        window: usize, 
        p: f64
    ) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if window == 0 {
            return Err("Window size must be positive");
        }
        
        let n = data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i >= window { i - window + 1 } else { 0 };
            let window_data = data.slice(ndarray::s![start..=i]);
            result[i] = Self::quantile(&window_data, p)?;
        }
        
        Ok(result)
    }
    
    /// Compute expanding quantiles
    pub fn expanding_quantile(data: &ArrayView1<f64>, p: f64) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let n = data.len();
        let mut result = Array1::zeros(n);
        let mut estimator = P2QuantileEstimator::new(p)?;
        
        for i in 0..n {
            estimator.update(data[i]);
            result[i] = estimator.quantile().unwrap_or(data[i]);
        }
        
        Ok(result)
    }
}

/// Standard quantile results
#[derive(Debug, Clone)]
pub struct StandardQuantiles {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
}

impl StandardQuantiles {
    /// Calculate interquartile range
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }
    
    /// Calculate range
    pub fn range(&self) -> f64 {
        self.max - self.min
    }
    
    /// Detect outliers using IQR method
    pub fn outlier_bounds(&self) -> (f64, f64) {
        let iqr = self.iqr();
        let lower_bound = self.q1 - 1.5 * iqr;
        let upper_bound = self.q3 + 1.5 * iqr;
        (lower_bound, upper_bound)
    }
    
    /// Check if a value is an outlier
    pub fn is_outlier(&self, value: f64) -> bool {
        let (lower, upper) = self.outlier_bounds();
        value < lower || value > upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_p2_single_quantile() {
        let mut estimator = P2QuantileEstimator::new(0.5).unwrap(); // Median
        
        // Add some data points
        for i in 1..=10 {
            estimator.update(i as f64);
        }
        
        let median = estimator.quantile().unwrap();
        
        // For data 1..10, median should be around 5.5
        assert_abs_diff_eq!(median, 5.5, epsilon = 1.0);
        assert_eq!(estimator.count(), 10);
    }
    
    #[test]
    fn test_p2_multiple_quantiles() {
        let mut estimator = P2MultiQuantileEstimator::new(vec![0.25, 0.5, 0.75]).unwrap();
        
        // Add data points
        for i in 1..=100 {
            estimator.update(i as f64);
        }
        
        let quantiles = estimator.quantiles();
        
        // Check that we got results for all quantiles
        assert_eq!(quantiles.len(), 3);
        assert!(quantiles[0].is_some()); // Q1
        assert!(quantiles[1].is_some()); // Median
        assert!(quantiles[2].is_some()); // Q3
        
        // Check approximate values
        let q1 = quantiles[0].unwrap();
        let median = quantiles[1].unwrap();
        let q3 = quantiles[2].unwrap();
        
        assert!(q1 < median);
        assert!(median < q3);
        assert_abs_diff_eq!(median, 50.5, epsilon = 5.0);
    }
    
    #[test]
    fn test_p2_batch_quantile() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let median = P2BatchQuantiles::quantile(&data.view(), 0.5).unwrap();
        
        assert_abs_diff_eq!(median, 5.5, epsilon = 1.0);
    }
    
    #[test]
    fn test_p2_batch_multiple_quantiles() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let quantiles = P2BatchQuantiles::quantiles(&data.view(), &[0.0, 0.25, 0.5, 0.75, 1.0]).unwrap();
        
        assert_eq!(quantiles.len(), 5);
        assert_eq!(quantiles[0], 1.0); // Min
        assert_eq!(quantiles[4], 10.0); // Max
        assert!(quantiles[1] < quantiles[2]); // Q1 < Median
        assert!(quantiles[2] < quantiles[3]); // Median < Q3
    }
    
    #[test]
    fn test_p2_standard_quantiles() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sq = P2BatchQuantiles::standard_quantiles(&data.view()).unwrap();
        
        assert_eq!(sq.min, 1.0);
        assert_eq!(sq.max, 10.0);
        assert_eq!(sq.range(), 9.0);
        assert!(sq.iqr() > 0.0);
        
        // Test outlier detection
        let (_lower, _upper) = sq.outlier_bounds();
        assert!(!sq.is_outlier(5.0)); // Should not be outlier
        assert!(sq.is_outlier(-10.0)); // Should be outlier
        assert!(sq.is_outlier(50.0)); // Should be outlier
    }
    
    #[test]
    fn test_p2_rolling_quantile() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rolling_median = P2BatchQuantiles::rolling_quantile(&data.view(), 3, 0.5).unwrap();
        
        assert_eq!(rolling_median.len(), data.len());
        
        // First few values should increase
        assert!(rolling_median[0] <= rolling_median[1]);
        assert!(rolling_median[1] <= rolling_median[2]);
    }
    
    #[test]
    fn test_p2_expanding_quantile() {
        let data = array![5.0, 1.0, 3.0, 9.0, 2.0];
        let expanding_median = P2BatchQuantiles::expanding_quantile(&data.view(), 0.5).unwrap();
        
        assert_eq!(expanding_median.len(), data.len());
        
        // First value should be the first data point
        assert_eq!(expanding_median[0], 5.0);
    }
    
    #[test]
    fn test_p2_estimator_reset() {
        let mut estimator = P2QuantileEstimator::new(0.5).unwrap();
        
        // Add some data
        for i in 1..=5 {
            estimator.update(i as f64);
        }
        
        assert_eq!(estimator.count(), 5);
        assert!(estimator.quantile().is_some());
        
        // Reset and check
        estimator.reset();
        assert_eq!(estimator.count(), 0);
        assert!(estimator.quantile().is_none());
    }
    
    #[test]
    fn test_p2_edge_cases() {
        // Test with single value
        let mut estimator = P2QuantileEstimator::new(0.5).unwrap();
        estimator.update(42.0);
        assert_eq!(estimator.quantile().unwrap(), 42.0);
        
        // Test with extreme quantiles
        let mut estimator_min = P2QuantileEstimator::new(0.0).unwrap();
        let mut estimator_max = P2QuantileEstimator::new(1.0).unwrap();
        
        for i in 1..=10 {
            estimator_min.update(i as f64);
            estimator_max.update(i as f64);
        }
        
        assert_abs_diff_eq!(estimator_min.quantile().unwrap(), 1.0, epsilon = 1.0);
        assert_abs_diff_eq!(estimator_max.quantile().unwrap(), 10.0, epsilon = 1.0);
    }
    
    #[test]
    fn test_p2_invalid_inputs() {
        // Invalid quantile values
        assert!(P2QuantileEstimator::new(-0.1).is_err());
        assert!(P2QuantileEstimator::new(1.1).is_err());
        
        // Empty data
        let empty_data = Array1::<f64>::zeros(0);
        assert!(P2BatchQuantiles::quantile(&empty_data.view(), 0.5).is_err());
        
        // Empty quantiles list
        assert!(P2MultiQuantileEstimator::new(vec![]).is_err());
    }
    
    #[test]
    fn test_standard_quantiles_methods() {
        let sq = StandardQuantiles {
            min: 1.0,
            q1: 3.0,
            median: 5.0,
            q3: 7.0,
            max: 10.0,
        };
        
        assert_eq!(sq.iqr(), 4.0); // 7 - 3
        assert_eq!(sq.range(), 9.0); // 10 - 1
        
        let (lower, upper) = sq.outlier_bounds();
        assert_eq!(lower, 3.0 - 1.5 * 4.0); // Q1 - 1.5*IQR
        assert_eq!(upper, 7.0 + 1.5 * 4.0); // Q3 + 1.5*IQR
        
        assert!(!sq.is_outlier(5.0)); // Normal value
        assert!(sq.is_outlier(-10.0)); // Below lower bound
        assert!(sq.is_outlier(20.0)); // Above upper bound
    }
}