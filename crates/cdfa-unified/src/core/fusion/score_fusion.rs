//! Score-based fusion algorithms
//! 
//! This module implements various methods for combining numerical scores
//! from multiple sources into a consensus result.

use crate::error::{CdfaError, Result};
use crate::types::*;
use crate::traits::FusionMethod;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Score-based fusion methods
pub struct ScoreFusion;

impl ScoreFusion {
    /// Simple average fusion
    pub fn average(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        Ok(data.mean_axis(Axis(0)).unwrap())
    }
    
    /// Weighted average fusion
    pub fn weighted_average(
        data: &ArrayView2<Float>,
        weights: &ArrayView1<Float>,
    ) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        if data.nrows() != weights.len() {
            return Err(CdfaError::dimension_mismatch(data.nrows(), weights.len()));
        }
        
        // Validate weights
        let weight_sum = weights.sum();
        if weight_sum <= 0.0 {
            return Err(CdfaError::invalid_input("Weights must sum to positive value"));
        }
        
        // Normalize weights
        let normalized_weights = weights / weight_sum;
        
        let mut result = Array1::zeros(data.ncols());
        for (i, &weight) in normalized_weights.iter().enumerate() {
            result = result + weight * &data.row(i);
        }
        
        Ok(result)
    }
    
    /// Normalized average (each source normalized to [0,1] first)
    pub fn normalized_average(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let mut normalized_data = Array2::zeros(data.dim());
        
        for (i, row) in data.rows().into_iter().enumerate() {
            let min_val = row.fold(Float::INFINITY, |acc, &x| acc.min(x));
            let max_val = row.fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));
            
            let range = max_val - min_val;
            if range > Float::EPSILON {
                for (j, &val) in row.iter().enumerate() {
                    normalized_data[[i, j]] = (val - min_val) / range;
                }
            } else {
                // Constant values, set to 0.5
                for j in 0..row.len() {
                    normalized_data[[i, j]] = 0.5;
                }
            }
        }
        
        Self::average(&normalized_data.view())
    }
    
    /// Standardized average (each source standardized to zero mean, unit variance)
    pub fn standardized_average(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let mut standardized_data = Array2::zeros(data.dim());
        
        for (i, row) in data.rows().into_iter().enumerate() {
            let mean = row.mean().unwrap();
            let std = row.std(0.0);
            
            if std > Float::EPSILON {
                for (j, &val) in row.iter().enumerate() {
                    standardized_data[[i, j]] = (val - mean) / std;
                }
            } else {
                // Constant values, set to 0
                for j in 0..row.len() {
                    standardized_data[[i, j]] = 0.0;
                }
            }
        }
        
        Self::average(&standardized_data.view())
    }
    
    /// Maximum fusion (element-wise maximum)
    pub fn maximum(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let mut result = Array1::from_elem(data.ncols(), Float::NEG_INFINITY);
        
        for row in data.rows() {
            for (j, &val) in row.iter().enumerate() {
                if val > result[j] {
                    result[j] = val;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Minimum fusion (element-wise minimum)
    pub fn minimum(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let mut result = Array1::from_elem(data.ncols(), Float::INFINITY);
        
        for row in data.rows() {
            for (j, &val) in row.iter().enumerate() {
                if val < result[j] {
                    result[j] = val;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Median fusion (element-wise median)
    pub fn median(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let n_items = data.ncols();
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let mut column_values: Vec<Float> = data.column(j).to_vec();
            column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let n = column_values.len();
            result[j] = if n % 2 == 0 {
                (column_values[n / 2 - 1] + column_values[n / 2]) / 2.0
            } else {
                column_values[n / 2]
            };
        }
        
        Ok(result)
    }
    
    /// Trimmed mean fusion (remove extreme values before averaging)
    pub fn trimmed_mean(data: &ArrayView2<Float>, trim_percent: Float) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        if !(0.0..=0.5).contains(&trim_percent) {
            return Err(CdfaError::invalid_input("Trim percent must be between 0.0 and 0.5"));
        }
        
        let n_items = data.ncols();
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let mut column_values: Vec<Float> = data.column(j).to_vec();
            column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let n = column_values.len();
            let trim_count = ((n as Float) * trim_percent).floor() as usize;
            
            if trim_count * 2 >= n {
                // If trimming would remove all data, use median
                result[j] = if n % 2 == 0 {
                    (column_values[n / 2 - 1] + column_values[n / 2]) / 2.0
                } else {
                    column_values[n / 2]
                };
            } else {
                let start = trim_count;
                let end = n - trim_count;
                let trimmed_sum: Float = column_values[start..end].iter().sum();
                result[j] = trimmed_sum / (end - start) as Float;
            }
        }
        
        Ok(result)
    }
    
    /// CombSUM fusion (sum of scores)
    pub fn comb_sum(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        Ok(data.sum_axis(Axis(0)))
    }
    
    /// CombMNZ fusion (sum of scores weighted by number of non-zero sources)
    pub fn comb_mnz(data: &ArrayView2<Float>, threshold: Float) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let n_items = data.ncols();
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let column = data.column(j);
            let sum: Float = column.sum();
            let nonzero_count = column.iter().filter(|&&x| x > threshold).count() as Float;
            
            result[j] = if nonzero_count > 0.0 {
                sum * nonzero_count
            } else {
                0.0
            };
        }
        
        Ok(result)
    }
    
    /// ISR (Inverse Square Rank) fusion
    pub fn isr_fusion(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        // Convert scores to ranks, then apply ISR weighting
        let rankings = crate::core::fusion::rank_fusion::scores_to_rankings(data)?;
        let n_items = rankings.ncols();
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let column = rankings.column(j);
            let isr_score: Float = column.iter()
                .map(|&rank| if rank > 0.0 { 1.0 / (rank * rank) } else { 0.0 })
                .sum();
            result[j] = isr_score;
        }
        
        Ok(result)
    }
    
    /// Geometric mean fusion
    pub fn geometric_mean(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let n_sources = data.nrows() as Float;
        let n_items = data.ncols();
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let column = data.column(j);
            
            // Check for non-positive values
            if column.iter().any(|&x| x <= 0.0) {
                result[j] = 0.0; // Geometric mean undefined for non-positive values
            } else {
                let log_sum: Float = column.iter().map(|&x| x.ln()).sum();
                result[j] = (log_sum / n_sources).exp();
            }
        }
        
        Ok(result)
    }
    
    /// Harmonic mean fusion
    pub fn harmonic_mean(data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        let n_sources = data.nrows() as Float;
        let n_items = data.ncols();
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let column = data.column(j);
            
            // Check for zero values
            if column.iter().any(|&x| x <= 0.0) {
                result[j] = 0.0; // Harmonic mean undefined for non-positive values
            } else {
                let reciprocal_sum: Float = column.iter().map(|&x| 1.0 / x).sum();
                result[j] = n_sources / reciprocal_sum;
            }
        }
        
        Ok(result)
    }
}

/// Adaptive score fusion that selects the best method based on data characteristics
pub struct AdaptiveScoreFusion {
    // Configuration parameters
    diversity_threshold: Float,
    consensus_threshold: Float,
}

impl AdaptiveScoreFusion {
    /// Create new adaptive fusion with default parameters
    pub fn new() -> Self {
        Self {
            diversity_threshold: 0.5,
            consensus_threshold: 0.7,
        }
    }
    
    /// Create adaptive fusion with custom parameters
    pub fn with_thresholds(diversity_threshold: Float, consensus_threshold: Float) -> Self {
        Self {
            diversity_threshold,
            consensus_threshold,
        }
    }
    
    /// Perform adaptive fusion
    pub fn fuse(&self, data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        // Analyze data characteristics
        let characteristics = self.analyze_data_characteristics(data)?;
        
        // Select fusion method based on characteristics
        let method = self.select_fusion_method(&characteristics);
        
        // Apply selected method
        match method {
            AdaptiveFusionMethod::RobustMedian => ScoreFusion::median(data),
            AdaptiveFusionMethod::TrimmedMean => ScoreFusion::trimmed_mean(data, 0.1),
            AdaptiveFusionMethod::StandardizedAverage => ScoreFusion::standardized_average(data),
            AdaptiveFusionMethod::SimpleAverage => ScoreFusion::average(data),
        }
    }
    
    /// Analyze data characteristics to guide method selection
    fn analyze_data_characteristics(&self, data: &ArrayView2<Float>) -> Result<DataCharacteristics> {
        let n_sources = data.nrows();
        let n_items = data.ncols();
        
        // Calculate diversity (average correlation between sources)
        let mut correlations = Vec::new();
        for i in 0..n_sources {
            for j in i + 1..n_sources {
                let corr = crate::core::diversity::pearson_correlation_fast(
                    &data.row(i),
                    &data.row(j),
                )?;
                correlations.push(corr.abs());
            }
        }
        
        let diversity = if correlations.is_empty() {
            1.0
        } else {
            correlations.iter().sum::<Float>() / correlations.len() as Float
        };
        
        // Calculate consensus (agreement across sources)
        let consensus = 1.0 - diversity; // Simple inverse relationship
        
        // Detect outliers (sources with very different distributions)
        let mut outlier_ratio = 0.0;
        for i in 0..n_sources {
            let row = data.row(i);
            let mean = row.mean().unwrap();
            let std = row.std(0.0);
            
            // Check if this source deviates significantly from others
            let mut is_outlier = false;
            for j in 0..n_sources {
                if i != j {
                    let other_row = data.row(j);
                    let other_mean = other_row.mean().unwrap();
                    if (mean - other_mean).abs() > 2.0 * std {
                        is_outlier = true;
                        break;
                    }
                }
            }
            
            if is_outlier {
                outlier_ratio += 1.0 / n_sources as Float;
            }
        }
        
        Ok(DataCharacteristics {
            diversity,
            consensus,
            outlier_ratio,
            n_sources,
            n_items,
        })
    }
    
    /// Select fusion method based on data characteristics
    fn select_fusion_method(&self, characteristics: &DataCharacteristics) -> AdaptiveFusionMethod {
        // Decision logic based on data characteristics
        if characteristics.outlier_ratio > 0.3 {
            // High outlier presence - use robust methods
            AdaptiveFusionMethod::RobustMedian
        } else if characteristics.outlier_ratio > 0.1 {
            // Moderate outliers - use trimmed mean
            AdaptiveFusionMethod::TrimmedMean
        } else if characteristics.diversity < self.diversity_threshold {
            // Low diversity (high agreement) - use standardized average
            AdaptiveFusionMethod::StandardizedAverage
        } else {
            // Default case - use simple average
            AdaptiveFusionMethod::SimpleAverage
        }
    }
}

impl Default for AdaptiveScoreFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// Data characteristics for adaptive fusion
#[derive(Debug, Clone)]
struct DataCharacteristics {
    diversity: Float,
    consensus: Float,
    outlier_ratio: Float,
    n_sources: usize,
    n_items: usize,
}

/// Adaptive fusion methods
#[derive(Debug, Clone, Copy)]
enum AdaptiveFusionMethod {
    RobustMedian,
    TrimmedMean,
    StandardizedAverage,
    SimpleAverage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_average_fusion() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ];
        
        let result = ScoreFusion::average(&data.view()).unwrap();
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 4.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_weighted_average_fusion() {
        let data = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let weights = array![0.7, 0.3];
        
        let result = ScoreFusion::weighted_average(&data.view(), &weights.view()).unwrap();
        assert_abs_diff_eq!(result[0], 0.7, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.3, epsilon = 1e-10);
    }
    
    #[test]
    fn test_normalized_average() {
        let data = array![
            [0.0, 5.0, 10.0],  // Range [0, 10] -> [0, 0.5, 1]
            [10.0, 15.0, 20.0] // Range [10, 20] -> [0, 0.5, 1]
        ];
        
        let result = ScoreFusion::normalized_average(&data.view()).unwrap();
        // Average of normalized values should be [0, 0.5, 1]
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_median_fusion() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [10.0, 11.0, 12.0] // Outlier row
        ];
        
        let result = ScoreFusion::median(&data.view()).unwrap();
        // Median should be robust to the outlier
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 4.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_trimmed_mean() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [100.0, 101.0, 102.0] // Outlier
        ];
        
        let result = ScoreFusion::trimmed_mean(&data.view(), 0.25).unwrap();
        // With 25% trimming, outlier should be removed
        // Remaining values: [1,2,3], [2,3,4], [3,4,5]
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 4.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_geometric_mean() {
        let data = array![
            [1.0, 4.0, 9.0],
            [4.0, 9.0, 16.0]
        ];
        
        let result = ScoreFusion::geometric_mean(&data.view()).unwrap();
        // Geometric mean of [1,4] = 2, [4,9] = 6, [9,16] = 12
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 12.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_harmonic_mean() {
        let data = array![
            [1.0, 2.0, 4.0],
            [4.0, 4.0, 4.0]
        ];
        
        let result = ScoreFusion::harmonic_mean(&data.view()).unwrap();
        // Harmonic mean of [1,4] = 1.6, [2,4] = 2.67, [4,4] = 4
        assert_abs_diff_eq!(result[0], 1.6, epsilon = 1e-10);
        assert!(result[1] > 2.6 && result[1] < 2.7);
        assert_abs_diff_eq!(result[2], 4.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_adaptive_fusion() {
        let adaptive = AdaptiveScoreFusion::new();
        
        // Test with low diversity data (should use standardized average)
        let low_diversity_data = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9]
        ];
        
        let result = adaptive.fuse(&low_diversity_data.view()).unwrap();
        assert_eq!(result.len(), 3);
        
        // Test with high diversity data
        let high_diversity_data = array![
            [1.0, 10.0, 100.0],
            [0.1, 1.0, 10.0],
            [100.0, 1000.0, 10000.0]
        ];
        
        let result2 = adaptive.fuse(&high_diversity_data.view()).unwrap();
        assert_eq!(result2.len(), 3);
    }
    
    #[test]
    fn test_input_validation() {
        let empty_data = Array2::<Float>::zeros((0, 0));
        assert!(ScoreFusion::average(&empty_data.view()).is_err());
        
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let wrong_weights = array![0.5]; // Wrong dimension
        assert!(ScoreFusion::weighted_average(&data.view(), &wrong_weights.view()).is_err());
        
        let _negative_weights = array![-0.5, 0.3];
        // This should still work as we normalize weights, but sum should be positive
        let zero_weights = array![0.0, 0.0];
        assert!(ScoreFusion::weighted_average(&data.view(), &zero_weights.view()).is_err());
    }
}

/// Simple wrapper for weighted average fusion
pub struct WeightedAverageFusion;

impl WeightedAverageFusion {
    pub fn new() -> Self {
        Self
    }
    
    pub fn fuse_weighted(&self, data: &ArrayView2<Float>, weights: &ArrayView1<Float>) -> Result<Array1<Float>> {
        ScoreFusion::weighted_average(data, weights)
    }
}

impl Default for WeightedAverageFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionMethod for WeightedAverageFusion {
    fn fuse(&self, data: &FloatArrayView2) -> crate::error::Result<FloatArray1> {
        // For trait implementation, use equal weights
        let weights = Array1::from_elem(data.nrows(), 1.0 / data.nrows() as Float);
        ScoreFusion::weighted_average(data, &weights.view())
    }
    
    fn name(&self) -> &'static str {
        "weighted_average_fusion"
    }
}