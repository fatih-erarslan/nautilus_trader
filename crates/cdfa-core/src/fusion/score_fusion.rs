use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Score-based fusion methods for combining multiple rankings or predictions
pub struct ScoreFusion;

impl ScoreFusion {
    /// Simple average fusion
    /// 
    /// Combines scores by taking the arithmetic mean
    pub fn average(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        if scores.is_empty() {
            return Err("Score matrix cannot be empty");
        }
        
        Ok(scores.mean_axis(Axis(0)).unwrap())
    }
    
    /// Weighted average fusion
    /// 
    /// Combines scores using provided weights for each source
    pub fn weighted_average(scores: &ArrayView2<f64>, weights: &ArrayView1<f64>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = scores.dim();
        
        if weights.len() != n_sources {
            return Err("Number of weights must match number of sources");
        }
        
        if weights.sum() == 0.0 {
            return Err("Weights cannot all be zero");
        }
        
        // Normalize weights
        let normalized_weights = weights / weights.sum();
        
        // Compute weighted sum
        let mut result = Array1::zeros(n_items);
        for (i, weight) in normalized_weights.iter().enumerate() {
            let row = scores.row(i);
            for j in 0..n_items {
                result[j] += *weight * row[j];
            }
        }
        
        Ok(result)
    }
    
    /// Min-max normalization followed by average fusion
    /// 
    /// Normalizes each source's scores to [0, 1] before averaging
    pub fn normalized_average(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = scores.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Score matrix cannot be empty");
        }
        
        let mut normalized_scores = Array2::zeros((n_sources, n_items));
        
        // Normalize each source
        for i in 0..n_sources {
            let row = scores.row(i);
            let min_val = row.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            if (max_val - min_val).abs() < f64::EPSILON {
                // All values are the same
                normalized_scores.row_mut(i).fill(0.5);
            } else {
                for j in 0..n_items {
                    normalized_scores[[i, j]] = (scores[[i, j]] - min_val) / (max_val - min_val);
                }
            }
        }
        
        Self::average(&normalized_scores.view())
    }
    
    /// Z-score normalization followed by average fusion
    /// 
    /// Standardizes each source's scores to have mean 0 and std 1
    pub fn standardized_average(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = scores.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Score matrix cannot be empty");
        }
        
        let mut standardized_scores = Array2::zeros((n_sources, n_items));
        
        // Standardize each source
        for i in 0..n_sources {
            let row = scores.row(i);
            let mean = row.mean().unwrap();
            let std = row.std(0.0);
            
            if std < f64::EPSILON {
                // Zero variance
                standardized_scores.row_mut(i).fill(0.0);
            } else {
                for j in 0..n_items {
                    standardized_scores[[i, j]] = (scores[[i, j]] - mean) / std;
                }
            }
        }
        
        Self::average(&standardized_scores.view())
    }
    
    /// Maximum score fusion
    /// 
    /// Takes the maximum score across all sources for each item
    pub fn maximum(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        if scores.is_empty() {
            return Err("Score matrix cannot be empty");
        }
        
        let n_items = scores.dim().1;
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            result[j] = scores.column(j).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }
        
        Ok(result)
    }
    
    /// Minimum score fusion
    /// 
    /// Takes the minimum score across all sources for each item
    pub fn minimum(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        if scores.is_empty() {
            return Err("Score matrix cannot be empty");
        }
        
        let n_items = scores.dim().1;
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            result[j] = scores.column(j).fold(f64::INFINITY, |a, &b| a.min(b));
        }
        
        Ok(result)
    }
    
    /// Median score fusion
    /// 
    /// Takes the median score across all sources for each item
    pub fn median(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = scores.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Score matrix cannot be empty");
        }
        
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let mut column_values: Vec<f64> = scores.column(j).to_vec();
            column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let median_val = if n_sources % 2 == 0 {
                (column_values[n_sources / 2 - 1] + column_values[n_sources / 2]) / 2.0
            } else {
                column_values[n_sources / 2]
            };
            
            result[j] = median_val;
        }
        
        Ok(result)
    }
    
    /// Trimmed mean fusion
    /// 
    /// Removes a percentage of highest and lowest scores before averaging
    pub fn trimmed_mean(scores: &ArrayView2<f64>, trim_percent: f64) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = scores.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Score matrix cannot be empty");
        }
        
        if trim_percent < 0.0 || trim_percent >= 0.5 {
            return Err("Trim percentage must be between 0 and 0.5");
        }
        
        let trim_count = ((n_sources as f64 * trim_percent) as usize).max(0);
        let keep_count = n_sources - 2 * trim_count;
        
        if keep_count == 0 {
            return Err("Trim percentage too high, no scores left");
        }
        
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let mut column_values: Vec<f64> = scores.column(j).to_vec();
            column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let sum: f64 = column_values[trim_count..n_sources - trim_count].iter().sum();
            result[j] = sum / keep_count as f64;
        }
        
        Ok(result)
    }
    
    /// CombSUM fusion
    /// 
    /// Simple sum of normalized scores
    pub fn comb_sum(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        let normalized = Self::normalized_average(scores)?;
        Ok(normalized * scores.dim().0 as f64)
    }
    
    /// CombMNZ (Multiply by Number of Non-Zero) fusion
    /// 
    /// Sum of scores multiplied by the number of non-zero scores
    pub fn comb_mnz(scores: &ArrayView2<f64>, threshold: f64) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = scores.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Score matrix cannot be empty");
        }
        
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let column = scores.column(j);
            let sum: f64 = column.sum();
            let non_zero_count = column.iter().filter(|&&x| x.abs() > threshold).count();
            
            result[j] = sum * non_zero_count as f64;
        }
        
        Ok(result)
    }
    
    /// ISR (Inverse Square Rank) fusion
    /// 
    /// Weights scores by inverse square of their rank position
    pub fn isr_fusion(scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = scores.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Score matrix cannot be empty");
        }
        
        let mut result = Array1::zeros(n_items);
        
        for i in 0..n_sources {
            // Convert scores to ranks
            let ranks = scores_to_ranks(&scores.row(i));
            
            // Apply ISR weighting
            for j in 0..n_items {
                result[j] += 1.0 / (ranks[j] as f64).powi(2);
            }
        }
        
        Ok(result)
    }
}

/// Convert scores to ranks (1-based, higher score = better rank)
fn scores_to_ranks(scores: &ArrayView1<f64>) -> Vec<usize> {
    let n = scores.len();
    let mut indexed: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    
    // Sort by score (descending)
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let mut ranks = vec![0; n];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank + 1; // 1-based ranking
    }
    
    ranks
}

/// Adaptive score fusion that selects the best method based on score characteristics
pub struct AdaptiveScoreFusion {
    methods: Vec<(&'static str, Box<dyn Fn(&ArrayView2<f64>) -> Result<Array1<f64>, &'static str>>)>,
}

impl AdaptiveScoreFusion {
    pub fn new() -> Self {
        let methods: Vec<(&'static str, Box<dyn Fn(&ArrayView2<f64>) -> Result<Array1<f64>, &'static str>>)> = vec![
            ("average", Box::new(|s| ScoreFusion::average(s))),
            ("normalized", Box::new(|s| ScoreFusion::normalized_average(s))),
            ("median", Box::new(|s| ScoreFusion::median(s))),
            ("trimmed_mean", Box::new(|s| ScoreFusion::trimmed_mean(s, 0.1))),
        ];
        
        Self { methods }
    }
    
    /// Select and apply the best fusion method based on score statistics
    pub fn fuse(&self, scores: &ArrayView2<f64>) -> Result<Array1<f64>, &'static str> {
        // Analyze score characteristics
        let variance_ratio = self.analyze_variance_ratio(scores);
        let outlier_ratio = self.analyze_outlier_ratio(scores);
        
        // Select method based on characteristics
        if outlier_ratio > 0.2 {
            // High outliers: use robust method
            ScoreFusion::trimmed_mean(scores, 0.2)
        } else if variance_ratio > 10.0 {
            // High variance differences: normalize first
            ScoreFusion::normalized_average(scores)
        } else {
            // Default to simple average
            ScoreFusion::average(scores)
        }
    }
    
    fn analyze_variance_ratio(&self, scores: &ArrayView2<f64>) -> f64 {
        let n_sources = scores.dim().0;
        if n_sources < 2 {
            return 1.0;
        }
        
        let mut variances = Vec::new();
        for i in 0..n_sources {
            let var = scores.row(i).std(0.0).powi(2);
            if var > f64::EPSILON {
                variances.push(var);
            }
        }
        
        if variances.len() < 2 {
            return 1.0;
        }
        
        let max_var = variances.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_var = variances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        max_var / min_var.max(f64::EPSILON)
    }
    
    fn analyze_outlier_ratio(&self, scores: &ArrayView2<f64>) -> f64 {
        let all_scores: Vec<f64> = scores.iter().cloned().collect();
        if all_scores.is_empty() {
            return 0.0;
        }
        
        let mean = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
        let std = (all_scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() 
                   / all_scores.len() as f64).sqrt();
        
        if std < f64::EPSILON {
            return 0.0;
        }
        
        let outliers = all_scores.iter()
            .filter(|&&x| (x - mean).abs() > 2.0 * std)
            .count();
        
        outliers as f64 / all_scores.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_average_fusion() {
        let scores = array![
            [0.8, 0.6, 0.9, 0.3],
            [0.7, 0.8, 0.6, 0.4],
            [0.9, 0.5, 0.8, 0.5]
        ];
        
        let result = ScoreFusion::average(&scores.view()).unwrap();
        let expected = array![0.8, 0.633333, 0.766667, 0.4];
        
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < 0.001);
        }
    }
    
    #[test]
    fn test_weighted_average_fusion() {
        let scores = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let weights = array![0.7, 0.3];
        
        let result = ScoreFusion::weighted_average(&scores.view(), &weights.view()).unwrap();
        assert!((result[0] - 0.7).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
    }
    
    #[test]
    fn test_normalized_fusion() {
        let scores = array![
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0]
        ];
        
        let result = ScoreFusion::normalized_average(&scores.view()).unwrap();
        
        // Both rows should contribute equally after normalization
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_median_fusion() {
        let scores = array![
            [1.0, 5.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 3.0, 9.0]
        ];
        
        let result = ScoreFusion::median(&scores.view()).unwrap();
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 6.0);
    }
    
    #[test]
    fn test_trimmed_mean() {
        let scores = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [100.0, 100.0] // Outlier
        ];
        
        let result = ScoreFusion::trimmed_mean(&scores.view(), 0.2).unwrap();
        
        // Should exclude the outlier
        assert!(result[0] < 10.0);
        assert!(result[1] < 10.0);
    }
    
    #[test]
    fn test_comb_mnz() {
        let scores = array![
            [0.5, 0.0, 0.3],
            [0.0, 0.4, 0.2],
            [0.6, 0.5, 0.0]
        ];
        
        let result = ScoreFusion::comb_mnz(&scores.view(), 0.01).unwrap();
        
        // First item: sum=1.1, count=2, result=2.2
        assert!((result[0] - 2.2).abs() < 1e-10);
        // Second item: sum=0.9, count=2, result=1.8
        assert!((result[1] - 1.8).abs() < 1e-10);
        // Third item: sum=0.5, count=2, result=1.0
        assert!((result[2] - 1.0).abs() < 1e-10);
    }
}