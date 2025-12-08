//! Spearman rank correlation implementation

use crate::error::{CdfaError, Result};
use crate::types::*;
use crate::traits::DiversityMethod;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float as FloatTrait, FromPrimitive};

/// Calculate Spearman rank correlation coefficient
pub fn spearman_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(CdfaError::invalid_input("Arrays must have the same length"));
    }
    
    if x.is_empty() {
        return Err(CdfaError::invalid_input("Arrays cannot be empty"));
    }
    
    let n = x.len() as Float;
    
    // Convert to ranks
    let ranks_x = compute_ranks(x);
    let ranks_y = compute_ranks(y);
    
    // Calculate differences
    let d_squared: Float = ranks_x.iter()
        .zip(ranks_y.iter())
        .map(|(rx, ry)| (*rx - *ry).powi(2))
        .sum();
    
    // Spearman correlation formula
    let rho = 1.0 - (6.0 * d_squared) / (n * (n * n - 1.0));
    
    Ok(rho)
}

/// Fast Spearman correlation implementation
pub fn spearman_correlation_fast(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    spearman_correlation(x, y)
}

/// Compute rank matrix for multiple columns
pub fn spearman_rank_matrix(data: &ArrayView2<Float>) -> Result<Array2<Float>> {
    let (n_rows, n_cols) = data.dim();
    let mut rank_matrix = Array2::zeros((n_rows, n_cols));
    
    for j in 0..n_cols {
        let column = data.column(j);
        let ranks = compute_ranks(&column);
        for (i, &rank) in ranks.iter().enumerate() {
            rank_matrix[[i, j]] = rank;
        }
    }
    
    Ok(rank_matrix)
}

/// Compute ranks for a single array
fn compute_ranks(data: &ArrayView1<Float>) -> Array1<Float> {
    let n = data.len();
    let mut indexed_data: Vec<(usize, Float)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    
    // Sort by value
    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut ranks = Array1::zeros(n);
    
    // Assign ranks (1-based)
    for (rank, &(original_index, _)) in indexed_data.iter().enumerate() {
        ranks[original_index] = (rank + 1) as Float;
    }
    
    // Handle ties by averaging ranks
    let mut i = 0;
    while i < n {
        let current_value = indexed_data[i].1;
        let mut j = i;
        
        // Find all elements with the same value
        while j < n && indexed_data[j].1 == current_value {
            j += 1;
        }
        
        // If there are ties, average the ranks
        if j - i > 1 {
            let avg_rank = (i + j + 1) as Float / 2.0;
            for k in i..j {
                let original_index = indexed_data[k].0;
                ranks[original_index] = avg_rank;
            }
        }
        
        i = j;
    }
    
    ranks
}

/// Spearman rank correlation diversity method
pub struct SpearmanDiversity;

impl SpearmanDiversity {
    /// Create a new Spearman diversity method
    pub fn new() -> Self {
        Self
    }
}

impl DiversityMethod for SpearmanDiversity {
    fn calculate(&self, data: &FloatArrayView2) -> Result<FloatArray1> {
        let n_features = data.ncols();
        let mut diversity_scores = FloatArray1::zeros(n_features);
        
        // Calculate pairwise Spearman correlation and convert to diversity scores
        for i in 0..n_features {
            let col_i = data.column(i);
            let mut sum_diversity = 0.0;
            let mut count = 0;
            
            for j in 0..n_features {
                if i != j {
                    let col_j = data.column(j);
                    let correlation = spearman_correlation(&col_i, &col_j)?;
                    sum_diversity += 1.0 - correlation.abs(); // Convert correlation to diversity
                    count += 1;
                }
            }
            
            diversity_scores[i] = if count > 0 { sum_diversity / count as Float } else { 0.0 };
        }
        
        Ok(diversity_scores)
    }
    
    fn name(&self) -> &'static str {
        "spearman"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_spearman_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = spearman_correlation(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_spearman_correlation_negative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
        
        let correlation = spearman_correlation(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(correlation, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_compute_ranks() {
        let data = array![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = compute_ranks(&data.view());
        
        // Expected ranks: [3, 1.5, 4, 1.5, 5] (ties averaged)
        assert_abs_diff_eq!(ranks[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[1], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[2], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[3], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[4], 5.0, epsilon = 1e-10);
    }
}