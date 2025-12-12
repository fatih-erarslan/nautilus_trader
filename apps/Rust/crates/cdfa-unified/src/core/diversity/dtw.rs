//! Dynamic Time Warping implementation

use crate::error::{CdfaError, Result};
use crate::types::*;
use ndarray::{Array2, ArrayView1};
use num_traits::{Float as FloatTrait, Zero, One};

/// Calculate Dynamic Time Warping distance
pub fn dynamic_time_warping(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    if x.is_empty() || y.is_empty() {
        return Err(CdfaError::invalid_input("Arrays cannot be empty"));
    }
    
    let n = x.len();
    let m = y.len();
    
    // Create distance matrix
    let mut dtw_matrix = Array2::from_elem((n + 1, m + 1), Float::infinity());
    dtw_matrix[[0, 0]] = 0.0;
    
    // Fill the DTW matrix
    for i in 1..=n {
        for j in 1..=m {
            let cost = (x[i - 1] - y[j - 1]).abs();
            
            let min_prev = Float::min(
                dtw_matrix[[i - 1, j]],     // Insertion
                Float::min(
                    dtw_matrix[[i, j - 1]], // Deletion
                    dtw_matrix[[i - 1, j - 1]] // Match
                )
            );
            
            dtw_matrix[[i, j]] = cost + min_prev;
        }
    }
    
    Ok(dtw_matrix[[n, m]])
}

/// Calculate DTW distance with window constraint
pub fn dynamic_time_warping_window(
    x: &ArrayView1<Float>,
    y: &ArrayView1<Float>,
    window: usize,
) -> Result<Float> {
    if x.is_empty() || y.is_empty() {
        return Err(CdfaError::invalid_input("Arrays cannot be empty"));
    }
    
    let n = x.len();
    let m = y.len();
    
    // Create distance matrix
    let mut dtw_matrix = Array2::from_elem((n + 1, m + 1), Float::infinity());
    dtw_matrix[[0, 0]] = 0.0;
    
    // Fill the DTW matrix with window constraint
    for i in 1..=n {
        let j_start = std::cmp::max(1, i.saturating_sub(window));
        let j_end = std::cmp::min(m + 1, i + window + 1);
        
        for j in j_start..j_end {
            let cost = (x[i - 1] - y[j - 1]).abs();
            
            let min_prev = Float::min(
                dtw_matrix[[i - 1, j]],     // Insertion
                Float::min(
                    dtw_matrix[[i, j - 1]], // Deletion
                    dtw_matrix[[i - 1, j - 1]] // Match
                )
            );
            
            dtw_matrix[[i, j]] = cost + min_prev;
        }
    }
    
    Ok(dtw_matrix[[n, m]])
}

/// Fast DTW implementation using approximation
pub fn fast_dtw(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    // For now, use regular DTW with a reasonable window
    let window = std::cmp::max(x.len(), y.len()) / 10;
    dynamic_time_warping_window(x, y, window)
}

/// Calculate DTW-based similarity (normalized)
pub fn dtw_similarity(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    let distance = dynamic_time_warping(x, y)?;
    
    // Normalize by the maximum possible distance
    let max_distance = x.len() as Float * (x.iter().map(|&v| v.abs()).fold(0.0, Float::max) + 
                                           y.iter().map(|&v| v.abs()).fold(0.0, Float::max));
    
    if max_distance > 0.0 {
        Ok(1.0 - distance / max_distance)
    } else {
        Ok(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_dtw_identical_sequences() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        
        let distance = dynamic_time_warping(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dtw_shifted_sequence() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![2.0, 3.0, 4.0];
        
        let distance = dynamic_time_warping(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(distance, 3.0, epsilon = 1e-10); // Each element differs by 1
    }
    
    #[test]
    fn test_dtw_with_window() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let distance = dynamic_time_warping_window(&x.view(), &y.view(), 2).unwrap();
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dtw_similarity() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        
        let similarity = dtw_similarity(&x.view(), &y.view()).unwrap();
        assert!(similarity > 0.9); // Should be high for identical sequences
    }
    
    #[test]
    fn test_fast_dtw() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.1, 2.1, 3.1, 4.1, 5.1];
        
        let distance = fast_dtw(&x.view(), &y.view()).unwrap();
        assert!(distance > 0.0);
        assert!(distance < 1.0); // Should be small for similar sequences
    }
}