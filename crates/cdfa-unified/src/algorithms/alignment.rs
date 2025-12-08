use crate::error::{CdfaError, Result};
use crate::types::*;
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;

/// Time series alignment algorithms
/// 
/// This module provides various methods for aligning time series data
/// including Dynamic Time Warping (DTW) and related algorithms
pub struct TimeSeriesAlignment;

impl TimeSeriesAlignment {
    /// Dynamic Time Warping (DTW) distance
    pub fn dtw_distance(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> std::result::Result<f64, &'static str> {
        if x.is_empty() || y.is_empty() {
            return Err("Input series cannot be empty");
        }
        
        let n = x.len();
        let m = y.len();
        
        // Initialize DTW matrix with infinity
        let mut dtw_matrix = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
        dtw_matrix[(0, 0)] = 0.0;
        
        // Fill DTW matrix
        for i in 1..=n {
            for j in 1..=m {
                let cost = (x[i-1] - y[j-1]).abs();
                dtw_matrix[(i, j)] = cost + dtw_matrix[(i-1, j)]
                    .min(dtw_matrix[(i, j-1)])
                    .min(dtw_matrix[(i-1, j-1)]);
            }
        }
        
        Ok(dtw_matrix[(n, m)])
    }
    
    /// DTW with constrained warping path (Sakoe-Chiba band)
    pub fn dtw_constrained(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>, 
        window: usize
    ) -> std::result::Result<f64, &'static str> {
        if x.is_empty() || y.is_empty() {
            return Err("Input series cannot be empty");
        }
        
        let n = x.len();
        let m = y.len();
        
        if window == 0 {
            return Err("Window size must be positive");
        }
        
        let mut dtw_matrix = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
        dtw_matrix[(0, 0)] = 0.0;
        
        for i in 1..=n {
            let j_start = ((i as i32 - window as i32).max(1)) as usize;
            let j_end = ((i + window).min(m)) + 1;
            
            for j in j_start..j_end {
                let cost = (x[i-1] - y[j-1]).abs();
                dtw_matrix[(i, j)] = cost + dtw_matrix[(i-1, j)]
                    .min(dtw_matrix[(i, j-1)])
                    .min(dtw_matrix[(i-1, j-1)]);
            }
        }
        
        Ok(dtw_matrix[(n, m)])
    }
    
    /// DTW with traceback path
    pub fn dtw_with_path(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>
    ) -> std::result::Result<(f64, Vec<(usize, usize)>), &'static str> {
        if x.is_empty() || y.is_empty() {
            return Err("Input series cannot be empty");
        }
        
        let n = x.len();
        let m = y.len();
        
        let mut dtw_matrix = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
        dtw_matrix[(0, 0)] = 0.0;
        
        // Fill DTW matrix
        for i in 1..=n {
            for j in 1..=m {
                let cost = (x[i-1] - y[j-1]).abs();
                dtw_matrix[(i, j)] = cost + dtw_matrix[(i-1, j)]
                    .min(dtw_matrix[(i, j-1)])
                    .min(dtw_matrix[(i-1, j-1)]);
            }
        }
        
        // Traceback to find optimal path
        let mut path = Vec::new();
        let mut i = n;
        let mut j = m;
        
        while i > 0 && j > 0 {
            path.push((i-1, j-1));
            
            let diag = dtw_matrix[(i-1, j-1)];
            let left = dtw_matrix[(i, j-1)];
            let up = dtw_matrix[(i-1, j)];
            
            if diag <= left && diag <= up {
                i -= 1;
                j -= 1;
            } else if left <= up {
                j -= 1;
            } else {
                i -= 1;
            }
        }
        
        // Add remaining path if needed
        while i > 0 {
            path.push((i-1, j));
            i -= 1;
        }
        while j > 0 {
            path.push((i, j-1));
            j -= 1;
        }
        
        path.reverse();
        Ok((dtw_matrix[(n, m)], path))
    }
    
    /// Longest Common Subsequence (LCS) alignment
    pub fn lcs_alignment(x: &ArrayView1<i32>, y: &ArrayView1<i32>) -> crate::error::CdfaResult<Vec<(usize, usize)>> {
        if x.is_empty() || y.is_empty() {
            return Err(crate::error::CdfaError::InvalidInput { message: "Input series cannot be empty".to_string() });
        }
        
        let n = x.len();
        let m = y.len();
        
        // Create LCS matrix
        let mut lcs_matrix: Array2<i32> = Array2::zeros((n + 1, m + 1));
        
        for i in 1..=n {
            for j in 1..=m {
                if x[i-1] == y[j-1] {
                    lcs_matrix[(i, j)] = lcs_matrix[(i-1, j-1)] + 1;
                } else {
                    lcs_matrix[(i, j)] = lcs_matrix[(i-1, j)].max(lcs_matrix[(i, j-1)]);
                }
            }
        }
        
        // Traceback to find alignment
        let mut alignment = Vec::new();
        let mut i = n;
        let mut j = m;
        
        while i > 0 && j > 0 {
            if x[i-1] == y[j-1] {
                alignment.push((i-1, j-1));
                i -= 1;
                j -= 1;
            } else if lcs_matrix[(i-1, j)] > lcs_matrix[(i, j-1)] {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        
        alignment.reverse();
        Ok(alignment)
    }
    
    /// Needleman-Wunsch global alignment
    pub fn needleman_wunsch(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>,
        match_score: f64,
        mismatch_penalty: f64,
        gap_penalty: f64,
        tolerance: f64
    ) -> std::result::Result<(f64, Vec<(Option<usize>, Option<usize>)>), &'static str> {
        if x.is_empty() || y.is_empty() {
            return Err("Input series cannot be empty");
        }
        
        let n = x.len();
        let m = y.len();
        
        // Initialize scoring matrix
        let mut score_matrix = Array2::zeros((n + 1, m + 1));
        
        // Initialize first row and column with gap penalties
        for i in 1..=n {
            score_matrix[(i, 0)] = score_matrix[(i-1, 0)] - gap_penalty;
        }
        for j in 1..=m {
            score_matrix[(0, j)] = score_matrix[(0, j-1)] - gap_penalty;
        }
        
        // Fill scoring matrix
        for i in 1..=n {
            for j in 1..=m {
                let is_match = (x[i-1] - y[j-1]).abs() <= tolerance;
                let match_mismatch_score = if is_match { match_score } else { -mismatch_penalty };
                
                let diagonal: Float = score_matrix[(i-1, j-1)] + match_mismatch_score;
                let up = score_matrix[(i-1, j)] - gap_penalty;
                let left = score_matrix[(i, j-1)] - gap_penalty;
                
                score_matrix[(i, j)] = diagonal.max(up).max(left);
            }
        }
        
        // Traceback to find alignment
        let mut alignment = Vec::new();
        let mut i = n;
        let mut j = m;
        
        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let is_match = (x[i-1] - y[j-1]).abs() <= tolerance;
                let match_mismatch_score = if is_match { match_score } else { -mismatch_penalty };
                
                let diagonal: Float = score_matrix[(i-1, j-1)] + match_mismatch_score;
                let up = score_matrix[(i-1, j)] - gap_penalty;
                let left = score_matrix[(i, j-1)] - gap_penalty;
                
                if score_matrix[(i, j)] == diagonal {
                    alignment.push((Some(i-1), Some(j-1)));
                    i -= 1;
                    j -= 1;
                } else if score_matrix[(i, j)] == up {
                    alignment.push((Some(i-1), None));
                    i -= 1;
                } else {
                    alignment.push((None, Some(j-1)));
                    j -= 1;
                }
            } else if i > 0 {
                alignment.push((Some(i-1), None));
                i -= 1;
            } else {
                alignment.push((None, Some(j-1)));
                j -= 1;
            }
        }
        
        alignment.reverse();
        Ok((score_matrix[(n, m)], alignment))
    }
    
    /// Smith-Waterman local alignment
    pub fn smith_waterman(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>,
        match_score: f64,
        mismatch_penalty: f64,
        gap_penalty: f64,
        tolerance: f64
    ) -> std::result::Result<(f64, Vec<(usize, usize)>), &'static str> {
        if x.is_empty() || y.is_empty() {
            return Err("Input series cannot be empty");
        }
        
        let n = x.len();
        let m = y.len();
        
        let mut score_matrix = Array2::zeros((n + 1, m + 1));
        let mut max_score = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;
        
        // Fill scoring matrix
        for i in 1..=n {
            for j in 1..=m {
                let is_match = (x[i-1] - y[j-1]).abs() <= tolerance;
                let match_mismatch_score = if is_match { match_score } else { -mismatch_penalty };
                
                let diagonal: Float = score_matrix[(i-1, j-1)] + match_mismatch_score;
                let up = score_matrix[(i-1, j)] - gap_penalty;
                let left = score_matrix[(i, j-1)] - gap_penalty;
                
                score_matrix[(i, j)] = diagonal.max(up).max(left).max(0.0);
                
                if score_matrix[(i, j)] > max_score {
                    max_score = score_matrix[(i, j)];
                    max_i = i;
                    max_j = j;
                }
            }
        }
        
        // Traceback from maximum score position
        let mut alignment = Vec::new();
        let mut i = max_i;
        let mut j = max_j;
        
        while i > 0 && j > 0 && score_matrix[(i, j)] > 0.0 {
            let is_match = (x[i-1] - y[j-1]).abs() <= tolerance;
            let match_mismatch_score = if is_match { match_score } else { -mismatch_penalty };
            
            let diagonal = score_matrix[(i-1, j-1)] + match_mismatch_score;
            let up = score_matrix[(i-1, j)] - gap_penalty;
            let left = score_matrix[(i, j-1)] - gap_penalty;
            
            if score_matrix[(i, j)] == diagonal {
                alignment.push((i-1, j-1));
                i -= 1;
                j -= 1;
            } else if score_matrix[(i, j)] == up {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        
        alignment.reverse();
        Ok((max_score, alignment))
    }
    
    /// Edit distance (Levenshtein distance) with traceback
    pub fn edit_distance(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>,
        tolerance: f64
    ) -> std::result::Result<(usize, Vec<EditOperation>), &'static str> {
        if x.is_empty() && y.is_empty() {
            return Ok((0, vec![]));
        }
        
        let n = x.len();
        let m = y.len();
        
        let mut dp = Array2::zeros((n + 1, m + 1));
        
        // Initialize base cases
        for i in 0..=n {
            dp[(i, 0)] = i;
        }
        for j in 0..=m {
            dp[(0, j)] = j;
        }
        
        // Fill DP table
        for i in 1..=n {
            for j in 1..=m {
                let is_match = (x[i-1] - y[j-1]).abs() <= tolerance;
                
                if is_match {
                    dp[(i, j)] = dp[(i-1, j-1)];
                } else {
                    dp[(i, j)] = 1 + dp[(i-1, j)]  // deletion
                        .min(dp[(i, j-1)])          // insertion
                        .min(dp[(i-1, j-1)]);       // substitution
                }
            }
        }
        
        // Traceback to find operations
        let mut operations = Vec::new();
        let mut i = n;
        let mut j = m;
        
        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let is_match = (x[i-1] - y[j-1]).abs() <= tolerance;
                
                if is_match && dp[(i, j)] == dp[(i-1, j-1)] {
                    operations.push(EditOperation::Match(i-1, j-1));
                    i -= 1;
                    j -= 1;
                } else if dp[(i, j)] == dp[(i-1, j-1)] + 1 {
                    operations.push(EditOperation::Substitute(i-1, j-1));
                    i -= 1;
                    j -= 1;
                } else if dp[(i, j)] == dp[(i-1, j)] + 1 {
                    operations.push(EditOperation::Delete(i-1));
                    i -= 1;
                } else {
                    operations.push(EditOperation::Insert(j-1));
                    j -= 1;
                }
            } else if i > 0 {
                operations.push(EditOperation::Delete(i-1));
                i -= 1;
            } else {
                operations.push(EditOperation::Insert(j-1));
                j -= 1;
            }
        }
        
        operations.reverse();
        Ok((dp[(n, m)], operations))
    }
    
    /// Cross-correlation based alignment
    pub fn cross_correlation_alignment(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>
    ) -> std::result::Result<(i32, f64), &'static str> {
        if x.is_empty() || y.is_empty() {
            return Err("Input series cannot be empty");
        }
        
        let n = x.len();
        let m = y.len();
        let max_lag = (n + m) / 2;
        
        let mut best_lag = 0i32;
        let mut max_correlation = f64::NEG_INFINITY;
        
        // Calculate means
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);
        
        // Calculate standard deviations
        let x_std = x.std(0.0);
        let y_std = y.std(0.0);
        
        if x_std < f64::EPSILON || y_std < f64::EPSILON {
            return Err("Input series have zero variance");
        }
        
        for lag in -(max_lag as i32)..(max_lag as i32) {
            let mut sum = 0.0;
            let mut count = 0;
            
            for i in 0..n {
                let j = i as i32 + lag;
                if j >= 0 && j < m as i32 {
                    let x_norm = (x[i] - x_mean) / x_std;
                    let y_norm = (y[j as usize] - y_mean) / y_std;
                    sum += x_norm * y_norm;
                    count += 1;
                }
            }
            
            if count > 0 {
                let correlation = sum / count as f64;
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_lag = lag;
                }
            }
        }
        
        Ok((best_lag, max_correlation))
    }
    
    /// Phase correlation for shift detection
    pub fn phase_correlation(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>
    ) -> std::result::Result<(f64, f64), &'static str> {
        if x.len() != y.len() {
            return Err("Input series must have same length");
        }
        
        if x.is_empty() {
            return Err("Input series cannot be empty");
        }
        
        let n = x.len();
        
        // Simple phase correlation approximation
        // In practice, this would use FFT
        let mut phase_diff_sum = 0.0;
        let mut amplitude_sum = 0.0;
        
        for i in 0..n-1 {
            let x_diff = x[i+1] - x[i];
            let y_diff = y[i+1] - y[i];
            
            if x_diff.abs() > f64::EPSILON && y_diff.abs() > f64::EPSILON {
                let phase_diff = (y_diff / x_diff).atan();
                phase_diff_sum += phase_diff;
                amplitude_sum += (x_diff * x_diff + y_diff * y_diff).sqrt();
            }
        }
        
        let avg_phase_diff = if n > 1 { phase_diff_sum / (n - 1) as f64 } else { 0.0 };
        let avg_amplitude = if n > 1 { amplitude_sum / (n - 1) as f64 } else { 0.0 };
        
        Ok((avg_phase_diff, avg_amplitude))
    }
}

/// Edit operations for sequence alignment
#[derive(Debug, Clone, PartialEq)]
pub enum EditOperation {
    Match(usize, usize),
    Substitute(usize, usize),
    Insert(usize),
    Delete(usize),
}

impl EditOperation {
    /// Get the operation type as a string
    pub fn operation_type(&self) -> &'static str {
        match self {
            EditOperation::Match(_, _) => "match",
            EditOperation::Substitute(_, _) => "substitute",
            EditOperation::Insert(_) => "insert",
            EditOperation::Delete(_) => "delete",
        }
    }
    
    /// Get the cost of the operation
    pub fn cost(&self) -> usize {
        match self {
            EditOperation::Match(_, _) => 0,
            EditOperation::Substitute(_, _) => 1,
            EditOperation::Insert(_) => 1,
            EditOperation::Delete(_) => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_dtw_distance() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.1, 2.1, 2.9, 4.1];
        
        let distance = TimeSeriesAlignment::dtw_distance(&x.view(), &y.view()).unwrap();
        assert!(distance < 1.0); // Should be small for similar series
    }
    
    #[test]
    fn test_dtw_with_path() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        
        let (distance, path) = TimeSeriesAlignment::dtw_with_path(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10);
        assert_eq!(path, vec![(0, 0), (1, 1), (2, 2)]);
    }
    
    #[test]
    fn test_dtw_constrained() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.1, 2.2, 3.1, 4.0, 5.1];
        
        let distance = TimeSeriesAlignment::dtw_constrained(&x.view(), &y.view(), 2).unwrap();
        assert!(distance >= 0.0);
    }
    
    #[test]
    fn test_lcs_alignment() {
        let x = array![1, 2, 3, 2, 4];
        let y = array![2, 1, 3, 2, 4, 5];
        
        let alignment = TimeSeriesAlignment::lcs_alignment(&x.view(), &y.view()).unwrap();
        assert!(!alignment.is_empty());
    }
    
    #[test]
    fn test_needleman_wunsch() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.1, 2.1, 3.1];
        
        let (score, alignment) = TimeSeriesAlignment::needleman_wunsch(
            &x.view(), &y.view(), 2.0, 1.0, 1.0, 0.2
        ).unwrap();
        
        assert!(score > 0.0);
        assert_eq!(alignment.len(), 3);
    }
    
    #[test]
    fn test_smith_waterman() {
        let x = array![1.0, 5.0, 2.0, 3.0, 6.0];
        let y = array![4.0, 2.0, 3.0, 7.0];
        
        let (score, alignment) = TimeSeriesAlignment::smith_waterman(
            &x.view(), &y.view(), 2.0, 1.0, 1.0, 0.2
        ).unwrap();
        
        assert!(score >= 0.0);
        assert!(!alignment.is_empty());
    }
    
    #[test]
    fn test_edit_distance() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.1, 2.1, 4.0];
        
        let (distance, operations) = TimeSeriesAlignment::edit_distance(
            &x.view(), &y.view(), 0.2
        ).unwrap();
        
        assert_eq!(distance, 1); // One substitution
        assert_eq!(operations.len(), 3);
        
        // Check operations
        assert_eq!(operations[0], EditOperation::Match(0, 0));
        assert_eq!(operations[1], EditOperation::Match(1, 1));
        assert_eq!(operations[2], EditOperation::Substitute(2, 2));
    }
    
    #[test]
    fn test_cross_correlation_alignment() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 3.0, 4.0, 5.0, 6.0]; // x shifted by 1 and +1
        
        let (_lag, correlation) = TimeSeriesAlignment::cross_correlation_alignment(
            &x.view(), &y.view()
        ).unwrap();
        
        assert!(correlation > 0.8); // Should have high correlation
    }
    
    #[test]
    fn test_phase_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        
        let (phase_diff, amplitude) = TimeSeriesAlignment::phase_correlation(
            &x.view(), &y.view()
        ).unwrap();
        
        assert!(phase_diff.abs() < 1e-10); // Should be very small for identical series
        assert!(amplitude > 0.0);
    }
    
    #[test]
    fn test_edit_operation_methods() {
        let op_match = EditOperation::Match(0, 0);
        let op_substitute = EditOperation::Substitute(1, 1);
        let op_insert = EditOperation::Insert(2);
        let op_delete = EditOperation::Delete(3);
        
        assert_eq!(op_match.operation_type(), "match");
        assert_eq!(op_substitute.operation_type(), "substitute");
        assert_eq!(op_insert.operation_type(), "insert");
        assert_eq!(op_delete.operation_type(), "delete");
        
        assert_eq!(op_match.cost(), 0);
        assert_eq!(op_substitute.cost(), 1);
        assert_eq!(op_insert.cost(), 1);
        assert_eq!(op_delete.cost(), 1);
    }
}