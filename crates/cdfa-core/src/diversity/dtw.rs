use ndarray::{Array1, Array2, ArrayView1};

/// Dynamic Time Warping distance
/// 
/// Measures similarity between two temporal sequences which may vary in speed
/// Returns the minimum distance between the two sequences
pub fn dynamic_time_warping(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if x.is_empty() || y.is_empty() {
        return Err("Sequences cannot be empty");
    }
    
    let n = x.len();
    let m = y.len();
    
    // Create cost matrix
    let mut cost = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
    cost[[0, 0]] = 0.0;
    
    // Fill the cost matrix
    for i in 1..=n {
        for j in 1..=m {
            let distance = (x[i-1] - y[j-1]).abs();
            cost[[i, j]] = distance + cost[[i-1, j]].min(cost[[i, j-1]]).min(cost[[i-1, j-1]]);
        }
    }
    
    Ok(cost[[n, m]])
}

/// DTW with window constraint (Sakoe-Chiba band)
/// 
/// Constrains the warping path to stay within a window of the diagonal
/// This improves performance and can give more meaningful results
pub fn dynamic_time_warping_window(
    x: &ArrayView1<f64>, 
    y: &ArrayView1<f64>, 
    window: usize
) -> Result<f64, &'static str> {
    if x.is_empty() || y.is_empty() {
        return Err("Sequences cannot be empty");
    }
    
    let n = x.len();
    let m = y.len();
    let w = window.max(((n as i32 - m as i32).abs()) as usize);
    
    // Create cost matrix
    let mut cost = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
    cost[[0, 0]] = 0.0;
    
    // Fill the cost matrix with window constraint
    for i in 1..=n {
        for j in ((i as i32 - w as i32).max(1) as usize)..=((i + w).min(m)) {
            let distance = (x[i-1] - y[j-1]).abs();
            cost[[i, j]] = distance + cost[[i-1, j]].min(cost[[i, j-1]]).min(cost[[i-1, j-1]]);
        }
    }
    
    Ok(cost[[n, m]])
}

/// Get the optimal warping path for DTW
pub fn dynamic_time_warping_path(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<Vec<(usize, usize)>, &'static str> {
    if x.is_empty() || y.is_empty() {
        return Err("Sequences cannot be empty");
    }
    
    let n = x.len();
    let m = y.len();
    
    // Create cost matrix
    let mut cost = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
    cost[[0, 0]] = 0.0;
    
    // Fill the cost matrix
    for i in 1..=n {
        for j in 1..=m {
            let distance = (x[i-1] - y[j-1]).abs();
            cost[[i, j]] = distance + cost[[i-1, j]].min(cost[[i, j-1]]).min(cost[[i-1, j-1]]);
        }
    }
    
    // Backtrack to find the path
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        
        // Find which direction we came from
        let diag = cost[[i-1, j-1]];
        let up = cost[[i-1, j]];
        let left = cost[[i, j-1]];
        
        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up < left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    
    // Add remaining steps if any
    while i > 0 {
        path.push((i - 1, j));
        i -= 1;
    }
    while j > 0 {
        path.push((i, j - 1));
        j -= 1;
    }
    
    path.reverse();
    Ok(path)
}

/// Multivariate DTW for sequences of vectors
pub fn multivariate_dtw(x: &ArrayView1<Array1<f64>>, y: &ArrayView1<Array1<f64>>) -> Result<f64, &'static str> {
    if x.is_empty() || y.is_empty() {
        return Err("Sequences cannot be empty");
    }
    
    // Check dimensionality
    let dim = x[0].len();
    for vec in x.iter().chain(y.iter()) {
        if vec.len() != dim {
            return Err("All vectors must have the same dimension");
        }
    }
    
    let n = x.len();
    let m = y.len();
    
    // Create cost matrix
    let mut cost = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
    cost[[0, 0]] = 0.0;
    
    // Fill the cost matrix
    for i in 1..=n {
        for j in 1..=m {
            // Euclidean distance between vectors
            let distance = euclidean_distance(&x[i-1], &y[j-1]);
            cost[[i, j]] = distance + cost[[i-1, j]].min(cost[[i, j-1]]).min(cost[[i-1, j-1]]);
        }
    }
    
    Ok(cost[[n, m]])
}

/// Fast DTW approximation using coarsening
pub fn fast_dtw(x: &ArrayView1<f64>, y: &ArrayView1<f64>, radius: usize) -> Result<f64, &'static str> {
    if x.is_empty() || y.is_empty() {
        return Err("Sequences cannot be empty");
    }
    
    // Base case: small sequences use standard DTW
    if x.len() <= radius * 2 || y.len() <= radius * 2 {
        return dynamic_time_warping(x, y);
    }
    
    // Downsample sequences
    let x_coarse = downsample(x);
    let y_coarse = downsample(y);
    
    // Get path at coarse resolution
    let coarse_path = dynamic_time_warping_path(&x_coarse.view(), &y_coarse.view())?;
    
    // Project path to finer resolution
    let projected_path = project_path(&coarse_path, x.len(), y.len());
    
    // Compute DTW with window around projected path
    dtw_with_path_constraint(x, y, &projected_path, radius)
}

// Helper functions

fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn downsample(x: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let new_len = (n + 1) / 2;
    let mut result = Array1::zeros(new_len);
    
    for i in 0..new_len {
        let idx = i * 2;
        if idx + 1 < n {
            result[i] = (x[idx] + x[idx + 1]) / 2.0;
        } else {
            result[i] = x[idx];
        }
    }
    
    result
}

fn project_path(coarse_path: &[(usize, usize)], n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut projected = Vec::new();
    
    for &(i, j) in coarse_path {
        // Project each coarse point to fine resolution
        let i_start = (i * 2).min(n - 1);
        let i_end = ((i + 1) * 2).min(n);
        let j_start = (j * 2).min(m - 1);
        let j_end = ((j + 1) * 2).min(m);
        
        for ii in i_start..i_end {
            for jj in j_start..j_end {
                projected.push((ii, jj));
            }
        }
    }
    
    // Remove duplicates while preserving order
    let mut seen = std::collections::HashSet::new();
    projected.retain(|&item| seen.insert(item));
    
    projected
}

fn dtw_with_path_constraint(
    x: &ArrayView1<f64>, 
    y: &ArrayView1<f64>, 
    path: &[(usize, usize)], 
    radius: usize
) -> Result<f64, &'static str> {
    let n = x.len();
    let m = y.len();
    
    // Create allowed region around path
    let mut allowed = Array2::from_elem((n, m), false);
    for &(i, j) in path {
        for di in 0..=radius {
            for dj in 0..=radius {
                if i >= di && j >= dj {
                    allowed[[i - di, j - dj]] = true;
                }
                if i + di < n && j + dj < m {
                    allowed[[i + di, j + dj]] = true;
                }
                if i >= di && j + dj < m {
                    allowed[[i - di, j + dj]] = true;
                }
                if i + di < n && j >= dj {
                    allowed[[i + di, j - dj]] = true;
                }
            }
        }
    }
    
    // Compute DTW only in allowed region
    let mut cost = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
    cost[[0, 0]] = 0.0;
    
    for i in 1..=n {
        for j in 1..=m {
            if allowed[[i-1, j-1]] {
                let distance = (x[i-1] - y[j-1]).abs();
                cost[[i, j]] = distance + cost[[i-1, j]].min(cost[[i, j-1]]).min(cost[[i-1, j-1]]);
            }
        }
    }
    
    Ok(cost[[n, m]])
}

/// DTW-based similarity score (normalized to [0, 1])
pub fn dtw_similarity(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, &'static str> {
    let dtw_dist = dynamic_time_warping(x, y)?;
    
    // Normalize by the sum of sequence lengths
    let max_dist = (x.len() + y.len()) as f64 * 
                   x.iter().chain(y.iter()).fold(0.0f64, |max, &val| max.max(val.abs()));
    
    if max_dist == 0.0 {
        Ok(1.0) // Both sequences are zero
    } else {
        Ok(1.0 - (dtw_dist / max_dist).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_dtw_identical_sequences() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let dist = dynamic_time_warping(&x.view(), &y.view()).unwrap();
        assert!(dist.abs() < 1e-10);
    }
    
    #[test]
    fn test_dtw_shifted_sequences() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let dist = dynamic_time_warping(&x.view(), &y.view()).unwrap();
        assert!(dist > 0.0 && dist < 10.0);
    }
    
    #[test]
    fn test_dtw_window_constraint() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let dist_full = dynamic_time_warping(&x.view(), &y.view()).unwrap();
        let dist_window = dynamic_time_warping_window(&x.view(), &y.view(), 2).unwrap();
        
        assert!((dist_full - dist_window).abs() < 1e-10);
    }
    
    #[test]
    fn test_dtw_path() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let path = dynamic_time_warping_path(&x.view(), &y.view()).unwrap();
        
        // Should follow diagonal for identical sequences
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], (0, 0));
        assert_eq!(path[1], (1, 1));
        assert_eq!(path[2], (2, 2));
    }
    
    #[test]
    fn test_multivariate_dtw() {
        let x = array![
            array![1.0, 2.0],
            array![2.0, 3.0],
            array![3.0, 4.0]
        ];
        let y = array![
            array![1.0, 2.0],
            array![2.0, 3.0],
            array![3.0, 4.0]
        ];
        
        let dist = multivariate_dtw(&x.view(), &y.view()).unwrap();
        assert!(dist.abs() < 1e-10);
    }
    
    #[test]
    fn test_fast_dtw() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        
        let dist_standard = dynamic_time_warping(&x.view(), &y.view()).unwrap();
        let dist_fast = fast_dtw(&x.view(), &y.view(), 2).unwrap();
        
        // Fast DTW should give similar result
        assert!((dist_standard - dist_fast).abs() < dist_standard * 0.1);
    }
    
    #[test]
    fn test_dtw_similarity() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let sim = dtw_similarity(&x.view(), &y.view()).unwrap();
        assert!((sim - 1.0).abs() < 1e-10); // Perfect similarity
        
        let z = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let sim2 = dtw_similarity(&x.view(), &z.view()).unwrap();
        assert!(sim2 < 1.0 && sim2 > 0.0); // Partial similarity
    }
}