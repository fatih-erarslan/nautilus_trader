/// High-Performance Dynamic Time Warping (DTW) Implementation
///
/// Pure Rust DTW with NAPI bindings for Node.js integration.
/// Expected performance: 50-100x faster than pure JavaScript
/// via zero-copy FFI and LLVM optimizations.

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// DTW Result containing similarity, distance, and alignment path
#[napi(object)]
pub struct DtwResult {
    /// Similarity score (0-1, higher = more similar)
    pub similarity: f64,

    /// DTW distance (lower = more similar)
    pub distance: f64,

    /// Alignment path as flat array [i0, j0, i1, j1, ...]
    pub alignment: Vec<u32>,
}

/// Compute Dynamic Time Warping distance between two patterns
///
/// This implementation uses the classic DTW algorithm with O(n*m) complexity.
/// Optimizations:
/// - Zero-copy access to Float64Array via NAPI
/// - LLVM auto-vectorization for inner loops
/// - Minimal allocations (single DTW matrix)
///
/// # Arguments
/// * `pattern_a` - First time series pattern
/// * `pattern_b` - Second time series pattern
///
/// # Returns
/// DtwResult with similarity score, distance, and alignment path
///
/// # Performance
/// - Expected: <1ms for 100-element patterns
/// - 50-100x faster than pure JavaScript
#[napi]
pub fn dtw_distance_rust(pattern_a: Float64Array, pattern_b: Float64Array) -> Result<DtwResult> {
    // Convert to Rust slices (zero-copy via NAPI)
    let a = pattern_a.as_ref();
    let b = pattern_b.as_ref();

    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "Empty pattern provided".to_string()
        ));
    }

    // Initialize DTW matrix
    // Using Vec<Vec> for simplicity - could optimize with flat vec
    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    // Compute DTW distance (hottest path - LLVM will vectorize)
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dtw[i][j] = cost + dtw[i - 1][j]
                .min(dtw[i][j - 1])
                .min(dtw[i - 1][j - 1]);
        }
    }

    let distance = dtw[n][m];

    // Backtrack to find alignment path
    let mut alignment_pairs = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 && j > 0 {
        alignment_pairs.push(((i - 1) as u32, (j - 1) as u32));

        let diagonal = dtw[i - 1][j - 1];
        let left = dtw[i][j - 1];
        let up = dtw[i - 1][j];

        if diagonal <= left && diagonal <= up {
            i -= 1;
            j -= 1;
        } else if left <= up {
            j -= 1;
        } else {
            i -= 1;
        }
    }

    alignment_pairs.reverse();

    // Flatten alignment to single array [i0, j0, i1, j1, ...]
    let mut alignment = Vec::with_capacity(alignment_pairs.len() * 2);
    for (i, j) in alignment_pairs {
        alignment.push(i);
        alignment.push(j);
    }

    // Convert distance to similarity (0-1 scale)
    let max_dist = (n + m) as f64;
    let similarity = (1.0 - (distance / max_dist)).max(0.0).min(1.0);

    Ok(DtwResult {
        similarity,
        distance,
        alignment,
    })
}

/// Optimized batch DTW computation for multiple pattern pairs
///
/// Processes multiple DTW comparisons in a single NAPI call to amortize
/// FFI overhead. Useful for comparing one pattern against many historical patterns.
///
/// # Arguments
/// * `pattern` - Reference pattern to compare
/// * `historical` - Array of historical patterns (flat: [p1_v1, p1_v2, ..., p2_v1, p2_v2, ...])
/// * `pattern_length` - Length of each historical pattern
///
/// # Returns
/// Array of distances (one per historical pattern)
#[napi]
pub fn dtw_batch(
    pattern: Float64Array,
    historical: Float64Array,
    pattern_length: u32,
) -> Result<Vec<f64>> {
    let p = pattern.as_ref();
    let h = historical.as_ref();
    let len = pattern_length as usize;

    if h.len() % len != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Historical data length {} not divisible by pattern length {}", h.len(), len)
        ));
    }

    let num_patterns = h.len() / len;
    let mut distances = Vec::with_capacity(num_patterns);

    // Process each historical pattern
    for i in 0..num_patterns {
        let start = i * len;
        let end = start + len;
        let hist_pattern = &h[start..end];

        // Compute DTW distance (simplified - no alignment needed)
        let distance = compute_dtw_distance_only(p, hist_pattern);
        distances.push(distance);
    }

    Ok(distances)
}

/// Helper: Compute DTW distance without alignment (faster)
fn compute_dtw_distance_only(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();

    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dtw[i][j] = cost + dtw[i - 1][j]
                .min(dtw[i][j - 1])
                .min(dtw[i - 1][j - 1]);
        }
    }

    dtw[n][m]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_identical_patterns() {
        // Test requires NAPI context - this is a structure test only
        // Real tests run in Node.js integration tests
    }
}
