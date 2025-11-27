//! Quantile Calibration Utilities
//!
//! Provides utilities for quantile estimation, validation, and diagnostics.

use std::cmp::Ordering;

/// Compute empirical quantile from data
///
/// Uses the weighted average method for non-integer indices.
///
/// # Arguments
/// * `data` - Input data (will be sorted)
/// * `quantile` - Quantile level in [0, 1]
///
/// # Returns
/// Estimated quantile value
///
/// # Panics
/// Panics if data is empty or quantile not in [0, 1]
pub fn compute_quantile(data: &[f32], quantile: f32) -> f32 {
    assert!(!data.is_empty(), "Cannot compute quantile of empty data");
    assert!(
        (0.0..=1.0).contains(&quantile),
        "Quantile must be in [0, 1]"
    );

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = sorted.len();

    if quantile == 0.0 {
        return sorted[0];
    }
    if quantile == 1.0 {
        return sorted[n - 1];
    }

    // Linear interpolation between order statistics
    let idx_f = quantile * (n - 1) as f32;
    let idx_lo = idx_f.floor() as usize;
    let idx_hi = idx_f.ceil() as usize;

    if idx_lo == idx_hi {
        sorted[idx_lo]
    } else {
        let weight = idx_f - idx_lo as f32;
        sorted[idx_lo] * (1.0 - weight) + sorted[idx_hi] * weight
    }
}

/// Compute multiple quantiles efficiently
///
/// Sorts data once then computes all requested quantiles.
///
/// # Arguments
/// * `data` - Input data
/// * `quantiles` - Quantile levels to compute
///
/// # Returns
/// Vector of estimated quantile values
pub fn compute_quantiles(data: &[f32], quantiles: &[f32]) -> Vec<f32> {
    assert!(!data.is_empty());

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    quantiles
        .iter()
        .map(|&q| quantile_from_sorted(&sorted, q))
        .collect()
}

/// Compute quantile from already sorted data
fn quantile_from_sorted(sorted: &[f32], quantile: f32) -> f32 {
    let n = sorted.len();

    if quantile == 0.0 {
        return sorted[0];
    }
    if quantile == 1.0 {
        return sorted[n - 1];
    }

    let idx_f = quantile * (n - 1) as f32;
    let idx_lo = idx_f.floor() as usize;
    let idx_hi = idx_f.ceil() as usize;

    if idx_lo == idx_hi {
        sorted[idx_lo]
    } else {
        let weight = idx_f - idx_lo as f32;
        sorted[idx_lo] * (1.0 - weight) + sorted[idx_hi] * weight
    }
}

/// Validate coverage on test set
///
/// # Arguments
/// * `y_true` - True values
/// * `intervals` - Prediction intervals as (lower, upper) tuples
///
/// # Returns
/// Empirical coverage rate
pub fn validate_coverage(y_true: &[f32], intervals: &[(f32, f32)]) -> f32 {
    assert_eq!(y_true.len(), intervals.len());

    let covered = y_true
        .iter()
        .zip(intervals.iter())
        .filter(|(&y, &(lo, hi))| lo <= y && y <= hi)
        .count();

    covered as f32 / y_true.len() as f32
}

/// Compute interval width statistics
///
/// # Returns
/// (mean, median, std_dev) of interval widths
pub fn interval_width_stats(intervals: &[(f32, f32)]) -> (f32, f32, f32) {
    let widths: Vec<f32> = intervals
        .iter()
        .map(|(lo, hi)| hi - lo)
        .collect();

    let mean = widths.iter().sum::<f32>() / widths.len() as f32;
    let median = compute_quantile(&widths, 0.5);

    let variance = widths
        .iter()
        .map(|w| (w - mean).powi(2))
        .sum::<f32>() / widths.len() as f32;
    let std_dev = variance.sqrt();

    (mean, median, std_dev)
}

/// Stratified coverage analysis
///
/// Splits data into bins and computes coverage per bin to detect
/// conditional coverage failures.
///
/// # Arguments
/// * `predictions` - Point predictions (used for binning)
/// * `y_true` - True values
/// * `intervals` - Prediction intervals
/// * `n_bins` - Number of bins for stratification
///
/// # Returns
/// Coverage rate per bin
pub fn stratified_coverage(
    predictions: &[f32],
    y_true: &[f32],
    intervals: &[(f32, f32)],
    n_bins: usize,
) -> Vec<f32> {
    assert_eq!(predictions.len(), y_true.len());
    assert_eq!(predictions.len(), intervals.len());

    // Determine bin edges
    let min_pred = predictions
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let max_pred = predictions
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let bin_width = (max_pred - min_pred) / n_bins as f32;

    // Compute coverage per bin
    let mut bin_coverages = vec![0.0; n_bins];
    let mut bin_counts = vec![0; n_bins];

    for ((&pred, &y), &(lo, hi)) in predictions
        .iter()
        .zip(y_true.iter())
        .zip(intervals.iter())
    {
        let bin_idx = ((pred - min_pred) / bin_width)
            .floor()
            .min((n_bins - 1) as f32) as usize;

        bin_counts[bin_idx] += 1;
        if lo <= y && y <= hi {
            bin_coverages[bin_idx] += 1.0;
        }
    }

    // Normalize by bin counts
    bin_coverages
        .iter()
        .zip(bin_counts.iter())
        .map(|(cov, count)| {
            if *count > 0 {
                cov / *count as f32
            } else {
                f32::NAN
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(compute_quantile(&data, 0.0), 1.0);
        assert_eq!(compute_quantile(&data, 1.0), 5.0);
        assert_eq!(compute_quantile(&data, 0.5), 3.0);

        // Test interpolation
        let q25 = compute_quantile(&data, 0.25);
        assert!(q25 > 1.0 && q25 < 3.0);
    }

    #[test]
    fn test_multiple_quantiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantiles = vec![0.25, 0.5, 0.75];

        let results = compute_quantiles(&data, &quantiles);

        assert_eq!(results.len(), 3);
        assert!(results[0] < results[1]);
        assert!(results[1] < results[2]);
    }

    #[test]
    fn test_coverage_validation() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let intervals = vec![
            (0.5, 1.5),
            (1.5, 2.5),
            (2.5, 3.5),
            (3.5, 4.5),
            (4.5, 5.5),
        ];

        let coverage = validate_coverage(&y_true, &intervals);
        assert_eq!(coverage, 1.0);

        // Test partial coverage
        let intervals_narrow = vec![
            (1.1, 1.2), // Doesn't cover
            (1.5, 2.5),
            (2.5, 3.5),
            (3.5, 4.5),
            (4.5, 5.5),
        ];

        let coverage = validate_coverage(&y_true, &intervals_narrow);
        assert_eq!(coverage, 0.8);
    }

    #[test]
    fn test_width_statistics() {
        let intervals = vec![
            (0.0, 1.0),
            (0.0, 2.0),
            (0.0, 3.0),
            (0.0, 4.0),
            (0.0, 5.0),
        ];

        let (mean, median, std) = interval_width_stats(&intervals);

        assert_eq!(mean, 3.0);
        assert_eq!(median, 3.0);
        assert!(std > 0.0);
    }

    #[test]
    fn test_stratified_coverage() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let intervals = vec![
            (0.5, 1.5),
            (1.5, 2.5),
            (2.5, 3.5),
            (3.5, 4.5),
            (4.5, 5.5),
        ];

        let coverages = stratified_coverage(&predictions, &y_true, &intervals, 2);

        assert_eq!(coverages.len(), 2);
        // All predictions covered
        for cov in &coverages {
            if !cov.is_nan() {
                assert!(*cov >= 0.0 && *cov <= 1.0);
            }
        }
    }
}
