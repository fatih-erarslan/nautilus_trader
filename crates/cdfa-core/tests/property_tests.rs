//! Property-based tests for cdfa-core using quickcheck
//!
//! Tests mathematical properties and invariants

use cdfa_core::prelude::*;
use ndarray::{Array1, Array2};
use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};
use approx::relative_eq;

#[derive(Clone, Debug)]
struct ValidSignal(Array1<f64>);

impl Arbitrary for ValidSignal {
    fn arbitrary(g: &mut Gen) -> Self {
        let size = usize::arbitrary(g) % 100 + 2; // 2-101 elements
        let values: Vec<f64> = (0..size)
            .map(|_| f64::arbitrary(g) % 10.0 - 5.0) // Range [-5, 5]
            .collect();
        ValidSignal(Array1::from_vec(values))
    }
}

#[derive(Clone, Debug)]
struct ProbabilityDistribution(Array1<f64>);

impl Arbitrary for ProbabilityDistribution {
    fn arbitrary(g: &mut Gen) -> Self {
        let size = usize::arbitrary(g) % 50 + 2; // 2-51 elements
        let values: Vec<f64> = (0..size)
            .map(|_| (f64::arbitrary(g) % 100.0).abs() + 1e-10)
            .collect();
        let sum: f64 = values.iter().sum();
        let normalized: Vec<f64> = values.iter().map(|&x| x / sum).collect();
        ProbabilityDistribution(Array1::from_vec(normalized))
    }
}

#[derive(Clone, Debug)]
struct ScoreMatrix(Array2<f64>);

impl Arbitrary for ScoreMatrix {
    fn arbitrary(g: &mut Gen) -> Self {
        let n_sources = (usize::arbitrary(g) % 8) + 2; // 2-9 sources
        let n_items = (usize::arbitrary(g) % 20) + 3;   // 3-22 items
        let values: Vec<f64> = (0..(n_sources * n_items))
            .map(|_| f64::arbitrary(g) % 1.0)
            .map(|x| x.abs()) // Ensure positive scores
            .collect();
        
        let matrix = Array2::from_shape_vec((n_sources, n_items), values).unwrap();
        ScoreMatrix(matrix)
    }
}

mod diversity_properties {
    use super::*;
    
    #[test]
    fn prop_kendall_tau_bounds() {
        fn check(ValidSignal(x): ValidSignal, ValidSignal(y): ValidSignal) -> TestResult {
            if x.len() != y.len() || x.len() < 2 {
                return TestResult::discard();
            }
            
            match kendall_tau(&x, &y) {
                Ok(tau) => TestResult::from_bool(-1.0 <= tau && tau <= 1.0),
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ValidSignal, ValidSignal) -> TestResult);
    }
    
    #[test]
    fn prop_kendall_tau_symmetry() {
        fn check(ValidSignal(x): ValidSignal, ValidSignal(y): ValidSignal) -> TestResult {
            if x.len() != y.len() || x.len() < 2 {
                return TestResult::discard();
            }
            
            match (kendall_tau(&x, &y), kendall_tau(&y, &x)) {
                (Ok(tau_xy), Ok(tau_yx)) => {
                    TestResult::from_bool(relative_eq!(tau_xy, tau_yx, epsilon = 1e-10))
                },
                _ => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ValidSignal, ValidSignal) -> TestResult);
    }
    
    #[test]
    fn prop_spearman_correlation_bounds() {
        fn check(ValidSignal(x): ValidSignal, ValidSignal(y): ValidSignal) -> TestResult {
            if x.len() != y.len() || x.len() < 2 {
                return TestResult::discard();
            }
            
            match spearman_correlation(&x, &y) {
                Ok(rho) => TestResult::from_bool(-1.0 <= rho && rho <= 1.0),
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ValidSignal, ValidSignal) -> TestResult);
    }
    
    #[test]
    fn prop_pearson_self_correlation() {
        fn check(ValidSignal(x): ValidSignal) -> TestResult {
            if x.len() < 2 || x.std(0.0) < 1e-10 {
                return TestResult::discard();
            }
            
            match pearson_correlation(&x, &x) {
                Ok(r) => TestResult::from_bool(relative_eq!(r, 1.0, epsilon = 1e-10)),
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ValidSignal) -> TestResult);
    }
    
    #[test]
    fn prop_jensen_shannon_non_negative() {
        fn check(ProbabilityDistribution(p): ProbabilityDistribution, 
                 ProbabilityDistribution(q): ProbabilityDistribution) -> TestResult {
            if p.len() != q.len() {
                return TestResult::discard();
            }
            
            match jensen_shannon_divergence(&p, &q) {
                Ok(jsd) => TestResult::from_bool(jsd >= 0.0),
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ProbabilityDistribution, ProbabilityDistribution) -> TestResult);
    }
    
    #[test]
    fn prop_jensen_shannon_symmetry() {
        fn check(ProbabilityDistribution(p): ProbabilityDistribution,
                 ProbabilityDistribution(q): ProbabilityDistribution) -> TestResult {
            if p.len() != q.len() {
                return TestResult::discard();
            }
            
            match (jensen_shannon_divergence(&p, &q), jensen_shannon_divergence(&q, &p)) {
                (Ok(jsd_pq), Ok(jsd_qp)) => {
                    TestResult::from_bool(relative_eq!(jsd_pq, jsd_qp, epsilon = 1e-10))
                },
                _ => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ProbabilityDistribution, ProbabilityDistribution) -> TestResult);
    }
    
    #[test]
    fn prop_dtw_non_negative() {
        fn check(ValidSignal(x): ValidSignal, ValidSignal(y): ValidSignal) -> TestResult {
            if x.is_empty() || y.is_empty() {
                return TestResult::discard();
            }
            
            match dynamic_time_warping(&x, &y) {
                Ok(dist) => TestResult::from_bool(dist >= 0.0),
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ValidSignal, ValidSignal) -> TestResult);
    }
    
    #[test]
    fn prop_dtw_self_distance_zero() {
        fn check(ValidSignal(x): ValidSignal) -> TestResult {
            if x.is_empty() {
                return TestResult::discard();
            }
            
            match dynamic_time_warping(&x, &x) {
                Ok(dist) => TestResult::from_bool(relative_eq!(dist, 0.0, epsilon = 1e-10)),
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ValidSignal) -> TestResult);
    }
}

mod fusion_properties {
    use super::*;
    
    #[test]
    fn prop_fusion_preserves_bounds() {
        fn check(ScoreMatrix(scores): ScoreMatrix) -> TestResult {
            if scores.is_empty() {
                return TestResult::discard();
            }
            
            // Normalize scores to [0, 1]
            let max_val = scores.fold(0.0f64, |a, &b| a.max(b));
            let normalized = if max_val > 0.0 {
                scores / max_val
            } else {
                scores.clone()
            };
            
            match CdfaFusion::fuse(&normalized.view(), FusionMethod::Average, None) {
                Ok(fused) => {
                    let all_in_bounds = fused.iter().all(|&x| 0.0 <= x && x <= 1.0);
                    TestResult::from_bool(all_in_bounds)
                },
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ScoreMatrix) -> TestResult);
    }
    
    #[test]
    fn prop_weighted_fusion_respects_weights() {
        fn check(ScoreMatrix(scores): ScoreMatrix) -> TestResult {
            if scores.is_empty() {
                return TestResult::discard();
            }
            
            let n_sources = scores.nrows();
            let weights = Array1::ones(n_sources) / n_sources as f64;
            
            let params = FusionParams {
                weights: Some(weights.clone()),
                ..Default::default()
            };
            
            match (
                CdfaFusion::fuse(&scores.view(), FusionMethod::WeightedAverage, Some(params)),
                CdfaFusion::fuse(&scores.view(), FusionMethod::Average, None)
            ) {
                (Ok(weighted), Ok(average)) => {
                    // With equal weights, should be same as average
                    let close = weighted.iter().zip(average.iter())
                        .all(|(w, a)| relative_eq!(w, a, epsilon = 1e-6));
                    TestResult::from_bool(close)
                },
                _ => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ScoreMatrix) -> TestResult);
    }
    
    #[test]
    fn prop_borda_count_ordering() {
        fn check(ScoreMatrix(scores): ScoreMatrix) -> TestResult {
            if scores.nrows() < 2 || scores.ncols() < 2 {
                return TestResult::discard();
            }
            
            match CdfaFusion::fuse(&scores.view(), FusionMethod::BordaCount, None) {
                Ok(fused) => {
                    // Borda scores should be non-negative
                    let all_non_negative = fused.iter().all(|&x| x >= 0.0);
                    TestResult::from_bool(all_non_negative)
                },
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ScoreMatrix) -> TestResult);
    }
    
    #[test]
    fn prop_scores_to_rankings_consistency() {
        fn check(ValidSignal(scores): ValidSignal) -> TestResult {
            if scores.is_empty() {
                return TestResult::discard();
            }
            
            let rankings = scores_to_rankings(&scores);
            
            // Check rankings are valid (1 to n)
            let n = scores.len();
            let valid_range = rankings.iter().all(|&r| 1 <= r && r <= n);
            
            // Check all ranks are unique
            let mut sorted_ranks = rankings.to_vec();
            sorted_ranks.sort();
            let all_unique = sorted_ranks.windows(2).all(|w| w[0] != w[1]);
            
            TestResult::from_bool(valid_range && all_unique)
        }
        
        quickcheck(check as fn(ValidSignal) -> TestResult);
    }
    
    #[test]
    fn prop_rankings_to_scores_bounds() {
        fn check(rankings: Vec<usize>) -> TestResult {
            if rankings.is_empty() || rankings.iter().any(|&r| r == 0) {
                return TestResult::discard();
            }
            
            let rankings_array = Array1::from_vec(rankings);
            let scores = rankings_to_scores(&rankings_array);
            
            // All scores should be in [0, 1]
            let all_in_bounds = scores.iter().all(|&s| 0.0 <= s && s <= 1.0);
            TestResult::from_bool(all_in_bounds)
        }
        
        quickcheck(check as fn(Vec<usize>) -> TestResult);
    }
}

mod correlation_matrix_properties {
    use super::*;
    
    #[test]
    fn prop_correlation_matrix_diagonal() {
        fn check(ScoreMatrix(data): ScoreMatrix) -> TestResult {
            if data.nrows() < 2 || data.ncols() < 2 {
                return TestResult::discard();
            }
            
            match pearson_correlation_matrix(&data.view()) {
                Ok(corr_matrix) => {
                    // Diagonal should be 1.0 (self-correlation)
                    let diagonal_ones = (0..corr_matrix.nrows())
                        .all(|i| relative_eq!(corr_matrix[[i, i]], 1.0, epsilon = 1e-6));
                    TestResult::from_bool(diagonal_ones)
                },
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ScoreMatrix) -> TestResult);
    }
    
    #[test]
    fn prop_correlation_matrix_symmetry() {
        fn check(ScoreMatrix(data): ScoreMatrix) -> TestResult {
            if data.nrows() < 2 || data.ncols() < 2 {
                return TestResult::discard();
            }
            
            match pearson_correlation_matrix(&data.view()) {
                Ok(corr_matrix) => {
                    let n = corr_matrix.nrows();
                    let symmetric = (0..n).all(|i| {
                        (0..n).all(|j| {
                            relative_eq!(corr_matrix[[i, j]], corr_matrix[[j, i]], epsilon = 1e-10)
                        })
                    });
                    TestResult::from_bool(symmetric)
                },
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ScoreMatrix) -> TestResult);
    }
    
    #[test]
    fn prop_correlation_matrix_bounds() {
        fn check(ScoreMatrix(data): ScoreMatrix) -> TestResult {
            if data.nrows() < 2 || data.ncols() < 2 {
                return TestResult::discard();
            }
            
            match pearson_correlation_matrix(&data.view()) {
                Ok(corr_matrix) => {
                    // All correlations should be in [-1, 1]
                    let all_in_bounds = corr_matrix.iter()
                        .all(|&r| -1.0 <= r && r <= 1.0);
                    TestResult::from_bool(all_in_bounds)
                },
                Err(_) => TestResult::discard(),
            }
        }
        
        quickcheck(check as fn(ScoreMatrix) -> TestResult);
    }
}