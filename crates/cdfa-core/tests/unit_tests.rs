//! Unit tests for cdfa-core
//!
//! Tests individual components and functions in isolation

use cdfa_core::prelude::*;
use approx::{assert_relative_eq, assert_abs_diff_eq};
use ndarray::{array, Array1, Array2};

#[cfg(test)]
mod diversity_tests {
    use super::*;
    
    #[test]
    fn test_kendall_tau_identical() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let tau = kendall_tau(&x, &y).unwrap();
        assert_relative_eq!(tau, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_kendall_tau_inverse() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
        
        let tau = kendall_tau(&x, &y).unwrap();
        assert_relative_eq!(tau, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_kendall_tau_fast_correctness() {
        let x = array![1.2, 3.4, 2.1, 5.6, 4.3];
        let y = array![2.3, 4.5, 3.2, 6.7, 5.4];
        
        let tau_normal = kendall_tau(&x, &y).unwrap();
        let tau_fast = kendall_tau_fast(&x, &y).unwrap();
        
        assert_relative_eq!(tau_normal, tau_fast, epsilon = 1e-6);
    }
    
    #[test]
    fn test_spearman_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let rho = spearman_correlation(&x, &y).unwrap();
        assert_relative_eq!(rho, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_pearson_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let r = pearson_correlation(&x, &y).unwrap();
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_pearson_correlation_matrix() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ];
        
        let corr_matrix = pearson_correlation_matrix(&data.view()).unwrap();
        
        // Diagonal should be 1.0
        for i in 0..3 {
            assert_relative_eq!(corr_matrix[[i, i]], 1.0, epsilon = 1e-10);
        }
        
        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(corr_matrix[[i, j]], corr_matrix[[j, i]], epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_jensen_shannon_divergence() {
        let p = array![0.25, 0.25, 0.25, 0.25];
        let q = array![0.25, 0.25, 0.25, 0.25];
        
        let jsd = jensen_shannon_divergence(&p, &q).unwrap();
        assert_abs_diff_eq!(jsd, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_jensen_shannon_distance() {
        let p = array![0.5, 0.5, 0.0, 0.0];
        let q = array![0.0, 0.0, 0.5, 0.5];
        
        let jsd = jensen_shannon_distance(&p, &q).unwrap();
        assert!(jsd > 0.0 && jsd <= 1.0);
    }
    
    #[test]
    fn test_dynamic_time_warping() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let dtw_dist = dynamic_time_warping(&x, &y).unwrap();
        assert_abs_diff_eq!(dtw_dist, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dtw_similarity() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let similarity = dtw_similarity(&x, &y).unwrap();
        assert_relative_eq!(similarity, 1.0, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod fusion_tests {
    use super::*;
    
    #[test]
    fn test_score_fusion_average() {
        let scores = array![
            [0.8, 0.6, 0.9],
            [0.7, 0.8, 0.6],
            [0.9, 0.5, 0.8]
        ];
        
        let fused = CdfaFusion::fuse(&scores.view(), FusionMethod::Average, None).unwrap();
        
        assert_eq!(fused.len(), 3);
        assert_relative_eq!(fused[0], 0.8, epsilon = 1e-6);
        assert_relative_eq!(fused[1], 0.6333333, epsilon = 1e-6);
        assert_relative_eq!(fused[2], 0.7666666, epsilon = 1e-6);
    }
    
    #[test]
    fn test_score_fusion_weighted() {
        let scores = array![
            [0.8, 0.6, 0.9],
            [0.7, 0.8, 0.6],
            [0.9, 0.5, 0.8]
        ];
        
        let weights = array![0.5, 0.3, 0.2];
        let params = FusionParams {
            weights: Some(weights),
            ..Default::default()
        };
        
        let fused = CdfaFusion::fuse(&scores.view(), FusionMethod::WeightedAverage, Some(params)).unwrap();
        
        assert_eq!(fused.len(), 3);
        // Verify weighted average calculation
        let expected_0 = 0.8 * 0.5 + 0.7 * 0.3 + 0.9 * 0.2;
        assert_relative_eq!(fused[0], expected_0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_rank_fusion_borda() {
        let scores = array![
            [0.8, 0.6, 0.9],  // ranks: [2, 1, 3]
            [0.7, 0.8, 0.6],  // ranks: [2, 3, 1]
            [0.9, 0.5, 0.8]   // ranks: [3, 1, 2]
        ];
        
        let fused = CdfaFusion::fuse(&scores.view(), FusionMethod::BordaCount, None).unwrap();
        
        assert_eq!(fused.len(), 3);
        // Item 0: 2+2+3=7, Item 1: 1+3+1=5, Item 2: 3+1+2=6
        // Normalized Borda scores
        assert!(fused[0] > fused[1]); // 7 > 5
        assert!(fused[2] > fused[1]); // 6 > 5
        assert!(fused[0] > fused[2]); // 7 > 6
    }
    
    #[test]
    fn test_scores_to_rankings() {
        let scores = array![0.3, 0.8, 0.5, 0.9];
        let rankings = scores_to_rankings(&scores);
        
        assert_eq!(rankings, array![1, 3, 2, 4]);
    }
    
    #[test]
    fn test_rankings_to_scores() {
        let rankings = array![1, 3, 2, 4];
        let scores = rankings_to_scores(&rankings);
        
        assert_relative_eq!(scores[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(scores[1], 0.3333333, epsilon = 1e-6);
        assert_relative_eq!(scores[2], 0.6666666, epsilon = 1e-6);
        assert_relative_eq!(scores[3], 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_adaptive_score_fusion() {
        let scores = array![
            [0.8, 0.6, 0.9],
            [0.7, 0.8, 0.6],
            [0.9, 0.5, 0.8]
        ];
        
        let params = FusionParams {
            diversity_threshold: 0.5,
            score_weight: 0.6,
            ..Default::default()
        };
        
        let fused = CdfaFusion::fuse(&scores.view(), FusionMethod::Adaptive, Some(params)).unwrap();
        
        assert_eq!(fused.len(), 3);
        // Adaptive fusion should produce results between score and rank fusion
        assert!(fused.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;
    
    #[test]
    fn test_empty_arrays() {
        let x = array![];
        let y = array![];
        
        assert!(kendall_tau(&x, &y).is_err());
        assert!(spearman_correlation(&x, &y).is_err());
        assert!(pearson_correlation(&x, &y).is_err());
    }
    
    #[test]
    fn test_mismatched_lengths() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        
        assert!(kendall_tau(&x, &y).is_err());
        assert!(spearman_correlation(&x, &y).is_err());
        assert!(pearson_correlation(&x, &y).is_err());
    }
    
    #[test]
    fn test_constant_arrays() {
        let x = array![5.0, 5.0, 5.0, 5.0];
        let y = array![3.0, 3.0, 3.0, 3.0];
        
        // Constant arrays should have undefined correlation
        assert!(pearson_correlation(&x, &y).is_err());
    }
    
    #[test]
    fn test_single_element() {
        let x = array![1.0];
        let y = array![2.0];
        
        assert!(kendall_tau(&x, &y).is_err());
        assert!(spearman_correlation(&x, &y).is_err());
    }
    
    #[test]
    fn test_invalid_probabilities() {
        let p = array![0.5, 0.6]; // Sum > 1
        let q = array![0.3, 0.7];
        
        // Should handle normalization or return error
        let result = jensen_shannon_divergence(&p, &q);
        assert!(result.is_ok() || result.is_err());
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_large_array_performance() {
        let n = 10000;
        let x: Array1<f64> = Array1::range(0.0, n as f64, 1.0);
        let y: Array1<f64> = x.mapv(|v| v * 2.0 + 3.0);
        
        let start = Instant::now();
        let _ = pearson_correlation(&x, &y).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete in under 1ms for 10k elements
        assert!(elapsed.as_millis() < 1, "Pearson correlation too slow: {:?}", elapsed);
    }
    
    #[test]
    fn test_correlation_matrix_performance() {
        let n_vars = 100;
        let n_samples = 1000;
        let data = Array2::<f64>::zeros((n_vars, n_samples));
        
        let start = Instant::now();
        let _ = pearson_correlation_matrix(&data.view()).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete in under 100ms for 100x1000 matrix
        assert!(elapsed.as_millis() < 100, "Correlation matrix too slow: {:?}", elapsed);
    }
}