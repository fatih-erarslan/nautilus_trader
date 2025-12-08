//! Diversity metrics for CDFA
//! 
//! This module provides various correlation and distance measures
//! to quantify diversity between different data streams or rankings.

pub mod kendall_tau;
pub mod spearman;
pub mod pearson;
pub mod jensen_shannon;
pub mod dtw;

// Re-export main functions
pub use kendall_tau::{kendall_tau, kendall_tau_fast};
pub use spearman::{spearman_correlation, spearman_correlation_fast};
pub use pearson::{pearson_correlation, pearson_correlation_fast, pearson_correlation_matrix, partial_correlation};
pub use jensen_shannon::{jensen_shannon_divergence, jensen_shannon_distance, jensen_shannon_divergence_empirical};
pub use dtw::{dynamic_time_warping, dynamic_time_warping_window, fast_dtw, dtw_similarity};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_all_metrics_available() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        // Test all metrics are accessible
        let _ = kendall_tau(&x.view(), &y.view()).unwrap();
        let _ = spearman_correlation(&x.view(), &y.view()).unwrap();
        let _ = pearson_correlation(&x.view(), &y.view()).unwrap();
        let _ = dynamic_time_warping(&x.view(), &y.view()).unwrap();
        
        // JS divergence needs probability distributions
        let p = array![0.2, 0.2, 0.2, 0.2, 0.2];
        let q = array![0.1, 0.2, 0.3, 0.2, 0.2];
        let _ = jensen_shannon_divergence(&p.view(), &q.view()).unwrap();
    }
}