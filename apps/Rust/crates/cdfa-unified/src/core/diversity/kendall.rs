//! Kendall tau diversity implementation

use crate::error::Result;
use crate::types::*;
use crate::traits::DiversityMethod;

/// Kendall tau rank correlation diversity method
pub struct KendallTauDiversity;

impl KendallTauDiversity {
    pub fn new() -> Self {
        Self
    }
}

impl DiversityMethod for KendallTauDiversity {
    fn calculate(&self, data: &FloatArrayView2) -> Result<FloatArray1> {
        use crate::core::diversity::kendall_tau;
        
        let n_features = data.ncols();
        let mut diversity_scores = FloatArray1::zeros(n_features);
        
        // Calculate pairwise Kendall tau and convert to diversity scores
        for i in 0..n_features {
            let col_i = data.column(i);
            let mut sum_diversity = 0.0;
            let mut count = 0;
            
            for j in 0..n_features {
                if i != j {
                    let col_j = data.column(j);
                    let tau = kendall_tau(&col_i, &col_j)?;
                    sum_diversity += 1.0 - tau.abs(); // Convert correlation to diversity
                    count += 1;
                }
            }
            
            diversity_scores[i] = if count > 0 { sum_diversity / count as Float } else { 0.0 };
        }
        
        Ok(diversity_scores)
    }
}