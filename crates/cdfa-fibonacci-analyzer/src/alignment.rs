//! Fibonacci level alignment scoring

use wide::f64x4;

/// Scorer for Fibonacci level alignments
#[derive(Debug)]
pub struct AlignmentScorer {
    threshold: f64,
    weight_decay: f64,
}

impl AlignmentScorer {
    pub fn new(threshold: f64, weight_decay: f64) -> Self {
        Self { threshold, weight_decay }
    }
    
    /// Score alignment of multiple Fibonacci levels
    pub fn score_alignment(&self, levels: &[f64], price: f64) -> f64 {
        let mut score = 0.0;
        let mut weight = 1.0;
        
        // Process levels in groups of 4 for SIMD
        let chunks = levels.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let levels_vec = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let price_vec = f64x4::splat(price);
            let diff = (levels_vec - price_vec).abs();
            let threshold_vec = f64x4::splat(self.threshold);
            
            // Calculate scores for each level
            let scores = threshold_vec - diff;
            let scores_array = scores.to_array();
            
            for s in scores_array {
                if s > 0.0 {
                    score += s * weight;
                    weight *= self.weight_decay;
                }
            }
        }
        
        // Process remaining levels
        for &level in remainder {
            let diff = (level - price).abs();
            if diff < self.threshold {
                score += (self.threshold - diff) * weight;
                weight *= self.weight_decay;
            }
        }
        
        score
    }
    
    /// Find clusters of aligned Fibonacci levels
    pub fn find_clusters(&self, levels: &[f64], min_cluster_size: usize) -> Vec<(f64, usize)> {
        let mut clusters = Vec::new();
        let mut i = 0;
        
        while i < levels.len() {
            let mut cluster_size = 1;
            let cluster_center = levels[i];
            
            for j in (i + 1)..levels.len() {
                if (levels[j] - cluster_center).abs() < self.threshold {
                    cluster_size += 1;
                } else {
                    break;
                }
            }
            
            if cluster_size >= min_cluster_size {
                let avg = levels[i..i + cluster_size].iter().sum::<f64>() / cluster_size as f64;
                clusters.push((avg, cluster_size));
                i += cluster_size;
            } else {
                i += 1;
            }
        }
        
        clusters
    }
    
    /// Calculate alignment score for current price relative to retracements and extensions
    pub fn calculate_alignment(
        &self,
        current_price: &f64,
        retracements: &crate::core::RetracementLevels,
        extensions: &crate::core::ExtensionLevels,
    ) -> Result<f64, crate::FibonacciError> {
        let mut all_levels = Vec::new();
        
        // Collect all retracement levels
        for (_, &level) in &retracements.levels {
            all_levels.push(level);
        }
        
        // Collect all extension levels
        for (_, &level) in &extensions.levels {
            all_levels.push(level);
        }
        
        // Sort levels for clustering
        all_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate alignment score based on proximity to levels
        let score = self.score_alignment(&all_levels, *current_price);
        
        Ok(score)
    }
}

impl Default for AlignmentScorer {
    fn default() -> Self {
        Self::new(0.0025, 0.8) // 0.25% threshold, 0.8 decay
    }
}