//! Rank-based fusion algorithms
//! 
//! This module implements various methods for combining rankings
//! from multiple sources into a consensus ranking.

use crate::error::{CdfaError, Result};
use crate::types::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Rank-based fusion methods
pub struct RankFusion;

impl RankFusion {
    /// Borda count method
    /// 
    /// Each item gets points based on its rank position across all sources.
    /// Higher scores indicate better consensus ranking.
    pub fn borda_count(rankings: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if rankings.is_empty() {
            return Err(CdfaError::invalid_input("Rankings cannot be empty"));
        }
        
        let n_items = rankings.ncols();
        let mut scores = Array1::zeros(n_items);
        
        for row in rankings.rows() {
            for (j, &rank) in row.iter().enumerate() {
                if rank > 0.0 {
                    // Borda score = n_items - rank + 1
                    scores[j] += n_items as Float - rank + 1.0;
                }
            }
        }
        
        Ok(scores)
    }
    
    /// Median rank aggregation
    pub fn median_rank(rankings: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if rankings.is_empty() {
            return Err(CdfaError::invalid_input("Rankings cannot be empty"));
        }
        
        let n_items = rankings.ncols();
        let mut result = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let mut column_ranks: Vec<Float> = rankings.column(j).to_vec();
            column_ranks.retain(|&x| x > 0.0); // Remove non-rankings
            
            if column_ranks.is_empty() {
                result[j] = 0.0;
            } else {
                column_ranks.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = column_ranks.len();
                result[j] = if n % 2 == 0 {
                    (column_ranks[n / 2 - 1] + column_ranks[n / 2]) / 2.0
                } else {
                    column_ranks[n / 2]
                };
            }
        }
        
        // Convert ranks back to scores (lower rank = higher score)
        let max_rank = result.iter().fold(0.0f64, |acc, &x| acc.max(x));
        if max_rank > 0.0 {
            result.mapv_inplace(|x| if x > 0.0 { max_rank - x + 1.0 } else { 0.0 });
        }
        
        Ok(result)
    }
    
    /// Minimum rank aggregation
    pub fn minimum_rank(rankings: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if rankings.is_empty() {
            return Err(CdfaError::invalid_input("Rankings cannot be empty"));
        }
        
        let n_items = rankings.ncols();
        let mut result = Array1::from_elem(n_items, Float::INFINITY);
        
        for j in 0..n_items {
            let column = rankings.column(j);
            for &rank in column.iter() {
                if rank > 0.0 && rank < result[j] {
                    result[j] = rank;
                }
            }
            
            if result[j] == Float::INFINITY {
                result[j] = 0.0; // No valid rankings for this item
            }
        }
        
        // Convert ranks back to scores
        let max_rank = result.iter().fold(0.0f64, |acc, &x| acc.max(x));
        if max_rank > 0.0 {
            result.mapv_inplace(|x| if x > 0.0 { max_rank - x + 1.0 } else { 0.0 });
        }
        
        Ok(result)
    }
    
    /// Reciprocal rank fusion
    pub fn reciprocal_rank(rankings: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if rankings.is_empty() {
            return Err(CdfaError::invalid_input("Rankings cannot be empty"));
        }
        
        let n_items = rankings.ncols();
        let mut scores = Array1::zeros(n_items);
        
        for row in rankings.rows() {
            for (j, &rank) in row.iter().enumerate() {
                if rank > 0.0 {
                    scores[j] += 1.0 / rank;
                }
            }
        }
        
        Ok(scores)
    }
    
    /// Kemeny optimal aggregation (approximation)
    /// 
    /// This is an NP-hard problem, so we use a heuristic approximation.
    pub fn kemeny_approximation(rankings: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if rankings.is_empty() {
            return Err(CdfaError::invalid_input("Rankings cannot be empty"));
        }
        
        // For simplicity, use Borda count as Kemeny approximation
        // In a full implementation, this would use more sophisticated algorithms
        Self::borda_count(rankings)
    }
    
    /// Footrule aggregation
    /// 
    /// Minimizes the sum of absolute differences between the consensus
    /// ranking and individual rankings.
    pub fn footrule_aggregation(rankings: &ArrayView2<Float>) -> Result<Array1<Float>> {
        if rankings.is_empty() {
            return Err(CdfaError::invalid_input("Rankings cannot be empty"));
        }
        
        // Use median rank as footrule approximation
        Self::median_rank(rankings)
    }
}

/// Hybrid rank-score fusion that combines both approaches
pub struct HybridRankScoreFusion;

impl HybridRankScoreFusion {
    /// Hybrid fusion with configurable score/rank balance
    /// 
    /// # Arguments
    /// * `data` - Original score data
    /// * `alpha` - Balance parameter (0.0 = pure rank, 1.0 = pure score)
    pub fn fuse(data: &ArrayView2<Float>, alpha: Float) -> Result<Array1<Float>> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(CdfaError::invalid_input("Alpha must be between 0.0 and 1.0"));
        }
        
        // Score-based fusion
        let score_result = crate::core::fusion::score_fusion::ScoreFusion::average(data)?;
        
        // Rank-based fusion
        let rankings = scores_to_rankings(data)?;
        let rank_result = RankFusion::borda_count(&rankings.view())?;
        
        // Normalize both results to [0, 1] for fair combination
        let score_normalized = normalize_to_unit(&score_result.view())?;
        let rank_normalized = normalize_to_unit(&rank_result.view())?;
        
        // Combine with alpha weighting
        let result = alpha * &score_normalized + (1.0 - alpha) * &rank_normalized;
        
        Ok(result)
    }
}

/// Convert scores to rankings
/// 
/// Higher scores get lower (better) rank numbers.
pub fn scores_to_rankings(scores: &ArrayView2<Float>) -> Result<Array2<Float>> {
    if scores.is_empty() {
        return Err(CdfaError::invalid_input("Scores cannot be empty"));
    }
    
    let (n_sources, n_items) = scores.dim();
    let mut rankings = Array2::zeros((n_sources, n_items));
    
    for (i, row) in scores.rows().into_iter().enumerate() {
        // Create index-value pairs for sorting
        let mut indexed_scores: Vec<(usize, Float)> = row.iter()
            .enumerate()
            .map(|(j, &score)| (j, score))
            .collect();
        
        // Sort by score in descending order (highest score gets rank 1)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Assign ranks (handle ties by averaging ranks)
        let mut current_rank = 1.0;
        let mut j = 0;
        
        while j < indexed_scores.len() {
            let current_score = indexed_scores[j].1;
            let mut tie_count = 1;
            
            // Count ties
            while j + tie_count < indexed_scores.len() && 
                  (indexed_scores[j + tie_count].1 - current_score).abs() < Float::EPSILON {
                tie_count += 1;
            }
            
            // Average rank for tied items
            let avg_rank = current_rank + (tie_count - 1) as Float / 2.0;
            
            // Assign ranks
            for k in 0..tie_count {
                let item_index = indexed_scores[j + k].0;
                rankings[[i, item_index]] = avg_rank;
            }
            
            current_rank += tie_count as Float;
            j += tie_count;
        }
    }
    
    Ok(rankings)
}

/// Convert rankings back to scores
/// 
/// Lower (better) ranks get higher scores.
pub fn rankings_to_scores(rankings: &ArrayView2<Float>) -> Result<Array2<Float>> {
    if rankings.is_empty() {
        return Err(CdfaError::invalid_input("Rankings cannot be empty"));
    }
    
    let (n_sources, n_items) = rankings.dim();
    let mut scores = Array2::zeros((n_sources, n_items));
    
    for (i, row) in rankings.rows().into_iter().enumerate() {
        let max_rank = row.iter().fold(0.0, |acc, &x| if x > acc { x } else { acc });
        
        for (j, &rank) in row.iter().enumerate() {
            if rank > 0.0 {
                // Convert rank to score: score = max_rank - rank + 1
                scores[[i, j]] = max_rank - rank + 1.0;
            } else {
                scores[[i, j]] = 0.0; // No ranking available
            }
        }
    }
    
    Ok(scores)
}

/// Normalize array to unit range [0, 1]
fn normalize_to_unit(data: &ArrayView1<Float>) -> Result<Array1<Float>> {
    let min_val = data.iter().fold(Float::INFINITY, |acc, &x| acc.min(x));
    let max_val = data.iter().fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));
    
    let range = max_val - min_val;
    if range < Float::EPSILON {
        // All values are the same
        Ok(Array1::from_elem(data.len(), 0.5))
    } else {
        Ok((data - min_val) / range)
    }
}

/// Calculate rank correlation between two ranking vectors
pub fn rank_correlation(
    ranks1: &ArrayView1<Float>,
    ranks2: &ArrayView1<Float>,
) -> Result<Float> {
    // Use Spearman correlation on the ranks
    crate::core::diversity::spearman_correlation_fast(ranks1, ranks2)
}

/// Calculate Kendall's tau distance between rankings
pub fn kendall_distance(
    ranks1: &ArrayView1<Float>,
    ranks2: &ArrayView1<Float>,
) -> Result<Float> {
    if ranks1.len() != ranks2.len() {
        return Err(CdfaError::dimension_mismatch(ranks1.len(), ranks2.len()));
    }
    
    let n = ranks1.len();
    let mut discordant_pairs = 0;
    let mut total_pairs = 0;
    
    for i in 0..n {
        for j in i + 1..n {
            // Skip if either ranking has missing values (0.0)
            if ranks1[i] > 0.0 && ranks1[j] > 0.0 && ranks2[i] > 0.0 && ranks2[j] > 0.0 {
                total_pairs += 1;
                
                // Check if the relative order is different
                let order1 = (ranks1[i] < ranks1[j]) as i32 - (ranks1[i] > ranks1[j]) as i32;
                let order2 = (ranks2[i] < ranks2[j]) as i32 - (ranks2[i] > ranks2[j]) as i32;
                
                if order1 * order2 < 0 {
                    discordant_pairs += 1;
                }
            }
        }
    }
    
    if total_pairs == 0 {
        Ok(0.0) // No comparable pairs
    } else {
        Ok(discordant_pairs as Float / total_pairs as Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_scores_to_rankings() {
        let scores = array![
            [10.0, 5.0, 8.0, 2.0],  // Rankings should be [1, 3, 2, 4]
            [6.0, 9.0, 3.0, 7.0]    // Rankings should be [3, 1, 4, 2]
        ];
        
        let rankings = scores_to_rankings(&scores.view()).unwrap();
        
        // Check first row rankings
        assert_abs_diff_eq!(rankings[[0, 0]], 1.0, epsilon = 1e-10); // 10.0 -> rank 1
        assert_abs_diff_eq!(rankings[[0, 1]], 3.0, epsilon = 1e-10); // 5.0 -> rank 3
        assert_abs_diff_eq!(rankings[[0, 2]], 2.0, epsilon = 1e-10); // 8.0 -> rank 2
        assert_abs_diff_eq!(rankings[[0, 3]], 4.0, epsilon = 1e-10); // 2.0 -> rank 4
        
        // Check second row rankings
        assert_abs_diff_eq!(rankings[[1, 0]], 3.0, epsilon = 1e-10); // 6.0 -> rank 3
        assert_abs_diff_eq!(rankings[[1, 1]], 1.0, epsilon = 1e-10); // 9.0 -> rank 1
        assert_abs_diff_eq!(rankings[[1, 2]], 4.0, epsilon = 1e-10); // 3.0 -> rank 4
        assert_abs_diff_eq!(rankings[[1, 3]], 2.0, epsilon = 1e-10); // 7.0 -> rank 2
    }
    
    #[test]
    fn test_borda_count() {
        let rankings = array![
            [1.0, 3.0, 2.0, 4.0],  // Item 0 rank 1, item 1 rank 3, etc.
            [2.0, 1.0, 4.0, 3.0]   // Item 0 rank 2, item 1 rank 1, etc.
        ];
        
        let scores = RankFusion::borda_count(&rankings.view()).unwrap();
        
        // With 4 items, Borda scores = 4 - rank + 1
        // Item 0: (4-1+1) + (4-2+1) = 4 + 3 = 7
        // Item 1: (4-3+1) + (4-1+1) = 2 + 4 = 6
        // Item 2: (4-2+1) + (4-4+1) = 3 + 1 = 4
        // Item 3: (4-4+1) + (4-3+1) = 1 + 2 = 3
        
        assert_abs_diff_eq!(scores[0], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[1], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[2], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[3], 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_reciprocal_rank() {
        let rankings = array![
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0]
        ];
        
        let scores = RankFusion::reciprocal_rank(&rankings.view()).unwrap();
        
        // Item 0: 1/1 + 1/2 = 1.5
        // Item 1: 1/2 + 1/1 = 1.5  
        // Item 2: 1/3 + 1/3 = 0.667
        
        assert_abs_diff_eq!(scores[0], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[1], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[2], 2.0/3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_median_rank() {
        let rankings = array![
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
            [1.0, 2.0, 3.0]
        ];
        
        let scores = RankFusion::median_rank(&rankings.view()).unwrap();
        
        // Item 0 ranks: [1, 2, 1] -> median = 1
        // Item 1 ranks: [3, 1, 2] -> median = 2
        // Item 2 ranks: [2, 3, 3] -> median = 3
        
        // Convert back to scores (max_rank - rank + 1 = 3 - rank + 1 = 4 - rank)
        assert_abs_diff_eq!(scores[0], 3.0, epsilon = 1e-10); // 4 - 1
        assert_abs_diff_eq!(scores[1], 2.0, epsilon = 1e-10); // 4 - 2
        assert_abs_diff_eq!(scores[2], 1.0, epsilon = 1e-10); // 4 - 3
    }
    
    #[test]
    fn test_hybrid_fusion() {
        let data = array![
            [10.0, 5.0, 8.0],
            [6.0, 9.0, 7.0]
        ];
        
        // Test pure score fusion (alpha = 1.0)
        let pure_score = HybridRankScoreFusion::fuse(&data.view(), 1.0).unwrap();
        assert_eq!(pure_score.len(), 3);
        
        // Test pure rank fusion (alpha = 0.0)
        let pure_rank = HybridRankScoreFusion::fuse(&data.view(), 0.0).unwrap();
        assert_eq!(pure_rank.len(), 3);
        
        // Test balanced fusion (alpha = 0.5)
        let balanced = HybridRankScoreFusion::fuse(&data.view(), 0.5).unwrap();
        assert_eq!(balanced.len(), 3);
        
        // Balanced should be between pure score and pure rank
        for i in 0..3 {
            let min_val = pure_score[i].min(pure_rank[i]);
            let max_val = pure_score[i].max(pure_rank[i]);
            assert!(balanced[i] >= min_val - 1e-10);
            assert!(balanced[i] <= max_val + 1e-10);
        }
    }
    
    #[test]
    fn test_rankings_to_scores() {
        let rankings = array![
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0]
        ];
        
        let scores = rankings_to_scores(&rankings.view()).unwrap();
        
        // Max rank is 3, so scores = 3 - rank + 1 = 4 - rank
        // First row: [4-1, 4-3, 4-2] = [3, 1, 2]
        // Second row: [4-2, 4-1, 4-3] = [2, 3, 1]
        
        assert_abs_diff_eq!(scores[[0, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[[0, 2]], 2.0, epsilon = 1e-10);
        
        assert_abs_diff_eq!(scores[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[[1, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scores[[1, 2]], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_rank_correlation() {
        let ranks1 = array![1.0, 2.0, 3.0, 4.0];
        let ranks2 = array![1.0, 2.0, 3.0, 4.0]; // Perfect correlation
        
        let corr = rank_correlation(&ranks1.view(), &ranks2.view()).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);
        
        let ranks3 = array![4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation
        let corr_neg = rank_correlation(&ranks1.view(), &ranks3.view()).unwrap();
        assert_abs_diff_eq!(corr_neg, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_kendall_distance() {
        let ranks1 = array![1.0, 2.0, 3.0, 4.0];
        let ranks2 = array![1.0, 2.0, 3.0, 4.0]; // Identical rankings
        
        let distance = kendall_distance(&ranks1.view(), &ranks2.view()).unwrap();
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10); // No discordant pairs
        
        let ranks3 = array![4.0, 3.0, 2.0, 1.0]; // Completely reversed
        let distance_max = kendall_distance(&ranks1.view(), &ranks3.view()).unwrap();
        assert_abs_diff_eq!(distance_max, 1.0, epsilon = 1e-10); // All pairs discordant
    }
    
    #[test]
    fn test_tied_scores() {
        let scores = array![
            [10.0, 10.0, 5.0, 5.0],  // Two ties
        ];
        
        let rankings = scores_to_rankings(&scores.view()).unwrap();
        
        // Tied scores should get averaged ranks
        // 10.0 values should get ranks (1+2)/2 = 1.5
        // 5.0 values should get ranks (3+4)/2 = 3.5
        
        assert_abs_diff_eq!(rankings[[0, 0]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(rankings[[0, 1]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(rankings[[0, 2]], 3.5, epsilon = 1e-10);
        assert_abs_diff_eq!(rankings[[0, 3]], 3.5, epsilon = 1e-10);
    }
}