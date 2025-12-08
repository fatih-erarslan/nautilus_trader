use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Rank-based fusion methods for combining multiple rankings
pub struct RankFusion;

impl RankFusion {
    /// Borda Count fusion
    /// 
    /// Each item gets points based on its rank position
    pub fn borda_count(rankings: &ArrayView2<usize>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = rankings.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Rankings matrix cannot be empty");
        }
        
        let mut borda_scores = Array1::zeros(n_items);
        
        for i in 0..n_sources {
            for j in 0..n_items {
                // Borda score: n_items - rank
                // Assumes ranks are 1-based
                let rank = rankings[[i, j]];
                if rank == 0 || rank > n_items {
                    return Err("Invalid rank value");
                }
                borda_scores[j] += (n_items - rank + 1) as f64;
            }
        }
        
        Ok(borda_scores)
    }
    
    /// Median Rank fusion
    /// 
    /// Takes the median rank for each item across all sources
    pub fn median_rank(rankings: &ArrayView2<usize>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = rankings.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Rankings matrix cannot be empty");
        }
        
        let mut median_ranks = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let mut ranks: Vec<usize> = rankings.column(j).to_vec();
            ranks.sort();
            
            let median = if n_sources % 2 == 0 {
                (ranks[n_sources / 2 - 1] + ranks[n_sources / 2]) as f64 / 2.0
            } else {
                ranks[n_sources / 2] as f64
            };
            
            median_ranks[j] = median;
        }
        
        // Convert to scores (lower rank = higher score)
        let max_rank = median_ranks.fold(0.0f64, |a, &b| a.max(b));
        Ok(max_rank + 1.0 - median_ranks)
    }
    
    /// Minimum Rank fusion
    /// 
    /// Takes the best (minimum) rank for each item
    pub fn minimum_rank(rankings: &ArrayView2<usize>) -> Result<Array1<f64>, &'static str> {
        let (_, n_items) = rankings.dim();
        
        if rankings.is_empty() {
            return Err("Rankings matrix cannot be empty");
        }
        
        let mut min_ranks = Array1::zeros(n_items);
        
        for j in 0..n_items {
            let min_rank = rankings.column(j).fold(usize::MAX, |a, &b| a.min(b));
            min_ranks[j] = min_rank as f64;
        }
        
        // Convert to scores
        let max_rank = min_ranks.fold(0.0f64, |a, &b| a.max(b));
        Ok(max_rank + 1.0 - min_ranks)
    }
    
    /// Reciprocal Rank fusion
    /// 
    /// Uses 1/rank as the score for each item
    pub fn reciprocal_rank(rankings: &ArrayView2<usize>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = rankings.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Rankings matrix cannot be empty");
        }
        
        let mut rr_scores = Array1::zeros(n_items);
        
        for i in 0..n_sources {
            for j in 0..n_items {
                let rank = rankings[[i, j]];
                if rank == 0 {
                    return Err("Ranks must be positive");
                }
                rr_scores[j] += 1.0 / rank as f64;
            }
        }
        
        Ok(rr_scores)
    }
    
    /// Weighted Borda Count
    /// 
    /// Borda count with source-specific weights
    pub fn weighted_borda_count(
        rankings: &ArrayView2<usize>, 
        weights: &ArrayView1<f64>
    ) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = rankings.dim();
        
        if weights.len() != n_sources {
            return Err("Number of weights must match number of sources");
        }
        
        if n_sources == 0 || n_items == 0 {
            return Err("Rankings matrix cannot be empty");
        }
        
        let normalized_weights = weights / weights.sum();
        let mut weighted_scores = Array1::zeros(n_items);
        
        for i in 0..n_sources {
            for j in 0..n_items {
                let rank = rankings[[i, j]];
                if rank == 0 || rank > n_items {
                    return Err("Invalid rank value");
                }
                let borda_score = (n_items - rank + 1) as f64;
                weighted_scores[j] += normalized_weights[i] * borda_score;
            }
        }
        
        Ok(weighted_scores)
    }
    
    /// Kemeny optimal aggregation (approximation)
    /// 
    /// Finds a consensus ranking that minimizes total disagreement
    pub fn kemeny_approximation(rankings: &ArrayView2<usize>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = rankings.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Rankings matrix cannot be empty");
        }
        
        // Build pairwise preference matrix
        let mut pairwise_prefs = Array2::<f64>::zeros((n_items, n_items));
        
        for source in 0..n_sources {
            for i in 0..n_items {
                for j in 0..n_items {
                    if i != j {
                        let rank_i = rankings[[source, i]];
                        let rank_j = rankings[[source, j]];
                        
                        if rank_i < rank_j {
                            pairwise_prefs[[i, j]] += 1.0;
                        }
                    }
                }
            }
        }
        
        // Compute Kemeny scores (simplified)
        let mut kemeny_scores = Array1::zeros(n_items);
        
        for i in 0..n_items {
            for j in 0..n_items {
                if i != j {
                    kemeny_scores[i] += pairwise_prefs[[i, j]];
                }
            }
        }
        
        Ok(kemeny_scores)
    }
    
    /// Footrule optimal aggregation
    /// 
    /// Minimizes the sum of rank distances
    pub fn footrule_aggregation(rankings: &ArrayView2<usize>) -> Result<Array1<f64>, &'static str> {
        let (n_sources, n_items) = rankings.dim();
        
        if n_sources == 0 || n_items == 0 {
            return Err("Rankings matrix cannot be empty");
        }
        
        // For each possible consensus ranking position, find the item that minimizes distance
        let mut consensus_scores = Array1::zeros(n_items);
        
        for target_rank in 1..=n_items {
            let mut min_distance = f64::INFINITY;
            let mut best_item = 0;
            
            for item in 0..n_items {
                let mut total_distance = 0.0;
                
                for source in 0..n_sources {
                    let actual_rank = rankings[[source, item]];
                    total_distance += (actual_rank as f64 - target_rank as f64).abs();
                }
                
                if total_distance < min_distance {
                    min_distance = total_distance;
                    best_item = item;
                }
            }
            
            // Higher score for lower target rank
            consensus_scores[best_item] = (n_items - target_rank + 1) as f64;
        }
        
        Ok(consensus_scores)
    }
}

/// Convert scores to rankings
pub fn scores_to_rankings(scores: &ArrayView2<f64>) -> Array2<usize> {
    let (n_sources, n_items) = scores.dim();
    let mut rankings = Array2::zeros((n_sources, n_items));
    
    for i in 0..n_sources {
        let row_scores = scores.row(i);
        let mut indexed: Vec<(usize, f64)> = row_scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        
        // Sort by score (descending)
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Assign ranks
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            rankings[[i, *idx]] = rank + 1; // 1-based ranking
        }
    }
    
    rankings
}

/// Convert rankings to scores (using Borda count)
pub fn rankings_to_scores(rankings: &ArrayView2<usize>) -> Result<Array2<f64>, &'static str> {
    let (n_sources, n_items) = rankings.dim();
    let mut scores = Array2::zeros((n_sources, n_items));
    
    for i in 0..n_sources {
        for j in 0..n_items {
            let rank = rankings[[i, j]];
            if rank == 0 || rank > n_items {
                return Err("Invalid rank value");
            }
            scores[[i, j]] = (n_items - rank + 1) as f64;
        }
    }
    
    Ok(scores)
}

/// Hybrid rank-score fusion
pub struct HybridRankScoreFusion;

impl HybridRankScoreFusion {
    /// Combine rank and score information
    pub fn fuse(
        scores: &ArrayView2<f64>, 
        alpha: f64
    ) -> Result<Array1<f64>, &'static str> {
        if alpha < 0.0 || alpha > 1.0 {
            return Err("Alpha must be between 0 and 1");
        }
        
        // Get score-based fusion
        let score_fusion = crate::fusion::score_fusion::ScoreFusion::normalized_average(scores)?;
        
        // Convert to rankings and get rank-based fusion
        let rankings = scores_to_rankings(scores);
        let rank_fusion = RankFusion::borda_count(&rankings.view())?;
        
        // Normalize rank fusion  
        let rank_sum = rank_fusion.sum();
        let score_sum = score_fusion.sum();
        let rank_fusion_norm = rank_fusion / rank_sum;
        let score_fusion_norm = score_fusion / score_sum;
        
        // Combine using alpha parameter
        Ok(alpha * score_fusion_norm + (1.0 - alpha) * rank_fusion_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_borda_count() {
        let rankings = array![
            [1, 2, 3],
            [2, 1, 3],
            [1, 3, 2]
        ];
        
        let scores = RankFusion::borda_count(&rankings.view()).unwrap();
        
        // Item 0: ranks [1,2,1] -> scores [3,2,3] -> total 8
        assert_eq!(scores[0], 8.0);
        // Item 1: ranks [2,1,3] -> scores [2,3,1] -> total 6
        assert_eq!(scores[1], 6.0);
        // Item 2: ranks [3,3,2] -> scores [1,1,2] -> total 4
        assert_eq!(scores[2], 4.0);
    }
    
    #[test]
    fn test_median_rank() {
        let rankings = array![
            [1, 3, 2],
            [2, 1, 3],
            [3, 2, 1]
        ];
        
        let scores = RankFusion::median_rank(&rankings.view()).unwrap();
        
        // Median ranks: [2, 2, 2]
        // All items have the same median rank, so same score
        assert_eq!(scores[0], scores[1]);
        assert_eq!(scores[1], scores[2]);
    }
    
    #[test]
    fn test_reciprocal_rank() {
        let rankings = array![
            [1, 2, 3],
            [1, 2, 3]
        ];
        
        let scores = RankFusion::reciprocal_rank(&rankings.view()).unwrap();
        
        // Item 0: 1/1 + 1/1 = 2.0
        assert_eq!(scores[0], 2.0);
        // Item 1: 1/2 + 1/2 = 1.0
        assert_eq!(scores[1], 1.0);
        // Item 2: 1/3 + 1/3 = 0.666...
        assert!((scores[2] - 2.0/3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_scores_to_rankings() {
        let scores = array![
            [0.9, 0.7, 0.8],
            [0.5, 0.6, 0.4]
        ];
        
        let rankings = scores_to_rankings(&scores.view());
        
        // First row: 0.9 > 0.8 > 0.7
        assert_eq!(rankings[[0, 0]], 1);
        assert_eq!(rankings[[0, 1]], 3);
        assert_eq!(rankings[[0, 2]], 2);
        
        // Second row: 0.6 > 0.5 > 0.4
        assert_eq!(rankings[[1, 0]], 2);
        assert_eq!(rankings[[1, 1]], 1);
        assert_eq!(rankings[[1, 2]], 3);
    }
    
    #[test]
    fn test_weighted_borda() {
        let rankings = array![
            [1, 2],
            [2, 1]
        ];
        let weights = array![0.7, 0.3];
        
        let scores = RankFusion::weighted_borda_count(&rankings.view(), &weights.view()).unwrap();
        
        // Item 0: 0.7*2 + 0.3*1 = 1.7
        assert!((scores[0] - 1.7).abs() < 1e-10);
        // Item 1: 0.7*1 + 0.3*2 = 1.3
        assert!((scores[1] - 1.3).abs() < 1e-10);
    }
}