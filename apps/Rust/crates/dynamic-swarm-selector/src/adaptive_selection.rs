//! Adaptive selection strategies for dynamic swarm optimization
//! 
//! This module implements adaptive selection strategies that learn from
//! performance feedback and adjust algorithm selection over time.

use crate::*;
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use rand::Rng;

/// Adaptive selection strategy
#[derive(Debug, Clone)]
pub enum AdaptiveStrategy {
    /// Epsilon-greedy strategy with exploration rate
    EpsilonGreedy { epsilon: f64 },
    /// Upper Confidence Bound strategy
    UpperConfidenceBound { confidence_level: f64 },
    /// Thompson Sampling strategy
    ThompsonSampling,
    /// Multi-armed bandit with reward tracking
    MultiArmedBandit { decay_factor: f64 },
    /// Reinforcement learning based selection
    ReinforcementLearning { learning_rate: f64 },
}

/// Adaptive selector that learns from performance feedback
pub struct AdaptiveSelector {
    strategy: AdaptiveStrategy,
    algorithm_rewards: Arc<RwLock<HashMap<SwarmAlgorithm, Vec<f64>>>>,
    selection_counts: Arc<RwLock<HashMap<SwarmAlgorithm, u32>>>,
    regime_adaptation: Arc<RwLock<HashMap<MarketRegime, HashMap<SwarmAlgorithm, f64>>>>,
    total_selections: Arc<RwLock<u32>>,
    performance_history: Arc<RwLock<Vec<(SwarmAlgorithm, f64, MarketRegime)>>>,
}

impl AdaptiveSelector {
    pub fn new(strategy: AdaptiveStrategy) -> Self {
        Self {
            strategy,
            algorithm_rewards: Arc::new(RwLock::new(HashMap::new())),
            selection_counts: Arc::new(RwLock::new(HashMap::new())),
            regime_adaptation: Arc::new(RwLock::new(HashMap::new())),
            total_selections: Arc::new(RwLock::new(0)),
            performance_history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Select algorithm using adaptive strategy
    pub async fn select_algorithm(
        &self,
        available_algorithms: &[SwarmAlgorithm],
        current_regime: MarketRegime,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        if available_algorithms.is_empty() {
            return Err(SwarmSelectionError::NoSuitableAlgorithm(current_regime));
        }
        
        let selected = match &self.strategy {
            AdaptiveStrategy::EpsilonGreedy { epsilon } => {
                self.epsilon_greedy_selection(available_algorithms, *epsilon, current_regime).await?
            }
            AdaptiveStrategy::UpperConfidenceBound { confidence_level } => {
                self.ucb_selection(available_algorithms, *confidence_level, current_regime).await?
            }
            AdaptiveStrategy::ThompsonSampling => {
                self.thompson_sampling_selection(available_algorithms, current_regime).await?
            }
            AdaptiveStrategy::MultiArmedBandit { decay_factor } => {
                self.multi_armed_bandit_selection(available_algorithms, *decay_factor, current_regime).await?
            }
            AdaptiveStrategy::ReinforcementLearning { learning_rate } => {
                self.reinforcement_learning_selection(available_algorithms, *learning_rate, current_regime).await?
            }
        };
        
        // Update selection count
        {
            let mut counts = self.selection_counts.write().await;
            *counts.entry(selected).or_insert(0) += 1;
        }
        
        {
            let mut total = self.total_selections.write().await;
            *total += 1;
        }
        
        Ok(selected)
    }
    
    /// Epsilon-greedy selection strategy
    async fn epsilon_greedy_selection(
        &self,
        available_algorithms: &[SwarmAlgorithm],
        epsilon: f64,
        current_regime: MarketRegime,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        let mut rng = rand::thread_rng();
        
        if rng.gen::<f64>() < epsilon {
            // Explore: random selection
            let index = rng.gen_range(0..available_algorithms.len());
            Ok(available_algorithms[index])
        } else {
            // Exploit: select best performing algorithm
            let best_algorithm = self.get_best_algorithm(available_algorithms, current_regime).await?;
            Ok(best_algorithm)
        }
    }
    
    /// Upper Confidence Bound selection strategy
    async fn ucb_selection(
        &self,
        available_algorithms: &[SwarmAlgorithm],
        confidence_level: f64,
        current_regime: MarketRegime,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        let total_selections = *self.total_selections.read().await;
        let selection_counts = self.selection_counts.read().await;
        let algorithm_rewards = self.algorithm_rewards.read().await;
        
        let mut best_algorithm = available_algorithms[0];
        let mut best_ucb_value = f64::NEG_INFINITY;
        
        for &algorithm in available_algorithms {
            let count = selection_counts.get(&algorithm).unwrap_or(&0);
            let rewards = algorithm_rewards.get(&algorithm).unwrap_or(&vec![]);
            
            let ucb_value = if *count == 0 {
                f64::INFINITY // Unselected algorithms get highest priority
            } else {
                let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
                let exploration_term = confidence_level * ((total_selections as f64).ln() / *count as f64).sqrt();
                mean_reward + exploration_term
            };
            
            if ucb_value > best_ucb_value {
                best_ucb_value = ucb_value;
                best_algorithm = algorithm;
            }
        }
        
        Ok(best_algorithm)
    }
    
    /// Thompson Sampling selection strategy
    async fn thompson_sampling_selection(
        &self,
        available_algorithms: &[SwarmAlgorithm],
        current_regime: MarketRegime,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        let mut rng = rand::thread_rng();
        let algorithm_rewards = self.algorithm_rewards.read().await;
        
        let mut best_algorithm = available_algorithms[0];
        let mut best_sample = f64::NEG_INFINITY;
        
        for &algorithm in available_algorithms {
            let rewards = algorithm_rewards.get(&algorithm).unwrap_or(&vec![]);
            
            // Beta distribution parameters (assuming rewards are in [0,1])
            let (alpha, beta) = if rewards.is_empty() {
                (1.0, 1.0) // Uninformed prior
            } else {
                let successes = rewards.iter().sum::<f64>();
                let failures = rewards.len() as f64 - successes;
                (successes + 1.0, failures + 1.0)
            };
            
            // Sample from beta distribution (simplified)
            let sample = rng.gen::<f64>().powf(1.0 / alpha) * (1.0 - rng.gen::<f64>()).powf(1.0 / beta);
            
            if sample > best_sample {
                best_sample = sample;
                best_algorithm = algorithm;
            }
        }
        
        Ok(best_algorithm)
    }
    
    /// Multi-armed bandit selection strategy
    async fn multi_armed_bandit_selection(
        &self,
        available_algorithms: &[SwarmAlgorithm],
        decay_factor: f64,
        current_regime: MarketRegime,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        let algorithm_rewards = self.algorithm_rewards.read().await;
        let mut best_algorithm = available_algorithms[0];
        let mut best_weighted_reward = f64::NEG_INFINITY;
        
        for &algorithm in available_algorithms {
            let rewards = algorithm_rewards.get(&algorithm).unwrap_or(&vec![]);
            
            // Calculate weighted reward with exponential decay
            let weighted_reward = if rewards.is_empty() {
                0.0
            } else {
                rewards.iter().enumerate().map(|(i, &reward)| {
                    reward * decay_factor.powi(rewards.len() as i32 - i as i32 - 1)
                }).sum::<f64>()
            };
            
            if weighted_reward > best_weighted_reward {
                best_weighted_reward = weighted_reward;
                best_algorithm = algorithm;
            }
        }
        
        Ok(best_algorithm)
    }
    
    /// Reinforcement learning selection strategy
    async fn reinforcement_learning_selection(
        &self,
        available_algorithms: &[SwarmAlgorithm],
        learning_rate: f64,
        current_regime: MarketRegime,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        let regime_adaptation = self.regime_adaptation.read().await;
        let regime_scores = regime_adaptation.get(&current_regime);
        
        if let Some(scores) = regime_scores {
            let mut best_algorithm = available_algorithms[0];
            let mut best_score = f64::NEG_INFINITY;
            
            for &algorithm in available_algorithms {
                let score = scores.get(&algorithm).unwrap_or(&0.0);
                if *score > best_score {
                    best_score = *score;
                    best_algorithm = algorithm;
                }
            }
            
            Ok(best_algorithm)
        } else {
            // No learning data for this regime, use random selection
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..available_algorithms.len());
            Ok(available_algorithms[index])
        }
    }
    
    /// Get the best performing algorithm for a regime
    async fn get_best_algorithm(
        &self,
        available_algorithms: &[SwarmAlgorithm],
        current_regime: MarketRegime,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        let algorithm_rewards = self.algorithm_rewards.read().await;
        
        let mut best_algorithm = available_algorithms[0];
        let mut best_average_reward = f64::NEG_INFINITY;
        
        for &algorithm in available_algorithms {
            let rewards = algorithm_rewards.get(&algorithm).unwrap_or(&vec![]);
            let average_reward = if rewards.is_empty() {
                0.0
            } else {
                rewards.iter().sum::<f64>() / rewards.len() as f64
            };
            
            if average_reward > best_average_reward {
                best_average_reward = average_reward;
                best_algorithm = algorithm;
            }
        }
        
        Ok(best_algorithm)
    }
    
    /// Update algorithm performance with feedback
    pub async fn update_performance(
        &self,
        algorithm: SwarmAlgorithm,
        reward: f64,
        regime: MarketRegime,
    ) -> SwarmSelectionResult<()> {
        // Update algorithm rewards
        {
            let mut rewards = self.algorithm_rewards.write().await;
            rewards.entry(algorithm).or_insert_with(Vec::new).push(reward);
        }
        
        // Update regime-specific adaptation
        {
            let mut regime_adaptation = self.regime_adaptation.write().await;
            let regime_scores = regime_adaptation.entry(regime).or_insert_with(HashMap::new);
            
            match &self.strategy {
                AdaptiveStrategy::ReinforcementLearning { learning_rate } => {
                    let current_score = regime_scores.get(&algorithm).unwrap_or(&0.0);
                    let new_score = current_score + learning_rate * (reward - current_score);
                    regime_scores.insert(algorithm, new_score);
                }
                _ => {
                    // For other strategies, use simple averaging
                    let current_score = regime_scores.get(&algorithm).unwrap_or(&0.0);
                    let new_score = (current_score + reward) / 2.0;
                    regime_scores.insert(algorithm, new_score);
                }
            }
        }
        
        // Update performance history
        {
            let mut history = self.performance_history.write().await;
            history.push((algorithm, reward, regime));
            
            // Keep only recent history (last 1000 entries)
            if history.len() > 1000 {
                history.remove(0);
            }
        }
        
        Ok(())
    }
    
    /// Get adaptation statistics
    pub async fn get_adaptation_stats(&self) -> AdaptationStatistics {
        let selection_counts = self.selection_counts.read().await;
        let algorithm_rewards = self.algorithm_rewards.read().await;
        let total_selections = *self.total_selections.read().await;
        
        let mut algorithm_stats = HashMap::new();
        
        for (algorithm, count) in selection_counts.iter() {
            let rewards = algorithm_rewards.get(algorithm).unwrap_or(&vec![]);
            let avg_reward = if rewards.is_empty() {
                0.0
            } else {
                rewards.iter().sum::<f64>() / rewards.len() as f64
            };
            
            algorithm_stats.insert(*algorithm, AlgorithmStats {
                selection_count: *count,
                average_reward: avg_reward,
                selection_ratio: if total_selections > 0 {
                    *count as f64 / total_selections as f64
                } else {
                    0.0
                },
            });
        }
        
        AdaptationStatistics {
            total_selections,
            algorithm_stats,
            strategy: self.strategy.clone(),
        }
    }
    
    /// Reset adaptation state
    pub async fn reset(&self) {
        let mut rewards = self.algorithm_rewards.write().await;
        let mut counts = self.selection_counts.write().await;
        let mut regime_adaptation = self.regime_adaptation.write().await;
        let mut total = self.total_selections.write().await;
        let mut history = self.performance_history.write().await;
        
        rewards.clear();
        counts.clear();
        regime_adaptation.clear();
        *total = 0;
        history.clear();
    }
}

/// Statistics for algorithm adaptation
#[derive(Debug, Clone)]
pub struct AdaptationStatistics {
    pub total_selections: u32,
    pub algorithm_stats: HashMap<SwarmAlgorithm, AlgorithmStats>,
    pub strategy: AdaptiveStrategy,
}

/// Statistics for individual algorithm
#[derive(Debug, Clone)]
pub struct AlgorithmStats {
    pub selection_count: u32,
    pub average_reward: f64,
    pub selection_ratio: f64,
}

impl AdaptationStatistics {
    /// Get the most selected algorithm
    pub fn most_selected_algorithm(&self) -> Option<SwarmAlgorithm> {
        self.algorithm_stats
            .iter()
            .max_by_key(|(_, stats)| stats.selection_count)
            .map(|(alg, _)| *alg)
    }
    
    /// Get the best performing algorithm
    pub fn best_performing_algorithm(&self) -> Option<SwarmAlgorithm> {
        self.algorithm_stats
            .iter()
            .max_by(|(_, a), (_, b)| a.average_reward.partial_cmp(&b.average_reward).unwrap())
            .map(|(alg, _)| *alg)
    }
    
    /// Calculate exploration ratio
    pub fn exploration_ratio(&self) -> f64 {
        if self.algorithm_stats.is_empty() {
            return 0.0;
        }
        
        let entropy = self.algorithm_stats.values()
            .map(|stats| {
                if stats.selection_ratio > 0.0 {
                    -stats.selection_ratio * stats.selection_ratio.log2()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        
        // Normalize by maximum entropy
        let max_entropy = (self.algorithm_stats.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_adaptive_selector_epsilon_greedy() {
        let selector = AdaptiveSelector::new(AdaptiveStrategy::EpsilonGreedy { epsilon: 0.1 });
        
        let algorithms = vec![
            SwarmAlgorithm::ParticleSwarm,
            SwarmAlgorithm::GeneticAlgorithm,
            SwarmAlgorithm::AntColony,
        ];
        
        let selected = selector.select_algorithm(&algorithms, MarketRegime::HighVolatility).await;
        assert!(selected.is_ok());
        
        // Update performance
        let result = selector.update_performance(
            SwarmAlgorithm::ParticleSwarm,
            0.85,
            MarketRegime::HighVolatility,
        ).await;
        assert!(result.is_ok());
        
        let stats = selector.get_adaptation_stats().await;
        assert_eq!(stats.total_selections, 1);
    }
    
    #[tokio::test]
    async fn test_adaptive_selector_ucb() {
        let selector = AdaptiveSelector::new(AdaptiveStrategy::UpperConfidenceBound { 
            confidence_level: 2.0 
        });
        
        let algorithms = vec![
            SwarmAlgorithm::ParticleSwarm,
            SwarmAlgorithm::GeneticAlgorithm,
        ];
        
        // First selection should be random since no history
        let selected = selector.select_algorithm(&algorithms, MarketRegime::HighVolatility).await;
        assert!(selected.is_ok());
        
        // Add some performance feedback
        selector.update_performance(
            SwarmAlgorithm::ParticleSwarm,
            0.9,
            MarketRegime::HighVolatility,
        ).await.unwrap();
        
        selector.update_performance(
            SwarmAlgorithm::GeneticAlgorithm,
            0.7,
            MarketRegime::HighVolatility,
        ).await.unwrap();
        
        let stats = selector.get_adaptation_stats().await;
        assert!(stats.algorithm_stats.len() >= 1);
    }
    
    #[tokio::test]
    async fn test_adaptation_statistics() {
        let selector = AdaptiveSelector::new(AdaptiveStrategy::EpsilonGreedy { epsilon: 0.1 });
        
        // Simulate some selections and updates
        selector.update_performance(
            SwarmAlgorithm::ParticleSwarm,
            0.9,
            MarketRegime::HighVolatility,
        ).await.unwrap();
        
        selector.update_performance(
            SwarmAlgorithm::GeneticAlgorithm,
            0.7,
            MarketRegime::HighVolatility,
        ).await.unwrap();
        
        let stats = selector.get_adaptation_stats().await;
        let best_performing = stats.best_performing_algorithm();
        assert_eq!(best_performing, Some(SwarmAlgorithm::ParticleSwarm));
        
        let exploration_ratio = stats.exploration_ratio();
        assert!(exploration_ratio >= 0.0 && exploration_ratio <= 1.0);
    }
}