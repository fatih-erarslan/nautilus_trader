//! Dynamic Swarm Selector Implementation
//! 
//! This module provides the core dynamic swarm selection functionality.

use crate::*;
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Dynamic swarm selector for bio-inspired optimization algorithms
pub struct DynamicSwarmSelector {
    performance_tracker: Arc<RwLock<HashMap<SwarmAlgorithm, SwarmPerformanceMetrics>>>,
    current_regime: Arc<RwLock<MarketRegime>>,
    selection_criteria: SelectionCriteria,
}

impl DynamicSwarmSelector {
    pub fn new(selection_criteria: SelectionCriteria) -> Self {
        Self {
            performance_tracker: Arc::new(RwLock::new(HashMap::new())),
            current_regime: Arc::new(RwLock::new(MarketRegime::Neutral)),
            selection_criteria,
        }
    }
    
    /// Select the best swarm algorithm for current market conditions
    pub async fn select_algorithm(&self, conditions: &MarketConditions) -> SwarmSelectionResult<SwarmAlgorithm> {
        let regime = self.current_regime.read().await;
        
        // Get compatible algorithms for current regime
        let compatible_algorithms = self.get_compatible_algorithms(&regime);
        
        if compatible_algorithms.is_empty() {
            return Err(SwarmSelectionError::NoSuitableAlgorithm(*regime));
        }
        
        // Select based on performance history and current conditions
        let best_algorithm = self.select_best_from_compatible(compatible_algorithms, conditions).await?;
        
        Ok(best_algorithm)
    }
    
    /// Get algorithms compatible with current market regime
    fn get_compatible_algorithms(&self, regime: &MarketRegime) -> Vec<SwarmAlgorithm> {
        vec![
            SwarmAlgorithm::ParticleSwarm,
            SwarmAlgorithm::AntColony,
            SwarmAlgorithm::GeneticAlgorithm,
            SwarmAlgorithm::DifferentialEvolution,
            SwarmAlgorithm::GreyWolf,
            SwarmAlgorithm::WhaleOptimization,
            SwarmAlgorithm::BatAlgorithm,
            SwarmAlgorithm::FireflyAlgorithm,
            SwarmAlgorithm::CuckooSearch,
            SwarmAlgorithm::ArtificialBeeColony,
            SwarmAlgorithm::BacterialForaging,
            SwarmAlgorithm::SocialSpider,
            SwarmAlgorithm::MothFlame,
            SwarmAlgorithm::SalpSwarm,
        ]
        .into_iter()
        .filter(|alg| alg.is_regime_compatible(regime))
        .collect()
    }
    
    /// Select the best algorithm from compatible ones
    async fn select_best_from_compatible(
        &self,
        compatible: Vec<SwarmAlgorithm>,
        conditions: &MarketConditions,
    ) -> SwarmSelectionResult<SwarmAlgorithm> {
        let performance_tracker = self.performance_tracker.read().await;
        
        // Score each algorithm based on performance and conditions
        let mut scored_algorithms: Vec<(SwarmAlgorithm, f64)> = compatible
            .into_iter()
            .map(|alg| {
                let score = self.calculate_algorithm_score(alg, conditions, &performance_tracker);
                (alg, score)
            })
            .collect();
        
        // Sort by score (highest first)
        scored_algorithms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(scored_algorithms.first().unwrap().0)
    }
    
    /// Calculate algorithm score based on performance and conditions
    fn calculate_algorithm_score(
        &self,
        algorithm: SwarmAlgorithm,
        conditions: &MarketConditions,
        performance_tracker: &HashMap<SwarmAlgorithm, SwarmPerformanceMetrics>,
    ) -> f64 {
        let mut score = 0.0;
        
        // Base score from algorithm characteristics
        score += match algorithm.convergence_profile() {
            ConvergenceProfile::Fast => 0.8,
            ConvergenceProfile::Steady => 0.7,
            ConvergenceProfile::Balanced => 0.9,
            ConvergenceProfile::Robust => 0.6,
            ConvergenceProfile::Aggressive => 0.75,
            ConvergenceProfile::Explorative => 0.65,
            ConvergenceProfile::Dynamic => 0.95,
            ConvergenceProfile::Moderate => 0.5,
        };
        
        // Adjust for computational complexity
        score += match algorithm.computational_complexity() {
            ComputationalComplexity::Low => 0.3,
            ComputationalComplexity::Medium => 0.2,
            ComputationalComplexity::High => 0.1,
            ComputationalComplexity::VeryHigh => 0.0,
        };
        
        // Historical performance bonus
        if let Some(metrics) = performance_tracker.get(&algorithm) {
            score += metrics.optimization_score * 0.4;
            score += metrics.stability_score * 0.3;
        }
        
        // Market condition adjustments
        if conditions.volatility > 0.5 {
            // High volatility - favor explorative algorithms
            if matches!(algorithm.convergence_profile(), ConvergenceProfile::Explorative) {
                score += 0.2;
            }
        } else {
            // Low volatility - favor exploitation algorithms
            if matches!(algorithm.convergence_profile(), ConvergenceProfile::Fast | ConvergenceProfile::Aggressive) {
                score += 0.2;
            }
        }
        
        score
    }
    
    /// Update algorithm performance metrics
    pub async fn update_performance(&self, feedback: PerformanceFeedback) -> SwarmSelectionResult<()> {
        let mut performance_tracker = self.performance_tracker.write().await;
        
        let metrics = SwarmPerformanceMetrics {
            algorithm: feedback.algorithm,
            regime: feedback.regime,
            optimization_score: feedback.optimization_quality,
            convergence_time: feedback.execution_time,
            function_evaluations: 0, // TODO: Extract from feedback
            success_rate: if feedback.success_indicator { 1.0 } else { 0.0 },
            stability_score: feedback.optimization_quality * 0.8,
            exploration_ratio: 0.5, // TODO: Calculate from algorithm behavior
            exploitation_ratio: 0.5, // TODO: Calculate from algorithm behavior
            diversity_index: 0.5, // TODO: Calculate from population diversity
        };
        
        performance_tracker.insert(feedback.algorithm, metrics);
        
        Ok(())
    }
    
    /// Update current market regime
    pub async fn update_regime(&self, new_regime: MarketRegime) {
        let mut regime = self.current_regime.write().await;
        *regime = new_regime;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    
    #[tokio::test]
    async fn test_selector_creation() {
        let criteria = SelectionCriteria {
            optimization_target: OptimizationTarget::Profit,
            time_constraint: Duration::minutes(5),
            accuracy_requirement: 0.95,
            computational_budget: ComputationalBudget {
                max_function_evaluations: 1000,
                max_computation_time: Duration::minutes(10),
                max_memory_usage: 1024 * 1024,
                parallelization_factor: 4,
            },
            risk_tolerance: 0.1,
            regime_specific: true,
        };
        
        let selector = DynamicSwarmSelector::new(criteria);
        
        let conditions = MarketConditions {
            volatility: 0.5,
            volume: 1000000.0,
            trend_strength: 0.3,
            liquidity: 0.8,
            correlation: 0.2,
            noise_level: 0.1,
            regime_stability: 0.7,
        };
        
        let algorithm = selector.select_algorithm(&conditions).await;
        assert!(algorithm.is_ok());
    }
    
    #[tokio::test]
    async fn test_performance_update() {
        let criteria = SelectionCriteria {
            optimization_target: OptimizationTarget::Profit,
            time_constraint: Duration::minutes(5),
            accuracy_requirement: 0.95,
            computational_budget: ComputationalBudget {
                max_function_evaluations: 1000,
                max_computation_time: Duration::minutes(10),
                max_memory_usage: 1024 * 1024,
                parallelization_factor: 4,
            },
            risk_tolerance: 0.1,
            regime_specific: true,
        };
        
        let selector = DynamicSwarmSelector::new(criteria);
        
        let feedback = PerformanceFeedback {
            algorithm: SwarmAlgorithm::ParticleSwarm,
            regime: MarketRegime::HighVolatility,
            optimization_quality: 0.85,
            execution_time: Duration::seconds(30),
            resource_usage: ResourceUsage {
                cpu_time: Duration::seconds(25),
                memory_peak: 512 * 1024,
                cache_hits: 800,
                cache_misses: 200,
                parallelization_efficiency: 0.9,
            },
            success_indicator: true,
            improvement_suggestions: vec!["Increase population size".to_string()],
        };
        
        let result = selector.update_performance(feedback).await;
        assert!(result.is_ok());
    }
}