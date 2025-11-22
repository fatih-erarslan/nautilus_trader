//! Organism Selection Logic for Consensus Voting
//!
//! Advanced organism selection with multi-criteria evaluation, performance-based
//! weighting, and real-time adaptation to market conditions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use tracing::{debug, instrument};
use uuid::Uuid;

use super::{ConsensusSessionId, PerformanceScore, VotingWeight};
use crate::organisms::*;

/// Criteria for organism selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    pub market_conditions: MarketConditions,
    pub required_traits: Vec<OrganismTrait>,
    pub minimum_fitness: f64,
    pub maximum_organisms: usize,
    pub diversity_requirement: f64,
    pub performance_window_secs: u64,
    pub emergence_sensitivity: f64,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            market_conditions: MarketConditions {
                volatility: 0.5,
                volume: 0.5,
                spread: 0.01,
                trend_strength: 0.3,
                noise_level: 0.2,
            },
            required_traits: vec![
                OrganismTrait::HighPerformance,
                OrganismTrait::FastExecution,
                OrganismTrait::RiskAware,
            ],
            minimum_fitness: 0.6,
            maximum_organisms: 5,
            diversity_requirement: 0.7,
            performance_window_secs: 3600, // 1 hour
            emergence_sensitivity: 0.8,
        }
    }
}

/// Organism traits for selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OrganismTrait {
    HighPerformance,
    FastExecution,
    RiskAware,
    Adaptive,
    Cooperative,
    Stealthy,
    Resilient,
    Efficient,
    Aggressive,
    Conservative,
}

/// Vote cast by an organism in consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismVote {
    pub session_id: ConsensusSessionId,
    pub organism_id: Uuid,
    pub score: f64, // 0.0 to 1.0
    pub weight: VotingWeight,
    pub confidence: f64, // 0.0 to 1.0
    pub timestamp: SystemTime,
    pub reasoning: Option<String>,
}

/// Organism performance evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismEvaluation {
    pub organism_id: Uuid,
    pub organism_type: String,
    pub fitness_score: f64,
    pub performance_metrics: PerformanceMetrics,
    pub trait_matching: HashMap<OrganismTrait, f64>,
    pub market_suitability: f64,
    pub emergence_potential: f64,
    pub selection_weight: VotingWeight,
}

/// Detailed performance metrics for organism evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub profit_ratio: f64,
    pub success_rate: f64,
    pub average_latency_ns: u64,
    pub resource_efficiency: f64,
    pub adaptability_score: f64,
    pub risk_adjusted_return: f64,
    pub market_correlation: f64,
    pub stability_index: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            profit_ratio: 0.0,
            success_rate: 0.0,
            average_latency_ns: 0,
            resource_efficiency: 0.0,
            adaptability_score: 0.0,
            risk_adjusted_return: 0.0,
            market_correlation: 0.0,
            stability_index: 0.0,
        }
    }
}

/// Advanced organism selector with multi-criteria evaluation
pub struct OrganismSelector {
    performance_history: HashMap<Uuid, Vec<PerformanceRecord>>,
    trait_weights: HashMap<OrganismTrait, f64>,
    market_adaptivity: f64,
}

/// Historical performance record
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceRecord {
    timestamp: SystemTime,
    performance: PerformanceMetrics,
    market_conditions: MarketConditions,
}

impl OrganismSelector {
    /// Create new organism selector with default configuration
    pub fn new() -> Self {
        let mut trait_weights = HashMap::new();
        trait_weights.insert(OrganismTrait::HighPerformance, 0.25);
        trait_weights.insert(OrganismTrait::FastExecution, 0.20);
        trait_weights.insert(OrganismTrait::RiskAware, 0.15);
        trait_weights.insert(OrganismTrait::Adaptive, 0.15);
        trait_weights.insert(OrganismTrait::Cooperative, 0.10);
        trait_weights.insert(OrganismTrait::Stealthy, 0.05);
        trait_weights.insert(OrganismTrait::Resilient, 0.05);
        trait_weights.insert(OrganismTrait::Efficient, 0.05);

        Self {
            performance_history: HashMap::new(),
            trait_weights,
            market_adaptivity: 0.8,
        }
    }

    /// Evaluate organisms against selection criteria
    #[instrument(skip(self, organisms))]
    pub async fn evaluate_organisms(
        &self,
        organisms: &[Box<dyn ParasiticOrganism + Send + Sync>],
        criteria: &SelectionCriteria,
    ) -> Result<Vec<OrganismEvaluation>, SelectionError> {
        let mut evaluations = Vec::new();

        for organism in organisms {
            let evaluation = self.evaluate_single_organism(organism, criteria).await?;
            evaluations.push(evaluation);
        }

        // Sort by selection weight (descending)
        evaluations.sort_by(|a, b| b.selection_weight.partial_cmp(&a.selection_weight).unwrap());

        // Apply diversity filtering
        let diverse_evaluations = self.apply_diversity_filter(evaluations, criteria)?;

        debug!(
            "Evaluated {} organisms, selected {} after diversity filtering",
            organisms.len(),
            diverse_evaluations.len()
        );

        Ok(diverse_evaluations)
    }

    /// Evaluate a single organism against criteria
    async fn evaluate_single_organism(
        &self,
        organism: &Box<dyn ParasiticOrganism + Send + Sync>,
        criteria: &SelectionCriteria,
    ) -> Result<OrganismEvaluation, SelectionError> {
        let organism_id = organism.id();
        let organism_type = organism.organism_type().to_string();
        let fitness_score = organism.fitness();

        // Skip organisms below minimum fitness
        if fitness_score < criteria.minimum_fitness {
            return Err(SelectionError::BelowMinimumFitness {
                organism_id,
                fitness: fitness_score,
                minimum: criteria.minimum_fitness,
            });
        }

        // Calculate performance metrics
        let performance_metrics = self
            .calculate_performance_metrics(organism, criteria)
            .await?;

        // Evaluate trait matching
        let trait_matching = self.evaluate_trait_matching(organism, &criteria.required_traits)?;

        // Calculate market suitability
        let market_suitability = self
            .calculate_market_suitability(organism, &criteria.market_conditions)
            .await?;

        // Estimate emergence potential
        let emergence_potential = self
            .estimate_emergence_potential(organism, criteria.emergence_sensitivity)
            .await?;

        // Calculate final selection weight
        let selection_weight = self.calculate_selection_weight(
            fitness_score,
            &performance_metrics,
            &trait_matching,
            market_suitability,
            emergence_potential,
            criteria,
        )?;

        Ok(OrganismEvaluation {
            organism_id,
            organism_type,
            fitness_score,
            performance_metrics,
            trait_matching,
            market_suitability,
            emergence_potential,
            selection_weight,
        })
    }

    /// Calculate comprehensive performance metrics
    async fn calculate_performance_metrics(
        &self,
        organism: &Box<dyn ParasiticOrganism + Send + Sync>,
        criteria: &SelectionCriteria,
    ) -> Result<PerformanceMetrics, SelectionError> {
        let resource_metrics = organism.resource_consumption();
        let strategy_params = organism.get_strategy_params();

        // Get historical performance if available
        let historical_performance =
            self.get_historical_performance(organism.id(), criteria.performance_window_secs);

        let profit_ratio = strategy_params.get("profit_ratio").cloned().unwrap_or(0.0);
        let success_rate = strategy_params.get("success_rate").cloned().unwrap_or(0.0);
        let average_latency_ns = resource_metrics.latency_overhead_ns;

        // Calculate resource efficiency (lower is better for CPU/memory)
        let resource_efficiency =
            1.0 / (1.0 + resource_metrics.cpu_usage + resource_metrics.memory_mb / 1000.0);

        // Calculate adaptability based on genetics
        let genetics = organism.get_genetics();
        let adaptability_score = (genetics.adaptability + genetics.resilience) / 2.0;

        // Calculate risk-adjusted return
        let risk_adjusted_return = profit_ratio / (1.0 + genetics.risk_tolerance);

        // Market correlation based on recent performance
        let market_correlation = historical_performance
            .map(|h| h.market_correlation)
            .unwrap_or(0.5);

        // Stability index based on fitness variance
        let stability_index = 1.0 - genetics.adaptability.abs(); // More stable = less adaptive

        Ok(PerformanceMetrics {
            profit_ratio,
            success_rate,
            average_latency_ns,
            resource_efficiency,
            adaptability_score,
            risk_adjusted_return,
            market_correlation,
            stability_index,
        })
    }

    /// Evaluate how well organism matches required traits
    fn evaluate_trait_matching(
        &self,
        organism: &Box<dyn ParasiticOrganism + Send + Sync>,
        required_traits: &[OrganismTrait],
    ) -> Result<HashMap<OrganismTrait, f64>, SelectionError> {
        let genetics = organism.get_genetics();
        let mut trait_scores = HashMap::new();

        for trait_type in required_traits {
            let score = match trait_type {
                OrganismTrait::HighPerformance => organism.fitness(),
                OrganismTrait::FastExecution => genetics.reaction_speed,
                OrganismTrait::RiskAware => 1.0 - genetics.risk_tolerance,
                OrganismTrait::Adaptive => genetics.adaptability,
                OrganismTrait::Cooperative => genetics.cooperation,
                OrganismTrait::Stealthy => genetics.stealth,
                OrganismTrait::Resilient => genetics.resilience,
                OrganismTrait::Efficient => genetics.efficiency,
                OrganismTrait::Aggressive => genetics.aggression,
                OrganismTrait::Conservative => 1.0 - genetics.aggression,
            };

            trait_scores.insert(trait_type.clone(), score);
        }

        Ok(trait_scores)
    }

    /// Calculate organism suitability for current market conditions
    async fn calculate_market_suitability(
        &self,
        organism: &Box<dyn ParasiticOrganism + Send + Sync>,
        market_conditions: &MarketConditions,
    ) -> Result<f64, SelectionError> {
        let genetics = organism.get_genetics();

        // Different organisms perform better in different market conditions
        let volatility_match = if market_conditions.volatility > 0.7 {
            genetics.aggression // High volatility favors aggressive organisms
        } else {
            genetics.resilience // Low volatility favors resilient organisms
        };

        let volume_match = if market_conditions.volume > 0.6 {
            genetics.reaction_speed // High volume needs fast reactions
        } else {
            genetics.stealth // Low volume favors stealth
        };

        let spread_match = if market_conditions.spread > 0.02 {
            genetics.efficiency // Wide spreads need efficiency
        } else {
            genetics.aggression // Tight spreads need aggression
        };

        let trend_match = if market_conditions.trend_strength > 0.5 {
            genetics.adaptability // Strong trends need adaptability
        } else {
            genetics.resilience // Weak trends need resilience
        };

        let noise_match = if market_conditions.noise_level > 0.3 {
            genetics.stealth // High noise favors stealth
        } else {
            genetics.reaction_speed // Low noise allows fast reactions
        };

        // Weighted average of all matches
        let suitability = (volatility_match * 0.25
            + volume_match * 0.20
            + spread_match * 0.20
            + trend_match * 0.20
            + noise_match * 0.15);

        Ok(suitability)
    }

    /// Estimate organism's emergence potential
    async fn estimate_emergence_potential(
        &self,
        organism: &Box<dyn ParasiticOrganism + Send + Sync>,
        emergence_sensitivity: f64,
    ) -> Result<f64, SelectionError> {
        let genetics = organism.get_genetics();

        // Emergence potential based on cooperation and adaptability
        let base_emergence = (genetics.cooperation + genetics.adaptability) / 2.0;

        // Amplify based on sensitivity setting
        let emergence_potential = base_emergence * emergence_sensitivity;

        Ok(emergence_potential.clamp(0.0, 1.0))
    }

    /// Calculate final selection weight using weighted scoring
    fn calculate_selection_weight(
        &self,
        fitness_score: f64,
        performance_metrics: &PerformanceMetrics,
        trait_matching: &HashMap<OrganismTrait, f64>,
        market_suitability: f64,
        emergence_potential: f64,
        criteria: &SelectionCriteria,
    ) -> Result<VotingWeight, SelectionError> {
        // Base weight from fitness
        let mut weight = fitness_score * 0.3;

        // Add performance component
        weight += (performance_metrics.profit_ratio * 0.15
            + performance_metrics.success_rate * 0.10
            + performance_metrics.resource_efficiency * 0.05
            + performance_metrics.risk_adjusted_return * 0.10);

        // Add trait matching component
        let trait_score: f64 = trait_matching
            .iter()
            .map(|(trait_type, score)| {
                let trait_weight = self.trait_weights.get(trait_type).unwrap_or(&0.1);
                score * trait_weight
            })
            .sum();
        weight += trait_score * 0.2;

        // Add market suitability
        weight += market_suitability * 0.15;

        // Add emergence potential
        weight += emergence_potential * 0.1;

        // Apply market adaptivity
        weight *= (1.0 + self.market_adaptivity * market_suitability);

        Ok(weight.clamp(0.0, 2.0)) // Cap at 2.0 for extreme cases
    }

    /// Apply diversity filtering to ensure organism variety
    fn apply_diversity_filter(
        &self,
        mut evaluations: Vec<OrganismEvaluation>,
        criteria: &SelectionCriteria,
    ) -> Result<Vec<OrganismEvaluation>, SelectionError> {
        if evaluations.len() <= criteria.maximum_organisms {
            return Ok(evaluations);
        }

        let mut selected = Vec::new();
        let mut organism_types = std::collections::HashSet::new();

        // First pass: select top performers of each type
        for evaluation in &evaluations {
            if organism_types.len() >= criteria.maximum_organisms {
                break;
            }

            if !organism_types.contains(&evaluation.organism_type) {
                organism_types.insert(evaluation.organism_type.clone());
                selected.push(evaluation.clone());
            }
        }

        // Second pass: fill remaining slots with best performers
        let remaining_slots = criteria.maximum_organisms.saturating_sub(selected.len());
        for evaluation in evaluations.iter().skip(selected.len()) {
            if selected.len() >= criteria.maximum_organisms {
                break;
            }

            // Check diversity requirement
            let diversity_score = self.calculate_diversity_score(&selected, evaluation);
            if diversity_score >= criteria.diversity_requirement {
                selected.push(evaluation.clone());
            }
        }

        Ok(selected)
    }

    /// Calculate diversity score for an organism against selected set
    fn calculate_diversity_score(
        &self,
        selected: &[OrganismEvaluation],
        candidate: &OrganismEvaluation,
    ) -> f64 {
        if selected.is_empty() {
            return 1.0;
        }

        let mut total_distance = 0.0;

        for selected_organism in selected {
            // Calculate "distance" based on organism type and traits
            let type_distance = if selected_organism.organism_type == candidate.organism_type {
                0.0
            } else {
                1.0
            };

            let trait_distance = candidate
                .trait_matching
                .iter()
                .map(|(trait_type, score)| {
                    let selected_score = selected_organism
                        .trait_matching
                        .get(trait_type)
                        .unwrap_or(&0.5);
                    (score - selected_score).abs()
                })
                .sum::<f64>()
                / candidate.trait_matching.len() as f64;

            total_distance += type_distance * 0.6 + trait_distance * 0.4;
        }

        total_distance / selected.len() as f64
    }

    /// Get historical performance for an organism
    fn get_historical_performance(
        &self,
        organism_id: Uuid,
        window_secs: u64,
    ) -> Option<&PerformanceMetrics> {
        let cutoff_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?
            .saturating_sub(std::time::Duration::from_secs(window_secs));

        self.performance_history
            .get(&organism_id)?
            .iter()
            .rev() // Most recent first
            .find(|record| {
                record
                    .timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d >= std::time::Duration::from_secs(cutoff_time.as_secs()))
                    .unwrap_or(false)
            })
            .map(|record| &record.performance)
    }

    /// Record performance for future evaluations
    pub fn record_performance(
        &mut self,
        organism_id: Uuid,
        performance: PerformanceMetrics,
        market_conditions: MarketConditions,
    ) {
        let record = PerformanceRecord {
            timestamp: SystemTime::now(),
            performance,
            market_conditions,
        };

        self.performance_history
            .entry(organism_id)
            .or_insert_with(Vec::new)
            .push(record);

        // Keep only recent history (last 1000 records per organism)
        if let Some(history) = self.performance_history.get_mut(&organism_id) {
            if history.len() > 1000 {
                history.remove(0);
            }
        }
    }

    /// Update trait weights based on performance feedback
    pub fn update_trait_weights(&mut self, trait_performance: HashMap<OrganismTrait, f64>) {
        const LEARNING_RATE: f64 = 0.1;

        for (trait_type, performance) in trait_performance {
            if let Some(current_weight) = self.trait_weights.get_mut(&trait_type) {
                *current_weight =
                    *current_weight * (1.0 - LEARNING_RATE) + performance * LEARNING_RATE;
                *current_weight = current_weight.clamp(0.01, 0.5); // Keep within reasonable bounds
            }
        }
    }
}

/// Errors that can occur during organism selection
#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("Organism {organism_id} fitness {fitness} below minimum {minimum}")]
    BelowMinimumFitness {
        organism_id: Uuid,
        fitness: f64,
        minimum: f64,
    },

    #[error("Performance calculation failed: {0}")]
    PerformanceCalculationFailed(String),

    #[error("Trait matching failed: {0}")]
    TraitMatchingFailed(String),

    #[error("Market suitability calculation failed: {0}")]
    MarketSuitabilityFailed(String),

    #[error("Diversity filtering failed: {0}")]
    DiversityFilteringFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organisms::BaseOrganism;

    #[test]
    fn test_selection_criteria_default() {
        let criteria = SelectionCriteria::default();
        assert_eq!(criteria.minimum_fitness, 0.6);
        assert_eq!(criteria.maximum_organisms, 5);
        assert!(criteria.diversity_requirement > 0.0);
    }

    #[test]
    fn test_organism_selector_creation() {
        let selector = OrganismSelector::new();
        assert!(selector.trait_weights.len() > 0);
        assert_eq!(selector.performance_history.len(), 0);
    }

    #[test]
    fn test_trait_matching_logic() {
        use crate::organisms::{BaseOrganism, OrganismGenetics};

        let mut base_organism = BaseOrganism::new();
        base_organism.genetics = OrganismGenetics {
            aggression: 0.8,
            adaptability: 0.9,
            efficiency: 0.7,
            resilience: 0.6,
            reaction_speed: 0.95,
            risk_tolerance: 0.3,
            cooperation: 0.5,
            stealth: 0.4,
        };

        // Mock organism for testing
        struct MockOrganism(BaseOrganism);

        #[async_trait::async_trait]
        impl ParasiticOrganism for MockOrganism {
            fn id(&self) -> Uuid {
                self.0.id
            }
            fn organism_type(&self) -> &'static str {
                "mock"
            }
            fn fitness(&self) -> f64 {
                self.0.fitness
            }
            fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
                vulnerability
            }
            async fn infect_pair(&self, _: &str, _: f64) -> Result<InfectionResult, OrganismError> {
                unimplemented!()
            }
            async fn adapt(&mut self, _: AdaptationFeedback) -> Result<(), OrganismError> {
                Ok(())
            }
            fn mutate(&mut self, _: f64) {}
            fn crossover(
                &self,
                _: &dyn ParasiticOrganism,
            ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
                unimplemented!()
            }
            fn get_genetics(&self) -> OrganismGenetics {
                self.0.genetics.clone()
            }
            fn set_genetics(&mut self, genetics: OrganismGenetics) {
                self.0.genetics = genetics;
            }
            fn should_terminate(&self) -> bool {
                false
            }
            fn resource_consumption(&self) -> ResourceMetrics {
                ResourceMetrics::default()
            }
            fn get_strategy_params(&self) -> HashMap<String, f64> {
                HashMap::new()
            }
        }

        let mock_organism: Box<dyn ParasiticOrganism + Send + Sync> =
            Box::new(MockOrganism(base_organism));
        let selector = OrganismSelector::new();

        let required_traits = vec![
            OrganismTrait::HighPerformance,
            OrganismTrait::FastExecution,
            OrganismTrait::Aggressive,
        ];

        let trait_matching = selector
            .evaluate_trait_matching(&mock_organism, &required_traits)
            .unwrap();

        assert_eq!(trait_matching.len(), 3);
        assert!(trait_matching.get(&OrganismTrait::FastExecution).unwrap() > &0.9);
        assert!(trait_matching.get(&OrganismTrait::Aggressive).unwrap() > &0.7);
    }

    #[test]
    fn test_diversity_score_calculation() {
        let selector = OrganismSelector::new();

        let evaluation1 = OrganismEvaluation {
            organism_id: Uuid::new_v4(),
            organism_type: "cuckoo".to_string(),
            fitness_score: 0.8,
            performance_metrics: PerformanceMetrics::default(),
            trait_matching: HashMap::new(),
            market_suitability: 0.7,
            emergence_potential: 0.6,
            selection_weight: 1.0,
        };

        let evaluation2 = OrganismEvaluation {
            organism_id: Uuid::new_v4(),
            organism_type: "wasp".to_string(), // Different type
            fitness_score: 0.7,
            performance_metrics: PerformanceMetrics::default(),
            trait_matching: HashMap::new(),
            market_suitability: 0.6,
            emergence_potential: 0.5,
            selection_weight: 0.9,
        };

        let selected = vec![evaluation1];
        let diversity_score = selector.calculate_diversity_score(&selected, &evaluation2);

        assert!(diversity_score > 0.5); // Different types should have good diversity
    }
}
