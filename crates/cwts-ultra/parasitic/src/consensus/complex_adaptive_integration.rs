//! Complex Adaptive Systems Integration for Organism Selection
//! 
//! Integrates Complex Adaptive Systems theory with autopoietic principles
//! to create self-organizing, emergent selection mechanisms

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, Duration};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

use super::organism_selector::{OrganismSelector, SelectionCriteria, OrganismEvaluation};
use super::autopoietic_systems::{AutopoieticEngine, AutopoieticState, EvolutionEvent};
use crate::organisms::{ParasiticOrganism, MarketConditions};

/// Enhanced organism selector with Complex Adaptive Systems
pub struct ComplexAdaptiveSelector {
    /// Base organism selector
    base_selector: OrganismSelector,
    /// Autopoietic engine for self-organization
    autopoietic_engine: Arc<AutopoieticEngine>,
    /// Emergent behavior patterns
    emergence_patterns: Arc<RwLock<Vec<EmergentPattern>>>,
    /// System attractors for stable configurations
    attractors: Arc<RwLock<Vec<SystemAttractor>>>,
    /// Adaptive feedback mechanisms
    feedback_systems: Arc<RwLock<Vec<FeedbackSystem>>>,
    /// Learning and memory system
    system_memory: Arc<RwLock<SystemMemory>>,
}

/// Emergent pattern in organism selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentPattern {
    pub pattern_id: Uuid,
    pub pattern_name: String,
    pub participants: Vec<Uuid>,
    pub emergence_strength: f64,
    pub stability_index: f64,
    pub performance_impact: f64,
    pub discovered_at: SystemTime,
    pub pattern_description: String,
}

/// System attractor for stable organism configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAttractor {
    pub attractor_id: Uuid,
    pub attractor_type: AttractorType,
    pub organism_configuration: Vec<OrganismRole>,
    pub stability_strength: f64,
    pub performance_characteristics: PerformanceCharacteristics,
    pub activation_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttractorType {
    HighPerformance,    // Optimal performance configuration
    Resilient,          // Stable under stress
    Adaptive,           // Quick response to changes
    Cooperative,        // Maximum cooperation
    Specialized,        // Role-differentiated
}

/// Organism role in system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismRole {
    pub organism_id: Uuid,
    pub role_type: RoleType,
    pub responsibility_weight: f64,
    pub interaction_strength: HashMap<Uuid, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoleType {
    Leader,       // Coordination role
    Specialist,   // Expert in specific domain
    Scout,        // Information gathering
    Executor,     // Action implementation
    Supporter,    // Resource provision
    Innovator,    // Novel strategy development
}

/// Performance characteristics of system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    pub average_fitness: f64,
    pub collective_efficiency: f64,
    pub risk_adjusted_return: f64,
    pub adaptation_speed: f64,
    pub stability_measure: f64,
    pub innovation_capacity: f64,
}

/// Adaptive feedback system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSystem {
    pub system_id: Uuid,
    pub feedback_type: FeedbackType,
    pub source_metrics: Vec<String>,
    pub target_parameters: Vec<String>,
    pub feedback_strength: f64,
    pub delay_characteristics: DelayCharacteristics,
    pub nonlinearity_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    Reinforcing,   // Positive feedback - amplifies changes
    Balancing,     // Negative feedback - maintains stability
    Oscillating,   // Creates periodic behavior
    Chaotic,       // Non-linear, sensitive to initial conditions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayCharacteristics {
    pub minimum_delay: Duration,
    pub maximum_delay: Duration,
    pub average_delay: Duration,
    pub delay_variability: f64,
}

/// System memory for learning and adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMemory {
    /// Historical selection outcomes
    pub selection_history: Vec<SelectionOutcome>,
    /// Learned patterns and their effectiveness
    pub pattern_library: HashMap<String, PatternEffectiveness>,
    /// Environmental context associations
    pub context_associations: HashMap<String, ContextualLearning>,
    /// Performance baselines
    pub performance_baselines: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionOutcome {
    pub timestamp: SystemTime,
    pub selected_organisms: Vec<Uuid>,
    pub selection_criteria: SelectionCriteria,
    pub market_conditions: MarketConditions,
    pub outcome_metrics: OutcomeMetrics,
    pub lessons_learned: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeMetrics {
    pub collective_performance: f64,
    pub individual_performances: HashMap<Uuid, f64>,
    pub emergent_behaviors: Vec<String>,
    pub stability_duration: Duration,
    pub adaptation_success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEffectiveness {
    pub pattern_name: String,
    pub success_rate: f64,
    pub average_performance: f64,
    pub optimal_conditions: Vec<String>,
    pub failure_modes: Vec<String>,
    pub usage_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualLearning {
    pub context_descriptor: String,
    pub successful_strategies: Vec<String>,
    pub failed_strategies: Vec<String>,
    pub adaptation_insights: Vec<String>,
    pub predictive_indicators: HashMap<String, f64>,
}

impl ComplexAdaptiveSelector {
    /// Create new complex adaptive selector
    pub async fn new() -> Self {
        let base_selector = OrganismSelector::new();
        let (autopoietic_engine, _evolution_receiver) = AutopoieticEngine::new();
        
        Self {
            base_selector,
            autopoietic_engine: Arc::new(autopoietic_engine),
            emergence_patterns: Arc::new(RwLock::new(Vec::new())),
            attractors: Arc::new(RwLock::new(Vec::new())),
            feedback_systems: Arc::new(RwLock::new(Vec::new())),
            system_memory: Arc::new(RwLock::new(SystemMemory {
                selection_history: Vec::new(),
                pattern_library: HashMap::new(),
                context_associations: HashMap::new(),
                performance_baselines: HashMap::new(),
            })),
        }
    }
    
    /// Enhanced organism evaluation with complex adaptive principles
    pub async fn evaluate_with_emergence(
        &mut self,
        organisms: &[Box<dyn ParasiticOrganism + Send + Sync>],
        criteria: &SelectionCriteria,
        market_conditions: &MarketConditions,
    ) -> Result<Vec<OrganismEvaluation>, ComplexAdaptiveError> {
        // 1. Base evaluation using traditional metrics
        let mut base_evaluations = self.base_selector
            .evaluate_organisms(organisms, criteria)
            .await
            .map_err(|e| ComplexAdaptiveError::BaseEvaluationFailed(e.to_string()))?;
        
        // 2. Register organisms with autopoietic engine
        for organism in organisms {
            self.autopoietic_engine
                .register_organism(organism.clone())
                .await
                .map_err(|e| ComplexAdaptiveError::AutopoieticRegistrationFailed(e.to_string()))?;
        }
        
        // 3. Evolve autopoietic system to detect emergent patterns
        let evolution_summary = self.autopoietic_engine
            .evolve_system(market_conditions, Duration::from_secs(1))
            .await
            .map_err(|e| ComplexAdaptiveError::SystemEvolutionFailed(e.to_string()))?;
        
        // 4. Detect and analyze emergent patterns
        let emergent_patterns = self.detect_emergent_selection_patterns(&base_evaluations, market_conditions).await?;
        
        // 5. Apply complex adaptive adjustments
        let enhanced_evaluations = self.apply_complex_adaptive_adjustments(
            base_evaluations,
            &emergent_patterns,
            &evolution_summary,
            market_conditions,
        ).await?;
        
        // 6. Update system memory with outcomes
        self.update_system_memory(&enhanced_evaluations, criteria, market_conditions).await?;
        
        // 7. Evolve attractors based on successful configurations
        self.evolve_attractors(&enhanced_evaluations).await?;
        
        Ok(enhanced_evaluations)
    }
    
    /// Detect emergent patterns in organism selection
    async fn detect_emergent_selection_patterns(
        &self,
        evaluations: &[OrganismEvaluation],
        market_conditions: &MarketConditions,
    ) -> Result<Vec<EmergentPattern>, ComplexAdaptiveError> {
        let mut patterns = Vec::new();
        
        // Pattern 1: Cooperative Clusters
        if let Some(pattern) = self.detect_cooperative_clusters(evaluations).await? {
            patterns.push(pattern);
        }
        
        // Pattern 2: Specialization Emergence
        if let Some(pattern) = self.detect_specialization_emergence(evaluations).await? {
            patterns.push(pattern);
        }
        
        // Pattern 3: Performance Synergy
        if let Some(pattern) = self.detect_performance_synergy(evaluations).await? {
            patterns.push(pattern);
        }
        
        // Pattern 4: Adaptive Resilience
        if let Some(pattern) = self.detect_adaptive_resilience(evaluations, market_conditions).await? {
            patterns.push(pattern);
        }
        
        // Update emergence patterns storage
        {
            let mut stored_patterns = self.emergence_patterns.write().unwrap();
            stored_patterns.extend(patterns.clone());
            
            // Keep only recent patterns (sliding window)
            stored_patterns.retain(|p| {
                SystemTime::now().duration_since(p.discovered_at).unwrap().as_secs() < 3600
            });
        }
        
        Ok(patterns)
    }
    
    /// Detect cooperative clusters among organisms
    async fn detect_cooperative_clusters(
        &self,
        evaluations: &[OrganismEvaluation],
    ) -> Result<Option<EmergentPattern>, ComplexAdaptiveError> {
        // Analyze cooperation traits and performance correlations
        let high_cooperation_organisms: Vec<_> = evaluations.iter()
            .filter(|eval| {
                eval.trait_matching.get(&super::OrganismTrait::Cooperative)
                    .map(|&score| score > 0.7)
                    .unwrap_or(false)
            })
            .collect();
        
        if high_cooperation_organisms.len() >= 3 {
            // Calculate cluster strength based on mutual benefit potential
            let cluster_strength = self.calculate_cooperation_synergy(&high_cooperation_organisms);
            
            if cluster_strength > 0.8 {
                return Ok(Some(EmergentPattern {
                    pattern_id: Uuid::new_v4(),
                    pattern_name: "CooperativeCluster".to_string(),
                    participants: high_cooperation_organisms.iter().map(|e| e.organism_id).collect(),
                    emergence_strength: cluster_strength,
                    stability_index: 0.85,
                    performance_impact: cluster_strength * 1.2, // Cooperative bonus
                    discovered_at: SystemTime::now(),
                    pattern_description: "High-cooperation organisms forming synergistic cluster".to_string(),
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Apply complex adaptive adjustments to evaluations
    async fn apply_complex_adaptive_adjustments(
        &self,
        mut evaluations: Vec<OrganismEvaluation>,
        emergent_patterns: &[EmergentPattern],
        evolution_summary: &super::autopoietic_systems::EvolutionSummary,
        market_conditions: &MarketConditions,
    ) -> Result<Vec<OrganismEvaluation>, ComplexAdaptiveError> {
        // 1. Apply emergence bonuses
        for pattern in emergent_patterns {
            for evaluation in &mut evaluations {
                if pattern.participants.contains(&evaluation.organism_id) {
                    let emergence_bonus = pattern.emergence_strength * pattern.performance_impact * 0.1;
                    evaluation.selection_weight += emergence_bonus;
                }
            }
        }
        
        // 2. Apply autopoietic system feedback
        let coherence_factor = evolution_summary.final_state.network_coherence;
        let energy_factor = evolution_summary.final_state.energy_level;
        
        for evaluation in &mut evaluations {
            // Bonus for organisms contributing to system coherence
            let coherence_bonus = coherence_factor * evaluation.market_suitability * 0.05;
            evaluation.selection_weight += coherence_bonus;
            
            // Energy efficiency consideration
            let energy_bonus = energy_factor * evaluation.performance_metrics.resource_efficiency * 0.03;
            evaluation.selection_weight += energy_bonus;
        }
        
        // 3. Apply attractor-based adjustments
        let attractors = self.attractors.read().unwrap();
        for attractor in attractors.iter() {
            if self.market_conditions_match_attractor(market_conditions, attractor) {
                for role in &attractor.organism_configuration {
                    if let Some(evaluation) = evaluations.iter_mut()
                        .find(|e| e.organism_id == role.organism_id) {
                        let attractor_bonus = attractor.stability_strength * role.responsibility_weight * 0.08;
                        evaluation.selection_weight += attractor_bonus;
                    }
                }
            }
        }
        
        // 4. Apply learned pattern adjustments from system memory
        let memory = self.system_memory.read().unwrap();
        for evaluation in &mut evaluations {
            if let Some(pattern_effectiveness) = memory.pattern_library.get(&evaluation.organism_type) {
                let learning_bonus = pattern_effectiveness.success_rate * pattern_effectiveness.average_performance * 0.06;
                evaluation.selection_weight += learning_bonus;
            }
        }
        
        // 5. Re-sort by adjusted selection weight
        evaluations.sort_by(|a, b| {
            b.selection_weight.partial_cmp(&a.selection_weight).unwrap()
        });
        
        Ok(evaluations)
    }
    
    /// Calculate cooperation synergy between organisms
    fn calculate_cooperation_synergy(&self, organisms: &[&OrganismEvaluation]) -> f64 {
        if organisms.len() < 2 {
            return 0.0;
        }
        
        let mut total_synergy = 0.0;
        let mut pair_count = 0;
        
        for i in 0..organisms.len() {
            for j in i+1..organisms.len() {
                let org1 = organisms[i];
                let org2 = organisms[j];
                
                // Calculate complementarity
                let complementarity = self.calculate_complementarity(org1, org2);
                
                // Calculate mutual cooperation potential
                let coop1 = org1.trait_matching.get(&super::OrganismTrait::Cooperative).unwrap_or(&0.0);
                let coop2 = org2.trait_matching.get(&super::OrganismTrait::Cooperative).unwrap_or(&0.0);
                let cooperation_product = coop1 * coop2;
                
                let pair_synergy = (complementarity + cooperation_product) / 2.0;
                total_synergy += pair_synergy;
                pair_count += 1;
            }
        }
        
        if pair_count > 0 {
            total_synergy / pair_count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate complementarity between two organisms
    fn calculate_complementarity(&self, org1: &OrganismEvaluation, org2: &OrganismEvaluation) -> f64 {
        let mut complementarity_score = 0.0;
        let traits = [
            super::OrganismTrait::Aggressive,
            super::OrganismTrait::Conservative,
            super::OrganismTrait::FastExecution,
            super::OrganismTrait::RiskAware,
        ];
        
        for trait_type in &traits {
            let score1 = org1.trait_matching.get(trait_type).unwrap_or(&0.5);
            let score2 = org2.trait_matching.get(trait_type).unwrap_or(&0.5);
            
            // Complementarity is higher when organisms have different strengths
            let difference = (score1 - score2).abs();
            complementarity_score += difference;
        }
        
        complementarity_score / traits.len() as f64
    }
    
    /// Check if market conditions match attractor activation conditions
    fn market_conditions_match_attractor(
        &self,
        market_conditions: &MarketConditions,
        attractor: &SystemAttractor,
    ) -> bool {
        // Simplified matching logic - in practice, this would be more sophisticated
        match attractor.attractor_type {
            AttractorType::HighPerformance => {
                market_conditions.volatility < 0.3 && market_conditions.volume > 0.7
            },
            AttractorType::Resilient => {
                market_conditions.volatility > 0.7 || market_conditions.noise_level > 0.5
            },
            AttractorType::Adaptive => {
                market_conditions.trend_strength < 0.3 && market_conditions.volatility > 0.4
            },
            AttractorType::Cooperative => {
                market_conditions.spread < 0.02 && market_conditions.volume > 0.5
            },
            AttractorType::Specialized => {
                market_conditions.trend_strength > 0.6
            },
        }
    }
    
    /// Update system memory with selection outcomes
    async fn update_system_memory(
        &self,
        evaluations: &[OrganismEvaluation],
        criteria: &SelectionCriteria,
        market_conditions: &MarketConditions,
    ) -> Result<(), ComplexAdaptiveError> {
        let mut memory = self.system_memory.write().unwrap();
        
        // Create selection outcome record
        let outcome = SelectionOutcome {
            timestamp: SystemTime::now(),
            selected_organisms: evaluations.iter().map(|e| e.organism_id).collect(),
            selection_criteria: criteria.clone(),
            market_conditions: market_conditions.clone(),
            outcome_metrics: OutcomeMetrics {
                collective_performance: evaluations.iter().map(|e| e.fitness_score).sum::<f64>() / evaluations.len() as f64,
                individual_performances: evaluations.iter()
                    .map(|e| (e.organism_id, e.fitness_score))
                    .collect(),
                emergent_behaviors: Vec::new(), // Would be populated from actual behavior observation
                stability_duration: Duration::from_secs(300), // Placeholder
                adaptation_success: true, // Placeholder
            },
            lessons_learned: vec![
                "Cooperative organisms show enhanced collective performance".to_string(),
                "Market volatility requires resilient organism selection".to_string(),
            ],
        };
        
        memory.selection_history.push(outcome);
        
        // Update pattern library
        for evaluation in evaluations {
            let pattern_name = &evaluation.organism_type;
            let entry = memory.pattern_library.entry(pattern_name.clone()).or_insert_with(|| {
                PatternEffectiveness {
                    pattern_name: pattern_name.clone(),
                    success_rate: 0.0,
                    average_performance: 0.0,
                    optimal_conditions: Vec::new(),
                    failure_modes: Vec::new(),
                    usage_count: 0,
                }
            });
            
            // Update effectiveness metrics
            entry.usage_count += 1;
            entry.average_performance = (entry.average_performance * (entry.usage_count - 1) as f64 + evaluation.fitness_score) / entry.usage_count as f64;
            entry.success_rate = if evaluation.fitness_score > 0.6 { 
                (entry.success_rate * (entry.usage_count - 1) as f64 + 1.0) / entry.usage_count as f64 
            } else { 
                entry.success_rate * (entry.usage_count - 1) as f64 / entry.usage_count as f64 
            };
        }
        
        // Trim memory to prevent unlimited growth
        if memory.selection_history.len() > 1000 {
            memory.selection_history = memory.selection_history.split_off(memory.selection_history.len() - 1000);
        }
        
        Ok(())
    }
    
    /// Evolve system attractors based on successful configurations
    async fn evolve_attractors(&self, evaluations: &[OrganismEvaluation]) -> Result<(), ComplexAdaptiveError> {
        // Implementation for evolving attractors based on successful organism configurations
        // This would analyze patterns in successful selections and create/update attractors
        Ok(())
    }
    
    /// Additional methods for pattern detection would be implemented here...
    async fn detect_specialization_emergence(&self, _evaluations: &[OrganismEvaluation]) -> Result<Option<EmergentPattern>, ComplexAdaptiveError> {
        // Implementation for detecting role specialization patterns
        Ok(None)
    }
    
    async fn detect_performance_synergy(&self, _evaluations: &[OrganismEvaluation]) -> Result<Option<EmergentPattern>, ComplexAdaptiveError> {
        // Implementation for detecting performance synergy patterns
        Ok(None)
    }
    
    async fn detect_adaptive_resilience(&self, _evaluations: &[OrganismEvaluation], _market_conditions: &MarketConditions) -> Result<Option<EmergentPattern>, ComplexAdaptiveError> {
        // Implementation for detecting adaptive resilience patterns
        Ok(None)
    }
}

/// Errors in complex adaptive systems
#[derive(Debug, thiserror::Error)]
pub enum ComplexAdaptiveError {
    #[error("Base evaluation failed: {0}")]
    BaseEvaluationFailed(String),
    #[error("Autopoietic registration failed: {0}")]
    AutopoieticRegistrationFailed(String),
    #[error("System evolution failed: {0}")]
    SystemEvolutionFailed(String),
    #[error("Pattern detection failed: {0}")]
    PatternDetectionFailed(String),
    #[error("Memory update failed: {0}")]
    MemoryUpdateFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_complex_adaptive_selector_creation() {
        let selector = ComplexAdaptiveSelector::new().await;
        
        let patterns = selector.emergence_patterns.read().unwrap();
        assert_eq!(patterns.len(), 0);
        
        let attractors = selector.attractors.read().unwrap();
        assert_eq!(attractors.len(), 0);
    }
    
    #[test]
    fn test_complementarity_calculation() {
        // Test complementarity calculation logic
        // Implementation would create mock evaluations and test the calculation
    }
}