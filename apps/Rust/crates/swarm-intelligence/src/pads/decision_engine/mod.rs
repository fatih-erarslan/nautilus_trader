//! # Adaptive Decision Engine
//!
//! Advanced decision-making engine with multi-criteria analysis, uncertainty quantification,
//! and adaptive learning capabilities for the PADS system.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, trace, instrument};
use ndarray::{Array1, Array2};

use crate::core::{
    PadsResult, PadsError, DecisionContext, DecisionLayer, AdaptiveCyclePhase
};
use crate::core::types::{
    DecisionAlternative, DecisionCriteria, PerformanceMetrics
};
use crate::core::traits::{DecisionMaker, Adaptive, AdaptationState};

pub mod multi_criteria;
pub mod decision_tree;
pub mod uncertainty;
pub mod optimization;
pub mod learning;

pub use multi_criteria::*;
pub use decision_tree::*;
pub use uncertainty::*;
pub use optimization::*;
pub use learning::*;

/// Main adaptive decision engine
#[derive(Debug)]
pub struct AdaptiveDecisionEngine {
    /// Engine configuration
    config: DecisionEngineConfig,
    
    /// Multi-criteria analysis module
    mca: Arc<RwLock<MultiCriteriaAnalyzer>>,
    
    /// Decision tree module
    decision_tree: Arc<RwLock<AdaptiveDecisionTree>>,
    
    /// Uncertainty quantifier
    uncertainty: Arc<RwLock<UncertaintyQuantifier>>,
    
    /// Decision optimizer
    optimizer: Arc<RwLock<DecisionOptimizer>>,
    
    /// Learning module
    learning: Arc<RwLock<DecisionLearning>>,
    
    /// Decision history
    history: Arc<RwLock<DecisionHistory>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

/// Decision engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEngineConfig {
    /// Maximum alternatives to consider
    pub max_alternatives: usize,
    
    /// Decision timeout
    pub decision_timeout: Duration,
    
    /// Uncertainty handling enabled
    pub uncertainty_handling: bool,
    
    /// Learning from outcomes enabled
    pub learning_enabled: bool,
    
    /// Optimization algorithms to use
    pub optimization_algorithms: Vec<OptimizationAlgorithm>,
    
    /// Multi-criteria methods
    pub mca_methods: Vec<McaMethod>,
    
    /// Risk tolerance (0.0 to 1.0)
    pub risk_tolerance: f64,
    
    /// Confidence threshold for decisions
    pub confidence_threshold: f64,
}

impl Default for DecisionEngineConfig {
    fn default() -> Self {
        Self {
            max_alternatives: 20,
            decision_timeout: Duration::from_secs(30),
            uncertainty_handling: true,
            learning_enabled: true,
            optimization_algorithms: vec![
                OptimizationAlgorithm::GeneticAlgorithm,
                OptimizationAlgorithm::ParticleSwarm,
                OptimizationAlgorithm::SimulatedAnnealing,
            ],
            mca_methods: vec![
                McaMethod::Topsis,
                McaMethod::Electre,
                McaMethod::Promethee,
                McaMethod::Ahp,
            ],
            risk_tolerance: 0.5,
            confidence_threshold: 0.7,
        }
    }
}

/// Optimization algorithms available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    GeneticAlgorithm,
    ParticleSwarm,
    SimulatedAnnealing,
    DifferentialEvolution,
    Tabu,
    AntColony,
}

/// Multi-criteria analysis methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McaMethod {
    /// Technique for Order Preference by Similarity to Ideal Solution
    Topsis,
    /// Elimination Et Choix Traduisant la Realité
    Electre,
    /// Preference Ranking Organization Method for Enrichment Evaluations
    Promethee,
    /// Analytic Hierarchy Process
    Ahp,
    /// Weighted Sum Model
    Wsm,
    /// Weighted Product Model
    Wpm,
}

/// Decision history record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionHistory {
    /// Historical decision records
    pub decisions: Vec<DecisionRecord>,
    
    /// Performance trends
    pub performance_trends: HashMap<String, Vec<f64>>,
    
    /// Learning insights
    pub insights: Vec<LearningInsight>,
    
    /// Adaptation events
    pub adaptations: Vec<AdaptationEvent>,
}

impl Default for DecisionHistory {
    fn default() -> Self {
        Self {
            decisions: Vec::new(),
            performance_trends: HashMap::new(),
            insights: Vec::new(),
            adaptations: Vec::new(),
        }
    }
}

/// Individual decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Decision identifier
    pub id: String,
    
    /// Decision context
    pub context: DecisionContext,
    
    /// Alternatives considered
    pub alternatives: Vec<DecisionAlternative>,
    
    /// Criteria used
    pub criteria: Vec<DecisionCriteria>,
    
    /// Selected alternative
    pub selected_alternative: String,
    
    /// Decision confidence
    pub confidence: f64,
    
    /// Uncertainty estimate
    pub uncertainty: f64,
    
    /// Processing time
    pub processing_time: Duration,
    
    /// Decision outcome (if available)
    pub outcome: Option<DecisionOutcome>,
    
    /// Timestamp
    pub timestamp: Instant,
}

/// Decision outcome for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Actual performance achieved
    pub actual_performance: PerformanceMetrics,
    
    /// Expected vs actual comparison
    pub expectation_variance: f64,
    
    /// Success indicator (0.0 to 1.0)
    pub success_score: f64,
    
    /// Lessons learned
    pub lessons: Vec<String>,
    
    /// Improvement suggestions
    pub improvements: Vec<String>,
    
    /// Outcome timestamp
    pub timestamp: Instant,
}

/// Learning insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningInsight {
    /// Insight identifier
    pub id: String,
    
    /// Insight type
    pub insight_type: InsightType,
    
    /// Insight description
    pub description: String,
    
    /// Confidence in insight
    pub confidence: f64,
    
    /// Supporting evidence
    pub evidence: Vec<String>,
    
    /// Actionable recommendations
    pub recommendations: Vec<String>,
    
    /// Discovery timestamp
    pub timestamp: Instant,
}

/// Types of learning insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    /// Pattern in decision outcomes
    Pattern,
    
    /// Correlation between variables
    Correlation,
    
    /// Optimization opportunity
    Optimization,
    
    /// Risk factor identification
    RiskFactor,
    
    /// Performance driver
    PerformanceDriver,
    
    /// Context dependency
    ContextDependency,
}

/// Adaptation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvent {
    /// Event identifier
    pub id: String,
    
    /// What was adapted
    pub component: String,
    
    /// Adaptation description
    pub description: String,
    
    /// Performance impact
    pub impact: f64,
    
    /// Confidence in adaptation
    pub confidence: f64,
    
    /// Timestamp
    pub timestamp: Instant,
}

impl AdaptiveDecisionEngine {
    /// Create a new adaptive decision engine
    #[instrument(skip(config))]
    pub async fn new(config: DecisionEngineConfig) -> PadsResult<Self> {
        info!("Initializing adaptive decision engine");
        
        let mca = Arc::new(RwLock::new(
            MultiCriteriaAnalyzer::new(config.mca_methods.clone()).await?
        ));
        
        let decision_tree = Arc::new(RwLock::new(
            AdaptiveDecisionTree::new().await?
        ));
        
        let uncertainty = Arc::new(RwLock::new(
            UncertaintyQuantifier::new().await?
        ));
        
        let optimizer = Arc::new(RwLock::new(
            DecisionOptimizer::new(config.optimization_algorithms.clone()).await?
        ));
        
        let learning = Arc::new(RwLock::new(
            DecisionLearning::new().await?
        ));
        
        let history = Arc::new(RwLock::new(DecisionHistory::default()));
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::new()));
        
        info!("Adaptive decision engine initialized successfully");
        
        Ok(Self {
            config,
            mca,
            decision_tree,
            uncertainty,
            optimizer,
            learning,
            history,
            metrics,
        })
    }
    
    /// Process a decision request
    #[instrument(skip(self, context))]
    pub async fn process_decision(
        &self,
        context: DecisionContext,
    ) -> PadsResult<DecisionResponse> {
        let start_time = Instant::now();
        debug!("Processing decision: {}", context.id);
        
        // Check if context is still valid
        if !context.is_valid() {
            return Err(PadsError::DecisionLayer {
                layer: context.layer,
                message: "Decision context expired".to_string(),
            });
        }
        
        // Generate alternatives
        let alternatives = self.generate_alternatives(&context).await?;
        
        // Define criteria based on context and layer
        let criteria = self.generate_criteria(&context).await?;
        
        // Evaluate alternatives using multi-criteria analysis
        let evaluations = self.evaluate_alternatives(&alternatives, &criteria).await?;
        
        // Quantify uncertainty if enabled
        let uncertainty_estimate = if self.config.uncertainty_handling {
            self.uncertainty.read().await.estimate_uncertainty(&context, &alternatives).await?
        } else {
            0.0
        };
        
        // Optimize decision if multiple good alternatives
        let optimized_selection = if alternatives.len() > 1 {
            self.optimizer.read().await.optimize_selection(
                &alternatives,
                &criteria,
                &evaluations,
                &context,
            ).await?
        } else {
            alternatives.first().map(|a| a.id.clone()).unwrap_or_default()
        };
        
        // Calculate decision confidence
        let confidence = self.calculate_confidence(
            &optimized_selection,
            &evaluations,
            uncertainty_estimate,
        ).await;
        
        // Check if confidence meets threshold
        if confidence < self.config.confidence_threshold {
            return Err(PadsError::DecisionLayer {
                layer: context.layer,
                message: format!("Decision confidence {} below threshold {}", 
                               confidence, self.config.confidence_threshold),
            });
        }
        
        let processing_time = start_time.elapsed();
        
        // Create decision record
        let decision_record = DecisionRecord {
            id: context.id.clone(),
            context: context.clone(),
            alternatives: alternatives.clone(),
            criteria: criteria.clone(),
            selected_alternative: optimized_selection.clone(),
            confidence,
            uncertainty: uncertainty_estimate,
            processing_time,
            outcome: None,
            timestamp: start_time,
        };
        
        // Store decision in history
        self.history.write().await.decisions.push(decision_record);
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.insert("decisions_processed".to_string(), 
                          metrics.get("decisions_processed").unwrap_or(&0.0) + 1.0);
            metrics.insert("avg_processing_time_ms".to_string(), 
                          processing_time.as_millis() as f64);
            metrics.insert("avg_confidence".to_string(), confidence);
        }
        
        // Generate reasoning
        let reasoning = self.generate_reasoning(
            &optimized_selection,
            &alternatives,
            &evaluations,
            &context,
        ).await;
        
        debug!("Decision processed: {} (confidence: {:.2})", 
               optimized_selection, confidence);
        
        Ok(DecisionResponse {
            decision_id: context.id,
            layer: context.layer,
            action: optimized_selection,
            confidence,
            reasoning,
            alternatives: alternatives.iter().map(|a| a.id.clone()).collect(),
            metadata: self.generate_metadata(&context, uncertainty_estimate).await,
            timestamp: Instant::now(),
        })
    }
    
    /// Learn from a decision outcome
    #[instrument(skip(self, decision_id, outcome))]
    pub async fn learn_from_outcome(
        &mut self,
        decision_id: String,
        outcome: DecisionOutcome,
    ) -> PadsResult<()> {
        info!("Learning from decision outcome: {}", decision_id);
        
        if !self.config.learning_enabled {
            return Ok(());
        }
        
        // Update decision record with outcome
        {
            let mut history = self.history.write().await;
            if let Some(record) = history.decisions.iter_mut()
                .find(|r| r.id == decision_id) {
                record.outcome = Some(outcome.clone());
            }
        }
        
        // Learn from the outcome
        self.learning.write().await.learn_from_outcome(&decision_id, &outcome).await?;
        
        // Generate insights
        let insights = self.learning.read().await.generate_insights().await?;
        
        // Store new insights
        {
            let mut history = self.history.write().await;
            history.insights.extend(insights);
        }
        
        // Adapt decision-making parameters
        self.adapt_parameters(&outcome).await?;
        
        Ok(())
    }
    
    /// Generate decision alternatives based on context
    async fn generate_alternatives(
        &self,
        context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionAlternative>> {
        debug!("Generating alternatives for decision: {}", context.id);
        
        let mut alternatives = Vec::new();
        
        // Generate alternatives based on decision layer and context
        match context.layer {
            DecisionLayer::Tactical => {
                alternatives.extend(self.generate_tactical_alternatives(context).await?);
            }
            DecisionLayer::Operational => {
                alternatives.extend(self.generate_operational_alternatives(context).await?);
            }
            DecisionLayer::Strategic => {
                alternatives.extend(self.generate_strategic_alternatives(context).await?);
            }
            DecisionLayer::MetaStrategic => {
                alternatives.extend(self.generate_meta_strategic_alternatives(context).await?);
            }
        }
        
        // Use decision tree to generate additional alternatives
        let tree_alternatives = self.decision_tree.read().await
            .generate_alternatives(context).await?;
        alternatives.extend(tree_alternatives);
        
        // Limit number of alternatives
        alternatives.truncate(self.config.max_alternatives);
        
        debug!("Generated {} alternatives", alternatives.len());
        Ok(alternatives)
    }
    
    /// Generate criteria for decision evaluation
    async fn generate_criteria(
        &self,
        context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionCriteria>> {
        let mut criteria = Vec::new();
        
        // Standard criteria for all decisions
        criteria.push(DecisionCriteria::new(
            "performance".to_string(),
            "Performance Impact".to_string(),
            0.3,
            true,
        ));
        
        criteria.push(DecisionCriteria::new(
            "risk".to_string(),
            "Risk Level".to_string(),
            0.2,
            false, // Lower risk is better
        ));
        
        criteria.push(DecisionCriteria::new(
            "cost".to_string(),
            "Implementation Cost".to_string(),
            0.2,
            false, // Lower cost is better
        ));
        
        criteria.push(DecisionCriteria::new(
            "time".to_string(),
            "Time to Implement".to_string(),
            0.15,
            false, // Shorter time is better
        ));
        
        criteria.push(DecisionCriteria::new(
            "adaptability".to_string(),
            "Future Adaptability".to_string(),
            0.15,
            true,
        ));
        
        // Layer-specific criteria
        match context.layer {
            DecisionLayer::Tactical => {
                criteria.push(DecisionCriteria::new(
                    "speed".to_string(),
                    "Execution Speed".to_string(),
                    0.4,
                    true,
                ));
            }
            DecisionLayer::Operational => {
                criteria.push(DecisionCriteria::new(
                    "efficiency".to_string(),
                    "Operational Efficiency".to_string(),
                    0.3,
                    true,
                ));
            }
            DecisionLayer::Strategic => {
                criteria.push(DecisionCriteria::new(
                    "alignment".to_string(),
                    "Strategic Alignment".to_string(),
                    0.35,
                    true,
                ));
            }
            DecisionLayer::MetaStrategic => {
                criteria.push(DecisionCriteria::new(
                    "transformation".to_string(),
                    "Transformation Potential".to_string(),
                    0.4,
                    true,
                ));
            }
        }
        
        Ok(criteria)
    }
    
    /// Generate tactical alternatives
    async fn generate_tactical_alternatives(
        &self,
        _context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionAlternative>> {
        // Placeholder implementation
        let alternatives = vec![
            DecisionAlternative::new(
                "tactical_001".to_string(),
                "Quick Response".to_string(),
                "Immediate tactical response".to_string(),
            )
            .with_criterion("performance".to_string(), 0.7)
            .with_criterion("speed".to_string(), 0.9)
            .with_risk(0.3),
            
            DecisionAlternative::new(
                "tactical_002".to_string(),
                "Cautious Approach".to_string(),
                "More cautious tactical approach".to_string(),
            )
            .with_criterion("performance".to_string(), 0.6)
            .with_criterion("speed".to_string(), 0.5)
            .with_risk(0.1),
        ];
        
        Ok(alternatives)
    }
    
    /// Generate operational alternatives
    async fn generate_operational_alternatives(
        &self,
        _context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionAlternative>> {
        // Placeholder implementation
        let alternatives = vec![
            DecisionAlternative::new(
                "operational_001".to_string(),
                "Process Optimization".to_string(),
                "Optimize current processes".to_string(),
            )
            .with_criterion("performance".to_string(), 0.8)
            .with_criterion("efficiency".to_string(), 0.9)
            .with_risk(0.2),
            
            DecisionAlternative::new(
                "operational_002".to_string(),
                "Resource Reallocation".to_string(),
                "Reallocate resources for better performance".to_string(),
            )
            .with_criterion("performance".to_string(), 0.7)
            .with_criterion("efficiency".to_string(), 0.6)
            .with_risk(0.4),
        ];
        
        Ok(alternatives)
    }
    
    /// Generate strategic alternatives
    async fn generate_strategic_alternatives(
        &self,
        _context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionAlternative>> {
        // Placeholder implementation
        let alternatives = vec![
            DecisionAlternative::new(
                "strategic_001".to_string(),
                "Long-term Investment".to_string(),
                "Invest in long-term capabilities".to_string(),
            )
            .with_criterion("performance".to_string(), 0.9)
            .with_criterion("alignment".to_string(), 0.8)
            .with_risk(0.5),
            
            DecisionAlternative::new(
                "strategic_002".to_string(),
                "Gradual Evolution".to_string(),
                "Gradually evolve current strategy".to_string(),
            )
            .with_criterion("performance".to_string(), 0.6)
            .with_criterion("alignment".to_string(), 0.7)
            .with_risk(0.2),
        ];
        
        Ok(alternatives)
    }
    
    /// Generate meta-strategic alternatives
    async fn generate_meta_strategic_alternatives(
        &self,
        _context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionAlternative>> {
        // Placeholder implementation
        let alternatives = vec![
            DecisionAlternative::new(
                "meta_001".to_string(),
                "System Transformation".to_string(),
                "Transform the entire system".to_string(),
            )
            .with_criterion("performance".to_string(), 0.95)
            .with_criterion("transformation".to_string(), 0.9)
            .with_risk(0.8),
            
            DecisionAlternative::new(
                "meta_002".to_string(),
                "Adaptive Evolution".to_string(),
                "Enable system to evolve adaptively".to_string(),
            )
            .with_criterion("performance".to_string(), 0.8)
            .with_criterion("transformation".to_string(), 0.7)
            .with_risk(0.4),
        ];
        
        Ok(alternatives)
    }
    
    /// Calculate decision confidence
    async fn calculate_confidence(
        &self,
        _selected_alternative: &str,
        evaluations: &HashMap<String, f64>,
        uncertainty: f64,
    ) -> f64 {
        // Base confidence from evaluation scores
        let max_score = evaluations.values().fold(0.0, |a, &b| a.max(b));
        let avg_score = evaluations.values().sum::<f64>() / evaluations.len() as f64;
        let score_variance = evaluations.values()
            .map(|&x| (x - avg_score).powi(2))
            .sum::<f64>() / evaluations.len() as f64;
        
        // Higher confidence if clear winner (low variance)
        let clarity_factor = 1.0 / (1.0 + score_variance);
        
        // Lower confidence with higher uncertainty
        let uncertainty_factor = 1.0 - uncertainty;
        
        // Combine factors
        (max_score * clarity_factor * uncertainty_factor).clamp(0.0, 1.0)
    }
    
    /// Generate reasoning for the decision
    async fn generate_reasoning(
        &self,
        selected_alternative: &str,
        alternatives: &[DecisionAlternative],
        evaluations: &HashMap<String, f64>,
        context: &DecisionContext,
    ) -> Vec<String> {
        let mut reasoning = Vec::new();
        
        // Find the selected alternative
        if let Some(alternative) = alternatives.iter().find(|a| a.id == selected_alternative) {
            reasoning.push(format!("Selected '{}' based on multi-criteria analysis", alternative.name));
            
            if let Some(score) = evaluations.get(&alternative.id) {
                reasoning.push(format!("Alternative scored {:.2} in evaluation", score));
            }
            
            // Add context-specific reasoning
            match context.layer {
                DecisionLayer::Tactical => {
                    reasoning.push("Prioritized speed and immediate impact".to_string());
                }
                DecisionLayer::Operational => {
                    reasoning.push("Focused on operational efficiency and resource utilization".to_string());
                }
                DecisionLayer::Strategic => {
                    reasoning.push("Considered long-term strategic alignment".to_string());
                }
                DecisionLayer::MetaStrategic => {
                    reasoning.push("Evaluated transformation potential and system-wide impact".to_string());
                }
            }
            
            // Add cycle phase reasoning
            match context.cycle_phase {
                AdaptiveCyclePhase::Growth => {
                    reasoning.push("Optimized for growth and expansion opportunities".to_string());
                }
                AdaptiveCyclePhase::Conservation => {
                    reasoning.push("Emphasized efficiency and risk management".to_string());
                }
                AdaptiveCyclePhase::Release => {
                    reasoning.push("Considered innovation and creative destruction".to_string());
                }
                AdaptiveCyclePhase::Reorganization => {
                    reasoning.push("Focused on renewal and transformation".to_string());
                }
            }
        }
        
        reasoning
    }
    
    /// Generate decision metadata
    async fn generate_metadata(
        &self,
        context: &DecisionContext,
        uncertainty: f64,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        metadata.insert("layer".to_string(), format!("{:?}", context.layer));
        metadata.insert("cycle_phase".to_string(), format!("{:?}", context.cycle_phase));
        metadata.insert("urgency".to_string(), context.urgency.to_string());
        metadata.insert("uncertainty".to_string(), uncertainty.to_string());
        metadata.insert("engine_version".to_string(), "1.0.0".to_string());
        
        metadata
    }
    
    /// Adapt decision-making parameters based on outcomes
    async fn adapt_parameters(&mut self, outcome: &DecisionOutcome) -> PadsResult<()> {
        // Adjust risk tolerance based on outcome success
        if outcome.success_score > 0.8 {
            // Successful outcome - can be slightly more risk-tolerant
            self.config.risk_tolerance = (self.config.risk_tolerance + 0.01).min(1.0);
        } else if outcome.success_score < 0.3 {
            // Poor outcome - be more risk-averse
            self.config.risk_tolerance = (self.config.risk_tolerance - 0.01).max(0.0);
        }
        
        // Adjust confidence threshold based on expectation variance
        if outcome.expectation_variance > 0.5 {
            // High variance - require higher confidence
            self.config.confidence_threshold = (self.config.confidence_threshold + 0.01).min(0.95);
        } else if outcome.expectation_variance < 0.1 {
            // Low variance - can accept lower confidence
            self.config.confidence_threshold = (self.config.confidence_threshold - 0.01).max(0.5);
        }
        
        // Record adaptation
        let adaptation = AdaptationEvent {
            id: format!("adapt-{}", uuid::Uuid::new_v4()),
            component: "decision_engine".to_string(),
            description: format!("Adapted risk tolerance to {:.3} and confidence threshold to {:.3}",
                               self.config.risk_tolerance, self.config.confidence_threshold),
            impact: (outcome.success_score - 0.5).abs(),
            confidence: 0.8,
            timestamp: Instant::now(),
        };
        
        self.history.write().await.adaptations.push(adaptation);
        
        Ok(())
    }
}

/// Decision response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResponse {
    /// Decision identifier
    pub decision_id: String,
    
    /// Processing layer
    pub layer: DecisionLayer,
    
    /// Recommended action
    pub action: String,
    
    /// Confidence in the decision (0.0 to 1.0)
    pub confidence: f64,
    
    /// Reasoning behind the decision
    pub reasoning: Vec<String>,
    
    /// Alternative options considered
    pub alternatives: Vec<String>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    
    /// Response timestamp
    pub timestamp: Instant,
}

#[async_trait::async_trait]
impl DecisionMaker for AdaptiveDecisionEngine {
    async fn generate_alternatives(
        &self,
        context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionAlternative>> {
        self.generate_alternatives(context).await
    }
    
    async fn evaluate_alternatives(
        &self,
        alternatives: &[DecisionAlternative],
        criteria: &[DecisionCriteria],
    ) -> PadsResult<HashMap<String, f64>> {
        self.mca.read().await.evaluate_alternatives(alternatives, criteria).await
    }
    
    async fn select_alternative(
        &self,
        evaluations: &HashMap<String, f64>,
        _context: &DecisionContext,
    ) -> PadsResult<String> {
        evaluations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| id.clone())
            .ok_or_else(|| PadsError::DecisionLayer {
                layer: DecisionLayer::Tactical,
                message: "No alternatives to select from".to_string(),
            })
    }
    
    async fn validate_decision(
        &self,
        alternative: &DecisionAlternative,
        context: &DecisionContext,
    ) -> PadsResult<bool> {
        // Basic validation - check if risk is acceptable
        let acceptable_risk = alternative.risk_level <= self.config.risk_tolerance;
        
        // Check if it meets minimum performance threshold
        let min_performance = context.constraints.get("min_performance").unwrap_or(&0.5);
        let performance_ok = alternative.expected_benefit >= *min_performance;
        
        Ok(acceptable_risk && performance_ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_decision_engine_creation() {
        let config = DecisionEngineConfig::default();
        let engine = AdaptiveDecisionEngine::new(config).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_alternative_generation() {
        let config = DecisionEngineConfig::default();
        let engine = AdaptiveDecisionEngine::new(config).await.unwrap();
        
        let context = DecisionContext::new(
            "test-001".to_string(),
            DecisionLayer::Tactical,
            AdaptiveCyclePhase::Growth,
        );
        
        let alternatives = engine.generate_alternatives(&context).await;
        assert!(alternatives.is_ok());
        assert!(!alternatives.unwrap().is_empty());
    }
    
    #[tokio::test]
    async fn test_decision_processing() {
        let config = DecisionEngineConfig::default();
        let engine = AdaptiveDecisionEngine::new(config).await.unwrap();
        
        let context = DecisionContext::new(
            "test-002".to_string(),
            DecisionLayer::Operational,
            AdaptiveCyclePhase::Conservation,
        );
        
        let response = engine.process_decision(context).await;
        assert!(response.is_ok());
        
        let response = response.unwrap();
        assert!(!response.action.is_empty());
        assert!(response.confidence > 0.0);
    }
}