//! Board Members with sophisticated cognitive architectures
//!
//! This module implements board members with specialized roles, cognitive biases,
//! and sophisticated decision-making capabilities harvested from the advanced
//! PADS cognitive architecture.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Board member types with specialized cognitive functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoardMemberType {
    /// Chief Executive Officer - Strategic oversight
    CEO {
        vision_horizon: u64,
        strategic_weight: f64,
        executive_authority: f64,
    },
    
    /// Chief Technology Officer - Technical innovation
    CTO {
        innovation_rate: f64,
        technical_debt_tolerance: f64,
        system_reliability_threshold: f64,
    },
    
    /// Chief Financial Officer - Financial oversight
    CFO {
        risk_tolerance: f64,
        capital_efficiency: f64,
        financial_controls: f64,
    },
    
    /// Chief Risk Officer - Risk management
    CRO {
        risk_appetite: f64,
        stress_test_severity: f64,
        compliance_strictness: f64,
    },
    
    /// Chief Marketing Officer - Market analysis
    CMO {
        market_insight: f64,
        trend_sensitivity: f64,
        customer_focus: f64,
    },
    
    /// Chief Operations Officer - Operational efficiency
    COO {
        efficiency_optimization: f64,
        process_innovation: f64,
        resource_allocation: f64,
    },
    
    /// Chief Data Officer - Data governance
    CDO {
        data_quality_threshold: f64,
        analytics_sophistication: f64,
        privacy_compliance: f64,
    },
    
    /// Chief Security Officer - Security protocols
    CSO {
        security_paranoia: f64,
        threat_detection: f64,
        incident_response: f64,
    },
    
    /// Independent Director - External perspective
    IndependentDirector {
        objectivity_score: f64,
        industry_expertise: f64,
        governance_focus: f64,
    },
    
    /// Lead Independent Director - Board governance
    LeadIndependentDirector {
        leadership_strength: f64,
        board_effectiveness: f64,
        stakeholder_balance: f64,
    },
    
    /// Audit Committee Chair - Financial oversight
    AuditChair {
        audit_thoroughness: f64,
        financial_expertise: f64,
        independence_score: f64,
    },
    
    /// Compensation Committee Chair - Incentive alignment
    CompensationChair {
        performance_linkage: f64,
        market_competitiveness: f64,
        stakeholder_alignment: f64,
    },
    
    /// Nominating Committee Chair - Board composition
    NominatingChair {
        diversity_commitment: f64,
        skills_matrix_optimization: f64,
        succession_planning: f64,
    },
    
    /// Risk Committee Chair - Risk governance
    RiskChair {
        risk_oversight: f64,
        scenario_planning: f64,
        crisis_management: f64,
    },
    
    /// Technology Committee Chair - Digital transformation
    TechChair {
        digital_fluency: f64,
        innovation_appetite: f64,
        cybersecurity_awareness: f64,
    },
    
    /// ESG Committee Chair - Sustainability focus
    ESGChair {
        sustainability_commitment: f64,
        stakeholder_capitalism: f64,
        long_term_thinking: f64,
    },
    
    /// Quantum Advisor - Quantum computing expertise
    QuantumAdvisor {
        quantum_literacy: f64,
        computational_advantage: f64,
        future_readiness: f64,
    },
    
    /// AI Ethics Advisor - AI governance
    AIEthicsAdvisor {
        ethical_framework: f64,
        bias_detection: f64,
        transparency_advocacy: f64,
    },
}

/// Cognitive archetypes for sophisticated reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveArchetype {
    /// Self-reflective reasoning with meta-cognition
    SelfReflection {
        introspection_depth: u32,
        meta_level: u32,
    },
    
    /// Quantum superposition of multiple states
    QuantumSuperposition {
        states: Vec<String>,
        coherence: f64,
    },
    
    /// Barbell strategy with safe/risk allocation
    Barbell {
        safe_allocation: f64,
        risk_allocation: f64,
    },
    
    /// Black swan awareness and tail risk sensitivity
    BlackSwan {
        tail_sensitivity: f64,
        impact_threshold: f64,
    },
    
    /// Quantum tunneling through decision barriers
    QuantumTunneler {
        barrier_height: f64,
        tunnel_probability: f64,
    },
    
    /// Antifragile reasoning that gains from volatility
    Antifragile {
        convexity: f64,
        gain_from_disorder: f64,
    },
    
    /// Systems thinking with emergence detection
    SystemsThinking {
        interconnection_awareness: f64,
        emergence_sensitivity: f64,
    },
    
    /// Adaptive learning with experience integration
    AdaptiveLearning {
        learning_rate: f64,
        memory_decay: f64,
    },
}

/// Board member with cognitive capabilities
#[derive(Debug, Clone)]
pub struct BoardMember {
    pub id: u64,
    pub name: String,
    pub member_type: BoardMemberType,
    pub cognitive_archetype: CognitiveArchetype,
    pub reputation: ReputationScore,
    pub tenure: u64,
    pub decision_weight: f64,
    pub voting_power: f64,
    pub expertise_domains: Vec<String>,
    pub recent_decisions: Vec<DecisionRecord>,
    pub performance_metrics: PerformanceMetrics,
    pub cognitive_biases: Vec<CognitiveBias>,
    pub weight: f64,
    pub capabilities: Vec<String>,
}

/// Decision record for board member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub timestamp: u64,
    pub decision_type: String,
    pub outcome: f64,
    pub confidence: f64,
    pub context: String,
    pub market_conditions: MarketConditions,
}

/// Performance metrics for board member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub decision_accuracy: f64,
    pub response_time: f64,
    pub influence_score: f64,
    pub collaboration_rating: f64,
    pub innovation_index: f64,
    pub risk_adjusted_return: f64,
    pub stakeholder_satisfaction: f64,
    pub governance_effectiveness: f64,
}

/// Cognitive biases that affect decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveBias {
    ConfirmationBias { strength: f64 },
    AnchoringBias { reference_point: f64 },
    AvailabilityBias { recency_weight: f64 },
    OverconfidenceBias { confidence_inflation: f64 },
    LossBias { loss_aversion: f64 },
    HerdingBias { social_influence: f64 },
    StatusQuoBias { change_resistance: f64 },
    FramingBias { context_sensitivity: f64 },
    OptimismBias { positive_skew: f64 },
    RecencyBias { temporal_weighting: f64 },
}

/// Market conditions context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub trend: f64,
    pub liquidity: f64,
    pub sentiment: f64,
    pub economic_indicators: HashMap<String, f64>,
}

/// Reputation scoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    /// Overall reputation score
    pub overall: f64,
    /// Decision accuracy history
    pub accuracy_score: f64,
    /// Consistency in decision making
    pub consistency_score: f64,
    /// Collaboration effectiveness
    pub collaboration_score: f64,
    /// Innovation contribution
    pub innovation_score: f64,
    /// Risk management effectiveness
    pub risk_score: f64,
}

/// Decision context for board member
#[derive(Debug, Clone)]
pub struct DecisionContext {
    pub timestamp: u64,
    pub decision_type: String,
    pub description: String,
    pub market_state: MarketState,
    pub market_conditions: MarketConditions,
    pub potential_upside: f64,
    pub potential_downside: f64,
    pub group_sentiment: Option<f64>,
    pub represents_change: bool,
    pub positive_framing: bool,
    pub urgency: f64,
    pub complexity: f64,
}

/// Decision result from board member
#[derive(Debug, Clone)]
pub struct DecisionResult {
    pub decision: String,
    pub confidence: f64,
    pub reasoning: String,
    pub weight: f64,
    pub cognitive_factors: CognitiveFactors,
}

/// Cognitive factors in decision
#[derive(Debug, Clone)]
pub struct CognitiveFactors {
    pub archetype_influence: f64,
    pub bias_adjustments: f64,
    pub reputation_factor: f64,
    pub experience_factor: f64,
}

/// Feedback for performance updates
#[derive(Debug, Clone)]
pub struct Feedback {
    pub performance: f64,
    pub performance_threshold: f64,
    pub stability_factor: f64,
    pub decision_quality: f64,
    pub coordination_score: f64,
    pub prediction_accuracy: f64,
    pub market_complexity: f64,
    pub volatility_survived: f64,
    pub response_time: f64,
    pub influence: f64,
}

/// Market state for cognitive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub trend: f64,
    pub momentum: f64,
}

impl BoardMember {
    /// Create new board member
    pub fn new(
        id: u64,
        name: String,
        member_type: BoardMemberType,
        cognitive_archetype: CognitiveArchetype,
    ) -> Self {
        let expertise_domains = Self::derive_expertise_domains(&member_type);
        let initial_weight = Self::calculate_initial_weight(&member_type);
        let cognitive_biases = Self::initialize_biases(&member_type);
        
        Self {
            id,
            name,
            member_type,
            cognitive_archetype,
            reputation: ReputationScore::new(),
            tenure: 0,
            decision_weight: initial_weight,
            voting_power: 1.0,
            expertise_domains,
            recent_decisions: Vec::new(),
            performance_metrics: PerformanceMetrics::default(),
            cognitive_biases,
            weight: initial_weight,
            capabilities: Vec::new(),
        }
    }
    
    /// Simple constructor for compatibility
    pub fn simple_new(name: String, weight: f64) -> Self {
        Self {
            id: 0,
            name: name.clone(),
            member_type: BoardMemberType::IndependentDirector {
                objectivity_score: 0.8,
                industry_expertise: 0.7,
                governance_focus: 0.9,
            },
            cognitive_archetype: CognitiveArchetype::SystemsThinking {
                interconnection_awareness: 0.8,
                emergence_sensitivity: 0.7,
            },
            reputation: ReputationScore::new(),
            tenure: 0,
            decision_weight: weight,
            voting_power: 1.0,
            expertise_domains: vec!["general_governance".to_string()],
            recent_decisions: Vec::new(),
            performance_metrics: PerformanceMetrics::default(),
            cognitive_biases: Vec::new(),
            weight,
            capabilities: Vec::new(),
        }
    }
    
    /// Derive expertise domains from member type
    fn derive_expertise_domains(member_type: &BoardMemberType) -> Vec<String> {
        match member_type {
            BoardMemberType::CEO { .. } => vec![
                "strategy".to_string(),
                "leadership".to_string(),
                "vision".to_string(),
                "execution".to_string(),
            ],
            BoardMemberType::CTO { .. } => vec![
                "technology".to_string(),
                "innovation".to_string(),
                "engineering".to_string(),
                "systems".to_string(),
            ],
            BoardMemberType::CFO { .. } => vec![
                "finance".to_string(),
                "accounting".to_string(),
                "capital_markets".to_string(),
                "financial_planning".to_string(),
            ],
            BoardMemberType::CRO { .. } => vec![
                "risk_management".to_string(),
                "compliance".to_string(),
                "regulatory".to_string(),
                "governance".to_string(),
            ],
            BoardMemberType::QuantumAdvisor { .. } => vec![
                "quantum_computing".to_string(),
                "quantum_algorithms".to_string(),
                "quantum_cryptography".to_string(),
                "quantum_advantage".to_string(),
            ],
            BoardMemberType::AIEthicsAdvisor { .. } => vec![
                "ai_ethics".to_string(),
                "algorithmic_fairness".to_string(),
                "responsible_ai".to_string(),
                "ai_governance".to_string(),
            ],
            _ => vec!["general_governance".to_string()],
        }
    }
    
    /// Calculate initial decision weight
    fn calculate_initial_weight(member_type: &BoardMemberType) -> f64 {
        match member_type {
            BoardMemberType::CEO { strategic_weight, .. } => *strategic_weight,
            BoardMemberType::CTO { .. } => 0.85,
            BoardMemberType::CFO { .. } => 0.90,
            BoardMemberType::CRO { .. } => 0.80,
            BoardMemberType::LeadIndependentDirector { leadership_strength, .. } => *leadership_strength,
            BoardMemberType::AuditChair { .. } => 0.75,
            BoardMemberType::QuantumAdvisor { .. } => 0.70,
            BoardMemberType::AIEthicsAdvisor { .. } => 0.65,
            _ => 0.60,
        }
    }
    
    /// Initialize cognitive biases based on member type
    fn initialize_biases(member_type: &BoardMemberType) -> Vec<CognitiveBias> {
        match member_type {
            BoardMemberType::CEO { .. } => vec![
                CognitiveBias::OverconfidenceBias { confidence_inflation: 0.15 },
                CognitiveBias::ConfirmationBias { strength: 0.10 },
                CognitiveBias::FramingBias { context_sensitivity: 0.20 },
            ],
            BoardMemberType::CFO { .. } => vec![
                CognitiveBias::LossBias { loss_aversion: 0.25 },
                CognitiveBias::AnchoringBias { reference_point: 0.15 },
                CognitiveBias::StatusQuoBias { change_resistance: 0.12 },
            ],
            BoardMemberType::CRO { .. } => vec![
                CognitiveBias::AvailabilityBias { recency_weight: 0.20 },
                CognitiveBias::LossBias { loss_aversion: 0.30 },
                CognitiveBias::ConfirmationBias { strength: 0.08 },
            ],
            BoardMemberType::QuantumAdvisor { .. } => vec![
                CognitiveBias::OptimismBias { positive_skew: 0.15 },
                CognitiveBias::FramingBias { context_sensitivity: 0.10 },
            ],
            BoardMemberType::IndependentDirector { .. } => vec![
                CognitiveBias::StatusQuoBias { change_resistance: 0.05 },
                CognitiveBias::FramingBias { context_sensitivity: 0.08 },
            ],
            _ => vec![
                CognitiveBias::ConfirmationBias { strength: 0.10 },
                CognitiveBias::HerdingBias { social_influence: 0.15 },
            ],
        }
    }
    
    /// Make decision based on context and cognitive archetype
    pub fn make_decision(&mut self, context: &DecisionContext) -> DecisionResult {
        // Apply cognitive archetype reasoning
        let archetype_score = self.compute_archetype_activation(&context.market_state);
        
        // Apply cognitive biases
        let biased_score = self.apply_cognitive_biases(archetype_score, context);
        
        // Factor in reputation and experience
        let reputation_factor = self.reputation.get_overall_score();
        let experience_factor = (self.tenure as f64).ln() / 10.0;
        
        // Calculate final decision score
        let final_score = biased_score * reputation_factor * (1.0 + experience_factor);
        
        // Generate decision
        let decision = if final_score > 0.6 {
            "APPROVE".to_string()
        } else if final_score < 0.4 {
            "REJECT".to_string()
        } else {
            "ABSTAIN".to_string()
        };
        
        let confidence = (final_score * 0.8 + 0.2).clamp(0.0, 1.0);
        
        // Record decision
        let decision_record = DecisionRecord {
            timestamp: context.timestamp,
            decision_type: context.decision_type.clone(),
            outcome: final_score,
            confidence,
            context: context.description.clone(),
            market_conditions: context.market_conditions.clone(),
        };
        
        self.recent_decisions.push(decision_record);
        
        // Keep only recent decisions
        if self.recent_decisions.len() > 100 {
            self.recent_decisions.drain(0..50);
        }
        
        DecisionResult {
            decision,
            confidence,
            reasoning: self.generate_reasoning(context, final_score),
            weight: self.decision_weight,
            cognitive_factors: CognitiveFactors {
                archetype_influence: archetype_score,
                bias_adjustments: biased_score - archetype_score,
                reputation_factor,
                experience_factor,
            },
        }
    }
    
    /// Compute activation based on cognitive archetype
    fn compute_archetype_activation(&self, market_state: &MarketState) -> f64 {
        match &self.cognitive_archetype {
            CognitiveArchetype::SelfReflection { introspection_depth, meta_level } => {
                let complexity_factor = (market_state.volatility * market_state.momentum).sqrt();
                let reflection_boost = (*introspection_depth as f64) / 10.0 * (*meta_level as f64) / 5.0;
                (0.5 + complexity_factor * reflection_boost).clamp(0.0, 1.0)
            }
            
            CognitiveArchetype::QuantumSuperposition { coherence, .. } => {
                let quantum_factor = market_state.volatility * coherence;
                (0.6 + quantum_factor * 0.4).clamp(0.0, 1.0)
            }
            
            CognitiveArchetype::Barbell { safe_allocation, risk_allocation } => {
                let risk_factor = market_state.volatility;
                if risk_factor > 0.5 {
                    safe_allocation * 0.8 + risk_allocation * 0.2
                } else {
                    safe_allocation * 0.3 + risk_allocation * 0.7
                }
            }
            
            CognitiveArchetype::BlackSwan { tail_sensitivity, impact_threshold } => {
                let tail_event = market_state.volatility > *impact_threshold;
                if tail_event {
                    tail_sensitivity * 0.9
                } else {
                    0.5
                }
            }
            
            CognitiveArchetype::QuantumTunneler { tunnel_probability, .. } => {
                let barrier_present = market_state.trend.abs() < 0.1;
                if barrier_present {
                    tunnel_probability * 0.8
                } else {
                    0.6
                }
            }
            
            CognitiveArchetype::Antifragile { convexity, gain_from_disorder } => {
                let disorder = market_state.volatility;
                0.5 + disorder * gain_from_disorder * convexity
            }
            
            CognitiveArchetype::SystemsThinking { interconnection_awareness, emergence_sensitivity } => {
                let system_complexity = (market_state.volume / 1000.0).ln() / 10.0;
                (interconnection_awareness * system_complexity + emergence_sensitivity * 0.5).clamp(0.0, 1.0)
            }
            
            CognitiveArchetype::AdaptiveLearning { learning_rate, .. } => {
                let recent_performance = self.performance_metrics.decision_accuracy;
                (0.5 + learning_rate * recent_performance).clamp(0.0, 1.0)
            }
        }
    }
    
    /// Apply cognitive biases to decision score
    fn apply_cognitive_biases(&self, base_score: f64, context: &DecisionContext) -> f64 {
        let mut adjusted_score = base_score;
        
        for bias in &self.cognitive_biases {
            match bias {
                CognitiveBias::ConfirmationBias { strength } => {
                    let prior_belief = self.get_prior_belief(&context.decision_type);
                    let bias_adjustment = (prior_belief - 0.5) * strength;
                    adjusted_score += bias_adjustment;
                }
                
                CognitiveBias::AnchoringBias { reference_point } => {
                    let anchor_influence = (*reference_point - adjusted_score) * 0.3;
                    adjusted_score += anchor_influence;
                }
                
                CognitiveBias::AvailabilityBias { recency_weight } => {
                    if let Some(recent_decision) = self.recent_decisions.last() {
                        let recency_influence = (recent_decision.outcome - 0.5) * recency_weight;
                        adjusted_score += recency_influence;
                    }
                }
                
                CognitiveBias::OverconfidenceBias { confidence_inflation } => {
                    if adjusted_score > 0.5 {
                        adjusted_score += confidence_inflation;
                    }
                }
                
                CognitiveBias::LossBias { loss_aversion } => {
                    if context.potential_downside > 0.1 {
                        adjusted_score -= loss_aversion * context.potential_downside;
                    }
                }
                
                CognitiveBias::HerdingBias { social_influence } => {
                    if let Some(group_sentiment) = context.group_sentiment {
                        let herd_influence = (group_sentiment - 0.5) * social_influence;
                        adjusted_score += herd_influence;
                    }
                }
                
                CognitiveBias::StatusQuoBias { change_resistance } => {
                    if context.represents_change {
                        adjusted_score -= change_resistance;
                    }
                }
                
                CognitiveBias::FramingBias { context_sensitivity } => {
                    let framing_effect = if context.positive_framing {
                        context_sensitivity
                    } else {
                        -context_sensitivity
                    };
                    adjusted_score += framing_effect;
                }
                
                CognitiveBias::OptimismBias { positive_skew } => {
                    if context.potential_upside > context.potential_downside {
                        adjusted_score += positive_skew * (context.potential_upside - context.potential_downside);
                    }
                }
                
                CognitiveBias::RecencyBias { temporal_weighting } => {
                    let time_decay = (-context.urgency * temporal_weighting).exp();
                    adjusted_score *= time_decay;
                }
            }
        }
        
        adjusted_score.clamp(0.0, 1.0)
    }
    
    /// Get prior belief about decision type
    fn get_prior_belief(&self, decision_type: &str) -> f64 {
        let relevant_decisions: Vec<&DecisionRecord> = self.recent_decisions
            .iter()
            .filter(|d| d.decision_type == decision_type)
            .collect();
        
        if relevant_decisions.is_empty() {
            0.5 // Neutral prior
        } else {
            let avg_outcome: f64 = relevant_decisions.iter()
                .map(|d| d.outcome)
                .sum::<f64>() / relevant_decisions.len() as f64;
            avg_outcome
        }
    }
    
    /// Generate reasoning for decision
    fn generate_reasoning(&self, context: &DecisionContext, score: f64) -> String {
        let mut reasoning = Vec::new();
        
        reasoning.push(format!("Board member {} ({:?}) decision analysis:", 
            self.name, self.member_type));
        reasoning.push(format!("Decision score: {:.3}", score));
        reasoning.push(format!("Confidence: {:.3}", score));
        reasoning.push(format!("Cognitive archetype influence: {:?}", self.cognitive_archetype));
        reasoning.push(format!("Reputation factor: {:.3}", self.reputation.get_overall_score()));
        reasoning.push(format!("Experience factor: {:.3}", (self.tenure as f64).ln() / 10.0));
        
        if !self.cognitive_biases.is_empty() {
            reasoning.push(format!("Active cognitive biases: {} biases applied", self.cognitive_biases.len()));
        }
        
        reasoning.push(format!("Market conditions: volatility={:.3}, trend={:.3}", 
            context.market_state.volatility, context.market_state.trend));
        
        reasoning.join("\n")
    }
    
    /// Update performance metrics
    pub fn update_performance(&mut self, feedback: &Feedback) {
        self.performance_metrics.decision_accuracy = 
            self.performance_metrics.decision_accuracy * 0.9 + feedback.performance * 0.1;
        
        self.performance_metrics.response_time = 
            self.performance_metrics.response_time * 0.9 + feedback.response_time * 0.1;
        
        self.performance_metrics.influence_score = 
            self.performance_metrics.influence_score * 0.9 + feedback.influence * 0.1;
        
        // Update reputation based on performance
        self.reputation.update_performance(feedback.performance);
        
        // Adjust decision weight based on performance
        self.decision_weight = self.calculate_dynamic_weight();
    }
    
    /// Calculate dynamic weight based on performance
    fn calculate_dynamic_weight(&self) -> f64 {
        let base_weight = Self::calculate_initial_weight(&self.member_type);
        let performance_factor = self.performance_metrics.decision_accuracy;
        let reputation_factor = self.reputation.get_overall_score();
        let experience_factor = (self.tenure as f64 / 365.0).ln().max(0.0) / 5.0;
        
        base_weight * performance_factor * reputation_factor * (1.0 + experience_factor)
    }
    
    /// Update reputation
    pub fn update_reputation(&mut self, new_reputation: f64) {
        self.reputation.overall = new_reputation;
    }
    
    /// Add capability
    pub fn add_capability(&mut self, capability: String) {
        if !self.capabilities.contains(&capability) {
            self.capabilities.push(capability);
        }
    }
}

impl ReputationScore {
    /// Create new reputation score
    pub fn new() -> Self {
        Self {
            overall: 0.5,
            accuracy_score: 0.5,
            consistency_score: 0.5,
            collaboration_score: 0.5,
            innovation_score: 0.5,
            risk_score: 0.5,
        }
    }
    
    /// Get overall reputation score
    pub fn get_overall_score(&self) -> f64 {
        self.overall
    }
    
    /// Update performance-based reputation
    pub fn update_performance(&mut self, performance: f64) {
        self.accuracy_score = self.accuracy_score * 0.9 + performance * 0.1;
        self.overall = (
            self.accuracy_score * 0.3 +
            self.consistency_score * 0.2 +
            self.collaboration_score * 0.2 +
            self.innovation_score * 0.15 +
            self.risk_score * 0.15
        );
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            decision_accuracy: 0.5,
            response_time: 1.0,
            influence_score: 0.5,
            collaboration_rating: 0.5,
            innovation_index: 0.5,
            risk_adjusted_return: 0.0,
            stakeholder_satisfaction: 0.5,
            governance_effectiveness: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_member_creation() {
        let member_type = BoardMemberType::CEO {
            vision_horizon: 1825,
            strategic_weight: 0.95,
            executive_authority: 0.90,
        };
        
        let archetype = CognitiveArchetype::SelfReflection {
            introspection_depth: 5,
            meta_level: 3,
        };
        
        let member = BoardMember::new(1, "Test CEO".to_string(), member_type, archetype);
        
        assert_eq!(member.name, "Test CEO");
        assert_eq!(member.expertise_domains.len(), 4);
        assert!(member.decision_weight > 0.0);
        assert!(!member.cognitive_biases.is_empty());
    }

    #[test]
    fn test_cognitive_archetype_activation() {
        let member_type = BoardMemberType::QuantumAdvisor {
            quantum_literacy: 0.95,
            computational_advantage: 0.85,
            future_readiness: 0.90,
        };
        
        let archetype = CognitiveArchetype::QuantumSuperposition {
            states: vec!["up".to_string(), "down".to_string()],
            coherence: 0.8,
        };
        
        let member = BoardMember::new(1, "Quantum Advisor".to_string(), member_type, archetype);
        
        let market_state = MarketState {
            price: 100.0,
            volume: 1000.0,
            volatility: 0.3,
            trend: 0.1,
            momentum: 0.05,
        };
        
        let activation = member.compute_archetype_activation(&market_state);
        assert!(activation >= 0.0 && activation <= 1.0);
    }

    #[test]
    fn test_cognitive_bias_application() {
        let mut member = BoardMember::new(
            1,
            "Test Member".to_string(),
            BoardMemberType::CFO {
                risk_tolerance: 0.6,
                capital_efficiency: 0.85,
                financial_controls: 0.90,
            },
            CognitiveArchetype::Barbell {
                safe_allocation: 0.7,
                risk_allocation: 0.3,
            },
        );
        
        let context = DecisionContext {
            timestamp: 1234567890,
            decision_type: "risk_decision".to_string(),
            description: "High risk investment".to_string(),
            market_state: MarketState {
                price: 100.0,
                volume: 1000.0,
                volatility: 0.3,
                trend: 0.0,
                momentum: 0.0,
            },
            market_conditions: MarketConditions {
                volatility: 0.3,
                trend: 0.0,
                liquidity: 0.7,
                sentiment: 0.5,
                economic_indicators: HashMap::new(),
            },
            potential_upside: 0.5,
            potential_downside: 0.4,
            group_sentiment: None,
            represents_change: true,
            positive_framing: false,
            urgency: 0.8,
            complexity: 0.9,
        };
        
        let result = member.make_decision(&context);
        
        assert!(result.confidence > 0.0);
        assert!(!result.reasoning.is_empty());
        assert!(result.weight > 0.0);
    }

    #[test]
    fn test_reputation_updates() {
        let mut reputation = ReputationScore::new();
        assert_eq!(reputation.overall, 0.5);
        
        reputation.update_performance(0.8);
        assert!(reputation.overall > 0.5);
        assert!(reputation.accuracy_score > 0.5);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.decision_accuracy, 0.5);
        assert_eq!(metrics.response_time, 1.0);
        assert_eq!(metrics.collaboration_rating, 0.5);
    }
}