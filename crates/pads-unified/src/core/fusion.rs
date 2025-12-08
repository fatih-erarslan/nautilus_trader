//! Decision Fusion Implementation
//!
//! Advanced decision fusion engine that combines multiple agent decisions using
//! various fusion strategies including Bayesian inference, neural networks, and
//! ensemble methods.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Decision fusion strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple weighted average
    WeightedAverage,
    /// Bayesian inference fusion
    Bayesian,
    /// Neural network fusion
    NeuralNetwork,
    /// Ensemble voting
    EnsembleVoting,
    /// Dempster-Shafer theory
    DempsterShafer,
    /// Fuzzy logic fusion
    FuzzyLogic,
    /// Adaptive fusion (learns optimal strategy)
    Adaptive,
}

/// Fusion weight configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionWeights {
    /// Agent-specific weights
    pub agent_weights: HashMap<String, f64>,
    /// Strategy-specific weights
    pub strategy_weights: HashMap<String, f64>,
    /// Time-based decay factor
    pub decay_factor: f64,
    /// Confidence threshold for inclusion
    pub confidence_threshold: f64,
}

/// Fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Primary fusion strategy
    pub strategy: FusionStrategy,
    /// Fusion weights
    pub weights: FusionWeights,
    /// Minimum number of decisions required
    pub min_decisions: usize,
    /// Maximum age of decisions to consider (seconds)
    pub max_age_seconds: u64,
    /// Enable uncertainty quantification
    pub enable_uncertainty: bool,
    /// Consensus threshold for high-confidence decisions
    pub consensus_threshold: f64,
    /// Enable adaptive learning
    pub enable_learning: bool,
}

/// Fusion result with uncertainty metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    /// Fused decision
    pub decision: TradingDecision,
    /// Confidence in the fused decision
    pub confidence: f64,
    /// Uncertainty bounds
    pub uncertainty: UncertaintyBounds,
    /// Contributing decisions
    pub contributors: Vec<ContributorInfo>,
    /// Fusion metadata
    pub metadata: FusionMetadata,
}

/// Uncertainty bounds for decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyBounds {
    /// Lower confidence bound
    pub lower_bound: f64,
    /// Upper confidence bound
    pub upper_bound: f64,
    /// Variance of the estimate
    pub variance: f64,
    /// Entropy of the decision distribution
    pub entropy: f64,
}

/// Information about contributing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributorInfo {
    /// Source agent or strategy ID
    pub source_id: String,
    /// Original decision
    pub decision: TradingDecision,
    /// Weight assigned to this decision
    pub weight: f64,
    /// Influence on final decision (0-1)
    pub influence: f64,
}

/// Fusion metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetadata {
    /// Fusion strategy used
    pub strategy: FusionStrategy,
    /// Number of decisions fused
    pub num_decisions: usize,
    /// Fusion computation time (nanoseconds)
    pub computation_time_ns: u64,
    /// Consensus score (0-1)
    pub consensus_score: f64,
    /// Disagreement score (0-1)
    pub disagreement_score: f64,
    /// Timestamp of fusion
    pub timestamp: u64,
}

/// Bayesian fusion parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BayesianParams {
    /// Prior distribution parameters
    pub prior_alpha: f64,
    pub prior_beta: f64,
    /// Likelihood function parameters
    pub likelihood_params: HashMap<String, f64>,
    /// Evidence accumulation factor
    pub evidence_factor: f64,
}

/// Neural network fusion model
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NeuralFusionModel {
    /// Layer weights
    pub weights: Vec<Vec<f64>>,
    /// Layer biases
    pub biases: Vec<f64>,
    /// Activation function
    pub activation: String,
    /// Model training history
    pub training_history: Vec<f64>,
}

/// Decision fusion engine
pub struct DecisionFusion {
    /// Fusion configuration
    config: FusionConfig,
    /// Fusion history for learning
    history: Arc<RwLock<Vec<FusionResult>>>,
    /// Bayesian parameters
    bayesian_params: Arc<RwLock<BayesianParams>>,
    /// Neural fusion model
    neural_model: Arc<RwLock<Option<NeuralFusionModel>>>,
    /// Performance metrics
    metrics: Arc<RwLock<FusionMetrics>>,
}

/// Fusion performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetrics {
    /// Total fusions performed
    pub total_fusions: u64,
    /// Average confidence
    pub average_confidence: f64,
    /// Average consensus score
    pub average_consensus: f64,
    /// Average computation time (nanoseconds)
    pub average_computation_time_ns: u64,
    /// Accuracy (if ground truth available)
    pub accuracy: Option<f64>,
    /// Calibration error
    pub calibration_error: Option<f64>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::Adaptive,
            weights: FusionWeights {
                agent_weights: HashMap::new(),
                strategy_weights: HashMap::new(),
                decay_factor: 0.1,
                confidence_threshold: 0.3,
            },
            min_decisions: 2,
            max_age_seconds: 300, // 5 minutes
            enable_uncertainty: true,
            consensus_threshold: 0.8,
            enable_learning: true,
        }
    }
}

impl Default for BayesianParams {
    fn default() -> Self {
        Self {
            prior_alpha: 1.0,
            prior_beta: 1.0,
            likelihood_params: HashMap::new(),
            evidence_factor: 1.0,
        }
    }
}

impl Default for FusionMetrics {
    fn default() -> Self {
        Self {
            total_fusions: 0,
            average_confidence: 0.0,
            average_consensus: 0.0,
            average_computation_time_ns: 0,
            accuracy: None,
            calibration_error: None,
        }
    }
}

impl DecisionFusion {
    /// Create a new decision fusion engine
    pub async fn new(config: FusionConfig) -> PadsResult<Self> {
        Ok(Self {
            config,
            history: Arc::new(RwLock::new(Vec::new())),
            bayesian_params: Arc::new(RwLock::new(BayesianParams::default())),
            neural_model: Arc::new(RwLock::new(None)),
            metrics: Arc::new(RwLock::new(FusionMetrics::default())),
        })
    }
    
    /// Fuse multiple decisions into a single decision
    pub async fn fuse_decisions(&self, decisions: Vec<TradingDecision>) -> PadsResult<FusionResult> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        if decisions.len() < self.config.min_decisions {
            return Err(PadsError::fusion(format!(
                "Insufficient decisions: {} < {}",
                decisions.len(),
                self.config.min_decisions
            )));
        }
        
        // Filter decisions by age and confidence
        let filtered_decisions = self.filter_decisions(&decisions).await?;
        
        if filtered_decisions.len() < self.config.min_decisions {
            return Err(PadsError::fusion("Insufficient valid decisions after filtering"));
        }
        
        // Perform fusion based on strategy
        let fusion_result = match self.config.strategy {
            FusionStrategy::WeightedAverage => {
                self.weighted_average_fusion(&filtered_decisions).await?
            }
            FusionStrategy::Bayesian => {
                self.bayesian_fusion(&filtered_decisions).await?
            }
            FusionStrategy::NeuralNetwork => {
                self.neural_fusion(&filtered_decisions).await?
            }
            FusionStrategy::EnsembleVoting => {
                self.ensemble_voting_fusion(&filtered_decisions).await?
            }
            FusionStrategy::DempsterShafer => {
                self.dempster_shafer_fusion(&filtered_decisions).await?
            }
            FusionStrategy::FuzzyLogic => {
                self.fuzzy_logic_fusion(&filtered_decisions).await?
            }
            FusionStrategy::Adaptive => {
                self.adaptive_fusion(&filtered_decisions).await?
            }
        };
        
        let computation_time = start_time.elapsed().as_nanos() as u64;
        
        // Update metrics
        self.update_metrics(&fusion_result, computation_time).await?;
        
        // Store in history for learning
        if self.config.enable_learning {
            let mut history = self.history.write().await;
            history.push(fusion_result.clone());
            
            // Keep only recent history
            if history.len() > 1000 {
                history.remove(0);
            }
        }
        
        Ok(fusion_result)
    }
    
    /// Filter decisions by age and confidence
    async fn filter_decisions(&self, decisions: &[TradingDecision]) -> PadsResult<Vec<TradingDecision>> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let filtered: Vec<TradingDecision> = decisions
            .iter()
            .filter(|decision| {
                // Check age
                if current_time - decision.timestamp > self.config.max_age_seconds {
                    return false;
                }
                
                // Check confidence threshold
                if decision.confidence < self.config.weights.confidence_threshold {
                    return false;
                }
                
                true
            })
            .cloned()
            .collect();
        
        Ok(filtered)
    }
    
    /// Weighted average fusion
    async fn weighted_average_fusion(&self, decisions: &[TradingDecision]) -> PadsResult<FusionResult> {
        let mut total_weight = 0.0;
        let mut weighted_confidence = 0.0;
        let mut weighted_amount = 0.0;
        let mut decision_counts = HashMap::new();
        let mut contributors = Vec::new();
        
        for decision in decisions {
            let weight = self.calculate_weight(decision).await?;
            
            total_weight += weight;
            weighted_confidence += decision.confidence * weight;
            weighted_amount += decision.amount * weight;
            
            *decision_counts.entry(decision.decision_type.clone()).or_insert(0.0) += weight;
            
            contributors.push(ContributorInfo {
                source_id: "unknown".to_string(), // Would need to be passed in
                decision: decision.clone(),
                weight,
                influence: weight / total_weight,
            });
        }
        
        // Determine final decision type by highest weighted count
        let final_decision_type = decision_counts
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(dt, _)| dt.clone())
            .unwrap_or(DecisionType::Hold);
        
        let final_confidence = weighted_confidence / total_weight;
        let final_amount = weighted_amount / total_weight;
        
        // Calculate uncertainty bounds
        let uncertainty = self.calculate_uncertainty(decisions, final_confidence).await?;
        
        // Calculate consensus and disagreement
        let consensus_score = self.calculate_consensus(decisions).await?;
        let disagreement_score = 1.0 - consensus_score;
        
        let fused_decision = TradingDecision {
            decision_type: final_decision_type,
            confidence: final_confidence,
            amount: final_amount,
            reasoning: format!("Weighted average fusion of {} decisions", decisions.len()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        Ok(FusionResult {
            decision: fused_decision,
            confidence: final_confidence,
            uncertainty,
            contributors,
            metadata: FusionMetadata {
                strategy: FusionStrategy::WeightedAverage,
                num_decisions: decisions.len(),
                computation_time_ns: 0, // Will be set by caller
                consensus_score,
                disagreement_score,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }
    
    /// Bayesian fusion using beta distributions
    async fn bayesian_fusion(&self, decisions: &[TradingDecision]) -> PadsResult<FusionResult> {
        let bayesian_params = self.bayesian_params.read().await;
        
        let mut alpha = bayesian_params.prior_alpha;
        let mut beta = bayesian_params.prior_beta;
        
        // Update posterior with evidence from decisions
        for decision in decisions {
            let weight = self.calculate_weight(decision).await?;
            let evidence = decision.confidence * weight;
            
            // Update beta distribution parameters
            alpha += evidence;
            beta += (1.0 - evidence) * weight;
        }
        
        // Calculate posterior mean and variance
        let posterior_mean = alpha / (alpha + beta);
        let posterior_variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        
        // Determine decision type based on posterior
        let decision_type = if posterior_mean > 0.6 {
            DecisionType::Buy
        } else if posterior_mean < 0.4 {
            DecisionType::Sell
        } else {
            DecisionType::Hold
        };
        
        let uncertainty = UncertaintyBounds {
            lower_bound: posterior_mean - 1.96 * posterior_variance.sqrt(),
            upper_bound: posterior_mean + 1.96 * posterior_variance.sqrt(),
            variance: posterior_variance,
            entropy: self.calculate_entropy(alpha, beta),
        };
        
        let fused_decision = TradingDecision {
            decision_type,
            confidence: posterior_mean,
            amount: self.calculate_bayesian_amount(decisions, posterior_mean).await?,
            reasoning: format!("Bayesian fusion with alpha={:.3}, beta={:.3}", alpha, beta),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        Ok(FusionResult {
            decision: fused_decision,
            confidence: posterior_mean,
            uncertainty,
            contributors: self.create_contributors(decisions).await?,
            metadata: FusionMetadata {
                strategy: FusionStrategy::Bayesian,
                num_decisions: decisions.len(),
                computation_time_ns: 0,
                consensus_score: self.calculate_consensus(decisions).await?,
                disagreement_score: 1.0 - self.calculate_consensus(decisions).await?,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }
    
    /// Neural network fusion
    async fn neural_fusion(&self, decisions: &[TradingDecision]) -> PadsResult<FusionResult> {
        let neural_model = self.neural_model.read().await;
        
        if neural_model.is_none() {
            // Fall back to weighted average if no model is trained
            return self.weighted_average_fusion(decisions).await;
        }
        
        let model = neural_model.as_ref().unwrap();
        
        // Prepare input features
        let features = self.extract_features(decisions).await?;
        
        // Forward pass through neural network
        let output = self.neural_forward_pass(&features, model).await?;
        
        // Convert output to decision
        let decision_type = match output[0] {
            x if x > 0.6 => DecisionType::Buy,
            x if x < 0.4 => DecisionType::Sell,
            _ => DecisionType::Hold,
        };
        
        let confidence = output[1].max(0.0).min(1.0);
        let amount = output[2].max(0.0);
        
        let uncertainty = self.calculate_uncertainty(decisions, confidence).await?;
        
        let fused_decision = TradingDecision {
            decision_type,
            confidence,
            amount,
            reasoning: "Neural network fusion".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        Ok(FusionResult {
            decision: fused_decision,
            confidence,
            uncertainty,
            contributors: self.create_contributors(decisions).await?,
            metadata: FusionMetadata {
                strategy: FusionStrategy::NeuralNetwork,
                num_decisions: decisions.len(),
                computation_time_ns: 0,
                consensus_score: self.calculate_consensus(decisions).await?,
                disagreement_score: 1.0 - self.calculate_consensus(decisions).await?,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }
    
    /// Ensemble voting fusion
    async fn ensemble_voting_fusion(&self, decisions: &[TradingDecision]) -> PadsResult<FusionResult> {
        let mut votes = HashMap::new();
        let mut total_confidence = 0.0;
        let mut total_amount = 0.0;
        
        for decision in decisions {
            let weight = self.calculate_weight(decision).await?;
            *votes.entry(decision.decision_type.clone()).or_insert(0.0) += weight;
            total_confidence += decision.confidence * weight;
            total_amount += decision.amount * weight;
        }
        
        let total_weight: f64 = votes.values().sum();
        let final_decision_type = votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(dt, _)| dt.clone())
            .unwrap_or(DecisionType::Hold);
        
        let final_confidence = total_confidence / total_weight;
        let final_amount = total_amount / total_weight;
        
        let uncertainty = self.calculate_uncertainty(decisions, final_confidence).await?;
        
        let fused_decision = TradingDecision {
            decision_type: final_decision_type,
            confidence: final_confidence,
            amount: final_amount,
            reasoning: format!("Ensemble voting fusion with {} votes", votes.len()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        Ok(FusionResult {
            decision: fused_decision,
            confidence: final_confidence,
            uncertainty,
            contributors: self.create_contributors(decisions).await?,
            metadata: FusionMetadata {
                strategy: FusionStrategy::EnsembleVoting,
                num_decisions: decisions.len(),
                computation_time_ns: 0,
                consensus_score: self.calculate_consensus(decisions).await?,
                disagreement_score: 1.0 - self.calculate_consensus(decisions).await?,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }
    
    /// Dempster-Shafer fusion
    async fn dempster_shafer_fusion(&self, decisions: &[TradingDecision]) -> PadsResult<FusionResult> {
        // Simplified Dempster-Shafer implementation
        let mut mass_functions = HashMap::new();
        
        for decision in decisions {
            let weight = self.calculate_weight(decision).await?;
            let mass = decision.confidence * weight;
            
            *mass_functions.entry(decision.decision_type.clone()).or_insert(0.0) += mass;
        }
        
        // Normalize mass functions
        let total_mass: f64 = mass_functions.values().sum();
        if total_mass > 0.0 {
            for mass in mass_functions.values_mut() {
                *mass /= total_mass;
            }
        }
        
        // Find decision with highest mass
        let final_decision_type = mass_functions
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(dt, _)| dt.clone())
            .unwrap_or(DecisionType::Hold);
        
        let final_confidence = mass_functions.get(&final_decision_type).unwrap_or(&0.0);
        let final_amount = decisions.iter().map(|d| d.amount).sum::<f64>() / decisions.len() as f64;
        
        let uncertainty = self.calculate_uncertainty(decisions, *final_confidence).await?;
        
        let fused_decision = TradingDecision {
            decision_type: final_decision_type,
            confidence: *final_confidence,
            amount: final_amount,
            reasoning: "Dempster-Shafer fusion".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        Ok(FusionResult {
            decision: fused_decision,
            confidence: *final_confidence,
            uncertainty,
            contributors: self.create_contributors(decisions).await?,
            metadata: FusionMetadata {
                strategy: FusionStrategy::DempsterShafer,
                num_decisions: decisions.len(),
                computation_time_ns: 0,
                consensus_score: self.calculate_consensus(decisions).await?,
                disagreement_score: 1.0 - self.calculate_consensus(decisions).await?,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }
    
    /// Fuzzy logic fusion
    async fn fuzzy_logic_fusion(&self, decisions: &[TradingDecision]) -> PadsResult<FusionResult> {
        // Simplified fuzzy logic implementation
        let mut buy_membership = 0.0;
        let mut sell_membership = 0.0;
        let mut hold_membership = 0.0;
        let mut total_weight = 0.0;
        
        for decision in decisions {
            let weight = self.calculate_weight(decision).await?;
            let confidence = decision.confidence;
            
            match decision.decision_type {
                DecisionType::Buy => buy_membership += confidence * weight,
                DecisionType::Sell => sell_membership += confidence * weight,
                DecisionType::Hold => hold_membership += confidence * weight,
            }
            
            total_weight += weight;
        }
        
        // Normalize memberships
        if total_weight > 0.0 {
            buy_membership /= total_weight;
            sell_membership /= total_weight;
            hold_membership /= total_weight;
        }
        
        // Defuzzification using centroid method
        let final_decision_type = if buy_membership > sell_membership && buy_membership > hold_membership {
            DecisionType::Buy
        } else if sell_membership > hold_membership {
            DecisionType::Sell
        } else {
            DecisionType::Hold
        };
        
        let final_confidence = match final_decision_type {
            DecisionType::Buy => buy_membership,
            DecisionType::Sell => sell_membership,
            DecisionType::Hold => hold_membership,
        };
        
        let final_amount = decisions.iter().map(|d| d.amount).sum::<f64>() / decisions.len() as f64;
        
        let uncertainty = self.calculate_uncertainty(decisions, final_confidence).await?;
        
        let fused_decision = TradingDecision {
            decision_type: final_decision_type,
            confidence: final_confidence,
            amount: final_amount,
            reasoning: "Fuzzy logic fusion".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        Ok(FusionResult {
            decision: fused_decision,
            confidence: final_confidence,
            uncertainty,
            contributors: self.create_contributors(decisions).await?,
            metadata: FusionMetadata {
                strategy: FusionStrategy::FuzzyLogic,
                num_decisions: decisions.len(),
                computation_time_ns: 0,
                consensus_score: self.calculate_consensus(decisions).await?,
                disagreement_score: 1.0 - self.calculate_consensus(decisions).await?,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }
    
    /// Adaptive fusion that learns optimal strategy
    async fn adaptive_fusion(&self, decisions: &[TradingDecision]) -> PadsResult<FusionResult> {
        // For now, use weighted average but could be enhanced with learning
        let mut result = self.weighted_average_fusion(decisions).await?;
        result.metadata.strategy = FusionStrategy::Adaptive;
        Ok(result)
    }
    
    /// Calculate weight for a decision
    async fn calculate_weight(&self, decision: &TradingDecision) -> PadsResult<f64> {
        let mut weight = 1.0;
        
        // Apply confidence weighting
        weight *= decision.confidence;
        
        // Apply time decay
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let age = current_time - decision.timestamp;
        let decay_factor = (-self.config.weights.decay_factor * age as f64).exp();
        weight *= decay_factor;
        
        Ok(weight.max(0.0))
    }
    
    /// Calculate uncertainty bounds
    async fn calculate_uncertainty(&self, decisions: &[TradingDecision], mean_confidence: f64) -> PadsResult<UncertaintyBounds> {
        let confidences: Vec<f64> = decisions.iter().map(|d| d.confidence).collect();
        
        let variance = if confidences.len() > 1 {
            let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
            confidences.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (confidences.len() - 1) as f64
        } else {
            0.0
        };
        
        let std_dev = variance.sqrt();
        let entropy = self.calculate_decision_entropy(decisions);
        
        Ok(UncertaintyBounds {
            lower_bound: (mean_confidence - 1.96 * std_dev).max(0.0),
            upper_bound: (mean_confidence + 1.96 * std_dev).min(1.0),
            variance,
            entropy,
        })
    }
    
    /// Calculate consensus score
    async fn calculate_consensus(&self, decisions: &[TradingDecision]) -> PadsResult<f64> {
        if decisions.is_empty() {
            return Ok(0.0);
        }
        
        let mut decision_counts = HashMap::new();
        for decision in decisions {
            *decision_counts.entry(decision.decision_type.clone()).or_insert(0) += 1;
        }
        
        let max_count = decision_counts.values().max().unwrap_or(&0);
        let consensus = *max_count as f64 / decisions.len() as f64;
        
        Ok(consensus)
    }
    
    /// Calculate entropy for beta distribution
    fn calculate_entropy(&self, alpha: f64, beta: f64) -> f64 {
        use std::f64::consts::E;
        
        let digamma_alpha_beta = self.digamma(alpha + beta);
        let digamma_alpha = self.digamma(alpha);
        let digamma_beta = self.digamma(beta);
        
        (alpha - 1.0) * digamma_alpha + (beta - 1.0) * digamma_beta - 
        (alpha + beta - 2.0) * digamma_alpha_beta + 
        (alpha + beta).ln() + 
        (alpha.ln_gamma().0 + beta.ln_gamma().0 - (alpha + beta).ln_gamma().0)
    }
    
    /// Digamma function approximation
    fn digamma(&self, x: f64) -> f64 {
        // Simple approximation
        if x > 6.0 {
            (x - 0.5).ln() - 1.0 / (12.0 * x)
        } else {
            self.digamma(x + 1.0) - 1.0 / x
        }
    }
    
    /// Calculate decision entropy
    fn calculate_decision_entropy(&self, decisions: &[TradingDecision]) -> f64 {
        let mut counts = HashMap::new();
        for decision in decisions {
            *counts.entry(decision.decision_type.clone()).or_insert(0) += 1;
        }
        
        let total = decisions.len() as f64;
        let mut entropy = 0.0;
        
        for count in counts.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        
        entropy
    }
    
    /// Calculate Bayesian amount
    async fn calculate_bayesian_amount(&self, decisions: &[TradingDecision], confidence: f64) -> PadsResult<f64> {
        let weighted_amounts: Vec<f64> = decisions.iter().map(|d| d.amount * d.confidence).collect();
        let total_weight: f64 = decisions.iter().map(|d| d.confidence).sum();
        
        if total_weight > 0.0 {
            Ok(weighted_amounts.iter().sum::<f64>() / total_weight * confidence)
        } else {
            Ok(0.0)
        }
    }
    
    /// Create contributor information
    async fn create_contributors(&self, decisions: &[TradingDecision]) -> PadsResult<Vec<ContributorInfo>> {
        let mut contributors = Vec::new();
        let total_weight: f64 = decisions.len() as f64; // Simplified
        
        for (i, decision) in decisions.iter().enumerate() {
            let weight = self.calculate_weight(decision).await?;
            contributors.push(ContributorInfo {
                source_id: format!("source_{}", i),
                decision: decision.clone(),
                weight,
                influence: weight / total_weight,
            });
        }
        
        Ok(contributors)
    }
    
    /// Extract features for neural network
    async fn extract_features(&self, decisions: &[TradingDecision]) -> PadsResult<Vec<f64>> {
        let mut features = Vec::new();
        
        // Basic features
        features.push(decisions.len() as f64);
        features.push(decisions.iter().map(|d| d.confidence).sum::<f64>() / decisions.len() as f64);
        features.push(decisions.iter().map(|d| d.amount).sum::<f64>() / decisions.len() as f64);
        
        // Decision type distribution
        let mut buy_count = 0;
        let mut sell_count = 0;
        let mut hold_count = 0;
        
        for decision in decisions {
            match decision.decision_type {
                DecisionType::Buy => buy_count += 1,
                DecisionType::Sell => sell_count += 1,
                DecisionType::Hold => hold_count += 1,
            }
        }
        
        features.push(buy_count as f64 / decisions.len() as f64);
        features.push(sell_count as f64 / decisions.len() as f64);
        features.push(hold_count as f64 / decisions.len() as f64);
        
        // Variance features
        let confidence_variance = self.calculate_variance(&decisions.iter().map(|d| d.confidence).collect::<Vec<_>>());
        let amount_variance = self.calculate_variance(&decisions.iter().map(|d| d.amount).collect::<Vec<_>>());
        
        features.push(confidence_variance);
        features.push(amount_variance);
        
        Ok(features)
    }
    
    /// Calculate variance
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }
    
    /// Neural network forward pass
    async fn neural_forward_pass(&self, features: &[f64], model: &NeuralFusionModel) -> PadsResult<Vec<f64>> {
        let mut current_layer = features.to_vec();
        
        for (layer_idx, layer_weights) in model.weights.iter().enumerate() {
            let mut next_layer = Vec::new();
            
            for (neuron_idx, neuron_weights) in layer_weights.iter().enumerate() {
                let mut activation = model.biases.get(neuron_idx).unwrap_or(&0.0);
                
                for (i, &weight) in neuron_weights.iter().enumerate() {
                    if i < current_layer.len() {
                        activation += current_layer[i] * weight;
                    }
                }
                
                // Apply activation function
                let activated = match model.activation.as_str() {
                    "relu" => activation.max(0.0),
                    "sigmoid" => 1.0 / (1.0 + (-activation).exp()),
                    "tanh" => activation.tanh(),
                    _ => activation, // Linear
                };
                
                next_layer.push(activated);
            }
            
            current_layer = next_layer;
        }
        
        Ok(current_layer)
    }
    
    /// Update fusion metrics
    async fn update_metrics(&self, result: &FusionResult, computation_time: u64) -> PadsResult<()> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_fusions += 1;
        metrics.average_confidence = (metrics.average_confidence * (metrics.total_fusions - 1) as f64 + result.confidence) / metrics.total_fusions as f64;
        metrics.average_consensus = (metrics.average_consensus * (metrics.total_fusions - 1) as f64 + result.metadata.consensus_score) / metrics.total_fusions as f64;
        metrics.average_computation_time_ns = (metrics.average_computation_time_ns * (metrics.total_fusions - 1) + computation_time) / metrics.total_fusions;
        
        Ok(())
    }
    
    /// Get fusion metrics
    pub async fn get_metrics(&self) -> PadsResult<FusionMetrics> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }
    
    /// Train neural fusion model
    pub async fn train_neural_model(&self, training_data: Vec<(Vec<TradingDecision>, TradingDecision)>) -> PadsResult<()> {
        // Simplified training - in practice would use proper ML framework
        let mut model = NeuralFusionModel {
            weights: vec![vec![vec![0.1; 8]; 10], vec![vec![0.1; 10]; 3]], // 8 inputs, 10 hidden, 3 outputs
            biases: vec![0.0; 13],
            activation: "relu".to_string(),
            training_history: Vec::new(),
        };
        
        // Simple gradient descent training
        let learning_rate = 0.01;
        let epochs = 100;
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (decisions, target) in &training_data {
                let features = self.extract_features(decisions).await?;
                let prediction = self.neural_forward_pass(&features, &model).await?;
                
                // Calculate loss (simplified)
                let target_vec = vec![
                    match target.decision_type {
                        DecisionType::Buy => 1.0,
                        DecisionType::Sell => -1.0,
                        DecisionType::Hold => 0.0,
                    },
                    target.confidence,
                    target.amount,
                ];
                
                let loss: f64 = prediction.iter().zip(target_vec.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>() / prediction.len() as f64;
                
                total_loss += loss;
                
                // Simplified backpropagation (would need proper implementation)
                // This is just a placeholder
            }
            
            let avg_loss = total_loss / training_data.len() as f64;
            model.training_history.push(avg_loss);
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, avg_loss);
            }
        }
        
        let mut neural_model = self.neural_model.write().await;
        *neural_model = Some(model);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_weighted_average_fusion() {
        let config = FusionConfig::default();
        let fusion = DecisionFusion::new(config).await.unwrap();
        
        let decisions = vec![
            TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.8,
                amount: 100.0,
                reasoning: "Test 1".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
            TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.9,
                amount: 200.0,
                reasoning: "Test 2".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
        ];
        
        let result = fusion.fuse_decisions(decisions).await.unwrap();
        
        assert_eq!(result.decision.decision_type, DecisionType::Buy);
        assert!(result.confidence > 0.8);
        assert!(result.decision.amount > 100.0);
    }
    
    #[tokio::test]
    async fn test_consensus_calculation() {
        let config = FusionConfig::default();
        let fusion = DecisionFusion::new(config).await.unwrap();
        
        let decisions = vec![
            TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.8,
                amount: 100.0,
                reasoning: "Test 1".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
            TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.9,
                amount: 200.0,
                reasoning: "Test 2".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
            TradingDecision {
                decision_type: DecisionType::Sell,
                confidence: 0.7,
                amount: 150.0,
                reasoning: "Test 3".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
        ];
        
        let consensus = fusion.calculate_consensus(&decisions).await.unwrap();
        assert_eq!(consensus, 2.0 / 3.0); // 2 out of 3 decisions are Buy
    }
    
    #[tokio::test]
    async fn test_uncertainty_calculation() {
        let config = FusionConfig::default();
        let fusion = DecisionFusion::new(config).await.unwrap();
        
        let decisions = vec![
            TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.8,
                amount: 100.0,
                reasoning: "Test 1".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
            TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.9,
                amount: 200.0,
                reasoning: "Test 2".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
        ];
        
        let uncertainty = fusion.calculate_uncertainty(&decisions, 0.85).await.unwrap();
        
        assert!(uncertainty.lower_bound <= 0.85);
        assert!(uncertainty.upper_bound >= 0.85);
        assert!(uncertainty.variance >= 0.0);
        assert!(uncertainty.entropy >= 0.0);
    }
}
