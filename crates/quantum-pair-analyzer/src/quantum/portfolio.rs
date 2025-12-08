//! Quantum Portfolio Optimizer
//!
//! This module implements quantum portfolio optimization algorithms
//! for optimal pair selection and portfolio construction.

use std::collections::HashMap;
use anyhow::{Result, Context};
use tracing::{info, debug, warn};
use serde::{Deserialize, Serialize};
use quantum_core::{QuantumResult, ComplexAmplitude};

use crate::{PairMetrics, OptimalPair, PairRecommendation, AnalyzerError};
use super::{QuantumConfig, QAOAResult};

/// Quantum portfolio optimizer
#[derive(Debug)]
pub struct QuantumPortfolioOptimizer {
    config: QuantumConfig,
    extraction_method: ExtractionMethod,
    selection_strategy: SelectionStrategy,
    ranking_algorithm: RankingAlgorithm,
}

/// Methods for extracting portfolio from quantum results
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExtractionMethod {
    /// Select most probable bitstrings
    MostProbable,
    /// Expectation value based selection
    ExpectationBased,
    /// Quantum annealing inspired
    QuantumAnnealing,
    /// Variational quantum eigensolver
    VQE,
    /// Sampling based extraction
    Sampling,
}

/// Portfolio selection strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Select top N pairs by probability
    TopN,
    /// Threshold-based selection
    Threshold,
    /// Proportional selection
    Proportional,
    /// Diversified selection
    Diversified,
    /// Risk-adjusted selection
    RiskAdjusted,
}

/// Ranking algorithms for portfolio construction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RankingAlgorithm {
    /// Quantum expectation value
    QuantumExpectation,
    /// Classical scoring
    ClassicalScore,
    /// Hybrid quantum-classical
    Hybrid,
    /// Sharpe ratio maximization
    SharpeRatio,
    /// Risk parity
    RiskParity,
}

/// Portfolio extraction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioExtractionResult {
    /// Selected pairs
    pub selected_pairs: Vec<OptimalPair>,
    /// Extraction confidence
    pub confidence: f64,
    /// Quantum advantage score
    pub quantum_advantage: f64,
    /// Portfolio metrics
    pub portfolio_metrics: PortfolioMetrics,
    /// Extraction metadata
    pub extraction_metadata: ExtractionMetadata,
}

/// Portfolio performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    /// Expected return
    pub expected_return: f64,
    /// Portfolio volatility
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Value at risk
    pub value_at_risk: f64,
    /// Diversification ratio
    pub diversification_ratio: f64,
    /// Correlation matrix
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Individual pair weights
    pub pair_weights: HashMap<String, f64>,
}

/// Extraction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionMetadata {
    /// Extraction method used
    pub method: ExtractionMethod,
    /// Selection strategy used
    pub strategy: SelectionStrategy,
    /// Ranking algorithm used
    pub ranking: RankingAlgorithm,
    /// Number of quantum measurements
    pub num_measurements: usize,
    /// Quantum state fidelity
    pub fidelity: f64,
    /// Extraction time
    pub extraction_time_ms: u64,
    /// Success probability
    pub success_probability: f64,
}

impl QuantumPortfolioOptimizer {
    /// Create a new quantum portfolio optimizer
    pub async fn new(config: QuantumConfig) -> Result<Self, AnalyzerError> {
        info!("Initializing quantum portfolio optimizer");
        
        Ok(Self {
            config,
            extraction_method: ExtractionMethod::MostProbable,
            selection_strategy: SelectionStrategy::TopN,
            ranking_algorithm: RankingAlgorithm::QuantumExpectation,
        })
    }
    
    /// Extract portfolio from QAOA result
    pub async fn extract_portfolio(
        &self,
        qaoa_result: &QAOAResult,
        pair_metrics: &[PairMetrics],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let start_time = std::time::Instant::now();
        debug!("Extracting portfolio from QAOA result with {} pairs", pair_metrics.len());
        
        // Extract optimal pairs using selected method
        let optimal_pairs = match self.extraction_method {
            ExtractionMethod::MostProbable => {
                self.extract_most_probable(qaoa_result, pair_metrics).await?
            }
            ExtractionMethod::ExpectationBased => {
                self.extract_expectation_based(qaoa_result, pair_metrics).await?
            }
            ExtractionMethod::QuantumAnnealing => {
                self.extract_quantum_annealing(qaoa_result, pair_metrics).await?
            }
            ExtractionMethod::VQE => {
                self.extract_vqe_based(qaoa_result, pair_metrics).await?
            }
            ExtractionMethod::Sampling => {
                self.extract_sampling_based(qaoa_result, pair_metrics).await?
            }
        };
        
        // Apply selection strategy
        let selected_pairs = self.apply_selection_strategy(&optimal_pairs).await?;
        
        // Rank and finalize portfolio
        let final_portfolio = self.rank_and_finalize(&selected_pairs, pair_metrics).await?;
        
        let duration = start_time.elapsed();
        info!("Portfolio extraction completed in {:?} with {} pairs", 
              duration, final_portfolio.len());
        
        Ok(final_portfolio)
    }
    
    /// Extract most probable bitstrings
    async fn extract_most_probable(
        &self,
        qaoa_result: &QAOAResult,
        pair_metrics: &[PairMetrics],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut probability_pairs: Vec<(usize, f64)> = qaoa_result.probabilities
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        
        // Sort by probability (descending)
        probability_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut optimal_pairs = Vec::new();
        let num_pairs = pair_metrics.len().min(qaoa_result.probabilities.len());
        
        // Take top bitstrings and extract selected pairs
        for (bitstring, probability) in probability_pairs.iter().take(10) {
            let selected_indices = self.bitstring_to_indices(*bitstring, num_pairs);
            
            for &idx in &selected_indices {
                if idx < pair_metrics.len() {
                    let pair_metric = &pair_metrics[idx];
                    let mut optimal_pair = OptimalPair::from_metrics(pair_metric.clone());
                    
                    // Adjust score based on quantum probability
                    optimal_pair.score = optimal_pair.score * probability;
                    optimal_pair.confidence = optimal_pair.confidence * probability;
                    
                    optimal_pairs.push(optimal_pair);
                }
            }
        }
        
        Ok(optimal_pairs)
    }
    
    /// Extract using expectation values
    async fn extract_expectation_based(
        &self,
        qaoa_result: &QAOAResult,
        pair_metrics: &[PairMetrics],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let num_pairs = pair_metrics.len().min(qaoa_result.probabilities.len());
        let mut optimal_pairs = Vec::new();
        
        // Calculate expectation value for each pair
        for (i, pair_metric) in pair_metrics.iter().enumerate().take(num_pairs) {
            let mut expectation = 0.0;
            
            // Sum over all bitstrings where this pair is selected
            for (bitstring, &probability) in qaoa_result.probabilities.iter().enumerate() {
                if (bitstring >> i) & 1 == 1 {
                    expectation += probability;
                }
            }
            
            // Create optimal pair with expectation-based score
            let mut optimal_pair = OptimalPair::from_metrics(pair_metric.clone());
            optimal_pair.score = expectation * optimal_pair.score;
            optimal_pair.confidence = expectation;
            
            optimal_pairs.push(optimal_pair);
        }
        
        Ok(optimal_pairs)
    }
    
    /// Extract using quantum annealing approach
    async fn extract_quantum_annealing(
        &self,
        qaoa_result: &QAOAResult,
        pair_metrics: &[PairMetrics],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        // Use temperature-based selection
        let temperature = 0.1;
        let mut optimal_pairs = Vec::new();
        
        // Calculate Boltzmann weights
        let mut weights = Vec::new();
        for (bitstring, &probability) in qaoa_result.probabilities.iter().enumerate() {
            if probability > 1e-10 {
                let energy = -probability.ln(); // Convert probability to energy
                let weight = (-energy / temperature).exp();
                weights.push((bitstring, weight));
            }
        }
        
        // Normalize weights
        let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
        if total_weight > 0.0 {
            for (_, weight) in weights.iter_mut() {
                *weight /= total_weight;
            }
        }
        
        // Sample from distribution
        let mut rng = rand::thread_rng();
        use rand::Rng;
        
        for _ in 0..10 {
            let rand_val: f64 = rng.gen();
            let mut cumulative = 0.0;
            
            for &(bitstring, weight) in &weights {
                cumulative += weight;
                if rand_val <= cumulative {
                    let selected_indices = self.bitstring_to_indices(bitstring, pair_metrics.len());
                    
                    for &idx in &selected_indices {
                        if idx < pair_metrics.len() {
                            let pair_metric = &pair_metrics[idx];
                            let mut optimal_pair = OptimalPair::from_metrics(pair_metric.clone());
                            optimal_pair.score = optimal_pair.score * weight;
                            optimal_pair.confidence = weight;
                            
                            optimal_pairs.push(optimal_pair);
                        }
                    }
                    break;
                }
            }
        }
        
        Ok(optimal_pairs)
    }
    
    /// Extract using VQE approach
    async fn extract_vqe_based(
        &self,
        qaoa_result: &QAOAResult,
        pair_metrics: &[PairMetrics],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        // Use ground state approximation
        let ground_state_idx = qaoa_result.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let selected_indices = self.bitstring_to_indices(ground_state_idx, pair_metrics.len());
        let mut optimal_pairs = Vec::new();
        
        for &idx in &selected_indices {
            if idx < pair_metrics.len() {
                let pair_metric = &pair_metrics[idx];
                let mut optimal_pair = OptimalPair::from_metrics(pair_metric.clone());
                
                // Boost score for ground state selection
                optimal_pair.score *= 1.2;
                optimal_pair.confidence = qaoa_result.probabilities[ground_state_idx];
                
                optimal_pairs.push(optimal_pair);
            }
        }
        
        Ok(optimal_pairs)
    }
    
    /// Extract using sampling approach
    async fn extract_sampling_based(
        &self,
        qaoa_result: &QAOAResult,
        pair_metrics: &[PairMetrics],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let num_samples = 100;
        let mut pair_counts = HashMap::new();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        
        // Sample from quantum distribution
        for _ in 0..num_samples {
            let rand_val: f64 = rng.gen();
            let mut cumulative = 0.0;
            
            for (bitstring, &probability) in qaoa_result.probabilities.iter().enumerate() {
                cumulative += probability;
                if rand_val <= cumulative {
                    let selected_indices = self.bitstring_to_indices(bitstring, pair_metrics.len());
                    
                    for &idx in &selected_indices {
                        *pair_counts.entry(idx).or_insert(0) += 1;
                    }
                    break;
                }
            }
        }
        
        // Convert counts to optimal pairs
        let mut optimal_pairs = Vec::new();
        for (&idx, &count) in &pair_counts {
            if idx < pair_metrics.len() {
                let pair_metric = &pair_metrics[idx];
                let mut optimal_pair = OptimalPair::from_metrics(pair_metric.clone());
                
                let frequency = count as f64 / num_samples as f64;
                optimal_pair.score = optimal_pair.score * frequency;
                optimal_pair.confidence = frequency;
                
                optimal_pairs.push(optimal_pair);
            }
        }
        
        Ok(optimal_pairs)
    }
    
    /// Apply selection strategy
    async fn apply_selection_strategy(
        &self,
        optimal_pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        match self.selection_strategy {
            SelectionStrategy::TopN => {
                self.select_top_n(optimal_pairs).await
            }
            SelectionStrategy::Threshold => {
                self.select_threshold(optimal_pairs).await
            }
            SelectionStrategy::Proportional => {
                self.select_proportional(optimal_pairs).await
            }
            SelectionStrategy::Diversified => {
                self.select_diversified(optimal_pairs).await
            }
            SelectionStrategy::RiskAdjusted => {
                self.select_risk_adjusted(optimal_pairs).await
            }
        }
    }
    
    /// Select top N pairs
    async fn select_top_n(
        &self,
        optimal_pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut pairs = optimal_pairs.to_vec();
        pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = self.config.max_qubits.min(10);
        pairs.truncate(n);
        
        Ok(pairs)
    }
    
    /// Select by threshold
    async fn select_threshold(
        &self,
        optimal_pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let threshold = 0.5;
        let selected: Vec<OptimalPair> = optimal_pairs
            .iter()
            .filter(|pair| pair.score >= threshold)
            .cloned()
            .collect();
        
        Ok(selected)
    }
    
    /// Select proportionally
    async fn select_proportional(
        &self,
        optimal_pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let total_score: f64 = optimal_pairs.iter().map(|p| p.score).sum();
        let mut selected = Vec::new();
        
        if total_score > 0.0 {
            let mut rng = rand::thread_rng();
            use rand::Rng;
            
            for _ in 0..self.config.max_qubits.min(10) {
                let rand_val: f64 = rng.gen_range(0.0..total_score);
                let mut cumulative = 0.0;
                
                for pair in optimal_pairs {
                    cumulative += pair.score;
                    if rand_val <= cumulative {
                        selected.push(pair.clone());
                        break;
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    /// Select for diversification
    async fn select_diversified(
        &self,
        optimal_pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut selected = Vec::new();
        let mut used_exchanges = std::collections::HashSet::new();
        
        // Sort by score
        let mut pairs = optimal_pairs.to_vec();
        pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select diverse pairs
        for pair in pairs {
            let exchange = &pair.pair_id.exchange;
            if !used_exchanges.contains(exchange) || used_exchanges.len() < 3 {
                selected.push(pair);
                used_exchanges.insert(exchange.clone());
                
                if selected.len() >= self.config.max_qubits.min(10) {
                    break;
                }
            }
        }
        
        Ok(selected)
    }
    
    /// Select with risk adjustment
    async fn select_risk_adjusted(
        &self,
        optimal_pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut risk_adjusted: Vec<OptimalPair> = optimal_pairs
            .iter()
            .map(|pair| {
                let mut adjusted_pair = pair.clone();
                // Adjust score by risk
                adjusted_pair.score = adjusted_pair.score / (1.0 + adjusted_pair.risk_score);
                adjusted_pair
            })
            .collect();
        
        risk_adjusted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        risk_adjusted.truncate(self.config.max_qubits.min(10));
        
        Ok(risk_adjusted)
    }
    
    /// Rank and finalize portfolio
    async fn rank_and_finalize(
        &self,
        selected_pairs: &[OptimalPair],
        pair_metrics: &[PairMetrics],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut final_pairs = match self.ranking_algorithm {
            RankingAlgorithm::QuantumExpectation => {
                self.rank_by_quantum_expectation(selected_pairs).await?
            }
            RankingAlgorithm::ClassicalScore => {
                self.rank_by_classical_score(selected_pairs).await?
            }
            RankingAlgorithm::Hybrid => {
                self.rank_by_hybrid(selected_pairs).await?
            }
            RankingAlgorithm::SharpeRatio => {
                self.rank_by_sharpe_ratio(selected_pairs).await?
            }
            RankingAlgorithm::RiskParity => {
                self.rank_by_risk_parity(selected_pairs).await?
            }
        };
        
        // Add recommendations
        for pair in &mut final_pairs {
            pair.recommendation = self.generate_recommendation(pair);
        }
        
        Ok(final_pairs)
    }
    
    /// Rank by quantum expectation
    async fn rank_by_quantum_expectation(
        &self,
        pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut ranked_pairs = pairs.to_vec();
        ranked_pairs.sort_by(|a, b| {
            let score_a = a.score * a.confidence;
            let score_b = b.score * b.confidence;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(ranked_pairs)
    }
    
    /// Rank by classical score
    async fn rank_by_classical_score(
        &self,
        pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut ranked_pairs = pairs.to_vec();
        ranked_pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(ranked_pairs)
    }
    
    /// Rank by hybrid approach
    async fn rank_by_hybrid(
        &self,
        pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut ranked_pairs = pairs.to_vec();
        ranked_pairs.sort_by(|a, b| {
            let hybrid_score_a = 0.6 * a.score + 0.4 * a.confidence;
            let hybrid_score_b = 0.6 * b.score + 0.4 * b.confidence;
            hybrid_score_b.partial_cmp(&hybrid_score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(ranked_pairs)
    }
    
    /// Rank by Sharpe ratio
    async fn rank_by_sharpe_ratio(
        &self,
        pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut ranked_pairs = pairs.to_vec();
        ranked_pairs.sort_by(|a, b| {
            let sharpe_a = a.expected_return / (a.risk_score + 1e-10);
            let sharpe_b = b.expected_return / (b.risk_score + 1e-10);
            sharpe_b.partial_cmp(&sharpe_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(ranked_pairs)
    }
    
    /// Rank by risk parity
    async fn rank_by_risk_parity(
        &self,
        pairs: &[OptimalPair],
    ) -> Result<Vec<OptimalPair>, AnalyzerError> {
        let mut ranked_pairs = pairs.to_vec();
        
        // Calculate risk contributions
        let total_risk: f64 = pairs.iter().map(|p| p.risk_score).sum();
        if total_risk > 0.0 {
            for pair in &mut ranked_pairs {
                let risk_contribution = pair.risk_score / total_risk;
                pair.score = pair.score * (1.0 - risk_contribution);
            }
        }
        
        ranked_pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(ranked_pairs)
    }
    
    /// Generate recommendation for a pair
    fn generate_recommendation(&self, pair: &OptimalPair) -> PairRecommendation {
        let combined_score = pair.score * pair.confidence;
        
        if combined_score > 0.8 {
            PairRecommendation::StrongBuy
        } else if combined_score > 0.6 {
            PairRecommendation::Buy
        } else if combined_score > 0.4 {
            PairRecommendation::Hold
        } else {
            PairRecommendation::Avoid
        }
    }
    
    /// Convert bitstring to selected indices
    fn bitstring_to_indices(&self, bitstring: usize, num_pairs: usize) -> Vec<usize> {
        let mut indices = Vec::new();
        
        for i in 0..num_pairs {
            if (bitstring >> i) & 1 == 1 {
                indices.push(i);
            }
        }
        
        indices
    }
    
    /// Calculate portfolio metrics
    pub async fn calculate_portfolio_metrics(
        &self,
        portfolio: &[OptimalPair],
    ) -> Result<PortfolioMetrics, AnalyzerError> {
        if portfolio.is_empty() {
            return Err(AnalyzerError::InvalidInput("Empty portfolio".to_string()));
        }
        
        let n = portfolio.len();
        let mut correlation_matrix = vec![vec![0.0; n]; n];
        let mut pair_weights = HashMap::new();
        
        // Calculate equal weights for simplicity
        let weight = 1.0 / n as f64;
        
        // Calculate metrics
        let expected_return: f64 = portfolio.iter()
            .map(|p| p.expected_return * weight)
            .sum();
        
        let volatility: f64 = portfolio.iter()
            .map(|p| p.risk_score * weight)
            .sum();
        
        let sharpe_ratio = if volatility > 0.0 {
            expected_return / volatility
        } else {
            0.0
        };
        
        let max_drawdown: f64 = portfolio.iter()
            .map(|p| p.risk_score)
            .fold(0.0, f64::max);
        
        let value_at_risk: f64 = portfolio.iter()
            .map(|p| p.risk_score * weight)
            .sum();
        
        let diversification_ratio = if n > 1 { 1.0 / (n as f64).sqrt() } else { 0.0 };
        
        // Build correlation matrix (simplified)
        for i in 0..n {
            for j in 0..n {
                correlation_matrix[i][j] = if i == j { 1.0 } else { 0.1 };
            }
        }
        
        // Build pair weights
        for (i, pair) in portfolio.iter().enumerate() {
            pair_weights.insert(pair.pair_id.symbol(), weight);
        }
        
        Ok(PortfolioMetrics {
            expected_return,
            volatility,
            sharpe_ratio,
            max_drawdown,
            value_at_risk,
            diversification_ratio,
            correlation_matrix,
            pair_weights,
        })
    }
    
    /// Set extraction method
    pub fn set_extraction_method(&mut self, method: ExtractionMethod) {
        self.extraction_method = method;
    }
    
    /// Set selection strategy
    pub fn set_selection_strategy(&mut self, strategy: SelectionStrategy) {
        self.selection_strategy = strategy;
    }
    
    /// Set ranking algorithm
    pub fn set_ranking_algorithm(&mut self, algorithm: RankingAlgorithm) {
        self.ranking_algorithm = algorithm;
    }
    
    /// Get current configuration
    pub fn get_configuration(&self) -> (ExtractionMethod, SelectionStrategy, RankingAlgorithm) {
        (self.extraction_method, self.selection_strategy, self.ranking_algorithm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PairId;
    use chrono::Utc;
    use super::super::qaoa::QAOAParameters;
    
    #[tokio::test]
    async fn test_portfolio_optimizer_creation() {
        let config = QuantumConfig::default();
        let optimizer = QuantumPortfolioOptimizer::new(config).await;
        assert!(optimizer.is_ok());
    }
    
    #[tokio::test]
    async fn test_portfolio_extraction() {
        let config = QuantumConfig::default();
        let optimizer = QuantumPortfolioOptimizer::new(config).await.unwrap();
        
        let qaoa_result = create_test_qaoa_result();
        let pair_metrics = create_test_pair_metrics();
        
        let portfolio = optimizer.extract_portfolio(&qaoa_result, &pair_metrics).await;
        assert!(portfolio.is_ok());
        
        let portfolio = portfolio.unwrap();
        assert!(portfolio.len() > 0);
    }
    
    #[tokio::test]
    async fn test_most_probable_extraction() {
        let config = QuantumConfig::default();
        let optimizer = QuantumPortfolioOptimizer::new(config).await.unwrap();
        
        let qaoa_result = create_test_qaoa_result();
        let pair_metrics = create_test_pair_metrics();
        
        let optimal_pairs = optimizer.extract_most_probable(&qaoa_result, &pair_metrics).await;
        assert!(optimal_pairs.is_ok());
        
        let optimal_pairs = optimal_pairs.unwrap();
        assert!(optimal_pairs.len() > 0);
    }
    
    #[tokio::test]
    async fn test_expectation_based_extraction() {
        let config = QuantumConfig::default();
        let optimizer = QuantumPortfolioOptimizer::new(config).await.unwrap();
        
        let qaoa_result = create_test_qaoa_result();
        let pair_metrics = create_test_pair_metrics();
        
        let optimal_pairs = optimizer.extract_expectation_based(&qaoa_result, &pair_metrics).await;
        assert!(optimal_pairs.is_ok());
        
        let optimal_pairs = optimal_pairs.unwrap();
        assert_eq!(optimal_pairs.len(), pair_metrics.len());
    }
    
    #[tokio::test]
    async fn test_selection_strategies() {
        let config = QuantumConfig::default();
        let mut optimizer = QuantumPortfolioOptimizer::new(config).await.unwrap();
        
        let optimal_pairs = create_test_optimal_pairs();
        
        // Test TopN strategy
        optimizer.set_selection_strategy(SelectionStrategy::TopN);
        let selected = optimizer.apply_selection_strategy(&optimal_pairs).await.unwrap();
        assert!(selected.len() <= 10);
        
        // Test Threshold strategy
        optimizer.set_selection_strategy(SelectionStrategy::Threshold);
        let selected = optimizer.apply_selection_strategy(&optimal_pairs).await.unwrap();
        assert!(selected.iter().all(|p| p.score >= 0.5));
        
        // Test Diversified strategy
        optimizer.set_selection_strategy(SelectionStrategy::Diversified);
        let selected = optimizer.apply_selection_strategy(&optimal_pairs).await.unwrap();
        assert!(selected.len() > 0);
    }
    
    #[tokio::test]
    async fn test_ranking_algorithms() {
        let config = QuantumConfig::default();
        let mut optimizer = QuantumPortfolioOptimizer::new(config).await.unwrap();
        
        let selected_pairs = create_test_optimal_pairs();
        let pair_metrics = create_test_pair_metrics();
        
        // Test QuantumExpectation ranking
        optimizer.set_ranking_algorithm(RankingAlgorithm::QuantumExpectation);
        let ranked = optimizer.rank_and_finalize(&selected_pairs, &pair_metrics).await.unwrap();
        assert!(ranked.len() > 0);
        
        // Test ClassicalScore ranking
        optimizer.set_ranking_algorithm(RankingAlgorithm::ClassicalScore);
        let ranked = optimizer.rank_and_finalize(&selected_pairs, &pair_metrics).await.unwrap();
        assert!(ranked.len() > 0);
        
        // Test Hybrid ranking
        optimizer.set_ranking_algorithm(RankingAlgorithm::Hybrid);
        let ranked = optimizer.rank_and_finalize(&selected_pairs, &pair_metrics).await.unwrap();
        assert!(ranked.len() > 0);
    }
    
    #[tokio::test]
    async fn test_portfolio_metrics_calculation() {
        let config = QuantumConfig::default();
        let optimizer = QuantumPortfolioOptimizer::new(config).await.unwrap();
        
        let portfolio = create_test_optimal_pairs();
        let metrics = optimizer.calculate_portfolio_metrics(&portfolio).await;
        assert!(metrics.is_ok());
        
        let metrics = metrics.unwrap();
        assert!(metrics.expected_return > 0.0);
        assert!(metrics.volatility >= 0.0);
        assert!(metrics.diversification_ratio >= 0.0);
        assert_eq!(metrics.correlation_matrix.len(), portfolio.len());
    }
    
    #[test]
    fn test_bitstring_to_indices() {
        let config = QuantumConfig::default();
        let optimizer = QuantumPortfolioOptimizer::new(config);
        
        // Test with futures - we'll need to block on the future
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let optimizer = optimizer.await.unwrap();
            
            // Test bitstring 5 (binary: 101) -> indices [0, 2]
            let indices = optimizer.bitstring_to_indices(5, 4);
            assert_eq!(indices, vec![0, 2]);
            
            // Test bitstring 7 (binary: 111) -> indices [0, 1, 2]
            let indices = optimizer.bitstring_to_indices(7, 4);
            assert_eq!(indices, vec![0, 1, 2]);
            
            // Test bitstring 0 (binary: 000) -> indices []
            let indices = optimizer.bitstring_to_indices(0, 4);
            assert!(indices.is_empty());
        });
    }
    
    #[test]
    fn test_recommendation_generation() {
        let config = QuantumConfig::default();
        
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let optimizer = QuantumPortfolioOptimizer::new(config).await.unwrap();
            
            // High score pair
            let high_score_pair = OptimalPair {
                pair_id: PairId::new("BTC", "USD", "binance"),
                score: 0.9,
                confidence: 0.95,
                expected_return: 0.15,
                risk_score: 0.05,
                recommendation: PairRecommendation::Hold, // Will be overwritten
                metrics: None,
            };
            
            let recommendation = optimizer.generate_recommendation(&high_score_pair);
            assert_eq!(recommendation, PairRecommendation::StrongBuy);
            
            // Low score pair
            let low_score_pair = OptimalPair {
                pair_id: PairId::new("ETH", "USD", "binance"),
                score: 0.3,
                confidence: 0.4,
                expected_return: 0.05,
                risk_score: 0.15,
                recommendation: PairRecommendation::Hold, // Will be overwritten
                metrics: None,
            };
            
            let recommendation = optimizer.generate_recommendation(&low_score_pair);
            assert_eq!(recommendation, PairRecommendation::Avoid);
        });
    }
    
    fn create_test_qaoa_result() -> QAOAResult {
        QAOAResult {
            optimal_parameters: QAOAParameters {
                beta: vec![0.5, 0.3],
                gamma: vec![1.0, 0.8],
                layers: 2,
                bounds: super::super::qaoa::ParameterBounds::default(),
                step_size: 0.1,
                tolerance: 1e-6,
            },
            objective_value: 0.85,
            iterations: 50,
            converged: true,
            final_state: vec![
                ComplexAmplitude::new(0.5, 0.0),
                ComplexAmplitude::new(0.5, 0.0),
                ComplexAmplitude::new(0.5, 0.0),
                ComplexAmplitude::new(0.5, 0.0),
            ],
            probabilities: vec![0.4, 0.3, 0.2, 0.1],
            optimization_trace: vec![],
            circuit_stats: super::super::qaoa::CircuitStatistics {
                total_gates: 20,
                depth: 8,
                gate_counts: std::collections::HashMap::new(),
                estimated_execution_time_ns: 10000,
                memory_usage_bytes: 1024,
            },
        }
    }
    
    fn create_test_pair_metrics() -> Vec<PairMetrics> {
        vec![
            PairMetrics {
                pair_id: PairId::new("BTC", "USD", "binance"),
                timestamp: Utc::now(),
                correlation_score: 0.5,
                cointegration_p_value: 0.01,
                volatility_ratio: 0.3,
                liquidity_ratio: 0.8,
                sentiment_divergence: 0.2,
                news_sentiment_score: 0.6,
                social_sentiment_score: 0.7,
                cuckoo_score: 0.0,
                firefly_score: 0.0,
                ant_colony_score: 0.0,
                quantum_entanglement: 0.0,
                quantum_advantage: 0.5,
                expected_return: 0.15,
                sharpe_ratio: 1.2,
                maximum_drawdown: 0.1,
                value_at_risk: 0.05,
                composite_score: 0.8,
                confidence: 0.9,
            },
            PairMetrics {
                pair_id: PairId::new("ETH", "USD", "binance"),
                timestamp: Utc::now(),
                correlation_score: -0.3,
                cointegration_p_value: 0.02,
                volatility_ratio: 0.4,
                liquidity_ratio: 0.7,
                sentiment_divergence: 0.1,
                news_sentiment_score: 0.5,
                social_sentiment_score: 0.6,
                cuckoo_score: 0.0,
                firefly_score: 0.0,
                ant_colony_score: 0.0,
                quantum_entanglement: 0.0,
                quantum_advantage: 0.3,
                expected_return: 0.12,
                sharpe_ratio: 1.0,
                maximum_drawdown: 0.15,
                value_at_risk: 0.07,
                composite_score: 0.7,
                confidence: 0.8,
            },
        ]
    }
    
    fn create_test_optimal_pairs() -> Vec<OptimalPair> {
        vec![
            OptimalPair {
                pair_id: PairId::new("BTC", "USD", "binance"),
                score: 0.9,
                confidence: 0.95,
                expected_return: 0.15,
                risk_score: 0.05,
                recommendation: PairRecommendation::StrongBuy,
                metrics: None,
            },
            OptimalPair {
                pair_id: PairId::new("ETH", "USD", "binance"),
                score: 0.7,
                confidence: 0.8,
                expected_return: 0.12,
                risk_score: 0.07,
                recommendation: PairRecommendation::Buy,
                metrics: None,
            },
            OptimalPair {
                pair_id: PairId::new("ADA", "USD", "binance"),
                score: 0.4,
                confidence: 0.6,
                expected_return: 0.08,
                risk_score: 0.12,
                recommendation: PairRecommendation::Hold,
                metrics: None,
            },
        ]
    }
}