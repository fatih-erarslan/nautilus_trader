//! # Logarithmic Market Scoring Rule (LMSR) Implementation
//!
//! Full LMSR implementation based on the sophisticated Python PADS system.
//! Provides market scoring, prediction aggregation, and decision fusion capabilities.

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Errors that can occur during LMSR operations
#[derive(thiserror::Error, Debug)]
pub enum LMSRError {
    #[error("Invalid market parameters: {message}")]
    InvalidParameters { message: String },
    
    #[error("Insufficient liquidity: {available} < {required}")]
    InsufficientLiquidity { available: f64, required: f64 },
    
    #[error("Market not found: {market_id}")]
    MarketNotFound { market_id: String },
    
    #[error("Invalid outcome: {outcome}")]
    InvalidOutcome { outcome: String },
    
    #[error("Calculation error: {message}")]
    CalculationError { message: String },
}

/// Result type for LMSR operations
pub type LMSRResult<T> = Result<T, LMSRError>;

/// LMSR configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRConfig {
    /// Liquidity parameter (b) - controls market sensitivity
    pub liquidity_parameter: f64,
    /// Minimum price threshold
    pub min_price: f64,
    /// Maximum price threshold
    pub max_price: f64,
    /// Transaction fee rate
    pub transaction_fee: f64,
    /// Maximum position size
    pub max_position_size: f64,
    /// Risk tolerance
    pub risk_tolerance: f64,
}

impl Default for LMSRConfig {
    fn default() -> Self {
        Self {
            liquidity_parameter: 100.0,
            min_price: 0.001,
            max_price: 0.999,
            transaction_fee: 0.001, // 0.1%
            max_position_size: 1000.0,
            risk_tolerance: 0.1,
        }
    }
}

/// Market outcome definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketOutcome {
    pub id: String,
    pub name: String,
    pub description: String,
    pub probability: f64,
    pub shares_outstanding: f64,
}

/// Market state for LMSR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub market_id: String,
    pub outcomes: HashMap<String, MarketOutcome>,
    pub total_volume: f64,
    pub last_updated: u64,
    pub market_status: MarketStatus,
}

/// Market status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketStatus {
    Active,
    Suspended,
    Resolved,
    Closed,
}

/// Trading position in LMSR market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRPosition {
    pub market_id: String,
    pub outcome_id: String,
    pub shares: f64,
    pub average_price: f64,
    pub unrealized_pnl: f64,
    pub created_at: u64,
}

/// LMSR transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRTransaction {
    pub transaction_id: String,
    pub market_id: String,
    pub outcome_id: String,
    pub transaction_type: TransactionType,
    pub shares: f64,
    pub price: f64,
    pub cost: f64,
    pub fee: f64,
    pub timestamp: u64,
}

/// Transaction type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    Buy,
    Sell,
    Redeem,
}

/// LMSR prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRPrediction {
    pub market_id: String,
    pub predicted_outcome: String,
    pub confidence: f64,
    pub probability_distribution: HashMap<String, f64>,
    pub expected_value: f64,
    pub risk_score: f64,
    pub recommendation: TradingAction,
}

/// Logarithmic Market Scoring Rule implementation
pub struct LMSR {
    config: LMSRConfig,
    markets: Arc<std::sync::RwLock<HashMap<String, MarketState>>>,
    positions: Arc<std::sync::RwLock<HashMap<String, Vec<LMSRPosition>>>>,
    transaction_history: Arc<std::sync::RwLock<Vec<LMSRTransaction>>>,
    performance_metrics: Arc<std::sync::RwLock<LMSRMetrics>>,
}

impl LMSR {
    /// Create new LMSR instance
    pub fn new(config: LMSRConfig) -> LMSRResult<Self> {
        if config.liquidity_parameter <= 0.0 {
            return Err(LMSRError::InvalidParameters {
                message: "Liquidity parameter must be positive".to_string()
            });
        }
        
        if config.min_price >= config.max_price {
            return Err(LMSRError::InvalidParameters {
                message: "Min price must be less than max price".to_string()
            });
        }
        
        Ok(Self {
            config,
            markets: Arc::new(std::sync::RwLock::new(HashMap::new())),
            positions: Arc::new(std::sync::RwLock::new(HashMap::new())),
            transaction_history: Arc::new(std::sync::RwLock::new(Vec::new())),
            performance_metrics: Arc::new(std::sync::RwLock::new(LMSRMetrics::default())),
        })
    }
    
    /// Create new market
    pub fn create_market(
        &self,
        market_id: String,
        outcomes: Vec<(String, String)>, // (id, name) pairs
    ) -> LMSRResult<()> {
        if outcomes.len() < 2 {
            return Err(LMSRError::InvalidParameters {
                message: "Market must have at least 2 outcomes".to_string()
            });
        }
        
        let mut market_outcomes = HashMap::new();
        let initial_probability = 1.0 / outcomes.len() as f64;
        
        for (outcome_id, outcome_name) in outcomes {
            market_outcomes.insert(outcome_id.clone(), MarketOutcome {
                id: outcome_id,
                name: outcome_name,
                description: "Market outcome".to_string(),
                probability: initial_probability,
                shares_outstanding: 0.0,
            });
        }
        
        let market_state = MarketState {
            market_id: market_id.clone(),
            outcomes: market_outcomes,
            total_volume: 0.0,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            market_status: MarketStatus::Active,
        };
        
        if let Ok(mut markets) = self.markets.write() {
            markets.insert(market_id, market_state);
        }
        
        Ok(())
    }
    
    /// Calculate cost of buying shares using LMSR formula
    pub fn calculate_cost(&self, market_id: &str, outcome_id: &str, shares: f64) -> LMSRResult<f64> {
        if shares <= 0.0 {
            return Err(LMSRError::InvalidParameters {
                message: "Shares must be positive".to_string()
            });
        }
        
        let markets = self.markets.read().map_err(|_| LMSRError::CalculationError {
            message: "Failed to read markets".to_string()
        })?;
        
        let market = markets.get(market_id).ok_or_else(|| LMSRError::MarketNotFound {
            market_id: market_id.to_string()
        })?;
        
        if !market.outcomes.contains_key(outcome_id) {
            return Err(LMSRError::InvalidOutcome {
                outcome: outcome_id.to_string()
            });
        }
        
        // LMSR cost function: C(q) = b * ln(Σ_i exp(q_i / b))
        // Cost of buying Δq shares: C(q + Δq) - C(q)
        
        let b = self.config.liquidity_parameter;
        
        // Current share quantities
        let current_shares: Vec<f64> = market.outcomes.values()
            .map(|outcome| outcome.shares_outstanding)
            .collect();
        
        // New share quantities after purchase
        let mut new_shares = current_shares.clone();
        if let Some(outcome_index) = market.outcomes.keys().position(|k| k == outcome_id) {
            new_shares[outcome_index] += shares;
        } else {
            return Err(LMSRError::InvalidOutcome {
                outcome: outcome_id.to_string()
            });
        }
        
        // Calculate cost function values
        let current_cost = self.calculate_cost_function(&current_shares, b);
        let new_cost = self.calculate_cost_function(&new_shares, b);
        
        let cost = new_cost - current_cost;
        let fee = cost * self.config.transaction_fee;
        
        Ok(cost + fee)
    }
    
    /// Calculate LMSR cost function
    fn calculate_cost_function(&self, shares: &[f64], b: f64) -> f64 {
        let sum_exp: f64 = shares.iter()
            .map(|&q| (q / b).exp())
            .sum();
        
        b * sum_exp.ln()
    }
    
    /// Calculate current price for an outcome
    pub fn calculate_price(&self, market_id: &str, outcome_id: &str) -> LMSRResult<f64> {
        let markets = self.markets.read().map_err(|_| LMSRError::CalculationError {
            message: "Failed to read markets".to_string()
        })?;
        
        let market = markets.get(market_id).ok_or_else(|| LMSRError::MarketNotFound {
            market_id: market_id.to_string()
        })?;
        
        if !market.outcomes.contains_key(outcome_id) {
            return Err(LMSRError::InvalidOutcome {
                outcome: outcome_id.to_string()
            });
        }
        
        let b = self.config.liquidity_parameter;
        let shares: Vec<f64> = market.outcomes.values()
            .map(|outcome| outcome.shares_outstanding)
            .collect();
        
        // Price = ∂C/∂q_i = exp(q_i / b) / Σ_j exp(q_j / b)
        let outcome_index = market.outcomes.keys().position(|k| k == outcome_id).unwrap();
        let outcome_shares = shares[outcome_index];
        
        let numerator = (outcome_shares / b).exp();
        let denominator: f64 = shares.iter().map(|&q| (q / b).exp()).sum();
        
        let price = numerator / denominator;
        Ok(price.clamp(self.config.min_price, self.config.max_price))
    }
    
    /// Buy shares in a market outcome
    pub fn buy_shares(
        &self,
        market_id: &str,
        outcome_id: &str,
        shares: f64,
    ) -> LMSRResult<LMSRTransaction> {
        if shares > self.config.max_position_size {
            return Err(LMSRError::InvalidParameters {
                message: format!("Position size {} exceeds maximum {}", shares, self.config.max_position_size)
            });
        }
        
        let cost = self.calculate_cost(market_id, outcome_id, shares)?;
        let price = cost / shares;
        
        // Update market state
        {
            let mut markets = self.markets.write().map_err(|_| LMSRError::CalculationError {
                message: "Failed to write markets".to_string()
            })?;
            
            if let Some(market) = markets.get_mut(market_id) {
                if let Some(outcome) = market.outcomes.get_mut(outcome_id) {
                    outcome.shares_outstanding += shares;
                    outcome.probability = self.calculate_price(market_id, outcome_id)?;
                }
                market.total_volume += cost;
                market.last_updated = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }
        }
        
        // Create transaction record
        let transaction = LMSRTransaction {
            transaction_id: uuid::Uuid::new_v4().to_string(),
            market_id: market_id.to_string(),
            outcome_id: outcome_id.to_string(),
            transaction_type: TransactionType::Buy,
            shares,
            price,
            cost,
            fee: cost * self.config.transaction_fee,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Record transaction
        if let Ok(mut history) = self.transaction_history.write() {
            history.push(transaction.clone());
        }
        
        // Update performance metrics
        if let Ok(mut metrics) = self.performance_metrics.write() {
            metrics.total_transactions += 1;
            metrics.total_volume += cost;
        }
        
        Ok(transaction)
    }
    
    /// Generate prediction based on current market state
    pub fn generate_prediction(&self, market_id: &str) -> LMSRResult<LMSRPrediction> {
        let markets = self.markets.read().map_err(|_| LMSRError::CalculationError {
            message: "Failed to read markets".to_string()
        })?;
        
        let market = markets.get(market_id).ok_or_else(|| LMSRError::MarketNotFound {
            market_id: market_id.to_string()
        })?;
        
        // Calculate probability distribution
        let mut probability_distribution = HashMap::new();
        let mut max_probability = 0.0;
        let mut predicted_outcome = String::new();
        
        for (outcome_id, outcome) in &market.outcomes {
            let probability = self.calculate_price(market_id, outcome_id)?;
            probability_distribution.insert(outcome_id.clone(), probability);
            
            if probability > max_probability {
                max_probability = probability;
                predicted_outcome = outcome_id.clone();
            }
        }
        
        // Calculate confidence based on probability separation
        let probabilities: Vec<f64> = probability_distribution.values().cloned().collect();
        let mean_prob = probabilities.iter().sum::<f64>() / probabilities.len() as f64;
        let variance: f64 = probabilities.iter()
            .map(|p| (p - mean_prob).powi(2))
            .sum::<f64>() / probabilities.len() as f64;
        let confidence = variance.sqrt() * 2.0; // Higher variance = higher confidence
        
        // Calculate expected value (simplified)
        let expected_value = max_probability;
        
        // Calculate risk score based on market volatility
        let risk_score = 1.0 - confidence;
        
        // Generate recommendation
        let recommendation = if max_probability > 0.6 && confidence > 0.3 {
            TradingAction::Buy
        } else if max_probability < 0.4 && confidence > 0.3 {
            TradingAction::Sell
        } else {
            TradingAction::Hold
        };
        
        Ok(LMSRPrediction {
            market_id: market_id.to_string(),
            predicted_outcome,
            confidence,
            probability_distribution,
            expected_value,
            risk_score,
            recommendation,
        })
    }
    
    /// Get market state
    pub fn get_market_state(&self, market_id: &str) -> LMSRResult<MarketState> {
        let markets = self.markets.read().map_err(|_| LMSRError::CalculationError {
            message: "Failed to read markets".to_string()
        })?;
        
        markets.get(market_id).cloned().ok_or_else(|| LMSRError::MarketNotFound {
            market_id: market_id.to_string()
        })
    }
    
    /// Get all markets
    pub fn get_all_markets(&self) -> HashMap<String, MarketState> {
        self.markets.read()
            .map(|markets| markets.clone())
            .unwrap_or_default()
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> LMSRMetrics {
        self.performance_metrics.read()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }
    
    /// Update market with external information
    pub fn update_market(
        &self,
        market_id: &str,
        external_probabilities: HashMap<String, f64>,
    ) -> LMSRResult<()> {
        let mut markets = self.markets.write().map_err(|_| LMSRError::CalculationError {
            message: "Failed to write markets".to_string()
        })?;
        
        if let Some(market) = markets.get_mut(market_id) {
            for (outcome_id, probability) in external_probabilities {
                if let Some(outcome) = market.outcomes.get_mut(&outcome_id) {
                    outcome.probability = probability.clamp(self.config.min_price, self.config.max_price);
                }
            }
            market.last_updated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
        
        Ok(())
    }
}

/// LMSR performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRMetrics {
    pub total_transactions: u64,
    pub total_volume: f64,
    pub total_markets: u64,
    pub average_prediction_accuracy: f64,
    pub average_spread: f64,
    pub liquidity_utilization: f64,
}

impl Default for LMSRMetrics {
    fn default() -> Self {
        Self {
            total_transactions: 0,
            total_volume: 0.0,
            total_markets: 0,
            average_prediction_accuracy: 0.0,
            average_spread: 0.0,
            liquidity_utilization: 0.0,
        }
    }
}

/// Factory function for creating LMSR instance
pub fn create_lmsr() -> LMSR {
    let config = LMSRConfig::default();
    LMSR::new(config).expect("Failed to create LMSR")
}

/// Factory function for creating trading-optimized LMSR
pub fn create_trading_lmsr() -> LMSR {
    let config = LMSRConfig {
        liquidity_parameter: 50.0, // Lower for more responsive pricing
        min_price: 0.01,
        max_price: 0.99,
        transaction_fee: 0.0005, // 0.05% for trading
        max_position_size: 10000.0,
        risk_tolerance: 0.05,
    };
    
    LMSR::new(config).expect("Failed to create trading LMSR")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lmsr_creation() {
        let config = LMSRConfig::default();
        let lmsr = LMSR::new(config);
        assert!(lmsr.is_ok());
    }
    
    #[test]
    fn test_market_creation() {
        let lmsr = create_lmsr();
        
        let outcomes = vec![
            ("bullish".to_string(), "Bullish outcome".to_string()),
            ("bearish".to_string(), "Bearish outcome".to_string()),
        ];
        
        let result = lmsr.create_market("test_market".to_string(), outcomes);
        assert!(result.is_ok());
        
        let market_state = lmsr.get_market_state("test_market");
        assert!(market_state.is_ok());
        
        let state = market_state.unwrap();
        assert_eq!(state.outcomes.len(), 2);
        assert!(state.outcomes.contains_key("bullish"));
        assert!(state.outcomes.contains_key("bearish"));
    }
    
    #[test]
    fn test_price_calculation() {
        let lmsr = create_lmsr();
        
        let outcomes = vec![
            ("up".to_string(), "Price goes up".to_string()),
            ("down".to_string(), "Price goes down".to_string()),
        ];
        
        lmsr.create_market("price_market".to_string(), outcomes).unwrap();
        
        let price_up = lmsr.calculate_price("price_market", "up");
        let price_down = lmsr.calculate_price("price_market", "down");
        
        assert!(price_up.is_ok());
        assert!(price_down.is_ok());
        
        let p_up = price_up.unwrap();
        let p_down = price_down.unwrap();
        
        // Prices should be approximately equal initially (0.5 each)
        assert!((p_up - 0.5).abs() < 0.1);
        assert!((p_down - 0.5).abs() < 0.1);
        assert!((p_up + p_down - 1.0).abs() < 0.01); // Should sum to ~1
    }
    
    #[test]
    fn test_cost_calculation() {
        let lmsr = create_lmsr();
        
        let outcomes = vec![
            ("yes".to_string(), "Yes".to_string()),
            ("no".to_string(), "No".to_string()),
        ];
        
        lmsr.create_market("binary_market".to_string(), outcomes).unwrap();
        
        let cost = lmsr.calculate_cost("binary_market", "yes", 10.0);
        assert!(cost.is_ok());
        
        let cost_value = cost.unwrap();
        assert!(cost_value > 0.0);
        assert!(cost_value < 100.0); // Should be reasonable for 10 shares
    }
    
    #[test]
    fn test_prediction_generation() {
        let lmsr = create_lmsr();
        
        let outcomes = vec![
            ("outcome_1".to_string(), "First outcome".to_string()),
            ("outcome_2".to_string(), "Second outcome".to_string()),
            ("outcome_3".to_string(), "Third outcome".to_string()),
        ];
        
        lmsr.create_market("prediction_market".to_string(), outcomes).unwrap();
        
        let prediction = lmsr.generate_prediction("prediction_market");
        assert!(prediction.is_ok());
        
        let pred = prediction.unwrap();
        assert_eq!(pred.market_id, "prediction_market");
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        assert_eq!(pred.probability_distribution.len(), 3);
        
        // Probabilities should sum to approximately 1
        let total_prob: f64 = pred.probability_distribution.values().sum();
        assert!((total_prob - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_buy_shares() {
        let lmsr = create_lmsr();
        
        let outcomes = vec![
            ("buy_outcome".to_string(), "Buy outcome".to_string()),
            ("sell_outcome".to_string(), "Sell outcome".to_string()),
        ];
        
        lmsr.create_market("trading_market".to_string(), outcomes).unwrap();
        
        let transaction = lmsr.buy_shares("trading_market", "buy_outcome", 5.0);
        assert!(transaction.is_ok());
        
        let tx = transaction.unwrap();
        assert_eq!(tx.market_id, "trading_market");
        assert_eq!(tx.outcome_id, "buy_outcome");
        assert_eq!(tx.shares, 5.0);
        assert!(tx.cost > 0.0);
        assert!(tx.fee >= 0.0);
        assert!(matches!(tx.transaction_type, TransactionType::Buy));
    }
    
    #[test]
    fn test_invalid_parameters() {
        let mut config = LMSRConfig::default();
        config.liquidity_parameter = -1.0;
        
        let lmsr = LMSR::new(config);
        assert!(lmsr.is_err());
        
        let config = LMSRConfig::default();
        let lmsr = LMSR::new(config).unwrap();
        
        // Test invalid market ID
        let price = lmsr.calculate_price("nonexistent_market", "outcome");
        assert!(price.is_err());
        
        // Test invalid outcome
        lmsr.create_market("test".to_string(), vec![("a".to_string(), "A".to_string())]).unwrap();
        let price = lmsr.calculate_price("test", "nonexistent_outcome");
        assert!(price.is_err());
    }
}