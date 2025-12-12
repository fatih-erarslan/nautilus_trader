//! Inventory Optimization for Market Making and Trading Strategies
//!
//! This crate provides optimization algorithms for managing inventory
//! in trading strategies, particularly for market making.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use nalgebra::DVector;
use anyhow::Result;
use async_trait::async_trait;

/// Inventory position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryPosition {
    pub symbol: String,
    pub quantity: f64,
    pub average_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Inventory constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryConstraints {
    pub max_position: f64,
    pub max_concentration: f64,
    pub risk_limit: f64,
    pub correlation_limit: f64,
}

/// Inventory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryOptimizationConfig {
    pub risk_aversion: f64,
    pub time_horizon: chrono::Duration,
    pub rebalancing_frequency: chrono::Duration,
    pub transaction_cost: f64,
    pub slippage_model: SlippageModel,
}

/// Slippage model for transaction costs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlippageModel {
    Linear { coefficient: f64 },
    SquareRoot { coefficient: f64 },
    Exponential { coefficient: f64 },
}

/// Inventory optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryOptimizationResult {
    pub target_positions: HashMap<String, f64>,
    pub trades: Vec<Trade>,
    pub expected_return: f64,
    pub expected_risk: f64,
    pub sharpe_ratio: f64,
    pub optimization_time: chrono::Duration,
}

/// Trade recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub quantity: f64,
    pub side: TradeSide,
    pub urgency: TradeUrgency,
    pub expected_cost: f64,
}

/// Trade side
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Trade urgency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeUrgency {
    Low,
    Medium,
    High,
    Urgent,
}

/// Inventory optimizer trait
#[async_trait]
pub trait InventoryOptimizer: Send + Sync {
    async fn optimize(
        &mut self,
        positions: &[InventoryPosition],
        constraints: &InventoryConstraints,
        config: &InventoryOptimizationConfig,
    ) -> Result<InventoryOptimizationResult>;
    
    async fn update_model(&mut self, market_data: &MarketData) -> Result<()>;
    fn get_risk_metrics(&self) -> RiskMetrics;
}

/// Market data for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub prices: HashMap<String, f64>,
    pub volatilities: HashMap<String, f64>,
    pub correlations: HashMap<(String, String), f64>,
    pub volumes: HashMap<String, f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub value_at_risk: f64,
    pub expected_shortfall: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub beta: f64,
    pub sharpe_ratio: f64,
}

/// Mean-variance optimizer
pub struct MeanVarianceOptimizer {
    config: InventoryOptimizationConfig,
    expected_returns: HashMap<String, f64>,
    covariance_matrix: HashMap<(String, String), f64>,
    risk_metrics: RiskMetrics,
}

impl MeanVarianceOptimizer {
    pub fn new(config: InventoryOptimizationConfig) -> Self {
        Self {
            config,
            expected_returns: HashMap::new(),
            covariance_matrix: HashMap::new(),
            risk_metrics: RiskMetrics {
                value_at_risk: 0.0,
                expected_shortfall: 0.0,
                max_drawdown: 0.0,
                volatility: 0.0,
                beta: 0.0,
                sharpe_ratio: 0.0,
            },
        }
    }

    /// Calculate portfolio variance
    pub fn calculate_portfolio_variance(
        &self,
        positions: &HashMap<String, f64>,
    ) -> f64 {
        let mut variance = 0.0;
        
        for (symbol1, weight1) in positions {
            for (symbol2, weight2) in positions {
                if let Some(cov) = self.covariance_matrix.get(&(symbol1.clone(), symbol2.clone())) {
                    variance += weight1 * weight2 * cov;
                }
            }
        }
        
        variance
    }

    /// Calculate portfolio expected return
    pub fn calculate_portfolio_return(
        &self,
        positions: &HashMap<String, f64>,
    ) -> f64 {
        let mut expected_return = 0.0;
        
        for (symbol, weight) in positions {
            if let Some(return_rate) = self.expected_returns.get(symbol) {
                expected_return += weight * return_rate;
            }
        }
        
        expected_return
    }

    /// Calculate transaction costs
    pub fn calculate_transaction_costs(
        &self,
        trades: &[Trade],
        market_data: &MarketData,
    ) -> f64 {
        let mut total_cost = 0.0;
        
        for trade in trades {
            let volume = market_data.volumes.get(&trade.symbol).unwrap_or(&1.0);
            let price = market_data.prices.get(&trade.symbol).unwrap_or(&1.0);
            
            let slippage = match &self.config.slippage_model {
                SlippageModel::Linear { coefficient } => {
                    coefficient * trade.quantity.abs()
                }
                SlippageModel::SquareRoot { coefficient } => {
                    coefficient * trade.quantity.abs().sqrt()
                }
                SlippageModel::Exponential { coefficient } => {
                    coefficient * (trade.quantity.abs().exp() - 1.0)
                }
            };
            
            total_cost += trade.quantity.abs() * price * (self.config.transaction_cost + slippage);
        }
        
        total_cost
    }

    /// Optimize using quadratic programming
    #[cfg(feature = "linear-programming")]
    pub fn optimize_quadratic_programming(
        &self,
        positions: &[InventoryPosition],
        constraints: &InventoryConstraints,
    ) -> Result<InventoryOptimizationResult> {
        // Stub implementation for quadratic programming optimization
        let mut target_positions = HashMap::new();
        let mut trades = Vec::new();
        
        for position in positions {
            // Simple rebalancing logic
            let current_weight = position.quantity / constraints.max_position;
            let target_weight = current_weight.clamp(-0.1, 0.1); // Simple bounds
            
            target_positions.insert(position.symbol.clone(), target_weight);
            
            let trade_quantity = target_weight - current_weight;
            if trade_quantity.abs() > 0.001 {
                trades.push(Trade {
                    symbol: position.symbol.clone(),
                    quantity: trade_quantity,
                    side: if trade_quantity > 0.0 { TradeSide::Buy } else { TradeSide::Sell },
                    urgency: TradeUrgency::Medium,
                    expected_cost: trade_quantity.abs() * 0.001, // Simple cost model
                });
            }
        }
        
        Ok(InventoryOptimizationResult {
            target_positions,
            trades,
            expected_return: 0.001, // Placeholder
            expected_risk: 0.02,    // Placeholder
            sharpe_ratio: 0.05,     // Placeholder
            optimization_time: chrono::Duration::milliseconds(10),
        })
    }
}

#[async_trait]
impl InventoryOptimizer for MeanVarianceOptimizer {
    async fn optimize(
        &mut self,
        positions: &[InventoryPosition],
        constraints: &InventoryConstraints,
        config: &InventoryOptimizationConfig,
    ) -> Result<InventoryOptimizationResult> {
        self.config = config.clone();
        
        #[cfg(feature = "linear-programming")]
        {
            self.optimize_quadratic_programming(positions, constraints)
        }
        
        #[cfg(not(feature = "linear-programming"))]
        {
            // Fallback to simple optimization
            let mut target_positions = HashMap::new();
            let mut trades = Vec::new();
            
            for position in positions {
                // Simple equal weight allocation
                let target_weight = 1.0 / positions.len() as f64;
                target_positions.insert(position.symbol.clone(), target_weight);
                
                let trade_quantity = target_weight - position.quantity;
                if trade_quantity.abs() > 0.001 {
                    trades.push(Trade {
                        symbol: position.symbol.clone(),
                        quantity: trade_quantity,
                        side: if trade_quantity > 0.0 { TradeSide::Buy } else { TradeSide::Sell },
                        urgency: TradeUrgency::Medium,
                        expected_cost: trade_quantity.abs() * 0.001,
                    });
                }
            }
            
            Ok(InventoryOptimizationResult {
                target_positions,
                trades,
                expected_return: 0.001,
                expected_risk: 0.02,
                sharpe_ratio: 0.05,
                optimization_time: chrono::Duration::milliseconds(5),
            })
        }
    }
    
    async fn update_model(&mut self, market_data: &MarketData) -> Result<()> {
        // Update expected returns and covariance matrix
        self.expected_returns.clear();
        self.covariance_matrix.clear();
        
        for (symbol, price) in &market_data.prices {
            // Simple momentum-based expected return
            self.expected_returns.insert(symbol.clone(), price * 0.0001);
            
            // Simple volatility-based variance
            let volatility = market_data.volatilities.get(symbol).unwrap_or(&0.2);
            self.covariance_matrix.insert((symbol.clone(), symbol.clone()), volatility * volatility);
        }
        
        // Update correlations
        for ((symbol1, symbol2), correlation) in &market_data.correlations {
            let vol1 = market_data.volatilities.get(symbol1).unwrap_or(&0.2);
            let vol2 = market_data.volatilities.get(symbol2).unwrap_or(&0.2);
            let covariance = correlation * vol1 * vol2;
            
            self.covariance_matrix.insert((symbol1.clone(), symbol2.clone()), covariance);
            self.covariance_matrix.insert((symbol2.clone(), symbol1.clone()), covariance);
        }
        
        Ok(())
    }
    
    fn get_risk_metrics(&self) -> RiskMetrics {
        self.risk_metrics.clone()
    }
}

/// Utility functions for inventory optimization
pub mod utils {
    use super::*;

    /// Create a simple inventory position
    pub fn create_position(symbol: &str, quantity: f64, price: f64) -> InventoryPosition {
        InventoryPosition {
            symbol: symbol.to_string(),
            quantity,
            average_price: price,
            market_value: quantity * price,
            unrealized_pnl: 0.0,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Calculate portfolio value
    pub fn calculate_portfolio_value(positions: &[InventoryPosition]) -> f64 {
        positions.iter().map(|p| p.market_value).sum()
    }

    /// Calculate portfolio risk
    pub fn calculate_portfolio_risk(positions: &[InventoryPosition]) -> f64 {
        let total_value = calculate_portfolio_value(positions);
        if total_value == 0.0 {
            return 0.0;
        }
        
        let variance = positions.iter()
            .map(|p| (p.market_value / total_value).powi(2) * 0.04) // Assume 20% volatility
            .sum::<f64>();
        
        variance.sqrt()
    }

    /// Rebalance portfolio to target weights
    pub fn rebalance_portfolio(
        current_positions: &[InventoryPosition],
        target_weights: &HashMap<String, f64>,
        total_value: f64,
    ) -> Vec<Trade> {
        let mut trades = Vec::new();
        
        for position in current_positions {
            let current_weight = position.market_value / total_value;
            let target_weight = target_weights.get(&position.symbol).unwrap_or(&0.0);
            
            let weight_diff = target_weight - current_weight;
            if weight_diff.abs() > 0.001 {
                let trade_value = weight_diff * total_value;
                let trade_quantity = trade_value / position.average_price;
                
                trades.push(Trade {
                    symbol: position.symbol.clone(),
                    quantity: trade_quantity,
                    side: if trade_quantity > 0.0 { TradeSide::Buy } else { TradeSide::Sell },
                    urgency: TradeUrgency::Medium,
                    expected_cost: trade_quantity.abs() * 0.001,
                });
            }
        }
        
        trades
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_mean_variance_optimizer() {
        let config = InventoryOptimizationConfig {
            risk_aversion: 0.5,
            time_horizon: chrono::Duration::hours(1),
            rebalancing_frequency: chrono::Duration::minutes(5),
            transaction_cost: 0.001,
            slippage_model: SlippageModel::Linear { coefficient: 0.0001 },
        };
        
        let mut optimizer = MeanVarianceOptimizer::new(config);
        
        let positions = vec![
            utils::create_position("AAPL", 100.0, 150.0),
            utils::create_position("GOOGL", 50.0, 2500.0),
        ];
        
        let constraints = InventoryConstraints {
            max_position: 10000.0,
            max_concentration: 0.5,
            risk_limit: 0.1,
            correlation_limit: 0.8,
        };
        
        let config = InventoryOptimizationConfig {
            risk_aversion: 0.5,
            time_horizon: chrono::Duration::hours(1),
            rebalancing_frequency: chrono::Duration::minutes(5),
            transaction_cost: 0.001,
            slippage_model: SlippageModel::Linear { coefficient: 0.0001 },
        };
        
        let result = optimizer.optimize(&positions, &constraints, &config).await.unwrap();
        
        assert!(!result.target_positions.is_empty());
        assert!(result.expected_return >= 0.0);
        assert!(result.expected_risk >= 0.0);
    }

    #[test]
    fn test_portfolio_value_calculation() {
        let positions = vec![
            utils::create_position("AAPL", 100.0, 150.0),
            utils::create_position("GOOGL", 50.0, 2500.0),
        ];
        
        let total_value = utils::calculate_portfolio_value(&positions);
        assert_eq!(total_value, 15000.0 + 125000.0);
    }
}