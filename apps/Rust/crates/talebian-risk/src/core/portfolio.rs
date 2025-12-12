//! Portfolio management for Talebian risk strategies

use crate::core::{TalebianRiskComponent, AssetType, RiskConfig};
use crate::error::{TalebianResult as Result, TalebianError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Antifragile portfolio implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilePortfolio {
    /// Portfolio identifier
    id: String,
    /// Portfolio configuration
    config: RiskConfig,
    /// Asset positions
    positions: HashMap<String, Position>,
    /// Portfolio value
    total_value: f64,
    /// Creation timestamp
    created_at: DateTime<Utc>,
    /// Last update timestamp
    updated_at: DateTime<Utc>,
}

/// Individual position in the portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Asset identifier
    pub asset: String,
    /// Asset type
    pub asset_type: AssetType,
    /// Position weight (0.0 to 1.0)
    pub weight: f64,
    /// Position value
    pub value: f64,
    /// Number of shares/units
    pub quantity: f64,
    /// Average entry price
    pub entry_price: f64,
    /// Current price
    pub current_price: f64,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
    /// Position timestamp
    pub timestamp: DateTime<Utc>,
}

impl AntifragilePortfolio {
    /// Create a new antifragile portfolio
    pub fn new(id: impl Into<String>, config: RiskConfig) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            config,
            positions: HashMap::new(),
            total_value: 0.0,
            created_at: now,
            updated_at: now,
        }
    }
    
    /// Add an asset to the portfolio
    pub fn add_asset(&mut self, asset: impl Into<String>, weight: f64, asset_type: AssetType) -> Result<()> {
        let asset = asset.into();
        
        if weight < 0.0 || weight > 1.0 {
            return Err(TalebianError::invalid_parameter(
                "weight",
                "Weight must be between 0.0 and 1.0"
            ));
        }
        
        let position = Position {
            asset: asset.clone(),
            asset_type,
            weight,
            value: 0.0,
            quantity: 0.0,
            entry_price: 0.0,
            current_price: 0.0,
            unrealized_pnl: 0.0,
            timestamp: Utc::now(),
        };
        
        self.positions.insert(asset, position);
        self.updated_at = Utc::now();
        
        Ok(())
    }
    
    /// Remove an asset from the portfolio
    pub fn remove_asset(&mut self, asset: &str) -> Result<()> {
        if self.positions.remove(asset).is_none() {
            return Err(TalebianError::portfolio_construction(
                format!("Asset {} not found in portfolio", asset)
            ));
        }
        
        self.updated_at = Utc::now();
        Ok(())
    }
    
    /// Update asset weight
    pub fn update_weight(&mut self, asset: &str, new_weight: f64) -> Result<()> {
        if new_weight < 0.0 || new_weight > 1.0 {
            return Err(TalebianError::invalid_parameter(
                "new_weight",
                "Weight must be between 0.0 and 1.0"
            ));
        }
        
        if let Some(position) = self.positions.get_mut(asset) {
            position.weight = new_weight;
            position.timestamp = Utc::now();
            self.updated_at = Utc::now();
            Ok(())
        } else {
            Err(TalebianError::portfolio_construction(
                format!("Asset {} not found in portfolio", asset)
            ))
        }
    }
    
    /// Update asset price
    pub fn update_price(&mut self, asset: &str, price: f64) -> Result<()> {
        if price <= 0.0 {
            return Err(TalebianError::invalid_parameter(
                "price",
                "Price must be positive"
            ));
        }
        
        if let Some(position) = self.positions.get_mut(asset) {
            position.current_price = price;
            
            if position.entry_price > 0.0 && position.quantity != 0.0 {
                position.value = position.quantity * price;
                position.unrealized_pnl = (price - position.entry_price) * position.quantity;
            }
            
            position.timestamp = Utc::now();
            self.updated_at = Utc::now();
            Ok(())
        } else {
            Err(TalebianError::portfolio_construction(
                format!("Asset {} not found in portfolio", asset)
            ))
        }
    }
    
    /// Calculate total portfolio value
    pub fn calculate_total_value(&mut self) -> f64 {
        self.total_value = self.positions.values()
            .map(|pos| pos.value)
            .sum();
        self.total_value
    }
    
    /// Get portfolio weights
    pub fn get_weights(&self) -> HashMap<String, f64> {
        self.positions.iter()
            .map(|(asset, pos)| (asset.clone(), pos.weight))
            .collect()
    }
    
    /// Get asset types
    pub fn get_asset_types(&self) -> HashMap<String, AssetType> {
        self.positions.iter()
            .map(|(asset, pos)| (asset.clone(), pos.asset_type))
            .collect()
    }
    
    /// Measure portfolio antifragility
    pub fn measure_antifragility(&self) -> Result<f64> {
        let mut antifragility_score = 0.0;
        let mut total_weight = 0.0;
        
        for position in self.positions.values() {
            let asset_antifragility = match position.asset_type {
                AssetType::Antifragile => 1.0,
                AssetType::Derivative => 0.8, // Options can be antifragile
                AssetType::Volatile => 0.3,   // Some upside potential
                AssetType::Alternative => 0.5,
                AssetType::Moderate => 0.1,
                AssetType::Safe => 0.0,       // No antifragility
                AssetType::Risky => 0.4,      // Some antifragile potential
                AssetType::Hedge => 0.2,      // Hedge instruments can be antifragile
            };
            
            antifragility_score += position.weight * asset_antifragility;
            total_weight += position.weight;
        }
        
        if total_weight > 0.0 {
            Ok(antifragility_score / total_weight)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate portfolio risk metrics
    pub fn calculate_risk_metrics(&self) -> Result<PortfolioRiskMetrics> {
        let total_weight: f64 = self.positions.values().map(|pos| pos.weight).sum();
        
        if total_weight == 0.0 {
            return Ok(PortfolioRiskMetrics::default());
        }
        
        // Calculate concentration metrics
        let weights: Vec<f64> = self.positions.values().map(|pos| pos.weight).collect();
        let hhi = weights.iter().map(|w| w * w).sum::<f64>();
        let effective_positions = if hhi > 0.0 { 1.0 / hhi } else { 0.0 };
        
        // Calculate asset type allocation
        let mut safe_allocation = 0.0;
        let mut risky_allocation = 0.0;
        let mut antifragile_allocation = 0.0;
        
        for position in self.positions.values() {
            match position.asset_type {
                AssetType::Safe => safe_allocation += position.weight,
                AssetType::Moderate => safe_allocation += position.weight * 0.5,
                AssetType::Volatile | AssetType::Derivative | AssetType::Risky => risky_allocation += position.weight,
                AssetType::Antifragile | AssetType::Alternative => antifragile_allocation += position.weight,
                AssetType::Hedge => antifragile_allocation += position.weight * 0.3, // Hedge is partially antifragile
            }
        }
        
        // Calculate tail risk
        let tail_risk = risky_allocation * 2.0 + antifragile_allocation * 1.5;
        
        // Calculate diversification ratio
        let diversification = effective_positions / self.positions.len() as f64;
        
        Ok(PortfolioRiskMetrics {
            total_weight,
            herfindahl_index: hhi,
            effective_positions,
            safe_allocation,
            risky_allocation,
            antifragile_allocation,
            tail_risk,
            diversification_ratio: diversification,
            antifragility_score: self.measure_antifragility()?,
        })
    }
    
    /// Get position for an asset
    pub fn get_position(&self, asset: &str) -> Option<&Position> {
        self.positions.get(asset)
    }
    
    /// Get all positions
    pub fn get_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }
    
    /// Check portfolio constraints
    pub fn validate_constraints(&self) -> Result<()> {
        let total_weight: f64 = self.positions.values().map(|pos| pos.weight).sum();
        
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(TalebianError::portfolio_construction(
                format!("Portfolio weights sum to {:.3}, should be 1.0", total_weight)
            ));
        }
        
        // Check individual position limits
        for (asset, position) in &self.positions {
            if position.weight > self.config.max_position_size {
                return Err(TalebianError::portfolio_construction(
                    format!("Asset {} weight {:.3} exceeds maximum {:.3}", 
                           asset, position.weight, self.config.max_position_size)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Rebalance portfolio to target weights
    pub fn rebalance(&mut self, target_weights: HashMap<String, f64>) -> Result<Vec<RebalanceAction>> {
        let mut actions = Vec::new();
        
        // Calculate required trades
        for (asset, &target_weight) in &target_weights {
            if let Some(position) = self.positions.get(asset) {
                let weight_diff = target_weight - position.weight;
                
                if weight_diff.abs() > 0.001 { // 0.1% threshold
                    actions.push(RebalanceAction {
                        asset: asset.clone(),
                        action_type: if weight_diff > 0.0 { ActionType::Buy } else { ActionType::Sell },
                        current_weight: position.weight,
                        target_weight,
                        weight_change: weight_diff,
                        estimated_cost: weight_diff.abs() * self.config.max_position_size * 0.001, // 0.1% transaction cost
                    });
                }
            } else {
                // New position
                actions.push(RebalanceAction {
                    asset: asset.clone(),
                    action_type: ActionType::Buy,
                    current_weight: 0.0,
                    target_weight,
                    weight_change: target_weight,
                    estimated_cost: target_weight * self.config.max_position_size * 0.001,
                });
            }
        }
        
        // Check for positions to close
        for (asset, position) in &self.positions {
            if !target_weights.contains_key(asset) {
                actions.push(RebalanceAction {
                    asset: asset.clone(),
                    action_type: ActionType::Sell,
                    current_weight: position.weight,
                    target_weight: 0.0,
                    weight_change: -position.weight,
                    estimated_cost: position.weight * self.config.max_position_size * 0.001,
                });
            }
        }
        
        Ok(actions)
    }
}

impl TalebianRiskComponent for AntifragilePortfolio {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn name(&self) -> &str {
        "Antifragile Portfolio"
    }
    
    fn validate(&self) -> Result<()> {
        self.validate_constraints()
    }
    
    fn reset(&mut self) {
        self.positions.clear();
        self.total_value = 0.0;
        self.updated_at = Utc::now();
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "AntifragilePortfolio".to_string());
        metadata.insert("id".to_string(), self.id.clone());
        metadata.insert("num_positions".to_string(), self.positions.len().to_string());
        metadata.insert("total_value".to_string(), self.total_value.to_string());
        metadata.insert("created_at".to_string(), self.created_at.to_rfc3339());
        metadata.insert("updated_at".to_string(), self.updated_at.to_rfc3339());
        metadata
    }
}

/// Portfolio risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioRiskMetrics {
    /// Total portfolio weight
    pub total_weight: f64,
    /// Herfindahl-Hirschman Index (concentration)
    pub herfindahl_index: f64,
    /// Effective number of positions
    pub effective_positions: f64,
    /// Allocation to safe assets
    pub safe_allocation: f64,
    /// Allocation to risky assets
    pub risky_allocation: f64,
    /// Allocation to antifragile assets
    pub antifragile_allocation: f64,
    /// Tail risk measure
    pub tail_risk: f64,
    /// Diversification ratio
    pub diversification_ratio: f64,
    /// Portfolio antifragility score
    pub antifragility_score: f64,
}

impl Default for PortfolioRiskMetrics {
    fn default() -> Self {
        Self {
            total_weight: 0.0,
            herfindahl_index: 0.0,
            effective_positions: 0.0,
            safe_allocation: 0.0,
            risky_allocation: 0.0,
            antifragile_allocation: 0.0,
            tail_risk: 0.0,
            diversification_ratio: 0.0,
            antifragility_score: 0.0,
        }
    }
}

/// Rebalancing action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceAction {
    /// Asset to rebalance
    pub asset: String,
    /// Type of action
    pub action_type: ActionType,
    /// Current weight
    pub current_weight: f64,
    /// Target weight
    pub target_weight: f64,
    /// Weight change required
    pub weight_change: f64,
    /// Estimated transaction cost
    pub estimated_cost: f64,
}

/// Type of rebalancing action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Buy more of the asset
    Buy,
    /// Sell some of the asset
    Sell,
    /// Hold current position
    Hold,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_portfolio_creation() {
        let config = RiskConfig::default();
        let portfolio = AntifragilePortfolio::new("test_portfolio", config);
        
        assert_eq!(portfolio.id(), "test_portfolio");
        assert_eq!(portfolio.name(), "Antifragile Portfolio");
        assert_eq!(portfolio.positions.len(), 0);
        assert_eq!(portfolio.total_value, 0.0);
    }
    
    #[test]
    fn test_add_asset() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        let result = portfolio.add_asset("BTC", 0.1, AssetType::Volatile);
        assert!(result.is_ok());
        
        assert_eq!(portfolio.positions.len(), 1);
        assert!(portfolio.positions.contains_key("BTC"));
        
        let position = portfolio.get_position("BTC").unwrap();
        assert_eq!(position.weight, 0.1);
        assert_eq!(position.asset_type, AssetType::Volatile);
    }
    
    #[test]
    fn test_invalid_weight() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        let result = portfolio.add_asset("BTC", 1.5, AssetType::Volatile);
        assert!(result.is_err());
        
        let result = portfolio.add_asset("BTC", -0.1, AssetType::Volatile);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_update_weight() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        portfolio.add_asset("BTC", 0.1, AssetType::Volatile).unwrap();
        
        let result = portfolio.update_weight("BTC", 0.2);
        assert!(result.is_ok());
        
        let position = portfolio.get_position("BTC").unwrap();
        assert_eq!(position.weight, 0.2);
    }
    
    #[test]
    fn test_update_price() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        portfolio.add_asset("BTC", 0.1, AssetType::Volatile).unwrap();
        
        // Set entry price and quantity first
        if let Some(position) = portfolio.positions.get_mut("BTC") {
            position.entry_price = 50000.0;
            position.quantity = 1.0;
        }
        
        let result = portfolio.update_price("BTC", 55000.0);
        assert!(result.is_ok());
        
        let position = portfolio.get_position("BTC").unwrap();
        assert_eq!(position.current_price, 55000.0);
        assert_eq!(position.value, 55000.0);
        assert_eq!(position.unrealized_pnl, 5000.0);
    }
    
    #[test]
    fn test_antifragility_measurement() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        portfolio.add_asset("BONDS", 0.5, AssetType::Safe).unwrap();
        portfolio.add_asset("GOLD", 0.3, AssetType::Antifragile).unwrap();
        portfolio.add_asset("OPTIONS", 0.2, AssetType::Derivative).unwrap();
        
        let antifragility = portfolio.measure_antifragility().unwrap();
        
        // Expected: 0.5*0.0 + 0.3*1.0 + 0.2*0.8 = 0.46
        assert!((antifragility - 0.46).abs() < 0.01);
    }
    
    #[test]
    fn test_risk_metrics() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        portfolio.add_asset("BONDS", 0.6, AssetType::Safe).unwrap();
        portfolio.add_asset("STOCKS", 0.3, AssetType::Volatile).unwrap();
        portfolio.add_asset("GOLD", 0.1, AssetType::Antifragile).unwrap();
        
        let metrics = portfolio.calculate_risk_metrics().unwrap();
        
        assert!((metrics.total_weight - 1.0).abs() < 0.01);
        assert_eq!(metrics.safe_allocation, 0.6);
        assert_eq!(metrics.risky_allocation, 0.3);
        assert_eq!(metrics.antifragile_allocation, 0.1);
        assert!(metrics.effective_positions > 0.0);
    }
    
    #[test]
    fn test_portfolio_validation() {
        let config = RiskConfig {
            max_position_size: 0.4,
            ..Default::default()
        };
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        portfolio.add_asset("STOCKS", 0.5, AssetType::Volatile).unwrap();
        portfolio.add_asset("BONDS", 0.5, AssetType::Safe).unwrap();
        
        // Should fail because STOCKS exceeds max position size
        let validation = portfolio.validate_constraints();
        assert!(validation.is_err());
    }
    
    #[test]
    fn test_rebalancing() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        portfolio.add_asset("STOCKS", 0.6, AssetType::Volatile).unwrap();
        portfolio.add_asset("BONDS", 0.4, AssetType::Safe).unwrap();
        
        let mut target_weights = HashMap::new();
        target_weights.insert("STOCKS".to_string(), 0.5);
        target_weights.insert("BONDS".to_string(), 0.5);
        
        let actions = portfolio.rebalance(target_weights).unwrap();
        
        assert_eq!(actions.len(), 2);
        
        // Find the STOCKS action
        let stocks_action = actions.iter().find(|a| a.asset == "STOCKS").unwrap();
        assert_eq!(stocks_action.action_type, ActionType::Sell);
        assert_eq!(stocks_action.weight_change, -0.1);
        
        // Find the BONDS action
        let bonds_action = actions.iter().find(|a| a.asset == "BONDS").unwrap();
        assert_eq!(bonds_action.action_type, ActionType::Buy);
        assert_eq!(bonds_action.weight_change, 0.1);
    }
    
    #[test]
    fn test_remove_asset() {
        let config = RiskConfig::default();
        let mut portfolio = AntifragilePortfolio::new("test", config);
        
        portfolio.add_asset("BTC", 0.1, AssetType::Volatile).unwrap();
        assert_eq!(portfolio.positions.len(), 1);
        
        let result = portfolio.remove_asset("BTC");
        assert!(result.is_ok());
        assert_eq!(portfolio.positions.len(), 0);
        
        // Try to remove non-existent asset
        let result = portfolio.remove_asset("ETH");
        assert!(result.is_err());
    }
}