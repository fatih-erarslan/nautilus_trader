//! Volatility modeling and hedging algorithms

use crate::{HedgeError, HedgeConfig, MarketData, utils::math};
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Volatility estimation methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VolatilityMethod {
    /// Historical volatility
    Historical,
    /// Exponentially weighted moving average
    EWMA,
    /// GARCH modeling
    GARCH,
    /// Stochastic volatility
    StochasticVol,
}

/// Volatility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityConfig {
    /// Estimation method
    pub estimation_method: VolatilityMethod,
    /// Lookback window
    pub lookback_window: usize,
    /// EWMA lambda parameter
    pub ewma_lambda: f64,
    /// GARCH parameters
    pub garch_alpha: f64,
    pub garch_beta: f64,
    /// Stochastic volatility parameters
    pub stoch_vol_kappa: f64,
    pub stoch_vol_theta: f64,
    pub stoch_vol_sigma: f64,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            estimation_method: VolatilityMethod::Historical,
            lookback_window: 252,
            ewma_lambda: 0.94,
            garch_alpha: 0.1,
            garch_beta: 0.85,
            stoch_vol_kappa: 2.0,
            stoch_vol_theta: 0.04,
            stoch_vol_sigma: 0.3,
        }
    }
}

/// Volatility hedger
#[derive(Debug, Clone)]
pub struct VolatilityHedger {
    /// Configuration
    config: HedgeConfig,
    /// Price history
    price_history: VecDeque<f64>,
    /// Return history
    return_history: VecDeque<f64>,
    /// Volatility estimates
    volatility_estimates: VecDeque<f64>,
    /// Current volatility
    current_volatility: f64,
    /// Grid levels
    grid_levels: Vec<GridLevel>,
}

impl VolatilityHedger {
    /// Create new volatility hedger
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            return_history: VecDeque::new(),
            volatility_estimates: VecDeque::new(),
            current_volatility: 0.0,
            grid_levels: Vec::new(),
        }
    }
    
    /// Update with market data
    pub fn update(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        self.price_history.push_back(market_data.close);
        
        // Calculate return
        if self.price_history.len() >= 2 {
            let current_price = market_data.close;
            let previous_price = self.price_history[self.price_history.len() - 2];
            let return_val = (current_price - previous_price) / previous_price;
            self.return_history.push_back(return_val);
        }
        
        // Keep only recent history
        let max_len = 1000;
        if self.price_history.len() > max_len {
            self.price_history.pop_front();
        }
        if self.return_history.len() > max_len {
            self.return_history.pop_front();
        }
        
        // Update volatility estimate
        self.update_volatility_estimate()?;
        
        // Update grid levels
        self.update_grid_levels(market_data.close)?;
        
        Ok(())
    }
    
    /// Update volatility estimate
    fn update_volatility_estimate(&mut self) -> Result<(), HedgeError> {
        if self.return_history.len() < 20 {
            return Ok(());
        }
        
        let returns: Vec<f64> = self.return_history.iter().copied().collect();
        
        match self.config.volatility_config.estimation_method {
            VolatilityMethod::Historical => {
                self.current_volatility = math::standard_deviation(&returns)?;
            }
            VolatilityMethod::EWMA => {
                self.current_volatility = self.calculate_ewma_volatility(&returns)?;
            }
            VolatilityMethod::GARCH => {
                self.current_volatility = self.calculate_garch_volatility(&returns)?;
            }
            VolatilityMethod::StochasticVol => {
                self.current_volatility = self.calculate_stochastic_volatility(&returns)?;
            }
        }
        
        self.volatility_estimates.push_back(self.current_volatility);
        
        // Keep only recent estimates
        if self.volatility_estimates.len() > 100 {
            self.volatility_estimates.pop_front();
        }
        
        Ok(())
    }
    
    /// Calculate EWMA volatility
    fn calculate_ewma_volatility(&self, returns: &[f64]) -> Result<f64, HedgeError> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let lambda = 0.94; // Decay factor
        let mut ewma_variance = returns[0].powi(2);
        
        for &return_val in returns.iter().skip(1) {
            ewma_variance = lambda * ewma_variance + (1.0 - lambda) * return_val.powi(2);
        }
        
        Ok(ewma_variance.sqrt())
    }
    
    /// Calculate GARCH volatility (simplified)
    fn calculate_garch_volatility(&self, returns: &[f64]) -> Result<f64, HedgeError> {
        if returns.len() < 10 {
            return Ok(math::standard_deviation(returns)?);
        }
        
        let params = &self.config.volatility_config.garch_params;
        let mut variance = returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;
        
        // Simplified GARCH(1,1) estimation
        for &return_val in returns.iter().rev().take(10) {
            variance = params.omega + params.alpha * return_val.powi(2) + params.beta * variance;
        }
        
        Ok(variance.sqrt())
    }
    
    /// Calculate stochastic volatility (simplified)
    fn calculate_stochastic_volatility(&self, returns: &[f64]) -> Result<f64, HedgeError> {
        // Simplified stochastic volatility using local volatility
        let window_size = 20;
        if returns.len() < window_size {
            return Ok(math::standard_deviation(returns)?);
        }
        
        let recent_returns = &returns[returns.len() - window_size..];
        let local_volatility = math::standard_deviation(recent_returns)?;
        
        Ok(local_volatility)
    }
    
    /// Update grid levels
    fn update_grid_levels(&mut self, current_price: f64) -> Result<(), HedgeError> {
        let grid_params = &self.config.volatility_config.grid_params;
        
        // Clear existing grid levels
        self.grid_levels.clear();
        
        // Calculate grid spacing based on volatility
        let grid_spacing = if grid_params.adaptive_sizing {
            grid_params.grid_spacing * (1.0 + self.current_volatility)
        } else {
            grid_params.grid_spacing
        };
        
        // Create grid levels above and below current price
        for i in 1..=grid_params.max_levels {
            let level_above = current_price * (1.0 + grid_spacing * i as f64);
            let level_below = current_price * (1.0 - grid_spacing * i as f64);
            
            self.grid_levels.push(GridLevel {
                price: level_above,
                side: GridSide::Sell,
                size: grid_params.grid_size,
                active: true,
            });
            
            self.grid_levels.push(GridLevel {
                price: level_below,
                side: GridSide::Buy,
                size: grid_params.grid_size,
                active: true,
            });
        }
        
        Ok(())
    }
    
    /// Get volatility forecast
    pub fn get_volatility_forecast(&self, horizon: usize) -> Result<f64, HedgeError> {
        if self.volatility_estimates.is_empty() {
            return Ok(0.0);
        }
        
        // Simple volatility forecast using historical mean
        let recent_estimates: Vec<f64> = self.volatility_estimates.iter()
            .rev()
            .take(horizon.min(self.volatility_estimates.len()))
            .copied()
            .collect();
        
        let forecast = recent_estimates.iter().sum::<f64>() / recent_estimates.len() as f64;
        Ok(forecast)
    }
    
    /// Get hedge recommendation
    pub fn get_hedge_recommendation(&self, target_volatility: f64) -> Result<VolatilityHedgeRecommendation, HedgeError> {
        let current_vol = self.current_volatility;
        let vol_diff = current_vol - target_volatility;
        
        let hedge_ratio = if vol_diff.abs() > self.config.volatility_config.rebalancing_threshold {
            -vol_diff / target_volatility
        } else {
            0.0
        };
        
        let recommendation = if vol_diff > 0.0 {
            VolatilityHedgeAction::SellVolatility
        } else if vol_diff < 0.0 {
            VolatilityHedgeAction::BuyVolatility
        } else {
            VolatilityHedgeAction::Hold
        };
        
        Ok(VolatilityHedgeRecommendation {
            action: recommendation,
            hedge_ratio,
            current_volatility: current_vol,
            target_volatility,
            confidence: self.calculate_confidence()?,
        })
    }
    
    /// Calculate confidence in volatility estimate
    fn calculate_confidence(&self) -> Result<f64, HedgeError> {
        if self.volatility_estimates.len() < 5 {
            return Ok(0.0);
        }
        
        let estimates: Vec<f64> = self.volatility_estimates.iter().copied().collect();
        let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let variance = estimates.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / estimates.len() as f64;
        
        // Confidence is inversely related to variance
        let confidence = 1.0 / (1.0 + variance);
        Ok(confidence)
    }
    
    /// Get current volatility
    pub fn get_current_volatility(&self) -> f64 {
        self.current_volatility
    }
    
    /// Get active grid levels
    pub fn get_active_grid_levels(&self) -> Vec<GridLevel> {
        self.grid_levels.iter()
            .filter(|level| level.active)
            .cloned()
            .collect()
    }
    
    /// Update grid level status
    pub fn update_grid_level(&mut self, price: f64, executed: bool) -> Result<(), HedgeError> {
        for level in &mut self.grid_levels {
            if (level.price - price).abs() < 0.01 {
                level.active = !executed;
                break;
            }
        }
        
        Ok(())
    }
}

/// Grid level
#[derive(Debug, Clone)]
pub struct GridLevel {
    /// Price level
    pub price: f64,
    /// Side (buy/sell)
    pub side: GridSide,
    /// Order size
    pub size: f64,
    /// Active status
    pub active: bool,
}

/// Grid side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridSide {
    Buy,
    Sell,
}

/// Volatility hedge recommendation
#[derive(Debug, Clone)]
pub struct VolatilityHedgeRecommendation {
    /// Recommended action
    pub action: VolatilityHedgeAction,
    /// Hedge ratio
    pub hedge_ratio: f64,
    /// Current volatility
    pub current_volatility: f64,
    /// Target volatility
    pub target_volatility: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Volatility hedge action
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityHedgeAction {
    BuyVolatility,
    SellVolatility,
    Hold,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_volatility_hedger_creation() {
        let config = HedgeConfig::default();
        let hedger = VolatilityHedger::new(config);
        
        assert_eq!(hedger.price_history.len(), 0);
        assert_eq!(hedger.current_volatility, 0.0);
    }
    
    #[test]
    fn test_volatility_hedger_update() {
        let config = HedgeConfig::default();
        let mut hedger = VolatilityHedger::new(config);
        
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            Utc::now(),
            [100.0, 105.0, 95.0, 102.0, 1000.0]
        );
        
        hedger.update(&market_data).unwrap();
        
        assert_eq!(hedger.price_history.len(), 1);
        assert_eq!(hedger.return_history.len(), 0);
        
        let market_data2 = MarketData::new(
            "BTCUSD".to_string(),
            Utc::now(),
            [102.0, 108.0, 98.0, 105.0, 1100.0]
        );
        
        hedger.update(&market_data2).unwrap();
        
        assert_eq!(hedger.price_history.len(), 2);
        assert_eq!(hedger.return_history.len(), 1);
    }
    
    #[test]
    fn test_volatility_forecast() {
        let config = HedgeConfig::default();
        let mut hedger = VolatilityHedger::new(config);
        
        // Add some volatility estimates
        hedger.volatility_estimates.push_back(0.1);
        hedger.volatility_estimates.push_back(0.15);
        hedger.volatility_estimates.push_back(0.12);
        
        let forecast = hedger.get_volatility_forecast(3).unwrap();
        assert!(forecast > 0.0);
        assert!(forecast < 1.0);
    }
}