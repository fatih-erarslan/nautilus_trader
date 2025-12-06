//! Pairs trading and statistical arbitrage algorithms

use crate::{HedgeError, HedgeConfig, MarketData, utils::math};
use std::collections::VecDeque;

/// Pairs trading engine
#[derive(Debug, Clone)]
pub struct PairsTrader {
    /// Configuration
    config: HedgeConfig,
    /// Price history for asset A
    price_history_a: VecDeque<f64>,
    /// Price history for asset B
    price_history_b: VecDeque<f64>,
    /// Spread history
    spread_history: VecDeque<f64>,
    /// Hedge ratio
    hedge_ratio: f64,
    /// Current position
    current_position: Option<PairPosition>,
}

impl PairsTrader {
    /// Create new pairs trader
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            config,
            price_history_a: VecDeque::new(),
            price_history_b: VecDeque::new(),
            spread_history: VecDeque::new(),
            hedge_ratio: 1.0,
            current_position: None,
        }
    }
    
    /// Update with market data
    pub fn update(&mut self, price_a: f64, price_b: f64) -> Result<(), HedgeError> {
        self.price_history_a.push_back(price_a);
        self.price_history_b.push_back(price_b);
        
        // Keep only lookback window
        let max_len = self.config.pairs_config.lookback_window;
        if self.price_history_a.len() > max_len {
            self.price_history_a.pop_front();
        }
        if self.price_history_b.len() > max_len {
            self.price_history_b.pop_front();
        }
        
        // Calculate spread
        let spread = price_a - self.hedge_ratio * price_b;
        self.spread_history.push_back(spread);
        
        if self.spread_history.len() > max_len {
            self.spread_history.pop_front();
        }
        
        // Update hedge ratio
        self.update_hedge_ratio()?;
        
        Ok(())
    }
    
    /// Update hedge ratio using cointegration
    fn update_hedge_ratio(&mut self) -> Result<(), HedgeError> {
        if self.price_history_a.len() < 30 || self.price_history_b.len() < 30 {
            return Ok(());
        }
        
        let prices_a: Vec<f64> = self.price_history_a.iter().copied().collect();
        let prices_b: Vec<f64> = self.price_history_b.iter().copied().collect();
        
        // Calculate hedge ratio using linear regression
        let correlation = math::correlation(&prices_a, &prices_b)?;
        let std_a = math::standard_deviation(&prices_a)?;
        let std_b = math::standard_deviation(&prices_b)?;
        
        if std_b > 0.0 {
            self.hedge_ratio = correlation * (std_a / std_b);
        }
        
        Ok(())
    }
    
    /// Check cointegration
    pub fn check_cointegration(&self) -> Result<bool, HedgeError> {
        if self.price_history_a.len() < 50 || self.price_history_b.len() < 50 {
            return Ok(false);
        }
        
        let prices_a: Vec<f64> = self.price_history_a.iter().copied().collect();
        let prices_b: Vec<f64> = self.price_history_b.iter().copied().collect();
        
        // Simplified cointegration test using correlation
        let correlation = math::correlation(&prices_a, &prices_b)?;
        
        Ok(correlation.abs() > self.config.pairs_config.cointegration_threshold)
    }
    
    /// Calculate z-score
    pub fn calculate_zscore(&self) -> Result<f64, HedgeError> {
        if self.spread_history.len() < 20 {
            return Ok(0.0);
        }
        
        let spreads: Vec<f64> = self.spread_history.iter().copied().collect();
        let mean = spreads.iter().sum::<f64>() / spreads.len() as f64;
        let std_dev = math::standard_deviation(&spreads)?;
        
        if std_dev > 0.0 {
            let current_spread = *self.spread_history.back().unwrap();
            Ok((current_spread - mean) / std_dev)
        } else {
            Ok(0.0)
        }
    }
    
    /// Generate trading signal
    pub fn generate_signal(&mut self) -> Result<Option<PairSignal>, HedgeError> {
        if !self.check_cointegration()? {
            return Ok(None);
        }
        
        let zscore = self.calculate_zscore()?;
        
        match self.current_position {
            None => {
                // No position, check for entry signals
                if zscore > self.config.pairs_config.zscore_entry {
                    // Spread is high, short A and long B
                    self.current_position = Some(PairPosition::Short);
                    Ok(Some(PairSignal::Short))
                } else if zscore < -self.config.pairs_config.zscore_entry {
                    // Spread is low, long A and short B
                    self.current_position = Some(PairPosition::Long);
                    Ok(Some(PairSignal::Long))
                } else {
                    Ok(None)
                }
            }
            Some(PairPosition::Long) => {
                // Long position, check for exit
                if zscore > -self.config.pairs_config.zscore_exit {
                    self.current_position = None;
                    Ok(Some(PairSignal::Exit))
                } else if zscore < -self.config.pairs_config.zscore_entry * 2.0 {
                    // Stop loss
                    self.current_position = None;
                    Ok(Some(PairSignal::StopLoss))
                } else {
                    Ok(None)
                }
            }
            Some(PairPosition::Short) => {
                // Short position, check for exit
                if zscore < self.config.pairs_config.zscore_exit {
                    self.current_position = None;
                    Ok(Some(PairSignal::Exit))
                } else if zscore > self.config.pairs_config.zscore_entry * 2.0 {
                    // Stop loss
                    self.current_position = None;
                    Ok(Some(PairSignal::StopLoss))
                } else {
                    Ok(None)
                }
            }
        }
    }
    
    /// Get current hedge ratio
    pub fn get_hedge_ratio(&self) -> f64 {
        self.hedge_ratio
    }
    
    /// Get current position
    pub fn get_position(&self) -> Option<PairPosition> {
        self.current_position
    }
    
    /// Calculate position size
    pub fn calculate_position_size(&self, portfolio_value: f64) -> f64 {
        portfolio_value * self.config.pairs_config.max_position_size
    }
}

/// Pair position
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairPosition {
    Long,
    Short,
}

/// Pair signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairSignal {
    Long,
    Short,
    Exit,
    StopLoss,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairs_trader_creation() {
        let config = HedgeConfig::default();
        let trader = PairsTrader::new(config);
        
        assert_eq!(trader.price_history_a.len(), 0);
        assert_eq!(trader.price_history_b.len(), 0);
        assert_eq!(trader.hedge_ratio, 1.0);
    }
    
    #[test]
    fn test_pairs_trader_update() {
        let config = HedgeConfig::default();
        let mut trader = PairsTrader::new(config);
        
        trader.update(100.0, 98.0).unwrap();
        trader.update(101.0, 99.0).unwrap();
        
        assert_eq!(trader.price_history_a.len(), 2);
        assert_eq!(trader.price_history_b.len(), 2);
        assert_eq!(trader.spread_history.len(), 2);
    }
}