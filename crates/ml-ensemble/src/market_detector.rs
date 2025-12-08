use crate::types::MarketCondition;
use anyhow::Result;
use ats_core::types::MarketData;
use ndarray::Array1;
use std::collections::VecDeque;

/// Market condition detector for adaptive model selection
pub struct MarketConditionDetector {
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    lookback_period: usize,
}

impl MarketConditionDetector {
    /// Create new market condition detector
    pub fn new() -> Self {
        Self {
            price_history: VecDeque::with_capacity(500),
            volume_history: VecDeque::with_capacity(500),
            lookback_period: 100,
        }
    }
    
    /// Detect current market condition
    pub fn detect_condition(&self, market_data: &MarketData) -> Result<MarketCondition> {
        // Update history
        let mid_price = (market_data.bid + market_data.ask) / 2.0;
        
        // If not enough history, default to ranging
        if self.price_history.len() < self.lookback_period {
            return Ok(MarketCondition::Ranging);
        }
        
        // Calculate metrics
        let returns = self.calculate_returns();
        let volatility = self.calculate_volatility(&returns);
        let trend_strength = self.calculate_trend_strength(&returns);
        let volume_ratio = self.calculate_volume_ratio();
        let price_efficiency = self.calculate_price_efficiency();
        
        // Detect anomalies first
        if self.is_anomalous(&returns, volatility, volume_ratio) {
            return Ok(MarketCondition::Anomalous);
        }
        
        // Detect market condition based on metrics
        if volatility > 0.02 {
            Ok(MarketCondition::HighVolatility)
        } else if volatility < 0.005 {
            Ok(MarketCondition::LowVolatility)
        } else if trend_strength > 0.7 {
            Ok(MarketCondition::Trending)
        } else if self.is_breakout(&returns, volume_ratio) {
            Ok(MarketCondition::Breakout)
        } else if self.is_reversal(&returns, price_efficiency) {
            Ok(MarketCondition::Reversal)
        } else {
            Ok(MarketCondition::Ranging)
        }
    }
    
    /// Update price and volume history
    pub fn update(&mut self, market_data: &MarketData) {
        let mid_price = (market_data.bid + market_data.ask) / 2.0;
        let volume = market_data.bid_size + market_data.ask_size;
        
        self.price_history.push_back(mid_price);
        self.volume_history.push_back(volume);
        
        // Limit history size
        if self.price_history.len() > 500 {
            self.price_history.pop_front();
        }
        if self.volume_history.len() > 500 {
            self.volume_history.pop_front();
        }
    }
    
    /// Calculate returns
    fn calculate_returns(&self) -> Vec<f64> {
        let prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(self.lookback_period)
            .rev()
            .copied()
            .collect();
        
        let mut returns = Vec::new();
        for i in 1..prices.len() {
            returns.push((prices[i] / prices[i - 1]) - 1.0);
        }
        
        returns
    }
    
    /// Calculate volatility
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }
    
    /// Calculate trend strength using directional movement
    fn calculate_trend_strength(&self, returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return 0.0;
        }
        
        let mut positive_moves = 0;
        let mut negative_moves = 0;
        
        for &ret in returns {
            if ret > 0.0 {
                positive_moves += 1;
            } else if ret < 0.0 {
                negative_moves += 1;
            }
        }
        
        let total_moves = positive_moves + negative_moves;
        if total_moves == 0 {
            return 0.0;
        }
        
        ((positive_moves as f64 - negative_moves as f64).abs() / total_moves as f64)
    }
    
    /// Calculate volume ratio (current vs average)
    fn calculate_volume_ratio(&self) -> f64 {
        if self.volume_history.len() < 20 {
            return 1.0;
        }
        
        let recent_volume: f64 = self.volume_history.iter()
            .rev()
            .take(5)
            .sum::<f64>() / 5.0;
        
        let avg_volume: f64 = self.volume_history.iter()
            .rev()
            .take(50)
            .sum::<f64>() / 50.0;
        
        if avg_volume > 0.0 {
            recent_volume / avg_volume
        } else {
            1.0
        }
    }
    
    /// Calculate price efficiency ratio
    fn calculate_price_efficiency(&self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.5;
        }
        
        let prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(20)
            .rev()
            .copied()
            .collect();
        
        // Calculate net change
        let net_change = (prices[prices.len() - 1] - prices[0]).abs();
        
        // Calculate sum of absolute changes
        let mut sum_changes = 0.0;
        for i in 1..prices.len() {
            sum_changes += (prices[i] - prices[i - 1]).abs();
        }
        
        if sum_changes > 0.0 {
            net_change / sum_changes
        } else {
            0.0
        }
    }
    
    /// Check if market is anomalous
    fn is_anomalous(&self, returns: &[f64], volatility: f64, volume_ratio: f64) -> bool {
        // Check for extreme returns
        let max_return = returns.iter().map(|r| r.abs()).fold(0.0, f64::max);
        if max_return > 0.05 {
            return true;
        }
        
        // Check for extreme volatility
        if volatility > 0.05 {
            return true;
        }
        
        // Check for extreme volume
        if volume_ratio > 3.0 || volume_ratio < 0.2 {
            return true;
        }
        
        false
    }
    
    /// Check for breakout conditions
    fn is_breakout(&self, returns: &[f64], volume_ratio: f64) -> bool {
        if returns.len() < 10 {
            return false;
        }
        
        // Recent strong move with increased volume
        let recent_return: f64 = returns.iter().rev().take(3).sum();
        let recent_avg = recent_return / 3.0;
        
        recent_avg.abs() > 0.01 && volume_ratio > 1.5
    }
    
    /// Check for reversal conditions
    fn is_reversal(&self, returns: &[f64], price_efficiency: f64) -> bool {
        if returns.len() < 20 {
            return false;
        }
        
        // Check for momentum shift
        let first_half: f64 = returns[..returns.len() / 2].iter().sum();
        let second_half: f64 = returns[returns.len() / 2..].iter().sum();
        
        // Signs are different and efficiency is low
        first_half.signum() != second_half.signum() && price_efficiency < 0.3
    }
}