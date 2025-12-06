//! Whale detection and momentum trading algorithms

use crate::{HedgeError, HedgeConfig, MarketData, utils::math};
use std::collections::VecDeque;

/// Whale detection engine
#[derive(Debug, Clone)]
pub struct WhaleDetector {
    /// Configuration
    config: HedgeConfig,
    /// Market data history
    market_history: VecDeque<MarketData>,
    /// Volume profile
    volume_profile: VecDeque<f64>,
    /// Price impact history
    price_impact_history: VecDeque<f64>,
    /// Momentum indicators
    momentum_indicators: MomentumIndicators,
    /// Detected whale activities
    whale_activities: VecDeque<WhaleActivity>,
}

impl WhaleDetector {
    /// Create new whale detector
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            config,
            market_history: VecDeque::new(),
            volume_profile: VecDeque::new(),
            price_impact_history: VecDeque::new(),
            momentum_indicators: MomentumIndicators::new(),
            whale_activities: VecDeque::new(),
        }
    }
    
    /// Update with market data
    pub fn update(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        self.market_history.push_back(market_data.clone());
        
        // Keep only recent history
        let max_len = self.config.whale_config.detection_window;
        if self.market_history.len() > max_len {
            self.market_history.pop_front();
        }
        
        // Update volume profile
        self.update_volume_profile(market_data)?;
        
        // Update price impact
        self.update_price_impact(market_data)?;
        
        // Update momentum indicators
        self.update_momentum_indicators(market_data)?;
        
        // Detect whale activity
        self.detect_whale_activity(market_data)?;
        
        Ok(())
    }
    
    /// Update volume profile
    fn update_volume_profile(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        self.volume_profile.push_back(market_data.volume);
        
        let max_len = self.config.whale_config.detection_window;
        if self.volume_profile.len() > max_len {
            self.volume_profile.pop_front();
        }
        
        Ok(())
    }
    
    /// Update price impact
    fn update_price_impact(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        if self.market_history.len() < 2 {
            return Ok(());
        }
        
        let current_price = market_data.close;
        let current_volume = market_data.volume;
        
        let previous_data = &self.market_history[self.market_history.len() - 2];
        let previous_price = previous_data.close;
        let previous_volume = previous_data.volume;
        
        // Calculate price impact as price change per unit volume
        let price_change = (current_price - previous_price) / previous_price;
        let volume_change = current_volume - previous_volume;
        
        let price_impact = if volume_change.abs() > 0.0 {
            price_change / (volume_change / previous_volume)
        } else {
            0.0
        };
        
        self.price_impact_history.push_back(price_impact);
        
        let max_len = self.config.whale_config.detection_window;
        if self.price_impact_history.len() > max_len {
            self.price_impact_history.pop_front();
        }
        
        Ok(())
    }
    
    /// Update momentum indicators
    fn update_momentum_indicators(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        if self.market_history.len() < 20 {
            return Ok(());
        }
        
        let prices: Vec<f64> = self.market_history.iter().map(|d| d.close).collect();
        let volumes: Vec<f64> = self.market_history.iter().map(|d| d.volume).collect();
        
        // Calculate momentum
        let short_window = 5;
        let long_window = 20;
        
        if prices.len() >= long_window {
            let short_ma = prices[prices.len() - short_window..].iter().sum::<f64>() / short_window as f64;
            let long_ma = prices[prices.len() - long_window..].iter().sum::<f64>() / long_window as f64;
            
            self.momentum_indicators.price_momentum = (short_ma - long_ma) / long_ma;
        }
        
        // Calculate volume momentum
        if volumes.len() >= long_window {
            let short_volume_ma = volumes[volumes.len() - short_window..].iter().sum::<f64>() / short_window as f64;
            let long_volume_ma = volumes[volumes.len() - long_window..].iter().sum::<f64>() / long_window as f64;
            
            self.momentum_indicators.volume_momentum = (short_volume_ma - long_volume_ma) / long_volume_ma;
        }
        
        // Calculate volatility momentum
        let returns = math::returns(&prices)?;
        if returns.len() >= short_window {
            let recent_returns = &returns[returns.len() - short_window..];
            let recent_volatility = math::standard_deviation(recent_returns)?;
            
            if returns.len() >= long_window {
                let long_returns = &returns[returns.len() - long_window..];
                let long_volatility = math::standard_deviation(long_returns)?;
                
                if long_volatility > 0.0 {
                    self.momentum_indicators.volatility_momentum = (recent_volatility - long_volatility) / long_volatility;
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect whale activity
    fn detect_whale_activity(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        if self.volume_profile.len() < 20 || self.price_impact_history.len() < 10 {
            return Ok(());
        }
        
        // Calculate volume threshold
        let volumes: Vec<f64> = self.volume_profile.iter().copied().collect();
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let volume_threshold = avg_volume * self.config.whale_config.volume_threshold;
        
        // Calculate price impact threshold
        let impacts: Vec<f64> = self.price_impact_history.iter().copied().collect();
        let avg_impact = impacts.iter().sum::<f64>() / impacts.len() as f64;
        let impact_threshold = self.config.whale_config.price_impact_threshold;
        
        // Check for whale activity
        let current_volume = market_data.volume;
        let current_impact = *self.price_impact_history.back().unwrap_or(&0.0);
        
        let volume_anomaly = current_volume > volume_threshold;
        let impact_anomaly = current_impact.abs() > impact_threshold;
        let momentum_anomaly = self.momentum_indicators.price_momentum.abs() > self.config.whale_config.momentum_threshold;
        
        if volume_anomaly && (impact_anomaly || momentum_anomaly) {
            let whale_type = self.classify_whale_activity(market_data)?;
            let confidence = self.calculate_whale_confidence(market_data)?;
            
            let activity = WhaleActivity {
                timestamp: market_data.timestamp,
                whale_type,
                volume: current_volume,
                price_impact: current_impact,
                momentum: self.momentum_indicators.price_momentum,
                confidence,
            };
            
            self.whale_activities.push_back(activity);
            
            // Keep only recent activities
            if self.whale_activities.len() > 100 {
                self.whale_activities.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Classify whale activity type
    fn classify_whale_activity(&self, market_data: &MarketData) -> Result<WhaleType, HedgeError> {
        let price_momentum = self.momentum_indicators.price_momentum;
        let volume_momentum = self.momentum_indicators.volume_momentum;
        
        if price_momentum > 0.0 && volume_momentum > 0.0 {
            Ok(WhaleType::Accumulation)
        } else if price_momentum < 0.0 && volume_momentum > 0.0 {
            Ok(WhaleType::Distribution)
        } else if price_momentum.abs() > self.config.whale_config.rapid_entry_threshold {
            Ok(WhaleType::RapidEntry)
        } else {
            Ok(WhaleType::Unknown)
        }
    }
    
    /// Calculate whale confidence
    fn calculate_whale_confidence(&self, market_data: &MarketData) -> Result<f64, HedgeError> {
        let volume_score = if self.volume_profile.len() > 0 {
            let avg_volume = self.volume_profile.iter().sum::<f64>() / self.volume_profile.len() as f64;
            (market_data.volume / avg_volume).min(5.0) / 5.0
        } else {
            0.0
        };
        
        let impact_score = if self.price_impact_history.len() > 0 {
            let current_impact = *self.price_impact_history.back().unwrap_or(&0.0);
            (current_impact.abs() / self.config.whale_config.price_impact_threshold).min(1.0)
        } else {
            0.0
        };
        
        let momentum_score = (self.momentum_indicators.price_momentum.abs() / self.config.whale_config.momentum_threshold).min(1.0);
        
        let confidence = (volume_score + impact_score + momentum_score) / 3.0;
        Ok(confidence.min(1.0))
    }
    
    /// Get trading signal based on whale activity
    pub fn get_trading_signal(&self) -> Result<Option<WhaleSignal>, HedgeError> {
        if self.whale_activities.is_empty() {
            return Ok(None);
        }
        
        let recent_activity = self.whale_activities.back().unwrap();
        
        if recent_activity.confidence < self.config.whale_config.confidence_threshold {
            return Ok(None);
        }
        
        match recent_activity.whale_type {
            WhaleType::Accumulation => {
                Ok(Some(WhaleSignal::Follow))
            }
            WhaleType::Distribution => {
                Ok(Some(WhaleSignal::Contrarian))
            }
            WhaleType::RapidEntry => {
                Ok(Some(WhaleSignal::RapidFollow))
            }
            WhaleType::Unknown => {
                Ok(None)
            }
        }
    }
    
    /// Get recent whale activities
    pub fn get_recent_activities(&self, count: usize) -> Vec<WhaleActivity> {
        self.whale_activities.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Get whale statistics
    pub fn get_whale_statistics(&self) -> WhaleStatistics {
        let total_activities = self.whale_activities.len();
        
        let accumulation_count = self.whale_activities.iter()
            .filter(|a| a.whale_type == WhaleType::Accumulation)
            .count();
        
        let distribution_count = self.whale_activities.iter()
            .filter(|a| a.whale_type == WhaleType::Distribution)
            .count();
        
        let rapid_entry_count = self.whale_activities.iter()
            .filter(|a| a.whale_type == WhaleType::RapidEntry)
            .count();
        
        let avg_confidence = if total_activities > 0 {
            self.whale_activities.iter()
                .map(|a| a.confidence)
                .sum::<f64>() / total_activities as f64
        } else {
            0.0
        };
        
        WhaleStatistics {
            total_activities,
            accumulation_count,
            distribution_count,
            rapid_entry_count,
            avg_confidence,
        }
    }
}

/// Momentum indicators
#[derive(Debug, Clone)]
pub struct MomentumIndicators {
    /// Price momentum
    pub price_momentum: f64,
    /// Volume momentum
    pub volume_momentum: f64,
    /// Volatility momentum
    pub volatility_momentum: f64,
}

impl MomentumIndicators {
    /// Create new momentum indicators
    pub fn new() -> Self {
        Self {
            price_momentum: 0.0,
            volume_momentum: 0.0,
            volatility_momentum: 0.0,
        }
    }
}

/// Whale activity
#[derive(Debug, Clone)]
pub struct WhaleActivity {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Whale type
    pub whale_type: WhaleType,
    /// Volume
    pub volume: f64,
    /// Price impact
    pub price_impact: f64,
    /// Momentum
    pub momentum: f64,
    /// Confidence
    pub confidence: f64,
}

/// Whale type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhaleType {
    Accumulation,
    Distribution,
    RapidEntry,
    Unknown,
}

/// Whale signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhaleSignal {
    Follow,
    Contrarian,
    RapidFollow,
}

/// Whale statistics
#[derive(Debug, Clone)]
pub struct WhaleStatistics {
    /// Total activities
    pub total_activities: usize,
    /// Accumulation count
    pub accumulation_count: usize,
    /// Distribution count
    pub distribution_count: usize,
    /// Rapid entry count
    pub rapid_entry_count: usize,
    /// Average confidence
    pub avg_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_whale_detector_creation() {
        let config = HedgeConfig::default();
        let detector = WhaleDetector::new(config);
        
        assert_eq!(detector.market_history.len(), 0);
        assert_eq!(detector.whale_activities.len(), 0);
    }
    
    #[test]
    fn test_whale_detector_update() {
        let config = HedgeConfig::default();
        let mut detector = WhaleDetector::new(config);
        
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            Utc::now(),
            [100.0, 105.0, 95.0, 102.0, 1000.0]
        );
        
        detector.update(&market_data).unwrap();
        
        assert_eq!(detector.market_history.len(), 1);
        assert_eq!(detector.volume_profile.len(), 1);
    }
    
    #[test]
    fn test_momentum_indicators() {
        let indicators = MomentumIndicators::new();
        
        assert_eq!(indicators.price_momentum, 0.0);
        assert_eq!(indicators.volume_momentum, 0.0);
        assert_eq!(indicators.volatility_momentum, 0.0);
    }
    
    #[test]
    fn test_whale_statistics() {
        let config = HedgeConfig::default();
        let detector = WhaleDetector::new(config);
        
        let stats = detector.get_whale_statistics();
        
        assert_eq!(stats.total_activities, 0);
        assert_eq!(stats.accumulation_count, 0);
        assert_eq!(stats.distribution_count, 0);
        assert_eq!(stats.rapid_entry_count, 0);
        assert_eq!(stats.avg_confidence, 0.0);
    }
}