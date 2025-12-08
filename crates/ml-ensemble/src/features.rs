use crate::types::FeatureConfig;
use anyhow::Result;
use ats_core::types::MarketData;
use ndarray::Array1;
use std::collections::VecDeque;

/// Feature engineering for ML models
pub struct FeatureEngineering {
    config: FeatureConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    spread_history: VecDeque<f64>,
    feature_names: Vec<String>,
}

impl FeatureEngineering {
    /// Create new feature engineering instance
    pub fn new(config: FeatureConfig) -> Self {
        let mut feature_names = Vec::new();
        
        // Price features
        for i in 1..=config.price_lags {
            feature_names.push(format!("price_lag_{}", i));
            feature_names.push(format!("return_lag_{}", i));
        }
        
        // Technical indicators
        for indicator in &config.technical_indicators {
            feature_names.push(indicator.clone());
        }
        
        // Market microstructure features
        if config.microstructure_features {
            feature_names.extend_from_slice(&[
                "bid_ask_spread".to_string(),
                "bid_size_ratio".to_string(),
                "ask_size_ratio".to_string(),
                "volume_imbalance".to_string(),
                "price_impact".to_string(),
            ]);
        }
        
        // Order flow features
        if config.order_flow_features {
            feature_names.extend_from_slice(&[
                "order_flow_imbalance".to_string(),
                "trade_intensity".to_string(),
                "volume_weighted_spread".to_string(),
            ]);
        }
        
        Self {
            config,
            price_history: VecDeque::with_capacity(1000),
            volume_history: VecDeque::with_capacity(1000),
            spread_history: VecDeque::with_capacity(1000),
            feature_names,
        }
    }
    
    /// Extract features from market data
    pub fn extract_features(&self, market_data: &MarketData) -> Result<Array1<f32>> {
        let mut features = Vec::new();
        
        // Price and return lags
        let price_features = self.extract_price_features(market_data);
        features.extend(price_features);
        
        // Technical indicators
        let technical_features = self.extract_technical_indicators();
        features.extend(technical_features);
        
        // Market microstructure features
        if self.config.microstructure_features {
            let micro_features = self.extract_microstructure_features(market_data);
            features.extend(micro_features);
        }
        
        // Order flow features
        if self.config.order_flow_features {
            let flow_features = self.extract_order_flow_features(market_data);
            features.extend(flow_features);
        }
        
        Ok(Array1::from_vec(features))
    }
    
    /// Update feature history
    pub fn update(&mut self, market_data: &MarketData) {
        let mid_price = (market_data.bid + market_data.ask) / 2.0;
        let volume = market_data.bid_size + market_data.ask_size;
        let spread = market_data.ask - market_data.bid;
        
        self.price_history.push_back(mid_price);
        self.volume_history.push_back(volume);
        self.spread_history.push_back(spread);
        
        // Limit history size
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
        }
        if self.volume_history.len() > 1000 {
            self.volume_history.pop_front();
        }
        if self.spread_history.len() > 1000 {
            self.spread_history.pop_front();
        }
    }
    
    /// Extract price-based features
    fn extract_price_features(&self, market_data: &MarketData) -> Vec<f32> {
        let mut features = Vec::new();
        let current_price = (market_data.bid + market_data.ask) / 2.0;
        
        // Price lags and returns
        for i in 1..=self.config.price_lags {
            if self.price_history.len() > i {
                let lag_price = self.price_history[self.price_history.len() - i];
                let price_ratio = current_price / lag_price;
                let return_value = price_ratio - 1.0;
                
                features.push(price_ratio as f32);
                features.push(return_value as f32);
            } else {
                features.push(1.0);
                features.push(0.0);
            }
        }
        
        features
    }
    
    /// Extract technical indicators
    fn extract_technical_indicators(&self) -> Vec<f32> {
        let mut features = Vec::new();
        
        for indicator in &self.config.technical_indicators {
            match indicator.as_str() {
                "RSI" => features.push(self.calculate_rsi() as f32),
                "MACD" => {
                    let (macd, signal) = self.calculate_macd();
                    features.push(macd as f32);
                }
                "BB" => {
                    let (upper, lower) = self.calculate_bollinger_bands();
                    features.push(upper as f32);
                }
                "ATR" => features.push(self.calculate_atr() as f32),
                _ => features.push(0.0),
            }
        }
        
        features
    }
    
    /// Extract market microstructure features
    fn extract_microstructure_features(&self, market_data: &MarketData) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Bid-ask spread (normalized)
        let spread = (market_data.ask - market_data.bid) / market_data.bid;
        features.push(spread as f32);
        
        // Bid/ask size ratios
        let total_size = market_data.bid_size + market_data.ask_size;
        let bid_ratio = if total_size > 0.0 {
            market_data.bid_size / total_size
        } else {
            0.5
        };
        let ask_ratio = if total_size > 0.0 {
            market_data.ask_size / total_size
        } else {
            0.5
        };
        features.push(bid_ratio as f32);
        features.push(ask_ratio as f32);
        
        // Volume imbalance
        let volume_imbalance = (market_data.bid_size - market_data.ask_size) / total_size.max(1.0);
        features.push(volume_imbalance as f32);
        
        // Price impact estimate
        let avg_spread = self.spread_history.iter().sum::<f64>() / self.spread_history.len().max(1) as f64;
        let price_impact = spread / avg_spread.max(0.0001);
        features.push(price_impact as f32);
        
        features
    }
    
    /// Extract order flow features
    fn extract_order_flow_features(&self, market_data: &MarketData) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Order flow imbalance
        let flow_imbalance = (market_data.bid_size - market_data.ask_size) / 
                           (market_data.bid_size + market_data.ask_size).max(1.0);
        features.push(flow_imbalance as f32);
        
        // Trade intensity (volume relative to average)
        let avg_volume = self.volume_history.iter().sum::<f64>() / self.volume_history.len().max(1) as f64;
        let current_volume = market_data.bid_size + market_data.ask_size;
        let trade_intensity = current_volume / avg_volume.max(1.0);
        features.push(trade_intensity as f32);
        
        // Volume-weighted spread
        let vw_spread = (market_data.ask - market_data.bid) * current_volume / avg_volume.max(1.0);
        features.push(vw_spread as f32);
        
        features
    }
    
    /// Calculate RSI
    fn calculate_rsi(&self) -> f64 {
        if self.price_history.len() < 14 {
            return 50.0;
        }
        
        let prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(15)
            .rev()
            .copied()
            .collect();
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / 14.0;
        let avg_loss = losses / 14.0;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    /// Calculate MACD
    fn calculate_macd(&self) -> (f64, f64) {
        if self.price_history.len() < 26 {
            return (0.0, 0.0);
        }
        
        // Simple EMA calculation
        let ema12 = self.calculate_ema(12);
        let ema26 = self.calculate_ema(26);
        let macd = ema12 - ema26;
        let signal = macd * 0.15; // Simplified signal line
        
        (macd, signal)
    }
    
    /// Calculate EMA
    fn calculate_ema(&self, period: usize) -> f64 {
        if self.price_history.len() < period {
            return self.price_history.back().copied().unwrap_or(0.0);
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = self.price_history[self.price_history.len() - period];
        
        for i in (self.price_history.len() - period + 1)..self.price_history.len() {
            ema = alpha * self.price_history[i] + (1.0 - alpha) * ema;
        }
        
        ema
    }
    
    /// Calculate Bollinger Bands
    fn calculate_bollinger_bands(&self) -> (f64, f64) {
        if self.price_history.len() < 20 {
            let current = self.price_history.back().copied().unwrap_or(0.0);
            return (current * 1.02, current * 0.98);
        }
        
        let prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(20)
            .copied()
            .collect();
        
        let mean = prices.iter().sum::<f64>() / 20.0;
        let variance = prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / 20.0;
        let std_dev = variance.sqrt();
        
        (mean + 2.0 * std_dev, mean - 2.0 * std_dev)
    }
    
    /// Calculate ATR
    fn calculate_atr(&self) -> f64 {
        if self.price_history.len() < 14 {
            return 0.0;
        }
        
        let mut tr_values = Vec::new();
        let prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(15)
            .rev()
            .copied()
            .collect();
        
        for i in 1..prices.len() {
            let high_low = (prices[i] - prices[i - 1]).abs();
            tr_values.push(high_low);
        }
        
        tr_values.iter().sum::<f64>() / 14.0
    }
    
    /// Get feature names
    pub fn get_feature_names(&self) -> Vec<String> {
        self.feature_names.clone()
    }
}