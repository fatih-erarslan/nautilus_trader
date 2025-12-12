//! Feature extraction for whale detection
//! 
//! This module provides high-performance feature extraction from market data,
//! including technical indicators and statistical features.

use ndarray::{Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use crate::error::{Result, WhaleMLError};

/// Market features structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFeatures {
    /// Price features
    pub price: f32,
    pub volume: f32,
    pub bid: Option<f32>,
    pub ask: Option<f32>,
    
    /// Technical indicators
    pub sma_20: f32,
    pub ema_20: f32,
    pub rsi_14: f32,
    pub bb_position: f32,  // Position within Bollinger Bands
    pub macd: f32,
    
    /// Volume indicators
    pub vwap: f32,
    pub volume_sma_20: f32,
    pub volume_ratio: f32,  // Current volume / average volume
    
    /// Market microstructure
    pub spread: Option<f32>,
    pub relative_spread: Option<f32>,
    
    /// Statistical features
    pub price_change_1m: f32,
    pub price_change_5m: f32,
    pub volatility: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

/// Feature extractor for market data
pub struct FeatureExtractor {
    /// Window size for indicators
    window_size: usize,
    
    /// Price history buffer
    price_buffer: VecDeque<f32>,
    
    /// Volume history buffer
    volume_buffer: VecDeque<f32>,
    
    /// Cumulative values for VWAP
    cumulative_pv: f64,
    cumulative_volume: f64,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            price_buffer: VecDeque::with_capacity(window_size * 2),
            volume_buffer: VecDeque::with_capacity(window_size * 2),
            cumulative_pv: 0.0,
            cumulative_volume: 0.0,
        }
    }
    
    /// Extract features from raw market data
    pub fn extract_features(
        &mut self,
        price: f32,
        volume: f32,
        bid: Option<f32>,
        ask: Option<f32>,
    ) -> Result<MarketFeatures> {
        // Update buffers
        self.price_buffer.push_back(price);
        self.volume_buffer.push_back(volume);
        
        // Maintain buffer size
        if self.price_buffer.len() > self.window_size * 2 {
            self.price_buffer.pop_front();
            self.volume_buffer.pop_front();
        }
        
        // Update VWAP
        self.cumulative_pv += price as f64 * volume as f64;
        self.cumulative_volume += volume as f64;
        
        // Calculate technical indicators
        let sma_20 = self.calculate_sma(&self.price_buffer, 20);
        let ema_20 = self.calculate_ema(&self.price_buffer, 20);
        let rsi_14 = self.calculate_rsi(&self.price_buffer, 14);
        let bb_position = self.calculate_bollinger_position(&self.price_buffer, 20);
        let macd = self.calculate_macd(&self.price_buffer);
        
        // Volume indicators
        let vwap = if self.cumulative_volume > 0.0 {
            (self.cumulative_pv / self.cumulative_volume) as f32
        } else {
            price
        };
        let volume_sma_20 = self.calculate_sma(&self.volume_buffer, 20);
        let volume_ratio = if volume_sma_20 > 0.0 {
            volume / volume_sma_20
        } else {
            1.0
        };
        
        // Market microstructure
        let (spread, relative_spread) = if let (Some(b), Some(a)) = (bid, ask) {
            let s = a - b;
            let rs = if price > 0.0 { s / price } else { 0.0 };
            (Some(s), Some(rs))
        } else {
            (None, None)
        };
        
        // Price changes
        let price_change_1m = if self.price_buffer.len() > 1 {
            let prev = self.price_buffer[self.price_buffer.len() - 2];
            (price - prev) / prev
        } else {
            0.0
        };
        
        let price_change_5m = if self.price_buffer.len() > 5 {
            let prev = self.price_buffer[self.price_buffer.len() - 6];
            (price - prev) / prev
        } else {
            0.0
        };
        
        // Statistical features
        let volatility = self.calculate_volatility(&self.price_buffer);
        let skewness = self.calculate_skewness(&self.price_buffer);
        let kurtosis = self.calculate_kurtosis(&self.price_buffer);
        
        Ok(MarketFeatures {
            price,
            volume,
            bid,
            ask,
            sma_20,
            ema_20,
            rsi_14,
            bb_position,
            macd,
            vwap,
            volume_sma_20,
            volume_ratio,
            spread,
            relative_spread,
            price_change_1m,
            price_change_5m,
            volatility,
            skewness,
            kurtosis,
        })
    }
    
    /// Convert features to tensor format
    pub fn features_to_array(features: &MarketFeatures) -> Array1<f32> {
        let mut arr = Array1::zeros(19);
        
        arr[0] = features.price;
        arr[1] = features.volume;
        arr[2] = features.sma_20;
        arr[3] = features.ema_20;
        arr[4] = features.rsi_14;
        arr[5] = features.bb_position;
        arr[6] = features.macd;
        arr[7] = features.vwap;
        arr[8] = features.volume_sma_20;
        arr[9] = features.volume_ratio;
        arr[10] = features.spread.unwrap_or(0.0);
        arr[11] = features.relative_spread.unwrap_or(0.0);
        arr[12] = features.price_change_1m;
        arr[13] = features.price_change_5m;
        arr[14] = features.volatility;
        arr[15] = features.skewness;
        arr[16] = features.kurtosis;
        arr[17] = features.bid.unwrap_or(features.price);
        arr[18] = features.ask.unwrap_or(features.price);
        
        arr
    }
    
    /// Calculate Simple Moving Average
    fn calculate_sma(&self, data: &VecDeque<f32>, window: usize) -> f32 {
        if data.len() < window {
            return data.back().copied().unwrap_or(0.0);
        }
        
        let sum: f32 = data.iter().rev().take(window).sum();
        sum / window as f32
    }
    
    /// Calculate Exponential Moving Average
    fn calculate_ema(&self, data: &VecDeque<f32>, window: usize) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        
        let alpha = 2.0 / (window as f32 + 1.0);
        let mut ema = data[0];
        
        for &price in data.iter().skip(1) {
            ema = alpha * price + (1.0 - alpha) * ema;
        }
        
        ema
    }
    
    /// Calculate Relative Strength Index
    fn calculate_rsi(&self, data: &VecDeque<f32>, window: usize) -> f32 {
        if data.len() < window + 1 {
            return 50.0;  // Neutral RSI
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in (data.len() - window)..data.len() {
            let change = data[i] - data[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / window as f32;
        let avg_loss = losses / window as f32;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    /// Calculate position within Bollinger Bands
    fn calculate_bollinger_position(&self, data: &VecDeque<f32>, window: usize) -> f32 {
        if data.len() < window {
            return 0.5;  // Middle of bands
        }
        
        let sma = self.calculate_sma(data, window);
        let recent_data: Vec<f32> = data.iter().rev().take(window).copied().collect();
        let variance: f32 = recent_data
            .iter()
            .map(|&x| (x - sma).powi(2))
            .sum::<f32>() / window as f32;
        let std_dev = variance.sqrt();
        
        let upper_band = sma + 2.0 * std_dev;
        let lower_band = sma - 2.0 * std_dev;
        
        let current_price = data.back().copied().unwrap_or(sma);
        
        if upper_band == lower_band {
            return 0.5;
        }
        
        (current_price - lower_band) / (upper_band - lower_band)
    }
    
    /// Calculate MACD
    fn calculate_macd(&self, data: &VecDeque<f32>) -> f32 {
        let ema_12 = self.calculate_ema(data, 12);
        let ema_26 = self.calculate_ema(data, 26);
        ema_12 - ema_26
    }
    
    /// Calculate volatility (standard deviation of returns)
    fn calculate_volatility(&self, data: &VecDeque<f32>) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let returns: Vec<f32> = data
            .iter()
            .zip(data.iter().skip(1))
            .map(|(prev, curr)| (curr - prev) / prev)
            .collect();
        
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f32>() / returns.len() as f32;
        let variance = returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f32>() / returns.len() as f32;
        
        variance.sqrt()
    }
    
    /// Calculate skewness
    fn calculate_skewness(&self, data: &VecDeque<f32>) -> f32 {
        if data.len() < 3 {
            return 0.0;
        }
        
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        let std_dev = self.calculate_std_dev(data, mean);
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let sum_cubed = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f32>();
        
        sum_cubed / n
    }
    
    /// Calculate kurtosis
    fn calculate_kurtosis(&self, data: &VecDeque<f32>) -> f32 {
        if data.len() < 4 {
            return 0.0;
        }
        
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        let std_dev = self.calculate_std_dev(data, mean);
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let sum_fourth = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f32>();
        
        (sum_fourth / n) - 3.0  // Excess kurtosis
    }
    
    /// Helper to calculate standard deviation
    fn calculate_std_dev(&self, data: &VecDeque<f32>, mean: f32) -> f32 {
        let variance = data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        
        variance.sqrt()
    }
}

/// Batch feature extraction for training
pub fn extract_features_batch(
    prices: &[f32],
    volumes: &[f32],
    window_size: usize,
    sequence_length: usize,
) -> Result<Array2<f32>> {
    if prices.len() != volumes.len() {
        return Err(WhaleMLError::InvalidDimensions {
            expected: format!("prices.len() == volumes.len()"),
            actual: format!("prices: {}, volumes: {}", prices.len(), volumes.len()),
        });
    }
    
    let mut extractor = FeatureExtractor::new(window_size);
    let mut all_features = Vec::new();
    
    // Extract features for each time step
    for i in 0..prices.len() {
        let features = extractor.extract_features(
            prices[i],
            volumes[i],
            None,  // No bid/ask in this simplified version
            None,
        )?;
        
        let feature_array = FeatureExtractor::features_to_array(&features);
        all_features.push(feature_array);
    }
    
    // Create sequences
    let num_sequences = all_features.len().saturating_sub(sequence_length);
    if num_sequences == 0 {
        return Err(WhaleMLError::InvalidDimensions {
            expected: format!("data length > sequence_length ({})", sequence_length),
            actual: format!("data length: {}", all_features.len()),
        });
    }
    
    let feature_dim = 19;
    let mut sequences = Array2::zeros((num_sequences, sequence_length * feature_dim));
    
    for i in 0..num_sequences {
        for j in 0..sequence_length {
            let start_idx = j * feature_dim;
            let end_idx = start_idx + feature_dim;
            sequences
                .slice_mut(ndarray::s![i, start_idx..end_idx])
                .assign(&all_features[i + j]);
        }
    }
    
    Ok(sequences)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_extraction() {
        let mut extractor = FeatureExtractor::new(20);
        
        // Generate some test data
        for i in 0..50 {
            let price = 45000.0 + (i as f32 * 10.0);
            let volume = 1000000.0 + (i as f32 * 1000.0);
            
            let features = extractor.extract_features(price, volume, None, None);
            assert!(features.is_ok());
            
            let features = features.unwrap();
            assert!(features.price > 0.0);
            assert!(features.volume > 0.0);
        }
    }
    
    #[test]
    fn test_batch_extraction() {
        let prices: Vec<f32> = (0..100).map(|i| 45000.0 + (i as f32 * 10.0)).collect();
        let volumes: Vec<f32> = (0..100).map(|i| 1000000.0 + (i as f32 * 1000.0)).collect();
        
        let result = extract_features_batch(&prices, &volumes, 20, 10);
        assert!(result.is_ok());
        
        let sequences = result.unwrap();
        assert_eq!(sequences.shape()[0], 90);  // 100 - 10
        assert_eq!(sequences.shape()[1], 10 * 19);  // sequence_length * features
    }
}