//! Financial applications of NHITS (Neural Hierarchical Interpolation for Time Series)
//! 
//! This module provides comprehensive financial forecasting capabilities using the
//! NHITS architecture enhanced with consciousness-aware mechanisms for superior
//! market prediction and risk management.

pub mod price_prediction;
pub mod volatility_modeling;
pub mod portfolio_optimization;
pub mod risk_metrics;
pub mod market_regime;
pub mod algorithmic_trading;
pub mod crypto_forecasting;
pub mod derivatives_pricing;
pub mod real_market_data;

use crate::ml::nhits::model::NHITSModel;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Common financial data structures and utilities
#[derive(Debug, Clone)]
pub struct FinancialTimeSeries {
    pub timestamps: Vec<i64>,
    pub open: Vec<f32>,
    pub high: Vec<f32>,
    pub low: Vec<f32>,
    pub close: Vec<f32>,
    pub volume: Vec<f32>,
    pub symbol: String,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub price_series: HashMap<String, FinancialTimeSeries>,
    pub market_indices: HashMap<String, FinancialTimeSeries>,
    pub economic_indicators: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub var_95: f32,
    pub var_99: f32,
    pub expected_shortfall: f32,
    pub max_drawdown: f32,
    pub sharpe_ratio: f32,
    pub sortino_ratio: f32,
    pub calmar_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub symbol: String,
    pub signal_type: SignalType,
    pub strength: f32,
    pub confidence: f32,
    pub timestamp: i64,
    pub target_price: Option<f32>,
    pub stop_loss: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    StrongBuy,
    StrongSell,
}

/// Consciousness-aware financial forecasting trait
pub trait ConsciousnessAwareForecasting {
    /// Predict with consciousness state awareness
    fn predict_conscious(&self, data: &Array2<f32>, consciousness_state: f32) -> Array2<f32>;
    
    /// Calculate prediction confidence based on consciousness coherence
    fn prediction_confidence(&self, prediction: &Array2<f32>, consciousness_state: f32) -> f32;
    
    /// Adapt model based on market consciousness patterns
    fn adapt_to_market_consciousness(&mut self, market_sentiment: f32, volatility_regime: f32);
}

/// Enhanced NHITS model for financial applications
pub struct FinancialNHITS {
    pub base_model: NHITSModel,
    pub consciousness_threshold: f32,
    pub market_regime_detector: Option<market_regime::MarketRegimeDetector>,
    pub risk_manager: Option<risk_metrics::RiskManager>,
}

impl FinancialNHITS {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_stacks: usize,
        num_blocks: usize,
        forecast_horizon: usize,
    ) -> Self {
        Self {
            base_model: NHITSModel::new(
                input_dim,
                hidden_dim,
                num_stacks,
                num_blocks,
                forecast_horizon,
            ),
            consciousness_threshold: 0.7,
            market_regime_detector: None,
            risk_manager: None,
        }
    }
    
    /// Initialize with financial-specific components
    pub fn with_financial_components(mut self) -> Self {
        self.market_regime_detector = Some(market_regime::MarketRegimeDetector::new());
        self.risk_manager = Some(risk_metrics::RiskManager::new());
        self
    }
}

impl ConsciousnessAwareForecasting for FinancialNHITS {
    fn predict_conscious(&self, data: &Array2<f32>, consciousness_state: f32) -> Array2<f32> {
        let base_prediction = self.base_model.forward(data);
        
        // Apply consciousness-aware adjustments
        if consciousness_state > self.consciousness_threshold {
            // High consciousness: More confident, stable predictions
            base_prediction * (1.0 + consciousness_state * 0.1)
        } else {
            // Low consciousness: More conservative, uncertainty-aware predictions
            base_prediction * (0.9 + consciousness_state * 0.1)
        }
    }
    
    fn prediction_confidence(&self, prediction: &Array2<f32>, consciousness_state: f32) -> f32 {
        let prediction_std = prediction.std(0.0);
        let base_confidence = 1.0 / (1.0 + prediction_std);
        
        // Consciousness state affects confidence calibration
        base_confidence * (0.5 + consciousness_state * 0.5)
    }
    
    fn adapt_to_market_consciousness(&mut self, market_sentiment: f32, volatility_regime: f32) {
        // Adjust consciousness threshold based on market conditions
        self.consciousness_threshold = 0.7 - (volatility_regime * 0.2);
        
        // Update model parameters based on market consciousness
        if market_sentiment < -0.5 {
            // Fear regime: Increase conservatism
            self.consciousness_threshold += 0.1;
        } else if market_sentiment > 0.5 {
            // Greed regime: Maintain vigilance
            self.consciousness_threshold += 0.05;
        }
    }
}

/// Utility functions for financial data processing
pub mod utils {
    use super::*;
    
    /// Calculate returns from price series
    pub fn calculate_returns(prices: &[f32]) -> Vec<f32> {
        prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
    
    /// Calculate log returns
    pub fn calculate_log_returns(prices: &[f32]) -> Vec<f32> {
        prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }
    
    /// Calculate rolling volatility
    pub fn rolling_volatility(returns: &[f32], window: usize) -> Vec<f32> {
        returns.windows(window)
            .map(|w| {
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                let variance = w.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>() / (w.len() - 1) as f32;
                variance.sqrt()
            })
            .collect()
    }
    
    /// Convert OHLCV to technical indicators
    pub fn ohlcv_to_features(series: &FinancialTimeSeries) -> Array2<f32> {
        let n = series.close.len();
        let mut features = Array2::zeros((n, 10));
        
        for i in 0..n {
            features[[i, 0]] = series.open[i];
            features[[i, 1]] = series.high[i];
            features[[i, 2]] = series.low[i];
            features[[i, 3]] = series.close[i];
            features[[i, 4]] = series.volume[i];
            
            if i > 0 {
                // Returns
                features[[i, 5]] = (series.close[i] - series.close[i-1]) / series.close[i-1];
                // High-Low spread
                features[[i, 6]] = (series.high[i] - series.low[i]) / series.close[i];
                // Volume change
                features[[i, 7]] = (series.volume[i] - series.volume[i-1]) / series.volume[i-1];
            }
            
            // Simple moving averages (if enough data)
            if i >= 10 {
                let sma_10 = series.close[i-9..=i].iter().sum::<f32>() / 10.0;
                features[[i, 8]] = series.close[i] / sma_10;
            }
            
            if i >= 20 {
                let sma_20 = series.close[i-19..=i].iter().sum::<f32>() / 20.0;
                features[[i, 9]] = series.close[i] / sma_20;
            }
        }
        
        features
    }
}