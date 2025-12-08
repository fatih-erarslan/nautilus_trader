//! Tengri quantum trading strategy implementation

use crate::{BacktestConfig, MarketData, TradingSignal, SignalType, Result, BacktestError};
use crate::portfolio::Portfolio;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Tengri quantum trading strategy
pub struct TengriQuantumStrategy {
    config: BacktestConfig,
    lookback_window: usize,
    price_history: HashMap<String, Vec<f64>>,
}

impl TengriQuantumStrategy {
    pub fn new(config: &BacktestConfig) -> Self {
        Self {
            config: config.clone(),
            lookback_window: 50, // 50 periods for technical analysis
            price_history: HashMap::new(),
        }
    }
    
    pub async fn generate_signals(
        &mut self,
        timestamp: &DateTime<Utc>,
        market_data: &HashMap<String, MarketData>,
        quantum_signals: &[TradingSignal],
        portfolio: &Portfolio,
    ) -> Result<Vec<TradingSignal>> {
        let mut signals = Vec::new();
        
        // Update price history
        for (symbol, data) in market_data {
            let prices = self.price_history.entry(symbol.clone()).or_insert_with(Vec::new);
            prices.push(data.close);
            
            // Keep only lookback window
            if prices.len() > self.lookback_window {
                prices.remove(0);
            }
            
            // Generate signals if we have enough history
            if prices.len() >= 20 {
                let signal = self.analyze_symbol(timestamp, symbol, data, quantum_signals, portfolio).await?;
                if let Some(s) = signal {
                    signals.push(s);
                }
            }
        }
        
        Ok(signals)
    }
    
    async fn analyze_symbol(
        &self,
        timestamp: &DateTime<Utc>,
        symbol: &str,
        data: &MarketData,
        quantum_signals: &[TradingSignal],
        portfolio: &Portfolio,
    ) -> Result<Option<TradingSignal>> {
        let prices = &self.price_history[symbol];
        
        // Calculate technical indicators
        let sma_20 = self.calculate_sma(prices, 20);
        let sma_50 = self.calculate_sma(prices, 50);
        let rsi = self.calculate_rsi(prices, 14);
        let macd = self.calculate_macd(prices);
        
        // Find corresponding quantum signal
        let quantum_signal = quantum_signals.iter()
            .find(|s| s.symbol == symbol);
        
        // Combined signal logic
        let mut signal_strength = 0.0;
        let mut confidence = 0.5;
        
        // Technical analysis component
        if data.close > sma_20 && sma_20 > sma_50 {
            signal_strength += 0.3; // Bullish trend
        } else if data.close < sma_20 && sma_20 < sma_50 {
            signal_strength -= 0.3; // Bearish trend
        }
        
        if rsi < 30.0 {
            signal_strength += 0.2; // Oversold
        } else if rsi > 70.0 {
            signal_strength -= 0.2; // Overbought
        }
        
        if macd.0 > macd.1 {
            signal_strength += 0.1; // MACD bullish
        } else {
            signal_strength -= 0.1; // MACD bearish
        }
        
        // Quantum signal component (higher weight if available)
        if let Some(q_signal) = quantum_signal {
            let quantum_weight = q_signal.confidence * 0.6; // Up to 60% weight
            match q_signal.signal_type {
                SignalType::StrongBuy => signal_strength += quantum_weight,
                SignalType::Buy => signal_strength += quantum_weight * 0.5,
                SignalType::Sell => signal_strength -= quantum_weight * 0.5,
                SignalType::StrongSell => signal_strength -= quantum_weight,
                SignalType::Hold => {} // No change
            }
            confidence = q_signal.confidence;
        }
        
        // Risk management filters
        let current_position = portfolio.get_position(symbol);
        let position_size = current_position.map(|p| p.quantity.abs()).unwrap_or(0.0);
        let max_position = self.config.initial_capital * self.config.risk_management.max_position_size;
        
        // Generate signal based on strength and position limits
        let signal_type = if signal_strength > 0.4 && position_size < max_position {
            if signal_strength > 0.7 {
                SignalType::StrongBuy
            } else {
                SignalType::Buy
            }
        } else if signal_strength < -0.4 && position_size > 0.0 {
            if signal_strength < -0.7 {
                SignalType::StrongSell
            } else {
                SignalType::Sell
            }
        } else {
            SignalType::Hold
        };
        
        if matches!(signal_type, SignalType::Hold) {
            return Ok(None);
        }
        
        // Calculate price targets and stop losses
        let volatility = self.calculate_volatility(prices, 20);
        let price_target = match signal_type {
            SignalType::StrongBuy | SignalType::Buy => {
                Some(data.close * (1.0 + volatility * 2.0))
            },
            SignalType::StrongSell | SignalType::Sell => {
                Some(data.close * (1.0 - volatility * 2.0))
            },
            _ => None,
        };
        
        let stop_loss = match signal_type {
            SignalType::StrongBuy | SignalType::Buy => {
                Some(data.close * (1.0 - self.config.risk_management.stop_loss))
            },
            SignalType::StrongSell | SignalType::Sell => {
                Some(data.close * (1.0 + self.config.risk_management.stop_loss))
            },
            _ => None,
        };
        
        Ok(Some(TradingSignal {
            timestamp: *timestamp,
            symbol: symbol.to_string(),
            signal_type,
            confidence,
            quantum_patterns: quantum_signal.map(|s| s.quantum_patterns.clone()).unwrap_or_default(),
            price_target,
            stop_loss,
        }))
    }
    
    fn calculate_sma(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }
        
        prices.iter().rev().take(period).sum::<f64>() / period as f64
    }
    
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI
        }
        
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
        for i in 1..=period {
            let change = prices[prices.len() - i] - prices[prices.len() - i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }
        
        let avg_gain = gains.iter().sum::<f64>() / period as f64;
        let avg_loss = losses.iter().sum::<f64>() / period as f64;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    fn calculate_macd(&self, prices: &[f64]) -> (f64, f64) {
        let ema_12 = self.calculate_ema(prices, 12);
        let ema_26 = self.calculate_ema(prices, 26);
        let macd_line = ema_12 - ema_26;
        
        // For simplicity, use a basic signal line calculation
        let signal_line = macd_line * 0.9; // Simplified signal line
        
        (macd_line, signal_line)
    }
    
    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        
        if prices.len() <= period {
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];
        
        for &price in prices.iter().skip(1) {
            ema = (price * multiplier) + (ema * (1.0 - multiplier));
        }
        
        ema
    }
    
    fn calculate_volatility(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return 0.02; // Default 2% volatility
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(period).cloned().collect();
        let returns: Vec<f64> = recent_prices.windows(2)
            .map(|w| (w[0] - w[1]) / w[1])
            .collect();
        
        if returns.is_empty() {
            return 0.02;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }
}