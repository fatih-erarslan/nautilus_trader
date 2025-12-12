//! Market data structures and utilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub timestamps: Option<Vec<i64>>,
}

impl MarketData {
    pub fn new(prices: Vec<f64>, volumes: Vec<f64>, highs: Vec<f64>, lows: Vec<f64>) -> Self {
        Self {
            prices,
            volumes,
            highs,
            lows,
            timestamps: None,
        }
    }

    pub fn with_timestamps(mut self, timestamps: Vec<i64>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }

    pub fn len(&self) -> usize {
        self.prices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    pub fn returns(&self) -> Vec<f64> {
        self.prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    pub fn log_returns(&self) -> Vec<f64> {
        self.prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    pub fn volatility(&self, window: usize) -> Vec<f64> {
        let returns = self.returns();
        let mut volatilities = Vec::new();
        
        for i in window..returns.len() {
            let window_returns = &returns[i-window..i];
            let mean = window_returns.iter().sum::<f64>() / window as f64;
            let variance = window_returns.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f64>() / (window - 1) as f64;
            volatilities.push(variance.sqrt());
        }
        
        volatilities
    }

    pub fn sma(&self, window: usize) -> Vec<f64> {
        self.prices
            .windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    pub fn ema(&self, window: usize) -> Vec<f64> {
        let alpha = 2.0 / (window as f64 + 1.0);
        let mut ema = Vec::with_capacity(self.prices.len());
        
        if !self.prices.is_empty() {
            ema.push(self.prices[0]);
            
            for &price in &self.prices[1..] {
                let prev_ema = ema.last().unwrap();
                ema.push(alpha * price + (1.0 - alpha) * prev_ema);
            }
        }
        
        ema
    }

    pub fn true_range(&self) -> Vec<f64> {
        let mut tr = Vec::new();
        
        for i in 1..self.len() {
            let high_low = self.highs[i] - self.lows[i];
            let high_close = (self.highs[i] - self.prices[i-1]).abs();
            let low_close = (self.lows[i] - self.prices[i-1]).abs();
            
            tr.push(high_low.max(high_close).max(low_close));
        }
        
        tr
    }

    pub fn atr(&self, window: usize) -> Vec<f64> {
        let tr = self.true_range();
        let alpha = 1.0 / window as f64;
        let mut atr = Vec::new();
        
        if !tr.is_empty() {
            atr.push(tr[0]);
            
            for &tr_val in &tr[1..] {
                let prev_atr = atr.last().unwrap();
                atr.push(alpha * tr_val + (1.0 - alpha) * prev_atr);
            }
        }
        
        atr
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub data: HashMap<String, Vec<f64>>,
    pub timestamps: Vec<i64>,
}

impl TimeSeriesData {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            timestamps: Vec::new(),
        }
    }

    pub fn add_series(&mut self, name: String, values: Vec<f64>) {
        self.data.insert(name, values);
    }

    pub fn get_series(&self, name: &str) -> Option<&Vec<f64>> {
        self.data.get(name)
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
}