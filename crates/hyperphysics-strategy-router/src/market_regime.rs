//! Market regime detection for adaptive routing
//!
//! Identifies market conditions to select appropriate trading strategies.

// Error types available for future extension
#[allow(unused_imports)]
use crate::{RouterError, Result};
use serde::{Deserialize, Serialize};

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Low volatility, trending market
    Trending,
    /// High volatility, mean-reverting
    MeanReverting,
    /// Very high volatility, crisis conditions
    HighVolatility,
    /// Low activity, range-bound
    RangeBound,
    /// Transition state between regimes
    Transitional,
}

impl MarketRegime {
    /// Get recommended expert weights for this regime
    pub fn expert_bias(&self) -> Vec<f64> {
        match self {
            MarketRegime::Trending => vec![1.0, 0.2, 0.1, 0.5, 0.1, 0.3],
            MarketRegime::MeanReverting => vec![0.2, 1.0, 0.1, 0.1, 0.8, 0.3],
            MarketRegime::HighVolatility => vec![0.1, 0.1, 0.3, 0.2, 0.3, 1.0],
            MarketRegime::RangeBound => vec![0.3, 0.5, 0.2, 0.3, 0.4, 0.2],
            MarketRegime::Transitional => vec![0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        }
    }
}

/// Configuration for regime detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Lookback window for volatility calculation
    pub volatility_window: usize,
    /// Lookback window for trend detection
    pub trend_window: usize,
    /// Threshold for high volatility regime
    pub high_vol_threshold: f64,
    /// Threshold for trending regime
    pub trend_threshold: f64,
    /// Threshold for range-bound detection
    pub range_threshold: f64,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            volatility_window: 20,
            trend_window: 50,
            high_vol_threshold: 0.03, // 3% daily vol
            trend_threshold: 0.5,     // R² for trend
            range_threshold: 0.02,    // 2% range
        }
    }
}

/// Market regime detector
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    config: RegimeConfig,
    /// Current detected regime
    current_regime: MarketRegime,
    /// Confidence in current regime
    confidence: f64,
    /// Recent returns for analysis
    returns_buffer: Vec<f64>,
}

impl RegimeDetector {
    /// Create new regime detector
    pub fn new(config: RegimeConfig) -> Self {
        Self {
            config,
            current_regime: MarketRegime::Transitional,
            confidence: 0.0,
            returns_buffer: Vec::new(),
        }
    }

    /// Update with new return observation
    pub fn update(&mut self, return_: f64) {
        self.returns_buffer.push(return_);

        // Keep buffer bounded
        let max_size = self.config.volatility_window.max(self.config.trend_window);
        if self.returns_buffer.len() > max_size * 2 {
            self.returns_buffer.drain(0..max_size);
        }

        // Detect regime if we have enough data
        if self.returns_buffer.len() >= self.config.volatility_window {
            self.detect_regime();
        }
    }

    /// Detect current market regime
    fn detect_regime(&mut self) {
        let vol = self.compute_volatility();
        let trend = self.compute_trend_strength();
        let range = self.compute_range();

        // Classification logic
        if vol > self.config.high_vol_threshold {
            self.current_regime = MarketRegime::HighVolatility;
            self.confidence = (vol / self.config.high_vol_threshold).min(1.0);
        } else if trend > self.config.trend_threshold {
            self.current_regime = MarketRegime::Trending;
            self.confidence = trend;
        } else if range < self.config.range_threshold {
            self.current_regime = MarketRegime::RangeBound;
            self.confidence = 1.0 - (range / self.config.range_threshold);
        } else if vol < self.config.high_vol_threshold / 2.0 {
            self.current_regime = MarketRegime::MeanReverting;
            self.confidence = 0.5 + (1.0 - vol / self.config.high_vol_threshold) * 0.5;
        } else {
            self.current_regime = MarketRegime::Transitional;
            self.confidence = 0.3;
        }
    }

    /// Compute realized volatility
    fn compute_volatility(&self) -> f64 {
        if self.returns_buffer.len() < 2 {
            return 0.0;
        }

        let n = self.config.volatility_window.min(self.returns_buffer.len());
        let recent: Vec<f64> = self.returns_buffer.iter().rev().take(n).copied().collect();

        let mean: f64 = recent.iter().sum::<f64>() / (n as f64);
        let variance: f64 = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n as f64);

        variance.sqrt()
    }

    /// Compute trend strength using R²
    fn compute_trend_strength(&self) -> f64 {
        if self.returns_buffer.len() < 3 {
            return 0.0;
        }

        let n = self.config.trend_window.min(self.returns_buffer.len());
        let recent: Vec<f64> = self.returns_buffer.iter().rev().take(n).copied().collect();

        // Cumulative returns
        let mut cumulative = Vec::with_capacity(n);
        let mut sum = 0.0;
        for r in recent.iter().rev() {
            sum += r;
            cumulative.push(sum);
        }

        // Linear regression R²
        let n_f = n as f64;
        let x_mean = (n_f - 1.0) / 2.0;
        let y_mean: f64 = cumulative.iter().sum::<f64>() / n_f;

        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        let mut ss_yy = 0.0;

        for (i, &y) in cumulative.iter().enumerate() {
            let x = i as f64;
            ss_xy += (x - x_mean) * (y - y_mean);
            ss_xx += (x - x_mean).powi(2);
            ss_yy += (y - y_mean).powi(2);
        }

        if ss_xx < 1e-10 || ss_yy < 1e-10 {
            return 0.0;
        }

        (ss_xy.powi(2) / (ss_xx * ss_yy)).min(1.0)
    }

    /// Compute price range
    fn compute_range(&self) -> f64 {
        if self.returns_buffer.is_empty() {
            return 0.0;
        }

        let n = self.config.volatility_window.min(self.returns_buffer.len());
        let recent: Vec<f64> = self.returns_buffer.iter().rev().take(n).copied().collect();

        // Cumulative returns
        let mut cumulative = Vec::with_capacity(n);
        let mut sum = 0.0;
        for r in recent.iter().rev() {
            sum += r;
            cumulative.push(sum);
        }

        let max = cumulative.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = cumulative.iter().cloned().fold(f64::INFINITY, f64::min);

        max - min
    }

    /// Get current regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get confidence in current regime
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Get expert bias for current regime
    pub fn expert_bias(&self) -> Vec<f64> {
        let base_bias = self.current_regime.expert_bias();

        // Scale by confidence
        base_bias.iter().map(|&b| b * self.confidence).collect()
    }

    /// Clear history
    pub fn reset(&mut self) {
        self.returns_buffer.clear();
        self.current_regime = MarketRegime::Transitional;
        self.confidence = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_detector_creation() {
        let detector = RegimeDetector::new(RegimeConfig::default());
        assert_eq!(detector.current_regime(), MarketRegime::Transitional);
    }

    #[test]
    fn test_trending_detection() {
        let mut detector = RegimeDetector::new(RegimeConfig {
            volatility_window: 5,
            trend_window: 5,
            ..Default::default()
        });

        // Feed consistent positive returns
        for _ in 0..20 {
            detector.update(0.01); // 1% daily return
        }

        // Should detect trending
        assert_eq!(detector.current_regime(), MarketRegime::Trending);
    }

    #[test]
    fn test_high_volatility_detection() {
        let mut detector = RegimeDetector::new(RegimeConfig {
            volatility_window: 5,
            high_vol_threshold: 0.02,
            ..Default::default()
        });

        // Feed alternating large returns
        for i in 0..20 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(0.05 * sign);
        }

        // Should detect high volatility
        assert_eq!(detector.current_regime(), MarketRegime::HighVolatility);
    }

    #[test]
    fn test_expert_bias() {
        let regime = MarketRegime::Trending;
        let bias = regime.expert_bias();

        assert!(!bias.is_empty());
        assert!(bias[0] > bias[1]); // Trending should favor first expert
    }
}
