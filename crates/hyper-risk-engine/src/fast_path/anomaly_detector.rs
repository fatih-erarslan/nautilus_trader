//! Fast anomaly detection for the pre-trade path.
//!
//! Uses a simplified autoencoder-style approach optimized for
//! low-latency inference.

use crate::core::types::{Order, Portfolio};

/// Anomaly detection configuration.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Number of input features.
    pub num_features: usize,
    /// Anomaly threshold (0-1).
    pub threshold: f64,
    /// Use adaptive threshold.
    pub adaptive: bool,
    /// EWMA decay for statistics.
    pub ewma_alpha: f64,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            num_features: 8,
            threshold: 0.7,
            adaptive: true,
            ewma_alpha: 0.01,
        }
    }
}

/// Anomaly score result.
#[derive(Debug, Clone)]
pub struct AnomalyScore {
    /// Overall anomaly score (0-1).
    pub score: f64,
    /// Is this classified as an anomaly?
    pub is_anomaly: bool,
    /// Individual feature contributions.
    pub feature_scores: Vec<f64>,
    /// Which features are most anomalous.
    pub top_features: Vec<usize>,
}

/// Fast anomaly detector.
#[derive(Debug)]
pub struct FastAnomalyDetector {
    /// Configuration.
    config: AnomalyConfig,
    /// Running mean of features.
    feature_mean: Vec<f64>,
    /// Running variance of features.
    feature_var: Vec<f64>,
    /// Observation count.
    count: u64,
}

impl FastAnomalyDetector {
    /// Create new anomaly detector.
    pub fn new(config: AnomalyConfig) -> Self {
        let n = config.num_features;
        Self {
            config,
            feature_mean: vec![0.0; n],
            feature_var: vec![1.0; n],
            count: 0,
        }
    }

    /// Score an order for anomalies.
    ///
    /// # Features
    /// 1. Order size (normalized)
    /// 2. Order size vs average position
    /// 3. Order size vs portfolio
    /// 4. Time since last order (if tracked)
    /// 5. Symbol activity level
    /// 6. Side imbalance
    /// 7. Drawdown level
    /// 8. Volatility regime
    #[inline]
    pub fn score(&self, order: &Order, portfolio: &Portfolio) -> AnomalyScore {
        let features = self.extract_features(order, portfolio);

        // Calculate z-scores
        let mut feature_scores: Vec<f64> = features
            .iter()
            .zip(self.feature_mean.iter())
            .zip(self.feature_var.iter())
            .map(|((f, m), v)| {
                if *v > 1e-8 {
                    ((f - m) / v.sqrt()).abs()
                } else {
                    (f - m).abs()
                }
            })
            .collect();

        // Cap extreme values
        for s in feature_scores.iter_mut() {
            *s = s.min(5.0); // Cap at 5 standard deviations
        }

        // Overall score: RMS of z-scores, normalized to 0-1
        let rms = (feature_scores.iter().map(|s| s * s).sum::<f64>()
            / feature_scores.len() as f64)
            .sqrt();

        // Convert to 0-1 using sigmoid-like transform
        let score = 1.0 - (-rms / 2.0).exp();

        // Find top anomalous features
        let mut indexed: Vec<(usize, f64)> = feature_scores.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_features: Vec<usize> = indexed.iter().take(3).map(|(i, _)| *i).collect();

        let threshold = if self.config.adaptive {
            self.adaptive_threshold()
        } else {
            self.config.threshold
        };

        AnomalyScore {
            score,
            is_anomaly: score > threshold,
            feature_scores,
            top_features,
        }
    }

    /// Extract features from order and portfolio.
    #[inline]
    fn extract_features(&self, order: &Order, portfolio: &Portfolio) -> Vec<f64> {
        let order_value = order.quantity.as_f64().abs();

        // Average position size
        let avg_pos_size = if portfolio.positions.is_empty() {
            order_value
        } else {
            portfolio.positions.iter()
                .map(|p| p.quantity.as_f64().abs())
                .sum::<f64>()
                / portfolio.positions.len() as f64
        };

        // Normalized features
        let f1 = order_value / 10000.0; // Order size (normalized to 10k)
        let f2 = if avg_pos_size > 0.0 {
            order_value / avg_pos_size
        } else {
            1.0
        };
        let f3 = if portfolio.total_value > 0.0 {
            order_value / portfolio.total_value * 100.0
        } else {
            0.0
        };
        let f4 = 0.5; // Placeholder for time feature
        let f5 = 0.5; // Placeholder for symbol activity
        let f6 = if order.side == crate::core::types::OrderSide::Buy { 1.0 } else { -1.0 };
        let f7 = portfolio.drawdown_pct() / 10.0; // Normalized drawdown
        let f8 = 0.5; // Placeholder for volatility regime

        vec![f1, f2, f3, f4, f5, f6, f7, f8]
    }

    /// Calculate adaptive threshold.
    fn adaptive_threshold(&self) -> f64 {
        // Start conservative, relax as we see more data
        let base = self.config.threshold;
        let adjustment = (-0.001 * self.count as f64).exp() * 0.2;
        base + adjustment
    }

    /// Update statistics with new observation.
    pub fn update(&mut self, order: &Order, portfolio: &Portfolio) {
        let features = self.extract_features(order, portfolio);
        let alpha = self.config.ewma_alpha;

        for (i, f) in features.iter().enumerate() {
            if i < self.feature_mean.len() {
                // Update mean
                let delta = f - self.feature_mean[i];
                self.feature_mean[i] += alpha * delta;

                // Update variance
                let delta2 = (f - self.feature_mean[i]).powi(2);
                self.feature_var[i] = (1.0 - alpha) * self.feature_var[i] + alpha * delta2;
            }
        }

        self.count += 1;
    }

    /// Get observation count.
    pub fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Symbol, OrderSide, Quantity, Timestamp};

    fn create_order(quantity: f64) -> Order {
        Order {
            symbol: Symbol::new("TEST"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(quantity),
            limit_price: None,
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let config = AnomalyConfig::default();
        let detector = FastAnomalyDetector::new(config);
        assert_eq!(detector.count(), 0);
    }

    #[test]
    fn test_normal_order_score() {
        let config = AnomalyConfig::default();
        let detector = FastAnomalyDetector::new(config);

        let order = create_order(100.0);
        let portfolio = Portfolio::new(100_000.0);

        let score = detector.score(&order, &portfolio);

        // Normal order should have low score
        assert!(score.score < 0.5);
        assert!(!score.is_anomaly);
    }

    #[test]
    fn test_extreme_order_score() {
        let config = AnomalyConfig {
            threshold: 0.5,
            ..Default::default()
        };
        let mut detector = FastAnomalyDetector::new(config);

        // Train on small orders
        let portfolio = Portfolio::new(100_000.0);
        for _ in 0..100 {
            let order = create_order(100.0);
            detector.update(&order, &portfolio);
        }

        // Now test with very large order
        let large_order = create_order(50000.0);
        let score = detector.score(&large_order, &portfolio);

        // Large order should have higher score
        assert!(score.score > 0.3);
    }

    #[test]
    fn test_feature_extraction() {
        let config = AnomalyConfig::default();
        let detector = FastAnomalyDetector::new(config);

        let order = create_order(100.0);
        let portfolio = Portfolio::new(100_000.0);

        let score = detector.score(&order, &portfolio);

        // Should have 8 feature scores
        assert_eq!(score.feature_scores.len(), 8);
        // Should have top 3 features
        assert_eq!(score.top_features.len(), 3);
    }
}
