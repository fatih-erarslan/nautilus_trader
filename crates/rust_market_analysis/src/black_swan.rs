//! Black Swan Event Detection
//! 
//! Detection of rare, extreme events with massive impact on financial markets.
//! Based on Nassim Taleb's work on extreme events and tail risk.

use crate::market_data::MarketData;
use anyhow::Result;
use std::collections::VecDeque;

pub struct BlackSwanDetector {
    tail_threshold: f64,
    extreme_volatility_multiplier: f64,
    correlation_breakdown_threshold: f64,
    volume_spike_threshold: f64,
    lookback_window: usize,
    event_memory: VecDeque<BlackSwanEvent>,
}

#[derive(Debug, Clone)]
pub struct BlackSwanMetrics {
    pub probability: f64,
    pub tail_risk_score: f64,
    pub volatility_regime_shift: f64,
    pub correlation_breakdown: f64,
    pub volume_anomaly: f64,
    pub market_stress_level: f64,
    pub fat_tail_indicator: f64,
    pub kurtosis_excess: f64,
    pub skew_extremity: f64,
    pub early_warning_signals: Vec<EarlyWarningSignal>,
}

#[derive(Debug, Clone)]
pub struct EarlyWarningSignal {
    pub signal_type: String,
    pub strength: f64,
    pub time_horizon: String, // "immediate", "short_term", "medium_term"
    pub description: String,
}

#[derive(Debug, Clone)]
struct BlackSwanEvent {
    timestamp: usize,
    magnitude: f64,
    duration: usize,
    event_type: SwanEventType,
    market_impact: f64,
}

#[derive(Debug, Clone)]
enum SwanEventType {
    PriceCollapse,
    VolatilityExplosion,
    LiquidityCrisis,
    CorrelationBreakdown,
    RegimeShift,
}

impl BlackSwanDetector {
    pub fn new() -> Self {
        Self {
            tail_threshold: 0.05,        // 5% extreme moves
            extreme_volatility_multiplier: 5.0,
            correlation_breakdown_threshold: 0.3,
            volume_spike_threshold: 10.0,
            lookback_window: 252,        // ~1 year of daily data
            event_memory: VecDeque::with_capacity(100),
        }
    }

    pub fn calculate_probability(&mut self, data: &MarketData) -> Result<f64> {
        if data.len() < 50 {
            return Ok(0.0);
        }

        let metrics = self.calculate_black_swan_metrics(data)?;
        self.update_event_memory(data, &metrics);

        // Weighted combination of different risk factors
        let probability = self.combine_risk_factors(&metrics);

        Ok(probability)
    }

    pub fn get_detailed_analysis(&mut self, data: &MarketData) -> Result<BlackSwanMetrics> {
        if data.len() < 50 {
            return Ok(BlackSwanMetrics::default());
        }

        let mut metrics = self.calculate_black_swan_metrics(data)?;
        metrics.early_warning_signals = self.generate_early_warning_signals(data, &metrics)?;

        Ok(metrics)
    }

    fn calculate_black_swan_metrics(&self, data: &MarketData) -> Result<BlackSwanMetrics> {
        let tail_risk_score = self.calculate_tail_risk(data)?;
        let volatility_regime_shift = self.detect_volatility_regime_shift(data)?;
        let correlation_breakdown = self.detect_correlation_breakdown(data)?;
        let volume_anomaly = self.detect_volume_anomalies(data)?;
        let market_stress_level = self.calculate_market_stress(data)?;
        let fat_tail_indicator = self.calculate_fat_tail_indicator(data)?;
        let kurtosis_excess = self.calculate_excess_kurtosis(data)?;
        let skew_extremity = self.calculate_skew_extremity(data)?;

        let probability = self.combine_risk_factors_detailed(
            tail_risk_score, volatility_regime_shift, correlation_breakdown,
            volume_anomaly, market_stress_level, fat_tail_indicator,
            kurtosis_excess, skew_extremity
        );

        Ok(BlackSwanMetrics {
            probability,
            tail_risk_score,
            volatility_regime_shift,
            correlation_breakdown,
            volume_anomaly,
            market_stress_level,
            fat_tail_indicator,
            kurtosis_excess,
            skew_extremity,
            early_warning_signals: Vec::new(), // Will be populated later
        })
    }

    /// Calculate tail risk using extreme value theory
    fn calculate_tail_risk(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 30 {
            return Ok(0.0);
        }

        // Sort returns to find tail distribution
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_returns.len();
        let tail_size = (n as f64 * self.tail_threshold) as usize;

        if tail_size == 0 {
            return Ok(0.0);
        }

        // Calculate tail risk score based on extreme negative returns
        let left_tail = &sorted_returns[..tail_size];
        let tail_mean = left_tail.iter().sum::<f64>() / tail_size as f64;
        let overall_mean = returns.iter().sum::<f64>() / returns.len() as f64;

        // Tail risk increases with more extreme negative tail
        let tail_deviation = (tail_mean - overall_mean).abs();
        let risk_score = (tail_deviation * 100.0).min(1.0); // Scale and cap

        Ok(risk_score)
    }

    /// Detect sudden regime shifts in volatility
    fn detect_volatility_regime_shift(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 40 {
            return Ok(0.0);
        }

        // Calculate rolling volatility with different windows
        let short_vol = self.calculate_rolling_volatility(&returns, 10);
        let long_vol = self.calculate_rolling_volatility(&returns, 30);

        if short_vol == 0.0 || long_vol == 0.0 {
            return Ok(0.0);
        }

        // Detect regime shift
        let volatility_ratio = short_vol / long_vol;
        let regime_shift_score = if volatility_ratio > self.extreme_volatility_multiplier {
            ((volatility_ratio - 1.0) / self.extreme_volatility_multiplier).min(1.0)
        } else {
            0.0
        };

        Ok(regime_shift_score)
    }

    /// Detect breakdown in normal correlations
    fn detect_correlation_breakdown(&self, data: &MarketData) -> Result<f64> {
        if data.len() < 60 {
            return Ok(0.0);
        }

        let returns = data.returns();
        let volumes = &data.volumes;

        // Calculate price-volume correlation (should be stable normally)
        let price_vol_corr = self.calculate_correlation(&returns, volumes);
        
        // Calculate rolling correlation stability
        let mut correlations = Vec::new();
        let window = 20;

        for i in window..returns.len() {
            let price_window = &returns[i-window..i];
            let volume_window = &volumes[i-window..i];
            let corr = self.calculate_correlation(price_window, volume_window);
            correlations.push(corr);
        }

        if correlations.is_empty() {
            return Ok(0.0);
        }

        // Calculate correlation stability
        let mean_corr = correlations.iter().sum::<f64>() / correlations.len() as f64;
        let corr_volatility = self.calculate_std(&correlations, mean_corr);

        // High correlation volatility indicates breakdown
        let breakdown_score = (corr_volatility * 5.0).min(1.0);

        Ok(breakdown_score)
    }

    /// Detect extreme volume anomalies
    fn detect_volume_anomalies(&self, data: &MarketData) -> Result<f64> {
        let volumes = &data.volumes;
        if volumes.len() < 20 {
            return Ok(0.0);
        }

        // Calculate volume z-scores
        let recent_volume = volumes[volumes.len() - 1];
        let window = 20.min(volumes.len());
        let historical_volumes = &volumes[volumes.len() - window..volumes.len() - 1];

        let mean_volume = historical_volumes.iter().sum::<f64>() / historical_volumes.len() as f64;
        let volume_std = self.calculate_std(historical_volumes, mean_volume);

        if volume_std == 0.0 {
            return Ok(0.0);
        }

        let z_score = (recent_volume - mean_volume) / volume_std;
        let anomaly_score = if z_score > self.volume_spike_threshold {
            (z_score / self.volume_spike_threshold / 2.0).min(1.0)
        } else {
            0.0
        };

        Ok(anomaly_score)
    }

    /// Calculate overall market stress level
    fn calculate_market_stress(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 30 {
            return Ok(0.0);
        }

        // Multiple stress indicators
        let volatility_stress = self.calculate_volatility_stress(&returns);
        let drawdown_stress = self.calculate_drawdown_stress(&data.prices);
        let momentum_stress = self.calculate_momentum_stress(&returns);

        // Combined stress score
        let stress_level = (
            volatility_stress * 0.4 +
            drawdown_stress * 0.3 +
            momentum_stress * 0.3
        ).min(1.0);

        Ok(stress_level)
    }

    /// Calculate fat tail indicator using power law distribution
    fn calculate_fat_tail_indicator(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 100 {
            return Ok(0.0);
        }

        // Sort absolute returns
        let mut abs_returns: Vec<f64> = returns.iter().map(|&r| r.abs()).collect();
        abs_returns.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Remove zeros
        abs_returns.retain(|&r| r > 1e-8);

        if abs_returns.len() < 20 {
            return Ok(0.0);
        }

        // Estimate tail exponent using Hill estimator
        let tail_size = (abs_returns.len() / 10).max(10); // Top 10% or at least 10 observations
        let hill_estimator = self.calculate_hill_estimator(&abs_returns, tail_size);

        // Fat tails have low tail exponent (< 4 for financial data)
        let fat_tail_score = if hill_estimator > 0.0 && hill_estimator < 4.0 {
            (4.0 - hill_estimator) / 4.0
        } else {
            0.0
        };

        Ok(fat_tail_score.min(1.0))
    }

    /// Calculate excess kurtosis (measure of tail heaviness)
    fn calculate_excess_kurtosis(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 30 {
            return Ok(0.0);
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        if variance <= 0.0 {
            return Ok(0.0);
        }

        let std_dev = variance.sqrt();
        let fourth_moment = returns.iter()
            .map(|&r| ((r - mean) / std_dev).powi(4))
            .sum::<f64>() / returns.len() as f64;

        // Excess kurtosis (normal distribution has kurtosis of 3)
        let excess_kurtosis = fourth_moment - 3.0;
        
        // Higher excess kurtosis indicates fatter tails
        let kurtosis_score = (excess_kurtosis / 10.0).max(0.0).min(1.0);

        Ok(kurtosis_score)
    }

    /// Calculate skewness extremity
    fn calculate_skew_extremity(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 30 {
            return Ok(0.0);
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        if variance <= 0.0 {
            return Ok(0.0);
        }

        let std_dev = variance.sqrt();
        let third_moment = returns.iter()
            .map(|&r| ((r - mean) / std_dev).powi(3))
            .sum::<f64>() / returns.len() as f64;

        // Extreme negative skew indicates crash risk
        let skew_extremity = if third_moment < 0.0 {
            (-third_moment / 3.0).min(1.0)
        } else {
            0.0
        };

        Ok(skew_extremity)
    }

    fn generate_early_warning_signals(&self, data: &MarketData, metrics: &BlackSwanMetrics) -> Result<Vec<EarlyWarningSignal>> {
        let mut signals = Vec::new();

        // Tail risk warning
        if metrics.tail_risk_score > 0.7 {
            signals.push(EarlyWarningSignal {
                signal_type: "extreme_tail_risk".to_string(),
                strength: metrics.tail_risk_score,
                time_horizon: "immediate".to_string(),
                description: "Extreme negative tail events detected".to_string(),
            });
        }

        // Volatility regime shift warning
        if metrics.volatility_regime_shift > 0.6 {
            signals.push(EarlyWarningSignal {
                signal_type: "volatility_explosion".to_string(),
                strength: metrics.volatility_regime_shift,
                time_horizon: "short_term".to_string(),
                description: "Sudden volatility regime shift detected".to_string(),
            });
        }

        // Market stress warning
        if metrics.market_stress_level > 0.8 {
            signals.push(EarlyWarningSignal {
                signal_type: "market_stress".to_string(),
                strength: metrics.market_stress_level,
                time_horizon: "immediate".to_string(),
                description: "Extreme market stress conditions".to_string(),
            });
        }

        // Fat tail warning
        if metrics.fat_tail_indicator > 0.5 {
            signals.push(EarlyWarningSignal {
                signal_type: "fat_tail_distribution".to_string(),
                strength: metrics.fat_tail_indicator,
                time_horizon: "medium_term".to_string(),
                description: "Distribution showing fat tail characteristics".to_string(),
            });
        }

        // Correlation breakdown warning
        if metrics.correlation_breakdown > 0.6 {
            signals.push(EarlyWarningSignal {
                signal_type: "correlation_breakdown".to_string(),
                strength: metrics.correlation_breakdown,
                time_horizon: "short_term".to_string(),
                description: "Normal market correlations breaking down".to_string(),
            });
        }

        Ok(signals)
    }

    fn combine_risk_factors(&self, metrics: &BlackSwanMetrics) -> f64 {
        // Weighted combination with emphasis on most dangerous signals
        let probability = (
            metrics.tail_risk_score * 0.25 +
            metrics.volatility_regime_shift * 0.20 +
            metrics.market_stress_level * 0.20 +
            metrics.fat_tail_indicator * 0.15 +
            metrics.correlation_breakdown * 0.10 +
            metrics.volume_anomaly * 0.05 +
            metrics.kurtosis_excess * 0.03 +
            metrics.skew_extremity * 0.02
        ).min(1.0);

        probability
    }

    fn combine_risk_factors_detailed(&self, tail_risk: f64, volatility_shift: f64, 
                                   correlation_breakdown: f64, volume_anomaly: f64,
                                   market_stress: f64, fat_tail: f64, 
                                   kurtosis: f64, skew: f64) -> f64 {
        (
            tail_risk * 0.25 +
            volatility_shift * 0.20 +
            market_stress * 0.20 +
            fat_tail * 0.15 +
            correlation_breakdown * 0.10 +
            volume_anomaly * 0.05 +
            kurtosis * 0.03 +
            skew * 0.02
        ).min(1.0)
    }

    fn update_event_memory(&mut self, data: &MarketData, metrics: &BlackSwanMetrics) {
        // Record significant events
        if metrics.probability > 0.8 {
            let event = BlackSwanEvent {
                timestamp: data.len() - 1,
                magnitude: metrics.probability,
                duration: 1,
                event_type: self.classify_event_type(metrics),
                market_impact: self.estimate_market_impact(data, metrics),
            };

            self.event_memory.push_back(event);

            // Keep memory manageable
            while self.event_memory.len() > 100 {
                self.event_memory.pop_front();
            }
        }
    }

    fn classify_event_type(&self, metrics: &BlackSwanMetrics) -> SwanEventType {
        if metrics.tail_risk_score > 0.8 {
            SwanEventType::PriceCollapse
        } else if metrics.volatility_regime_shift > 0.8 {
            SwanEventType::VolatilityExplosion
        } else if metrics.volume_anomaly > 0.8 {
            SwanEventType::LiquidityCrisis
        } else if metrics.correlation_breakdown > 0.8 {
            SwanEventType::CorrelationBreakdown
        } else {
            SwanEventType::RegimeShift
        }
    }

    fn estimate_market_impact(&self, data: &MarketData, metrics: &BlackSwanMetrics) -> f64 {
        // Estimate potential market impact based on current conditions
        let current_volatility = if data.len() > 20 {
            let returns = data.returns();
            let recent_returns = &returns[returns.len().saturating_sub(20)..];
            self.calculate_std(recent_returns, 0.0)
        } else {
            0.02
        };

        // Impact increases with metrics severity and current volatility
        (metrics.probability * current_volatility * 100.0).min(1.0)
    }

    // Helper methods
    fn calculate_rolling_volatility(&self, returns: &[f64], window: usize) -> f64 {
        if returns.len() < window {
            return 0.0;
        }

        let recent_returns = &returns[returns.len() - window..];
        let mean = recent_returns.iter().sum::<f64>() / window as f64;
        let variance = recent_returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (window - 1) as f64;

        variance.sqrt()
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_y2: f64 = y.iter().map(|&yi| yi * yi).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn calculate_std(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let actual_mean = if mean == 0.0 {
            values.iter().sum::<f64>() / values.len() as f64
        } else {
            mean
        };

        let variance = values.iter()
            .map(|&x| (x - actual_mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    fn calculate_hill_estimator(&self, sorted_data: &[f64], k: usize) -> f64 {
        if k >= sorted_data.len() || k == 0 {
            return 0.0;
        }

        let threshold = sorted_data[k - 1];
        let mut sum = 0.0;

        for i in 0..k {
            if sorted_data[i] > 0.0 && threshold > 0.0 {
                sum += (sorted_data[i] / threshold).ln();
            }
        }

        if sum > 0.0 {
            k as f64 / sum
        } else {
            0.0
        }
    }

    fn calculate_volatility_stress(&self, returns: &[f64]) -> f64 {
        let current_vol = self.calculate_rolling_volatility(returns, 10);
        let historical_vol = self.calculate_rolling_volatility(returns, 50);

        if historical_vol > 0.0 {
            (current_vol / historical_vol).min(5.0) / 5.0
        } else {
            0.0
        }
    }

    fn calculate_drawdown_stress(&self, prices: &[f64]) -> f64 {
        if prices.len() < 20 {
            return 0.0;
        }

        let window = 50.min(prices.len());
        let recent_prices = &prices[prices.len() - window..];
        
        let mut peak = recent_prices[0];
        let mut max_drawdown = 0.0;

        for &price in recent_prices {
            if price > peak {
                peak = price;
            }
            let drawdown = (peak - price) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        (max_drawdown * 5.0).min(1.0) // Scale to 0-1
    }

    fn calculate_momentum_stress(&self, returns: &[f64]) -> f64 {
        if returns.len() < 10 {
            return 0.0;
        }

        let recent_returns: f64 = returns.iter().rev().take(10).sum();
        let momentum_stress = (-recent_returns * 10.0).max(0.0).min(1.0);

        momentum_stress
    }
}

impl Default for BlackSwanMetrics {
    fn default() -> Self {
        Self {
            probability: 0.0,
            tail_risk_score: 0.0,
            volatility_regime_shift: 0.0,
            correlation_breakdown: 0.0,
            volume_anomaly: 0.0,
            market_stress_level: 0.0,
            fat_tail_indicator: 0.0,
            kurtosis_excess: 0.0,
            skew_extremity: 0.0,
            early_warning_signals: Vec::new(),
        }
    }
}

impl Default for BlackSwanDetector {
    fn default() -> Self {
        Self::new()
    }
}