//! Self-Organized Criticality Analysis
//! 
//! Implementation of SOC theory for financial markets, detecting systems
//! approaching critical thresholds where small changes can trigger large events.

use crate::market_data::MarketData;
use anyhow::Result;
use std::collections::VecDeque;

pub struct SOCAnalyzer {
    avalanche_threshold: f64,
    correlation_window: usize,
    fractal_scales: Vec<usize>,
    power_law_threshold: f64,
    event_history: VecDeque<CriticalEvent>,
}

#[derive(Debug, Clone)]
pub struct CriticalityMetrics {
    pub overall_level: f64,
    pub avalanche_risk: f64,
    pub correlation_length: f64,
    pub fractal_dimension: f64,
    pub power_law_exponent: f64,
    pub system_stress: f64,
    pub phase_transition_proximity: f64,
}

#[derive(Debug, Clone)]
struct CriticalEvent {
    magnitude: f64,
    duration: usize,
    timestamp: usize,
    event_type: EventType,
}

#[derive(Debug, Clone)]
enum EventType {
    PriceAvalanche,
    VolumeSpike,
    CorrelationBreakdown,
    RegimeShift,
}

impl SOCAnalyzer {
    pub fn new() -> Self {
        Self {
            avalanche_threshold: 0.05, // 5% price move
            correlation_window: 50,
            fractal_scales: vec![5, 10, 20, 50, 100],
            power_law_threshold: 0.8,
            event_history: VecDeque::with_capacity(1000),
        }
    }

    pub fn analyze_criticality(&mut self, data: &MarketData) -> Result<f64> {
        if data.len() < 100 {
            return Ok(0.0);
        }

        let metrics = self.calculate_criticality_metrics(data)?;
        self.update_event_history(data, &metrics);

        // Combine metrics into overall criticality level
        let criticality_level = self.combine_criticality_indicators(
            metrics.avalanche_risk, 
            metrics.correlation_length, 
            metrics.fractal_dimension,
            metrics.power_law_exponent,
            metrics.system_stress,
            metrics.phase_transition_proximity
        );

        Ok(criticality_level)
    }

    fn calculate_criticality_metrics(&self, data: &MarketData) -> Result<CriticalityMetrics> {
        let avalanche_risk = self.calculate_avalanche_risk(data)?;
        let correlation_length = self.calculate_correlation_length(data)?;
        let fractal_dimension = self.calculate_fractal_dimension(data)?;
        let power_law_exponent = self.calculate_power_law_exponent(data)?;
        let system_stress = self.calculate_system_stress(data)?;
        let phase_transition_proximity = self.detect_phase_transition_proximity(data)?;

        let overall_level = self.combine_criticality_indicators(
            avalanche_risk, correlation_length, fractal_dimension,
            power_law_exponent, system_stress, phase_transition_proximity
        );

        Ok(CriticalityMetrics {
            overall_level,
            avalanche_risk,
            correlation_length,
            fractal_dimension,
            power_law_exponent,
            system_stress,
            phase_transition_proximity,
        })
    }

    /// Detect potential for avalanche-like price movements
    fn calculate_avalanche_risk(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 50 {
            return Ok(0.0);
        }

        // Calculate clustering of small events (sand pile model)
        let mut small_events = 0;
        let mut medium_events = 0;
        let mut large_events = 0;

        for &ret in &returns {
            let abs_ret = ret.abs();
            if abs_ret < 0.01 {
                small_events += 1;
            } else if abs_ret < 0.03 {
                medium_events += 1;
            } else {
                large_events += 1;
            }
        }

        // SOC systems show power-law distribution of event sizes
        let total_events = small_events + medium_events + large_events;
        if total_events == 0 {
            return Ok(0.0);
        }

        let small_ratio = small_events as f64 / total_events as f64;
        let medium_ratio = medium_events as f64 / total_events as f64;
        let large_ratio = large_events as f64 / total_events as f64;

        // Calculate power-law fitness
        let expected_small = 0.7; // Expected ratio for SOC system
        let expected_medium = 0.25;
        let expected_large = 0.05;

        let power_law_deviation = 
            (small_ratio - expected_small).abs() +
            (medium_ratio - expected_medium).abs() +
            (large_ratio - expected_large).abs();

        // Risk increases as system deviates from power-law distribution
        // and accumulates more small events
        let avalanche_risk = (small_ratio * 2.0 - power_law_deviation).clamp(0.0, 1.0);

        Ok(avalanche_risk)
    }

    /// Calculate correlation length - how far perturbations propagate
    fn calculate_correlation_length(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < self.correlation_window * 2 {
            return Ok(0.0);
        }

        let mut max_correlation_length = 0.0;

        // Calculate correlations at different lags
        for lag in 1..=20 {
            let mut correlation_sum = 0.0;
            let mut count = 0;

            for i in lag..returns.len() {
                let corr = returns[i] * returns[i - lag];
                correlation_sum += corr;
                count += 1;
            }

            let avg_correlation = if count > 0 {
                correlation_sum / count as f64
            } else {
                0.0
            };

            if avg_correlation.abs() > 0.1 {
                max_correlation_length = lag as f64;
            }
        }

        // Normalize to 0-1 scale
        Ok((max_correlation_length / 20.0).min(1.0))
    }

    /// Calculate fractal dimension of price series
    fn calculate_fractal_dimension(&self, data: &MarketData) -> Result<f64> {
        let prices = &data.prices;
        if prices.len() < 100 {
            return Ok(0.0);
        }

        let mut dimensions = Vec::new();

        for &scale in &self.fractal_scales {
            if scale >= prices.len() {
                continue;
            }

            let dimension = self.box_counting_dimension(prices, scale);
            dimensions.push(dimension);
        }

        if dimensions.is_empty() {
            return Ok(0.0);
        }

        // Average fractal dimension
        let avg_dimension = dimensions.iter().sum::<f64>() / dimensions.len() as f64;
        
        // Higher fractal dimension indicates approaching criticality
        Ok((avg_dimension - 1.0).max(0.0).min(1.0))
    }

    fn box_counting_dimension(&self, prices: &[f64], scale: usize) -> f64 {
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_price <= min_price {
            return 1.0;
        }

        let box_size = (max_price - min_price) / scale as f64;
        let mut occupied_boxes = std::collections::HashSet::new();

        for &price in prices {
            let box_index = ((price - min_price) / box_size).floor() as i32;
            occupied_boxes.insert(box_index);
        }

        let n_boxes = occupied_boxes.len() as f64;
        
        if n_boxes <= 1.0 || scale <= 1 {
            1.0
        } else {
            n_boxes.ln() / (scale as f64).ln()
        }
    }

    /// Calculate power-law exponent of return distribution
    fn calculate_power_law_exponent(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 100 {
            return Ok(0.0);
        }

        // Sort absolute returns
        let mut abs_returns: Vec<f64> = returns.iter().map(|&r| r.abs()).collect();
        abs_returns.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Remove zeros and very small values
        abs_returns.retain(|&r| r > 1e-6);

        if abs_returns.len() < 20 {
            return Ok(0.0);
        }

        // Calculate power-law exponent using rank-frequency analysis
        let mut log_ranks = Vec::new();
        let mut log_values = Vec::new();

        for (i, &value) in abs_returns.iter().enumerate() {
            let rank = (i + 1) as f64;
            log_ranks.push(rank.ln());
            log_values.push(value.ln());
        }

        // Linear regression to find slope
        let exponent = self.linear_regression_slope(&log_ranks, &log_values);
        
        // Power-law exponent around 2-3 indicates criticality in financial markets
        let criticality_score = if exponent.abs() >= 2.0 && exponent.abs() <= 4.0 {
            1.0 - (exponent.abs() - 2.5).abs() / 1.5
        } else {
            0.0
        };

        Ok(criticality_score.clamp(0.0, 1.0))
    }

    /// Measure overall system stress
    fn calculate_system_stress(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        let volumes = &data.volumes;
        
        if returns.len() < 50 || volumes.len() < 50 {
            return Ok(0.0);
        }

        // Volatility stress
        let volatility = self.calculate_rolling_volatility(&returns, 20);
        let vol_stress = if volatility > 0.0 {
            (volatility * 100.0).min(10.0) / 10.0
        } else {
            0.0
        };

        // Volume stress (deviation from normal)
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let recent_volume = volumes.iter().rev().take(10).sum::<f64>() / 10.0;
        let vol_ratio_stress = if avg_volume > 0.0 {
            (recent_volume / avg_volume - 1.0).abs().min(2.0) / 2.0
        } else {
            0.0
        };

        // Price level stress (distance from moving average)
        let sma = data.sma(50);
        let current_price = data.prices[data.len() - 1];
        let sma_current = sma.last().unwrap_or(&current_price);
        let price_stress = ((current_price - sma_current) / sma_current).abs().min(0.5) / 0.5;

        // Combined stress
        let system_stress = (vol_stress * 0.4 + vol_ratio_stress * 0.3 + price_stress * 0.3).min(1.0);

        Ok(system_stress)
    }

    /// Detect proximity to phase transitions
    fn detect_phase_transition_proximity(&self, data: &MarketData) -> Result<f64> {
        if data.len() < 100 {
            return Ok(0.0);
        }

        // Look for early warning signals of phase transitions
        let returns = data.returns();
        
        // 1. Critical slowing down (increasing autocorrelation)
        let autocorr = self.calculate_autocorrelation(&returns, 1);
        let slowing_score = autocorr.abs().min(1.0);

        // 2. Increasing variance (flickering)
        let recent_var = self.calculate_rolling_variance(&returns, 20);
        let historical_var = self.calculate_rolling_variance(&returns, 100);
        let variance_ratio = if historical_var > 0.0 {
            (recent_var / historical_var).min(3.0) / 3.0
        } else {
            0.0
        };

        // 3. Skewness changes (asymmetry in fluctuations)
        let skewness = self.calculate_skewness(&returns[returns.len().saturating_sub(50)..]);
        let skew_score = skewness.abs().min(2.0) / 2.0;

        // Combined proximity score
        let proximity = (slowing_score * 0.4 + variance_ratio * 0.4 + skew_score * 0.2).min(1.0);

        Ok(proximity)
    }

    fn combine_criticality_indicators(&self, avalanche_risk: f64, correlation_length: f64,
                                    fractal_dimension: f64, power_law_exponent: f64,
                                    system_stress: f64, phase_transition_proximity: f64) -> f64 {
        // Weighted combination of all indicators
        let criticality = (
            avalanche_risk * 0.25 +
            correlation_length * 0.15 +
            fractal_dimension * 0.20 +
            power_law_exponent * 0.15 +
            system_stress * 0.15 +
            phase_transition_proximity * 0.10
        ).min(1.0);

        criticality
    }

    fn update_event_history(&mut self, data: &MarketData, metrics: &CriticalityMetrics) {
        let current_index = data.len() - 1;
        
        // Detect critical events
        if metrics.avalanche_risk > 0.8 {
            self.event_history.push_back(CriticalEvent {
                magnitude: metrics.avalanche_risk,
                duration: 1,
                timestamp: current_index,
                event_type: EventType::PriceAvalanche,
            });
        }

        if metrics.phase_transition_proximity > 0.7 {
            self.event_history.push_back(CriticalEvent {
                magnitude: metrics.phase_transition_proximity,
                duration: 1,
                timestamp: current_index,
                event_type: EventType::RegimeShift,
            });
        }

        // Keep only recent events
        while self.event_history.len() > 1000 {
            self.event_history.pop_front();
        }
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

    fn calculate_rolling_variance(&self, returns: &[f64], window: usize) -> f64 {
        if returns.len() < window {
            return 0.0;
        }

        let recent_returns = &returns[returns.len() - window..];
        let mean = recent_returns.iter().sum::<f64>() / window as f64;
        recent_returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (window - 1) as f64
    }

    fn calculate_autocorrelation(&self, returns: &[f64], lag: usize) -> f64 {
        if returns.len() <= lag {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in lag..returns.len() {
            numerator += (returns[i] - mean) * (returns[i - lag] - mean);
        }

        for &r in returns {
            denominator += (r - mean).powi(2);
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn calculate_skewness(&self, returns: &[f64]) -> f64 {
        if returns.len() < 3 {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        if variance <= 0.0 {
            return 0.0;
        }

        let std_dev = variance.sqrt();
        let skewness = returns.iter()
            .map(|&r| ((r - mean) / std_dev).powi(3))
            .sum::<f64>() / returns.len() as f64;

        skewness
    }

    fn linear_regression_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }
}

impl Default for SOCAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}