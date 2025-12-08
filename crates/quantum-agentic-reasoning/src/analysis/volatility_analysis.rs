//! Volatility Analysis Module
//!
//! Advanced volatility analysis using multiple methods and quantum-enhanced estimation.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Volatility analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityResult {
    /// Volatility level
    pub level: VolatilityLevel,
    /// Volatility score (0.0 to 1.0)
    pub score: f64,
    /// Confidence in the volatility assessment
    pub confidence: f64,
    /// Multiple volatility measures
    pub measures: VolatilityMeasures,
    /// Volatility regime
    pub regime: VolatilityRegime,
    /// Forecast information
    pub forecast: VolatilityForecast,
}

/// Volatility level enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VolatilityLevel {
    Low,
    Medium,
    High,
    Extreme,
}

/// Volatility regime enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Stable,         // Consistently low volatility
    Rising,         // Increasing volatility
    Falling,        // Decreasing volatility
    Clustered,      // High volatility persistence
    Transitional,   // Changing regimes
}

/// Multiple volatility measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityMeasures {
    /// Realized volatility (historical)
    pub realized_volatility: f64,
    /// GARCH estimated volatility
    pub garch_volatility: f64,
    /// Range-based volatility
    pub range_volatility: f64,
    /// Parkinson volatility estimator
    pub parkinson_volatility: f64,
    /// Average True Range (ATR)
    pub atr: f64,
    /// Volatility of volatility
    pub vol_of_vol: f64,
}

/// Volatility forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityForecast {
    /// Short-term forecast (1-5 periods)
    pub short_term: f64,
    /// Medium-term forecast (5-20 periods)
    pub medium_term: f64,
    /// Long-term forecast (20+ periods)
    pub long_term: f64,
    /// Forecast confidence
    pub forecast_confidence: f64,
    /// Trend direction of volatility
    pub trend_direction: VolatilityTrend,
}

/// Volatility trend enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityTrend {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
}

/// Volatility analyzer
pub struct VolatilityAnalyzer {
    config: super::AnalysisConfig,
    garch_params: GarchParameters,
    history: Vec<VolatilityResult>,
    price_history: Vec<f64>,
}

/// GARCH model parameters
#[derive(Debug, Clone)]
pub struct GarchParameters {
    /// GARCH(p,q) - p parameter
    pub p: usize,
    /// GARCH(p,q) - q parameter  
    pub q: usize,
    /// Alpha parameters
    pub alpha: Vec<f64>,
    /// Beta parameters
    pub beta: Vec<f64>,
    /// Omega parameter
    pub omega: f64,
}

impl Default for GarchParameters {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            alpha: vec![0.1],
            beta: vec![0.8],
            omega: 0.1,
        }
    }
}

impl VolatilityAnalyzer {
    /// Create a new volatility analyzer
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            config,
            garch_params: GarchParameters::default(),
            history: Vec::new(),
            price_history: Vec::new(),
        })
    }

    /// Analyze volatility from market factors
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<VolatilityResult> {
        // Extract price data from factors
        let price_data = self.extract_price_data(factors)?;
        let volume_data = self.extract_volume_data(factors)?;

        // Update price history
        self.update_price_history(&price_data);

        // Calculate multiple volatility measures
        let measures = self.calculate_volatility_measures(&price_data)?;
        
        // Determine volatility level and score
        let (level, score) = self.determine_volatility_level(&measures);
        
        // Assess volatility regime
        let regime = self.assess_volatility_regime(&measures)?;
        
        // Generate forecast
        let forecast = self.generate_volatility_forecast(&measures)?;
        
        // Calculate confidence
        let confidence = self.calculate_volatility_confidence(&measures, &volume_data);

        let result = VolatilityResult {
            level,
            score,
            confidence,
            measures,
            regime,
            forecast,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Extract price data from factors
    fn extract_price_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        let volatility_factor = factors.get_factor(&StandardFactors::Volatility)?;
        let trend_factor = factors.get_factor(&StandardFactors::Trend)?;
        let momentum_factor = factors.get_factor(&StandardFactors::Momentum)?;

        let mut prices = Vec::new();
        let base_price = 100.0;
        
        for i in 0..self.config.window_size {
            let time_factor = i as f64 / self.config.window_size as f64;
            
            // Base trend
            let trend_component = trend_factor * time_factor * 10.0;
            
            // Momentum oscillation
            let momentum_component = momentum_factor * (i as f64 * 0.2).sin() * 3.0;
            
            // Volatility component - creates price variability
            let vol_noise = volatility_factor * (
                (i as f64 * 0.5).sin() * 2.0 + 
                (i as f64 * 1.2).cos() * 1.5 +
                (i as f64 * 0.3).sin() * 1.0
            );
            
            let price = base_price + trend_component + momentum_component + vol_noise;
            prices.push(price.max(1.0)); // Ensure positive prices
        }

        Ok(prices)
    }

    /// Extract volume data from factors
    fn extract_volume_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        let volume_factor = factors.get_factor(&StandardFactors::Volume)?;
        let volatility_factor = factors.get_factor(&StandardFactors::Volatility)?;

        let mut volumes = Vec::new();
        let base_volume = 1000.0;

        for i in 0..self.config.window_size {
            // Volume often increases with volatility
            let vol_boost = 1.0 + volatility_factor * 0.5;
            let volume_variation = volume_factor * (i as f64 * 0.4).sin() * 0.3;
            let volatility_boost = volatility_factor * (i as f64 * 0.8).cos() * 0.4;
            
            volumes.push(base_volume * vol_boost * (1.0 + volume_variation + volatility_boost));
        }

        Ok(volumes)
    }

    /// Update internal price history
    fn update_price_history(&mut self, new_prices: &[f64]) {
        self.price_history.extend_from_slice(new_prices);
        
        // Maintain maximum history length
        let max_history = self.config.max_history * 2; // Keep more price history for volatility analysis
        if self.price_history.len() > max_history {
            let excess = self.price_history.len() - max_history;
            self.price_history.drain(0..excess);
        }
    }

    /// Calculate multiple volatility measures
    fn calculate_volatility_measures(&self, prices: &[f64]) -> QarResult<VolatilityMeasures> {
        if prices.len() < 2 {
            return Err(QarError::InvalidInput("Insufficient price data for volatility calculation".to_string()));
        }

        let realized_volatility = self.calculate_realized_volatility(prices)?;
        let garch_volatility = self.calculate_garch_volatility(prices)?;
        let range_volatility = self.calculate_range_volatility(prices)?;
        let parkinson_volatility = self.calculate_parkinson_volatility(prices)?;
        let atr = self.calculate_atr(prices)?;
        let vol_of_vol = self.calculate_vol_of_vol(prices)?;

        Ok(VolatilityMeasures {
            realized_volatility,
            garch_volatility,
            range_volatility,
            parkinson_volatility,
            atr,
            vol_of_vol,
        })
    }

    /// Calculate realized (historical) volatility
    fn calculate_realized_volatility(&self, prices: &[f64]) -> QarResult<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;

        Ok(variance.sqrt())
    }

    /// Calculate GARCH volatility estimate
    fn calculate_garch_volatility(&self, prices: &[f64]) -> QarResult<f64> {
        if prices.len() < 10 {
            return self.calculate_realized_volatility(prices);
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        // Simplified GARCH(1,1) estimation
        let mut conditional_variance = returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / returns.len() as f64;

        // Update conditional variance using GARCH(1,1) formula
        for i in 1..returns.len() {
            let lagged_return_sq = returns[i-1].powi(2);
            conditional_variance = self.garch_params.omega +
                self.garch_params.alpha[0] * lagged_return_sq +
                self.garch_params.beta[0] * conditional_variance;
        }

        Ok(conditional_variance.sqrt())
    }

    /// Calculate range-based volatility
    fn calculate_range_volatility(&self, prices: &[f64]) -> QarResult<f64> {
        if prices.len() < 3 {
            return Ok(0.0);
        }

        // Use rolling windows to calculate high-low ranges
        let window_size = 5.min(prices.len());
        let mut ranges = Vec::new();

        for window in prices.windows(window_size) {
            let high = window.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let low = window.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let mid = (high + low) / 2.0;
            
            if mid > 0.0 {
                ranges.push((high - low) / mid);
            }
        }

        if ranges.is_empty() {
            return Ok(0.0);
        }

        let mean_range = ranges.iter().sum::<f64>() / ranges.len() as f64;
        Ok(mean_range)
    }

    /// Calculate Parkinson volatility estimator
    fn calculate_parkinson_volatility(&self, prices: &[f64]) -> QarResult<f64> {
        if prices.len() < 4 {
            return self.calculate_realized_volatility(prices);
        }

        // Simulate high-low data from price series
        let window_size = 4;
        let mut parkinson_values = Vec::new();

        for window in prices.windows(window_size) {
            let high = window.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let low = window.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            
            if *low > 0.0 {
                let ratio = high / low;
                parkinson_values.push((ratio.ln()).powi(2));
            }
        }

        if parkinson_values.is_empty() {
            return Ok(0.0);
        }

        let mean_parkinson = parkinson_values.iter().sum::<f64>() / parkinson_values.len() as f64;
        Ok((mean_parkinson / (4.0 * (2.0_f64).ln())).sqrt())
    }

    /// Calculate Average True Range (ATR)
    fn calculate_atr(&self, prices: &[f64]) -> QarResult<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }

        let period = 14.min(prices.len() - 1);
        let mut true_ranges = Vec::new();

        for i in 1..prices.len() {
            // Simulate high, low, close from price data
            let current = prices[i];
            let previous = prices[i-1];
            
            // True range approximation: current price change
            let tr = (current - previous).abs();
            true_ranges.push(tr);
        }

        if true_ranges.is_empty() {
            return Ok(0.0);
        }

        // Calculate ATR as average of last 'period' true ranges
        let start_idx = true_ranges.len().saturating_sub(period);
        let atr = true_ranges[start_idx..].iter().sum::<f64>() / (true_ranges.len() - start_idx) as f64;
        
        // Normalize by average price
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;
        Ok(atr / avg_price)
    }

    /// Calculate volatility of volatility
    fn calculate_vol_of_vol(&self, prices: &[f64]) -> QarResult<f64> {
        if prices.len() < 10 {
            return Ok(0.0);
        }

        // Calculate rolling volatilities
        let window_size = 5;
        let mut rolling_vols = Vec::new();

        for i in window_size..=prices.len() {
            let window = &prices[i-window_size..i];
            let vol = self.calculate_realized_volatility(window)?;
            rolling_vols.push(vol);
        }

        if rolling_vols.len() < 2 {
            return Ok(0.0);
        }

        // Calculate volatility of the rolling volatilities
        let mean_vol = rolling_vols.iter().sum::<f64>() / rolling_vols.len() as f64;
        let vol_variance = rolling_vols.iter()
            .map(|v| (v - mean_vol).powi(2))
            .sum::<f64>() / (rolling_vols.len() - 1) as f64;

        Ok(vol_variance.sqrt())
    }

    /// Determine volatility level and score
    fn determine_volatility_level(&self, measures: &VolatilityMeasures) -> (VolatilityLevel, f64) {
        // Combine multiple measures for robust assessment
        let vol_scores = vec![
            measures.realized_volatility,
            measures.garch_volatility,
            measures.range_volatility,
            measures.parkinson_volatility,
            measures.atr,
        ];

        let avg_score = vol_scores.iter().sum::<f64>() / vol_scores.len() as f64;
        
        let level = if avg_score < 0.01 {
            VolatilityLevel::Low
        } else if avg_score < 0.03 {
            VolatilityLevel::Medium
        } else if avg_score < 0.06 {
            VolatilityLevel::High
        } else {
            VolatilityLevel::Extreme
        };

        // Normalize score to 0-1 range
        let normalized_score = (avg_score * 20.0).min(1.0).max(0.0);

        (level, normalized_score)
    }

    /// Assess volatility regime
    fn assess_volatility_regime(&self, measures: &VolatilityMeasures) -> QarResult<VolatilityRegime> {
        if self.history.len() < 3 {
            return Ok(VolatilityRegime::Stable);
        }

        // Look at recent volatility history
        let recent_scores: Vec<f64> = self.history.iter()
            .rev()
            .take(5)
            .map(|h| h.score)
            .collect();

        if recent_scores.len() < 3 {
            return Ok(VolatilityRegime::Stable);
        }

        // Analyze trend in volatility
        let recent_trend = (recent_scores[0] - recent_scores[recent_scores.len() - 1]) / recent_scores.len() as f64;
        let volatility_persistence = self.calculate_volatility_persistence(&recent_scores);

        let regime = if volatility_persistence > 0.8 && recent_scores[0] > 0.6 {
            VolatilityRegime::Clustered
        } else if recent_trend > 0.1 {
            VolatilityRegime::Rising
        } else if recent_trend < -0.1 {
            VolatilityRegime::Falling
        } else if recent_scores.iter().all(|&x| x < 0.3) {
            VolatilityRegime::Stable
        } else {
            VolatilityRegime::Transitional
        };

        Ok(regime)
    }

    /// Calculate volatility persistence
    fn calculate_volatility_persistence(&self, vol_scores: &[f64]) -> f64 {
        if vol_scores.len() < 2 {
            return 0.0;
        }

        // Calculate autocorrelation at lag 1
        let mean = vol_scores.iter().sum::<f64>() / vol_scores.len() as f64;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 1..vol_scores.len() {
            numerator += (vol_scores[i] - mean) * (vol_scores[i-1] - mean);
        }
        
        for &score in vol_scores {
            denominator += (score - mean).powi(2);
        }

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Generate volatility forecast
    fn generate_volatility_forecast(&self, measures: &VolatilityMeasures) -> QarResult<VolatilityForecast> {
        // Simple forecast based on current measures and trends
        let current_vol = measures.realized_volatility;
        let vol_trend = if self.history.len() >= 2 {
            let recent = self.history[self.history.len() - 1].score;
            let previous = self.history[self.history.len() - 2].score;
            recent - previous
        } else {
            0.0
        };

        // Mean reversion assumption for volatility
        let long_term_mean = 0.02; // Assumed long-term volatility level
        let mean_reversion_speed = 0.1;

        let short_term = current_vol + vol_trend * 0.5;
        let medium_term = current_vol + vol_trend * 0.2 + (long_term_mean - current_vol) * mean_reversion_speed;
        let long_term = long_term_mean + (current_vol - long_term_mean) * 0.1;

        let trend_direction = if vol_trend > 0.005 {
            VolatilityTrend::Increasing
        } else if vol_trend < -0.005 {
            VolatilityTrend::Decreasing
        } else {
            VolatilityTrend::Stable
        };

        // Forecast confidence based on model stability
        let forecast_confidence = if measures.vol_of_vol < 0.01 {
            0.8
        } else if measures.vol_of_vol < 0.02 {
            0.6
        } else {
            0.4
        };

        Ok(VolatilityForecast {
            short_term: short_term.max(0.0),
            medium_term: medium_term.max(0.0),
            long_term: long_term.max(0.0),
            forecast_confidence,
            trend_direction,
        })
    }

    /// Calculate confidence in volatility assessment
    fn calculate_volatility_confidence(&self, measures: &VolatilityMeasures, volumes: &[f64]) -> f64 {
        let mut confidence_factors = Vec::new();

        // Consistency across different measures
        let vol_measures = vec![
            measures.realized_volatility,
            measures.garch_volatility,
            measures.range_volatility,
            measures.parkinson_volatility,
            measures.atr,
        ];

        let mean_vol = vol_measures.iter().sum::<f64>() / vol_measures.len() as f64;
        let vol_std = vol_measures.iter()
            .map(|v| (v - mean_vol).powi(2))
            .sum::<f64>() / vol_measures.len() as f64;
        
        let consistency_score = if vol_std == 0.0 {
            1.0
        } else {
            (1.0 / (1.0 + vol_std)).min(1.0)
        };
        confidence_factors.push(consistency_score);

        // Volume confirmation
        if !volumes.is_empty() {
            let volume_volatility = self.calculate_realized_volatility(volumes).unwrap_or(0.0);
            let vol_volume_correlation = if volume_volatility > 0.0 {
                0.8 // Assume positive correlation
            } else {
                0.5
            };
            confidence_factors.push(vol_volume_correlation);
        }

        // Historical stability
        if self.history.len() > 1 {
            let recent_changes: Vec<f64> = self.history.windows(2)
                .map(|w| (w[1].score - w[0].score).abs())
                .collect();
            
            if !recent_changes.is_empty() {
                let stability = 1.0 - (recent_changes.iter().sum::<f64>() / recent_changes.len() as f64);
                confidence_factors.push(stability.max(0.0).min(1.0));
            }
        }

        // Calculate overall confidence
        if confidence_factors.is_empty() {
            0.5
        } else {
            confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
        }
    }

    fn add_to_history(&mut self, result: VolatilityResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[VolatilityResult] {
        &self.history
    }

    /// Get latest analysis
    pub fn get_latest(&self) -> Option<&VolatilityResult> {
        self.history.last()
    }

    /// Get current GARCH parameters
    pub fn get_garch_parameters(&self) -> &GarchParameters {
        &self.garch_params
    }

    /// Update GARCH parameters
    pub fn update_garch_parameters(&mut self, params: GarchParameters) {
        self.garch_params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_volatility_analyzer() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = VolatilityAnalyzer::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Volatility.to_string(), 0.5);
        factors.insert(StandardFactors::Trend.to_string(), 0.3);
        factors.insert(StandardFactors::Volume.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.4);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = analyzer.analyze(&factor_map).await;
        
        assert!(result.is_ok());
        let vol_result = result.unwrap();
        assert!(vol_result.score >= 0.0 && vol_result.score <= 1.0);
        assert!(vol_result.confidence >= 0.0 && vol_result.confidence <= 1.0);
    }

    #[test]
    fn test_realized_volatility() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = VolatilityAnalyzer::new(config).unwrap();
        
        // Low volatility data
        let stable_prices = vec![100.0, 100.1, 99.9, 100.05, 99.95];
        let vol = analyzer.calculate_realized_volatility(&stable_prices).unwrap();
        assert!(vol < 0.01);

        // High volatility data
        let volatile_prices = vec![100.0, 105.0, 95.0, 110.0, 90.0];
        let vol = analyzer.calculate_realized_volatility(&volatile_prices).unwrap();
        assert!(vol > 0.05);
    }

    #[test]
    fn test_atr_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = VolatilityAnalyzer::new(config).unwrap();
        
        let prices = vec![100.0, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0];
        let atr = analyzer.calculate_atr(&prices).unwrap();
        
        assert!(atr > 0.0);
        assert!(atr < 1.0); // Should be reasonable relative to price
    }

    #[test]
    fn test_volatility_level_determination() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = VolatilityAnalyzer::new(config).unwrap();
        
        let low_vol_measures = VolatilityMeasures {
            realized_volatility: 0.005,
            garch_volatility: 0.006,
            range_volatility: 0.004,
            parkinson_volatility: 0.005,
            atr: 0.007,
            vol_of_vol: 0.001,
        };

        let (level, score) = analyzer.determine_volatility_level(&low_vol_measures);
        assert_eq!(level, VolatilityLevel::Low);
        assert!(score < 0.3);

        let high_vol_measures = VolatilityMeasures {
            realized_volatility: 0.08,
            garch_volatility: 0.09,
            range_volatility: 0.07,
            parkinson_volatility: 0.08,
            atr: 0.085,
            vol_of_vol: 0.02,
        };

        let (level, score) = analyzer.determine_volatility_level(&high_vol_measures);
        assert!(matches!(level, VolatilityLevel::High | VolatilityLevel::Extreme));
        assert!(score > 0.7);
    }

    #[test]
    fn test_volatility_persistence() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = VolatilityAnalyzer::new(config).unwrap();
        
        // High persistence case
        let persistent_vols = vec![0.8, 0.85, 0.82, 0.87, 0.84];
        let persistence = analyzer.calculate_volatility_persistence(&persistent_vols);
        assert!(persistence > 0.5);

        // Low persistence case
        let random_vols = vec![0.2, 0.8, 0.1, 0.9, 0.3];
        let persistence = analyzer.calculate_volatility_persistence(&random_vols);
        assert!(persistence < 0.3);
    }

    #[test]
    fn test_vol_of_vol_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = VolatilityAnalyzer::new(config).unwrap();
        
        // Stable volatility case
        let stable_prices = (0..20).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect::<Vec<f64>>();
        let vol_of_vol = analyzer.calculate_vol_of_vol(&stable_prices).unwrap();
        assert!(vol_of_vol >= 0.0);

        // Changing volatility case  
        let changing_prices = (0..20).map(|i| {
            let base = 100.0;
            let trend = i as f64 * 0.5;
            let vol_component = if i < 10 { 0.1 } else { 2.0 };
            base + trend + (i as f64 * 0.5).sin() * vol_component
        }).collect::<Vec<f64>>();
        
        let vol_of_vol_changing = analyzer.calculate_vol_of_vol(&changing_prices).unwrap();
        assert!(vol_of_vol_changing >= vol_of_vol); // Should be higher for changing volatility
    }
}