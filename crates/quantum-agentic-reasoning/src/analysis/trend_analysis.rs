//! Trend Analysis Module
//!
//! Advanced trend analysis using multiple timeframes and quantum-enhanced detection.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendResult {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Trend confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Trend duration in periods
    pub duration: usize,
    /// Multi-timeframe analysis
    pub timeframe_analysis: TimeframeAnalysis,
    /// Trend quality metrics
    pub quality_metrics: TrendQualityMetrics,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Sideways,
    Unknown,
}

/// Multi-timeframe trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeframeAnalysis {
    /// Short-term trend (e.g., 1H)
    pub short_term: TrendInfo,
    /// Medium-term trend (e.g., 4H)
    pub medium_term: TrendInfo,
    /// Long-term trend (e.g., 1D)
    pub long_term: TrendInfo,
    /// Trend alignment score
    pub alignment_score: f64,
}

/// Individual timeframe trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendInfo {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f64,
    /// Trend age in periods
    pub age: usize,
    /// Trend stability
    pub stability: f64,
}

/// Trend quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendQualityMetrics {
    /// Consistency score (0.0 to 1.0)
    pub consistency: f64,
    /// Momentum score (0.0 to 1.0)
    pub momentum: f64,
    /// Volume confirmation (0.0 to 1.0)
    pub volume_confirmation: f64,
    /// Structure quality (0.0 to 1.0)
    pub structure_quality: f64,
    /// Volatility score (0.0 to 1.0, lower is better for trends)
    pub volatility_score: f64,
}

/// Trend analyzer
pub struct TrendAnalyzer {
    config: super::AnalysisConfig,
    moving_averages: MovingAverageConfig,
    momentum_indicators: MomentumConfig,
    history: Vec<TrendResult>,
}

/// Moving average configuration
#[derive(Debug, Clone)]
pub struct MovingAverageConfig {
    /// Short-term MA period
    pub short_period: usize,
    /// Medium-term MA period
    pub medium_period: usize,
    /// Long-term MA period
    pub long_period: usize,
    /// MA type
    pub ma_type: MovingAverageType,
}

/// Moving average types
#[derive(Debug, Clone)]
pub enum MovingAverageType {
    Simple,
    Exponential,
    Weighted,
    Adaptive,
}

/// Momentum indicator configuration
#[derive(Debug, Clone)]
pub struct MomentumConfig {
    /// RSI period
    pub rsi_period: usize,
    /// MACD fast period
    pub macd_fast: usize,
    /// MACD slow period
    pub macd_slow: usize,
    /// MACD signal period
    pub macd_signal: usize,
    /// Stochastic period
    pub stoch_period: usize,
}

impl Default for MovingAverageConfig {
    fn default() -> Self {
        Self {
            short_period: 9,
            medium_period: 21,
            long_period: 50,
            ma_type: MovingAverageType::Exponential,
        }
    }
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            stoch_period: 14,
        }
    }
}

impl TrendAnalyzer {
    /// Create a new trend analyzer
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            config,
            moving_averages: MovingAverageConfig::default(),
            momentum_indicators: MomentumConfig::default(),
            history: Vec::new(),
        })
    }

    /// Analyze trend from market factors
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<TrendResult> {
        // Extract price data from factors
        let price_data = self.extract_price_data(factors)?;
        let volume_data = self.extract_volume_data(factors)?;

        // Perform trend analysis
        let direction = self.determine_trend_direction(&price_data)?;
        let strength = self.calculate_trend_strength(&price_data, &direction)?;
        let confidence = self.calculate_trend_confidence(&price_data, &volume_data, &direction)?;
        let duration = self.estimate_trend_duration(&price_data, &direction)?;
        let timeframe_analysis = self.analyze_multiple_timeframes(&price_data)?;
        let quality_metrics = self.calculate_quality_metrics(&price_data, &volume_data, &direction)?;

        let result = TrendResult {
            direction,
            strength,
            confidence,
            duration,
            timeframe_analysis,
            quality_metrics,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Extract price data from factors
    fn extract_price_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        // Use trend and momentum factors to generate synthetic price data
        let trend_factor = factors.get_factor(&StandardFactors::Trend)?;
        let momentum_factor = factors.get_factor(&StandardFactors::Momentum)?;
        let volatility_factor = factors.get_factor(&StandardFactors::Volatility)?;

        let mut prices = Vec::new();
        let base_price = 100.0;
        
        for i in 0..self.config.window_size {
            let time_factor = i as f64 / self.config.window_size as f64;
            
            // Trend component
            let trend_component = trend_factor * time_factor * 20.0;
            
            // Momentum component
            let momentum_component = momentum_factor * (i as f64 * 0.1).sin() * 5.0;
            
            // Volatility/noise component
            let volatility_component = volatility_factor * (i as f64 * 0.3).cos() * 2.0;
            
            let price = base_price + trend_component + momentum_component + volatility_component;
            prices.push(price);
        }

        Ok(prices)
    }

    /// Extract volume data from factors
    fn extract_volume_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        let volume_factor = factors.get_factor(&StandardFactors::Volume)?;
        let trend_factor = factors.get_factor(&StandardFactors::Trend)?;

        let mut volumes = Vec::new();
        let base_volume = 1000.0;

        for i in 0..self.config.window_size {
            // Volume tends to increase with strong trends
            let trend_volume = base_volume * (1.0 + trend_factor.abs() * 0.5);
            let noise = (i as f64 * 0.2).sin() * volume_factor * 200.0;
            volumes.push(trend_volume + noise);
        }

        Ok(volumes)
    }

    /// Determine overall trend direction
    fn determine_trend_direction(&self, prices: &[f64]) -> QarResult<TrendDirection> {
        if prices.len() < 3 {
            return Ok(TrendDirection::Unknown);
        }

        // Calculate multiple moving averages
        let short_ma = self.calculate_moving_average(prices, self.moving_averages.short_period)?;
        let medium_ma = self.calculate_moving_average(prices, self.moving_averages.medium_period)?;
        let long_ma = self.calculate_moving_average(prices, self.moving_averages.long_period)?;

        let current_price = prices[prices.len() - 1];
        
        // Determine trend based on MA alignment and current price position
        if current_price > short_ma && short_ma > medium_ma && medium_ma > long_ma {
            Ok(TrendDirection::Bullish)
        } else if current_price < short_ma && short_ma < medium_ma && medium_ma < long_ma {
            Ok(TrendDirection::Bearish)
        } else if (current_price - long_ma).abs() / long_ma < 0.02 {
            Ok(TrendDirection::Sideways)
        } else {
            Ok(TrendDirection::Unknown)
        }
    }

    /// Calculate trend strength
    fn calculate_trend_strength(&self, prices: &[f64], direction: &TrendDirection) -> QarResult<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }

        match direction {
            TrendDirection::Unknown => Ok(0.0),
            TrendDirection::Sideways => {
                let volatility = self.calculate_volatility(prices);
                Ok((1.0 - volatility).max(0.0).min(1.0))
            },
            _ => {
                // Calculate strength based on price momentum and consistency
                let price_change = (prices[prices.len() - 1] - prices[0]) / prices[0];
                let raw_strength = price_change.abs();
                
                // Adjust for direction consistency
                let consistency = self.calculate_direction_consistency(prices, direction);
                let adjusted_strength = raw_strength * consistency;
                
                Ok(adjusted_strength.min(1.0).max(0.0))
            }
        }
    }

    /// Calculate trend confidence
    fn calculate_trend_confidence(&self, prices: &[f64], volumes: &[f64], direction: &TrendDirection) -> QarResult<f64> {
        if prices.is_empty() {
            return Ok(0.0);
        }

        let mut confidence_factors = Vec::new();

        // Price action confidence
        let price_confidence = self.calculate_price_action_confidence(prices, direction);
        confidence_factors.push(price_confidence);

        // Volume confirmation
        if !volumes.is_empty() {
            let volume_confidence = self.calculate_volume_confirmation(prices, volumes, direction);
            confidence_factors.push(volume_confidence);
        }

        // Momentum confirmation
        let momentum_confidence = self.calculate_momentum_confirmation(prices, direction);
        confidence_factors.push(momentum_confidence);

        // Calculate overall confidence as weighted average
        let overall_confidence = confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64;
        Ok(overall_confidence.min(1.0).max(0.0))
    }

    /// Estimate trend duration
    fn estimate_trend_duration(&self, prices: &[f64], direction: &TrendDirection) -> QarResult<usize> {
        if prices.len() < 3 {
            return Ok(0);
        }

        let mut duration = 0;
        let window_size = 5; // Look at 5-period windows

        for i in (window_size..prices.len()).rev() {
            let window = &prices[i - window_size..i];
            let window_direction = self.determine_window_direction(window);
            
            if self.directions_match(direction, &window_direction) {
                duration += window_size;
            } else {
                break;
            }
        }

        Ok(duration)
    }

    /// Analyze multiple timeframes
    fn analyze_multiple_timeframes(&self, prices: &[f64]) -> QarResult<TimeframeAnalysis> {
        if prices.len() < 30 {
            return Err(QarError::InvalidInput("Insufficient data for multi-timeframe analysis".to_string()));
        }

        // Simulate different timeframes by sampling data
        let short_term = self.analyze_timeframe_window(&prices[prices.len() - 10..], 10)?;
        let medium_term = self.analyze_timeframe_window(&prices[prices.len() - 20..], 20)?;
        let long_term = self.analyze_timeframe_window(prices, prices.len())?;

        // Calculate alignment score
        let alignment_score = self.calculate_timeframe_alignment(&short_term, &medium_term, &long_term);

        Ok(TimeframeAnalysis {
            short_term,
            medium_term,
            long_term,
            alignment_score,
        })
    }

    /// Calculate quality metrics
    fn calculate_quality_metrics(&self, prices: &[f64], volumes: &[f64], direction: &TrendDirection) -> QarResult<TrendQualityMetrics> {
        let consistency = self.calculate_direction_consistency(prices, direction);
        let momentum = self.calculate_momentum_score(prices);
        let volume_confirmation = if !volumes.is_empty() {
            self.calculate_volume_confirmation(prices, volumes, direction)
        } else {
            0.5 // Neutral if no volume data
        };
        let structure_quality = self.calculate_structure_quality(prices);
        let volatility_score = 1.0 - self.calculate_volatility(prices); // Lower volatility = better for trends

        Ok(TrendQualityMetrics {
            consistency,
            momentum,
            volume_confirmation,
            structure_quality,
            volatility_score: volatility_score.max(0.0).min(1.0),
        })
    }

    /// Helper methods
    fn calculate_moving_average(&self, prices: &[f64], period: usize) -> QarResult<f64> {
        if prices.is_empty() || period == 0 || period > prices.len() {
            return Err(QarError::InvalidInput("Invalid parameters for moving average".to_string()));
        }

        let start_idx = prices.len() - period;
        match self.moving_averages.ma_type {
            MovingAverageType::Simple => {
                let sum: f64 = prices[start_idx..].iter().sum();
                Ok(sum / period as f64)
            },
            MovingAverageType::Exponential => {
                let alpha = 2.0 / (period as f64 + 1.0);
                let mut ema = prices[start_idx];
                
                for &price in &prices[start_idx + 1..] {
                    ema = alpha * price + (1.0 - alpha) * ema;
                }
                
                Ok(ema)
            },
            _ => {
                // Fallback to simple moving average for other types
                let sum: f64 = prices[start_idx..].iter().sum();
                Ok(sum / period as f64)
            }
        }
    }

    fn calculate_volatility(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    fn calculate_direction_consistency(&self, prices: &[f64], direction: &TrendDirection) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let moves: Vec<bool> = prices.windows(2)
            .map(|w| {
                let up_move = w[1] > w[0];
                match direction {
                    TrendDirection::Bullish => up_move,
                    TrendDirection::Bearish => !up_move,
                    TrendDirection::Sideways => (w[1] - w[0]).abs() / w[0] < 0.01, // Small moves
                    TrendDirection::Unknown => true, // Always consistent for unknown
                }
            })
            .collect();

        let consistent_moves = moves.iter().filter(|&&m| m).count();
        consistent_moves as f64 / moves.len() as f64
    }

    fn calculate_price_action_confidence(&self, prices: &[f64], direction: &TrendDirection) -> f64 {
        if prices.len() < 5 {
            return 0.0;
        }

        // Calculate confidence based on higher highs/lower lows pattern
        match direction {
            TrendDirection::Bullish => {
                let recent_high = prices[prices.len() - 5..].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let earlier_high = prices[..prices.len() - 5].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
                if recent_high > earlier_high { 0.8 } else { 0.3 }
            },
            TrendDirection::Bearish => {
                let recent_low = prices[prices.len() - 5..].iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let earlier_low = prices[..prices.len() - 5].iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&f64::INFINITY);
                if recent_low < earlier_low { 0.8 } else { 0.3 }
            },
            TrendDirection::Sideways => {
                let range = prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() 
                          - prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;
                if range / avg_price < 0.05 { 0.8 } else { 0.4 }
            },
            TrendDirection::Unknown => 0.1,
        }
    }

    fn calculate_volume_confirmation(&self, prices: &[f64], volumes: &[f64], direction: &TrendDirection) -> f64 {
        if prices.len() != volumes.len() || prices.len() < 2 {
            return 0.5; // Neutral if invalid data
        }

        let price_changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
        let volume_changes: Vec<f64> = volumes.windows(2).map(|w| w[1] - w[0]).collect();

        let mut confirmations = 0;
        let total_periods = price_changes.len();

        for (price_change, volume_change) in price_changes.iter().zip(&volume_changes) {
            let confirmation = match direction {
                TrendDirection::Bullish => *price_change > 0.0 && *volume_change > 0.0,
                TrendDirection::Bearish => *price_change < 0.0 && *volume_change > 0.0,
                TrendDirection::Sideways => volume_change.abs() < volumes.iter().sum::<f64>() / volumes.len() as f64 * 0.1,
                TrendDirection::Unknown => true,
            };

            if confirmation {
                confirmations += 1;
            }
        }

        confirmations as f64 / total_periods as f64
    }

    fn calculate_momentum_confirmation(&self, prices: &[f64], direction: &TrendDirection) -> f64 {
        if prices.len() < self.momentum_indicators.rsi_period + 1 {
            return 0.5;
        }

        // Simple RSI calculation
        let rsi = self.calculate_rsi(prices, self.momentum_indicators.rsi_period);
        
        match direction {
            TrendDirection::Bullish => if rsi > 50.0 && rsi < 80.0 { 0.8 } else { 0.4 },
            TrendDirection::Bearish => if rsi < 50.0 && rsi > 20.0 { 0.8 } else { 0.4 },
            TrendDirection::Sideways => if rsi > 40.0 && rsi < 60.0 { 0.8 } else { 0.4 },
            TrendDirection::Unknown => 0.3,
        }
    }

    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI
        }

        let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
        let recent_changes = &changes[changes.len() - period..];

        let gains: f64 = recent_changes.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = recent_changes.iter().filter(|&&x| x < 0.0).map(|x| x.abs()).sum();

        if losses == 0.0 {
            return 100.0;
        }

        let rs = gains / losses;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_momentum_score(&self, prices: &[f64]) -> f64 {
        if prices.len() < 10 {
            return 0.0;
        }

        let recent_change = (prices[prices.len() - 1] - prices[prices.len() - 5]) / prices[prices.len() - 5];
        let earlier_change = (prices[prices.len() - 5] - prices[prices.len() - 10]) / prices[prices.len() - 10];

        let momentum = if earlier_change != 0.0 {
            (recent_change - earlier_change).abs()
        } else {
            recent_change.abs()
        };

        momentum.min(1.0).max(0.0)
    }

    fn calculate_structure_quality(&self, prices: &[f64]) -> f64 {
        if prices.len() < 5 {
            return 0.0;
        }

        // Count significant highs and lows
        let mut structure_points = 0;
        let total_points = prices.len() - 4;

        for i in 2..prices.len() - 2 {
            let is_high = prices[i] > prices[i-1] && prices[i] > prices[i+1] &&
                         prices[i] > prices[i-2] && prices[i] > prices[i+2];
            let is_low = prices[i] < prices[i-1] && prices[i] < prices[i+1] &&
                        prices[i] < prices[i-2] && prices[i] < prices[i+2];
            
            if is_high || is_low {
                structure_points += 1;
            }
        }

        // Good structure has some significant points but not too many (would indicate choppiness)
        let ideal_ratio = 0.15; // 15% of points being structure points is good
        let actual_ratio = structure_points as f64 / total_points as f64;
        
        1.0 - (actual_ratio - ideal_ratio).abs() / ideal_ratio
    }

    fn analyze_timeframe_window(&self, window: &[f64], age: usize) -> QarResult<TrendInfo> {
        if window.len() < 3 {
            return Ok(TrendInfo {
                direction: TrendDirection::Unknown,
                strength: 0.0,
                age,
                stability: 0.0,
            });
        }

        let direction = self.determine_window_direction(window);
        let strength = self.calculate_window_strength(window, &direction);
        let stability = self.calculate_window_stability(window);

        Ok(TrendInfo {
            direction,
            strength,
            age,
            stability,
        })
    }

    fn determine_window_direction(&self, window: &[f64]) -> TrendDirection {
        if window.len() < 2 {
            return TrendDirection::Unknown;
        }

        let total_change = (window[window.len() - 1] - window[0]) / window[0];
        let avg_price = window.iter().sum::<f64>() / window.len() as f64;
        let price_range = window.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() 
                         - window.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        if total_change > price_range / avg_price * 0.1 {
            TrendDirection::Bullish
        } else if total_change < -price_range / avg_price * 0.1 {
            TrendDirection::Bearish
        } else {
            TrendDirection::Sideways
        }
    }

    fn calculate_window_strength(&self, window: &[f64], direction: &TrendDirection) -> f64 {
        if window.len() < 2 {
            return 0.0;
        }

        let change = (window[window.len() - 1] - window[0]) / window[0];
        match direction {
            TrendDirection::Bullish | TrendDirection::Bearish => change.abs().min(1.0),
            TrendDirection::Sideways => {
                let volatility = self.calculate_volatility(window);
                (1.0 - volatility).max(0.0).min(1.0)
            },
            TrendDirection::Unknown => 0.0,
        }
    }

    fn calculate_window_stability(&self, window: &[f64]) -> f64 {
        if window.len() < 3 {
            return 0.0;
        }

        // Calculate how consistently the window moves in the same direction
        let changes: Vec<f64> = window.windows(2).map(|w| w[1] - w[0]).collect();
        let positive_changes = changes.iter().filter(|&&x| x > 0.0).count();
        let negative_changes = changes.iter().filter(|&&x| x < 0.0).count();
        
        let max_directional = positive_changes.max(negative_changes);
        max_directional as f64 / changes.len() as f64
    }

    fn calculate_timeframe_alignment(&self, short: &TrendInfo, medium: &TrendInfo, long: &TrendInfo) -> f64 {
        let trends = [&short.direction, &medium.direction, &long.direction];
        
        // Count how many trends are in the same direction
        let bullish_count = trends.iter().filter(|&&ref d| *d == TrendDirection::Bullish).count();
        let bearish_count = trends.iter().filter(|&&ref d| *d == TrendDirection::Bearish).count();
        let sideways_count = trends.iter().filter(|&&ref d| *d == TrendDirection::Sideways).count();
        
        let max_alignment = bullish_count.max(bearish_count).max(sideways_count);
        max_alignment as f64 / 3.0
    }

    fn directions_match(&self, dir1: &TrendDirection, dir2: &TrendDirection) -> bool {
        match (dir1, dir2) {
            (TrendDirection::Bullish, TrendDirection::Bullish) => true,
            (TrendDirection::Bearish, TrendDirection::Bearish) => true,
            (TrendDirection::Sideways, TrendDirection::Sideways) => true,
            _ => false,
        }
    }

    fn add_to_history(&mut self, result: TrendResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[TrendResult] {
        &self.history
    }

    /// Get latest analysis
    pub fn get_latest(&self) -> Option<&TrendResult> {
        self.history.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_trend_analyzer() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = TrendAnalyzer::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.8);
        factors.insert(StandardFactors::Momentum.to_string(), 0.7);
        factors.insert(StandardFactors::Volume.to_string(), 0.6);
        factors.insert(StandardFactors::Volatility.to_string(), 0.3);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = analyzer.analyze(&factor_map).await;
        
        assert!(result.is_ok());
        let trend_result = result.unwrap();
        assert!(trend_result.strength >= 0.0 && trend_result.strength <= 1.0);
        assert!(trend_result.confidence >= 0.0 && trend_result.confidence <= 1.0);
    }

    #[test]
    fn test_moving_average_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = TrendAnalyzer::new(config).unwrap();
        
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let ma = analyzer.calculate_moving_average(&prices, 3).unwrap();
        
        // Should be average of last 3 prices: (102 + 103 + 104) / 3 = 103
        assert!((ma - 103.0).abs() < 0.01);
    }

    #[test]
    fn test_trend_direction_determination() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = TrendAnalyzer::new(config).unwrap();
        
        // Strong uptrend
        let uptrend_prices = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0];
        let direction = analyzer.determine_trend_direction(&uptrend_prices).unwrap();
        assert_eq!(direction, TrendDirection::Bullish);

        // Strong downtrend
        let downtrend_prices = vec![118.0, 116.0, 114.0, 112.0, 110.0, 108.0, 106.0, 104.0, 102.0, 100.0];
        let direction = analyzer.determine_trend_direction(&downtrend_prices).unwrap();
        assert_eq!(direction, TrendDirection::Bearish);
    }

    #[test]
    fn test_rsi_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = TrendAnalyzer::new(config).unwrap();
        
        let rising_prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0];
        let rsi = analyzer.calculate_rsi(&rising_prices, 14);
        
        // RSI should be above 50 for rising prices
        assert!(rsi > 50.0);
        assert!(rsi <= 100.0);
    }

    #[test]
    fn test_direction_consistency() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = TrendAnalyzer::new(config).unwrap();
        
        let consistent_up = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let consistency = analyzer.calculate_direction_consistency(&consistent_up, &TrendDirection::Bullish);
        assert!(consistency > 0.8);

        let choppy_prices = vec![100.0, 99.0, 101.0, 98.0, 102.0];
        let consistency = analyzer.calculate_direction_consistency(&choppy_prices, &TrendDirection::Bullish);
        assert!(consistency < 0.6);
    }
}