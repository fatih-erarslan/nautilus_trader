//! Market Analysis Module
//!
//! Core market analysis functionality including price action analysis,
//! volume analysis, and market structure analysis.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use crate::quantum::QuantumState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysisResult {
    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Price trend analysis
    pub price_trend: PriceTrend,
    /// Volume analysis
    pub volume_analysis: VolumeAnalysis,
    /// Market structure
    pub market_structure: MarketStructure,
    /// Support and resistance levels
    pub support_resistance: SupportResistance,
    /// Analysis confidence
    pub confidence: f64,
}

/// Price trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTrend {
    /// Trend direction
    pub direction: super::TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Trend duration in bars
    pub duration: usize,
    /// Price momentum
    pub momentum: f64,
    /// Trend quality score
    pub quality: f64,
}

/// Volume analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAnalysis {
    /// Volume trend
    pub volume_trend: VolumeTrend,
    /// Volume profile
    pub volume_profile: VolumeProfile,
    /// Volume-price relationship
    pub volume_price_divergence: f64,
    /// Average volume ratio
    pub avg_volume_ratio: f64,
}

/// Volume trend enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolumeTrend {
    Increasing,
    Decreasing,
    Stable,
    Diverging,
}

/// Volume profile analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    /// Point of control (highest volume price level)
    pub poc: f64,
    /// Value area high
    pub value_area_high: f64,
    /// Value area low
    pub value_area_low: f64,
    /// Volume distribution
    pub distribution: Vec<(f64, f64)>, // (price, volume)
}

/// Market structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStructure {
    /// Market phase
    pub phase: MarketPhase,
    /// Structure quality
    pub quality: f64,
    /// Break of structure signals
    pub bos_signals: Vec<StructureBreak>,
    /// Change of character signals
    pub choch_signals: Vec<CharacterChange>,
}

/// Market phase enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketPhase {
    Accumulation,
    MarkupTrend,
    Distribution,
    MarkdownTrend,
    Transition,
}

/// Structure break signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureBreak {
    /// Break timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Break price level
    pub price_level: f64,
    /// Break strength
    pub strength: f64,
    /// Break direction
    pub direction: super::TrendDirection,
}

/// Character change signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterChange {
    /// Change timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Previous character
    pub previous_character: MarketCharacter,
    /// New character
    pub new_character: MarketCharacter,
    /// Change confidence
    pub confidence: f64,
}

/// Market character enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketCharacter {
    Bullish,
    Bearish,
    Neutral,
    Choppy,
}

/// Support and resistance levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportResistance {
    /// Support levels
    pub support_levels: Vec<PriceLevel>,
    /// Resistance levels
    pub resistance_levels: Vec<PriceLevel>,
    /// Key levels (high importance)
    pub key_levels: Vec<PriceLevel>,
}

/// Price level information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Price value
    pub price: f64,
    /// Level strength
    pub strength: f64,
    /// Number of touches
    pub touches: usize,
    /// Level age
    pub age: chrono::Duration,
    /// Level type
    pub level_type: LevelType,
}

/// Price level type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LevelType {
    Support,
    Resistance,
    KeyLevel,
    Pivot,
    Fibonacci,
}

/// Market analyzer for comprehensive analysis
pub struct MarketAnalyzer {
    config: super::AnalysisConfig,
    history: Vec<MarketAnalysisResult>,
}

impl MarketAnalyzer {
    /// Create a new market analyzer
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            config,
            history: Vec::new(),
        })
    }

    /// Perform comprehensive market analysis
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<MarketAnalysisResult> {
        let timestamp = chrono::Utc::now();

        // Extract price and volume data from factors
        let price_data = self.extract_price_data(factors)?;
        let volume_data = self.extract_volume_data(factors)?;

        // Perform analysis components
        let price_trend = self.analyze_price_trend(&price_data).await?;
        let volume_analysis = self.analyze_volume(&volume_data, &price_data).await?;
        let market_structure = self.analyze_market_structure(&price_data).await?;
        let support_resistance = self.identify_support_resistance(&price_data).await?;

        // Calculate overall confidence
        let confidence = self.calculate_analysis_confidence(&price_trend, &volume_analysis, &market_structure);

        let result = MarketAnalysisResult {
            timestamp,
            price_trend,
            volume_analysis,
            market_structure,
            support_resistance,
            confidence,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Extract price data from factors
    fn extract_price_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        // In a real implementation, this would extract OHLCV data
        // For now, use trend factor as proxy
        let trend_factor = factors.get_factor(&StandardFactors::Trend)?;
        let momentum_factor = factors.get_factor(&StandardFactors::Momentum)?;
        
        // Generate synthetic price data based on factors
        let mut prices = Vec::new();
        for i in 0..self.config.window_size {
            let base_price = 100.0;
            let trend_component = trend_factor * (i as f64 / self.config.window_size as f64);
            let momentum_component = momentum_factor * (i as f64).sin();
            let noise = (i as f64 * 0.1).sin() * 0.5;
            
            prices.push(base_price + trend_component * 10.0 + momentum_component * 2.0 + noise);
        }
        
        Ok(prices)
    }

    /// Extract volume data from factors
    fn extract_volume_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        // Generate synthetic volume data
        let volume_factor = factors.get_factor(&StandardFactors::Volume)?;
        let mut volumes = Vec::new();
        
        for i in 0..self.config.window_size {
            let base_volume = 1000.0;
            let volume_component = volume_factor * (1.0 + (i as f64 * 0.2).sin() * 0.3);
            volumes.push(base_volume * volume_component);
        }
        
        Ok(volumes)
    }

    /// Analyze price trend
    async fn analyze_price_trend(&self, prices: &[f64]) -> QarResult<PriceTrend> {
        if prices.len() < 2 {
            return Err(QarError::InvalidInput("Insufficient price data".to_string()));
        }

        // Calculate trend direction and strength
        let price_change = prices[prices.len() - 1] - prices[0];
        let price_range = prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() 
                         - prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let direction = if price_change > price_range * 0.1 {
            super::TrendDirection::Bullish
        } else if price_change < -price_range * 0.1 {
            super::TrendDirection::Bearish
        } else {
            super::TrendDirection::Sideways
        };

        let strength = (price_change.abs() / price_range).min(1.0).max(0.0);

        // Calculate momentum using rate of change
        let momentum = if prices.len() >= 10 {
            let recent_change = prices[prices.len() - 1] - prices[prices.len() - 10];
            let earlier_change = prices[prices.len() - 10] - prices[prices.len() - 20.min(prices.len())];
            (recent_change - earlier_change) / price_range
        } else {
            0.0
        };

        let quality = self.calculate_trend_quality(prices, &direction);

        Ok(PriceTrend {
            direction,
            strength,
            duration: prices.len(),
            momentum,
            quality,
        })
    }

    /// Analyze volume patterns
    async fn analyze_volume(&self, volumes: &[f64], prices: &[f64]) -> QarResult<VolumeAnalysis> {
        if volumes.is_empty() || prices.is_empty() {
            return Err(QarError::InvalidInput("Insufficient volume/price data".to_string()));
        }

        let volume_trend = self.determine_volume_trend(volumes);
        let volume_profile = self.calculate_volume_profile(volumes, prices);
        let volume_price_divergence = self.calculate_vp_divergence(volumes, prices);
        let avg_volume_ratio = volumes.iter().sum::<f64>() / volumes.len() as f64 / 1000.0; // Normalized

        Ok(VolumeAnalysis {
            volume_trend,
            volume_profile,
            volume_price_divergence,
            avg_volume_ratio,
        })
    }

    /// Analyze market structure
    async fn analyze_market_structure(&self, prices: &[f64]) -> QarResult<MarketStructure> {
        let phase = self.determine_market_phase(prices);
        let quality = self.calculate_structure_quality(prices);
        let bos_signals = self.detect_structure_breaks(prices);
        let choch_signals = self.detect_character_changes(prices);

        Ok(MarketStructure {
            phase,
            quality,
            bos_signals,
            choch_signals,
        })
    }

    /// Identify support and resistance levels
    async fn identify_support_resistance(&self, prices: &[f64]) -> QarResult<SupportResistance> {
        let support_levels = self.find_support_levels(prices);
        let resistance_levels = self.find_resistance_levels(prices);
        let key_levels = self.identify_key_levels(&support_levels, &resistance_levels);

        Ok(SupportResistance {
            support_levels,
            resistance_levels,
            key_levels,
        })
    }

    /// Calculate trend quality score
    fn calculate_trend_quality(&self, prices: &[f64], direction: &super::TrendDirection) -> f64 {
        if prices.len() < 3 {
            return 0.0;
        }

        let mut quality_score = 0.0;
        let mut consistent_moves = 0;
        let total_moves = prices.len() - 1;

        for i in 1..prices.len() {
            let price_change = prices[i] - prices[i - 1];
            let is_consistent = match direction {
                super::TrendDirection::Bullish => price_change > 0.0,
                super::TrendDirection::Bearish => price_change < 0.0,
                _ => true, // Sideways trends are always "consistent"
            };

            if is_consistent {
                consistent_moves += 1;
            }
        }

        quality_score = consistent_moves as f64 / total_moves as f64;
        quality_score.min(1.0).max(0.0)
    }

    /// Determine volume trend
    fn determine_volume_trend(&self, volumes: &[f64]) -> VolumeTrend {
        if volumes.len() < 2 {
            return VolumeTrend::Stable;
        }

        let first_half_avg = volumes[..volumes.len() / 2].iter().sum::<f64>() / (volumes.len() / 2) as f64;
        let second_half_avg = volumes[volumes.len() / 2..].iter().sum::<f64>() / (volumes.len() - volumes.len() / 2) as f64;

        let change_ratio = (second_half_avg - first_half_avg) / first_half_avg;

        if change_ratio > 0.1 {
            VolumeTrend::Increasing
        } else if change_ratio < -0.1 {
            VolumeTrend::Decreasing
        } else {
            VolumeTrend::Stable
        }
    }

    /// Calculate volume profile
    fn calculate_volume_profile(&self, volumes: &[f64], prices: &[f64]) -> VolumeProfile {
        if volumes.is_empty() || prices.is_empty() {
            return VolumeProfile {
                poc: 0.0,
                value_area_high: 0.0,
                value_area_low: 0.0,
                distribution: Vec::new(),
            };
        }

        // Simplified volume profile calculation
        let min_price = prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_price = prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        let mut distribution = Vec::new();
        let price_bins = 20;
        let bin_size = (max_price - min_price) / price_bins as f64;

        for i in 0..price_bins {
            let price_level = min_price + (i as f64 * bin_size);
            let volume_at_level = volumes.iter().enumerate()
                .filter(|(idx, _)| {
                    let price = prices[*idx];
                    price >= price_level && price < price_level + bin_size
                })
                .map(|(_, vol)| *vol)
                .sum::<f64>();
            
            distribution.push((price_level, volume_at_level));
        }

        // Find POC (highest volume level)
        let poc = distribution.iter()
            .max_by(|(_, vol1), (_, vol2)| vol1.partial_cmp(vol2).unwrap())
            .map(|(price, _)| *price)
            .unwrap_or(0.0);

        VolumeProfile {
            poc,
            value_area_high: max_price * 0.9, // Simplified
            value_area_low: min_price * 1.1,  // Simplified
            distribution,
        }
    }

    /// Calculate volume-price divergence
    fn calculate_vp_divergence(&self, volumes: &[f64], prices: &[f64]) -> f64 {
        if volumes.len() != prices.len() || volumes.len() < 2 {
            return 0.0;
        }

        let price_changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
        let volume_changes: Vec<f64> = volumes.windows(2).map(|w| w[1] - w[0]).collect();

        if price_changes.is_empty() {
            return 0.0;
        }

        // Calculate correlation between price and volume changes
        let price_mean = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
        let volume_mean = volume_changes.iter().sum::<f64>() / volume_changes.len() as f64;

        let numerator: f64 = price_changes.iter().zip(&volume_changes)
            .map(|(p, v)| (p - price_mean) * (v - volume_mean))
            .sum();

        let price_variance: f64 = price_changes.iter().map(|p| (p - price_mean).powi(2)).sum();
        let volume_variance: f64 = volume_changes.iter().map(|v| (v - volume_mean).powi(2)).sum();

        if price_variance == 0.0 || volume_variance == 0.0 {
            0.0
        } else {
            numerator / (price_variance * volume_variance).sqrt()
        }
    }

    /// Additional helper methods
    fn determine_market_phase(&self, prices: &[f64]) -> MarketPhase {
        // Simplified market phase detection
        if prices.len() < 10 {
            return MarketPhase::Transition;
        }

        let recent_volatility = self.calculate_volatility(&prices[prices.len() - 10..]);
        let overall_trend = (prices[prices.len() - 1] - prices[0]) / prices[0];

        if recent_volatility < 0.02 {
            if overall_trend > 0.1 {
                MarketPhase::Accumulation
            } else if overall_trend < -0.1 {
                MarketPhase::Distribution
            } else {
                MarketPhase::Transition
            }
        } else {
            if overall_trend > 0.05 {
                MarketPhase::MarkupTrend
            } else if overall_trend < -0.05 {
                MarketPhase::MarkdownTrend
            } else {
                MarketPhase::Transition
            }
        }
    }

    fn calculate_structure_quality(&self, prices: &[f64]) -> f64 {
        if prices.len() < 5 {
            return 0.0;
        }

        // Calculate structure quality based on higher highs/lower lows consistency
        let mut hh_ll_score = 0.0;
        let mut trend_consistency = 0.0;

        // Simplified quality calculation
        let price_range = prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() 
                         - prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        if price_range > 0.0 {
            let volatility = self.calculate_volatility(prices);
            hh_ll_score = (1.0 - volatility).max(0.0).min(1.0);
        }

        (hh_ll_score + trend_consistency) / 2.0
    }

    fn detect_structure_breaks(&self, prices: &[f64]) -> Vec<StructureBreak> {
        let mut breaks = Vec::new();
        
        // Simplified structure break detection
        if prices.len() < 10 {
            return breaks;
        }

        for i in 10..prices.len() {
            let current_price = prices[i];
            let prev_high = prices[i-10..i].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let prev_low = prices[i-10..i].iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

            if current_price > *prev_high * 1.01 { // 1% break
                breaks.push(StructureBreak {
                    timestamp: chrono::Utc::now(),
                    price_level: current_price,
                    strength: (current_price - prev_high) / prev_high,
                    direction: super::TrendDirection::Bullish,
                });
            } else if current_price < *prev_low * 0.99 { // 1% break
                breaks.push(StructureBreak {
                    timestamp: chrono::Utc::now(),
                    price_level: current_price,
                    strength: (prev_low - current_price) / prev_low,
                    direction: super::TrendDirection::Bearish,
                });
            }
        }

        breaks
    }

    fn detect_character_changes(&self, prices: &[f64]) -> Vec<CharacterChange> {
        let mut changes = Vec::new();
        
        // Simplified character change detection
        if prices.len() < 20 {
            return changes;
        }

        let window_size = 10;
        for i in window_size..prices.len() - window_size {
            let prev_trend = self.analyze_window_character(&prices[i-window_size..i]);
            let curr_trend = self.analyze_window_character(&prices[i..i+window_size]);

            if prev_trend != curr_trend {
                changes.push(CharacterChange {
                    timestamp: chrono::Utc::now(),
                    previous_character: prev_trend,
                    new_character: curr_trend,
                    confidence: 0.8, // Simplified
                });
            }
        }

        changes
    }

    fn analyze_window_character(&self, window: &[f64]) -> MarketCharacter {
        if window.len() < 2 {
            return MarketCharacter::Neutral;
        }

        let trend = (window[window.len() - 1] - window[0]) / window[0];
        let volatility = self.calculate_volatility(window);

        if volatility > 0.05 {
            MarketCharacter::Choppy
        } else if trend > 0.02 {
            MarketCharacter::Bullish
        } else if trend < -0.02 {
            MarketCharacter::Bearish
        } else {
            MarketCharacter::Neutral
        }
    }

    fn find_support_levels(&self, prices: &[f64]) -> Vec<PriceLevel> {
        let mut levels = Vec::new();
        
        // Find local minima as support levels
        for i in 2..prices.len() - 2 {
            if prices[i] < prices[i-1] && prices[i] < prices[i+1] &&
               prices[i] < prices[i-2] && prices[i] < prices[i+2] {
                levels.push(PriceLevel {
                    price: prices[i],
                    strength: 0.7, // Simplified
                    touches: 1,
                    age: chrono::Duration::zero(),
                    level_type: LevelType::Support,
                });
            }
        }

        levels
    }

    fn find_resistance_levels(&self, prices: &[f64]) -> Vec<PriceLevel> {
        let mut levels = Vec::new();
        
        // Find local maxima as resistance levels
        for i in 2..prices.len() - 2 {
            if prices[i] > prices[i-1] && prices[i] > prices[i+1] &&
               prices[i] > prices[i-2] && prices[i] > prices[i+2] {
                levels.push(PriceLevel {
                    price: prices[i],
                    strength: 0.7, // Simplified
                    touches: 1,
                    age: chrono::Duration::zero(),
                    level_type: LevelType::Resistance,
                });
            }
        }

        levels
    }

    fn identify_key_levels(&self, support: &[PriceLevel], resistance: &[PriceLevel]) -> Vec<PriceLevel> {
        let mut key_levels = Vec::new();
        
        // Identify key levels based on strength and confluence
        for level in support.iter().chain(resistance.iter()) {
            if level.strength > 0.8 {
                let mut key_level = level.clone();
                key_level.level_type = LevelType::KeyLevel;
                key_levels.push(key_level);
            }
        }

        key_levels
    }

    fn calculate_volatility(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;

        variance.sqrt()
    }

    fn calculate_analysis_confidence(&self, price_trend: &PriceTrend, volume_analysis: &VolumeAnalysis, market_structure: &MarketStructure) -> f64 {
        let trend_confidence = price_trend.quality;
        let volume_confidence = match volume_analysis.volume_trend {
            VolumeTrend::Increasing | VolumeTrend::Decreasing => 0.8,
            VolumeTrend::Stable => 0.6,
            VolumeTrend::Diverging => 0.4,
        };
        let structure_confidence = market_structure.quality;

        (trend_confidence + volume_confidence + structure_confidence) / 3.0
    }

    fn add_to_history(&mut self, result: MarketAnalysisResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[MarketAnalysisResult] {
        &self.history
    }

    /// Get latest analysis
    pub fn get_latest(&self) -> Option<&MarketAnalysisResult> {
        self.history.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_market_analyzer() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = MarketAnalyzer::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = analyzer.analyze(&factor_map).await;
        
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
    }

    #[test]
    fn test_trend_quality_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = MarketAnalyzer::new(config).unwrap();
        
        let uptrend_prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let quality = analyzer.calculate_trend_quality(&uptrend_prices, &super::super::TrendDirection::Bullish);
        assert!(quality > 0.5);

        let choppy_prices = vec![100.0, 99.0, 101.0, 98.0, 102.0];
        let quality = analyzer.calculate_trend_quality(&choppy_prices, &super::super::TrendDirection::Bullish);
        assert!(quality < 0.5);
    }

    #[test]
    fn test_volume_trend_detection() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = MarketAnalyzer::new(config).unwrap();
        
        let increasing_volume = vec![100.0, 110.0, 120.0, 130.0, 140.0];
        let trend = analyzer.determine_volume_trend(&increasing_volume);
        assert!(matches!(trend, VolumeTrend::Increasing));

        let decreasing_volume = vec![140.0, 130.0, 120.0, 110.0, 100.0];
        let trend = analyzer.determine_volume_trend(&decreasing_volume);
        assert!(matches!(trend, VolumeTrend::Decreasing));
    }

    #[test]
    fn test_market_phase_detection() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = MarketAnalyzer::new(config).unwrap();
        
        let trending_prices = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0];
        let phase = analyzer.determine_market_phase(&trending_prices);
        assert!(matches!(phase, MarketPhase::MarkupTrend));
    }
}