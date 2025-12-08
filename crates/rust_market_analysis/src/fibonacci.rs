//! Fibonacci Pattern Recognition
//! 
//! Advanced detection of Fibonacci retracements, extensions, and geometric patterns
//! in price movements for identifying key support/resistance levels.

use crate::market_data::MarketData;
use anyhow::Result;
use std::collections::HashMap;

pub struct FibonacciAnalyzer {
    fibonacci_ratios: Vec<f64>,
    extension_ratios: Vec<f64>,
    min_swing_size: f64,
    lookback_period: usize,
    confluence_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct FibonacciLevels {
    pub retracement_levels: Vec<FibLevel>,
    pub extension_levels: Vec<FibLevel>,
    pub current_position: f64,
    pub key_level_proximity: f64,
    pub confluence_zones: Vec<ConfluenceZone>,
    pub golden_ratio_signals: Vec<GoldenRatioSignal>,
    pub elliott_wave_context: Option<ElliottWaveContext>,
}

#[derive(Debug, Clone)]
pub struct FibLevel {
    pub price: f64,
    pub ratio: f64,
    pub level_type: String,
    pub strength: f64,
    pub touches: usize,
    pub last_touch_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ConfluenceZone {
    pub price_center: f64,
    pub price_range: (f64, f64),
    pub confluence_count: usize,
    pub strength_score: f64,
    pub level_types: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GoldenRatioSignal {
    pub signal_type: String,
    pub strength: f64,
    pub price_target: f64,
    pub probability: f64,
}

#[derive(Debug, Clone)]
pub struct ElliottWaveContext {
    pub wave_count: usize,
    pub current_wave: String,
    pub fibonacci_confirmation: f64,
}

impl FibonacciAnalyzer {
    pub fn new() -> Self {
        Self {
            fibonacci_ratios: vec![0.236, 0.382, 0.50, 0.618, 0.786],
            extension_ratios: vec![1.272, 1.414, 1.618, 2.0, 2.618],
            min_swing_size: 0.02, // 2% minimum swing
            lookback_period: 200,
            confluence_threshold: 0.005, // 0.5% price range for confluence
        }
    }

    pub fn find_fibonacci_levels(&mut self, data: &MarketData) -> Result<Vec<f64>> {
        let levels = self.analyze_fibonacci_patterns(data)?;
        
        // Extract just the price levels for simple integration
        let mut all_levels = Vec::new();
        all_levels.extend(levels.retracement_levels.iter().map(|l| l.price));
        all_levels.extend(levels.extension_levels.iter().map(|l| l.price));
        
        all_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(all_levels)
    }

    pub fn analyze_fibonacci_patterns(&mut self, data: &MarketData) -> Result<FibonacciLevels> {
        if data.len() < 50 {
            return Ok(FibonacciLevels::default());
        }

        // Find significant swing points
        let swing_points = self.identify_swing_points(data)?;
        
        // Calculate Fibonacci retracements and extensions
        let retracement_levels = self.calculate_retracement_levels(data, &swing_points)?;
        let extension_levels = self.calculate_extension_levels(data, &swing_points)?;
        
        // Find confluence zones
        let confluence_zones = self.find_confluence_zones(&retracement_levels, &extension_levels)?;
        
        // Analyze golden ratio patterns
        let golden_ratio_signals = self.analyze_golden_ratio_patterns(data, &swing_points)?;
        
        // Elliott Wave context
        let elliott_wave_context = self.analyze_elliott_wave_context(data, &swing_points)?;
        
        // Calculate current position relative to key levels
        let current_price = data.prices[data.len() - 1];
        let current_position = self.calculate_current_position(current_price, &retracement_levels);
        let key_level_proximity = self.calculate_key_level_proximity(current_price, &retracement_levels, &extension_levels);

        Ok(FibonacciLevels {
            retracement_levels,
            extension_levels,
            current_position,
            key_level_proximity,
            confluence_zones,
            golden_ratio_signals,
            elliott_wave_context,
        })
    }

    fn identify_swing_points(&self, data: &MarketData) -> Result<Vec<SwingPoint>> {
        let mut swing_points = Vec::new();
        let prices = &data.prices;
        let volumes = &data.volumes;
        
        if prices.len() < 10 {
            return Ok(swing_points);
        }

        // Find local highs and lows
        for i in 5..prices.len()-5 {
            let current = prices[i];
            let is_high = self.is_local_high(prices, i, 5);
            let is_low = self.is_local_low(prices, i, 5);
            
            if is_high || is_low {
                let swing_type = if is_high { SwingType::High } else { SwingType::Low };
                let strength = self.calculate_swing_strength(data, i, swing_type);
                
                // Only include significant swings
                if strength > 0.3 {
                    swing_points.push(SwingPoint {
                        index: i,
                        price: current,
                        swing_type,
                        strength,
                        volume: volumes.get(i).copied().unwrap_or(0.0),
                    });
                }
            }
        }

        // Filter to most significant swings
        swing_points.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        swing_points.truncate(20); // Keep top 20 swings
        swing_points.sort_by_key(|s| s.index); // Sort by time

        Ok(swing_points)
    }

    fn calculate_retracement_levels(&self, data: &MarketData, swings: &[SwingPoint]) -> Result<Vec<FibLevel>> {
        let mut levels = Vec::new();
        
        // Find recent significant move for retracement calculation
        if let Some(major_move) = self.find_major_move(swings) {
            let (start_price, end_price) = major_move;
            let move_size = (end_price - start_price).abs();
            
            // Skip if move is too small
            if move_size / start_price < self.min_swing_size {
                return Ok(levels);
            }

            for &ratio in &self.fibonacci_ratios {
                let retracement_price = if end_price > start_price {
                    // Uptrend retracement
                    end_price - (move_size * ratio)
                } else {
                    // Downtrend retracement
                    end_price + (move_size * ratio)
                };

                let strength = self.calculate_level_strength(data, retracement_price);
                let touches = self.count_level_touches(data, retracement_price, 0.01);

                levels.push(FibLevel {
                    price: retracement_price,
                    ratio,
                    level_type: "retracement".to_string(),
                    strength,
                    touches,
                    last_touch_index: None,
                });
            }
        }

        Ok(levels)
    }

    fn calculate_extension_levels(&self, data: &MarketData, swings: &[SwingPoint]) -> Result<Vec<FibLevel>> {
        let mut levels = Vec::new();
        
        // Find AB-CD pattern or recent impulse move
        if let Some(extension_base) = self.find_extension_base(swings) {
            let (start_price, end_price, direction) = extension_base;
            let move_size = (end_price - start_price).abs();
            
            for &ratio in &self.extension_ratios {
                let extension_price = if direction > 0.0 {
                    end_price + (move_size * ratio)
                } else {
                    end_price - (move_size * ratio)
                };

                let strength = self.calculate_level_strength(data, extension_price);
                let touches = self.count_level_touches(data, extension_price, 0.01);

                levels.push(FibLevel {
                    price: extension_price,
                    ratio,
                    level_type: "extension".to_string(),
                    strength,
                    touches,
                    last_touch_index: None,
                });
            }
        }

        Ok(levels)
    }

    fn find_confluence_zones(&self, retracements: &[FibLevel], extensions: &[FibLevel]) -> Result<Vec<ConfluenceZone>> {
        let mut confluence_zones = Vec::new();
        let mut all_levels = Vec::new();
        
        // Combine all levels
        for level in retracements {
            all_levels.push((level.price, "retracement".to_string(), level.strength));
        }
        for level in extensions {
            all_levels.push((level.price, "extension".to_string(), level.strength));
        }

        // Sort by price
        all_levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find confluence areas
        for i in 0..all_levels.len() {
            let (base_price, base_type, base_strength) = &all_levels[i];
            let mut confluence_levels = vec![(base_price.clone(), base_type.clone())];
            let mut total_strength = *base_strength;

            // Check nearby levels
            for j in (i+1)..all_levels.len() {
                let (price, level_type, strength) = &all_levels[j];
                let price_diff_pct = (price - base_price).abs() / base_price;
                
                if price_diff_pct <= self.confluence_threshold {
                    confluence_levels.push((price.clone(), level_type.clone()));
                    total_strength += strength;
                } else {
                    break; // Levels are sorted, so we can break
                }
            }

            // Create confluence zone if multiple levels are close
            if confluence_levels.len() >= 2 {
                let prices: Vec<f64> = confluence_levels.iter().map(|(p, _)| *p).collect();
                let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let center_price = (min_price + max_price) / 2.0;

                confluence_zones.push(ConfluenceZone {
                    price_center: center_price,
                    price_range: (min_price, max_price),
                    confluence_count: confluence_levels.len(),
                    strength_score: total_strength / confluence_levels.len() as f64,
                    level_types: confluence_levels.into_iter().map(|(_, t)| t).collect(),
                });
            }
        }

        // Sort by strength
        confluence_zones.sort_by(|a, b| b.strength_score.partial_cmp(&a.strength_score).unwrap());
        confluence_zones.truncate(5); // Keep top 5 confluence zones

        Ok(confluence_zones)
    }

    fn analyze_golden_ratio_patterns(&self, data: &MarketData, swings: &[SwingPoint]) -> Result<Vec<GoldenRatioSignal>> {
        let mut signals = Vec::new();
        
        // Golden ratio (1.618) is key in Fibonacci analysis
        let golden_ratio = 1.618;
        let current_price = data.prices[data.len() - 1];

        // Look for golden ratio relationships in price movements
        for i in 0..swings.len().saturating_sub(2) {
            let swing1 = &swings[i];
            let swing2 = &swings[i + 1];
            let swing3 = &swings[i + 2];

            // Check if moves follow golden ratio proportions
            let move1 = (swing2.price - swing1.price).abs();
            let move2 = (swing3.price - swing2.price).abs();

            if move1 > 0.0 {
                let ratio = move2 / move1;
                
                // Golden ratio extension pattern
                if (ratio - golden_ratio).abs() < 0.1 {
                    let target_price = if swing3.price > swing2.price {
                        swing3.price + move2 * golden_ratio
                    } else {
                        swing3.price - move2 * golden_ratio
                    };

                    signals.push(GoldenRatioSignal {
                        signal_type: "golden_extension".to_string(),
                        strength: 0.8,
                        price_target: target_price,
                        probability: 0.7,
                    });
                }

                // Golden ratio retracement pattern
                if (ratio - (1.0 / golden_ratio)).abs() < 0.05 {
                    signals.push(GoldenRatioSignal {
                        signal_type: "golden_retracement".to_string(),
                        strength: 0.6,
                        price_target: swing2.price,
                        probability: 0.6,
                    });
                }
            }
        }

        Ok(signals)
    }

    fn analyze_elliott_wave_context(&self, data: &MarketData, swings: &[SwingPoint]) -> Result<Option<ElliottWaveContext>> {
        if swings.len() < 5 {
            return Ok(None);
        }

        // Simple Elliott Wave counting (5-wave impulse pattern)
        let mut wave_count = 0;
        let mut fibonacci_confirmation: f64 = 0.0;

        // Check if recent swings follow Elliott Wave proportions
        for i in 0..swings.len().saturating_sub(4) {
            let waves = &swings[i..i+5];
            
            // Wave relationships in Elliott Wave theory
            let wave1 = (waves[1].price - waves[0].price).abs();
            let wave3 = (waves[3].price - waves[2].price).abs();
            let wave5 = (waves[4].price - waves[3].price).abs();

            if wave1 > 0.0 && wave3 > 0.0 {
                // Wave 3 should be 1.618 * Wave 1 (ideally)
                let wave3_ratio = wave3 / wave1;
                if (wave3_ratio - 1.618).abs() < 0.3 {
                    fibonacci_confirmation += 0.4;
                }

                // Wave 5 should be 0.618 * Wave 1 or equal to Wave 1
                if wave5 > 0.0 {
                    let wave5_ratio = wave5 / wave1;
                    if (wave5_ratio - 0.618).abs() < 0.2 || (wave5_ratio - 1.0).abs() < 0.2 {
                        fibonacci_confirmation += 0.3;
                    }
                }
            }

            wave_count = 5; // Found potential 5-wave pattern
        }

        let current_wave = if wave_count >= 5 {
            "wave_5_completion"
        } else {
            "wave_development"
        }.to_string();

        Ok(Some(ElliottWaveContext {
            wave_count,
            current_wave,
            fibonacci_confirmation: fibonacci_confirmation.min(1.0),
        }))
    }

    // Helper methods
    fn is_local_high(&self, prices: &[f64], index: usize, lookback: usize) -> bool {
        let current = prices[index];
        let start = index.saturating_sub(lookback);
        let end = (index + lookback + 1).min(prices.len());

        for i in start..end {
            if i != index && prices[i] >= current {
                return false;
            }
        }
        true
    }

    fn is_local_low(&self, prices: &[f64], index: usize, lookback: usize) -> bool {
        let current = prices[index];
        let start = index.saturating_sub(lookback);
        let end = (index + lookback + 1).min(prices.len());

        for i in start..end {
            if i != index && prices[i] <= current {
                return false;
            }
        }
        true
    }

    fn calculate_swing_strength(&self, data: &MarketData, index: usize, swing_type: SwingType) -> f64 {
        // Strength based on price range and volume
        let price_range = self.calculate_local_price_range(data, index, 10);
        let volume_strength = self.calculate_volume_strength(data, index);
        let duration_strength = self.calculate_duration_strength(data, index, swing_type);

        (price_range * 0.4 + volume_strength * 0.3 + duration_strength * 0.3).min(1.0)
    }

    fn find_major_move(&self, swings: &[SwingPoint]) -> Option<(f64, f64)> {
        if swings.len() < 2 {
            return None;
        }

        // Find the most recent significant move
        let mut max_move_size = 0.0;
        let mut major_move = None;

        for i in 0..swings.len()-1 {
            let start = &swings[i];
            let end = &swings[i + 1];
            let move_size = (end.price - start.price).abs();

            if move_size > max_move_size {
                max_move_size = move_size;
                major_move = Some((start.price, end.price));
            }
        }

        major_move
    }

    fn find_extension_base(&self, swings: &[SwingPoint]) -> Option<(f64, f64, f64)> {
        if swings.len() < 3 {
            return None;
        }

        // Use last significant move for extension calculation
        let len = swings.len();
        let start = &swings[len - 3];
        let end = &swings[len - 1];
        let direction = if end.price > start.price { 1.0 } else { -1.0 };

        Some((start.price, end.price, direction))
    }

    fn calculate_level_strength(&self, data: &MarketData, level_price: f64) -> f64 {
        // Strength based on how often price respected this level
        let touches = self.count_level_touches(data, level_price, 0.01);
        let recency = self.calculate_level_recency(data, level_price);
        
        let touch_strength = (touches as f64 / 10.0).min(1.0);
        (touch_strength * 0.7 + recency * 0.3).min(1.0)
    }

    fn count_level_touches(&self, data: &MarketData, level_price: f64, tolerance: f64) -> usize {
        let mut touches = 0;
        let tolerance_range = level_price * tolerance;

        for (i, &price) in data.prices.iter().enumerate() {
            if (price - level_price).abs() <= tolerance_range {
                // Check if this was a turning point
                if i > 5 && i < data.prices.len() - 5 {
                    let before_avg = data.prices[i-5..i].iter().sum::<f64>() / 5.0;
                    let after_avg = data.prices[i+1..i+6].iter().sum::<f64>() / 5.0;
                    
                    // Check for bounce or rejection
                    if (before_avg > level_price && after_avg < level_price) ||
                       (before_avg < level_price && after_avg > level_price) {
                        touches += 1;
                    }
                }
            }
        }

        touches
    }

    fn calculate_level_recency(&self, data: &MarketData, level_price: f64) -> f64 {
        let current_price = data.prices[data.len() - 1];
        let distance = (current_price - level_price).abs() / current_price;
        
        // Closer levels are more relevant
        (1.0 - distance.min(1.0)).max(0.0)
    }

    fn calculate_current_position(&self, current_price: f64, levels: &[FibLevel]) -> f64 {
        if levels.is_empty() {
            return 0.5;
        }

        // Find position relative to Fibonacci levels (0.0 = bottom, 1.0 = top)
        let min_level = levels.iter().map(|l| l.price).fold(f64::INFINITY, f64::min);
        let max_level = levels.iter().map(|l| l.price).fold(f64::NEG_INFINITY, f64::max);

        if max_level > min_level {
            ((current_price - min_level) / (max_level - min_level)).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    fn calculate_key_level_proximity(&self, current_price: f64, retracements: &[FibLevel], extensions: &[FibLevel]) -> f64 {
        let mut min_distance = f64::INFINITY;

        // Find closest key level
        for level in retracements.iter().chain(extensions.iter()) {
            let distance = (current_price - level.price).abs() / current_price;
            if distance < min_distance {
                min_distance = distance;
            }
        }

        // Return proximity score (higher = closer to key level)
        if min_distance == f64::INFINITY {
            0.0
        } else {
            (1.0 - min_distance.min(1.0)).max(0.0)
        }
    }

    fn calculate_local_price_range(&self, data: &MarketData, index: usize, window: usize) -> f64 {
        let start = index.saturating_sub(window);
        let end = (index + window + 1).min(data.prices.len());
        let window_prices = &data.prices[start..end];

        if window_prices.is_empty() {
            return 0.0;
        }

        let min_price = window_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = window_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let current_price = data.prices[index];

        if max_price > min_price {
            (max_price - min_price) / current_price
        } else {
            0.0
        }
    }

    fn calculate_volume_strength(&self, data: &MarketData, index: usize) -> f64 {
        if data.volumes.is_empty() || index >= data.volumes.len() {
            return 0.5;
        }

        let current_volume = data.volumes[index];
        let start = index.saturating_sub(20);
        let window_volumes = &data.volumes[start..index];

        if window_volumes.is_empty() {
            return 0.5;
        }

        let avg_volume = window_volumes.iter().sum::<f64>() / window_volumes.len() as f64;
        
        if avg_volume > 0.0 {
            (current_volume / avg_volume / 3.0).min(1.0) // Normalize to 0-1
        } else {
            0.5
        }
    }

    fn calculate_duration_strength(&self, data: &MarketData, index: usize, swing_type: SwingType) -> f64 {
        // Longer duration swings are generally stronger
        let window = 10;
        let start = index.saturating_sub(window);
        let end = (index + window + 1).min(data.prices.len());
        
        let duration = end - start;
        (duration as f64 / (window * 2) as f64).min(1.0)
    }
}

// Supporting structures
#[derive(Debug, Clone)]
struct SwingPoint {
    index: usize,
    price: f64,
    swing_type: SwingType,
    strength: f64,
    volume: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SwingType {
    High,
    Low,
}

impl Default for FibonacciLevels {
    fn default() -> Self {
        Self {
            retracement_levels: Vec::new(),
            extension_levels: Vec::new(),
            current_position: 0.5,
            key_level_proximity: 0.0,
            confluence_zones: Vec::new(),
            golden_ratio_signals: Vec::new(),
            elliott_wave_context: None,
        }
    }
}

impl Default for FibonacciAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}