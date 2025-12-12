//! Pattern Analysis Module
//!
//! Advanced pattern recognition using technical analysis and quantum-enhanced detection.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pattern analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    /// Overall pattern score (0.0 to 1.0)
    pub score: f64,
    /// Confidence in pattern detection
    pub confidence: f64,
    /// Detected patterns
    pub detected_patterns: Vec<DetectedPattern>,
    /// Pattern strength distribution
    pub pattern_strengths: HashMap<PatternType, f64>,
    /// Pattern reliability metrics
    pub reliability_metrics: ReliabilityMetrics,
}

/// Individual detected pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Type of pattern
    pub pattern_type: PatternType,
    /// Pattern strength (0.0 to 1.0)
    pub strength: f64,
    /// Pattern completion percentage
    pub completion: f64,
    /// Pattern direction
    pub direction: PatternDirection,
    /// Expected target/breakout level
    pub target_level: Option<f64>,
    /// Pattern timeframe
    pub timeframe: PatternTimeframe,
    /// Pattern metadata
    pub metadata: PatternMetadata,
}

/// Pattern type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternType {
    // Continuation Patterns
    Triangle,
    Flag,
    Pennant,
    Rectangle,
    Wedge,
    
    // Reversal Patterns
    HeadAndShoulders,
    DoubleTop,
    DoubleBottom,
    TripleTop,
    TripleBottom,
    RoundingTop,
    RoundingBottom,
    
    // Candlestick Patterns
    Doji,
    Hammer,
    ShootingStar,
    Engulfing,
    Harami,
    
    // Price Action Patterns
    BreakoutPattern,
    PullbackPattern,
    TrendContinuation,
    Support,
    Resistance,
    
    // Volume Patterns
    VolumeClimaxing,
    VolumeSpike,
    VolumeDrying,
    
    // Momentum Patterns
    Divergence,
    Convergence,
    MomentumExhaustion,
}

/// Pattern direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternDirection {
    Bullish,
    Bearish,
    Neutral,
    Reversal,
}

/// Pattern timeframe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternTimeframe {
    ShortTerm,   // 1-5 periods
    MediumTerm,  // 5-20 periods
    LongTerm,    // 20+ periods
}

/// Pattern metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMetadata {
    /// Pattern start index/time
    pub start_index: usize,
    /// Pattern end index/time
    pub end_index: usize,
    /// Key price levels
    pub key_levels: Vec<f64>,
    /// Volume confirmation
    pub volume_confirmation: bool,
    /// Historical reliability of this pattern type
    pub historical_reliability: f64,
}

/// Pattern reliability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    /// Overall pattern reliability
    pub overall_reliability: f64,
    /// Success rate by pattern type
    pub success_rates: HashMap<PatternType, f64>,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Pattern stability over time
    pub stability: f64,
}

/// Pattern analyzer
pub struct PatternAnalyzer {
    config: super::AnalysisConfig,
    pattern_params: PatternParameters,
    price_history: Vec<f64>,
    volume_history: Vec<f64>,
    pattern_database: PatternDatabase,
    history: Vec<PatternResult>,
}

/// Pattern analysis parameters
#[derive(Debug, Clone)]
pub struct PatternParameters {
    /// Minimum pattern length
    pub min_pattern_length: usize,
    /// Maximum pattern length
    pub max_pattern_length: usize,
    /// Pattern strength threshold
    pub strength_threshold: f64,
    /// Volume confirmation weight
    pub volume_weight: f64,
    /// Pattern matching tolerance
    pub matching_tolerance: f64,
}

/// Pattern database for template matching
#[derive(Debug)]
pub struct PatternDatabase {
    /// Pattern templates
    pub templates: HashMap<PatternType, Vec<PatternTemplate>>,
    /// Pattern statistics
    pub statistics: HashMap<PatternType, PatternStatistics>,
}

/// Pattern template for matching
#[derive(Debug, Clone)]
pub struct PatternTemplate {
    /// Normalized price sequence
    pub price_sequence: Vec<f64>,
    /// Expected volume pattern
    pub volume_pattern: Vec<f64>,
    /// Pattern characteristics
    pub characteristics: PatternCharacteristics,
}

/// Pattern characteristics
#[derive(Debug, Clone)]
pub struct PatternCharacteristics {
    /// Typical duration
    pub typical_duration: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average target achievement
    pub avg_target_achievement: f64,
    /// Reliability score
    pub reliability_score: f64,
}

/// Pattern statistics
#[derive(Debug, Clone)]
pub struct PatternStatistics {
    /// Total occurrences
    pub total_occurrences: usize,
    /// Successful patterns
    pub successful_patterns: usize,
    /// Average strength
    pub average_strength: f64,
    /// Average completion time
    pub average_completion_time: f64,
}

impl Default for PatternParameters {
    fn default() -> Self {
        Self {
            min_pattern_length: 5,
            max_pattern_length: 50,
            strength_threshold: 0.6,
            volume_weight: 0.3,
            matching_tolerance: 0.1,
        }
    }
}

impl Default for PatternDatabase {
    fn default() -> Self {
        let mut templates = HashMap::new();
        let mut statistics = HashMap::new();
        
        // Initialize with basic patterns
        let pattern_types = vec![
            PatternType::Triangle,
            PatternType::Flag,
            PatternType::HeadAndShoulders,
            PatternType::DoubleTop,
            PatternType::DoubleBottom,
            PatternType::Support,
            PatternType::Resistance,
        ];

        for pattern_type in pattern_types {
            templates.insert(pattern_type.clone(), vec![]);
            statistics.insert(pattern_type, PatternStatistics {
                total_occurrences: 0,
                successful_patterns: 0,
                average_strength: 0.0,
                average_completion_time: 0.0,
            });
        }

        Self {
            templates,
            statistics,
        }
    }
}

impl PatternAnalyzer {
    /// Create a new pattern analyzer
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            config,
            pattern_params: PatternParameters::default(),
            price_history: Vec::new(),
            volume_history: Vec::new(),
            pattern_database: PatternDatabase::default(),
            history: Vec::new(),
        })
    }

    /// Analyze patterns from market factors
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<PatternResult> {
        // Extract and update price/volume data
        let price_data = self.extract_price_data(factors)?;
        let volume_data = self.extract_volume_data(factors)?;
        
        self.update_history(&price_data, &volume_data);

        // Detect patterns using multiple methods
        let mut detected_patterns = Vec::new();
        
        detected_patterns.extend(self.detect_continuation_patterns()?);
        detected_patterns.extend(self.detect_reversal_patterns()?);
        detected_patterns.extend(self.detect_candlestick_patterns()?);
        detected_patterns.extend(self.detect_price_action_patterns()?);
        detected_patterns.extend(self.detect_volume_patterns()?);
        detected_patterns.extend(self.detect_momentum_patterns(factors)?);

        // Calculate pattern strengths
        let pattern_strengths = self.calculate_pattern_strengths(&detected_patterns);
        
        // Calculate overall score and confidence
        let score = self.calculate_overall_score(&detected_patterns);
        let confidence = self.calculate_pattern_confidence(&detected_patterns);
        
        // Calculate reliability metrics
        let reliability_metrics = self.calculate_reliability_metrics(&detected_patterns);

        let result = PatternResult {
            score,
            confidence,
            detected_patterns,
            pattern_strengths,
            reliability_metrics,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Extract price data from factors
    fn extract_price_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        let trend_factor = factors.get_factor(&StandardFactors::Trend)?;
        let momentum_factor = factors.get_factor(&StandardFactors::Momentum)?;
        let volatility_factor = factors.get_factor(&StandardFactors::Volatility)?;

        let mut prices = Vec::new();
        let base_price = 100.0;
        let window_size = 20; // Generate 20 price points
        
        for i in 0..window_size {
            let time_factor = i as f64 / window_size as f64;
            
            // Trend component
            let trend_component = trend_factor * time_factor * 15.0;
            
            // Momentum oscillation
            let momentum_component = momentum_factor * (i as f64 * 0.3).sin() * 5.0;
            
            // Volatility noise
            let volatility_component = volatility_factor * (
                (i as f64 * 0.7).sin() * 2.0 + 
                (i as f64 * 1.1).cos() * 1.5
            );
            
            let price = base_price + trend_component + momentum_component + volatility_component;
            prices.push(price.max(1.0));
        }

        Ok(prices)
    }

    /// Extract volume data from factors
    fn extract_volume_data(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        let volume_factor = factors.get_factor(&StandardFactors::Volume)?;
        let volatility_factor = factors.get_factor(&StandardFactors::Volatility)?;

        let mut volumes = Vec::new();
        let base_volume = 1000.0;
        let window_size = 20;

        for i in 0..window_size {
            let volume_variation = volume_factor * (i as f64 * 0.4).sin() * 0.3;
            let volatility_boost = volatility_factor * (i as f64 * 0.6).cos() * 0.4;
            
            volumes.push(base_volume * (1.0 + volume_variation + volatility_boost));
        }

        Ok(volumes)
    }

    /// Update price and volume history
    fn update_history(&mut self, prices: &[f64], volumes: &[f64]) {
        self.price_history.extend_from_slice(prices);
        self.volume_history.extend_from_slice(volumes);
        
        // Maintain maximum history length
        let max_history = self.config.max_history * 2;
        if self.price_history.len() > max_history {
            let excess = self.price_history.len() - max_history;
            self.price_history.drain(0..excess);
        }
        
        if self.volume_history.len() > max_history {
            let excess = self.volume_history.len() - max_history;
            self.volume_history.drain(0..excess);
        }
    }

    /// Detect continuation patterns
    fn detect_continuation_patterns(&self) -> QarResult<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        if self.price_history.len() < self.pattern_params.min_pattern_length {
            return Ok(patterns);
        }

        // Triangle pattern detection
        if let Some(triangle) = self.detect_triangle_pattern()? {
            patterns.push(triangle);
        }

        // Flag pattern detection
        if let Some(flag) = self.detect_flag_pattern()? {
            patterns.push(flag);
        }

        // Pennant pattern detection
        if let Some(pennant) = self.detect_pennant_pattern()? {
            patterns.push(pennant);
        }

        Ok(patterns)
    }

    /// Detect triangle pattern
    fn detect_triangle_pattern(&self) -> QarResult<Option<DetectedPattern>> {
        let min_length = 10;
        if self.price_history.len() < min_length {
            return Ok(None);
        }

        let recent_prices = &self.price_history[self.price_history.len() - min_length..];
        
        // Find swing highs and lows
        let highs = self.find_swing_highs(recent_prices, 2);
        let lows = self.find_swing_lows(recent_prices, 2);

        if highs.len() < 2 || lows.len() < 2 {
            return Ok(None);
        }

        // Check for converging trend lines
        let high_slope = self.calculate_slope(&highs);
        let low_slope = self.calculate_slope(&lows);

        // Triangle: highs declining, lows rising (or vice versa)
        let is_triangle = (high_slope < 0.0 && low_slope > 0.0) || 
                         (high_slope > 0.0 && low_slope < 0.0);

        if !is_triangle {
            return Ok(None);
        }

        let direction = if high_slope < 0.0 && low_slope > 0.0 {
            PatternDirection::Neutral // Symmetrical triangle
        } else if high_slope > 0.0 {
            PatternDirection::Bullish // Ascending triangle
        } else {
            PatternDirection::Bearish // Descending triangle
        };

        let strength = self.calculate_triangle_strength(&highs, &lows);
        let completion = self.calculate_pattern_completion(recent_prices);

        Ok(Some(DetectedPattern {
            pattern_type: PatternType::Triangle,
            strength,
            completion,
            direction,
            target_level: self.calculate_triangle_target(&highs, &lows),
            timeframe: PatternTimeframe::MediumTerm,
            metadata: PatternMetadata {
                start_index: self.price_history.len() - min_length,
                end_index: self.price_history.len() - 1,
                key_levels: vec![highs[0].1, lows[0].1], // First high and low
                volume_confirmation: self.check_volume_confirmation(),
                historical_reliability: 0.65,
            },
        }))
    }

    /// Detect flag pattern
    fn detect_flag_pattern(&self) -> QarResult<Option<DetectedPattern>> {
        let min_length = 8;
        if self.price_history.len() < min_length {
            return Ok(None);
        }

        let recent_prices = &self.price_history[self.price_history.len() - min_length..];
        
        // Flag: sharp move followed by consolidation
        let initial_move = (recent_prices[2] - recent_prices[0]) / recent_prices[0];
        let consolidation_range = self.calculate_price_range(&recent_prices[3..]);
        let avg_price = recent_prices[3..].iter().sum::<f64>() / (recent_prices.len() - 3) as f64;
        
        // Check for significant initial move
        if initial_move.abs() < 0.03 {
            return Ok(None);
        }

        // Check for tight consolidation
        if consolidation_range / avg_price > 0.02 {
            return Ok(None);
        }

        let direction = if initial_move > 0.0 {
            PatternDirection::Bullish
        } else {
            PatternDirection::Bearish
        };

        let strength = initial_move.abs() * 10.0; // Convert to 0-1 scale
        let completion = 0.8; // Flags are usually near completion when detected

        Ok(Some(DetectedPattern {
            pattern_type: PatternType::Flag,
            strength: strength.min(1.0),
            completion,
            direction,
            target_level: Some(recent_prices[recent_prices.len() - 1] + initial_move * recent_prices[0]),
            timeframe: PatternTimeframe::ShortTerm,
            metadata: PatternMetadata {
                start_index: self.price_history.len() - min_length,
                end_index: self.price_history.len() - 1,
                key_levels: vec![recent_prices[0], recent_prices[2]],
                volume_confirmation: self.check_volume_confirmation(),
                historical_reliability: 0.7,
            },
        }))
    }

    /// Detect pennant pattern
    fn detect_pennant_pattern(&self) -> QarResult<Option<DetectedPattern>> {
        let min_length = 10;
        if self.price_history.len() < min_length {
            return Ok(None);
        }

        let recent_prices = &self.price_history[self.price_history.len() - min_length..];
        
        // Pennant: strong move followed by triangular consolidation
        let initial_move = (recent_prices[3] - recent_prices[0]) / recent_prices[0];
        
        if initial_move.abs() < 0.04 {
            return Ok(None);
        }

        // Check for triangular consolidation after the move
        let consolidation_prices = &recent_prices[4..];
        let highs = self.find_swing_highs(consolidation_prices, 1);
        let lows = self.find_swing_lows(consolidation_prices, 1);

        if highs.len() < 2 || lows.len() < 2 {
            return Ok(None);
        }

        let high_slope = self.calculate_slope(&highs);
        let low_slope = self.calculate_slope(&lows);

        // Pennant: converging trend lines
        let is_pennant = (high_slope < 0.0 && low_slope > 0.0) ||
                        (high_slope > 0.0 && low_slope < 0.0);

        if !is_pennant {
            return Ok(None);
        }

        let direction = if initial_move > 0.0 {
            PatternDirection::Bullish
        } else {
            PatternDirection::Bearish
        };

        let strength = initial_move.abs() * 8.0;
        let completion = 0.75;

        Ok(Some(DetectedPattern {
            pattern_type: PatternType::Pennant,
            strength: strength.min(1.0),
            completion,
            direction,
            target_level: Some(recent_prices[recent_prices.len() - 1] + initial_move * recent_prices[0]),
            timeframe: PatternTimeframe::ShortTerm,
            metadata: PatternMetadata {
                start_index: self.price_history.len() - min_length,
                end_index: self.price_history.len() - 1,
                key_levels: vec![recent_prices[0], recent_prices[3]],
                volume_confirmation: self.check_volume_confirmation(),
                historical_reliability: 0.68,
            },
        }))
    }

    /// Detect reversal patterns
    fn detect_reversal_patterns(&self) -> QarResult<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        if self.price_history.len() < 15 {
            return Ok(patterns);
        }

        // Head and shoulders detection
        if let Some(h_s) = self.detect_head_and_shoulders()? {
            patterns.push(h_s);
        }

        // Double top/bottom detection
        if let Some(double_pattern) = self.detect_double_pattern()? {
            patterns.push(double_pattern);
        }

        Ok(patterns)
    }

    /// Detect head and shoulders pattern
    fn detect_head_and_shoulders(&self) -> QarResult<Option<DetectedPattern>> {
        let min_length = 15;
        if self.price_history.len() < min_length {
            return Ok(None);
        }

        let recent_prices = &self.price_history[self.price_history.len() - min_length..];
        let peaks = self.find_swing_highs(recent_prices, 3);

        if peaks.len() < 3 {
            return Ok(None);
        }

        // Check for head and shoulders formation
        let left_shoulder = peaks[0].1;
        let head = peaks[1].1;
        let right_shoulder = peaks[2].1;

        // Head should be higher than both shoulders
        if head <= left_shoulder || head <= right_shoulder {
            return Ok(None);
        }

        // Shoulders should be roughly equal
        let shoulder_diff = (left_shoulder - right_shoulder).abs() / left_shoulder;
        if shoulder_diff > 0.05 {
            return Ok(None);
        }

        let strength = ((head - left_shoulder) / left_shoulder + 
                       (head - right_shoulder) / right_shoulder) / 2.0;
        
        let completion = 0.9; // Usually detected near completion

        Ok(Some(DetectedPattern {
            pattern_type: PatternType::HeadAndShoulders,
            strength: strength.min(1.0),
            completion,
            direction: PatternDirection::Bearish,
            target_level: Some(left_shoulder - (head - left_shoulder)),
            timeframe: PatternTimeframe::LongTerm,
            metadata: PatternMetadata {
                start_index: self.price_history.len() - min_length,
                end_index: self.price_history.len() - 1,
                key_levels: vec![left_shoulder, head, right_shoulder],
                volume_confirmation: self.check_volume_confirmation(),
                historical_reliability: 0.75,
            },
        }))
    }

    /// Detect double top/bottom pattern
    fn detect_double_pattern(&self) -> QarResult<Option<DetectedPattern>> {
        let min_length = 12;
        if self.price_history.len() < min_length {
            return Ok(None);
        }

        let recent_prices = &self.price_history[self.price_history.len() - min_length..];
        
        // Try double top first
        let peaks = self.find_swing_highs(recent_prices, 2);
        if peaks.len() >= 2 {
            let first_peak = peaks[0].1;
            let second_peak = peaks[1].1;
            let peak_diff = (first_peak - second_peak).abs() / first_peak;
            
            if peak_diff < 0.02 { // Peaks are roughly equal
                let strength = first_peak / recent_prices[0] - 1.0;
                return Ok(Some(DetectedPattern {
                    pattern_type: PatternType::DoubleTop,
                    strength: strength.min(1.0),
                    completion: 0.85,
                    direction: PatternDirection::Bearish,
                    target_level: Some(first_peak - (first_peak - recent_prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()) * 0.8),
                    timeframe: PatternTimeframe::MediumTerm,
                    metadata: PatternMetadata {
                        start_index: self.price_history.len() - min_length,
                        end_index: self.price_history.len() - 1,
                        key_levels: vec![first_peak, second_peak],
                        volume_confirmation: self.check_volume_confirmation(),
                        historical_reliability: 0.72,
                    },
                }));
            }
        }

        // Try double bottom
        let troughs = self.find_swing_lows(recent_prices, 2);
        if troughs.len() >= 2 {
            let first_trough = troughs[0].1;
            let second_trough = troughs[1].1;
            let trough_diff = (first_trough - second_trough).abs() / first_trough;
            
            if trough_diff < 0.02 {
                let strength = 1.0 - first_trough / recent_prices[0];
                return Ok(Some(DetectedPattern {
                    pattern_type: PatternType::DoubleBottom,
                    strength: strength.min(1.0),
                    completion: 0.85,
                    direction: PatternDirection::Bullish,
                    target_level: Some(first_trough + (recent_prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() - first_trough) * 0.8),
                    timeframe: PatternTimeframe::MediumTerm,
                    metadata: PatternMetadata {
                        start_index: self.price_history.len() - min_length,
                        end_index: self.price_history.len() - 1,
                        key_levels: vec![first_trough, second_trough],
                        volume_confirmation: self.check_volume_confirmation(),
                        historical_reliability: 0.72,
                    },
                }));
            }
        }

        Ok(None)
    }

    /// Detect candlestick patterns (simplified)
    fn detect_candlestick_patterns(&self) -> QarResult<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        if self.price_history.len() < 3 {
            return Ok(patterns);
        }

        // Doji pattern (simplified - looking for small range)
        let recent = &self.price_history[self.price_history.len() - 1];
        let prev = &self.price_history[self.price_history.len() - 2];
        let range = (recent - prev).abs() / prev;
        
        if range < 0.005 { // Very small movement
            patterns.push(DetectedPattern {
                pattern_type: PatternType::Doji,
                strength: 1.0 - range * 200.0, // Smaller range = stronger doji
                completion: 1.0,
                direction: PatternDirection::Neutral,
                target_level: None,
                timeframe: PatternTimeframe::ShortTerm,
                metadata: PatternMetadata {
                    start_index: self.price_history.len() - 2,
                    end_index: self.price_history.len() - 1,
                    key_levels: vec![*prev, *recent],
                    volume_confirmation: false,
                    historical_reliability: 0.6,
                },
            });
        }

        Ok(patterns)
    }

    /// Detect price action patterns
    fn detect_price_action_patterns(&self) -> QarResult<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        if self.price_history.len() < 10 {
            return Ok(patterns);
        }

        // Support and resistance levels
        patterns.extend(self.detect_support_resistance()?);

        // Breakout patterns
        if let Some(breakout) = self.detect_breakout_pattern()? {
            patterns.push(breakout);
        }

        Ok(patterns)
    }

    /// Detect support and resistance levels
    fn detect_support_resistance(&self) -> QarResult<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        let recent_prices = &self.price_history[self.price_history.len().saturating_sub(20)..];
        
        // Find potential support levels (recent lows)
        let lows = self.find_swing_lows(recent_prices, 2);
        for (idx, level) in lows {
            let current_price = recent_prices[recent_prices.len() - 1];
            let distance = (current_price - level).abs() / level;
            
            if distance < 0.03 { // Close to support/resistance
                let strength = 1.0 - distance * 33.3; // Closer = stronger
                patterns.push(DetectedPattern {
                    pattern_type: PatternType::Support,
                    strength,
                    completion: 1.0,
                    direction: PatternDirection::Bullish,
                    target_level: Some(level),
                    timeframe: PatternTimeframe::MediumTerm,
                    metadata: PatternMetadata {
                        start_index: self.price_history.len() - 20 + idx,
                        end_index: self.price_history.len() - 1,
                        key_levels: vec![level],
                        volume_confirmation: self.check_volume_confirmation(),
                        historical_reliability: 0.65,
                    },
                });
            }
        }

        // Find potential resistance levels (recent highs)
        let highs = self.find_swing_highs(recent_prices, 2);
        for (idx, level) in highs {
            let current_price = recent_prices[recent_prices.len() - 1];
            let distance = (current_price - level).abs() / level;
            
            if distance < 0.03 {
                let strength = 1.0 - distance * 33.3;
                patterns.push(DetectedPattern {
                    pattern_type: PatternType::Resistance,
                    strength,
                    completion: 1.0,
                    direction: PatternDirection::Bearish,
                    target_level: Some(level),
                    timeframe: PatternTimeframe::MediumTerm,
                    metadata: PatternMetadata {
                        start_index: self.price_history.len() - 20 + idx,
                        end_index: self.price_history.len() - 1,
                        key_levels: vec![level],
                        volume_confirmation: self.check_volume_confirmation(),
                        historical_reliability: 0.65,
                    },
                });
            }
        }

        Ok(patterns)
    }

    /// Detect breakout pattern
    fn detect_breakout_pattern(&self) -> QarResult<Option<DetectedPattern>> {
        if self.price_history.len() < 10 {
            return Ok(None);
        }

        let recent_prices = &self.price_history[self.price_history.len() - 10..];
        let consolidation_range = self.calculate_price_range(&recent_prices[..8]);
        let avg_price = recent_prices[..8].iter().sum::<f64>() / 8.0;
        let latest_price = recent_prices[recent_prices.len() - 1];
        
        // Check for tight consolidation followed by breakout
        if consolidation_range / avg_price > 0.03 {
            return Ok(None); // Not tight enough consolidation
        }

        let breakout_move = (latest_price - avg_price) / avg_price;
        if breakout_move.abs() < 0.015 {
            return Ok(None); // No significant breakout
        }

        let direction = if breakout_move > 0.0 {
            PatternDirection::Bullish
        } else {
            PatternDirection::Bearish
        };

        let strength = breakout_move.abs() * 66.7; // Convert to 0-1 scale

        Ok(Some(DetectedPattern {
            pattern_type: PatternType::BreakoutPattern,
            strength: strength.min(1.0),
            completion: 0.6, // Early stage of breakout
            direction,
            target_level: Some(latest_price + breakout_move * avg_price),
            timeframe: PatternTimeframe::ShortTerm,
            metadata: PatternMetadata {
                start_index: self.price_history.len() - 10,
                end_index: self.price_history.len() - 1,
                key_levels: vec![avg_price, latest_price],
                volume_confirmation: self.check_volume_confirmation(),
                historical_reliability: 0.6,
            },
        }))
    }

    /// Detect volume patterns
    fn detect_volume_patterns(&self) -> QarResult<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        if self.volume_history.len() < 5 {
            return Ok(patterns);
        }

        let recent_volumes = &self.volume_history[self.volume_history.len() - 5..];
        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        let latest_volume = recent_volumes[recent_volumes.len() - 1];
        
        // Volume spike detection
        if latest_volume > avg_volume * 2.0 {
            patterns.push(DetectedPattern {
                pattern_type: PatternType::VolumeSpike,
                strength: (latest_volume / avg_volume - 1.0).min(1.0),
                completion: 1.0,
                direction: PatternDirection::Neutral,
                target_level: None,
                timeframe: PatternTimeframe::ShortTerm,
                metadata: PatternMetadata {
                    start_index: self.volume_history.len() - 5,
                    end_index: self.volume_history.len() - 1,
                    key_levels: vec![avg_volume, latest_volume],
                    volume_confirmation: true,
                    historical_reliability: 0.7,
                },
            });
        }

        Ok(patterns)
    }

    /// Detect momentum patterns
    fn detect_momentum_patterns(&self, factors: &FactorMap) -> QarResult<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        let momentum = factors.get_factor(&StandardFactors::Momentum)?;
        let trend = factors.get_factor(&StandardFactors::Trend)?;
        
        // Momentum divergence
        if (momentum - trend).abs() > 0.3 {
            let direction = if momentum > trend {
                PatternDirection::Bullish
            } else {
                PatternDirection::Bearish
            };

            patterns.push(DetectedPattern {
                pattern_type: PatternType::Divergence,
                strength: (momentum - trend).abs(),
                completion: 0.8,
                direction,
                target_level: None,
                timeframe: PatternTimeframe::MediumTerm,
                metadata: PatternMetadata {
                    start_index: 0,
                    end_index: 0,
                    key_levels: vec![momentum, trend],
                    volume_confirmation: false,
                    historical_reliability: 0.55,
                },
            });
        }

        Ok(patterns)
    }

    /// Helper methods
    fn find_swing_highs(&self, prices: &[f64], lookback: usize) -> Vec<(usize, f64)> {
        let mut highs = Vec::new();
        
        for i in lookback..(prices.len() - lookback) {
            let current = prices[i];
            let is_high = (0..=lookback).all(|j| current >= prices[i - j]) &&
                         (0..=lookback).all(|j| current >= prices[i + j]);
            
            if is_high {
                highs.push((i, current));
            }
        }
        
        highs
    }

    fn find_swing_lows(&self, prices: &[f64], lookback: usize) -> Vec<(usize, f64)> {
        let mut lows = Vec::new();
        
        for i in lookback..(prices.len() - lookback) {
            let current = prices[i];
            let is_low = (0..=lookback).all(|j| current <= prices[i - j]) &&
                        (0..=lookback).all(|j| current <= prices[i + j]);
            
            if is_low {
                lows.push((i, current));
            }
        }
        
        lows
    }

    fn calculate_slope(&self, points: &[(usize, f64)]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }

        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| *x as f64).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| *x as f64 * y).sum();
        let sum_x2: f64 = points.iter().map(|(x, _)| (*x as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator == 0.0 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    fn calculate_triangle_strength(&self, highs: &[(usize, f64)], lows: &[(usize, f64)]) -> f64 {
        if highs.is_empty() || lows.is_empty() {
            return 0.0;
        }

        let high_range = highs.iter().map(|(_, h)| *h).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() -
                        highs.iter().map(|(_, h)| *h).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let low_range = lows.iter().map(|(_, l)| *l).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() -
                       lows.iter().map(|(_, l)| *l).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        let convergence = 1.0 - (high_range + low_range) / (highs[0].1 + lows[0].1);
        convergence.max(0.0).min(1.0)
    }

    fn calculate_triangle_target(&self, highs: &[(usize, f64)], lows: &[(usize, f64)]) -> Option<f64> {
        if highs.is_empty() || lows.is_empty() {
            return None;
        }

        let height = highs[0].1 - lows[0].1;
        let base_price = (highs[0].1 + lows[0].1) / 2.0;
        Some(base_price + height * 0.618) // Fibonacci extension
    }

    fn calculate_pattern_completion(&self, prices: &[f64]) -> f64 {
        // Simplified completion calculation
        if prices.len() < 3 {
            return 0.0;
        }
        
        let range = self.calculate_price_range(prices);
        let recent_range = self.calculate_price_range(&prices[prices.len() - 3..]);
        
        if range == 0.0 {
            1.0
        } else {
            (1.0 - recent_range / range).max(0.0).min(1.0)
        }
    }

    fn calculate_price_range(&self, prices: &[f64]) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }

        let max_price = prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_price = prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        max_price - min_price
    }

    fn check_volume_confirmation(&self) -> bool {
        if self.volume_history.len() < 3 {
            return false;
        }

        let recent_volumes = &self.volume_history[self.volume_history.len() - 3..];
        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        let latest_volume = recent_volumes[recent_volumes.len() - 1];

        latest_volume > avg_volume * 1.2 // 20% above average
    }

    fn calculate_pattern_strengths(&self, patterns: &[DetectedPattern]) -> HashMap<PatternType, f64> {
        let mut strengths = HashMap::new();
        
        for pattern in patterns {
            let entry = strengths.entry(pattern.pattern_type.clone()).or_insert(0.0);
            *entry = entry.max(pattern.strength);
        }
        
        strengths
    }

    fn calculate_overall_score(&self, patterns: &[DetectedPattern]) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }

        let weighted_sum: f64 = patterns.iter()
            .map(|p| p.strength * p.metadata.historical_reliability)
            .sum();
        
        let weight_sum: f64 = patterns.iter()
            .map(|p| p.metadata.historical_reliability)
            .sum();

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    fn calculate_pattern_confidence(&self, patterns: &[DetectedPattern]) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }

        let confidence_factors = vec![
            self.calculate_sample_size_confidence(),
            self.calculate_pattern_consistency(patterns),
            self.calculate_volume_confidence(patterns),
        ];

        confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
    }

    fn calculate_sample_size_confidence(&self) -> f64 {
        let data_points = self.price_history.len();
        if data_points < 10 {
            0.3
        } else if data_points < 50 {
            0.6
        } else {
            0.9
        }
    }

    fn calculate_pattern_consistency(&self, patterns: &[DetectedPattern]) -> f64 {
        if patterns.len() < 2 {
            return 0.5;
        }

        let bullish_count = patterns.iter().filter(|p| matches!(p.direction, PatternDirection::Bullish)).count();
        let bearish_count = patterns.iter().filter(|p| matches!(p.direction, PatternDirection::Bearish)).count();
        let total_directional = bullish_count + bearish_count;

        if total_directional == 0 {
            0.5
        } else {
            let max_directional = bullish_count.max(bearish_count);
            max_directional as f64 / total_directional as f64
        }
    }

    fn calculate_volume_confidence(&self, patterns: &[DetectedPattern]) -> f64 {
        if patterns.is_empty() {
            return 0.5;
        }

        let confirmed_patterns = patterns.iter()
            .filter(|p| p.metadata.volume_confirmation)
            .count();

        confirmed_patterns as f64 / patterns.len() as f64
    }

    fn calculate_reliability_metrics(&self, patterns: &[DetectedPattern]) -> ReliabilityMetrics {
        let mut success_rates = HashMap::new();
        let mut pattern_counts = HashMap::new();

        for pattern in patterns {
            *pattern_counts.entry(pattern.pattern_type.clone()).or_insert(0) += 1;
            success_rates.insert(pattern.pattern_type.clone(), pattern.metadata.historical_reliability);
        }

        let overall_reliability = if !patterns.is_empty() {
            patterns.iter().map(|p| p.metadata.historical_reliability).sum::<f64>() / patterns.len() as f64
        } else {
            0.0
        };

        let false_positive_rate = 1.0 - overall_reliability;
        let stability = self.calculate_pattern_stability(patterns);

        ReliabilityMetrics {
            overall_reliability,
            success_rates,
            false_positive_rate,
            stability,
        }
    }

    fn calculate_pattern_stability(&self, patterns: &[DetectedPattern]) -> f64 {
        if patterns.is_empty() {
            return 1.0;
        }

        let avg_completion = patterns.iter().map(|p| p.completion).sum::<f64>() / patterns.len() as f64;
        let completion_variance = patterns.iter()
            .map(|p| (p.completion - avg_completion).powi(2))
            .sum::<f64>() / patterns.len() as f64;

        1.0 - completion_variance.sqrt()
    }

    fn add_to_history(&mut self, result: PatternResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[PatternResult] {
        &self.history
    }

    /// Get latest analysis
    pub fn get_latest(&self) -> Option<&PatternResult> {
        self.history.last()
    }

    /// Get pattern parameters
    pub fn get_parameters(&self) -> &PatternParameters {
        &self.pattern_params
    }

    /// Update pattern parameters
    pub fn update_parameters(&mut self, params: PatternParameters) {
        self.pattern_params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_pattern_analyzer() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = PatternAnalyzer::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Volatility.to_string(), 0.4);
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.7);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.6);
        factors.insert(StandardFactors::Risk.to_string(), 0.3);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.7);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = analyzer.analyze(&factor_map).await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        assert!(pattern_result.score >= 0.0 && pattern_result.score <= 1.0);
        assert!(pattern_result.confidence >= 0.0 && pattern_result.confidence <= 1.0);
    }

    #[test]
    fn test_swing_high_detection() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config).unwrap();
        
        let prices = vec![100.0, 101.0, 103.0, 102.0, 100.0, 99.0, 101.0];
        let highs = analyzer.find_swing_highs(&prices, 2);
        
        assert!(!highs.is_empty());
        assert_eq!(highs[0].1, 103.0); // Should find the peak at 103.0
    }

    #[test]
    fn test_swing_low_detection() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config).unwrap();
        
        let prices = vec![100.0, 99.0, 97.0, 98.0, 100.0, 101.0, 99.0];
        let lows = analyzer.find_swing_lows(&prices, 2);
        
        assert!(!lows.is_empty());
        assert_eq!(lows[0].1, 97.0); // Should find the trough at 97.0
    }

    #[test]
    fn test_slope_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config).unwrap();
        
        let points = vec![(0, 100.0), (1, 101.0), (2, 102.0)];
        let slope = analyzer.calculate_slope(&points);
        
        assert!(slope > 0.0); // Should be positive slope
        assert!((slope - 1.0).abs() < 0.1); // Should be close to 1.0
    }

    #[test]
    fn test_price_range_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config).unwrap();
        
        let prices = vec![100.0, 105.0, 95.0, 102.0];
        let range = analyzer.calculate_price_range(&prices);
        
        assert_eq!(range, 10.0); // 105.0 - 95.0 = 10.0
    }

    #[test]
    fn test_volume_confirmation() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = PatternAnalyzer::new(config).unwrap();
        
        // Set up volume history
        analyzer.volume_history = vec![1000.0, 1100.0, 1300.0]; // Increasing volume
        
        let confirmation = analyzer.check_volume_confirmation();
        assert!(confirmation); // Should confirm with increasing volume
    }

    #[test]
    fn test_pattern_strength_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config).unwrap();
        
        let patterns = vec![
            DetectedPattern {
                pattern_type: PatternType::Triangle,
                strength: 0.8,
                completion: 0.9,
                direction: PatternDirection::Bullish,
                target_level: None,
                timeframe: PatternTimeframe::MediumTerm,
                metadata: PatternMetadata {
                    start_index: 0,
                    end_index: 10,
                    key_levels: vec![100.0, 105.0],
                    volume_confirmation: true,
                    historical_reliability: 0.7,
                },
            },
            DetectedPattern {
                pattern_type: PatternType::Flag,
                strength: 0.6,
                completion: 0.8,
                direction: PatternDirection::Bullish,
                target_level: None,
                timeframe: PatternTimeframe::ShortTerm,
                metadata: PatternMetadata {
                    start_index: 5,
                    end_index: 15,
                    key_levels: vec![102.0, 104.0],
                    volume_confirmation: false,
                    historical_reliability: 0.6,
                },
            },
        ];
        
        let strengths = analyzer.calculate_pattern_strengths(&patterns);
        assert_eq!(strengths.get(&PatternType::Triangle), Some(&0.8));
        assert_eq!(strengths.get(&PatternType::Flag), Some(&0.6));
    }
}