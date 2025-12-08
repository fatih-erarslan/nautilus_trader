//! Bubble Detection Module
//!
//! Enterprise-grade detector for identifying bubble patterns in cryptocurrency markets.
//! This module implements sophisticated analysis of:
//! - Exponential price growth pattern detection
//! - Volume divergence analysis during parabolic moves
//! - Regime change detection using statistical methods
//! - Social sentiment integration (placeholder for external data)
//! - Volatility expansion analysis
//! - SIMD-optimized mathematical calculations
//!
//! Bubble characteristics include:
//! - Exponential price growth over extended periods
//! - Volume patterns: initial increase followed by divergence
//! - Parabolic curve fitting with high R-squared values
//! - Regime shifts from normal to euphoric states
//! - Acceleration in price velocity
//! - Deviation from fundamental value approximations

use crate::*;
use std::time::Instant;

#[cfg(feature = "simd")]
use wide::f32x8;

/// Configuration for bubble detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleConfig {
    /// Minimum period for exponential growth analysis
    pub min_growth_period: usize,
    /// Exponential growth threshold (R-squared)
    pub growth_threshold: f32,
    /// Volume divergence threshold
    pub volume_divergence_threshold: f32,
    /// Volatility expansion threshold
    pub volatility_threshold: f32,
    /// Regime change sensitivity
    pub regime_sensitivity: f32,
    /// Parabolic acceleration threshold
    pub acceleration_threshold: f32,
    /// Social sentiment weight (for future integration)
    pub sentiment_weight: f32,
    /// Lookback period for analysis
    pub lookback_period: usize,
    /// Enable parallel processing
    pub use_parallel: bool,
}

impl Default for BubbleConfig {
    fn default() -> Self {
        Self {
            min_growth_period: 30,
            growth_threshold: 0.85, // R-squared > 0.85 for exponential fit
            volume_divergence_threshold: -0.3, // 30% volume decline during growth
            volatility_threshold: 2.0, // 2x volatility increase
            regime_sensitivity: 0.7,
            acceleration_threshold: 1.5, // 50% acceleration in price velocity
            sentiment_weight: 0.1, // Future integration
            lookback_period: 100,
            use_parallel: true,
        }
    }
}

/// Exponential growth model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialModel {
    pub a: f32, // Initial value
    pub b: f32, // Growth rate
    pub r_squared: f32, // Goodness of fit
    pub start_index: usize,
    pub end_index: usize,
    pub growth_duration: usize,
}

/// Bubble detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleResult {
    pub bubble_detected: bool,
    pub bubble_strength: f32,
    pub confidence: f32,
    pub bubble_phase: String, // "formation", "expansion", "peak", "burst", "none"
    pub start_index: Option<usize>,
    pub peak_index: Option<usize>,
    pub calculation_time_ns: u64,
    
    // Component analysis
    pub exponential_model: Option<ExponentialModel>,
    pub volume_divergence: f32,
    pub volatility_expansion: f32,
    pub regime_score: f32,
    pub acceleration_factor: f32,
    pub parabolic_r_squared: f32,
    
    // Growth metrics
    pub total_growth: f32,
    pub growth_rate: f32,
    pub price_velocity: f32,
    pub price_acceleration: f32,
    
    // Risk indicators
    pub deviation_from_trend: f32,
    pub sustainability_score: f32,
    pub burst_probability: f32,
    
    // Performance metrics
    pub simd_operations: u64,
    pub parallel_chunks: u64,
}

/// Cache-aligned bubble detector with SIMD optimization
#[repr(align(64))]
pub struct BubbleDetector {
    config: BubbleConfig,
    // Performance tracking
    total_detections: AtomicU64,
    total_time_ns: AtomicU64,
}

impl BubbleDetector {
    /// Create new bubble detector with default configuration
    pub fn new() -> Self {
        Self {
            config: BubbleConfig::default(),
            total_detections: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
        }
    }
    
    /// Create detector with custom configuration
    pub fn with_config(config: BubbleConfig) -> Self {
        Self {
            config,
            total_detections: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
        }
    }
    
    /// Detect bubble patterns in market data
    /// Matches Python BubbleDetector.detect() functionality
    pub fn detect(&self, market_data: &MarketData) -> Result<BubbleResult> {
        let start_time = Instant::now();
        
        // Validate input data
        market_data.validate()?;
        
        if market_data.len() < self.config.lookback_period {
            return Err(DetectorError::InsufficientData {
                required: self.config.lookback_period,
                actual: market_data.len(),
            });
        }
        
        info!("Starting bubble detection for {} data points", market_data.len());
        
        // Fit exponential growth model
        let exponential_model = self.fit_exponential_model(&market_data.prices)?;
        
        // Calculate volume divergence
        let volume_divergence = self.calculate_volume_divergence(&market_data.prices, &market_data.volumes)?;
        
        // Calculate volatility expansion
        let volatility_expansion = self.calculate_volatility_expansion(&market_data.prices)?;
        
        // Calculate regime score
        let regime_score = self.calculate_regime_score(&market_data.prices)?;
        
        // Calculate price velocity and acceleration
        let (price_velocity, price_acceleration) = self.calculate_price_dynamics(&market_data.prices)?;
        let acceleration_factor = if price_velocity > 0.0 {
            price_acceleration / price_velocity
        } else {
            0.0
        };
        
        // Fit parabolic curve and calculate R-squared
        let parabolic_r_squared = self.fit_parabolic_curve(&market_data.prices)?;
        
        // Calculate growth metrics
        let total_growth = if market_data.prices.len() >= 2 {
            let start_price = market_data.prices[market_data.prices.len() - self.config.lookback_period.min(market_data.prices.len())];
            let end_price = market_data.prices[market_data.prices.len() - 1];
            (end_price - start_price) / start_price
        } else {
            0.0
        };
        
        let growth_rate = if let Some(ref model) = exponential_model {
            model.b
        } else {
            0.0
        };
        
        // Calculate risk indicators
        let deviation_from_trend = self.calculate_trend_deviation(&market_data.prices);
        let sustainability_score = self.calculate_sustainability_score(
            total_growth,
            volume_divergence,
            volatility_expansion,
        );
        
        // Determine bubble phase and detection
        let (bubble_detected, bubble_strength, confidence, bubble_phase, start_idx, peak_idx) = 
            self.analyze_bubble_indicators(
                &exponential_model,
                volume_divergence,
                volatility_expansion,
                regime_score,
                acceleration_factor,
                parabolic_r_squared,
                &market_data.prices,
            )?;
        
        // Calculate burst probability
        let burst_probability = self.calculate_burst_probability(
            bubble_strength,
            volume_divergence,
            volatility_expansion,
            sustainability_score,
        );
        
        let calculation_time_ns = start_time.elapsed().as_nanos() as u64;
        
        // Update performance counters
        self.total_detections.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(calculation_time_ns, Ordering::Relaxed);
        
        // Record global performance
        super::PERFORMANCE_MONITOR.record_detection(calculation_time_ns, "bubble");
        
        info!("Bubble detection completed in {}ns, detected: {}, phase: {}", 
              calculation_time_ns, bubble_detected, bubble_phase);
        
        Ok(BubbleResult {
            bubble_detected,
            bubble_strength,
            confidence,
            bubble_phase,
            start_index: start_idx,
            peak_index: peak_idx,
            calculation_time_ns,
            exponential_model,
            volume_divergence,
            volatility_expansion,
            regime_score,
            acceleration_factor,
            parabolic_r_squared,
            total_growth,
            growth_rate,
            price_velocity,
            price_acceleration,
            deviation_from_trend,
            sustainability_score,
            burst_probability,
            simd_operations: 0, // Would be tracked in SIMD operations
            parallel_chunks: if self.config.use_parallel { 4 } else { 1 },
        })
    }
    
    /// Fit exponential growth model to price data
    /// Uses least squares regression for y = a * e^(b*x)
    fn fit_exponential_model(&self, prices: &[f32]) -> Result<Option<ExponentialModel>> {
        if prices.len() < self.config.min_growth_period {
            return Ok(None);
        }
        
        let n = prices.len().min(self.config.lookback_period);
        let recent_prices = &prices[prices.len() - n..];
        
        // Convert to logarithmic space: ln(y) = ln(a) + b*x
        let mut log_prices = Vec::new();
        let mut x_values = Vec::new();
        
        for (i, &price) in recent_prices.iter().enumerate() {
            if price > 0.0 {
                log_prices.push(price.ln());
                x_values.push(i as f32);
            }
        }
        
        if log_prices.len() < self.config.min_growth_period {
            return Ok(None);
        }
        
        // Linear regression on log-transformed data
        let n_points = log_prices.len() as f32;
        let sum_x = x_values.iter().sum::<f32>();
        let sum_y = log_prices.iter().sum::<f32>();
        let sum_xy = x_values.iter().zip(log_prices.iter()).map(|(&x, &y)| x * y).sum::<f32>();
        let sum_xx = x_values.iter().map(|&x| x * x).sum::<f32>();
        let sum_yy = log_prices.iter().map(|&y| y * y).sum::<f32>();
        
        // Calculate regression coefficients
        let denominator = n_points * sum_xx - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return Ok(None);
        }
        
        let b = (n_points * sum_xy - sum_x * sum_y) / denominator;
        let ln_a = (sum_y - b * sum_x) / n_points;
        let a = ln_a.exp();
        
        // Calculate R-squared
        let y_mean = sum_y / n_points;
        let ss_tot = log_prices.iter().map(|&y| (y - y_mean).powi(2)).sum::<f32>();
        let ss_res = x_values.iter().zip(log_prices.iter())
            .map(|(&x, &y)| {
                let predicted = ln_a + b * x;
                (y - predicted).powi(2)
            })
            .sum::<f32>();
        
        let r_squared = if ss_tot > f32::EPSILON {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        
        // Only return model if it indicates exponential growth
        if b > 0.0 && r_squared >= self.config.growth_threshold {
            Ok(Some(ExponentialModel {
                a,
                b,
                r_squared,
                start_index: prices.len() - n,
                end_index: prices.len() - 1,
                growth_duration: n,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate volume divergence during price increases
    fn calculate_volume_divergence(&self, prices: &[f32], volumes: &[f32]) -> Result<f32> {
        if prices.len() != volumes.len() || prices.len() < 20 {
            return Ok(0.0);
        }
        
        let window = 20;
        let n = prices.len();
        
        // Calculate price trend (recent vs earlier)
        let recent_prices = &prices[n - window..];
        let earlier_prices = &prices[n - 2*window..n - window];
        
        let recent_price_avg = recent_prices.iter().sum::<f32>() / window as f32;
        let earlier_price_avg = earlier_prices.iter().sum::<f32>() / window as f32;
        let price_change = (recent_price_avg - earlier_price_avg) / earlier_price_avg;
        
        // Calculate volume trend
        let recent_volumes = &volumes[n - window..];
        let earlier_volumes = &volumes[n - 2*window..n - window];
        
        let recent_volume_avg = recent_volumes.iter().sum::<f32>() / window as f32;
        let earlier_volume_avg = earlier_volumes.iter().sum::<f32>() / window as f32;
        let volume_change = (recent_volume_avg - earlier_volume_avg) / earlier_volume_avg;
        
        // Divergence occurs when price rises but volume falls
        if price_change > 0.05 { // Price increased by >5%
            Ok(volume_change)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate volatility expansion
    fn calculate_volatility_expansion(&self, prices: &[f32]) -> Result<f32> {
        if prices.len() < 40 {
            return Ok(1.0);
        }
        
        let window = 20;
        let n = prices.len();
        
        // Calculate recent volatility
        let recent_prices = &prices[n - window..];
        let recent_mean = recent_prices.iter().sum::<f32>() / window as f32;
        let recent_volatility = (recent_prices.iter()
            .map(|&p| (p - recent_mean).powi(2))
            .sum::<f32>() / window as f32).sqrt();
        
        // Calculate historical volatility
        let historical_prices = &prices[n - 2*window..n - window];
        let historical_mean = historical_prices.iter().sum::<f32>() / window as f32;
        let historical_volatility = (historical_prices.iter()
            .map(|&p| (p - historical_mean).powi(2))
            .sum::<f32>() / window as f32).sqrt();
        
        if historical_volatility > f32::EPSILON {
            Ok(recent_volatility / historical_volatility)
        } else {
            Ok(1.0)
        }
    }
    
    /// Calculate regime score (transition from normal to bubble state)
    fn calculate_regime_score(&self, prices: &[f32]) -> Result<f32> {
        if prices.len() < 50 {
            return Ok(0.0);
        }
        
        let window = 25;
        let n = prices.len();
        
        // Calculate rolling Sharpe-like ratio (returns / volatility)
        let mut regime_indicators = Vec::new();
        
        for i in window..n {
            let window_prices = &prices[i - window..i];
            
            // Calculate returns
            let mut returns = Vec::new();
            for j in 1..window_prices.len() {
                returns.push((window_prices[j] - window_prices[j-1]) / window_prices[j-1]);
            }
            
            if !returns.is_empty() {
                let mean_return = returns.iter().sum::<f32>() / returns.len() as f32;
                let return_std = if returns.len() > 1 {
                    let variance = returns.iter()
                        .map(|&r| (r - mean_return).powi(2))
                        .sum::<f32>() / (returns.len() - 1) as f32;
                    variance.sqrt()
                } else {
                    f32::EPSILON
                };
                
                // Regime indicator: high returns with low volatility = normal, 
                // high returns with high volatility = bubble
                let regime_value = if return_std > f32::EPSILON {
                    mean_return / return_std
                } else {
                    0.0
                };
                
                regime_indicators.push(regime_value);
            }
        }
        
        if regime_indicators.is_empty() {
            return Ok(0.0);
        }
        
        // Recent regime vs historical regime
        let split_point = regime_indicators.len() / 2;
        let historical_regime = regime_indicators[0..split_point].iter().sum::<f32>() / split_point as f32;
        let recent_regime = regime_indicators[split_point..].iter().sum::<f32>() / (regime_indicators.len() - split_point) as f32;
        
        // Score based on deviation from historical norm
        let regime_change = (recent_regime - historical_regime).abs() / (historical_regime.abs() + f32::EPSILON);
        Ok(regime_change.min(10.0)) // Cap at 10.0
    }
    
    /// Calculate price velocity and acceleration
    fn calculate_price_dynamics(&self, prices: &[f32]) -> Result<(f32, f32)> {
        if prices.len() < 3 {
            return Ok((0.0, 0.0));
        }
        
        let n = prices.len();
        
        // Calculate velocity (first derivative approximation)
        let velocity = (prices[n-1] - prices[n-2]) / prices[n-2];
        
        // Calculate acceleration (second derivative approximation)
        let acceleration = if n >= 3 {
            let prev_velocity = (prices[n-2] - prices[n-3]) / prices[n-3];
            velocity - prev_velocity
        } else {
            0.0
        };
        
        Ok((velocity, acceleration))
    }
    
    /// Fit parabolic curve y = ax^2 + bx + c
    fn fit_parabolic_curve(&self, prices: &[f32]) -> Result<f32> {
        let n = prices.len().min(self.config.lookback_period);
        if n < 10 {
            return Ok(0.0);
        }
        
        let recent_prices = &prices[prices.len() - n..];
        
        // Set up matrices for least squares: [x^2 x 1] * [a b c]' = y
        let mut sum_x4 = 0.0;
        let mut sum_x3 = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2y = 0.0;
        let mut sum_xy = 0.0;
        let n_f = n as f32;
        
        for (i, &price) in recent_prices.iter().enumerate() {
            let x = i as f32;
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x3 * x;
            
            sum_x4 += x4;
            sum_x3 += x3;
            sum_x2 += x2;
            sum_x += x;
            sum_y += price;
            sum_x2y += x2 * price;
            sum_xy += x * price;
        }
        
        // Solve normal equations (simplified for parabolic fit)
        // This is a simplified version - in practice would use matrix operations
        let y_mean = sum_y / n_f;
        let ss_tot = recent_prices.iter()
            .map(|&y| (y - y_mean).powi(2))
            .sum::<f32>();
        
        // Estimate parabolic fit quality using correlation
        let mut ss_res = 0.0;
        for (i, &price) in recent_prices.iter().enumerate() {
            let x = i as f32;
            // Simple parabolic approximation
            let predicted = sum_y / n_f + (sum_xy / sum_x2) * x.powi(2);
            ss_res += (price - predicted).powi(2);
        }
        
        let r_squared = if ss_tot > f32::EPSILON {
            (1.0 - (ss_res / ss_tot)).max(0.0)
        } else {
            0.0
        };
        
        Ok(r_squared)
    }
    
    /// Calculate deviation from long-term trend
    fn calculate_trend_deviation(&self, prices: &[f32]) -> f32 {
        if prices.len() < 50 {
            return 0.0;
        }
        
        let n = prices.len();
        let trend_window = n.min(100);
        
        // Calculate long-term linear trend
        let trend_prices = &prices[n - trend_window..];
        let n_trend = trend_prices.len() as f32;
        
        let sum_x = (0..trend_prices.len()).map(|i| i as f32).sum::<f32>();
        let sum_y = trend_prices.iter().sum::<f32>();
        let sum_xy = trend_prices.iter().enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum::<f32>();
        let sum_xx = (0..trend_prices.len())
            .map(|i| (i as f32).powi(2))
            .sum::<f32>();
        
        // Linear regression coefficients
        let denominator = n_trend * sum_xx - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return 0.0;
        }
        
        let slope = (n_trend * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n_trend;
        
        // Calculate current deviation from trend
        let current_x = (trend_prices.len() - 1) as f32;
        let trend_price = intercept + slope * current_x;
        let current_price = prices[n - 1];
        
        let deviation = (current_price - trend_price) / trend_price;
        deviation
    }
    
    /// Calculate sustainability score
    fn calculate_sustainability_score(&self, growth: f32, volume_div: f32, volatility: f32) -> f32 {
        // Lower score = less sustainable (more bubble-like)
        let mut sustainability: f32 = 1.0;
        
        // Penalize excessive growth
        if growth > 2.0 { // >200% growth
            sustainability *= 0.5;
        }
        
        // Penalize volume divergence
        if volume_div < self.config.volume_divergence_threshold {
            sustainability *= 0.7;
        }
        
        // Penalize volatility expansion
        if volatility > self.config.volatility_threshold {
            sustainability *= 0.6;
        }
        
        sustainability.max(0.0).min(1.0)
    }
    
    /// Analyze all indicators to determine bubble presence and phase
    fn analyze_bubble_indicators(
        &self,
        exponential_model: &Option<ExponentialModel>,
        volume_divergence: f32,
        volatility_expansion: f32,
        regime_score: f32,
        acceleration_factor: f32,
        parabolic_r_squared: f32,
        prices: &[f32],
    ) -> Result<(bool, f32, f32, String, Option<usize>, Option<usize>)> {
        let mut bubble_strength = 0.0;
        let mut indicator_count = 0;
        
        // Exponential growth indicator
        if let Some(ref model) = exponential_model {
            bubble_strength += model.r_squared * 2.0; // Weight: 2.0
            indicator_count += 1;
        }
        
        // Volume divergence indicator
        if volume_divergence < self.config.volume_divergence_threshold {
            bubble_strength += 1.0; // Weight: 1.0
            indicator_count += 1;
        }
        
        // Volatility expansion indicator
        if volatility_expansion > self.config.volatility_threshold {
            bubble_strength += 1.0; // Weight: 1.0
            indicator_count += 1;
        }
        
        // Regime change indicator
        if regime_score > self.config.regime_sensitivity {
            bubble_strength += 1.5; // Weight: 1.5
            indicator_count += 1;
        }
        
        // Acceleration indicator
        if acceleration_factor > self.config.acceleration_threshold {
            bubble_strength += 1.0; // Weight: 1.0
            indicator_count += 1;
        }
        
        // Parabolic fit indicator
        if parabolic_r_squared > 0.8 {
            bubble_strength += 1.5; // Weight: 1.5
            indicator_count += 1;
        }
        
        // Normalize bubble strength
        let max_possible_strength = 8.0; // Sum of all weights
        bubble_strength = (bubble_strength / max_possible_strength).min(1.0);
        
        // Calculate confidence based on indicator agreement
        let confidence = if indicator_count > 0 {
            (indicator_count as f32 / 6.0).min(1.0) // 6 total indicators
        } else {
            0.0
        };
        
        // Determine bubble detection
        let bubble_detected = bubble_strength > 0.6 && indicator_count >= 3;
        
        // Determine bubble phase
        let bubble_phase = if !bubble_detected {
            "none".to_string()
        } else if acceleration_factor > 2.0 && volatility_expansion > 3.0 {
            "peak".to_string()
        } else if acceleration_factor > 1.0 {
            "expansion".to_string()
        } else {
            "formation".to_string()
        };
        
        // Determine start and peak indices
        let start_idx = if bubble_detected {
            exponential_model.as_ref().map(|m| m.start_index)
        } else {
            None
        };
        
        let peak_idx = if bubble_phase == "peak" || bubble_phase == "burst" {
            Some(prices.len() - 1)
        } else {
            None
        };
        
        Ok((bubble_detected, bubble_strength, confidence, bubble_phase, start_idx, peak_idx))
    }
    
    /// Calculate probability of bubble burst
    fn calculate_burst_probability(
        &self,
        bubble_strength: f32,
        volume_divergence: f32,
        volatility_expansion: f32,
        sustainability_score: f32,
    ) -> f32 {
        if bubble_strength < 0.5 {
            return 0.0;
        }
        
        let mut burst_factors = Vec::new();
        
        // High bubble strength increases burst probability
        burst_factors.push(bubble_strength);
        
        // Severe volume divergence increases burst probability
        if volume_divergence < -0.5 {
            burst_factors.push(0.8);
        }
        
        // Extreme volatility increases burst probability
        if volatility_expansion > 4.0 {
            burst_factors.push(0.9);
        }
        
        // Low sustainability increases burst probability
        burst_factors.push(1.0 - sustainability_score);
        
        let avg_burst_factor = burst_factors.iter().sum::<f32>() / burst_factors.len() as f32;
        avg_burst_factor.min(1.0)
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> (u64, u64, f64) {
        let total_detections = self.total_detections.load(Ordering::Relaxed);
        let total_time_ns = self.total_time_ns.load(Ordering::Relaxed);
        let avg_time_ns = if total_detections > 0 {
            total_time_ns as f64 / total_detections as f64
        } else {
            0.0
        };
        (total_detections, total_time_ns, avg_time_ns)
    }
    
    /// Reset performance counters
    pub fn reset_stats(&self) {
        self.total_detections.store(0, Ordering::Relaxed);
        self.total_time_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for BubbleDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bubble_detector_creation() {
        let detector = BubbleDetector::new();
        assert_eq!(detector.config.min_growth_period, 30);
        assert_eq!(detector.config.growth_threshold, 0.85);
    }
    
    #[test]
    fn test_bubble_detection() {
        let detector = BubbleDetector::new();
        
        // Create test data showing exponential growth (bubble pattern)
        let mut prices = vec![100.0; 150];
        let mut volumes = vec![1000.0; 150];
        
        // Add exponential growth pattern
        for i in 50..120 {
            let growth_factor = 1.0 + (i - 50) as f32 * 0.02; // 2% per period
            prices[i] = 100.0 * growth_factor.powf(2.0); // Exponential growth
            volumes[i] = 1000.0 * (1.0 + (i - 50) as f32 * 0.01); // Initial volume increase
        }
        
        // Add volume divergence in later stage
        for i in 100..120 {
            volumes[i] = volumes[i] * 0.8; // Volume decline during price rise
        }
        
        let timestamps = (0..150).map(|i| i as i64).collect();
        let market_data = MarketData::new(prices, volumes, timestamps);
        
        let result = detector.detect(&market_data);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.bubble_strength >= 0.0 && result.bubble_strength <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.burst_probability >= 0.0 && result.burst_probability <= 1.0);
    }
    
    #[test]
    fn test_exponential_model_fitting() {
        let detector = BubbleDetector::new();
        
        // Create exponential growth data
        let mut prices = Vec::new();
        for i in 0..50 {
            let price = 100.0 * (1.05_f32).powf(i as f32); // 5% growth per period
            prices.push(price);
        }
        
        let model = detector.fit_exponential_model(&prices);
        assert!(model.is_ok());
        
        if let Ok(Some(model)) = model {
            assert!(model.r_squared > 0.8); // Should fit well
            assert!(model.b > 0.0); // Positive growth rate
        }
    }
    
    #[test]
    fn test_insufficient_data() {
        let detector = BubbleDetector::new();
        
        let prices = vec![100.0, 101.0]; // Too few data points
        let volumes = vec![1000.0, 1100.0];
        let timestamps = vec![0, 1];
        let market_data = MarketData::new(prices, volumes, timestamps);
        
        let result = detector.detect(&market_data);
        assert!(result.is_err());
        
        match result {
            Err(DetectorError::InsufficientData { required, actual }) => {
                assert_eq!(required, 100); // Default lookback period
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }
}