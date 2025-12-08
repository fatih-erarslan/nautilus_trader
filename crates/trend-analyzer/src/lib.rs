use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use polars::prelude::*;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use ta::{indicators::*, DataItem};
use thiserror::Error;

pub mod indicators;
pub mod market_regime;
pub mod multi_timeframe;
pub mod pattern_recognition;

#[derive(Error, Debug)]
pub enum TrendError {
    #[error("Data fetch error: {0}")]
    DataFetchError(#[from] reqwest::Error),
    
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    
    #[error("Invalid timeframe: {0}")]
    InvalidTimeframe(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendMetrics {
    pub timeframe: String,
    pub trend_strength: f64,
    pub momentum_score: f64,
    pub volume_confirmation: f64,
    pub breakout_probability: f64,
    pub support_levels: Vec<f64>,
    pub resistance_levels: Vec<f64>,
    pub volatility: f64,
    pub trend_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendScore {
    pub symbol: String,
    pub overall_score: f64,
    pub trend_direction: TrendDirection,
    pub confidence: f64,
    pub timeframe_scores: Vec<TrendMetrics>,
    pub market_regime: MarketRegime,
    pub key_levels: KeyLevels,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    StrongBullish,
    Bullish,
    Neutral,
    Bearish,
    StrongBearish,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending(TrendingRegime),
    Ranging(RangingRegime),
    Volatile(VolatileRegime),
    Breakout(BreakoutRegime),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingRegime {
    pub direction: TrendDirection,
    pub strength: f64,
    pub duration_candles: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangingRegime {
    pub range_high: f64,
    pub range_low: f64,
    pub oscillation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatileRegime {
    pub volatility_percentile: f64,
    pub average_true_range: f64,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakoutRegime {
    pub breakout_level: f64,
    pub volume_surge: f64,
    pub confirmation_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Extreme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyLevels {
    pub pivot_point: f64,
    pub support_levels: Vec<SupportResistance>,
    pub resistance_levels: Vec<SupportResistance>,
    pub fibonacci_levels: Vec<FibonacciLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportResistance {
    pub price: f64,
    pub strength: f64,
    pub touches: u32,
    pub last_test: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciLevel {
    pub level: f64,
    pub price: f64,
    pub retracement_pct: f64,
}

#[derive(Clone)]
pub struct TrendAnalyzer {
    pub timeframes: Vec<String>,
    pub indicator_set: IndicatorSet,
    pub regime_detector: MarketRegimeDetector,
    cache: Arc<DashMap<String, (DateTime<Utc>, DataFrame)>>,
    client: reqwest::Client,
}

#[derive(Clone)]
pub struct IndicatorSet {
    pub ema_periods: Vec<usize>,
    pub sma_periods: Vec<usize>,
    pub rsi_period: usize,
    pub macd_config: MacdConfig,
    pub bb_config: BollingerConfig,
    pub atr_period: usize,
}

#[derive(Clone)]
pub struct MacdConfig {
    pub fast: usize,
    pub slow: usize,
    pub signal: usize,
}

#[derive(Clone)]
pub struct BollingerConfig {
    pub period: usize,
    pub std_dev: f64,
}

#[derive(Clone)]
pub struct MarketRegimeDetector {
    pub volatility_window: usize,
    pub trend_window: usize,
    pub breakout_threshold: f64,
}

impl Default for IndicatorSet {
    fn default() -> Self {
        Self {
            ema_periods: vec![9, 21, 55, 200],
            sma_periods: vec![20, 50, 100, 200],
            rsi_period: 14,
            macd_config: MacdConfig {
                fast: 12,
                slow: 26,
                signal: 9,
            },
            bb_config: BollingerConfig {
                period: 20,
                std_dev: 2.0,
            },
            atr_period: 14,
        }
    }
}

impl TrendAnalyzer {
    pub fn new(timeframes: Vec<String>) -> Self {
        Self {
            timeframes,
            indicator_set: IndicatorSet::default(),
            regime_detector: MarketRegimeDetector {
                volatility_window: 20,
                trend_window: 50,
                breakout_threshold: 2.0,
            },
            cache: Arc::new(DashMap::new()),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
        }
    }

    pub async fn analyze_pair_trends(&self, symbol: &str) -> Result<TrendScore, TrendError> {
        let mut timeframe_scores = Vec::new();
        
        for timeframe in &self.timeframes {
            let data = self.fetch_ohlcv(symbol, timeframe).await?;
            let metrics = self.analyze_timeframe(&data, timeframe)?;
            timeframe_scores.push(metrics);
        }
        
        let overall_score = self.aggregate_scores(&timeframe_scores);
        let market_regime = self.detect_market_regime(&timeframe_scores)?;
        let key_levels = self.calculate_key_levels(symbol).await?;
        
        Ok(TrendScore {
            symbol: symbol.to_string(),
            overall_score: overall_score.0,
            trend_direction: overall_score.1,
            confidence: overall_score.2,
            timeframe_scores,
            market_regime,
            key_levels,
        })
    }

    async fn fetch_ohlcv(&self, symbol: &str, timeframe: &str) -> Result<DataFrame, TrendError> {
        let cache_key = format!("{}-{}", symbol, timeframe);
        
        // Check cache first
        if let Some(cached) = self.cache.get(&cache_key) {
            let (timestamp, ref data) = *cached;
            if (Utc::now() - timestamp).num_minutes() < 5 {
                return Ok(data.clone());
            }
        }
        
        // Fetch from exchange (placeholder - implement actual exchange API)
        let url = format!("https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit=500", 
                         symbol.replace("/", ""), timeframe);
        
        let response: Vec<Vec<serde_json::Value>> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;
        
        // Convert to DataFrame
        let mut opens = Vec::new();
        let mut highs = Vec::new();
        let mut lows = Vec::new();
        let mut closes = Vec::new();
        let mut volumes = Vec::new();
        let mut timestamps = Vec::new();
        
        for kline in response {
            if kline.len() >= 6 {
                timestamps.push(kline[0].as_i64().unwrap_or(0));
                opens.push(kline[1].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0));
                highs.push(kline[2].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0));
                lows.push(kline[3].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0));
                closes.push(kline[4].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0));
                volumes.push(kline[5].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0));
            }
        }
        
        let df = DataFrame::new(vec![
            Series::new("timestamp", timestamps),
            Series::new("open", opens),
            Series::new("high", highs),
            Series::new("low", lows),
            Series::new("close", closes),
            Series::new("volume", volumes),
        ])?;
        
        // Cache the data
        self.cache.insert(cache_key, (Utc::now(), df.clone()));
        
        Ok(df)
    }

    fn analyze_timeframe(&self, data: &DataFrame, timeframe: &str) -> Result<TrendMetrics, TrendError> {
        let closes = data.column("close")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let volumes = data.column("volume")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        // Calculate trend strength using EMA confluence
        let trend_strength = self.calculate_trend_strength(&closes)?;
        
        // Calculate momentum
        let momentum_score = self.calculate_momentum(&closes)?;
        
        // Volume analysis
        let volume_confirmation = self.analyze_volume_trend(&closes, &volumes)?;
        
        // Breakout detection
        let breakout_probability = self.detect_breakout_probability(&closes, &volumes)?;
        
        // Support/Resistance levels
        let (support_levels, resistance_levels) = self.find_key_levels(&closes)?;
        
        // Volatility
        let volatility = self.calculate_volatility(&closes)?;
        
        // Trend quality
        let trend_quality = self.assess_trend_quality(&closes)?;
        
        Ok(TrendMetrics {
            timeframe: timeframe.to_string(),
            trend_strength,
            momentum_score,
            volume_confirmation,
            breakout_probability,
            support_levels,
            resistance_levels,
            volatility,
            trend_quality,
        })
    }

    fn calculate_trend_strength(&self, closes: &[f64]) -> Result<f64, TrendError> {
        if closes.len() < 200 {
            return Err(TrendError::InsufficientData("Need at least 200 candles".to_string()));
        }
        
        // Calculate multiple EMAs
        let ema9 = calculate_ema(closes, 9);
        let ema21 = calculate_ema(closes, 21);
        let ema55 = calculate_ema(closes, 55);
        let ema200 = calculate_ema(closes, 200);
        
        let last_idx = closes.len() - 1;
        let current_price = closes[last_idx];
        
        // Score based on EMA alignment
        let mut score = 0.0;
        
        // Check if EMAs are properly aligned for trend
        if ema9[last_idx] > ema21[last_idx] && 
           ema21[last_idx] > ema55[last_idx] && 
           ema55[last_idx] > ema200[last_idx] {
            score += 0.4; // Bullish alignment
        } else if ema9[last_idx] < ema21[last_idx] && 
                  ema21[last_idx] < ema55[last_idx] && 
                  ema55[last_idx] < ema200[last_idx] {
            score -= 0.4; // Bearish alignment
        }
        
        // Distance from EMAs
        let ema_distance = (current_price - ema200[last_idx]).abs() / ema200[last_idx];
        score += (ema_distance * 0.3).min(0.3);
        
        // Slope of trend
        let trend_slope = (ema55[last_idx] - ema55[last_idx.saturating_sub(20)]) / ema55[last_idx];
        score += trend_slope.abs().min(0.3);
        
        Ok(score)
    }

    fn calculate_momentum(&self, closes: &[f64]) -> Result<f64, TrendError> {
        // RSI momentum
        let rsi = calculate_rsi(closes, self.indicator_set.rsi_period);
        let last_rsi = rsi.last().copied().unwrap_or(50.0);
        
        // Rate of change
        let roc_period = 10;
        let roc = if closes.len() > roc_period {
            let current = closes.last().unwrap();
            let past = closes[closes.len() - roc_period - 1];
            (current - past) / past * 100.0
        } else {
            0.0
        };
        
        // MACD momentum
        let (macd_line, signal_line, _) = calculate_macd(
            closes,
            self.indicator_set.macd_config.fast,
            self.indicator_set.macd_config.slow,
            self.indicator_set.macd_config.signal,
        );
        
        let macd_momentum = if !macd_line.is_empty() && !signal_line.is_empty() {
            let last_macd = macd_line.last().unwrap();
            let last_signal = signal_line.last().unwrap();
            (last_macd - last_signal) / closes.last().unwrap() * 100.0
        } else {
            0.0
        };
        
        // Combine momentum indicators
        let rsi_score = (last_rsi - 50.0) / 50.0; // Normalize to [-1, 1]
        let roc_score = roc.max(-20.0).min(20.0) / 20.0; // Cap and normalize
        let macd_score = macd_momentum.max(-2.0).min(2.0) / 2.0;
        
        Ok((rsi_score * 0.3 + roc_score * 0.3 + macd_score * 0.4).max(-1.0).min(1.0))
    }

    fn analyze_volume_trend(&self, closes: &[f64], volumes: &[f64]) -> Result<f64, TrendError> {
        if closes.len() != volumes.len() || closes.len() < 20 {
            return Err(TrendError::InsufficientData("Volume data mismatch".to_string()));
        }
        
        // Volume moving average
        let vol_ma = calculate_sma(volumes, 20);
        let current_vol = volumes.last().unwrap();
        let avg_vol = vol_ma.last().unwrap();
        
        // Volume ratio
        let vol_ratio = current_vol / avg_vol;
        
        // Price-volume correlation
        let price_changes: Vec<f64> = closes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let vol_changes: Vec<f64> = volumes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(1.0))
            .collect();
        
        let correlation = calculate_correlation(&price_changes, &vol_changes);
        
        // Score based on volume confirmation
        let mut score = 0.0;
        
        // Higher volume on price increase is bullish
        if vol_ratio > 1.2 && price_changes.last().copied().unwrap_or(0.0) > 0.0 {
            score += 0.5;
        }
        
        // Positive correlation between price and volume
        if correlation > 0.3 {
            score += 0.3;
        }
        
        // Volume trend
        let vol_trend = if vol_ma.len() > 10 {
            let recent_avg = vol_ma[vol_ma.len()-5..].iter().sum::<f64>() / 5.0;
            let older_avg = vol_ma[vol_ma.len()-10..vol_ma.len()-5].iter().sum::<f64>() / 5.0;
            (recent_avg - older_avg) / older_avg
        } else {
            0.0
        };
        
        score += vol_trend.max(-0.2).min(0.2);
        
        Ok(score.max(0.0).min(1.0))
    }

    fn detect_breakout_probability(&self, closes: &[f64], volumes: &[f64]) -> Result<f64, TrendError> {
        if closes.len() < 50 {
            return Ok(0.0);
        }
        
        // Bollinger Bands
        let (upper, middle, lower) = calculate_bollinger_bands(
            closes,
            self.indicator_set.bb_config.period,
            self.indicator_set.bb_config.std_dev,
        );
        
        let current_price = closes.last().unwrap();
        let current_upper = upper.last().unwrap();
        let current_lower = lower.last().unwrap();
        let current_middle = middle.last().unwrap();
        
        // ATR for volatility
        let atr = calculate_atr(closes, closes, closes, self.indicator_set.atr_period);
        let current_atr = atr.last().copied().unwrap_or(0.0);
        
        let mut probability: f64 = 0.0;
        
        // Price near bands
        let band_width = current_upper - current_lower;
        let price_position = (current_price - current_lower) / band_width;
        
        if price_position > 0.9 || price_position < 0.1 {
            probability += 0.3;
        }
        
        // Volatility squeeze (bands contracting)
        let recent_bandwidth = &upper[upper.len().saturating_sub(20)..]
            .iter()
            .zip(&lower[lower.len().saturating_sub(20)..])
            .map(|(u, l)| u - l)
            .collect::<Vec<_>>();
        
        if !recent_bandwidth.is_empty() {
            let avg_bandwidth = recent_bandwidth.iter().sum::<f64>() / recent_bandwidth.len() as f64;
            if band_width < avg_bandwidth * 0.8 {
                probability += 0.3; // Squeeze detected
            }
        }
        
        // Volume surge
        let vol_ma = calculate_sma(volumes, 20);
        if let (Some(&current_vol), Some(&avg_vol)) = (volumes.last(), vol_ma.last()) {
            if current_vol > avg_vol * 1.5 {
                probability += 0.2;
            }
        }
        
        // Momentum building
        let rsi = calculate_rsi(closes, 14);
        if let Some(&current_rsi) = rsi.last() {
            if current_rsi > 70.0 || current_rsi < 30.0 {
                probability += 0.2;
            }
        }
        
        Ok(probability.min(1.0))
    }

    fn find_key_levels(&self, closes: &[f64]) -> Result<(Vec<f64>, Vec<f64>), TrendError> {
        if closes.len() < 20 {
            return Ok((vec![], vec![]));
        }
        
        let mut support_levels = Vec::new();
        let mut resistance_levels = Vec::new();
        
        // Find local minima and maxima
        for i in 10..closes.len()-10 {
            let window_before = &closes[i-10..i];
            let window_after = &closes[i+1..i+11];
            
            let current = closes[i];
            
            // Local minimum (support)
            if window_before.iter().all(|&x| x >= current) && 
               window_after.iter().all(|&x| x >= current) {
                support_levels.push(current);
            }
            
            // Local maximum (resistance)
            if window_before.iter().all(|&x| x <= current) && 
               window_after.iter().all(|&x| x <= current) {
                resistance_levels.push(current);
            }
        }
        
        // Sort and deduplicate
        support_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        resistance_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Keep only significant levels
        support_levels = consolidate_levels(support_levels, 0.001);
        resistance_levels = consolidate_levels(resistance_levels, 0.001);
        
        // Limit to top 3 each
        support_levels.truncate(3);
        resistance_levels.truncate(3);
        
        Ok((support_levels, resistance_levels))
    }

    fn calculate_volatility(&self, closes: &[f64]) -> Result<f64, TrendError> {
        if closes.len() < 20 {
            return Ok(0.0);
        }
        
        // Calculate returns
        let returns: Vec<f64> = closes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        // Standard deviation of returns
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(variance.sqrt() * 100.0) // Percentage volatility
    }

    fn assess_trend_quality(&self, closes: &[f64]) -> Result<f64, TrendError> {
        if closes.len() < 50 {
            return Ok(0.5);
        }
        
        // Linear regression for trend line
        let x: Vec<f64> = (0..closes.len()).map(|i| i as f64).collect();
        let (slope, intercept) = linear_regression(&x, closes);
        
        // Calculate R-squared
        let mean_y = closes.iter().sum::<f64>() / closes.len() as f64;
        let ss_tot: f64 = closes.iter().map(|y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(closes)
            .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
            .sum();
        
        let r_squared = 1.0 - (ss_res / ss_tot);
        
        // Trend consistency (fewer reversals)
        let reversals = count_trend_reversals(closes);
        let reversal_rate = reversals as f64 / closes.len() as f64;
        
        // Combine metrics
        let quality = r_squared * 0.7 + (1.0 - reversal_rate.min(0.5)) * 0.3;
        
        Ok(quality.max(0.0).min(1.0))
    }

    fn aggregate_scores(&self, scores: &[TrendMetrics]) -> (f64, TrendDirection, f64) {
        if scores.is_empty() {
            return (0.5, TrendDirection::Neutral, 0.0);
        }
        
        // Weight scores by timeframe (higher timeframes get more weight)
        let weights: Vec<f64> = (0..scores.len())
            .map(|i| 1.0 + (i as f64 * 0.5))
            .collect();
        
        let total_weight: f64 = weights.iter().sum();
        
        let weighted_trend_strength: f64 = scores.iter()
            .zip(&weights)
            .map(|(s, w)| s.trend_strength * w)
            .sum::<f64>() / total_weight;
        
        let weighted_momentum: f64 = scores.iter()
            .zip(&weights)
            .map(|(s, w)| s.momentum_score * w)
            .sum::<f64>() / total_weight;
        
        let avg_quality: f64 = scores.iter()
            .map(|s| s.trend_quality)
            .sum::<f64>() / scores.len() as f64;
        
        // Determine direction
        let direction = match weighted_trend_strength {
            x if x > 0.6 => TrendDirection::StrongBullish,
            x if x > 0.2 => TrendDirection::Bullish,
            x if x < -0.6 => TrendDirection::StrongBearish,
            x if x < -0.2 => TrendDirection::Bearish,
            _ => TrendDirection::Neutral,
        };
        
        // Overall score combines trend, momentum, and quality
        let overall = (weighted_trend_strength.abs() * 0.4 + 
                      weighted_momentum.abs() * 0.3 + 
                      avg_quality * 0.3).max(0.0).min(1.0);
        
        // Confidence based on consistency across timeframes
        let consistency = 1.0 - scores.iter()
            .map(|s| (s.trend_strength - weighted_trend_strength).abs())
            .sum::<f64>() / scores.len() as f64;
        
        (overall, direction, consistency)
    }

    fn detect_market_regime(&self, scores: &[TrendMetrics]) -> Result<MarketRegime, TrendError> {
        if scores.is_empty() {
            return Ok(MarketRegime::Ranging(RangingRegime {
                range_high: 0.0,
                range_low: 0.0,
                oscillation_count: 0,
            }));
        }
        
        // Analyze metrics to determine regime
        let avg_volatility = scores.iter().map(|s| s.volatility).sum::<f64>() / scores.len() as f64;
        let avg_trend_strength = scores.iter().map(|s| s.trend_strength.abs()).sum::<f64>() / scores.len() as f64;
        let avg_breakout_prob = scores.iter().map(|s| s.breakout_probability).sum::<f64>() / scores.len() as f64;
        
        // Breakout regime
        if avg_breakout_prob > 0.7 {
            return Ok(MarketRegime::Breakout(BreakoutRegime {
                breakout_level: 0.0, // Would need actual price data
                volume_surge: scores.last().unwrap().volume_confirmation,
                confirmation_strength: avg_breakout_prob,
            }));
        }
        
        // Volatile regime
        if avg_volatility > 2.5 {
            return Ok(MarketRegime::Volatile(VolatileRegime {
                volatility_percentile: 90.0,
                average_true_range: avg_volatility,
                risk_level: if avg_volatility > 4.0 { RiskLevel::Extreme } 
                           else if avg_volatility > 3.0 { RiskLevel::High }
                           else { RiskLevel::Medium },
            }));
        }
        
        // Trending regime
        if avg_trend_strength > 0.5 {
            let direction = if scores.last().unwrap().trend_strength > 0.0 {
                TrendDirection::Bullish
            } else {
                TrendDirection::Bearish
            };
            
            return Ok(MarketRegime::Trending(TrendingRegime {
                direction,
                strength: avg_trend_strength,
                duration_candles: 0, // Would need historical data
            }));
        }
        
        // Default to ranging
        Ok(MarketRegime::Ranging(RangingRegime {
            range_high: scores.iter()
                .flat_map(|s| &s.resistance_levels)
                .copied()
                .fold(0.0_f64, f64::max),
            range_low: scores.iter()
                .flat_map(|s| &s.support_levels)
                .copied()
                .fold(f64::INFINITY, f64::min),
            oscillation_count: 0,
        }))
    }

    async fn calculate_key_levels(&self, symbol: &str) -> Result<KeyLevels, TrendError> {
        // Get daily data for key levels
        let daily_data = self.fetch_ohlcv(symbol, "1d").await?;
        
        let highs = daily_data.column("high")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let lows = daily_data.column("low")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let closes = daily_data.column("close")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        // Calculate pivot point
        let last_high = highs.last().copied().unwrap_or(0.0);
        let last_low = lows.last().copied().unwrap_or(0.0);
        let last_close = closes.last().copied().unwrap_or(0.0);
        
        let pivot_point = (last_high + last_low + last_close) / 3.0;
        
        // Calculate support/resistance from pivot
        let r1 = 2.0 * pivot_point - last_low;
        let s1 = 2.0 * pivot_point - last_high;
        let r2 = pivot_point + (last_high - last_low);
        let s2 = pivot_point - (last_high - last_low);
        
        // Find historical S/R levels
        let (hist_support, hist_resistance) = self.find_key_levels(&closes)?;
        
        // Combine pivot and historical levels
        let mut support_levels = vec![
            SupportResistance {
                price: s1,
                strength: 0.7,
                touches: 0,
                last_test: Utc::now(),
            },
            SupportResistance {
                price: s2,
                strength: 0.5,
                touches: 0,
                last_test: Utc::now(),
            },
        ];
        
        let mut resistance_levels = vec![
            SupportResistance {
                price: r1,
                strength: 0.7,
                touches: 0,
                last_test: Utc::now(),
            },
            SupportResistance {
                price: r2,
                strength: 0.5,
                touches: 0,
                last_test: Utc::now(),
            },
        ];
        
        // Add historical levels
        for price in hist_support {
            support_levels.push(SupportResistance {
                price,
                strength: 0.8,
                touches: count_price_touches(&lows, price, 0.001),
                last_test: Utc::now(),
            });
        }
        
        for price in hist_resistance {
            resistance_levels.push(SupportResistance {
                price,
                strength: 0.8,
                touches: count_price_touches(&highs, price, 0.001),
                last_test: Utc::now(),
            });
        }
        
        // Calculate Fibonacci levels
        let fibonacci_levels = if !highs.is_empty() && !lows.is_empty() {
            let swing_high = highs.iter().copied().fold(0.0_f64, f64::max);
            let swing_low = lows.iter().copied().fold(f64::INFINITY, f64::min);
            let diff = swing_high - swing_low;
            
            vec![
                FibonacciLevel {
                    level: 0.236,
                    price: swing_high - diff * 0.236,
                    retracement_pct: 23.6,
                },
                FibonacciLevel {
                    level: 0.382,
                    price: swing_high - diff * 0.382,
                    retracement_pct: 38.2,
                },
                FibonacciLevel {
                    level: 0.500,
                    price: swing_high - diff * 0.500,
                    retracement_pct: 50.0,
                },
                FibonacciLevel {
                    level: 0.618,
                    price: swing_high - diff * 0.618,
                    retracement_pct: 61.8,
                },
            ]
        } else {
            vec![]
        };
        
        Ok(KeyLevels {
            pivot_point,
            support_levels,
            resistance_levels,
            fibonacci_levels,
        })
    }
}

// Helper functions
fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![];
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = vec![0.0; data.len()];
    
    // Initialize with SMA
    if data.len() >= period {
        ema[period - 1] = data[..period].iter().sum::<f64>() / period as f64;
        
        for i in period..data.len() {
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
        }
    }
    
    ema
}

fn calculate_sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }
    
    data.windows(period)
        .map(|w| w.iter().sum::<f64>() / period as f64)
        .collect()
}

fn calculate_rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![];
    }
    
    let mut gains = vec![];
    let mut losses = vec![];
    
    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    let mut rsi = vec![];
    let mut avg_gain = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss = losses[..period].iter().sum::<f64>() / period as f64;
    
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        
        let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
        rsi.push(100.0 - (100.0 / (1.0 + rs)));
    }
    
    rsi
}

fn calculate_macd(data: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = calculate_ema(data, fast);
    let ema_slow = calculate_ema(data, slow);
    
    let macd_line: Vec<f64> = ema_fast.iter()
        .zip(&ema_slow)
        .map(|(f, s)| f - s)
        .collect();
    
    let signal_line = calculate_ema(&macd_line, signal);
    
    let histogram: Vec<f64> = macd_line.iter()
        .zip(&signal_line)
        .map(|(m, s)| m - s)
        .collect();
    
    (macd_line, signal_line, histogram)
}

fn calculate_bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sma = calculate_sma(data, period);
    let mut upper = vec![];
    let mut lower = vec![];
    
    for i in 0..sma.len() {
        let window_start = i;
        let window_end = i + period;
        let window = &data[window_start..window_end];
        
        let mean = sma[i];
        let variance = window.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;
        let std_deviation = variance.sqrt();
        
        upper.push(mean + std_dev * std_deviation);
        lower.push(mean - std_dev * std_deviation);
    }
    
    (upper, sma, lower)
}

fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return vec![];
    }
    
    let mut true_ranges = vec![];
    
    for i in 1..highs.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        
        true_ranges.push(high_low.max(high_close).max(low_close));
    }
    
    calculate_ema(&true_ranges, period)
}

fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_y2: f64 = y.iter().map(|yi| yi * yi).sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    (slope, intercept)
}

fn count_trend_reversals(data: &[f64]) -> usize {
    if data.len() < 3 {
        return 0;
    }
    
    let mut reversals = 0;
    let mut prev_direction = if data[1] > data[0] { 1 } else { -1 };
    
    for i in 2..data.len() {
        let current_direction = if data[i] > data[i - 1] { 1 } else { -1 };
        if current_direction != prev_direction {
            reversals += 1;
            prev_direction = current_direction;
        }
    }
    
    reversals
}

fn consolidate_levels(mut levels: Vec<f64>, threshold: f64) -> Vec<f64> {
    if levels.is_empty() {
        return levels;
    }
    
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut consolidated = vec![levels[0]];
    
    for &level in &levels[1..] {
        let last = consolidated.last().unwrap();
        if (level - last) / last > threshold {
            consolidated.push(level);
        }
    }
    
    consolidated
}

fn count_price_touches(prices: &[f64], level: f64, tolerance: f64) -> u32 {
    prices.iter()
        .filter(|&&price| (price - level).abs() / level <= tolerance)
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ema = calculate_ema(&data, 3);
        assert_eq!(ema.len(), data.len());
        assert!(ema[2] > 0.0);
    }

    #[test]
    fn test_trend_direction() {
        assert_eq!(TrendDirection::Bullish, TrendDirection::Bullish);
    }
}