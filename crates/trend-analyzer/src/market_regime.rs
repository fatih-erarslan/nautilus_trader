use crate::*;

pub struct RegimeAnalyzer {
    volatility_lookback: usize,
    trend_lookback: usize,
    volume_lookback: usize,
}

impl RegimeAnalyzer {
    pub fn new() -> Self {
        Self {
            volatility_lookback: 20,
            trend_lookback: 50,
            volume_lookback: 20,
        }
    }
    
    pub fn detect_regime(
        &self,
        ohlcv: &DataFrame,
    ) -> Result<MarketRegime, TrendError> {
        let closes = ohlcv.column("close")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let highs = ohlcv.column("high")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let lows = ohlcv.column("low")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let volumes = ohlcv.column("volume")?.f64()?.to_vec()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        // Calculate various metrics
        let volatility_score = self.calculate_volatility_score(&closes)?;
        let trend_score = self.calculate_trend_score(&closes)?;
        let volume_score = self.calculate_volume_score(&volumes)?;
        let breakout_score = self.calculate_breakout_score(&highs, &lows, &closes, &volumes)?;
        
        // Decision logic for regime
        if breakout_score > 0.7 {
            Ok(MarketRegime::Breakout(BreakoutRegime {
                breakout_level: *closes.last().unwrap(),
                volume_surge: volume_score,
                confirmation_strength: breakout_score,
            }))
        } else if volatility_score > 0.7 {
            let risk_level = match volatility_score {
                x if x > 0.9 => RiskLevel::Extreme,
                x if x > 0.8 => RiskLevel::High,
                x if x > 0.6 => RiskLevel::Medium,
                _ => RiskLevel::Low,
            };
            
            Ok(MarketRegime::Volatile(VolatileRegime {
                volatility_percentile: volatility_score * 100.0,
                average_true_range: self.calculate_atr(&highs, &lows, &closes),
                risk_level,
            }))
        } else if trend_score.0.abs() > 0.5 {
            let direction = if trend_score.0 > 0.0 {
                TrendDirection::Bullish
            } else {
                TrendDirection::Bearish
            };
            
            Ok(MarketRegime::Trending(TrendingRegime {
                direction,
                strength: trend_score.0.abs(),
                duration_candles: trend_score.1,
            }))
        } else {
            // Default to ranging
            let (range_high, range_low) = self.calculate_range(&highs, &lows);
            let oscillation_count = self.count_oscillations(&closes, range_high, range_low);
            
            Ok(MarketRegime::Ranging(RangingRegime {
                range_high,
                range_low,
                oscillation_count,
            }))
        }
    }
    
    fn calculate_volatility_score(&self, closes: &[f64]) -> Result<f64, TrendError> {
        if closes.len() < self.volatility_lookback {
            return Ok(0.5);
        }
        
        // Calculate returns
        let returns: Vec<f64> = closes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        // Recent volatility
        let recent_start = returns.len().saturating_sub(self.volatility_lookback);
        let recent_returns = &returns[recent_start..];
        
        let recent_vol = self.calculate_std_dev(recent_returns);
        
        // Historical volatility for comparison
        let hist_vol = self.calculate_std_dev(&returns);
        
        // Volatility percentile
        let vol_percentile = if hist_vol > 0.0 {
            recent_vol / hist_vol
        } else {
            1.0
        };
        
        Ok(vol_percentile.min(1.0))
    }
    
    fn calculate_trend_score(&self, closes: &[f64]) -> Result<(f64, u32), TrendError> {
        if closes.len() < self.trend_lookback {
            return Ok((0.0, 0));
        }
        
        // Linear regression on recent data
        let recent_start = closes.len().saturating_sub(self.trend_lookback);
        let recent_closes = &closes[recent_start..];
        
        let x: Vec<f64> = (0..recent_closes.len()).map(|i| i as f64).collect();
        let (slope, _) = linear_regression(&x, recent_closes);
        
        // Normalize slope
        let avg_price = recent_closes.iter().sum::<f64>() / recent_closes.len() as f64;
        let normalized_slope = slope / avg_price * 100.0;
        
        // Count consecutive trend candles
        let mut trend_duration = 1;
        let mut prev_direction = if recent_closes[1] > recent_closes[0] { 1 } else { -1 };
        
        for i in 2..recent_closes.len() {
            let current_direction = if recent_closes[i] > recent_closes[i - 1] { 1 } else { -1 };
            if current_direction == prev_direction {
                trend_duration += 1;
            } else {
                break;
            }
        }
        
        Ok((normalized_slope.max(-1.0).min(1.0), trend_duration))
    }
    
    fn calculate_volume_score(&self, volumes: &[f64]) -> Result<f64, TrendError> {
        if volumes.len() < self.volume_lookback {
            return Ok(0.5);
        }
        
        let recent_start = volumes.len().saturating_sub(self.volume_lookback);
        let recent_volumes = &volumes[recent_start..];
        
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let recent_avg = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        
        let volume_ratio = recent_avg / avg_volume;
        
        Ok((volume_ratio - 1.0).max(0.0).min(1.0))
    }
    
    fn calculate_breakout_score(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
        volumes: &[f64],
    ) -> Result<f64, TrendError> {
        if closes.len() < 50 {
            return Ok(0.0);
        }
        
        let mut score: f64 = 0.0;
        
        // Check if price is breaking recent high/low
        let recent_high = highs[highs.len().saturating_sub(20)..].iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let recent_low = lows[lows.len().saturating_sub(20)..].iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        
        let current_price = *closes.last().unwrap();
        
        if current_price > recent_high {
            score += 0.4;
        } else if current_price < recent_low {
            score += 0.4;
        }
        
        // Volume confirmation
        let vol_score = self.calculate_volume_score(volumes)?;
        if vol_score > 0.5 {
            score += 0.3;
        }
        
        // Volatility expansion
        let vol_score = self.calculate_volatility_score(closes)?;
        if vol_score > 0.6 {
            score += 0.3;
        }
        
        Ok(score.min(1.0))
    }
    
    fn calculate_range(&self, highs: &[f64], lows: &[f64]) -> (f64, f64) {
        let recent_start = highs.len().saturating_sub(self.trend_lookback);
        
        let range_high = highs[recent_start..].iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range_low = lows[recent_start..].iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        
        (range_high, range_low)
    }
    
    fn count_oscillations(&self, closes: &[f64], range_high: f64, range_low: f64) -> u32 {
        let range_mid = (range_high + range_low) / 2.0;
        let mut oscillations = 0;
        let mut above_mid = closes[0] > range_mid;
        
        for &close in &closes[1..] {
            let currently_above = close > range_mid;
            if currently_above != above_mid {
                oscillations += 1;
                above_mid = currently_above;
            }
        }
        
        oscillations
    }
    
    fn calculate_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> f64 {
        let atr_values = calculate_atr(highs, lows, closes, 14);
        atr_values.last().copied().unwrap_or(0.0)
    }
    
    fn calculate_std_dev(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        variance.sqrt()
    }
}

pub struct RegimeTransitionDetector {
    lookback_periods: usize,
    transition_threshold: f64,
}

impl RegimeTransitionDetector {
    pub fn new() -> Self {
        Self {
            lookback_periods: 10,
            transition_threshold: 0.7,
        }
    }
    
    pub fn detect_transition(
        &self,
        current_regime: &MarketRegime,
        historical_regimes: &[MarketRegime],
    ) -> Option<RegimeTransition> {
        if historical_regimes.len() < self.lookback_periods {
            return None;
        }
        
        // Count regime occurrences
        let mut regime_counts = std::collections::HashMap::new();
        
        for regime in historical_regimes.iter().rev().take(self.lookback_periods) {
            let regime_type = match regime {
                MarketRegime::Trending(_) => "trending",
                MarketRegime::Ranging(_) => "ranging",
                MarketRegime::Volatile(_) => "volatile",
                MarketRegime::Breakout(_) => "breakout",
            };
            *regime_counts.entry(regime_type).or_insert(0) += 1;
        }
        
        // Find dominant historical regime
        let dominant_regime = regime_counts.iter()
            .max_by_key(|&(_, count)| count)
            .map(|(regime, _)| *regime)?;
        
        let current_type = match current_regime {
            MarketRegime::Trending(_) => "trending",
            MarketRegime::Ranging(_) => "ranging",
            MarketRegime::Volatile(_) => "volatile",
            MarketRegime::Breakout(_) => "breakout",
        };
        
        // Check if transition is occurring
        if dominant_regime != current_type {
            let transition_strength = regime_counts.get(dominant_regime).copied().unwrap_or(0) as f64
                / self.lookback_periods as f64;
            
            if transition_strength >= self.transition_threshold {
                return Some(RegimeTransition {
                    from: dominant_regime.to_string(),
                    to: current_type.to_string(),
                    confidence: transition_strength,
                    expected_duration: self.estimate_duration(current_type),
                });
            }
        }
        
        None
    }
    
    fn estimate_duration(&self, regime_type: &str) -> u32 {
        // Historical average durations (in candles)
        match regime_type {
            "trending" => 50,
            "ranging" => 30,
            "volatile" => 20,
            "breakout" => 10,
            _ => 25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegimeTransition {
    pub from: String,
    pub to: String,
    pub confidence: f64,
    pub expected_duration: u32,
}