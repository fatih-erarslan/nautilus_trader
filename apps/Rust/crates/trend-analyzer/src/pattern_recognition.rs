use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartPattern {
    HeadAndShoulders(HeadAndShouldersPattern),
    DoubleTop(DoubleTopPattern),
    DoubleBottom(DoubleBottomPattern),
    Triangle(TrianglePattern),
    Flag(FlagPattern),
    Wedge(WedgePattern),
    Cup(CupPattern),
    Channel(ChannelPattern),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadAndShouldersPattern {
    pub left_shoulder: f64,
    pub head: f64,
    pub right_shoulder: f64,
    pub neckline: f64,
    pub pattern_type: PatternType,
    pub completion_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleTopPattern {
    pub first_peak: f64,
    pub second_peak: f64,
    pub valley: f64,
    pub pattern_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleBottomPattern {
    pub first_bottom: f64,
    pub second_bottom: f64,
    pub peak: f64,
    pub pattern_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrianglePattern {
    pub pattern_type: TriangleType,
    pub upper_trendline: Vec<(usize, f64)>,
    pub lower_trendline: Vec<(usize, f64)>,
    pub apex_distance: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagPattern {
    pub pole_start: f64,
    pub pole_end: f64,
    pub flag_high: f64,
    pub flag_low: f64,
    pub direction: TrendDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WedgePattern {
    pub pattern_type: WedgeType,
    pub upper_line: Vec<(usize, f64)>,
    pub lower_line: Vec<(usize, f64)>,
    pub convergence_point: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CupPattern {
    pub left_peak: f64,
    pub bottom: f64,
    pub right_peak: f64,
    pub handle_low: f64,
    pub pattern_depth: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelPattern {
    pub upper_channel: Vec<(usize, f64)>,
    pub lower_channel: Vec<(usize, f64)>,
    pub channel_width: f64,
    pub channel_slope: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    Bullish,
    Bearish,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriangleType {
    Ascending,
    Descending,
    Symmetrical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WedgeType {
    Rising,
    Falling,
}

pub struct PatternRecognizer {
    min_pattern_bars: usize,
    similarity_threshold: f64,
}

impl PatternRecognizer {
    pub fn new() -> Self {
        Self {
            min_pattern_bars: 20,
            similarity_threshold: 0.95,
        }
    }
    
    pub fn detect_patterns(&self, ohlcv: &DataFrame) -> Vec<ChartPattern> {
        let mut patterns = vec![];
        
        if let Ok(highs) = ohlcv.column("high").and_then(|c| c.f64()) {
            if let Ok(lows) = ohlcv.column("low").and_then(|c| c.f64()) {
                if let Ok(closes) = ohlcv.column("close").and_then(|c| c.f64()) {
                    let highs: Vec<f64> = highs.to_vec().into_iter().flatten().collect();
                    let lows: Vec<f64> = lows.to_vec().into_iter().flatten().collect();
                    let closes: Vec<f64> = closes.to_vec().into_iter().flatten().collect();
                    
                    // Detect various patterns
                    if let Some(pattern) = self.detect_head_and_shoulders(&highs, &lows) {
                        patterns.push(ChartPattern::HeadAndShoulders(pattern));
                    }
                    
                    if let Some(pattern) = self.detect_double_top(&highs) {
                        patterns.push(ChartPattern::DoubleTop(pattern));
                    }
                    
                    if let Some(pattern) = self.detect_double_bottom(&lows) {
                        patterns.push(ChartPattern::DoubleBottom(pattern));
                    }
                    
                    if let Some(pattern) = self.detect_triangle(&highs, &lows) {
                        patterns.push(ChartPattern::Triangle(pattern));
                    }
                    
                    if let Some(pattern) = self.detect_flag(&highs, &lows, &closes) {
                        patterns.push(ChartPattern::Flag(pattern));
                    }
                    
                    if let Some(pattern) = self.detect_wedge(&highs, &lows) {
                        patterns.push(ChartPattern::Wedge(pattern));
                    }
                    
                    if let Some(pattern) = self.detect_cup(&highs, &lows) {
                        patterns.push(ChartPattern::Cup(pattern));
                    }
                    
                    if let Some(pattern) = self.detect_channel(&highs, &lows) {
                        patterns.push(ChartPattern::Channel(pattern));
                    }
                }
            }
        }
        
        patterns
    }
    
    fn detect_head_and_shoulders(&self, highs: &[f64], lows: &[f64]) -> Option<HeadAndShouldersPattern> {
        if highs.len() < 50 {
            return None;
        }
        
        // Find potential peaks
        let peaks = self.find_peaks(highs, 5);
        
        if peaks.len() < 3 {
            return None;
        }
        
        // Look for H&S pattern in last 3 peaks
        let recent_peaks = &peaks[peaks.len().saturating_sub(3)..];
        
        if recent_peaks.len() == 3 {
            let (idx1, peak1) = recent_peaks[0];
            let (idx2, peak2) = recent_peaks[1];
            let (idx3, peak3) = recent_peaks[2];
            
            // Check if middle peak is highest (head)
            if peak2 > peak1 && peak2 > peak3 {
                // Check shoulder symmetry
                let shoulder_diff = (peak1 - peak3).abs() / peak1;
                if shoulder_diff < 0.05 { // 5% tolerance
                    // Find neckline
                    let valley1 = lows[idx1..idx2].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let valley2 = lows[idx2..idx3].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let neckline = (valley1 + valley2) / 2.0;
                    
                    return Some(HeadAndShouldersPattern {
                        left_shoulder: peak1,
                        head: peak2,
                        right_shoulder: peak3,
                        neckline,
                        pattern_type: PatternType::Bearish,
                        completion_percentage: self.calculate_completion(highs, idx3, neckline),
                    });
                }
            }
        }
        
        None
    }
    
    fn detect_double_top(&self, highs: &[f64]) -> Option<DoubleTopPattern> {
        let peaks = self.find_peaks(highs, 5);
        
        if peaks.len() < 2 {
            return None;
        }
        
        // Check last two peaks
        let (idx1, peak1) = peaks[peaks.len() - 2];
        let (idx2, peak2) = peaks[peaks.len() - 1];
        
        // Check if peaks are similar height
        let peak_diff = (peak1 - peak2).abs() / peak1;
        if peak_diff < 0.03 { // 3% tolerance
            // Find valley between peaks
            let valley = highs[idx1..idx2].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            let confidence = 1.0 - peak_diff;
            
            return Some(DoubleTopPattern {
                first_peak: peak1,
                second_peak: peak2,
                valley,
                pattern_confidence: confidence,
            });
        }
        
        None
    }
    
    fn detect_double_bottom(&self, lows: &[f64]) -> Option<DoubleBottomPattern> {
        let troughs = self.find_troughs(lows, 5);
        
        if troughs.len() < 2 {
            return None;
        }
        
        // Check last two troughs
        let (idx1, trough1) = troughs[troughs.len() - 2];
        let (idx2, trough2) = troughs[troughs.len() - 1];
        
        // Check if troughs are similar depth
        let trough_diff = (trough1 - trough2).abs() / trough1;
        if trough_diff < 0.03 { // 3% tolerance
            // Find peak between troughs
            let peak = lows[idx1..idx2].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            let confidence = 1.0 - trough_diff;
            
            return Some(DoubleBottomPattern {
                first_bottom: trough1,
                second_bottom: trough2,
                peak,
                pattern_confidence: confidence,
            });
        }
        
        None
    }
    
    fn detect_triangle(&self, highs: &[f64], lows: &[f64]) -> Option<TrianglePattern> {
        if highs.len() < 30 {
            return None;
        }
        
        // Get recent data
        let lookback = 30;
        let start_idx = highs.len().saturating_sub(lookback);
        
        let recent_highs = &highs[start_idx..];
        let recent_lows = &lows[start_idx..];
        
        // Find peaks and troughs
        let peaks = self.find_peaks(recent_highs, 3);
        let troughs = self.find_troughs(recent_lows, 3);
        
        if peaks.len() < 2 || troughs.len() < 2 {
            return None;
        }
        
        // Fit trendlines
        let upper_line = self.fit_trendline(&peaks);
        let lower_line = self.fit_trendline(&troughs);
        
        // Determine triangle type
        let upper_slope = self.calculate_slope(&upper_line);
        let lower_slope = self.calculate_slope(&lower_line);
        
        let pattern_type = if upper_slope > 0.0001 && lower_slope.abs() < 0.0001 {
            TriangleType::Ascending
        } else if upper_slope.abs() < 0.0001 && lower_slope < -0.0001 {
            TriangleType::Descending
        } else if upper_slope < -0.0001 && lower_slope > 0.0001 {
            TriangleType::Symmetrical
        } else {
            return None;
        };
        
        // Calculate apex
        let apex_distance = self.calculate_apex_distance(&upper_line, &lower_line);
        
        Some(TrianglePattern {
            pattern_type,
            upper_trendline: upper_line,
            lower_trendline: lower_line,
            apex_distance,
        })
    }
    
    fn detect_flag(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> Option<FlagPattern> {
        if closes.len() < 20 {
            return None;
        }
        
        // Look for strong directional move (pole)
        let pole_period = 10;
        let flag_period = 10;
        
        if closes.len() < pole_period + flag_period {
            return None;
        }
        
        let pole_start_idx = closes.len() - pole_period - flag_period;
        let pole_end_idx = closes.len() - flag_period;
        
        let pole_start = closes[pole_start_idx];
        let pole_end = closes[pole_end_idx];
        let pole_move = (pole_end - pole_start) / pole_start;
        
        // Need strong move for pole (>5%)
        if pole_move.abs() > 0.05 {
            // Check consolidation in flag
            let flag_highs = &highs[pole_end_idx..];
            let flag_lows = &lows[pole_end_idx..];
            
            let flag_high = flag_highs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let flag_low = flag_lows.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            let flag_range = (flag_high - flag_low) / flag_low;
            
            // Flag should be tight consolidation (<2% range)
            if flag_range < 0.02 {
                let direction = if pole_move > 0.0 {
                    TrendDirection::Bullish
                } else {
                    TrendDirection::Bearish
                };
                
                return Some(FlagPattern {
                    pole_start,
                    pole_end,
                    flag_high,
                    flag_low,
                    direction,
                });
            }
        }
        
        None
    }
    
    fn detect_wedge(&self, highs: &[f64], lows: &[f64]) -> Option<WedgePattern> {
        if highs.len() < 20 {
            return None;
        }
        
        let lookback = 20;
        let start_idx = highs.len().saturating_sub(lookback);
        
        let recent_highs = &highs[start_idx..];
        let recent_lows = &lows[start_idx..];
        
        let peaks = self.find_peaks(recent_highs, 3);
        let troughs = self.find_troughs(recent_lows, 3);
        
        if peaks.len() < 2 || troughs.len() < 2 {
            return None;
        }
        
        let upper_line = self.fit_trendline(&peaks);
        let lower_line = self.fit_trendline(&troughs);
        
        let upper_slope = self.calculate_slope(&upper_line);
        let lower_slope = self.calculate_slope(&lower_line);
        
        // Both lines should be sloping in same direction
        if upper_slope * lower_slope > 0.0 {
            // Lines should be converging
            let convergence_point = self.calculate_convergence(&upper_line, &lower_line);
            
            if convergence_point > 0 && convergence_point < 50 {
                let pattern_type = if upper_slope > 0.0 {
                    WedgeType::Rising
                } else {
                    WedgeType::Falling
                };
                
                return Some(WedgePattern {
                    pattern_type,
                    upper_line,
                    lower_line,
                    convergence_point,
                });
            }
        }
        
        None
    }
    
    fn detect_cup(&self, highs: &[f64], lows: &[f64]) -> Option<CupPattern> {
        if highs.len() < 30 {
            return None;
        }
        
        // Look for U-shaped pattern
        let lookback = 30;
        let start_idx = highs.len().saturating_sub(lookback);
        let end_idx = highs.len();
        
        let section = &lows[start_idx..end_idx];
        
        // Find minimum (bottom of cup)
        let (bottom_idx, bottom) = section.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &v)| (i, v))?;
        
        // Check left and right sides
        if bottom_idx > 5 && bottom_idx < section.len() - 5 {
            let left_peak = highs[start_idx..start_idx + bottom_idx].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let right_peak = highs[start_idx + bottom_idx..end_idx].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            // Peaks should be similar height
            let peak_diff = (left_peak - right_peak).abs() / left_peak;
            if peak_diff < 0.05 {
                // Calculate pattern depth
                let pattern_depth = (left_peak - bottom) / left_peak;
                
                // Look for handle (small dip after right peak)
                let handle_low = if end_idx < highs.len() - 5 {
                    lows[end_idx..end_idx + 5].iter().fold(f64::INFINITY, |a, &b| a.min(b))
                } else {
                    right_peak * 0.98
                };
                
                return Some(CupPattern {
                    left_peak,
                    bottom,
                    right_peak,
                    handle_low,
                    pattern_depth,
                });
            }
        }
        
        None
    }
    
    fn detect_channel(&self, highs: &[f64], lows: &[f64]) -> Option<ChannelPattern> {
        if highs.len() < 20 {
            return None;
        }
        
        let lookback = 20;
        let start_idx = highs.len().saturating_sub(lookback);
        
        let recent_highs = &highs[start_idx..];
        let recent_lows = &lows[start_idx..];
        
        let peaks = self.find_peaks(recent_highs, 3);
        let troughs = self.find_troughs(recent_lows, 3);
        
        if peaks.len() < 2 || troughs.len() < 2 {
            return None;
        }
        
        let upper_line = self.fit_trendline(&peaks);
        let lower_line = self.fit_trendline(&troughs);
        
        let upper_slope = self.calculate_slope(&upper_line);
        let lower_slope = self.calculate_slope(&lower_line);
        
        // Slopes should be parallel (similar)
        let slope_diff = (upper_slope - lower_slope).abs();
        let avg_slope = (upper_slope + lower_slope) / 2.0;
        
        if slope_diff < avg_slope.abs() * 0.2 { // 20% tolerance
            // Calculate channel width
            let channel_width = upper_line.iter()
                .zip(&lower_line)
                .map(|((_, u), (_, l))| u - l)
                .sum::<f64>() / upper_line.len() as f64;
            
            return Some(ChannelPattern {
                upper_channel: upper_line,
                lower_channel: lower_line,
                channel_width,
                channel_slope: avg_slope,
            });
        }
        
        None
    }
    
    fn find_peaks(&self, data: &[f64], window: usize) -> Vec<(usize, f64)> {
        let mut peaks = vec![];
        
        for i in window..data.len() - window {
            let current = data[i];
            let is_peak = data[i - window..i].iter().all(|&v| v <= current) &&
                         data[i + 1..=i + window].iter().all(|&v| v <= current);
            
            if is_peak {
                peaks.push((i, current));
            }
        }
        
        peaks
    }
    
    fn find_troughs(&self, data: &[f64], window: usize) -> Vec<(usize, f64)> {
        let mut troughs = vec![];
        
        for i in window..data.len() - window {
            let current = data[i];
            let is_trough = data[i - window..i].iter().all(|&v| v >= current) &&
                           data[i + 1..=i + window].iter().all(|&v| v >= current);
            
            if is_trough {
                troughs.push((i, current));
            }
        }
        
        troughs
    }
    
    fn fit_trendline(&self, points: &[(usize, f64)]) -> Vec<(usize, f64)> {
        if points.len() < 2 {
            return vec![];
        }
        
        let x: Vec<f64> = points.iter().map(|(idx, _)| *idx as f64).collect();
        let y: Vec<f64> = points.iter().map(|(_, val)| *val).collect();
        
        let (slope, intercept) = linear_regression(&x, &y);
        
        points.iter()
            .map(|(idx, _)| (*idx, slope * (*idx as f64) + intercept))
            .collect()
    }
    
    fn calculate_slope(&self, line: &[(usize, f64)]) -> f64 {
        if line.len() < 2 {
            return 0.0;
        }
        
        let first = line.first().unwrap();
        let last = line.last().unwrap();
        
        (last.1 - first.1) / ((last.0 - first.0) as f64)
    }
    
    fn calculate_apex_distance(&self, upper: &[(usize, f64)], lower: &[(usize, f64)]) -> usize {
        let upper_slope = self.calculate_slope(upper);
        let lower_slope = self.calculate_slope(lower);
        
        if (upper_slope - lower_slope).abs() < 0.00001 {
            return 999; // Parallel lines
        }
        
        // Calculate intersection point
        let (u_idx, u_val) = upper.first().unwrap();
        let (l_idx, l_val) = lower.first().unwrap();
        
        let x_intersect = (l_val - u_val + upper_slope * (*u_idx as f64) - lower_slope * (*l_idx as f64)) 
                         / (upper_slope - lower_slope);
        
        (x_intersect - *u_idx.max(l_idx) as f64).max(0.0) as usize
    }
    
    fn calculate_convergence(&self, upper: &[(usize, f64)], lower: &[(usize, f64)]) -> usize {
        self.calculate_apex_distance(upper, lower)
    }
    
    fn calculate_completion(&self, data: &[f64], pattern_end: usize, target_level: f64) -> f64 {
        if pattern_end >= data.len() {
            return 0.0;
        }
        
        let current_price = data.last().unwrap();
        let pattern_price = data[pattern_end];
        
        let expected_move = (target_level - pattern_price).abs();
        let actual_move = (current_price - pattern_price).abs();
        
        (actual_move / expected_move).min(1.0) * 100.0
    }
}