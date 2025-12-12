use crate::*;

pub struct AdvancedIndicators;

impl AdvancedIndicators {
    pub fn ichimoku_cloud(
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> IchimokuResult {
        let conversion_period = 9;
        let base_period = 26;
        let span_b_period = 52;
        let displacement = 26;
        
        // Conversion Line (Tenkan-sen)
        let conversion_line = Self::donchian_middle(high, low, conversion_period);
        
        // Base Line (Kijun-sen)
        let base_line = Self::donchian_middle(high, low, base_period);
        
        // Leading Span A (Senkou Span A)
        let span_a: Vec<f64> = conversion_line.iter()
            .zip(&base_line)
            .map(|(c, b)| (c + b) / 2.0)
            .collect();
        
        // Leading Span B (Senkou Span B)
        let span_b = Self::donchian_middle(high, low, span_b_period);
        
        // Lagging Span (Chikou Span) - close displaced backwards
        let lagging_span = close.to_vec();
        
        IchimokuResult {
            conversion_line,
            base_line,
            span_a,
            span_b,
            lagging_span,
            displacement,
        }
    }
    
    fn donchian_middle(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
        if high.len() < period || low.len() < period {
            return vec![];
        }
        
        (0..=high.len() - period)
            .map(|i| {
                let h_max = high[i..i + period].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let l_min = low[i..i + period].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                (h_max + l_min) / 2.0
            })
            .collect()
    }
    
    pub fn vwap(price: &[f64], volume: &[f64]) -> Vec<f64> {
        if price.len() != volume.len() || price.is_empty() {
            return vec![];
        }
        
        let mut cumulative_pv = 0.0;
        let mut cumulative_volume = 0.0;
        let mut vwap = Vec::with_capacity(price.len());
        
        for i in 0..price.len() {
            cumulative_pv += price[i] * volume[i];
            cumulative_volume += volume[i];
            
            if cumulative_volume > 0.0 {
                vwap.push(cumulative_pv / cumulative_volume);
            } else {
                vwap.push(price[i]);
            }
        }
        
        vwap
    }
    
    pub fn super_trend(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        atr_period: usize,
        multiplier: f64,
    ) -> (Vec<f64>, Vec<bool>) {
        let atr = calculate_atr(high, low, close, atr_period);
        let hl_avg: Vec<f64> = high.iter()
            .zip(low)
            .map(|(h, l)| (h + l) / 2.0)
            .collect();
        
        let mut upper_band = vec![];
        let mut lower_band = vec![];
        let mut super_trend = vec![];
        let mut trend = vec![];
        
        for i in 0..atr.len() {
            let idx = i + atr_period - 1;
            if idx < hl_avg.len() {
                let ub = hl_avg[idx] + multiplier * atr[i];
                let lb = hl_avg[idx] - multiplier * atr[i];
                
                upper_band.push(ub);
                lower_band.push(lb);
                
                // Determine trend
                if i == 0 {
                    super_trend.push(ub);
                    trend.push(false);
                } else {
                    let prev_trend = trend[i - 1];
                    let current_close = close[idx];
                    
                    if prev_trend {
                        if current_close <= lower_band[i] {
                            super_trend.push(lower_band[i]);
                            trend.push(false);
                        } else {
                            super_trend.push(upper_band[i]);
                            trend.push(true);
                        }
                    } else {
                        if current_close >= upper_band[i] {
                            super_trend.push(upper_band[i]);
                            trend.push(true);
                        } else {
                            super_trend.push(lower_band[i]);
                            trend.push(false);
                        }
                    }
                }
            }
        }
        
        (super_trend, trend)
    }
    
    pub fn keltner_channels(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        ema_period: usize,
        atr_period: usize,
        multiplier: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema = calculate_ema(close, ema_period);
        let atr = calculate_atr(high, low, close, atr_period);
        
        let min_len = ema.len().min(atr.len());
        
        let upper: Vec<f64> = (0..min_len)
            .map(|i| ema[i] + multiplier * atr[i])
            .collect();
        
        let lower: Vec<f64> = (0..min_len)
            .map(|i| ema[i] - multiplier * atr[i])
            .collect();
        
        (upper, ema[..min_len].to_vec(), lower)
    }
    
    pub fn donchian_channels(
        high: &[f64],
        low: &[f64],
        period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut upper = vec![];
        let mut lower = vec![];
        let mut middle = vec![];
        
        for i in period..high.len() {
            let h_max = high[i - period..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let l_min = low[i - period..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            upper.push(h_max);
            lower.push(l_min);
            middle.push((h_max + l_min) / 2.0);
        }
        
        (upper, middle, lower)
    }
    
    pub fn stochastic_oscillator(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k_period: usize,
        d_period: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut k_values = vec![];
        
        for i in k_period..close.len() {
            let highest = high[i - k_period..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = low[i - k_period..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            let k = if highest != lowest {
                (close[i - 1] - lowest) / (highest - lowest) * 100.0
            } else {
                50.0
            };
            
            k_values.push(k);
        }
        
        let d_values = calculate_sma(&k_values, d_period);
        
        (k_values, d_values)
    }
    
    pub fn williams_r(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        period: usize,
    ) -> Vec<f64> {
        let mut williams_r = vec![];
        
        for i in period..close.len() {
            let highest = high[i - period..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = low[i - period..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            let wr = if highest != lowest {
                (highest - close[i - 1]) / (highest - lowest) * -100.0
            } else {
                -50.0
            };
            
            williams_r.push(wr);
        }
        
        williams_r
    }
    
    pub fn commodity_channel_index(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        period: usize,
    ) -> Vec<f64> {
        let typical_price: Vec<f64> = high.iter()
            .zip(low)
            .zip(close)
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect();
        
        let sma = calculate_sma(&typical_price, period);
        let mut cci = vec![];
        
        for i in 0..sma.len() {
            let start_idx = i;
            let end_idx = i + period;
            
            if end_idx <= typical_price.len() {
                let window = &typical_price[start_idx..end_idx];
                let mean_deviation = window.iter()
                    .map(|&tp| (tp - sma[i]).abs())
                    .sum::<f64>() / period as f64;
                
                if mean_deviation != 0.0 {
                    cci.push((typical_price[end_idx - 1] - sma[i]) / (0.015 * mean_deviation));
                } else {
                    cci.push(0.0);
                }
            }
        }
        
        cci
    }
    
    pub fn money_flow_index(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        period: usize,
    ) -> Vec<f64> {
        let typical_price: Vec<f64> = high.iter()
            .zip(low)
            .zip(close)
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect();
        
        let raw_money_flow: Vec<f64> = typical_price.iter()
            .zip(volume)
            .map(|(tp, v)| tp * v)
            .collect();
        
        let mut mfi = vec![];
        
        for i in period..typical_price.len() {
            let mut positive_flow = 0.0;
            let mut negative_flow = 0.0;
            
            for j in 1..period {
                let idx = i - period + j;
                if typical_price[idx] > typical_price[idx - 1] {
                    positive_flow += raw_money_flow[idx];
                } else if typical_price[idx] < typical_price[idx - 1] {
                    negative_flow += raw_money_flow[idx];
                }
            }
            
            let money_ratio = if negative_flow > 0.0 {
                positive_flow / negative_flow
            } else {
                100.0
            };
            
            mfi.push(100.0 - (100.0 / (1.0 + money_ratio)));
        }
        
        mfi
    }
}

#[derive(Debug, Clone)]
pub struct IchimokuResult {
    pub conversion_line: Vec<f64>,
    pub base_line: Vec<f64>,
    pub span_a: Vec<f64>,
    pub span_b: Vec<f64>,
    pub lagging_span: Vec<f64>,
    pub displacement: usize,
}