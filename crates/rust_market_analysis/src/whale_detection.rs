//! Whale Detection System
//! 
//! Advanced algorithms to detect large institutional trades and whale activity
//! in cryptocurrency markets through volume analysis and order flow patterns.

use crate::market_data::MarketData;
use anyhow::Result;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct WhaleSignals {
    pub major_whale_detected: bool,
    pub whale_direction: f64, // -1.0 to 1.0 (sell to buy)
    pub whale_strength: f64,  // 0.0 to 1.0
    pub accumulation_score: f64,
    pub distribution_score: f64,
    pub unusual_volume: bool,
    pub large_order_flow: f64,
}

pub struct WhaleDetector {
    volume_threshold_multiplier: f64,
    price_impact_threshold: f64,
    accumulation_window: usize,
    large_trade_percentile: f64,
    recent_trades: VecDeque<TradeData>,
}

#[derive(Debug, Clone)]
struct TradeData {
    volume: f64,
    price: f64,
    price_change: f64,
    volume_ratio: f64,
    timestamp_index: usize,
}

impl WhaleDetector {
    pub fn new() -> Self {
        Self {
            volume_threshold_multiplier: 3.0,
            price_impact_threshold: 0.02,
            accumulation_window: 50,
            large_trade_percentile: 95.0,
            recent_trades: VecDeque::with_capacity(1000),
        }
    }

    pub fn detect_whale_activity(&mut self, data: &MarketData) -> Result<WhaleSignals> {
        if data.len() < 50 {
            return Ok(WhaleSignals::default());
        }

        // Update internal trade data
        self.update_trade_data(data);

        // Detect different types of whale activity
        let volume_anomalies = self.detect_volume_anomalies(data)?;
        let price_impact = self.detect_price_impact_trades(data)?;
        let accumulation = self.detect_accumulation_pattern(data)?;
        let distribution = self.detect_distribution_pattern(data)?;
        let order_flow = self.analyze_order_flow(data)?;

        // Combine signals
        let whale_strength = self.calculate_whale_strength(&volume_anomalies, &price_impact, &order_flow);
        let whale_direction = self.determine_whale_direction(&accumulation, &distribution, &order_flow);
        
        let major_whale_detected = whale_strength > 0.7 && (accumulation.score > 0.6 || distribution.score > 0.6);

        Ok(WhaleSignals {
            major_whale_detected,
            whale_direction,
            whale_strength,
            accumulation_score: accumulation.score,
            distribution_score: distribution.score,
            unusual_volume: volume_anomalies.unusual_detected,
            large_order_flow: order_flow.net_flow,
        })
    }

    fn update_trade_data(&mut self, data: &MarketData) {
        let start_idx = if data.len() > 100 { data.len() - 100 } else { 0 };
        
        for i in start_idx..data.len() {
            if i > 0 {
                let trade = TradeData {
                    volume: data.volumes[i],
                    price: data.prices[i],
                    price_change: (data.prices[i] - data.prices[i-1]) / data.prices[i-1],
                    volume_ratio: self.calculate_volume_ratio(data, i),
                    timestamp_index: i,
                };
                
                self.recent_trades.push_back(trade);
                
                if self.recent_trades.len() > 1000 {
                    self.recent_trades.pop_front();
                }
            }
        }
    }

    fn detect_volume_anomalies(&self, data: &MarketData) -> Result<VolumeAnomalies> {
        let volumes = &data.volumes;
        
        if volumes.len() < 20 {
            return Ok(VolumeAnomalies::default());
        }

        // Calculate rolling volume statistics
        let mut anomalies = Vec::new();
        let window = 20;

        for i in window..volumes.len() {
            let window_volumes = &volumes[i-window..i];
            let mean_volume = window_volumes.iter().sum::<f64>() / window as f64;
            let std_volume = self.calculate_std(window_volumes, mean_volume);
            
            let current_volume = volumes[i];
            let z_score = if std_volume > 0.0 {
                (current_volume - mean_volume) / std_volume
            } else {
                0.0
            };

            if z_score > self.volume_threshold_multiplier {
                anomalies.push(VolumeAnomaly {
                    index: i,
                    volume: current_volume,
                    z_score,
                    volume_ratio: current_volume / mean_volume,
                });
            }
        }

        let unusual_detected = !anomalies.is_empty() && 
            anomalies.iter().any(|a| a.z_score > 4.0);

        let total_anomaly_strength = anomalies.iter()
            .map(|a| a.z_score.min(10.0) / 10.0)
            .sum::<f64>() / anomalies.len().max(1) as f64;

        Ok(VolumeAnomalies {
            unusual_detected,
            anomalies,
            strength: total_anomaly_strength,
        })
    }

    fn detect_price_impact_trades(&self, data: &MarketData) -> Result<PriceImpactSignals> {
        let mut impact_trades = Vec::new();
        
        for i in 1..data.len() {
            let price_change = (data.prices[i] - data.prices[i-1]).abs() / data.prices[i-1];
            let volume_ratio = self.calculate_volume_ratio(data, i);
            
            if price_change > self.price_impact_threshold && volume_ratio > 2.0 {
                impact_trades.push(PriceImpactTrade {
                    index: i,
                    price_impact: price_change,
                    volume_ratio,
                    direction: if data.prices[i] > data.prices[i-1] { 1.0 } else { -1.0 },
                });
            }
        }

        let total_impact = impact_trades.iter()
            .map(|t| t.price_impact * t.volume_ratio)
            .sum::<f64>();

        let count = impact_trades.len();

        Ok(PriceImpactSignals {
            impact_trades,
            total_impact,
            count,
        })
    }

    fn detect_accumulation_pattern(&self, data: &MarketData) -> Result<AccumulationSignal> {
        if data.len() < self.accumulation_window {
            return Ok(AccumulationSignal::default());
        }

        let start_idx = data.len() - self.accumulation_window;
        let window_data = &data.prices[start_idx..];
        let window_volumes = &data.volumes[start_idx..];

        // Look for: increasing volume + sideways/up price action
        let price_trend = self.calculate_trend(window_data);
        let volume_trend = self.calculate_trend(window_volumes);
        
        // Calculate volume-weighted price trend
        let vwap = self.calculate_vwap(&data.prices[start_idx..], &data.volumes[start_idx..]);
        let current_price = data.prices[data.len() - 1];
        let vwap_position = (current_price - vwap) / vwap;

        // Accumulation characteristics:
        // - Increasing volume
        // - Price staying above VWAP or consolidating
        // - Higher lows pattern
        let accumulation_score = if volume_trend > 0.0 && price_trend >= -0.02 && vwap_position > -0.01 {
            let higher_lows = self.detect_higher_lows(window_data);
            let volume_strength = (volume_trend * 2.0).min(1.0);
            let price_stability = (1.0 - price_trend.abs()).max(0.0);
            
            (volume_strength * 0.4 + price_stability * 0.3 + higher_lows * 0.3).min(1.0)
        } else {
            0.0
        };

        Ok(AccumulationSignal {
            score: accumulation_score,
            volume_trend,
            price_trend,
            vwap_position,
        })
    }

    fn detect_distribution_pattern(&self, data: &MarketData) -> Result<DistributionSignal> {
        if data.len() < self.accumulation_window {
            return Ok(DistributionSignal::default());
        }

        let start_idx = data.len() - self.accumulation_window;
        let window_data = &data.prices[start_idx..];
        let window_volumes = &data.volumes[start_idx..];

        let price_trend = self.calculate_trend(window_data);
        let volume_trend = self.calculate_trend(window_volumes);
        
        let vwap = self.calculate_vwap(&data.prices[start_idx..], &data.volumes[start_idx..]);
        let current_price = data.prices[data.len() - 1];
        let vwap_position = (current_price - vwap) / vwap;

        // Distribution characteristics:
        // - High volume on down moves
        // - Price below VWAP
        // - Lower highs pattern
        let distribution_score = if volume_trend > 0.0 && price_trend < 0.0 && vwap_position < 0.0 {
            let lower_highs = self.detect_lower_highs(window_data);
            let volume_on_decline = self.calculate_volume_on_decline(&data.prices[start_idx..], &data.volumes[start_idx..]);
            
            (volume_on_decline * 0.5 + lower_highs * 0.3 + (-vwap_position).min(1.0) * 0.2).min(1.0)
        } else {
            0.0
        };

        Ok(DistributionSignal {
            score: distribution_score,
            volume_trend,
            price_trend,
            vwap_position,
        })
    }

    fn analyze_order_flow(&self, data: &MarketData) -> Result<OrderFlowAnalysis> {
        let mut net_flow = 0.0;
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        // Estimate order flow based on price and volume
        for i in 1..data.len() {
            let price_change = data.prices[i] - data.prices[i-1];
            let volume = data.volumes[i];
            
            if price_change > 0.0 {
                buy_volume += volume;
                net_flow += volume;
            } else if price_change < 0.0 {
                sell_volume += volume;
                net_flow -= volume;
            } else {
                // For neutral moves, split volume
                buy_volume += volume * 0.5;
                sell_volume += volume * 0.5;
            }
        }

        let total_volume = buy_volume + sell_volume;
        let flow_ratio = if total_volume > 0.0 {
            net_flow / total_volume
        } else {
            0.0
        };

        Ok(OrderFlowAnalysis {
            net_flow: flow_ratio,
            buy_volume,
            sell_volume,
            flow_strength: flow_ratio.abs(),
        })
    }

    // Helper methods
    fn calculate_volume_ratio(&self, data: &MarketData, index: usize) -> f64 {
        let window = 20;
        let start = if index >= window { index - window } else { 0 };
        let window_volumes = &data.volumes[start..index];
        
        if window_volumes.is_empty() {
            return 1.0;
        }
        
        let avg_volume = window_volumes.iter().sum::<f64>() / window_volumes.len() as f64;
        if avg_volume > 0.0 {
            data.volumes[index] / avg_volume
        } else {
            1.0
        }
    }

    fn calculate_std(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let first_half = &values[..values.len()/2];
        let second_half = &values[values.len()/2..];
        
        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        (second_avg - first_avg) / first_avg
    }

    fn calculate_vwap(&self, prices: &[f64], volumes: &[f64]) -> f64 {
        let mut total_pv = 0.0;
        let mut total_volume = 0.0;
        
        for (price, volume) in prices.iter().zip(volumes.iter()) {
            total_pv += price * volume;
            total_volume += volume;
        }
        
        if total_volume > 0.0 {
            total_pv / total_volume
        } else {
            prices.iter().sum::<f64>() / prices.len() as f64
        }
    }

    fn detect_higher_lows(&self, prices: &[f64]) -> f64 {
        if prices.len() < 10 {
            return 0.0;
        }
        
        let mut lows = Vec::new();
        
        // Find local lows
        for i in 2..prices.len()-2 {
            if prices[i] < prices[i-1] && prices[i] < prices[i+1] &&
               prices[i] < prices[i-2] && prices[i] < prices[i+2] {
                lows.push(prices[i]);
            }
        }
        
        if lows.len() < 2 {
            return 0.0;
        }
        
        // Check if lows are generally increasing
        let mut higher_count = 0;
        for i in 1..lows.len() {
            if lows[i] > lows[i-1] {
                higher_count += 1;
            }
        }
        
        higher_count as f64 / (lows.len() - 1) as f64
    }

    fn detect_lower_highs(&self, prices: &[f64]) -> f64 {
        if prices.len() < 10 {
            return 0.0;
        }
        
        let mut highs = Vec::new();
        
        // Find local highs
        for i in 2..prices.len()-2 {
            if prices[i] > prices[i-1] && prices[i] > prices[i+1] &&
               prices[i] > prices[i-2] && prices[i] > prices[i+2] {
                highs.push(prices[i]);
            }
        }
        
        if highs.len() < 2 {
            return 0.0;
        }
        
        // Check if highs are generally decreasing
        let mut lower_count = 0;
        for i in 1..highs.len() {
            if highs[i] < highs[i-1] {
                lower_count += 1;
            }
        }
        
        lower_count as f64 / (highs.len() - 1) as f64
    }

    fn calculate_volume_on_decline(&self, prices: &[f64], volumes: &[f64]) -> f64 {
        let mut decline_volume = 0.0;
        let mut total_volume = 0.0;
        
        for i in 1..prices.len() {
            total_volume += volumes[i];
            if prices[i] < prices[i-1] {
                decline_volume += volumes[i];
            }
        }
        
        if total_volume > 0.0 {
            decline_volume / total_volume
        } else {
            0.0
        }
    }

    fn calculate_whale_strength(&self, volume_anomalies: &VolumeAnomalies, 
                               price_impact: &PriceImpactSignals, 
                               order_flow: &OrderFlowAnalysis) -> f64 {
        let volume_component = volume_anomalies.strength * 0.4;
        let impact_component = (price_impact.total_impact / 10.0).min(1.0) * 0.4;
        let flow_component = order_flow.flow_strength * 0.2;
        
        volume_component + impact_component + flow_component
    }

    fn determine_whale_direction(&self, accumulation: &AccumulationSignal,
                                distribution: &DistributionSignal,
                                order_flow: &OrderFlowAnalysis) -> f64 {
        let acc_signal = accumulation.score * 1.0;  // Bullish
        let dist_signal = distribution.score * -1.0; // Bearish
        let flow_signal = order_flow.net_flow * 0.5;
        
        (acc_signal + dist_signal + flow_signal).clamp(-1.0, 1.0)
    }
}

// Supporting structures
#[derive(Debug, Clone, Default)]
struct VolumeAnomalies {
    unusual_detected: bool,
    anomalies: Vec<VolumeAnomaly>,
    strength: f64,
}

#[derive(Debug, Clone)]
struct VolumeAnomaly {
    index: usize,
    volume: f64,
    z_score: f64,
    volume_ratio: f64,
}

#[derive(Debug, Clone, Default)]
struct PriceImpactSignals {
    impact_trades: Vec<PriceImpactTrade>,
    total_impact: f64,
    count: usize,
}

#[derive(Debug, Clone)]
struct PriceImpactTrade {
    index: usize,
    price_impact: f64,
    volume_ratio: f64,
    direction: f64,
}

#[derive(Debug, Clone, Default)]
struct AccumulationSignal {
    score: f64,
    volume_trend: f64,
    price_trend: f64,
    vwap_position: f64,
}

#[derive(Debug, Clone, Default)]
struct DistributionSignal {
    score: f64,
    volume_trend: f64,
    price_trend: f64,
    vwap_position: f64,
}

#[derive(Debug, Clone, Default)]
struct OrderFlowAnalysis {
    net_flow: f64,
    buy_volume: f64,
    sell_volume: f64,
    flow_strength: f64,
}

impl Default for WhaleSignals {
    fn default() -> Self {
        Self {
            major_whale_detected: false,
            whale_direction: 0.0,
            whale_strength: 0.0,
            accumulation_score: 0.0,
            distribution_score: 0.0,
            unusual_volume: false,
            large_order_flow: 0.0,
        }
    }
}

impl Default for WhaleDetector {
    fn default() -> Self {
        Self::new()
    }
}