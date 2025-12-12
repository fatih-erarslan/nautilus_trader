use crate::*;
use std::collections::HashMap;

pub struct MultiTimeframeAnalyzer {
    timeframe_weights: HashMap<String, f64>,
    correlation_threshold: f64,
}

impl MultiTimeframeAnalyzer {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("1m".to_string(), 0.05);
        weights.insert("5m".to_string(), 0.10);
        weights.insert("15m".to_string(), 0.15);
        weights.insert("30m".to_string(), 0.20);
        weights.insert("1h".to_string(), 0.20);
        weights.insert("4h".to_string(), 0.15);
        weights.insert("1d".to_string(), 0.15);
        
        Self {
            timeframe_weights: weights,
            correlation_threshold: 0.6,
        }
    }
    
    pub fn analyze_confluence(
        &self,
        timeframe_data: HashMap<String, TrendMetrics>,
    ) -> ConfluenceResult {
        // Calculate weighted scores
        let mut weighted_trend = 0.0;
        let mut weighted_momentum = 0.0;
        let mut weighted_volume = 0.0;
        let mut total_weight = 0.0;
        
        for (tf, metrics) in &timeframe_data {
            if let Some(weight) = self.timeframe_weights.get(tf) {
                weighted_trend += metrics.trend_strength * weight;
                weighted_momentum += metrics.momentum_score * weight;
                weighted_volume += metrics.volume_confirmation * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            weighted_trend /= total_weight;
            weighted_momentum /= total_weight;
            weighted_volume /= total_weight;
        }
        
        // Calculate alignment score
        let alignment_score = self.calculate_alignment(&timeframe_data);
        
        // Determine confluence strength
        let confluence_strength = (weighted_trend.abs() + weighted_momentum.abs() + alignment_score) / 3.0;
        
        // Find key support/resistance across timeframes
        let unified_levels = self.unify_key_levels(&timeframe_data);
        
        ConfluenceResult {
            overall_trend: weighted_trend,
            overall_momentum: weighted_momentum,
            overall_volume: weighted_volume,
            alignment_score,
            confluence_strength,
            unified_support: unified_levels.0,
            unified_resistance: unified_levels.1,
            timeframe_agreement: self.calculate_agreement(&timeframe_data),
        }
    }
    
    fn calculate_alignment(&self, timeframe_data: &HashMap<String, TrendMetrics>) -> f64 {
        let trends: Vec<f64> = timeframe_data.values()
            .map(|m| m.trend_strength)
            .collect();
        
        if trends.len() < 2 {
            return 0.5;
        }
        
        // Check if all trends point in same direction
        let all_positive = trends.iter().all(|&t| t > 0.0);
        let all_negative = trends.iter().all(|&t| t < 0.0);
        
        if all_positive || all_negative {
            // Calculate variance for alignment strength
            let mean = trends.iter().sum::<f64>() / trends.len() as f64;
            let variance = trends.iter()
                .map(|t| (t - mean).powi(2))
                .sum::<f64>() / trends.len() as f64;
            
            // Lower variance = better alignment
            let alignment = 1.0 - variance.sqrt().min(1.0);
            alignment
        } else {
            // Mixed signals, calculate correlation
            let correlations = self.calculate_timeframe_correlations(&trends);
            correlations.iter().sum::<f64>() / correlations.len().max(1) as f64
        }
    }
    
    fn calculate_timeframe_correlations(&self, trends: &[f64]) -> Vec<f64> {
        let mut correlations = vec![];
        
        for i in 0..trends.len() {
            for j in i + 1..trends.len() {
                let corr = if trends[i].signum() == trends[j].signum() {
                    1.0 - (trends[i] - trends[j]).abs()
                } else {
                    0.0
                };
                correlations.push(corr);
            }
        }
        
        correlations
    }
    
    fn unify_key_levels(
        &self,
        timeframe_data: &HashMap<String, TrendMetrics>,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut all_support = vec![];
        let mut all_resistance = vec![];
        
        // Collect all levels with weights
        for (tf, metrics) in timeframe_data {
            let weight = self.timeframe_weights.get(tf).copied().unwrap_or(0.1);
            
            for &level in &metrics.support_levels {
                all_support.push((level, weight));
            }
            
            for &level in &metrics.resistance_levels {
                all_resistance.push((level, weight));
            }
        }
        
        // Cluster nearby levels
        let clustered_support = self.cluster_levels(all_support);
        let clustered_resistance = self.cluster_levels(all_resistance);
        
        (clustered_support, clustered_resistance)
    }
    
    fn cluster_levels(&self, levels: Vec<(f64, f64)>) -> Vec<f64> {
        if levels.is_empty() {
            return vec![];
        }
        
        // Sort by price
        let mut sorted_levels = levels;
        sorted_levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let mut clusters = vec![];
        let mut current_cluster = vec![sorted_levels[0]];
        let cluster_threshold = 0.001; // 0.1% difference
        
        for i in 1..sorted_levels.len() {
            let (price, weight) = sorted_levels[i];
            let cluster_center = current_cluster.iter()
                .map(|(p, _)| p)
                .sum::<f64>() / current_cluster.len() as f64;
            
            if (price - cluster_center).abs() / cluster_center < cluster_threshold {
                current_cluster.push((price, weight));
            } else {
                // Finalize current cluster
                let weighted_price = current_cluster.iter()
                    .map(|(p, w)| p * w)
                    .sum::<f64>() / current_cluster.iter()
                    .map(|(_, w)| w)
                    .sum::<f64>();
                
                clusters.push(weighted_price);
                current_cluster = vec![(price, weight)];
            }
        }
        
        // Don't forget last cluster
        if !current_cluster.is_empty() {
            let weighted_price = current_cluster.iter()
                .map(|(p, w)| p * w)
                .sum::<f64>() / current_cluster.iter()
                .map(|(_, w)| w)
                .sum::<f64>();
            clusters.push(weighted_price);
        }
        
        // Return top 3 strongest levels
        clusters.into_iter().take(3).collect()
    }
    
    fn calculate_agreement(&self, timeframe_data: &HashMap<String, TrendMetrics>) -> TimeframeAgreement {
        let mut bullish_count = 0;
        let mut bearish_count = 0;
        let mut neutral_count = 0;
        
        for metrics in timeframe_data.values() {
            if metrics.trend_strength > 0.2 {
                bullish_count += 1;
            } else if metrics.trend_strength < -0.2 {
                bearish_count += 1;
            } else {
                neutral_count += 1;
            }
        }
        
        let total = timeframe_data.len();
        
        TimeframeAgreement {
            bullish_percentage: (bullish_count as f64 / total as f64) * 100.0,
            bearish_percentage: (bearish_count as f64 / total as f64) * 100.0,
            neutral_percentage: (neutral_count as f64 / total as f64) * 100.0,
            consensus: if bullish_count > total / 2 {
                "Bullish".to_string()
            } else if bearish_count > total / 2 {
                "Bearish".to_string()
            } else {
                "Mixed".to_string()
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfluenceResult {
    pub overall_trend: f64,
    pub overall_momentum: f64,
    pub overall_volume: f64,
    pub alignment_score: f64,
    pub confluence_strength: f64,
    pub unified_support: Vec<f64>,
    pub unified_resistance: Vec<f64>,
    pub timeframe_agreement: TimeframeAgreement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeframeAgreement {
    pub bullish_percentage: f64,
    pub bearish_percentage: f64,
    pub neutral_percentage: f64,
    pub consensus: String,
}

pub struct TimeframeSynchronizer {
    base_timeframe: String,
    target_timeframes: Vec<String>,
}

impl TimeframeSynchronizer {
    pub fn new(base: String, targets: Vec<String>) -> Self {
        Self {
            base_timeframe: base,
            target_timeframes: targets,
        }
    }
    
    pub fn align_signals(
        &self,
        base_signal: &TrendMetrics,
        target_signals: HashMap<String, TrendMetrics>,
    ) -> AlignedSignals {
        let mut confirmations = vec![];
        let mut conflicts = vec![];
        
        for (tf, signal) in target_signals {
            let alignment = self.calculate_signal_alignment(base_signal, &signal);
            
            if alignment > 0.7 {
                confirmations.push(SignalConfirmation {
                    timeframe: tf,
                    strength: alignment,
                    metrics: signal.clone(),
                });
            } else if alignment < 0.3 {
                conflicts.push(SignalConflict {
                    timeframe: tf,
                    divergence: 1.0 - alignment,
                    metrics: signal.clone(),
                });
            }
        }
        
        let overall_confidence = self.calculate_overall_confidence(&confirmations, &conflicts);
        
        AlignedSignals {
            base_timeframe: self.base_timeframe.clone(),
            base_signal: base_signal.clone(),
            confirmations,
            conflicts,
            overall_confidence,
        }
    }
    
    fn calculate_signal_alignment(&self, base: &TrendMetrics, target: &TrendMetrics) -> f64 {
        let trend_alignment = 1.0 - (base.trend_strength - target.trend_strength).abs();
        let momentum_alignment = 1.0 - (base.momentum_score - target.momentum_score).abs();
        let volume_alignment = 1.0 - (base.volume_confirmation - target.volume_confirmation).abs();
        
        (trend_alignment + momentum_alignment + volume_alignment) / 3.0
    }
    
    fn calculate_overall_confidence(
        &self,
        confirmations: &[SignalConfirmation],
        conflicts: &[SignalConflict],
    ) -> f64 {
        let confirmation_score = confirmations.iter()
            .map(|c| c.strength)
            .sum::<f64>() / confirmations.len().max(1) as f64;
        
        let conflict_penalty = conflicts.iter()
            .map(|c| c.divergence)
            .sum::<f64>() / (conflicts.len().max(1) as f64 * 2.0);
        
        (confirmation_score - conflict_penalty).max(0.0).min(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct AlignedSignals {
    pub base_timeframe: String,
    pub base_signal: TrendMetrics,
    pub confirmations: Vec<SignalConfirmation>,
    pub conflicts: Vec<SignalConflict>,
    pub overall_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct SignalConfirmation {
    pub timeframe: String,
    pub strength: f64,
    pub metrics: TrendMetrics,
}

#[derive(Debug, Clone)]
pub struct SignalConflict {
    pub timeframe: String,
    pub divergence: f64,
    pub metrics: TrendMetrics,
}