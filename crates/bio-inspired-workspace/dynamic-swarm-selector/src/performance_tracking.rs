//! Performance tracking and monitoring for swarm algorithms
//! 
//! This module provides functionality to track and analyze the performance
//! of different swarm algorithms across various market conditions.

use crate::*;
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};

/// Performance tracker for swarm algorithms
pub struct PerformanceTracker {
    metrics_history: Arc<RwLock<HashMap<SwarmAlgorithm, Vec<SwarmPerformanceMetrics>>>>,
    regime_performance: Arc<RwLock<HashMap<(SwarmAlgorithm, MarketRegime), Vec<f64>>>>,
    real_time_metrics: Arc<RwLock<HashMap<SwarmAlgorithm, RealTimeMetrics>>>,
}

/// Real-time performance metrics
#[derive(Debug, Clone)]
pub struct RealTimeMetrics {
    pub current_score: f64,
    pub moving_average_score: f64,
    pub volatility: f64,
    pub trend: f64,
    pub last_updated: DateTime<Utc>,
    pub sample_count: u32,
}

impl RealTimeMetrics {
    pub fn new() -> Self {
        Self {
            current_score: 0.0,
            moving_average_score: 0.0,
            volatility: 0.0,
            trend: 0.0,
            last_updated: Utc::now(),
            sample_count: 0,
        }
    }
    
    pub fn update(&mut self, new_score: f64) {
        self.current_score = new_score;
        self.sample_count += 1;
        
        // Update moving average
        let alpha = 0.1; // Smoothing factor
        if self.sample_count == 1 {
            self.moving_average_score = new_score;
        } else {
            self.moving_average_score = alpha * new_score + (1.0 - alpha) * self.moving_average_score;
        }
        
        // Update volatility (simplified standard deviation)
        let deviation = new_score - self.moving_average_score;
        self.volatility = alpha * deviation.abs() + (1.0 - alpha) * self.volatility;
        
        // Update trend (simplified momentum)
        self.trend = new_score - self.moving_average_score;
        
        self.last_updated = Utc::now();
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            regime_performance: Arc::new(RwLock::new(HashMap::new())),
            real_time_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Record performance metrics for an algorithm
    pub async fn record_performance(&self, metrics: SwarmPerformanceMetrics) -> anyhow::Result<()> {
        // Update metrics history
        {
            let mut history = self.metrics_history.write().await;
            history.entry(metrics.algorithm)
                .or_insert_with(Vec::new)
                .push(metrics.clone());
        }
        
        // Update regime-specific performance
        {
            let mut regime_perf = self.regime_performance.write().await;
            regime_perf.entry((metrics.algorithm, metrics.regime))
                .or_insert_with(Vec::new)
                .push(metrics.optimization_score);
        }
        
        // Update real-time metrics
        {
            let mut real_time = self.real_time_metrics.write().await;
            let rt_metrics = real_time.entry(metrics.algorithm)
                .or_insert_with(RealTimeMetrics::new);
            rt_metrics.update(metrics.optimization_score);
        }
        
        Ok(())
    }
    
    /// Get historical performance for an algorithm
    pub async fn get_performance_history(&self, algorithm: SwarmAlgorithm) -> Option<Vec<SwarmPerformanceMetrics>> {
        let history = self.metrics_history.read().await;
        history.get(&algorithm).cloned()
    }
    
    /// Get average performance for an algorithm in a specific regime
    pub async fn get_regime_performance(&self, algorithm: SwarmAlgorithm, regime: MarketRegime) -> Option<f64> {
        let regime_perf = self.regime_performance.read().await;
        let scores = regime_perf.get(&(algorithm, regime))?;
        
        if scores.is_empty() {
            return None;
        }
        
        Some(scores.iter().sum::<f64>() / scores.len() as f64)
    }
    
    /// Get real-time metrics for an algorithm
    pub async fn get_real_time_metrics(&self, algorithm: SwarmAlgorithm) -> Option<RealTimeMetrics> {
        let real_time = self.real_time_metrics.read().await;
        real_time.get(&algorithm).cloned()
    }
    
    /// Get performance ranking of algorithms
    pub async fn get_performance_ranking(&self) -> Vec<(SwarmAlgorithm, f64)> {
        let real_time = self.real_time_metrics.read().await;
        let mut ranking: Vec<(SwarmAlgorithm, f64)> = real_time
            .iter()
            .map(|(alg, metrics)| (*alg, metrics.moving_average_score))
            .collect();
        
        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranking
    }
    
    /// Get algorithm recommendations based on performance
    pub async fn get_recommendations(&self, regime: MarketRegime) -> Vec<SwarmAlgorithm> {
        let regime_perf = self.regime_performance.read().await;
        let mut regime_scores: Vec<(SwarmAlgorithm, f64)> = Vec::new();
        
        for ((alg, reg), scores) in regime_perf.iter() {
            if *reg == regime && !scores.is_empty() {
                let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
                regime_scores.push((*alg, avg_score));
            }
        }
        
        regime_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        regime_scores.into_iter().map(|(alg, _)| alg).collect()
    }
    
    /// Calculate algorithm adaptation score
    pub async fn calculate_adaptation_score(&self, algorithm: SwarmAlgorithm) -> f64 {
        let real_time = self.real_time_metrics.read().await;
        
        if let Some(metrics) = real_time.get(&algorithm) {
            // Adaptation score based on trend and volatility
            let trend_score = if metrics.trend > 0.0 { 1.0 } else { -1.0 };
            let volatility_penalty = metrics.volatility * 0.5;
            let recency_bonus = if metrics.last_updated > Utc::now() - Duration::hours(1) {
                0.2
            } else {
                0.0
            };
            
            (trend_score - volatility_penalty + recency_bonus).max(0.0)
        } else {
            0.0
        }
    }
    
    /// Get performance statistics for an algorithm
    pub async fn get_statistics(&self, algorithm: SwarmAlgorithm) -> Option<PerformanceStatistics> {
        let history = self.metrics_history.read().await;
        let metrics = history.get(&algorithm)?;
        
        if metrics.is_empty() {
            return None;
        }
        
        let scores: Vec<f64> = metrics.iter().map(|m| m.optimization_score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();
        
        let min_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        Some(PerformanceStatistics {
            algorithm,
            sample_count: metrics.len() as u32,
            mean_score: mean,
            std_deviation: std_dev,
            min_score,
            max_score,
            success_rate: metrics.iter().map(|m| m.success_rate).sum::<f64>() / metrics.len() as f64,
            avg_convergence_time: metrics.iter().map(|m| m.convergence_time).sum::<chrono::Duration>() / metrics.len() as i32,
        })
    }
    
    /// Clear old performance data
    pub async fn cleanup_old_data(&self, _cutoff_time: DateTime<Utc>) -> anyhow::Result<()> {
        let mut history = self.metrics_history.write().await;
        
        for (_, metrics) in history.iter_mut() {
            metrics.retain(|m| {
                m.convergence_time > chrono::Duration::zero() && 
                m.optimization_score > 0.0 // Basic validity check
            });
        }
        
        Ok(())
    }
}

/// Performance statistics summary
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub algorithm: SwarmAlgorithm,
    pub sample_count: u32,
    pub mean_score: f64,
    pub std_deviation: f64,
    pub min_score: f64,
    pub max_score: f64,
    pub success_rate: f64,
    pub avg_convergence_time: Duration,
}

impl PerformanceStatistics {
    /// Calculate coefficient of variation
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean_score != 0.0 {
            self.std_deviation / self.mean_score
        } else {
            0.0
        }
    }
    
    /// Calculate performance stability (inverse of coefficient of variation)
    pub fn stability_score(&self) -> f64 {
        let cv = self.coefficient_of_variation();
        if cv > 0.0 {
            1.0 / (1.0 + cv)
        } else {
            1.0
        }
    }
    
    /// Calculate overall performance score
    pub fn overall_score(&self) -> f64 {
        let score_component = self.mean_score * 0.4;
        let success_component = self.success_rate * 0.3;
        let stability_component = self.stability_score() * 0.2;
        let efficiency_component = if self.avg_convergence_time.num_seconds() > 0 {
            (1.0 / self.avg_convergence_time.num_seconds() as f64) * 0.1
        } else {
            0.0
        };
        
        score_component + success_component + stability_component + efficiency_component
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();
        
        let metrics = SwarmPerformanceMetrics {
            algorithm: SwarmAlgorithm::ParticleSwarm,
            regime: MarketRegime::HighVolatility,
            optimization_score: 0.85,
            convergence_time: Duration::seconds(30),
            function_evaluations: 1000,
            success_rate: 0.9,
            stability_score: 0.8,
            exploration_ratio: 0.6,
            exploitation_ratio: 0.4,
            diversity_index: 0.7,
        };
        
        let result = tracker.record_performance(metrics.clone()).await;
        assert!(result.is_ok());
        
        let history = tracker.get_performance_history(SwarmAlgorithm::ParticleSwarm).await;
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 1);
        
        let regime_perf = tracker.get_regime_performance(
            SwarmAlgorithm::ParticleSwarm,
            MarketRegime::HighVolatility
        ).await;
        assert!(regime_perf.is_some());
        assert_eq!(regime_perf.unwrap(), 0.85);
    }
    
    #[tokio::test]
    async fn test_real_time_metrics() {
        let mut metrics = RealTimeMetrics::new();
        
        metrics.update(0.8);
        assert_eq!(metrics.current_score, 0.8);
        assert_eq!(metrics.moving_average_score, 0.8);
        
        metrics.update(0.9);
        assert_eq!(metrics.current_score, 0.9);
        assert!(metrics.moving_average_score > 0.8);
        assert!(metrics.moving_average_score < 0.9);
    }
    
    #[test]
    fn test_performance_statistics() {
        let stats = PerformanceStatistics {
            algorithm: SwarmAlgorithm::ParticleSwarm,
            sample_count: 100,
            mean_score: 0.8,
            std_deviation: 0.1,
            min_score: 0.6,
            max_score: 0.95,
            success_rate: 0.9,
            avg_convergence_time: Duration::seconds(30),
        };
        
        let cv = stats.coefficient_of_variation();
        assert_eq!(cv, 0.1 / 0.8);
        
        let stability = stats.stability_score();
        assert!(stability > 0.0 && stability <= 1.0);
        
        let overall = stats.overall_score();
        assert!(overall > 0.0);
    }
}