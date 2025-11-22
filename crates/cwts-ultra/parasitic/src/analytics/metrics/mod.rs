//! Organism Metrics Module
//!
//! Comprehensive tracking of all 10 parasitic organism types with
//! performance analytics, resource efficiency, and lifecycle management.

use crate::analytics::{AnalyticsError, OrganismAnalyticsSummary, OrganismPerformanceData};
use crate::organisms::OrganismFactory;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;

/// Comprehensive organism metrics tracking system
pub struct OrganismMetrics {
    /// Per-organism performance data
    organism_data: Arc<DashMap<Uuid, OrganismTrackingData>>,

    /// Aggregate statistics cache
    aggregate_cache: Arc<TokioRwLock<AggregateStatistics>>,

    /// Organism type performance summaries
    type_summaries: Arc<RwLock<HashMap<String, TypePerformanceSummary>>>,

    /// Configuration
    config: OrganismMetricsConfig,
}

/// Configuration for organism metrics tracking
#[derive(Debug, Clone)]
pub struct OrganismMetricsConfig {
    pub max_history_per_organism: usize,
    pub performance_window: ChronoDuration,
    pub efficiency_calculation_samples: usize,
    pub trend_analysis_window: ChronoDuration,
}

impl Default for OrganismMetricsConfig {
    fn default() -> Self {
        Self {
            max_history_per_organism: 1000,
            performance_window: ChronoDuration::hours(1),
            efficiency_calculation_samples: 100,
            trend_analysis_window: ChronoDuration::minutes(30),
        }
    }
}

/// Comprehensive tracking data for each organism
#[derive(Debug, Clone)]
pub struct OrganismTrackingData {
    pub organism_id: Uuid,
    pub organism_type: String,
    pub creation_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub performance_history: VecDeque<OrganismPerformanceData>,
    pub lifecycle_info: LifecycleInfo,
    pub current_metrics: CurrentMetrics,
    pub trend_analysis: TrendAnalysis,
}

/// Organism lifecycle information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleInfo {
    pub creation_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub total_runtime: ChronoDuration,
    pub active_periods: Vec<ActivePeriod>,
    pub status: OrganismStatus,
}

/// Active period tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivePeriod {
    pub start: DateTime<Utc>,
    pub end: Option<DateTime<Utc>>,
    pub activity_level: f64,
}

/// Organism status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrganismStatus {
    Active,
    Idle,
    Degraded,
    Terminated,
}

/// Current performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentMetrics {
    pub performance_score: f64,
    pub resource_efficiency: f64,
    pub total_trades: u64,
    pub total_profit: f64,
    pub average_latency_ns: u64,
    pub current_success_rate: f64,
    pub last_profit_per_trade: f64,
}

impl Default for CurrentMetrics {
    fn default() -> Self {
        Self {
            performance_score: 0.0,
            resource_efficiency: 0.0,
            total_trades: 0,
            total_profit: 0.0,
            average_latency_ns: 0,
            current_success_rate: 0.0,
            last_profit_per_trade: 0.0,
        }
    }
}

/// Trend analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub performance_trend: f64, // Positive = improving, Negative = degrading
    pub efficiency_trend: f64,
    pub profit_trend: f64,
    pub stability_score: f64,
    pub prediction_confidence: f64,
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            performance_trend: 0.0,
            efficiency_trend: 0.0,
            profit_trend: 0.0,
            stability_score: 1.0,
            prediction_confidence: 0.5,
        }
    }
}

/// Aggregate statistics across all organisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStatistics {
    pub total_organisms: usize,
    pub active_organisms: usize,
    pub total_trades: u64,
    pub total_profit: f64,
    pub average_performance_score: f64,
    pub average_success_rate: f64,
    pub average_resource_efficiency: f64,
    pub top_performing_type: String,
    pub last_update: DateTime<Utc>,
}

impl Default for AggregateStatistics {
    fn default() -> Self {
        Self {
            total_organisms: 0,
            active_organisms: 0,
            total_trades: 0,
            total_profit: 0.0,
            average_performance_score: 0.0,
            average_success_rate: 0.0,
            average_resource_efficiency: 0.0,
            top_performing_type: "none".to_string(),
            last_update: Utc::now(),
        }
    }
}

/// Performance summary by organism type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypePerformanceSummary {
    pub organism_type: String,
    pub total_instances: usize,
    pub average_performance_score: f64,
    pub total_profit: f64,
    pub average_efficiency: f64,
    pub best_performer_id: Option<Uuid>,
    pub worst_performer_id: Option<Uuid>,
}

/// Organism ranking for leaderboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismRanking {
    pub organism_id: Uuid,
    pub organism_type: String,
    pub rank: usize,
    pub performance_score: f64,
    pub total_profit: f64,
    pub resource_efficiency: f64,
    pub trades_executed: u64,
}

/// Time series data point for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub performance_score: f64,
    pub profit: f64,
    pub success_rate: f64,
    pub latency_ns: u64,
}

impl OrganismMetrics {
    /// Create new organism metrics tracker
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            organism_data: Arc::new(DashMap::new()),
            aggregate_cache: Arc::new(TokioRwLock::new(AggregateStatistics::default())),
            type_summaries: Arc::new(RwLock::new(HashMap::new())),
            config: OrganismMetricsConfig::default(),
        })
    }

    /// Check if metrics system is initialized
    pub fn is_initialized(&self) -> bool {
        true // Always initialized after construction
    }

    /// Get all tracked organism types (should be all available types)
    pub fn get_tracked_organism_types(&self) -> Vec<String> {
        OrganismFactory::available_types()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get list of currently active organisms
    pub fn get_active_organisms(&self) -> Vec<OrganismAnalyticsSummary> {
        self.organism_data
            .iter()
            .filter(|entry| entry.value().lifecycle_info.status == OrganismStatus::Active)
            .map(|entry| self.create_summary_from_tracking_data(entry.value()))
            .collect()
    }

    /// Update metrics for an organism
    pub async fn update_metrics(
        &mut self,
        data: &OrganismPerformanceData,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let organism_id = data.organism_id;

        // Get or create tracking data for this organism
        // Fix E0597/E0716: Avoid temporary value lifetime issues by handling separately
        let needs_creation = !self.organism_data.contains_key(&organism_id);

        if needs_creation {
            // Create new tracking data
            let new_tracking_data = OrganismTrackingData {
                organism_id,
                organism_type: data.organism_type.clone(),
                creation_time: Utc::now(),
                last_update: data.timestamp,
                performance_history: VecDeque::with_capacity(self.config.max_history_per_organism),
                lifecycle_info: LifecycleInfo {
                    creation_time: Utc::now(),
                    last_update: data.timestamp,
                    total_runtime: ChronoDuration::zero(),
                    active_periods: vec![ActivePeriod {
                        start: data.timestamp,
                        end: None,
                        activity_level: 1.0,
                    }],
                    status: OrganismStatus::Active,
                },
                current_metrics: CurrentMetrics::default(),
                trend_analysis: TrendAnalysis::default(),
            };

            self.organism_data.insert(organism_id, new_tracking_data);
        }

        // Fix E0716: Get the entry and then get value_mut to avoid temporary
        let mut entry = self.organism_data.get_mut(&organism_id).unwrap();
        let tracking_data = entry.value_mut();

        // Update performance history
        if tracking_data.performance_history.len() >= self.config.max_history_per_organism {
            tracking_data.performance_history.pop_front();
        }
        tracking_data.performance_history.push_back(data.clone());

        // Update current metrics
        self.update_current_metrics(tracking_data, data);

        // Update lifecycle info
        tracking_data.lifecycle_info.last_update = data.timestamp;
        tracking_data.lifecycle_info.total_runtime =
            data.timestamp - tracking_data.lifecycle_info.creation_time;
        tracking_data.last_update = data.timestamp;

        // Update trend analysis
        self.update_trend_analysis(tracking_data);

        // Update type summaries
        self.update_type_summaries(&data.organism_type, tracking_data)
            .await;

        // Update aggregate statistics
        self.update_aggregate_statistics().await;

        Ok(())
    }

    /// Get performance summary for specific organism
    pub async fn get_organism_summary(
        &self,
        organism_id: Uuid,
    ) -> Result<OrganismAnalyticsSummary, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(tracking_data) = self.organism_data.get(&organism_id) {
            Ok(self.create_summary_from_tracking_data(tracking_data.value()))
        } else {
            Err(format!("Organism {} not found", organism_id).into())
        }
    }

    /// Get summaries for all organisms
    pub async fn get_all_organism_summaries(
        &self,
    ) -> Result<Vec<OrganismAnalyticsSummary>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self
            .organism_data
            .iter()
            .map(|entry| self.create_summary_from_tracking_data(entry.value()))
            .collect())
    }

    /// Get organism rankings sorted by performance
    pub async fn get_organism_rankings(
        &self,
    ) -> Result<Vec<OrganismRanking>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rankings: Vec<_> = self
            .organism_data
            .iter()
            .map(|entry| {
                let data = entry.value();
                OrganismRanking {
                    organism_id: data.organism_id,
                    organism_type: data.organism_type.clone(),
                    rank: 0, // Will be set after sorting
                    performance_score: data.current_metrics.performance_score,
                    total_profit: data.current_metrics.total_profit,
                    resource_efficiency: data.current_metrics.resource_efficiency,
                    trades_executed: data.current_metrics.total_trades,
                }
            })
            .collect();

        // Sort by performance score (descending)
        rankings.sort_by(|a, b| {
            b.performance_score
                .partial_cmp(&a.performance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, ranking) in rankings.iter_mut().enumerate() {
            ranking.rank = i + 1;
        }

        Ok(rankings)
    }

    /// Check if organism is currently active
    pub async fn is_organism_active(&self, organism_id: Uuid) -> bool {
        if let Some(tracking_data) = self.organism_data.get(&organism_id) {
            tracking_data.lifecycle_info.status == OrganismStatus::Active
        } else {
            false
        }
    }

    /// Get organism lifecycle information
    pub async fn get_organism_lifecycle_info(
        &self,
        organism_id: Uuid,
    ) -> Result<LifecycleInfo, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(tracking_data) = self.organism_data.get(&organism_id) {
            Ok(tracking_data.lifecycle_info.clone())
        } else {
            Err(format!("Organism {} not found", organism_id).into())
        }
    }

    /// Get time series data for trend analysis
    pub async fn get_organism_time_series(
        &self,
        organism_id: Uuid,
        duration: ChronoDuration,
    ) -> Result<Vec<TimeSeriesPoint>, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(tracking_data) = self.organism_data.get(&organism_id) {
            let cutoff_time = Utc::now() - duration;

            let time_series: Vec<_> = tracking_data
                .performance_history
                .iter()
                .filter(|data| data.timestamp >= cutoff_time)
                .map(|data| {
                    let performance_score = self.calculate_performance_score_from_data(data);
                    TimeSeriesPoint {
                        timestamp: data.timestamp,
                        performance_score,
                        profit: data.profit,
                        success_rate: data.success_rate,
                        latency_ns: data.latency_ns,
                    }
                })
                .collect();

            Ok(time_series)
        } else {
            Err(format!("Organism {} not found", organism_id).into())
        }
    }

    /// Calculate performance trend for organism
    pub async fn calculate_performance_trend(
        &self,
        organism_id: Uuid,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(tracking_data) = self.organism_data.get(&organism_id) {
            Ok(tracking_data.trend_analysis.performance_trend)
        } else {
            Err(format!("Organism {} not found", organism_id).into())
        }
    }

    /// Get aggregate statistics across all organisms
    pub async fn get_aggregate_statistics(
        &self,
    ) -> Result<AggregateStatistics, Box<dyn std::error::Error + Send + Sync>> {
        let cache = self.aggregate_cache.read().await;
        Ok(cache.clone())
    }

    // Private helper methods

    fn create_summary_from_tracking_data(
        &self,
        tracking_data: &OrganismTrackingData,
    ) -> OrganismAnalyticsSummary {
        OrganismAnalyticsSummary {
            organism_id: tracking_data.organism_id,
            organism_type: tracking_data.organism_type.clone(),
            performance_score: tracking_data.current_metrics.performance_score,
            total_trades: tracking_data.current_metrics.total_trades,
            total_profit: tracking_data.current_metrics.total_profit,
            average_latency_ns: tracking_data.current_metrics.average_latency_ns,
            success_rate: tracking_data.current_metrics.current_success_rate,
            resource_efficiency: tracking_data.current_metrics.resource_efficiency,
            last_active: tracking_data.last_update,
        }
    }

    fn update_current_metrics(
        &self,
        tracking_data: &mut OrganismTrackingData,
        data: &OrganismPerformanceData,
    ) {
        // Update totals
        tracking_data.current_metrics.total_trades += data.trades_executed;
        tracking_data.current_metrics.total_profit += data.profit;
        tracking_data.current_metrics.current_success_rate = data.success_rate;

        // Calculate profit per trade
        if tracking_data.current_metrics.total_trades > 0 {
            tracking_data.current_metrics.last_profit_per_trade =
                tracking_data.current_metrics.total_profit
                    / tracking_data.current_metrics.total_trades as f64;
        }

        // Update average latency (exponentially weighted moving average)
        let alpha = 0.1;
        if tracking_data.current_metrics.average_latency_ns == 0 {
            tracking_data.current_metrics.average_latency_ns = data.latency_ns;
        } else {
            tracking_data.current_metrics.average_latency_ns =
                ((1.0 - alpha) * tracking_data.current_metrics.average_latency_ns as f64
                    + alpha * data.latency_ns as f64) as u64;
        }

        // Calculate resource efficiency
        tracking_data.current_metrics.resource_efficiency =
            self.calculate_resource_efficiency(data);

        // Calculate performance score
        tracking_data.current_metrics.performance_score =
            self.calculate_performance_score_from_data(data);
    }

    fn calculate_resource_efficiency(&self, data: &OrganismPerformanceData) -> f64 {
        // Resource efficiency = profit per unit of resource consumed
        let total_resource_cost = data.resource_usage.cpu_usage * 0.1 +           // CPU weight
            data.resource_usage.memory_mb * 0.01 +          // Memory weight  
            data.resource_usage.network_bandwidth_kbps * 0.001 + // Network weight
            data.resource_usage.api_calls_per_second * 0.05; // API calls weight

        if total_resource_cost > 0.0 {
            (data.profit / total_resource_cost).clamp(0.0, 10.0) // Clamp to reasonable range
        } else {
            0.0
        }
    }

    fn calculate_performance_score_from_data(&self, data: &OrganismPerformanceData) -> f64 {
        // Multi-factor performance score
        let latency_score = (1.0 - (data.latency_ns as f64 / 1_000_000.0)).clamp(0.0, 1.0); // Normalize to 1ms
        let throughput_score = (data.throughput / 100.0).clamp(0.0, 1.0); // Normalize to 100 TPS
        let success_score = data.success_rate;
        let profit_score = (data.profit / 1000.0).clamp(0.0, 1.0); // Normalize to $1000

        // Weighted combination
        0.25 * latency_score + 0.25 * throughput_score + 0.3 * success_score + 0.2 * profit_score
    }

    fn update_trend_analysis(&self, tracking_data: &mut OrganismTrackingData) {
        if tracking_data.performance_history.len() < 10 {
            return; // Need sufficient data for trend analysis
        }

        let recent_data: Vec<_> = tracking_data
            .performance_history
            .iter()
            .rev()
            .take(20)
            .collect();

        if recent_data.len() < 10 {
            return;
        }

        // Calculate performance trend using linear regression
        let performance_scores: Vec<f64> = recent_data
            .iter()
            .rev()
            .map(|data| self.calculate_performance_score_from_data(data))
            .collect();

        tracking_data.trend_analysis.performance_trend = self.calculate_trend(&performance_scores);

        // Calculate other trends
        let profits: Vec<f64> = recent_data.iter().rev().map(|data| data.profit).collect();
        tracking_data.trend_analysis.profit_trend = self.calculate_trend(&profits);

        let efficiencies: Vec<f64> = recent_data
            .iter()
            .rev()
            .map(|data| self.calculate_resource_efficiency(data))
            .collect();
        tracking_data.trend_analysis.efficiency_trend = self.calculate_trend(&efficiencies);

        // Calculate stability score (inverse of variance)
        let variance = self.calculate_variance(&performance_scores);
        tracking_data.trend_analysis.stability_score = (1.0 / (1.0 + variance)).clamp(0.0, 1.0);

        // Prediction confidence based on data consistency
        tracking_data.trend_analysis.prediction_confidence =
            (tracking_data.trend_analysis.stability_score * 0.7
                + (recent_data.len() as f64 / 50.0).clamp(0.0, 1.0) * 0.3)
                .clamp(0.0, 1.0);
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        // Simple linear trend calculation
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        if n * sum_x2 - sum_x.powi(2) != 0.0 {
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
        } else {
            0.0
        }
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }

    async fn update_type_summaries(
        &self,
        organism_type: &str,
        tracking_data: &OrganismTrackingData,
    ) {
        let mut summaries = self.type_summaries.write();

        let summary = summaries
            .entry(organism_type.to_string())
            .or_insert_with(|| TypePerformanceSummary {
                organism_type: organism_type.to_string(),
                total_instances: 0,
                average_performance_score: 0.0,
                total_profit: 0.0,
                average_efficiency: 0.0,
                best_performer_id: None,
                worst_performer_id: None,
            });

        // Update type summary (simplified - would be more complex in real implementation)
        summary.total_instances += 1;
        summary.total_profit += tracking_data.current_metrics.total_profit;
        summary.average_performance_score = (summary.average_performance_score
            * (summary.total_instances - 1) as f64
            + tracking_data.current_metrics.performance_score)
            / summary.total_instances as f64;
    }

    async fn update_aggregate_statistics(&self) {
        let mut cache = self.aggregate_cache.write().await;

        let total_organisms = self.organism_data.len();
        let active_organisms = self
            .organism_data
            .iter()
            .filter(|entry| entry.value().lifecycle_info.status == OrganismStatus::Active)
            .count();

        let (total_trades, total_profit, sum_performance, sum_success_rate, sum_efficiency) = self
            .organism_data
            .iter()
            .fold((0u64, 0.0f64, 0.0f64, 0.0f64, 0.0f64), |acc, entry| {
                let metrics = &entry.value().current_metrics;
                (
                    acc.0 + metrics.total_trades,
                    acc.1 + metrics.total_profit,
                    acc.2 + metrics.performance_score,
                    acc.3 + metrics.current_success_rate,
                    acc.4 + metrics.resource_efficiency,
                )
            });

        cache.total_organisms = total_organisms;
        cache.active_organisms = active_organisms;
        cache.total_trades = total_trades;
        cache.total_profit = total_profit;

        if total_organisms > 0 {
            cache.average_performance_score = sum_performance / total_organisms as f64;
            cache.average_success_rate = sum_success_rate / total_organisms as f64;
            cache.average_resource_efficiency = sum_efficiency / total_organisms as f64;
        }

        // Find top performing organism type
        let type_summaries = self.type_summaries.read();
        cache.top_performing_type = type_summaries
            .values()
            .max_by(|a, b| {
                a.average_performance_score
                    .partial_cmp(&b.average_performance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|summary| summary.organism_type.clone())
            .unwrap_or_else(|| "none".to_string());

        cache.last_update = Utc::now();
    }
}
