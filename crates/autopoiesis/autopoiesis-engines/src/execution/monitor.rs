//! Execution monitoring implementation

use crate::prelude::*;
use crate::models::{Order, Trade};
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::{HashMap, VecDeque};

/// Execution monitor for tracking and analyzing execution performance
#[derive(Debug, Clone)]
pub struct ExecutionMonitor {
    /// Monitor configuration
    config: ExecutionMonitorConfig,
    
    /// Execution tracking data
    tracking_data: ExecutionTrackingData,
    
    /// Performance metrics
    metrics: ExecutionPerformanceMetrics,
    
    /// Alert history
    alert_history: VecDeque<ExecutionAlert>,
}

#[derive(Debug, Clone)]
pub struct ExecutionMonitorConfig {
    /// Monitoring frequency in seconds
    pub monitoring_frequency_seconds: u32,
    
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    
    /// Performance window size
    pub performance_window_hours: u32,
    
    /// Maximum alerts to retain
    pub max_alert_history: usize,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub max_slippage_bps: f64,
    pub max_execution_time_seconds: u32,
    pub min_fill_rate: f64,
    pub max_rejection_rate: f64,
    pub max_latency_ms: u32,
}

#[derive(Debug, Clone, Default)]
struct ExecutionTrackingData {
    active_orders: HashMap<uuid::Uuid, OrderTracking>,
    completed_executions: VecDeque<ExecutionRecord>,
    venue_performance: HashMap<String, VenuePerformance>,
}

#[derive(Debug, Clone)]
struct OrderTracking {
    order: Order,
    start_time: DateTime<Utc>,
    trades: Vec<Trade>,
    status_updates: Vec<StatusUpdate>,
    expected_completion: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct StatusUpdate {
    timestamp: DateTime<Utc>,
    status: String,
    details: String,
}

#[derive(Debug, Clone)]
struct ExecutionRecord {
    order_id: uuid::Uuid,
    venue: String,
    algorithm: String,
    start_time: DateTime<Utc>,
    completion_time: DateTime<Utc>,
    total_quantity: Decimal,
    filled_quantity: Decimal,
    average_price: Decimal,
    slippage_bps: f64,
    fees_paid: Decimal,
    execution_quality_score: f64,
}

#[derive(Debug, Clone, Default)]
struct VenuePerformance {
    venue_name: String,
    total_orders: u64,
    successful_orders: u64,
    average_latency_ms: f64,
    average_slippage_bps: f64,
    reliability_score: f64,
    last_updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionPerformanceMetrics {
    pub total_orders_monitored: u64,
    pub average_execution_time_seconds: f64,
    pub average_slippage_bps: f64,
    pub fill_rate: f64,
    pub rejection_rate: f64,
    pub average_latency_ms: f64,
    pub cost_efficiency_score: f64,
    pub execution_quality_score: f64,
    pub venue_distribution: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ExecutionAlert {
    pub timestamp: DateTime<Utc>,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub order_id: Option<uuid::Uuid>,
    pub venue: Option<String>,
    pub message: String,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    HighSlippage,
    SlowExecution,
    LowFillRate,
    HighRejectionRate,
    VenueLatency,
    ExecutionQuality,
    SystemAnomaly,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl Default for ExecutionMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_frequency_seconds: 10,
            alert_thresholds: AlertThresholds {
                max_slippage_bps: 20.0,
                max_execution_time_seconds: 300,
                min_fill_rate: 0.90,
                max_rejection_rate: 0.05,
                max_latency_ms: 200,
            },
            performance_window_hours: 24,
            max_alert_history: 1000,
        }
    }
}

impl ExecutionMonitor {
    /// Create a new execution monitor
    pub fn new(config: ExecutionMonitorConfig) -> Self {
        Self {
            config,
            tracking_data: ExecutionTrackingData::default(),
            metrics: ExecutionPerformanceMetrics::default(),
            alert_history: VecDeque::new(),
        }
    }

    /// Start monitoring an order
    pub async fn start_monitoring_order(&mut self, order: Order) -> Result<()> {
        let tracking = OrderTracking {
            order: order.clone(),
            start_time: Utc::now(),
            trades: Vec::new(),
            status_updates: Vec::new(),
            expected_completion: Utc::now() + Duration::minutes(5),
        };

        self.tracking_data.active_orders.insert(order.id, tracking);
        info!("Started monitoring order {}", order.id);
        Ok(())
    }

    /// Update order with trade information
    pub async fn update_order_trade(&mut self, order_id: uuid::Uuid, trade: Trade) -> Result<()> {
        if let Some(tracking) = self.tracking_data.active_orders.get_mut(&order_id) {
            tracking.trades.push(trade.clone());
            
            let status_update = StatusUpdate {
                timestamp: Utc::now(),
                status: "Trade Executed".to_string(),
                details: format!("Quantity: {}, Price: {}", trade.quantity, trade.price),
            };
            tracking.status_updates.push(status_update);

            // Check for alerts
            self.check_trade_alerts(&trade).await?;
        }

        Ok(())
    }

    /// Complete order monitoring
    pub async fn complete_order_monitoring(&mut self, order_id: uuid::Uuid) -> Result<ExecutionRecord> {
        if let Some(tracking) = self.tracking_data.active_orders.remove(&order_id) {
            let completion_time = Utc::now();
            let execution_time = (completion_time - tracking.start_time).num_seconds() as f64;

            // Calculate metrics
            let total_quantity = tracking.order.quantity;
            let filled_quantity: Decimal = tracking.trades.iter().map(|t| t.quantity).sum();
            let total_value: Decimal = tracking.trades.iter().map(|t| t.quantity * t.price).sum();
            let average_price = if filled_quantity > Decimal::ZERO {
                total_value / filled_quantity
            } else {
                Decimal::ZERO
            };

            // Calculate slippage (simplified)
            let expected_price = tracking.order.price.unwrap_or(average_price);
            let slippage_bps = if expected_price > Decimal::ZERO {
                ((average_price - expected_price) / expected_price * Decimal::from(10000)).to_f64().unwrap_or(0.0)
            } else {
                0.0
            };

            let fees_paid: Decimal = tracking.trades.iter().map(|t| t.fee).sum();
            let execution_quality_score = self.calculate_execution_quality_score(slippage_bps, execution_time, filled_quantity, total_quantity);

            let execution_record = ExecutionRecord {
                order_id,
                venue: "default".to_string(), // Would be determined from trades
                algorithm: "unknown".to_string(), // Would be stored in tracking
                start_time: tracking.start_time,
                completion_time,
                total_quantity,
                filled_quantity,
                average_price,
                slippage_bps,
                fees_paid,
                execution_quality_score,
            };

            // Update metrics
            self.update_performance_metrics(&execution_record).await;

            // Store execution record
            self.tracking_data.completed_executions.push_back(execution_record.clone());
            
            // Maintain history size
            while self.tracking_data.completed_executions.len() > 10000 {
                self.tracking_data.completed_executions.pop_front();
            }

            // Check for completion alerts
            self.check_completion_alerts(&execution_record).await?;

            Ok(execution_record)
        } else {
            Err(Error::Execution(format!("Order {} not found in monitoring", order_id)))
        }
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> ExecutionPerformanceMetrics {
        self.metrics.clone()
    }

    /// Get recent alerts
    pub async fn get_recent_alerts(&self, count: usize) -> Vec<ExecutionAlert> {
        self.alert_history.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Generate performance report
    pub async fn generate_performance_report(&self, hours: u32) -> Result<PerformanceReport> {
        let cutoff_time = Utc::now() - Duration::hours(hours as i64);
        
        let recent_executions: Vec<&ExecutionRecord> = self.tracking_data.completed_executions
            .iter()
            .filter(|record| record.completion_time > cutoff_time)
            .collect();

        if recent_executions.is_empty() {
            return Ok(PerformanceReport::empty());
        }

        let total_orders = recent_executions.len() as u64;
        let total_quantity: Decimal = recent_executions.iter().map(|r| r.filled_quantity).sum();
        let total_fees: Decimal = recent_executions.iter().map(|r| r.fees_paid).sum();
        
        let average_slippage = recent_executions.iter()
            .map(|r| r.slippage_bps)
            .sum::<f64>() / total_orders as f64;

        let average_execution_time = recent_executions.iter()
            .map(|r| (r.completion_time - r.start_time).num_seconds() as f64)
            .sum::<f64>() / total_orders as f64;

        let fill_rate = recent_executions.iter()
            .map(|r| (r.filled_quantity / r.total_quantity).to_f64().unwrap_or(0.0))
            .sum::<f64>() / total_orders as f64;

        let average_quality_score = recent_executions.iter()
            .map(|r| r.execution_quality_score)
            .sum::<f64>() / total_orders as f64;

        // Venue performance breakdown
        let mut venue_stats: HashMap<String, VenueStats> = HashMap::new();
        for record in &recent_executions {
            let stats = venue_stats.entry(record.venue.clone()).or_insert(VenueStats::default());
            stats.order_count += 1;
            stats.total_slippage += record.slippage_bps;
            stats.total_fees += record.fees_paid;
        }

        for stats in venue_stats.values_mut() {
            if stats.order_count > 0 {
                stats.average_slippage = stats.total_slippage / stats.order_count as f64;
            }
        }

        Ok(PerformanceReport {
            period_hours: hours,
            total_orders,
            total_quantity,
            total_fees,
            average_slippage_bps: average_slippage,
            average_execution_time_seconds: average_execution_time,
            fill_rate,
            average_quality_score,
            venue_performance: venue_stats,
            generated_at: Utc::now(),
        })
    }

    async fn check_trade_alerts(&mut self, trade: &Trade) -> Result<()> {
        // Implementation would check for various trade-level alerts
        // For now, this is a placeholder
        Ok(())
    }

    async fn check_completion_alerts(&mut self, record: &ExecutionRecord) -> Result<()> {
        let mut alerts = Vec::new();

        // Check slippage threshold
        if record.slippage_bps.abs() > self.config.alert_thresholds.max_slippage_bps {
            alerts.push(ExecutionAlert {
                timestamp: Utc::now(),
                alert_type: AlertType::HighSlippage,
                severity: if record.slippage_bps.abs() > self.config.alert_thresholds.max_slippage_bps * 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                order_id: Some(record.order_id),
                venue: Some(record.venue.clone()),
                message: format!("High slippage detected: {:.2} bps", record.slippage_bps),
                metrics: [("slippage_bps".to_string(), record.slippage_bps)].iter().cloned().collect(),
            });
        }

        // Check execution time threshold
        let execution_time_seconds = (record.completion_time - record.start_time).num_seconds() as u32;
        if execution_time_seconds > self.config.alert_thresholds.max_execution_time_seconds {
            alerts.push(ExecutionAlert {
                timestamp: Utc::now(),
                alert_type: AlertType::SlowExecution,
                severity: AlertSeverity::Warning,
                order_id: Some(record.order_id),
                venue: Some(record.venue.clone()),
                message: format!("Slow execution detected: {} seconds", execution_time_seconds),
                metrics: [("execution_time_seconds".to_string(), execution_time_seconds as f64)].iter().cloned().collect(),
            });
        }

        // Check fill rate
        let fill_rate = (record.filled_quantity / record.total_quantity).to_f64().unwrap_or(0.0);
        if fill_rate < self.config.alert_thresholds.min_fill_rate {
            alerts.push(ExecutionAlert {
                timestamp: Utc::now(),
                alert_type: AlertType::LowFillRate,
                severity: AlertSeverity::Warning,
                order_id: Some(record.order_id),
                venue: Some(record.venue.clone()),
                message: format!("Low fill rate detected: {:.2}%", fill_rate * 100.0),
                metrics: [("fill_rate".to_string(), fill_rate)].iter().cloned().collect(),
            });
        }

        // Add alerts to history
        for alert in alerts {
            self.alert_history.push_back(alert);
            
            // Maintain alert history size
            while self.alert_history.len() > self.config.max_alert_history {
                self.alert_history.pop_front();
            }
        }

        Ok(())
    }

    fn calculate_execution_quality_score(&self, slippage_bps: f64, execution_time_seconds: f64, filled_quantity: Decimal, total_quantity: Decimal) -> f64 {
        let fill_rate = (filled_quantity / total_quantity).to_f64().unwrap_or(0.0);
        
        // Normalize metrics to 0-1 scale
        let slippage_score = (1.0 - (slippage_bps.abs() / 100.0)).max(0.0);
        let time_score = (1.0 - (execution_time_seconds / 300.0)).max(0.0); // 5 minutes max
        let fill_score = fill_rate;

        // Weighted average
        (slippage_score * 0.4 + time_score * 0.3 + fill_score * 0.3).min(1.0)
    }

    async fn update_performance_metrics(&mut self, record: &ExecutionRecord) {
        self.metrics.total_orders_monitored += 1;
        
        let n = self.metrics.total_orders_monitored as f64;
        
        // Update moving averages
        self.metrics.average_execution_time_seconds = 
            (self.metrics.average_execution_time_seconds * (n - 1.0) + (record.completion_time - record.start_time).num_seconds() as f64) / n;
        
        self.metrics.average_slippage_bps = 
            (self.metrics.average_slippage_bps * (n - 1.0) + record.slippage_bps) / n;
        
        let fill_rate = (record.filled_quantity / record.total_quantity).to_f64().unwrap_or(0.0);
        self.metrics.fill_rate = 
            (self.metrics.fill_rate * (n - 1.0) + fill_rate) / n;
        
        self.metrics.execution_quality_score = 
            (self.metrics.execution_quality_score * (n - 1.0) + record.execution_quality_score) / n;

        // Update venue distribution
        let venue_count = self.metrics.venue_distribution.entry(record.venue.clone()).or_insert(0.0);
        *venue_count += 1.0;
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub period_hours: u32,
    pub total_orders: u64,
    pub total_quantity: Decimal,
    pub total_fees: Decimal,
    pub average_slippage_bps: f64,
    pub average_execution_time_seconds: f64,
    pub fill_rate: f64,
    pub average_quality_score: f64,
    pub venue_performance: HashMap<String, VenueStats>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct VenueStats {
    pub order_count: u64,
    pub total_slippage: f64,
    pub average_slippage: f64,
    pub total_fees: Decimal,
}

impl PerformanceReport {
    fn empty() -> Self {
        Self {
            period_hours: 0,
            total_orders: 0,
            total_quantity: Decimal::ZERO,
            total_fees: Decimal::ZERO,
            average_slippage_bps: 0.0,
            average_execution_time_seconds: 0.0,
            fill_rate: 0.0,
            average_quality_score: 0.0,
            venue_performance: HashMap::new(),
            generated_at: Utc::now(),
        }
    }
}