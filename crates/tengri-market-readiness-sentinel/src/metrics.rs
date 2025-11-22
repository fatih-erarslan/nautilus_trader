//! Metrics collection and reporting for market readiness validation

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::error::MarketReadinessError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub validation_metrics: ValidationMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub system_metrics: SystemMetrics,
    pub trading_metrics: TradingMetrics,
    pub risk_metrics: RiskMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub total_validations: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub warning_validations: u64,
    pub average_validation_time_ms: f64,
    pub validation_success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub disk_usage_percent: f64,
    pub network_latency_ms: f64,
    pub throughput_per_second: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub uptime_seconds: u64,
    pub active_connections: u64,
    pub thread_count: u64,
    pub load_average: f64,
    pub available_memory_mb: f64,
    pub disk_io_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub orders_per_second: f64,
    pub fill_rate: f64,
    pub average_latency_ms: f64,
    pub rejection_rate: f64,
    pub slippage_bps: f64,
    pub market_impact_bps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub current_var_95: f64,
    pub current_var_99: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub position_utilization: f64,
    pub leverage_ratio: f64,
}

#[derive(Debug)]
pub struct MetricsCollector {
    config: Arc<MarketReadinessConfig>,
    current_snapshot: Arc<RwLock<MetricsSnapshot>>,
    historical_snapshots: Arc<RwLock<Vec<MetricsSnapshot>>>,
    validation_stats: Arc<RwLock<ValidationStats>>,
    start_time: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct ValidationStats {
    total_count: u64,
    success_count: u64,
    failure_count: u64,
    warning_count: u64,
    total_duration_ms: u64,
}

impl MetricsCollector {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        let start_time = Utc::now();
        
        let initial_snapshot = MetricsSnapshot {
            timestamp: start_time,
            validation_metrics: ValidationMetrics {
                total_validations: 0,
                successful_validations: 0,
                failed_validations: 0,
                warning_validations: 0,
                average_validation_time_ms: 0.0,
                validation_success_rate: 1.0,
            },
            performance_metrics: PerformanceMetrics {
                cpu_usage_percent: 0.0,
                memory_usage_mb: 0.0,
                disk_usage_percent: 0.0,
                network_latency_ms: 0.0,
                throughput_per_second: 0.0,
                error_rate: 0.0,
            },
            system_metrics: SystemMetrics {
                uptime_seconds: 0,
                active_connections: 0,
                thread_count: 0,
                load_average: 0.0,
                available_memory_mb: 0.0,
                disk_io_rate: 0.0,
            },
            trading_metrics: TradingMetrics {
                orders_per_second: 0.0,
                fill_rate: 0.0,
                average_latency_ms: 0.0,
                rejection_rate: 0.0,
                slippage_bps: 0.0,
                market_impact_bps: 0.0,
            },
            risk_metrics: RiskMetrics {
                current_var_95: 0.0,
                current_var_99: 0.0,
                max_drawdown: 0.0,
                sharpe_ratio: 0.0,
                position_utilization: 0.0,
                leverage_ratio: 0.0,
            },
        };
        
        Ok(Self {
            config,
            current_snapshot: Arc::new(RwLock::new(initial_snapshot)),
            historical_snapshots: Arc::new(RwLock::new(Vec::new())),
            validation_stats: Arc::new(RwLock::new(ValidationStats {
                total_count: 0,
                success_count: 0,
                failure_count: 0,
                warning_count: 0,
                total_duration_ms: 0,
            })),
            start_time,
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing metrics collector...");
        
        // Start metrics collection background task
        self.start_metrics_collection().await?;
        
        info!("Metrics collector initialized successfully");
        Ok(())
    }

    pub async fn record_validation(&self, duration_ms: u64) -> Result<()> {
        let mut stats = self.validation_stats.write().await;
        stats.total_count += 1;
        stats.success_count += 1;
        stats.total_duration_ms += duration_ms;
        
        self.update_validation_metrics().await?;
        Ok(())
    }

    pub async fn get_current_snapshot(&self) -> MetricsSnapshot {
        self.current_snapshot.read().await.clone()
    }

    async fn start_metrics_collection(&self) -> Result<()> {
        let current_snapshot = self.current_snapshot.clone();
        let start_time = self.start_time;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Collect current metrics
                if let Ok(snapshot) = Self::collect_system_metrics(start_time).await {
                    // Update current snapshot
                    let mut current = current_snapshot.write().await;
                    current.timestamp = snapshot.timestamp;
                    current.performance_metrics = snapshot.performance_metrics;
                    current.system_metrics = snapshot.system_metrics;
                    current.trading_metrics = snapshot.trading_metrics;
                    current.risk_metrics = snapshot.risk_metrics;
                }
            }
        });
        
        Ok(())
    }

    async fn update_validation_metrics(&self) -> Result<()> {
        let stats = self.validation_stats.read().await;
        let mut snapshot = self.current_snapshot.write().await;
        
        snapshot.validation_metrics.total_validations = stats.total_count;
        snapshot.validation_metrics.successful_validations = stats.success_count;
        snapshot.validation_metrics.failed_validations = stats.failure_count;
        snapshot.validation_metrics.warning_validations = stats.warning_count;
        
        if stats.total_count > 0 {
            snapshot.validation_metrics.average_validation_time_ms = 
                stats.total_duration_ms as f64 / stats.total_count as f64;
            snapshot.validation_metrics.validation_success_rate = 
                stats.success_count as f64 / stats.total_count as f64;
        }
        
        Ok(())
    }

    async fn collect_system_metrics(start_time: DateTime<Utc>) -> Result<MetricsSnapshot> {
        let now = Utc::now();
        let uptime = now.signed_duration_since(start_time).num_seconds() as u64;
        
        // Collect system metrics (simplified - would use actual system APIs)
        let performance_metrics = PerformanceMetrics {
            cpu_usage_percent: 30.0,
            memory_usage_mb: 2048.0,
            disk_usage_percent: 45.0,
            network_latency_ms: 50.0,
            throughput_per_second: 1000.0,
            error_rate: 0.01,
        };
        
        let system_metrics = SystemMetrics {
            uptime_seconds: uptime,
            active_connections: 50,
            thread_count: 20,
            load_average: 1.5,
            available_memory_mb: 4096.0,
            disk_io_rate: 100.0,
        };
        
        let trading_metrics = TradingMetrics {
            orders_per_second: 50.0,
            fill_rate: 0.95,
            average_latency_ms: 25.0,
            rejection_rate: 0.02,
            slippage_bps: 2.5,
            market_impact_bps: 5.0,
        };
        
        let risk_metrics = RiskMetrics {
            current_var_95: 50000.0,
            current_var_99: 75000.0,
            max_drawdown: 0.05,
            sharpe_ratio: 1.8,
            position_utilization: 0.6,
            leverage_ratio: 2.0,
        };
        
        Ok(MetricsSnapshot {
            timestamp: now,
            validation_metrics: ValidationMetrics {
                total_validations: 0,
                successful_validations: 0,
                failed_validations: 0,
                warning_validations: 0,
                average_validation_time_ms: 0.0,
                validation_success_rate: 1.0,
            },
            performance_metrics,
            system_metrics,
            trading_metrics,
            risk_metrics,
        })
    }
}
