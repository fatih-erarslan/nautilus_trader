use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};

use crate::observability::{
    ObservabilityManager, NeuralMetrics, SystemHealthMetrics, TradingMetrics, 
    AnomalyAlert, AlertSeverity
};

/// Dashboard Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub refresh_interval_ms: u64,
    pub metrics_retention_hours: u32,
    pub alert_thresholds: AlertThresholds,
    pub visualization_settings: VisualizationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub neural_latency_ms: f64,
    pub neural_accuracy_min: f64,
    pub system_cpu_max: f64,
    pub system_memory_max: f64,
    pub trading_latency_ms: f64,
    pub risk_score_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSettings {
    pub chart_type: String,
    pub time_window_minutes: u32,
    pub aggregation_interval_seconds: u32,
    pub color_scheme: String,
}

/// Real-time Dashboard Widget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardWidget {
    pub id: String,
    pub widget_type: WidgetType,
    pub title: String,
    pub data: WidgetData,
    pub status: WidgetStatus,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    MetricGauge,
    TimeSeriesChart,
    AlertTable,
    SystemOverview,
    NetworkTopology,
    PerformanceHeatmap,
    TradingStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetData {
    Scalar(f64),
    TimeSeries(Vec<(u64, f64)>),
    Table(Vec<HashMap<String, String>>),
    Chart(ChartData),
    Status(StatusData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub labels: Vec<String>,
    pub datasets: Vec<Dataset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub label: String,
    pub data: Vec<f64>,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusData {
    pub status: String,
    pub color: String,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Operational Dashboard for Real-time Monitoring
pub struct OperationalDashboard {
    observability: Arc<ObservabilityManager>,
    config: DashboardConfig,
    widgets: Arc<RwLock<HashMap<String, DashboardWidget>>>,
    alert_history: Arc<RwLock<Vec<AnomalyAlert>>>,
}

impl OperationalDashboard {
    pub fn new(observability: Arc<ObservabilityManager>) -> Self {
        let config = DashboardConfig {
            refresh_interval_ms: 1000,
            metrics_retention_hours: 24,
            alert_thresholds: AlertThresholds {
                neural_latency_ms: 10.0,
                neural_accuracy_min: 95.0,
                system_cpu_max: 90.0,
                system_memory_max: 85.0,
                trading_latency_ms: 100.0,
                risk_score_max: 0.8,
            },
            visualization_settings: VisualizationSettings {
                chart_type: "line".to_string(),
                time_window_minutes: 60,
                aggregation_interval_seconds: 60,
                color_scheme: "dark".to_string(),
            },
        };
        
        Self {
            observability,
            config,
            widgets: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Initialize dashboard with default widgets
    #[instrument(skip(self))]
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing operational dashboard");
        
        let mut widgets = self.widgets.write().await;
        
        // Neural Network Performance Widget
        widgets.insert("neural_performance".to_string(), DashboardWidget {
            id: "neural_performance".to_string(),
            widget_type: WidgetType::MetricGauge,
            title: "Neural Network Performance".to_string(),
            data: WidgetData::Scalar(0.0),
            status: WidgetStatus::Unknown,
            last_updated: 0,
        });
        
        // System Health Widget
        widgets.insert("system_health".to_string(), DashboardWidget {
            id: "system_health".to_string(),
            widget_type: WidgetType::SystemOverview,
            title: "System Health Overview".to_string(),
            data: WidgetData::Status(StatusData {
                status: "Unknown".to_string(),
                color: "#666666".to_string(),
                details: HashMap::new(),
            }),
            status: WidgetStatus::Unknown,
            last_updated: 0,
        });
        
        // Trading Performance Widget
        widgets.insert("trading_performance".to_string(), DashboardWidget {
            id: "trading_performance".to_string(),
            widget_type: WidgetType::TradingStatus,
            title: "Trading System Status".to_string(),
            data: WidgetData::Status(StatusData {
                status: "Unknown".to_string(),
                color: "#666666".to_string(),
                details: HashMap::new(),
            }),
            status: WidgetStatus::Unknown,
            last_updated: 0,
        });
        
        // Latency Time Series Widget
        widgets.insert("latency_chart".to_string(), DashboardWidget {
            id: "latency_chart".to_string(),
            widget_type: WidgetType::TimeSeriesChart,
            title: "System Latency Trends".to_string(),
            data: WidgetData::TimeSeries(Vec::new()),
            status: WidgetStatus::Unknown,
            last_updated: 0,
        });
        
        // Active Alerts Widget
        widgets.insert("active_alerts".to_string(), DashboardWidget {
            id: "active_alerts".to_string(),
            widget_type: WidgetType::AlertTable,
            title: "Active Alerts".to_string(),
            data: WidgetData::Table(Vec::new()),
            status: WidgetStatus::Healthy,
            last_updated: 0,
        });
        
        info!("Dashboard initialized with {} widgets", widgets.len());
        Ok(())
    }
    
    /// Update all dashboard widgets with latest metrics
    #[instrument(skip(self))]
    pub async fn update_dashboard(&self) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;
        
        // Update neural performance widget
        if let Some(neural_metrics) = self.observability.get_neural_metrics_summary().await? {
            self.update_neural_performance_widget(&neural_metrics, timestamp).await?;
        }
        
        // Update system health widget
        if let Some(system_metrics) = self.observability.get_system_health_summary().await? {
            self.update_system_health_widget(&system_metrics, timestamp).await?;
        }
        
        // Update trading performance widget
        if let Some(trading_metrics) = self.observability.get_trading_metrics_summary().await? {
            self.update_trading_performance_widget(&trading_metrics, timestamp).await?;
        }
        
        // Update alerts widget
        let alerts = self.observability.get_active_alerts().await;
        self.update_alerts_widget(&alerts, timestamp).await?;
        
        Ok(())
    }
    
    /// Update neural performance widget
    async fn update_neural_performance_widget(
        &self, 
        metrics: &NeuralMetrics, 
        timestamp: u64
    ) -> Result<()> {
        let mut widgets = self.widgets.write().await;
        
        if let Some(widget) = widgets.get_mut("neural_performance") {
            let status = if metrics.accuracy_percentage >= self.config.alert_thresholds.neural_accuracy_min {
                WidgetStatus::Healthy
            } else if metrics.accuracy_percentage >= 90.0 {
                WidgetStatus::Warning
            } else {
                WidgetStatus::Critical
            };
            
            widget.data = WidgetData::Chart(ChartData {
                labels: vec![
                    "Latency (ms)".to_string(),
                    "Throughput (ops/s)".to_string(),
                    "Accuracy (%)".to_string(),
                    "GPU Util (%)".to_string(),
                ],
                datasets: vec![Dataset {
                    label: "Current".to_string(),
                    data: vec![
                        metrics.inference_latency_ns as f64 / 1_000_000.0,
                        metrics.throughput_ops_per_sec,
                        metrics.accuracy_percentage,
                        metrics.gpu_utilization_percent,
                    ],
                    color: match status {
                        WidgetStatus::Healthy => "#4CAF50".to_string(),
                        WidgetStatus::Warning => "#FF9800".to_string(),
                        WidgetStatus::Critical => "#F44336".to_string(),
                        _ => "#666666".to_string(),
                    },
                }],
            });
            
            widget.status = status;
            widget.last_updated = timestamp;
        }
        
        Ok(())
    }
    
    /// Update system health widget
    async fn update_system_health_widget(
        &self, 
        metrics: &SystemHealthMetrics, 
        timestamp: u64
    ) -> Result<()> {
        let mut widgets = self.widgets.write().await;
        
        if let Some(widget) = widgets.get_mut("system_health") {
            let status = if metrics.cpu_usage_percent < self.config.alert_thresholds.system_cpu_max 
                && metrics.memory_usage_percent < self.config.alert_thresholds.system_memory_max {
                WidgetStatus::Healthy
            } else if metrics.cpu_usage_percent < 95.0 && metrics.memory_usage_percent < 95.0 {
                WidgetStatus::Warning
            } else {
                WidgetStatus::Critical
            };
            
            let mut details = HashMap::new();
            details.insert("CPU Usage".to_string(), format!("{:.1}%", metrics.cpu_usage_percent));
            details.insert("Memory Usage".to_string(), format!("{:.1}%", metrics.memory_usage_percent));
            details.insert("Disk Usage".to_string(), format!("{:.1}%", metrics.disk_usage_percent));
            details.insert("Network RX".to_string(), format!("{:.1} Mbps", metrics.network_rx_mbps));
            details.insert("Network TX".to_string(), format!("{:.1} Mbps", metrics.network_tx_mbps));
            details.insert("Processes".to_string(), metrics.process_count.to_string());
            
            widget.data = WidgetData::Status(StatusData {
                status: match status {
                    WidgetStatus::Healthy => "Healthy".to_string(),
                    WidgetStatus::Warning => "Warning".to_string(),
                    WidgetStatus::Critical => "Critical".to_string(),
                    _ => "Unknown".to_string(),
                },
                color: match status {
                    WidgetStatus::Healthy => "#4CAF50".to_string(),
                    WidgetStatus::Warning => "#FF9800".to_string(),
                    WidgetStatus::Critical => "#F44336".to_string(),
                    _ => "#666666".to_string(),
                },
                details,
            });
            
            widget.status = status;
            widget.last_updated = timestamp;
        }
        
        Ok(())
    }
    
    /// Update trading performance widget
    async fn update_trading_performance_widget(
        &self, 
        metrics: &TradingMetrics, 
        timestamp: u64
    ) -> Result<()> {
        let mut widgets = self.widgets.write().await;
        
        if let Some(widget) = widgets.get_mut("trading_performance") {
            let status = if metrics.risk_score < self.config.alert_thresholds.risk_score_max 
                && metrics.order_fill_latency_ms < self.config.alert_thresholds.trading_latency_ms {
                WidgetStatus::Healthy
            } else if metrics.risk_score < 0.9 && metrics.order_fill_latency_ms < 200.0 {
                WidgetStatus::Warning
            } else {
                WidgetStatus::Critical
            };
            
            let mut details = HashMap::new();
            details.insert("Orders/sec".to_string(), format!("{:.1}", metrics.orders_per_second));
            details.insert("Fill Latency".to_string(), format!("{:.2}ms", metrics.order_fill_latency_ms));
            details.insert("Market Data Latency".to_string(), format!("{:.2}ms", metrics.market_data_latency_ms));
            details.insert("Risk Score".to_string(), format!("{:.3}", metrics.risk_score));
            details.insert("P&L".to_string(), format!("${:.2}", metrics.pnl_realtime));
            details.insert("Portfolio Value".to_string(), format!("${:.2}", metrics.portfolio_value));
            details.insert("Open Positions".to_string(), metrics.open_positions.to_string());
            details.insert("Alerts".to_string(), metrics.alerts_triggered.to_string());
            
            widget.data = WidgetData::Status(StatusData {
                status: match status {
                    WidgetStatus::Healthy => "Trading Active".to_string(),
                    WidgetStatus::Warning => "Risk Elevated".to_string(),
                    WidgetStatus::Critical => "Risk Critical".to_string(),
                    _ => "Unknown".to_string(),
                },
                color: match status {
                    WidgetStatus::Healthy => "#4CAF50".to_string(),
                    WidgetStatus::Warning => "#FF9800".to_string(),
                    WidgetStatus::Critical => "#F44336".to_string(),
                    _ => "#666666".to_string(),
                },
                details,
            });
            
            widget.status = status;
            widget.last_updated = timestamp;
        }
        
        Ok(())
    }
    
    /// Update alerts widget
    async fn update_alerts_widget(&self, alerts: &[AnomalyAlert], timestamp: u64) -> Result<()> {
        let mut widgets = self.widgets.write().await;
        
        if let Some(widget) = widgets.get_mut("active_alerts") {
            let mut table_data = Vec::new();
            
            for alert in alerts.iter().take(10) { // Show only latest 10 alerts
                let mut row = HashMap::new();
                row.insert("Severity".to_string(), format!("{:?}", alert.severity));
                row.insert("Component".to_string(), alert.component.clone());
                row.insert("Metric".to_string(), alert.metric_name.clone());
                row.insert("Value".to_string(), format!("{:.2}", alert.current_value));
                row.insert("Threshold".to_string(), format!("{:.2}", alert.threshold));
                row.insert("Description".to_string(), alert.description.clone());
                row.insert("Action".to_string(), alert.recommended_action.clone());
                
                let alert_time = std::time::UNIX_EPOCH + std::time::Duration::from_millis(alert.timestamp);
                row.insert("Time".to_string(), format!("{:?}", alert_time));
                
                table_data.push(row);
            }
            
            widget.data = WidgetData::Table(table_data);
            
            // Determine overall alert status
            let status = if alerts.is_empty() {
                WidgetStatus::Healthy
            } else if alerts.iter().any(|a| matches!(a.severity, AlertSeverity::Critical)) {
                WidgetStatus::Critical
            } else if alerts.iter().any(|a| matches!(a.severity, AlertSeverity::High)) {
                WidgetStatus::Warning
            } else {
                WidgetStatus::Healthy
            };
            
            widget.status = status;
            widget.last_updated = timestamp;
        }
        
        Ok(())
    }
    
    /// Generate dashboard summary for external consumption
    #[instrument(skip(self))]
    pub async fn generate_dashboard_summary(&self) -> Result<DashboardSummary> {
        let widgets = self.widgets.read().await;
        let alerts = self.observability.get_active_alerts().await;
        
        let overall_status = if alerts.iter().any(|a| matches!(a.severity, AlertSeverity::Critical)) {
            "CRITICAL"
        } else if alerts.iter().any(|a| matches!(a.severity, AlertSeverity::High)) {
            "WARNING"
        } else if alerts.is_empty() {
            "HEALTHY"
        } else {
            "INFO"
        };
        
        let widget_count = widgets.len();
        let alert_count = alerts.len();
        let critical_alerts = alerts.iter().filter(|a| matches!(a.severity, AlertSeverity::Critical)).count();
        
        Ok(DashboardSummary {
            overall_status: overall_status.to_string(),
            total_widgets: widget_count as u32,
            active_alerts: alert_count as u32,
            critical_alerts: critical_alerts as u32,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
            widgets: widgets.values().cloned().collect(),
        })
    }
    
    /// Export dashboard configuration
    pub fn export_config(&self) -> DashboardConfig {
        self.config.clone()
    }
    
    /// Get specific widget by ID
    pub async fn get_widget(&self, widget_id: &str) -> Option<DashboardWidget> {
        let widgets = self.widgets.read().await;
        widgets.get(widget_id).cloned()
    }
    
    /// Get all widgets
    pub async fn get_all_widgets(&self) -> HashMap<String, DashboardWidget> {
        self.widgets.read().await.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    pub overall_status: String,
    pub total_widgets: u32,
    pub active_alerts: u32,
    pub critical_alerts: u32,
    pub last_updated: u64,
    pub widgets: Vec<DashboardWidget>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::observability::ObservabilityManager;
    
    #[tokio::test]
    async fn test_dashboard_initialization() {
        let observability = Arc::new(ObservabilityManager::new().unwrap());
        let dashboard = OperationalDashboard::new(observability);
        
        let result = dashboard.initialize().await;
        assert!(result.is_ok());
        
        let widgets = dashboard.get_all_widgets().await;
        assert!(widgets.len() > 0);
        assert!(widgets.contains_key("neural_performance"));
        assert!(widgets.contains_key("system_health"));
    }
    
    #[tokio::test]
    async fn test_dashboard_summary_generation() {
        let observability = Arc::new(ObservabilityManager::new().unwrap());
        let dashboard = OperationalDashboard::new(observability);
        
        dashboard.initialize().await.unwrap();
        
        let summary = dashboard.generate_dashboard_summary().await.unwrap();
        assert_eq!(summary.overall_status, "HEALTHY");
        assert!(summary.total_widgets > 0);
    }
}