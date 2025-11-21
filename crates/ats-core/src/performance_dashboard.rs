//! Real-Time Performance Dashboard for Nanosecond Precision Monitoring
//!
//! This module provides real-time monitoring and visualization of nanosecond-precision
//! performance metrics with automated alerting and regression detection.

use crate::nanosecond_validator::{NanosecondValidator, ValidationResult};
use crate::error::AtsCoreError;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use std::thread;

/// Real-time performance dashboard
pub struct PerformanceDashboard {
    validator: NanosecondValidator,
    metrics_history: Arc<Mutex<HashMap<String, VecDeque<PerformanceMetric>>>>,
    alerts: Arc<Mutex<Vec<PerformanceAlert>>>,
    monitoring_active: Arc<std::sync::atomic::AtomicBool>,
    regression_threshold: f64,
}

/// Performance metric with timestamp
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub timestamp: SystemTime,
    pub latency_ns: u64,
    pub success_rate: f64,
    pub target_ns: u64,
    pub operation: String,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub timestamp: SystemTime,
    pub alert_type: AlertType,
    pub operation: String,
    pub message: String,
    pub severity: Severity,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    LatencyTargetMissed,
    PerformanceRegression,
    SuccessRateDropped,
    SystemOverload,
}

#[derive(Debug, Clone)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

impl PerformanceDashboard {
    /// Create a new performance dashboard
    pub fn new() -> Result<Self, AtsCoreError> {
        Ok(Self {
            validator: NanosecondValidator::new()?,
            metrics_history: Arc::new(Mutex::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(Vec::new())),
            monitoring_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            regression_threshold: 0.1, // 10% regression threshold
        })
    }
    
    /// Start real-time monitoring
    pub fn start_monitoring(&self, monitoring_interval: Duration) -> Result<(), AtsCoreError> {
        self.monitoring_active.store(true, std::sync::atomic::Ordering::Relaxed);
        
        let validator = NanosecondValidator::new()?;
        let metrics_history = Arc::clone(&self.metrics_history);
        let alerts = Arc::clone(&self.alerts);
        let monitoring_active = Arc::clone(&self.monitoring_active);
        let regression_threshold = self.regression_threshold;
        
        thread::spawn(move || {
            let mut last_display = Instant::now();
            
            while monitoring_active.load(std::sync::atomic::Ordering::Relaxed) {
                // Collect current performance metrics
                if let Err(e) = Self::collect_current_metrics(&validator, &metrics_history, &alerts, regression_threshold) {
                    eprintln!("Error collecting metrics: {:?}", e);
                }
                
                // Display dashboard every 5 seconds
                if last_display.elapsed() >= Duration::from_secs(5) {
                    Self::display_dashboard(&metrics_history, &alerts);
                    last_display = Instant::now();
                }
                
                thread::sleep(monitoring_interval);
            }
        });
        
        Ok(())
    }
    
    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Collect current performance metrics
    fn collect_current_metrics(
        validator: &NanosecondValidator,
        metrics_history: &Arc<Mutex<HashMap<String, VecDeque<PerformanceMetric>>>>,
        alerts: &Arc<Mutex<Vec<PerformanceAlert>>>,
        regression_threshold: f64,
    ) -> Result<(), AtsCoreError> {
        // Test trading decision performance
        let validation_result = validator.validate_custom(|| {
            let mut sum = 0.0;
            for i in 0..10 {
                sum += (i as f64).sin();
            }
        }, "trading_decision", 500, 0.99)?;
            
        let metric = PerformanceMetric {
            timestamp: SystemTime::now(),
            latency_ns: validation_result.median_ns,
            success_rate: validation_result.actual_success_rate,
            target_ns: validation_result.target_ns,
            operation: "trading_decision".to_string(),
        };
            
        // Store metric in history
        {
            let mut history = metrics_history.lock().unwrap();
            let operation_history = history.entry("trading_decision".to_string()).or_insert_with(VecDeque::new);
            operation_history.push_back(metric.clone());
            
            // Keep only last 100 metrics
            if operation_history.len() > 100 {
                operation_history.pop_front();
            }
        }
        
        // Check for alerts
        Self::check_for_alerts(&metric, &validation_result, metrics_history, alerts, regression_threshold);
        
        Ok(())
    }
    
    /// Check for performance alerts
    fn check_for_alerts(
        metric: &PerformanceMetric,
        validation_result: &ValidationResult,
        metrics_history: &Arc<Mutex<HashMap<String, VecDeque<PerformanceMetric>>>>,
        alerts: &Arc<Mutex<Vec<PerformanceAlert>>>,
        regression_threshold: f64,
    ) {
        let mut new_alerts = Vec::new();
        
        // Check latency target
        if !validation_result.passed {
            new_alerts.push(PerformanceAlert {
                timestamp: SystemTime::now(),
                alert_type: AlertType::LatencyTargetMissed,
                operation: metric.operation.clone(),
                message: format!(
                    "Latency target missed: {}ns > {}ns (success rate: {:.2}%)",
                    metric.latency_ns, metric.target_ns, metric.success_rate * 100.0
                ),
                severity: Severity::Critical,
            });
        }
        
        // Check for performance regression
        {
            let history = metrics_history.lock().unwrap();
            if let Some(operation_history) = history.get(&metric.operation) {
                if operation_history.len() >= 10 {
                    let recent_avg = operation_history.iter().rev().take(5)
                        .map(|m| m.latency_ns as f64)
                        .sum::<f64>() / 5.0;
                    
                    let baseline_avg = operation_history.iter().take(5)
                        .map(|m| m.latency_ns as f64)
                        .sum::<f64>() / 5.0;
                    
                    let regression = (recent_avg - baseline_avg) / baseline_avg;
                    
                    if regression > regression_threshold {
                        new_alerts.push(PerformanceAlert {
                            timestamp: SystemTime::now(),
                            alert_type: AlertType::PerformanceRegression,
                            operation: metric.operation.clone(),
                            message: format!(
                                "Performance regression detected: {:.1}% increase in latency",
                                regression * 100.0
                            ),
                            severity: Severity::Warning,
                        });
                    }
                }
            }
        }
        
        // Check success rate
        if metric.success_rate < 0.99 {
            new_alerts.push(PerformanceAlert {
                timestamp: SystemTime::now(),
                alert_type: AlertType::SuccessRateDropped,
                operation: metric.operation.clone(),
                message: format!(
                    "Success rate dropped: {:.2}% < 99%",
                    metric.success_rate * 100.0
                ),
                severity: Severity::Warning,
            });
        }
        
        // Add alerts
        if !new_alerts.is_empty() {
            let mut alerts_guard = alerts.lock().unwrap();
            alerts_guard.extend(new_alerts);
            
            // Keep only last 50 alerts
            if alerts_guard.len() > 50 {
                alerts_guard.truncate(50);
            }
        }
    }
    
    /// Display real-time dashboard
    fn display_dashboard(
        metrics_history: &Arc<Mutex<HashMap<String, VecDeque<PerformanceMetric>>>>,
        alerts: &Arc<Mutex<Vec<PerformanceAlert>>>,
    ) {
        println!("\x1B[2J\x1B[1;1H"); // Clear screen and move cursor to top
        
        println!("🚀 NANOSECOND PRECISION PERFORMANCE DASHBOARD");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Updated: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!();
        
        // Display current metrics
        {
            let history = metrics_history.lock().unwrap();
            
            println!("📊 CURRENT PERFORMANCE METRICS");
            println!("┌─────────────────┬──────────────┬──────────────┬──────────────┬────────────┐");
            println!("│ Operation       │ Latency (ns) │ Target (ns)  │ Success Rate │ Status     │");
            println!("├─────────────────┼──────────────┼──────────────┼──────────────┼────────────┤");
            
            for (operation, metrics) in history.iter() {
                if let Some(latest) = metrics.back() {
                    let status = if latest.latency_ns <= latest.target_ns && latest.success_rate >= 0.99 {
                        "✅ PASS"
                    } else {
                        "❌ FAIL"
                    };
                    
                    println!("│ {:<15} │ {:>12} │ {:>12} │ {:>11.2}% │ {:<10} │",
                             operation,
                             latest.latency_ns,
                             latest.target_ns,
                             latest.success_rate * 100.0,
                             status);
                }
            }
            
            println!("└─────────────────┴──────────────┴──────────────┴──────────────┴────────────┘");
            println!();
            
            // Display latency trends
            println!("📈 LATENCY TRENDS (Last 10 measurements)");
            for (operation, metrics) in history.iter() {
                if metrics.len() >= 2 {
                    let recent: Vec<u64> = metrics.iter().rev().take(10).map(|m| m.latency_ns).collect();
                    let trend = Self::calculate_trend(&recent);
                    
                    let trend_indicator = if trend > 5.0 {
                        "📈 RISING"
                    } else if trend < -5.0 {
                        "📉 FALLING"
                    } else {
                        "📊 STABLE"
                    };
                    
                    println!("  {}: {} ({:+.1}%)", operation, trend_indicator, trend);
                }
            }
        }
        
        println!();
        
        // Display recent alerts
        {
            let alerts_guard = alerts.lock().unwrap();
            let recent_alerts: Vec<_> = alerts_guard.iter().rev().take(5).collect();
            
            if !recent_alerts.is_empty() {
                println!("🚨 RECENT ALERTS");
                println!("┌─────────────────┬─────────────┬──────────────────────────────────────────────────┐");
                println!("│ Time            │ Severity    │ Message                                          │");
                println!("├─────────────────┼─────────────┼──────────────────────────────────────────────────┤");
                
                for alert in recent_alerts {
                    let time_str = alert.timestamp
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| {
                            let datetime = chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                                .unwrap_or_default();
                            datetime.format("%H:%M:%S").to_string()
                        })
                        .unwrap_or_else(|_| "Unknown".to_string());
                    
                    let severity_str = match alert.severity {
                        Severity::Info => "ℹ️  INFO",
                        Severity::Warning => "⚠️  WARN",
                        Severity::Critical => "🚨 CRIT",
                    };
                    
                    let message = if alert.message.len() > 48 {
                        format!("{}...", &alert.message[..45])
                    } else {
                        alert.message.clone()
                    };
                    
                    println!("│ {:<15} │ {:<11} │ {:<48} │", time_str, severity_str, message);
                }
                
                println!("└─────────────────┴─────────────┴──────────────────────────────────────────────────┘");
            } else {
                println!("✅ NO RECENT ALERTS - All systems performing within targets");
            }
        }
        
        println!();
        println!("🎯 Performance Targets: Trading <500ns, Whale <200ns, GPU <100ns, API <50ns");
        println!("⏱️  Next update in 5 seconds... (Ctrl+C to stop)");
    }
    
    /// Calculate trend percentage
    fn calculate_trend(values: &[u64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let first_half: f64 = values.iter().take(values.len() / 2).map(|&v| v as f64).sum::<f64>() / (values.len() / 2) as f64;
        let second_half: f64 = values.iter().skip(values.len() / 2).map(|&v| v as f64).sum::<f64>() / (values.len() - values.len() / 2) as f64;
        
        if first_half > 0.0 {
            ((second_half - first_half) / first_half) * 100.0
        } else {
            0.0
        }
    }
    
    /// Generate performance report
    pub fn generate_performance_report(&self) -> Result<PerformanceReport, AtsCoreError> {
        let history = self.metrics_history.lock().unwrap();
        let alerts = self.alerts.lock().unwrap();
        
        let mut report = PerformanceReport {
            generated_at: SystemTime::now(),
            operations: HashMap::new(),
            total_alerts: alerts.len(),
            critical_alerts: alerts.iter().filter(|a| matches!(a.severity, Severity::Critical)).count(),
            warning_alerts: alerts.iter().filter(|a| matches!(a.severity, Severity::Warning)).count(),
        };
        
        for (operation, metrics) in history.iter() {
            if let Some(latest) = metrics.back() {
                let latencies: Vec<u64> = metrics.iter().map(|m| m.latency_ns).collect();
                let success_rates: Vec<f64> = metrics.iter().map(|m| m.success_rate).collect();
                
                report.operations.insert(operation.clone(), OperationReport {
                    current_latency_ns: latest.latency_ns,
                    target_latency_ns: latest.target_ns,
                    current_success_rate: latest.success_rate,
                    avg_latency_ns: latencies.iter().sum::<u64>() / latencies.len() as u64,
                    min_latency_ns: *latencies.iter().min().unwrap_or(&0),
                    max_latency_ns: *latencies.iter().max().unwrap_or(&0),
                    avg_success_rate: success_rates.iter().sum::<f64>() / success_rates.len() as f64,
                    measurements_count: metrics.len(),
                    target_met: latest.latency_ns <= latest.target_ns && latest.success_rate >= 0.99,
                });
            }
        }
        
        Ok(report)
    }
}

/// Performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub generated_at: SystemTime,
    pub operations: HashMap<String, OperationReport>,
    pub total_alerts: usize,
    pub critical_alerts: usize,
    pub warning_alerts: usize,
}

#[derive(Debug, Clone)]
pub struct OperationReport {
    pub current_latency_ns: u64,
    pub target_latency_ns: u64,
    pub current_success_rate: f64,
    pub avg_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub avg_success_rate: f64,
    pub measurements_count: usize,
    pub target_met: bool,
}

impl PerformanceReport {
    /// Check if all operations meet targets
    pub fn all_targets_met(&self) -> bool {
        self.operations.values().all(|op| op.target_met)
    }
    
    /// Display comprehensive report
    pub fn display_report(&self) {
        println!("📊 COMPREHENSIVE PERFORMANCE REPORT");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Generated: {:?}", self.generated_at);
        println!("Overall Status: {}", if self.all_targets_met() { "✅ ALL TARGETS MET" } else { "❌ SOME TARGETS MISSED" });
        println!("Alerts: {} total ({} critical, {} warning)", self.total_alerts, self.critical_alerts, self.warning_alerts);
        println!();
        
        for (operation, report) in &self.operations {
            println!("🎯 {} Performance:", operation.to_uppercase());
            println!("  Current Latency: {}ns (target: {}ns) {}", 
                     report.current_latency_ns, 
                     report.target_latency_ns,
                     if report.current_latency_ns <= report.target_latency_ns { "✅" } else { "❌" });
            println!("  Success Rate: {:.2}% {}", 
                     report.current_success_rate * 100.0,
                     if report.current_success_rate >= 0.99 { "✅" } else { "❌" });
            println!("  Latency Range: {}ns - {}ns (avg: {}ns)", 
                     report.min_latency_ns, report.max_latency_ns, report.avg_latency_ns);
            println!("  Measurements: {}", report.measurements_count);
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_dashboard_creation() {
        let dashboard = PerformanceDashboard::new();
        assert!(dashboard.is_ok());
    }
    
    #[test]
    fn test_performance_report_generation() {
        let dashboard = PerformanceDashboard::new().unwrap();
        let report = dashboard.generate_performance_report().unwrap();
        
        // Report should be generated successfully
        assert!(report.generated_at <= SystemTime::now());
    }
    
    #[test]
    fn test_trend_calculation() {
        let values = vec![100, 110, 120, 130, 140];
        let trend = PerformanceDashboard::calculate_trend(&values);
        
        // Should show positive trend
        assert!(trend > 0.0);
    }
}