//! Comprehensive metrics and monitoring for TENGRI compliance engine

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge, 
    register_counter, register_gauge, register_histogram, register_int_counter, register_int_gauge,
    Encoder, TextEncoder, Registry,
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use crate::error::ComplianceResult;

/// Comprehensive metrics for compliance monitoring
pub struct ComplianceMetrics {
    // Trade metrics
    trades_processed: IntCounter,
    trades_approved: IntCounter,
    trades_rejected: IntCounter,
    trade_processing_time: Histogram,
    
    // Rule metrics
    rules_evaluated: IntCounter,
    rule_violations: IntCounter,
    rule_evaluation_time: Histogram,
    
    // Circuit breaker metrics
    circuit_breakers_triggered: IntCounter,
    kill_switch_activations: IntCounter,
    
    // Risk metrics
    current_portfolio_risk: Gauge,
    daily_pnl: Gauge,
    position_concentration: Gauge,
    leverage_ratio: Gauge,
    
    // Surveillance metrics
    suspicious_patterns_detected: IntCounter,
    wash_trading_alerts: IntCounter,
    spoofing_alerts: IntCounter,
    volume_anomalies: IntCounter,
    
    // Performance metrics
    compliance_engine_uptime: Gauge,
    memory_usage_bytes: Gauge,
    audit_records_count: IntGauge,
    
    // Error metrics
    compliance_errors: IntCounter,
    system_errors: IntCounter,
    
    // Custom registry for isolation
    registry: Registry,
    
    // Real-time statistics
    real_time_stats: Arc<RwLock<RealTimeStats>>,
    
    // Per-trader metrics
    trader_metrics: Arc<DashMap<String, TraderMetrics>>,
    
    // Per-symbol metrics
    symbol_metrics: Arc<DashMap<String, SymbolMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeStats {
    pub trades_per_second: f64,
    pub rejection_rate: f64,
    pub average_processing_time_ms: f64,
    pub active_circuit_breakers: u32,
    pub high_risk_traders: u32,
    pub total_portfolio_value: f64,
    pub system_health_score: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraderMetrics {
    pub trader_id: String,
    pub trades_today: u32,
    pub rejected_trades: u32,
    pub total_volume: f64,
    pub pnl_today: f64,
    pub risk_score: f64,
    pub violations_count: u32,
    pub last_trade_time: Option<DateTime<Utc>>,
    pub status: TraderStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraderStatus {
    Active,
    Restricted,
    Suspended,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMetrics {
    pub symbol: String,
    pub trade_count: u32,
    pub total_volume: f64,
    pub price_volatility: f64,
    pub surveillance_alerts: u32,
    pub last_trade_price: Option<f64>,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl ComplianceMetrics {
    pub fn new() -> ComplianceResult<Self> {
        let registry = Registry::new();
        
        let trades_processed = register_int_counter!(
            "tengri_trades_processed_total",
            "Total number of trades processed by TENGRI"
        )?;
        
        let trades_approved = register_int_counter!(
            "tengri_trades_approved_total", 
            "Total number of trades approved"
        )?;
        
        let trades_rejected = register_int_counter!(
            "tengri_trades_rejected_total",
            "Total number of trades rejected"
        )?;
        
        let trade_processing_time = register_histogram!(
            "tengri_trade_processing_duration_seconds",
            "Time taken to process trades",
            vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )?;
        
        let rules_evaluated = register_int_counter!(
            "tengri_rules_evaluated_total",
            "Total number of compliance rules evaluated"
        )?;
        
        let rule_violations = register_int_counter!(
            "tengri_rule_violations_total",
            "Total number of rule violations detected"
        )?;
        
        let rule_evaluation_time = register_histogram!(
            "tengri_rule_evaluation_duration_seconds",
            "Time taken to evaluate compliance rules",
            vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
        )?;
        
        let circuit_breakers_triggered = register_int_counter!(
            "tengri_circuit_breakers_triggered_total",
            "Total number of circuit breaker activations"
        )?;
        
        let kill_switch_activations = register_int_counter!(
            "tengri_kill_switch_activations_total",
            "Total number of kill switch activations"
        )?;
        
        let current_portfolio_risk = register_gauge!(
            "tengri_portfolio_risk_score",
            "Current portfolio risk score (0-1)"
        )?;
        
        let daily_pnl = register_gauge!(
            "tengri_daily_pnl_usd",
            "Current daily profit and loss in USD"
        )?;
        
        let position_concentration = register_gauge!(
            "tengri_position_concentration_max",
            "Maximum position concentration percentage"
        )?;
        
        let leverage_ratio = register_gauge!(
            "tengri_leverage_ratio",
            "Current portfolio leverage ratio"
        )?;
        
        let suspicious_patterns_detected = register_int_counter!(
            "tengri_suspicious_patterns_total",
            "Total number of suspicious patterns detected"
        )?;
        
        let wash_trading_alerts = register_int_counter!(
            "tengri_wash_trading_alerts_total",
            "Total number of wash trading alerts"
        )?;
        
        let spoofing_alerts = register_int_counter!(
            "tengri_spoofing_alerts_total",
            "Total number of spoofing alerts"
        )?;
        
        let volume_anomalies = register_int_counter!(
            "tengri_volume_anomalies_total",
            "Total number of volume anomalies detected"
        )?;
        
        let compliance_engine_uptime = register_gauge!(
            "tengri_uptime_seconds",
            "Compliance engine uptime in seconds"
        )?;
        
        let memory_usage_bytes = register_gauge!(
            "tengri_memory_usage_bytes",
            "Memory usage of compliance engine"
        )?;
        
        let audit_records_count = register_int_gauge!(
            "tengri_audit_records_count",
            "Number of audit records stored"
        )?;
        
        let compliance_errors = register_int_counter!(
            "tengri_compliance_errors_total",
            "Total number of compliance errors"
        )?;
        
        let system_errors = register_int_counter!(
            "tengri_system_errors_total",
            "Total number of system errors"
        )?;
        
        Ok(Self {
            trades_processed,
            trades_approved,
            trades_rejected,
            trade_processing_time,
            rules_evaluated,
            rule_violations,
            rule_evaluation_time,
            circuit_breakers_triggered,
            kill_switch_activations,
            current_portfolio_risk,
            daily_pnl,
            position_concentration,
            leverage_ratio,
            suspicious_patterns_detected,
            wash_trading_alerts,
            spoofing_alerts,
            volume_anomalies,
            compliance_engine_uptime,
            memory_usage_bytes,
            audit_records_count,
            compliance_errors,
            system_errors,
            registry,
            real_time_stats: Arc::new(RwLock::new(RealTimeStats::default())),
            trader_metrics: Arc::new(DashMap::new()),
            symbol_metrics: Arc::new(DashMap::new()),
        })
    }

    // Trade metrics
    pub fn record_trade_processed(&self) {
        self.trades_processed.inc();
    }

    pub fn record_trade_approved(&self) {
        self.trades_approved.inc();
    }

    pub fn record_trade_rejected(&self) {
        self.trades_rejected.inc();
    }

    pub fn record_trade_processing_time(&self, duration_seconds: f64) {
        self.trade_processing_time.observe(duration_seconds);
    }

    // Rule metrics
    pub fn record_rules_evaluated(&self, count: u64) {
        self.rules_evaluated.inc_by(count);
    }

    pub fn record_rule_violation(&self) {
        self.rule_violations.inc();
    }

    pub fn record_rule_evaluation_time(&self, duration_seconds: f64) {
        self.rule_evaluation_time.observe(duration_seconds);
    }

    // Circuit breaker metrics
    pub fn record_circuit_breaker_triggered(&self) {
        self.circuit_breakers_triggered.inc();
    }

    pub fn record_kill_switch_activation(&self) {
        self.kill_switch_activations.inc();
    }

    // Risk metrics
    pub fn update_portfolio_risk(&self, risk_score: f64) {
        self.current_portfolio_risk.set(risk_score);
    }

    pub fn update_daily_pnl(&self, pnl: f64) {
        self.daily_pnl.set(pnl);
    }

    pub fn update_position_concentration(&self, concentration: f64) {
        self.position_concentration.set(concentration);
    }

    pub fn update_leverage_ratio(&self, leverage: f64) {
        self.leverage_ratio.set(leverage);
    }

    // Surveillance metrics
    pub fn record_suspicious_pattern(&self, pattern_type: &str) {
        self.suspicious_patterns_detected.inc();
        
        match pattern_type {
            "WashTrading" => self.wash_trading_alerts.inc(),
            "Spoofing" => self.spoofing_alerts.inc(),
            "UnusualVolume" => self.volume_anomalies.inc(),
            _ => {}
        }
    }

    // System metrics
    pub fn update_uptime(&self, uptime_seconds: f64) {
        self.compliance_engine_uptime.set(uptime_seconds);
    }

    pub fn update_memory_usage(&self, bytes: f64) {
        self.memory_usage_bytes.set(bytes);
    }

    pub fn update_audit_records_count(&self, count: i64) {
        self.audit_records_count.set(count);
    }

    // Error metrics
    pub fn record_compliance_error(&self) {
        self.compliance_errors.inc();
    }

    pub fn record_system_error(&self) {
        self.system_errors.inc();
    }

    // Trader-specific metrics
    pub fn update_trader_metrics(&self, trader_id: String, metrics: TraderMetrics) {
        self.trader_metrics.insert(trader_id, metrics);
    }

    pub fn get_trader_metrics(&self, trader_id: &str) -> Option<TraderMetrics> {
        self.trader_metrics.get(trader_id).map(|m| m.clone())
    }

    // Symbol-specific metrics
    pub fn update_symbol_metrics(&self, symbol: String, metrics: SymbolMetrics) {
        self.symbol_metrics.insert(symbol, metrics);
    }

    pub fn get_symbol_metrics(&self, symbol: &str) -> Option<SymbolMetrics> {
        self.symbol_metrics.get(symbol).map(|m| m.clone())
    }

    // Real-time statistics
    pub fn update_real_time_stats(&self, stats: RealTimeStats) {
        *self.real_time_stats.write() = stats;
    }

    pub fn get_real_time_stats(&self) -> RealTimeStats {
        self.real_time_stats.read().clone()
    }

    // Calculate derived metrics
    pub fn calculate_rejection_rate(&self) -> f64 {
        let approved = self.trades_approved.get() as f64;
        let rejected = self.trades_rejected.get() as f64;
        let total = approved + rejected;
        
        if total > 0.0 {
            (rejected / total) * 100.0
        } else {
            0.0
        }
    }

    pub fn calculate_system_health_score(&self) -> f64 {
        let rejection_rate = self.calculate_rejection_rate();
        let error_rate = self.compliance_errors.get() as f64 / (self.trades_processed.get() as f64).max(1.0);
        
        // Simple health score calculation (would be more sophisticated in production)
        let health = 100.0 - (rejection_rate * 0.5) - (error_rate * 100.0 * 2.0);
        health.max(0.0).min(100.0)
    }

    // Export metrics in Prometheus format
    pub fn export_metrics(&self) -> ComplianceResult<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    // Get comprehensive dashboard data
    pub fn get_dashboard_data(&self) -> DashboardData {
        let real_time_stats = self.get_real_time_stats();
        
        let top_traders: Vec<TraderMetrics> = self.trader_metrics
            .iter()
            .map(|entry| entry.value().clone())
            .collect::<Vec<_>>()
            .into_iter()
            .take(10)
            .collect();
        
        let high_risk_symbols: Vec<SymbolMetrics> = self.symbol_metrics
            .iter()
            .filter(|entry| matches!(entry.value().risk_level, RiskLevel::High | RiskLevel::Critical))
            .map(|entry| entry.value().clone())
            .collect();
        
        DashboardData {
            real_time_stats,
            total_trades_processed: self.trades_processed.get(),
            total_trades_approved: self.trades_approved.get(),
            total_trades_rejected: self.trades_rejected.get(),
            total_rule_violations: self.rule_violations.get(),
            circuit_breakers_active: self.circuit_breakers_triggered.get(),
            suspicious_patterns_count: self.suspicious_patterns_detected.get(),
            system_health_score: self.calculate_system_health_score(),
            rejection_rate: self.calculate_rejection_rate(),
            top_traders,
            high_risk_symbols,
            last_updated: Utc::now(),
        }
    }
}

impl Default for RealTimeStats {
    fn default() -> Self {
        Self {
            trades_per_second: 0.0,
            rejection_rate: 0.0,
            average_processing_time_ms: 0.0,
            active_circuit_breakers: 0,
            high_risk_traders: 0,
            total_portfolio_value: 0.0,
            system_health_score: 100.0,
            last_updated: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DashboardData {
    pub real_time_stats: RealTimeStats,
    pub total_trades_processed: u64,
    pub total_trades_approved: u64,
    pub total_trades_rejected: u64,
    pub total_rule_violations: u64,
    pub circuit_breakers_active: u64,
    pub suspicious_patterns_count: u64,
    pub system_health_score: f64,
    pub rejection_rate: f64,
    pub top_traders: Vec<TraderMetrics>,
    pub high_risk_symbols: Vec<SymbolMetrics>,
    pub last_updated: DateTime<Utc>,
}

/// Performance tracker for detailed analysis
pub struct PerformanceTracker {
    metrics: Arc<ComplianceMetrics>,
    start_time: std::time::Instant,
}

impl PerformanceTracker {
    pub fn new(metrics: Arc<ComplianceMetrics>) -> Self {
        Self {
            metrics,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn track_trade_processing<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64();
        self.metrics.record_trade_processing_time(duration);
        result
    }

    pub fn track_rule_evaluation<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64();
        self.metrics.record_rule_evaluation_time(duration);
        result
    }

    pub fn get_uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}