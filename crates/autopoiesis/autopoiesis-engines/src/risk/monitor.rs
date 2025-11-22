//! Risk monitoring implementation

use crate::prelude::*;
use crate::models::{Position, MarketData};
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::{HashMap, VecDeque};

/// Risk monitor for continuous risk surveillance
#[derive(Debug, Clone)]
pub struct RiskMonitor {
    /// Monitor configuration
    config: RiskMonitorConfig,
    
    /// Risk alerts buffer
    alerts: VecDeque<RiskAlert>,
    
    /// Monitoring metrics
    metrics: MonitoringMetrics,
    
    /// Risk snapshots for trend analysis
    risk_snapshots: VecDeque<RiskSnapshot>,
}

#[derive(Debug, Clone)]
pub struct RiskMonitorConfig {
    /// Monitoring frequency in seconds
    pub monitoring_frequency_seconds: u32,
    
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    
    /// Maximum alerts to retain
    pub max_alerts: usize,
    
    /// Risk snapshot frequency
    pub snapshot_frequency_minutes: u32,
    
    /// Maximum snapshots to retain
    pub max_snapshots: usize,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// VaR threshold for alerts
    pub var_threshold: f64,
    
    /// Drawdown threshold for alerts
    pub drawdown_threshold: f64,
    
    /// Leverage threshold for alerts
    pub leverage_threshold: f64,
    
    /// Concentration threshold for alerts
    pub concentration_threshold: f64,
    
    /// Volatility threshold for alerts
    pub volatility_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct RiskAlert {
    pub timestamp: DateTime<Utc>,
    pub alert_type: RiskAlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub affected_positions: Vec<String>,
    pub current_value: f64,
    pub threshold_value: f64,
    pub recommendation: String,
}

#[derive(Debug, Clone)]
pub enum RiskAlertType {
    HighVaR,
    ExcessiveDrawdown,
    HighLeverage,
    Concentration,
    HighVolatility,
    CorrelationSpike,
    LiquidityRisk,
    MarketStress,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Default)]
struct MonitoringMetrics {
    total_alerts_generated: u64,
    critical_alerts_count: u64,
    average_response_time_ms: f64,
    last_monitoring_cycle: Option<DateTime<Utc>>,
    monitoring_uptime_pct: f64,
}

#[derive(Debug, Clone)]
struct RiskSnapshot {
    timestamp: DateTime<Utc>,
    portfolio_value: Decimal,
    var_95: f64,
    current_drawdown: f64,
    leverage_ratio: f64,
    largest_position_pct: f64,
    portfolio_volatility: f64,
    correlation_risk: f64,
}

#[derive(Debug, Clone)]
pub struct RiskReport {
    pub timestamp: DateTime<Utc>,
    pub current_risk_level: RiskLevel,
    pub portfolio_summary: PortfolioSummary,
    pub risk_metrics: RiskMetricsSummary,
    pub recent_alerts: Vec<RiskAlert>,
    pub recommendations: Vec<String>,
    pub trend_analysis: TrendAnalysis,
}

#[derive(Debug, Clone)]
pub enum RiskLevel {
    Low,
    Moderate,
    High,
    Extreme,
}

#[derive(Debug, Clone)]
pub struct PortfolioSummary {
    pub total_value: Decimal,
    pub position_count: usize,
    pub largest_position_pct: f64,
    pub sector_concentration: HashMap<String, f64>,
    pub geographic_concentration: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct RiskMetricsSummary {
    pub var_95_1day: f64,
    pub var_99_1day: f64,
    pub expected_shortfall: f64,
    pub current_drawdown: f64,
    pub max_drawdown_30d: f64,
    pub portfolio_volatility: f64,
    pub sharpe_ratio: f64,
    pub leverage_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub risk_trend: TrendDirection,
    pub volatility_trend: TrendDirection,
    pub correlation_trend: TrendDirection,
    pub concentration_trend: TrendDirection,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
}

impl Default for RiskMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_frequency_seconds: 30,
            alert_thresholds: AlertThresholds {
                var_threshold: 0.02,
                drawdown_threshold: 0.10,
                leverage_threshold: 2.5,
                concentration_threshold: 0.25,
                volatility_threshold: 0.30,
            },
            max_alerts: 1000,
            snapshot_frequency_minutes: 15,
            max_snapshots: 2880, // 30 days of 15-minute snapshots
        }
    }
}

impl RiskMonitor {
    /// Create a new risk monitor
    pub fn new(config: RiskMonitorConfig) -> Self {
        Self {
            config,
            alerts: VecDeque::new(),
            metrics: MonitoringMetrics::default(),
            risk_snapshots: VecDeque::new(),
        }
    }

    /// Perform risk monitoring cycle
    pub async fn monitor_risks(
        &mut self,
        positions: &[Position],
        market_data: &[MarketData],
    ) -> Result<Vec<RiskAlert>> {
        let monitoring_start = std::time::Instant::now();
        let mut new_alerts = Vec::new();

        // Take risk snapshot
        let snapshot = self.create_risk_snapshot(positions, market_data).await?;
        self.add_risk_snapshot(snapshot.clone());

        // Check VaR limits
        if let Some(alert) = self.check_var_limits(&snapshot).await? {
            new_alerts.push(alert);
        }

        // Check drawdown limits
        if let Some(alert) = self.check_drawdown_limits(&snapshot).await? {
            new_alerts.push(alert);
        }

        // Check leverage limits
        if let Some(alert) = self.check_leverage_limits(&snapshot).await? {
            new_alerts.push(alert);
        }

        // Check concentration limits
        if let Some(alert) = self.check_concentration_limits(positions).await? {
            new_alerts.push(alert);
        }

        // Check volatility limits
        if let Some(alert) = self.check_volatility_limits(&snapshot).await? {
            new_alerts.push(alert);
        }

        // Check correlation risks
        if let Some(alert) = self.check_correlation_risks(positions).await? {
            new_alerts.push(alert);
        }

        // Add new alerts to buffer
        for alert in &new_alerts {
            self.add_alert(alert.clone());
        }

        // Update metrics
        let monitoring_time = monitoring_start.elapsed().as_millis() as f64;
        self.update_monitoring_metrics(monitoring_time).await;

        Ok(new_alerts)
    }

    /// Generate comprehensive risk report
    pub async fn generate_risk_report(
        &self,
        positions: &[Position],
        market_data: &[MarketData],
    ) -> Result<RiskReport> {
        let current_snapshot = self.create_risk_snapshot(positions, market_data).await?;
        
        // Determine current risk level
        let risk_level = self.assess_risk_level(&current_snapshot);

        // Create portfolio summary
        let portfolio_summary = self.create_portfolio_summary(positions).await?;

        // Create risk metrics summary
        let risk_metrics = RiskMetricsSummary {
            var_95_1day: current_snapshot.var_95,
            var_99_1day: current_snapshot.var_95 * 1.3, // Approximation
            expected_shortfall: current_snapshot.var_95 * 1.5,
            current_drawdown: current_snapshot.current_drawdown,
            max_drawdown_30d: self.calculate_max_drawdown_30d(),
            portfolio_volatility: current_snapshot.portfolio_volatility,
            sharpe_ratio: 0.8, // Would calculate from returns
            leverage_ratio: current_snapshot.leverage_ratio,
        };

        // Get recent alerts
        let recent_alerts: Vec<RiskAlert> = self.alerts.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        // Generate recommendations
        let recommendations = self.generate_recommendations(&current_snapshot);

        // Perform trend analysis
        let trend_analysis = self.analyze_trends();

        Ok(RiskReport {
            timestamp: Utc::now(),
            current_risk_level: risk_level,
            portfolio_summary,
            risk_metrics,
            recent_alerts,
            recommendations,
            trend_analysis,
        })
    }

    /// Get active alerts
    pub fn get_active_alerts(&self, severity_filter: Option<AlertSeverity>) -> Vec<&RiskAlert> {
        let cutoff = Utc::now() - Duration::hours(24);
        
        self.alerts.iter()
            .filter(|alert| alert.timestamp > cutoff)
            .filter(|alert| {
                if let Some(ref filter) = severity_filter {
                    matches!((&alert.severity, filter), 
                        (AlertSeverity::Critical, AlertSeverity::Critical) |
                        (AlertSeverity::Emergency, AlertSeverity::Emergency) |
                        (AlertSeverity::Warning, AlertSeverity::Warning) |
                        (AlertSeverity::Info, AlertSeverity::Info))
                } else {
                    true
                }
            })
            .collect()
    }

    /// Clear old alerts and snapshots
    pub async fn cleanup(&mut self) {
        // Remove old alerts
        let alert_cutoff = Utc::now() - Duration::days(7);
        self.alerts.retain(|alert| alert.timestamp > alert_cutoff);

        // Remove old snapshots
        let snapshot_cutoff = Utc::now() - Duration::days(30);
        self.risk_snapshots.retain(|snapshot| snapshot.timestamp > snapshot_cutoff);
    }

    async fn create_risk_snapshot(&self, positions: &[Position], _market_data: &[MarketData]) -> Result<RiskSnapshot> {
        let portfolio_value: Decimal = positions.iter().map(|p| p.quantity * p.mark_price).sum();
        
        // Calculate VaR (simplified)
        let var_95 = self.calculate_portfolio_var(positions);
        
        // Calculate current drawdown
        let current_drawdown = self.calculate_current_drawdown(positions);
        
        // Calculate leverage
        let leverage_ratio = self.calculate_leverage_ratio(positions);
        
        // Find largest position percentage
        let largest_position_pct = if portfolio_value > Decimal::ZERO {
            positions.iter()
                .map(|p| (p.quantity * p.mark_price / portfolio_value).to_f64().unwrap_or(0.0))
                .fold(0.0f64, f64::max)
        } else {
            0.0
        };

        // Calculate portfolio volatility (simplified)
        let portfolio_volatility = 0.15; // Would calculate from historical data

        // Calculate correlation risk (simplified)
        let correlation_risk = 0.3; // Would calculate from correlation matrix

        Ok(RiskSnapshot {
            timestamp: Utc::now(),
            portfolio_value,
            var_95,
            current_drawdown,
            leverage_ratio,
            largest_position_pct,
            portfolio_volatility,
            correlation_risk,
        })
    }

    fn add_risk_snapshot(&mut self, snapshot: RiskSnapshot) {
        self.risk_snapshots.push_back(snapshot);
        
        // Maintain snapshot buffer size
        while self.risk_snapshots.len() > self.config.max_snapshots {
            self.risk_snapshots.pop_front();
        }
    }

    fn add_alert(&mut self, alert: RiskAlert) {
        if matches!(alert.severity, AlertSeverity::Critical | AlertSeverity::Emergency) {
            self.metrics.critical_alerts_count += 1;
        }

        self.alerts.push_back(alert);
        
        // Maintain alert buffer size
        while self.alerts.len() > self.config.max_alerts {
            self.alerts.pop_front();
        }
    }

    async fn check_var_limits(&self, snapshot: &RiskSnapshot) -> Result<Option<RiskAlert>> {
        if snapshot.var_95 > self.config.alert_thresholds.var_threshold {
            Ok(Some(RiskAlert {
                timestamp: Utc::now(),
                alert_type: RiskAlertType::HighVaR,
                severity: if snapshot.var_95 > self.config.alert_thresholds.var_threshold * 1.5 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                message: format!("Portfolio VaR exceeds threshold: {:.2}%", snapshot.var_95 * 100.0),
                affected_positions: vec![], // Would identify specific positions
                current_value: snapshot.var_95,
                threshold_value: self.config.alert_thresholds.var_threshold,
                recommendation: "Consider reducing position sizes or hedging exposure".to_string(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn check_drawdown_limits(&self, snapshot: &RiskSnapshot) -> Result<Option<RiskAlert>> {
        if snapshot.current_drawdown > self.config.alert_thresholds.drawdown_threshold {
            Ok(Some(RiskAlert {
                timestamp: Utc::now(),
                alert_type: RiskAlertType::ExcessiveDrawdown,
                severity: if snapshot.current_drawdown > self.config.alert_thresholds.drawdown_threshold * 1.5 {
                    AlertSeverity::Emergency
                } else {
                    AlertSeverity::Critical
                },
                message: format!("Portfolio drawdown exceeds threshold: {:.2}%", snapshot.current_drawdown * 100.0),
                affected_positions: vec![],
                current_value: snapshot.current_drawdown,
                threshold_value: self.config.alert_thresholds.drawdown_threshold,
                recommendation: "Consider stopping trading or reducing positions".to_string(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn check_leverage_limits(&self, snapshot: &RiskSnapshot) -> Result<Option<RiskAlert>> {
        if snapshot.leverage_ratio > self.config.alert_thresholds.leverage_threshold {
            Ok(Some(RiskAlert {
                timestamp: Utc::now(),
                alert_type: RiskAlertType::HighLeverage,
                severity: AlertSeverity::Warning,
                message: format!("Portfolio leverage exceeds threshold: {:.2}x", snapshot.leverage_ratio),
                affected_positions: vec![],
                current_value: snapshot.leverage_ratio,
                threshold_value: self.config.alert_thresholds.leverage_threshold,
                recommendation: "Reduce leverage by closing positions or adding capital".to_string(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn check_concentration_limits(&self, positions: &[Position]) -> Result<Option<RiskAlert>> {
        let total_value: Decimal = positions.iter().map(|p| p.quantity * p.mark_price).sum();
        
        if total_value <= Decimal::ZERO {
            return Ok(None);
        }

        for position in positions {
            let position_value = position.quantity * position.mark_price;
            let concentration_pct = (position_value / total_value).to_f64().unwrap_or(0.0);
            
            if concentration_pct > self.config.alert_thresholds.concentration_threshold {
                return Ok(Some(RiskAlert {
                    timestamp: Utc::now(),
                    alert_type: RiskAlertType::Concentration,
                    severity: AlertSeverity::Warning,
                    message: format!("High concentration in {}: {:.2}%", position.symbol, concentration_pct * 100.0),
                    affected_positions: vec![position.symbol.clone()],
                    current_value: concentration_pct,
                    threshold_value: self.config.alert_thresholds.concentration_threshold,
                    recommendation: format!("Consider diversifying by reducing {} position", position.symbol),
                }));
            }
        }

        Ok(None)
    }

    async fn check_volatility_limits(&self, snapshot: &RiskSnapshot) -> Result<Option<RiskAlert>> {
        if snapshot.portfolio_volatility > self.config.alert_thresholds.volatility_threshold {
            Ok(Some(RiskAlert {
                timestamp: Utc::now(),
                alert_type: RiskAlertType::HighVolatility,
                severity: AlertSeverity::Info,
                message: format!("Portfolio volatility is elevated: {:.2}%", snapshot.portfolio_volatility * 100.0),
                affected_positions: vec![],
                current_value: snapshot.portfolio_volatility,
                threshold_value: self.config.alert_thresholds.volatility_threshold,
                recommendation: "Monitor market conditions and consider reducing risk".to_string(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn check_correlation_risks(&self, _positions: &[Position]) -> Result<Option<RiskAlert>> {
        // Simplified correlation risk check
        let correlation_risk = 0.3; // Would calculate from actual correlations
        
        if correlation_risk > 0.7 {
            Ok(Some(RiskAlert {
                timestamp: Utc::now(),
                alert_type: RiskAlertType::CorrelationSpike,
                severity: AlertSeverity::Warning,
                message: "High correlation detected between portfolio positions".to_string(),
                affected_positions: vec![],
                current_value: correlation_risk,
                threshold_value: 0.7,
                recommendation: "Consider adding uncorrelated assets to portfolio".to_string(),
            }))
        } else {
            Ok(None)
        }
    }

    fn calculate_portfolio_var(&self, _positions: &[Position]) -> f64 {
        // Simplified VaR calculation
        0.015 // 1.5% daily VaR
    }

    fn calculate_current_drawdown(&self, _positions: &[Position]) -> f64 {
        // Simplified drawdown calculation
        0.05 // 5% drawdown
    }

    fn calculate_leverage_ratio(&self, positions: &[Position]) -> f64 {
        let total_exposure: Decimal = positions.iter().map(|p| (p.quantity * p.mark_price).abs()).sum();
        let equity: Decimal = positions.iter().map(|p| p.quantity * p.mark_price).sum();
        
        if equity > Decimal::ZERO {
            (total_exposure / equity).to_f64().unwrap_or(1.0)
        } else {
            1.0
        }
    }

    fn assess_risk_level(&self, snapshot: &RiskSnapshot) -> RiskLevel {
        let mut risk_score = 0;

        if snapshot.var_95 > self.config.alert_thresholds.var_threshold {
            risk_score += 1;
        }
        if snapshot.current_drawdown > self.config.alert_thresholds.drawdown_threshold {
            risk_score += 2;
        }
        if snapshot.leverage_ratio > self.config.alert_thresholds.leverage_threshold {
            risk_score += 1;
        }
        if snapshot.largest_position_pct > self.config.alert_thresholds.concentration_threshold {
            risk_score += 1;
        }

        match risk_score {
            0 => RiskLevel::Low,
            1..=2 => RiskLevel::Moderate,
            3..=4 => RiskLevel::High,
            _ => RiskLevel::Extreme,
        }
    }

    async fn create_portfolio_summary(&self, positions: &[Position]) -> Result<PortfolioSummary> {
        let total_value: Decimal = positions.iter().map(|p| p.quantity * p.mark_price).sum();
        let position_count = positions.len();
        
        let largest_position_pct = if total_value > Decimal::ZERO {
            positions.iter()
                .map(|p| (p.quantity * p.mark_price / total_value).to_f64().unwrap_or(0.0))
                .fold(0.0f64, f64::max)
        } else {
            0.0
        };

        // Simplified sector and geographic concentrations
        let mut sector_concentration = HashMap::new();
        sector_concentration.insert("Technology".to_string(), 0.4);
        sector_concentration.insert("Finance".to_string(), 0.3);
        sector_concentration.insert("Other".to_string(), 0.3);

        let mut geographic_concentration = HashMap::new();
        geographic_concentration.insert("US".to_string(), 0.6);
        geographic_concentration.insert("EU".to_string(), 0.2);
        geographic_concentration.insert("APAC".to_string(), 0.2);

        Ok(PortfolioSummary {
            total_value,
            position_count,
            largest_position_pct,
            sector_concentration,
            geographic_concentration,
        })
    }

    fn calculate_max_drawdown_30d(&self) -> f64 {
        self.risk_snapshots.iter()
            .map(|s| s.current_drawdown)
            .fold(0.0f64, f64::max)
    }

    fn generate_recommendations(&self, snapshot: &RiskSnapshot) -> Vec<String> {
        let mut recommendations = Vec::new();

        if snapshot.var_95 > self.config.alert_thresholds.var_threshold {
            recommendations.push("Consider reducing position sizes to lower portfolio VaR".to_string());
        }

        if snapshot.current_drawdown > self.config.alert_thresholds.drawdown_threshold * 0.8 {
            recommendations.push("Monitor drawdown closely and consider defensive measures".to_string());
        }

        if snapshot.leverage_ratio > self.config.alert_thresholds.leverage_threshold * 0.8 {
            recommendations.push("Consider reducing leverage by closing positions or adding capital".to_string());
        }

        if snapshot.largest_position_pct > self.config.alert_thresholds.concentration_threshold * 0.8 {
            recommendations.push("Consider diversifying portfolio to reduce concentration risk".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Portfolio risk levels are within acceptable limits".to_string());
        }

        recommendations
    }

    fn analyze_trends(&self) -> TrendAnalysis {
        if self.risk_snapshots.len() < 10 {
            return TrendAnalysis {
                risk_trend: TrendDirection::Stable,
                volatility_trend: TrendDirection::Stable,
                correlation_trend: TrendDirection::Stable,
                concentration_trend: TrendDirection::Stable,
            };
        }

        // Analyze recent vs older snapshots
        let recent_count = 5;
        let recent_snapshots: Vec<&RiskSnapshot> = self.risk_snapshots.iter().rev().take(recent_count).collect();
        let older_snapshots: Vec<&RiskSnapshot> = self.risk_snapshots.iter().rev().skip(recent_count).take(recent_count).collect();

        let recent_avg_var = recent_snapshots.iter().map(|s| s.var_95).sum::<f64>() / recent_count as f64;
        let older_avg_var = older_snapshots.iter().map(|s| s.var_95).sum::<f64>() / older_snapshots.len() as f64;

        let risk_trend = if recent_avg_var > older_avg_var * 1.1 {
            TrendDirection::Increasing
        } else if recent_avg_var < older_avg_var * 0.9 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        // Similar analysis for other metrics (simplified)
        TrendAnalysis {
            risk_trend,
            volatility_trend: TrendDirection::Stable,
            correlation_trend: TrendDirection::Stable,
            concentration_trend: TrendDirection::Stable,
        }
    }

    async fn update_monitoring_metrics(&mut self, monitoring_time_ms: f64) {
        self.metrics.total_alerts_generated += 1;
        self.metrics.last_monitoring_cycle = Some(Utc::now());
        
        // Update average response time
        if self.metrics.total_alerts_generated == 1 {
            self.metrics.average_response_time_ms = monitoring_time_ms;
        } else {
            self.metrics.average_response_time_ms = 
                (self.metrics.average_response_time_ms * (self.metrics.total_alerts_generated - 1) as f64 + monitoring_time_ms) 
                / self.metrics.total_alerts_generated as f64;
        }
    }

    /// Get monitoring metrics
    pub fn get_monitoring_metrics(&self) -> &MonitoringMetrics {
        &self.metrics
    }
}