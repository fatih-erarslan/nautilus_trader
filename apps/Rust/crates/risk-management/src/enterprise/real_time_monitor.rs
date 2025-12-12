//! Real-Time Risk Monitoring System
//! 
//! Provides ultra-low latency risk monitoring with <50ms calculation times
//! and automated alert generation for risk limit breaches.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock, Notify};
use tokio::time;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::*;

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitorConfig {
    /// Monitoring frequency in milliseconds
    pub monitoring_frequency_ms: u64,
    
    /// VaR confidence levels to monitor
    pub var_confidence_levels: Vec<f64>,
    
    /// Historical window for calculations
    pub historical_window_days: u32,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// Cache size for calculations
    pub cache_size: usize,
    
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

impl Default for RealTimeMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_frequency_ms: 100, // 100ms monitoring cycle
            var_confidence_levels: vec![0.95, 0.99, 0.999],
            historical_window_days: 252, // 1 year of trading days
            enable_simd: true,
            enable_gpu: true,
            cache_size: 10000,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum VaR as percentage of portfolio value
    pub max_var_percent: f64,
    
    /// Maximum concentration in single position
    pub max_position_concentration: f64,
    
    /// Maximum sector concentration
    pub max_sector_concentration: f64,
    
    /// Maximum drawdown threshold
    pub max_drawdown_percent: f64,
    
    /// Minimum liquidity score
    pub min_liquidity_score: f64,
    
    /// Maximum leverage ratio
    pub max_leverage_ratio: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_var_percent: 5.0,
            max_position_concentration: 10.0,
            max_sector_concentration: 25.0,
            max_drawdown_percent: 15.0,
            min_liquidity_score: 0.7,
            max_leverage_ratio: 3.0,
        }
    }
}

/// Real-time risk calculation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeRiskCalculation {
    pub portfolio_id: Uuid,
    pub var_95: f64,
    pub var_99: f64,
    pub var_999: f64,
    pub expected_shortfall_95: f64,
    pub expected_shortfall_99: f64,
    pub max_drawdown: f64,
    pub concentration_risk: ConcentrationRisk,
    pub liquidity_risk: LiquidityRisk,
    pub leverage_ratio: f64,
    pub beta: Option<f64>,
    pub correlation_breakdown: bool,
    pub calculation_time: Duration,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Concentration risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationRisk {
    pub top_position_percent: f64,
    pub top_5_positions_percent: f64,
    pub herfindahl_index: f64,
    pub sector_concentrations: HashMap<String, f64>,
    pub asset_class_concentrations: HashMap<AssetClass, f64>,
}

/// Liquidity risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityRisk {
    pub weighted_liquidity_score: f64,
    pub illiquid_positions_percent: f64,
    pub time_to_liquidate_days: f64,
    pub liquidity_concentration: f64,
}

/// Real-time risk monitor implementation
#[derive(Debug)]
pub struct RealTimeRiskMonitor {
    config: RealTimeMonitorConfig,
    portfolio_cache: Arc<RwLock<HashMap<Uuid, Portfolio>>>,
    market_data_cache: Arc<RwLock<HashMap<String, MarketData>>>,
    calculation_cache: Arc<RwLock<HashMap<String, RealTimeRiskCalculation>>>,
    alert_sender: mpsc::UnboundedSender<RiskAlert>,
    performance_tracker: Arc<RwLock<MonitoringPerformanceTracker>>,
    shutdown_notify: Arc<Notify>,
    is_running: Arc<RwLock<bool>>,
}

impl RealTimeRiskMonitor {
    /// Create new real-time risk monitor
    pub async fn new(
        config: RealTimeMonitorConfig,
        alert_sender: mpsc::UnboundedSender<RiskAlert>,
    ) -> Result<Self> {
        info!("Initializing Real-Time Risk Monitor");
        
        let monitor = Self {
            config,
            portfolio_cache: Arc::new(RwLock::new(HashMap::new())),
            market_data_cache: Arc::new(RwLock::new(HashMap::new())),
            calculation_cache: Arc::new(RwLock::new(HashMap::new())),
            alert_sender,
            performance_tracker: Arc::new(RwLock::new(MonitoringPerformanceTracker::new())),
            shutdown_notify: Arc::new(Notify::new()),
            is_running: Arc::new(RwLock::new(false)),
        };
        
        info!("Real-Time Risk Monitor initialized successfully");
        Ok(monitor)
    }
    
    /// Start real-time monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            warn!("Real-time monitoring is already running");
            return Ok(());
        }
        
        info!("Starting real-time risk monitoring");
        *is_running = true;
        drop(is_running);
        
        // Start monitoring loop
        let monitor_clone = self.clone_for_task().await;
        tokio::spawn(async move {
            monitor_clone.monitoring_loop().await;
        });
        
        info!("Real-time risk monitoring started");
        Ok(())
    }
    
    /// Stop real-time monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        info!("Stopping real-time risk monitoring");
        
        let mut is_running = self.is_running.write().await;
        *is_running = false;
        drop(is_running);
        
        self.shutdown_notify.notify_one();
        
        info!("Real-time risk monitoring stopped");
        Ok(())
    }
    
    /// Update portfolio data
    pub async fn update_portfolio(&self, portfolio: Portfolio) -> Result<()> {
        let mut cache = self.portfolio_cache.write().await;
        cache.insert(portfolio.id, portfolio);
        Ok(())
    }
    
    /// Update market data
    pub async fn update_market_data(&self, market_data: Vec<MarketData>) -> Result<()> {
        let mut cache = self.market_data_cache.write().await;
        for data in market_data {
            cache.insert(data.symbol.clone(), data);
        }
        Ok(())
    }
    
    /// Calculate real-time risk metrics for a portfolio
    pub async fn calculate_portfolio_risk(&self, portfolio: &Portfolio) -> Result<RealTimeRiskCalculation> {
        let start_time = Instant::now();
        
        // Track performance
        {
            let mut tracker = self.performance_tracker.write().await;
            tracker.start_calculation();
        }
        
        // Calculate VaR at different confidence levels
        let var_results = self.calculate_var_multiple_levels(portfolio).await?;
        
        // Calculate concentration risk
        let concentration_risk = self.calculate_concentration_risk(portfolio).await?;
        
        // Calculate liquidity risk
        let liquidity_risk = self.calculate_liquidity_risk(portfolio).await?;
        
        // Calculate leverage
        let leverage_ratio = self.calculate_leverage_ratio(portfolio);
        
        // Calculate beta (if benchmark available)
        let beta = self.calculate_portfolio_beta(portfolio).await?;
        
        // Check for correlation breakdown
        let correlation_breakdown = self.detect_correlation_breakdown(portfolio).await?;
        
        // Calculate maximum drawdown
        let max_drawdown = self.calculate_max_drawdown(portfolio).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Update performance tracker
        {
            let mut tracker = self.performance_tracker.write().await;
            tracker.end_calculation(calculation_time);
        }
        
        // Check performance target
        if calculation_time > Duration::from_millis(50) {
            warn!(
                "Risk calculation took {:?}, exceeding 50ms target for portfolio {}",
                calculation_time, portfolio.id
            );
        }
        
        let result = RealTimeRiskCalculation {
            portfolio_id: portfolio.id,
            var_95: var_results.get(&0.95).copied().unwrap_or(0.0),
            var_99: var_results.get(&0.99).copied().unwrap_or(0.0),
            var_999: var_results.get(&0.999).copied().unwrap_or(0.0),
            expected_shortfall_95: self.calculate_expected_shortfall(portfolio, 0.95).await?,
            expected_shortfall_99: self.calculate_expected_shortfall(portfolio, 0.99).await?,
            max_drawdown,
            concentration_risk,
            liquidity_risk,
            leverage_ratio,
            beta,
            correlation_breakdown,
            calculation_time,
            timestamp: chrono::Utc::now(),
        };
        
        // Cache the result
        let cache_key = format!("{}_{}", portfolio.id, chrono::Utc::now().timestamp());
        {
            let mut cache = self.calculation_cache.write().await;
            cache.insert(cache_key, result.clone());
        }
        
        // Check for alerts
        self.check_and_send_alerts(&result).await?;
        
        Ok(result)
    }
    
    /// Main monitoring loop
    async fn monitoring_loop(&self) {
        let mut interval = time::interval(Duration::from_millis(self.config.monitoring_frequency_ms));
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.perform_monitoring_cycle().await {
                        error!("Error in monitoring cycle: {}", e);
                    }
                }
                _ = self.shutdown_notify.notified() => {
                    info!("Monitoring loop shutdown requested");
                    break;
                }
            }
            
            // Check if monitoring should continue
            let is_running = *self.is_running.read().await;
            if !is_running {
                break;
            }
        }
        
        info!("Real-time monitoring loop stopped");
    }
    
    /// Perform a single monitoring cycle
    async fn perform_monitoring_cycle(&self) -> Result<()> {
        let portfolios = {
            let cache = self.portfolio_cache.read().await;
            cache.values().cloned().collect::<Vec<_>>()
        };
        
        for portfolio in portfolios {
            if let Err(e) = self.calculate_portfolio_risk(&portfolio).await {
                error!("Error calculating risk for portfolio {}: {}", portfolio.id, e);
            }
        }
        
        Ok(())
    }
    
    /// Calculate VaR at multiple confidence levels
    async fn calculate_var_multiple_levels(&self, portfolio: &Portfolio) -> Result<HashMap<f64, f64>> {
        let mut results = HashMap::new();
        
        // Simplified VaR calculation - in production, this would use more sophisticated methods
        for &confidence_level in &self.config.var_confidence_levels {
            let var = self.calculate_historical_var(portfolio, confidence_level)?;
            results.insert(confidence_level, var);
        }
        
        Ok(results)
    }
    
    /// Calculate historical VaR
    fn calculate_historical_var(&self, portfolio: &Portfolio, confidence_level: f64) -> Result<f64> {
        if portfolio.positions.is_empty() {
            return Ok(0.0);
        }
        
        // Simplified calculation - extract returns from positions
        let mut returns = Vec::new();
        for position in &portfolio.positions {
            let return_pct = position.unrealized_pnl / position.cost_basis.max(1.0);
            returns.push(return_pct);
        }
        
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let var_index = ((1.0 - confidence_level) * returns.len() as f64).floor() as usize;
        let var_percentile = returns[var_index.min(returns.len() - 1)];
        
        Ok(-var_percentile * portfolio.total_market_value)
    }
    
    /// Calculate expected shortfall (CVaR)
    async fn calculate_expected_shortfall(&self, portfolio: &Portfolio, confidence_level: f64) -> Result<f64> {
        let var = self.calculate_historical_var(portfolio, confidence_level)?;
        
        // Simplified ES calculation
        // In production, this would be more sophisticated
        Ok(var * 1.3) // Typical relationship between VaR and ES
    }
    
    /// Calculate concentration risk metrics
    async fn calculate_concentration_risk(&self, portfolio: &Portfolio) -> Result<ConcentrationRisk> {
        if portfolio.positions.is_empty() {
            return Ok(ConcentrationRisk {
                top_position_percent: 0.0,
                top_5_positions_percent: 0.0,
                herfindahl_index: 0.0,
                sector_concentrations: HashMap::new(),
                asset_class_concentrations: HashMap::new(),
            });
        }
        
        // Calculate position weights
        let total_value = portfolio.total_market_value.max(1.0);
        let mut position_weights: Vec<f64> = portfolio.positions
            .iter()
            .map(|p| (p.market_value.abs() / total_value) * 100.0)
            .collect();
        
        position_weights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        // Top position concentration
        let top_position_percent = position_weights.first().copied().unwrap_or(0.0);
        
        // Top 5 positions concentration
        let top_5_positions_percent = position_weights.iter().take(5).sum();
        
        // Herfindahl Index
        let herfindahl_index = position_weights.iter()
            .map(|w| (w / 100.0).powi(2))
            .sum::<f64>();
        
        // Sector concentrations
        let mut sector_concentrations = HashMap::new();
        for position in &portfolio.positions {
            if let Some(sector) = &position.sector {
                let weight = (position.market_value.abs() / total_value) * 100.0;
                *sector_concentrations.entry(sector.clone()).or_insert(0.0) += weight;
            }
        }
        
        // Asset class concentrations
        let mut asset_class_concentrations = HashMap::new();
        for position in &portfolio.positions {
            let weight = (position.market_value.abs() / total_value) * 100.0;
            *asset_class_concentrations.entry(position.asset_class.clone()).or_insert(0.0) += weight;
        }
        
        Ok(ConcentrationRisk {
            top_position_percent,
            top_5_positions_percent,
            herfindahl_index,
            sector_concentrations,
            asset_class_concentrations,
        })
    }
    
    /// Calculate liquidity risk metrics
    async fn calculate_liquidity_risk(&self, portfolio: &Portfolio) -> Result<LiquidityRisk> {
        if portfolio.positions.is_empty() {
            return Ok(LiquidityRisk {
                weighted_liquidity_score: 1.0,
                illiquid_positions_percent: 0.0,
                time_to_liquidate_days: 0.0,
                liquidity_concentration: 0.0,
            });
        }
        
        let market_data_cache = self.market_data_cache.read().await;
        let total_value = portfolio.total_market_value.max(1.0);
        
        let mut weighted_liquidity_score = 0.0;
        let mut illiquid_value = 0.0;
        let mut total_liquidation_time = 0.0;
        
        for position in &portfolio.positions {
            let weight = position.market_value.abs() / total_value;
            
            // Get liquidity score from market data
            let liquidity_score = market_data_cache
                .get(&position.symbol)
                .and_then(|md| md.liquidity_score)
                .unwrap_or(0.5); // Default liquidity score
            
            weighted_liquidity_score += weight * liquidity_score;
            
            // Consider positions with liquidity score < 0.3 as illiquid
            if liquidity_score < 0.3 {
                illiquid_value += position.market_value.abs();
            }
            
            // Estimate time to liquidate based on liquidity score
            let liquidation_time = match liquidity_score {
                s if s > 0.8 => 0.25, // 6 hours
                s if s > 0.6 => 1.0,  // 1 day
                s if s > 0.4 => 3.0,  // 3 days
                s if s > 0.2 => 7.0,  // 1 week
                _ => 30.0,            // 1 month
            };
            
            total_liquidation_time += weight * liquidation_time;
        }
        
        let illiquid_positions_percent = (illiquid_value / total_value) * 100.0;
        
        // Calculate liquidity concentration risk
        let liquidity_concentration = self.calculate_liquidity_concentration(portfolio, &market_data_cache);
        
        Ok(LiquidityRisk {
            weighted_liquidity_score,
            illiquid_positions_percent,
            time_to_liquidate_days: total_liquidation_time,
            liquidity_concentration,
        })
    }
    
    /// Calculate liquidity concentration
    fn calculate_liquidity_concentration(
        &self,
        portfolio: &Portfolio,
        market_data_cache: &HashMap<String, MarketData>,
    ) -> f64 {
        // Calculate Herfindahl index for liquidity concentration
        let total_value = portfolio.total_market_value.max(1.0);
        
        let liquidity_weighted_concentration: f64 = portfolio.positions
            .iter()
            .map(|position| {
                let weight = position.market_value.abs() / total_value;
                let liquidity_score = market_data_cache
                    .get(&position.symbol)
                    .and_then(|md| md.liquidity_score)
                    .unwrap_or(0.5);
                
                // Higher concentration risk for less liquid positions
                weight * (1.0 - liquidity_score)
            })
            .sum();
        
        liquidity_weighted_concentration
    }
    
    /// Calculate leverage ratio
    fn calculate_leverage_ratio(&self, portfolio: &Portfolio) -> f64 {
        let gross_exposure: f64 = portfolio.positions
            .iter()
            .map(|p| p.market_value.abs())
            .sum();
        
        let net_asset_value = portfolio.total_market_value + portfolio.cash;
        
        if net_asset_value <= 0.0 {
            return 0.0;
        }
        
        gross_exposure / net_asset_value
    }
    
    /// Calculate portfolio beta
    async fn calculate_portfolio_beta(&self, portfolio: &Portfolio) -> Result<Option<f64>> {
        // Simplified beta calculation
        // In production, this would use regression against benchmark
        if portfolio.benchmark.is_none() {
            return Ok(None);
        }
        
        // Placeholder calculation
        Ok(Some(1.0))
    }
    
    /// Detect correlation breakdown
    async fn detect_correlation_breakdown(&self, _portfolio: &Portfolio) -> Result<bool> {
        // Simplified correlation breakdown detection
        // In production, this would analyze correlation matrices
        Ok(false)
    }
    
    /// Calculate maximum drawdown
    async fn calculate_max_drawdown(&self, portfolio: &Portfolio) -> Result<f64> {
        // Simplified drawdown calculation
        let current_unrealized_pnl = portfolio.total_unrealized_pnl;
        let total_value = portfolio.total_market_value;
        
        if total_value <= 0.0 {
            return Ok(0.0);
        }
        
        // This is a simplified calculation
        // In production, you'd track high-water marks over time
        Ok((current_unrealized_pnl.min(0.0) / total_value * 100.0).abs())
    }
    
    /// Check for alerts and send them
    async fn check_and_send_alerts(&self, calculation: &RealTimeRiskCalculation) -> Result<()> {
        let mut alerts = Vec::new();
        
        // Check VaR threshold
        let portfolio_value = self.get_portfolio_value(calculation.portfolio_id).await?;
        let var_percent = (calculation.var_95 / portfolio_value) * 100.0;
        
        if var_percent > self.config.alert_thresholds.max_var_percent {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                level: AlertLevel::Critical,
                title: "VaR Limit Exceeded".to_string(),
                description: format!("Portfolio VaR ({:.2}%) exceeds threshold ({:.2}%)", 
                                   var_percent, self.config.alert_thresholds.max_var_percent),
                metric_name: "VaR_95".to_string(),
                current_value: var_percent,
                threshold_value: self.config.alert_thresholds.max_var_percent,
                portfolio_id: Some(calculation.portfolio_id),
                position_symbol: None,
                recommended_action: "Reduce position sizes or hedge portfolio".to_string(),
                auto_action_taken: false,
                timestamp: chrono::Utc::now(),
            });
        }
        
        // Check concentration risk
        if calculation.concentration_risk.top_position_percent > self.config.alert_thresholds.max_position_concentration {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                level: AlertLevel::Warning,
                title: "Position Concentration Risk".to_string(),
                description: format!("Top position concentration ({:.2}%) exceeds threshold ({:.2}%)", 
                                   calculation.concentration_risk.top_position_percent,
                                   self.config.alert_thresholds.max_position_concentration),
                metric_name: "Position_Concentration".to_string(),
                current_value: calculation.concentration_risk.top_position_percent,
                threshold_value: self.config.alert_thresholds.max_position_concentration,
                portfolio_id: Some(calculation.portfolio_id),
                position_symbol: None,
                recommended_action: "Diversify holdings or reduce position size".to_string(),
                auto_action_taken: false,
                timestamp: chrono::Utc::now(),
            });
        }
        
        // Check drawdown
        if calculation.max_drawdown > self.config.alert_thresholds.max_drawdown_percent {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                level: AlertLevel::Critical,
                title: "Maximum Drawdown Exceeded".to_string(),
                description: format!("Current drawdown ({:.2}%) exceeds threshold ({:.2}%)", 
                                   calculation.max_drawdown,
                                   self.config.alert_thresholds.max_drawdown_percent),
                metric_name: "Max_Drawdown".to_string(),
                current_value: calculation.max_drawdown,
                threshold_value: self.config.alert_thresholds.max_drawdown_percent,
                portfolio_id: Some(calculation.portfolio_id),
                position_symbol: None,
                recommended_action: "Implement stop-loss or hedging strategy".to_string(),
                auto_action_taken: false,
                timestamp: chrono::Utc::now(),
            });
        }
        
        // Check liquidity risk
        if calculation.liquidity_risk.weighted_liquidity_score < self.config.alert_thresholds.min_liquidity_score {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                level: AlertLevel::Warning,
                title: "Low Portfolio Liquidity".to_string(),
                description: format!("Portfolio liquidity score ({:.2}) below threshold ({:.2})", 
                                   calculation.liquidity_risk.weighted_liquidity_score,
                                   self.config.alert_thresholds.min_liquidity_score),
                metric_name: "Liquidity_Score".to_string(),
                current_value: calculation.liquidity_risk.weighted_liquidity_score,
                threshold_value: self.config.alert_thresholds.min_liquidity_score,
                portfolio_id: Some(calculation.portfolio_id),
                position_symbol: None,
                recommended_action: "Increase allocation to liquid assets".to_string(),
                auto_action_taken: false,
                timestamp: chrono::Utc::now(),
            });
        }
        
        // Check leverage
        if calculation.leverage_ratio > self.config.alert_thresholds.max_leverage_ratio {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                level: AlertLevel::Critical,
                title: "High Leverage Detected".to_string(),
                description: format!("Leverage ratio ({:.2}) exceeds threshold ({:.2})", 
                                   calculation.leverage_ratio,
                                   self.config.alert_thresholds.max_leverage_ratio),
                metric_name: "Leverage_Ratio".to_string(),
                current_value: calculation.leverage_ratio,
                threshold_value: self.config.alert_thresholds.max_leverage_ratio,
                portfolio_id: Some(calculation.portfolio_id),
                position_symbol: None,
                recommended_action: "Reduce leverage or increase capital".to_string(),
                auto_action_taken: false,
                timestamp: chrono::Utc::now(),
            });
        }
        
        // Send all alerts
        for alert in alerts {
            if let Err(e) = self.alert_sender.send(alert) {
                error!("Failed to send risk alert: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Get portfolio value by ID
    async fn get_portfolio_value(&self, portfolio_id: Uuid) -> Result<f64> {
        let cache = self.portfolio_cache.read().await;
        Ok(cache.get(&portfolio_id)
            .map(|p| p.total_market_value)
            .unwrap_or(0.0))
    }
    
    /// Clone for task execution
    async fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            portfolio_cache: Arc::clone(&self.portfolio_cache),
            market_data_cache: Arc::clone(&self.market_data_cache),
            calculation_cache: Arc::clone(&self.calculation_cache),
            alert_sender: self.alert_sender.clone(),
            performance_tracker: Arc::clone(&self.performance_tracker),
            shutdown_notify: Arc::clone(&self.shutdown_notify),
            is_running: Arc::clone(&self.is_running),
        }
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<MonitoringPerformanceMetrics> {
        let tracker = self.performance_tracker.read().await;
        Ok(tracker.get_metrics())
    }
}

/// Performance tracker for monitoring
#[derive(Debug)]
struct MonitoringPerformanceTracker {
    total_calculations: u64,
    total_calculation_time: Duration,
    max_calculation_time: Duration,
    min_calculation_time: Duration,
    start_time: Option<Instant>,
}

impl MonitoringPerformanceTracker {
    fn new() -> Self {
        Self {
            total_calculations: 0,
            total_calculation_time: Duration::from_nanos(0),
            max_calculation_time: Duration::from_nanos(0),
            min_calculation_time: Duration::from_secs(u64::MAX),
            start_time: None,
        }
    }
    
    fn start_calculation(&mut self) {
        self.start_time = Some(Instant::now());
    }
    
    fn end_calculation(&mut self, duration: Duration) {
        self.total_calculations += 1;
        self.total_calculation_time += duration;
        
        if duration > self.max_calculation_time {
            self.max_calculation_time = duration;
        }
        
        if duration < self.min_calculation_time {
            self.min_calculation_time = duration;
        }
        
        self.start_time = None;
    }
    
    fn get_metrics(&self) -> MonitoringPerformanceMetrics {
        let avg_calculation_time = if self.total_calculations > 0 {
            self.total_calculation_time / self.total_calculations as u32
        } else {
            Duration::from_nanos(0)
        };
        
        MonitoringPerformanceMetrics {
            total_calculations: self.total_calculations,
            avg_calculation_time,
            max_calculation_time: self.max_calculation_time,
            min_calculation_time: if self.min_calculation_time == Duration::from_secs(u64::MAX) {
                Duration::from_nanos(0)
            } else {
                self.min_calculation_time
            },
        }
    }
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringPerformanceMetrics {
    pub total_calculations: u64,
    pub avg_calculation_time: Duration,
    pub max_calculation_time: Duration,
    pub min_calculation_time: Duration,
}