//! Risk management system for Tengri trading strategy
//! 
//! Provides comprehensive risk management including position sizing,
//! portfolio risk monitoring, correlation analysis, and emergency controls.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use statrs::statistics::Statistics;

use crate::{Result, TengriError};
use crate::config::RiskConfig;
use crate::types::{
    TradingSignal, Position, PortfolioMetrics, RiskMetrics, PositionSide
};

/// Risk management engine for portfolio and position risk control
pub struct RiskManager {
    config: RiskConfig,
    position_limits: Arc<RwLock<HashMap<String, PositionLimit>>>,
    correlation_matrix: Arc<RwLock<Array2<f64>>>,
    risk_metrics_cache: Arc<RwLock<RiskMetrics>>,
    daily_pnl_history: Arc<RwLock<Vec<f64>>>,
    volatility_cache: Arc<RwLock<HashMap<String, f64>>>,
    var_calculator: VaRCalculator,
}

/// Position limit configuration for individual instruments
#[derive(Debug, Clone)]
pub struct PositionLimit {
    pub symbol: String,
    pub max_size: f64,
    pub max_value: f64,
    pub current_size: f64,
    pub current_value: f64,
    pub last_updated: DateTime<Utc>,
}

/// Risk evaluation result for signal processing
#[derive(Debug, Clone)]
pub struct RiskEvaluation {
    pub approved: bool,
    pub reason: String,
    pub max_position_size: f64,
    pub confidence_adjustment: f64,
    pub risk_score: f64,
}

/// Value at Risk calculator
pub struct VaRCalculator {
    confidence_level: f64,
    lookback_period: usize,
    monte_carlo_simulations: usize,
}

/// Portfolio correlation analyzer
pub struct CorrelationAnalyzer {
    window_size: usize,
    min_correlation_threshold: f64,
}

/// Kelly Criterion calculator for optimal position sizing
pub struct KellyCriterion;

impl RiskManager {
    /// Create new risk manager with configuration
    pub async fn new(config: RiskConfig) -> Result<Self> {
        let var_calculator = VaRCalculator::new(0.95, 252, 10000); // 95% confidence, 1-year lookback, 10k simulations
        
        Ok(Self {
            config,
            position_limits: Arc::new(RwLock::new(HashMap::new())),
            correlation_matrix: Arc::new(RwLock::new(Array2::zeros((0, 0)))),
            risk_metrics_cache: Arc::new(RwLock::new(RiskMetrics::default())),
            daily_pnl_history: Arc::new(RwLock::new(Vec::new())),
            volatility_cache: Arc::new(RwLock::new(HashMap::new())),
            var_calculator,
        })
    }

    /// Evaluate a trading signal for risk compliance
    pub async fn evaluate_signal(&self, signal: &TradingSignal, current_position: Option<&Position>) -> Result<RiskEvaluation> {
        let mut evaluation = RiskEvaluation {
            approved: true,
            reason: String::new(),
            max_position_size: 0.0,
            confidence_adjustment: 1.0,
            risk_score: 0.0,
        };

        // Calculate maximum allowed position size
        evaluation.max_position_size = self.calculate_position_size(signal).await?;

        // Check position limits
        if let Some(position) = current_position {
            let position_limit_check = self.check_position_limits(&signal.symbol, position).await?;
            if !position_limit_check.0 {
                evaluation.approved = false;
                evaluation.reason = position_limit_check.1;
                return Ok(evaluation);
            }
        }

        // Check portfolio-level risk limits
        let portfolio_risk_check = self.check_portfolio_risk_limits().await?;
        if !portfolio_risk_check.0 {
            evaluation.approved = false;
            evaluation.reason = portfolio_risk_check.1;
            return Ok(evaluation);
        }

        // Check correlation limits
        let correlation_check = self.check_correlation_limits(&signal.symbol).await?;
        if !correlation_check.0 {
            evaluation.approved = false;
            evaluation.reason = correlation_check.1;
            return Ok(evaluation);
        }

        // Calculate risk score
        evaluation.risk_score = self.calculate_signal_risk_score(signal, current_position).await?;

        // Adjust confidence based on risk factors
        evaluation.confidence_adjustment = self.calculate_confidence_adjustment(&evaluation).await?;

        // Final approval check
        if evaluation.risk_score > 80.0 {
            evaluation.approved = false;
            evaluation.reason = format!("Risk score too high: {:.1}", evaluation.risk_score);
        }

        Ok(evaluation)
    }

    /// Calculate optimal position size for a signal
    pub async fn calculate_position_size(&self, signal: &TradingSignal) -> Result<f64> {
        let symbol = &signal.symbol;
        
        // Base position size from configuration
        let base_size = self.config.position_sizing.base_size;
        
        // Get current volatility for this symbol
        let volatility = {
            let vol_cache = self.volatility_cache.read().await;
            vol_cache.get(symbol).copied().unwrap_or(0.02) // Default 2% volatility
        };

        // Adjust size based on volatility
        let volatility_adjustment = match self.config.position_sizing.method.as_str() {
            "volatility_adjusted" => {
                let vol_lookback = self.config.position_sizing.volatility_lookback as usize;
                self.calculate_volatility_adjusted_size(base_size, volatility, vol_lookback).await?
            }
            "kelly" => {
                self.calculate_kelly_position_size(signal, volatility).await?
            }
            "risk_parity" => {
                self.calculate_risk_parity_size(symbol, base_size).await?
            }
            "fixed" => base_size,
            _ => base_size,
        };

        // Apply signal strength adjustment
        let signal_adjusted_size = volatility_adjustment * signal.strength.abs();

        // Apply maximum position size limit
        let max_size = self.config.max_position_size;
        let final_size = signal_adjusted_size.min(max_size);

        // Apply leverage limit
        let leveraged_size = final_size * self.config.position_sizing.max_leverage;

        Ok(leveraged_size)
    }

    /// Calculate volatility-adjusted position size
    async fn calculate_volatility_adjusted_size(&self, base_size: f64, volatility: f64, lookback: usize) -> Result<f64> {
        // Target volatility (e.g., 1% daily)
        let target_volatility = 0.01;
        
        // Adjust position size inversely to volatility
        let volatility_ratio = target_volatility / volatility.max(0.001); // Avoid division by zero
        let adjusted_size = base_size * volatility_ratio;
        
        // Cap the adjustment to reasonable bounds
        Ok(adjusted_size.max(base_size * 0.1).min(base_size * 3.0))
    }

    /// Calculate Kelly Criterion optimal position size
    async fn calculate_kelly_position_size(&self, signal: &TradingSignal, volatility: f64) -> Result<f64> {
        // Kelly formula: f = (bp - q) / b
        // where f = fraction of capital to wager
        //       b = odds of winning (1 for even odds)
        //       p = probability of winning
        //       q = probability of losing (1 - p)
        
        let win_probability = signal.confidence; // Use signal confidence as win probability
        let loss_probability = 1.0 - win_probability;
        
        // Estimate odds based on signal strength and volatility
        let expected_return = signal.strength * 0.02; // Assume 2% expected move
        let odds = expected_return / volatility;
        
        // Kelly fraction
        let kelly_fraction = (odds * win_probability - loss_probability) / odds;
        
        // Apply fractional Kelly to reduce risk
        let fractional_kelly = kelly_fraction * self.config.position_sizing.kelly_fraction;
        
        // Convert to position size (as fraction of portfolio)
        Ok(fractional_kelly.max(0.0).min(self.config.max_position_size))
    }

    /// Calculate risk parity position size
    async fn calculate_risk_parity_size(&self, symbol: &str, base_size: f64) -> Result<f64> {
        // Risk parity aims to equalize risk contribution across positions
        let volatility = {
            let vol_cache = self.volatility_cache.read().await;
            vol_cache.get(symbol).copied().unwrap_or(0.02)
        };
        
        // Target risk per position (e.g., 1% portfolio volatility contribution)
        let target_risk_contribution = 0.01;
        
        // Position size = target_risk / volatility
        let risk_parity_size = target_risk_contribution / volatility;
        
        Ok(risk_parity_size.min(self.config.max_position_size))
    }

    /// Check position limits for a specific symbol
    async fn check_position_limits(&self, symbol: &str, position: &Position) -> Result<(bool, String)> {
        let position_limits = self.position_limits.read().await;
        
        if let Some(limit) = position_limits.get(symbol) {
            if position.size >= limit.max_size {
                return Ok((false, format!("Position size limit exceeded for {}: {} >= {}", 
                    symbol, position.size, limit.max_size)));
            }
            
            let position_value = position.value();
            if position_value >= limit.max_value {
                return Ok((false, format!("Position value limit exceeded for {}: {:.2} >= {:.2}", 
                    symbol, position_value, limit.max_value)));
            }
        }

        Ok((true, String::new()))
    }

    /// Check portfolio-level risk limits
    async fn check_portfolio_risk_limits(&self) -> Result<(bool, String)> {
        let risk_metrics = self.risk_metrics_cache.read().await;
        
        // Check leverage limits
        if risk_metrics.leverage > self.config.position_sizing.max_leverage {
            return Ok((false, format!("Portfolio leverage limit exceeded: {:.2}x > {:.2}x", 
                risk_metrics.leverage, self.config.position_sizing.max_leverage)));
        }

        // Check daily loss limits
        if risk_metrics.current_daily_loss < -self.config.max_daily_loss {
            return Ok((false, format!("Daily loss limit exceeded: {:.2}% < -{:.2}%", 
                risk_metrics.current_daily_loss * 100.0, self.config.max_daily_loss * 100.0)));
        }

        // Check portfolio loss limits
        if risk_metrics.current_daily_loss < -self.config.max_portfolio_loss {
            return Ok((false, format!("Portfolio loss limit exceeded: {:.2}% < -{:.2}%", 
                risk_metrics.current_daily_loss * 100.0, self.config.max_portfolio_loss * 100.0)));
        }

        Ok((true, String::new()))
    }

    /// Check correlation limits to avoid concentration risk
    async fn check_correlation_limits(&self, symbol: &str) -> Result<(bool, String)> {
        let correlation_matrix = self.correlation_matrix.read().await;
        
        // This would require mapping symbols to matrix indices
        // For now, implement a simplified check
        
        // Count highly correlated positions
        let correlation_threshold = self.config.correlation_limits.max_correlation;
        let max_correlated = self.config.correlation_limits.max_correlated_positions;
        
        // This is a placeholder - would need actual correlation calculation
        let correlated_count = 1; // Simplified
        
        if correlated_count as u32 >= max_correlated {
            return Ok((false, format!("Too many correlated positions: {} >= {}", 
                correlated_count, max_correlated)));
        }

        Ok((true, String::new()))
    }

    /// Calculate risk score for a signal
    async fn calculate_signal_risk_score(&self, signal: &TradingSignal, current_position: Option<&Position>) -> Result<f64> {
        let mut risk_score = 0.0;

        // Signal quality risk (inverse of confidence)
        risk_score += (1.0 - signal.confidence) * 30.0;

        // Volatility risk
        let volatility = {
            let vol_cache = self.volatility_cache.read().await;
            vol_cache.get(&signal.symbol).copied().unwrap_or(0.02)
        };
        risk_score += (volatility - 0.02) * 1000.0; // Scale volatility risk

        // Position concentration risk
        if let Some(position) = current_position {
            let position_concentration = position.size / self.config.max_position_size;
            risk_score += position_concentration * 20.0;
        }

        // Market regime risk
        // This would use market state analysis
        risk_score += 10.0; // Base market risk

        Ok(risk_score.max(0.0).min(100.0))
    }

    /// Calculate confidence adjustment factor
    async fn calculate_confidence_adjustment(&self, evaluation: &RiskEvaluation) -> Result<f64> {
        let mut adjustment = 1.0;

        // Reduce confidence for high risk scores
        if evaluation.risk_score > 50.0 {
            adjustment *= 1.0 - ((evaluation.risk_score - 50.0) / 50.0) * 0.3;
        }

        // Increase confidence for low risk scores
        if evaluation.risk_score < 20.0 {
            adjustment *= 1.0 + (20.0 - evaluation.risk_score) / 20.0 * 0.2;
        }

        Ok(adjustment.max(0.1).min(1.5))
    }

    /// Calculate portfolio risk metrics
    pub async fn calculate_portfolio_risk(&self, positions: &HashMap<String, Position>, portfolio_metrics: &PortfolioMetrics) -> Result<RiskMetrics> {
        let mut risk_metrics = RiskMetrics::default();
        risk_metrics.timestamp = Utc::now();

        // Calculate leverage
        let total_position_value: f64 = positions.values().map(|p| p.value()).sum();
        if portfolio_metrics.total_value > 0.0 {
            risk_metrics.leverage = total_position_value / portfolio_metrics.total_value;
        }

        // Calculate concentration (largest position as % of portfolio)
        if portfolio_metrics.total_value > 0.0 && !positions.is_empty() {
            let largest_position_value = positions.values()
                .map(|p| p.value())
                .fold(0.0, f64::max);
            risk_metrics.concentration = largest_position_value / portfolio_metrics.total_value;
        }

        // Calculate current daily loss
        risk_metrics.current_daily_loss = portfolio_metrics.realized_pnl + portfolio_metrics.unrealized_pnl;

        // Calculate margin usage (simplified)
        risk_metrics.margin_usage = risk_metrics.leverage * 0.1; // Assume 10% margin requirement

        // Market correlation (placeholder)
        risk_metrics.market_correlation = 0.5; // Would need market index data

        // Risk-adjusted return (Sharpe ratio approximation)
        if let Some(sharpe) = portfolio_metrics.sharpe_ratio {
            risk_metrics.risk_adjusted_return = sharpe;
        }

        // Risk score (0-100)
        risk_metrics.risk_score = self.calculate_portfolio_risk_score(&risk_metrics).await?;

        // Update cache
        {
            let mut cache = self.risk_metrics_cache.write().await;
            *cache = risk_metrics.clone();
        }

        Ok(risk_metrics)
    }

    /// Calculate overall portfolio risk score
    async fn calculate_portfolio_risk_score(&self, risk_metrics: &RiskMetrics) -> Result<f64> {
        let mut score = 0.0;

        // Leverage risk
        if risk_metrics.leverage > 2.0 {
            score += (risk_metrics.leverage - 2.0) * 20.0;
        }

        // Concentration risk
        if risk_metrics.concentration > 0.2 {
            score += (risk_metrics.concentration - 0.2) * 100.0;
        }

        // Loss risk
        if risk_metrics.current_daily_loss < 0.0 {
            score += (-risk_metrics.current_daily_loss) * 200.0;
        }

        // Margin usage risk
        if risk_metrics.margin_usage > 0.8 {
            score += (risk_metrics.margin_usage - 0.8) * 100.0;
        }

        Ok(score.max(0.0).min(100.0))
    }

    /// Update volatility cache for risk calculations
    pub async fn update_volatility(&self, symbol: &str, volatility: f64) -> Result<()> {
        let mut vol_cache = self.volatility_cache.write().await;
        vol_cache.insert(symbol.to_string(), volatility);
        Ok(())
    }

    /// Update position limits
    pub async fn update_position_limits(&self, symbol: &str, max_size: f64, max_value: f64) -> Result<()> {
        let mut limits = self.position_limits.write().await;
        limits.insert(symbol.to_string(), PositionLimit {
            symbol: symbol.to_string(),
            max_size,
            max_value,
            current_size: 0.0,
            current_value: 0.0,
            last_updated: Utc::now(),
        });
        Ok(())
    }

    /// Get current risk metrics
    pub async fn get_risk_metrics(&self) -> RiskMetrics {
        self.risk_metrics_cache.read().await.clone()
    }

    /// Emergency risk check - returns true if trading should be halted
    pub async fn emergency_risk_check(&self, portfolio_metrics: &PortfolioMetrics) -> Result<bool> {
        // Check for emergency stop conditions
        
        // Catastrophic loss
        if portfolio_metrics.max_drawdown < -self.config.max_portfolio_loss * 2.0 {
            tracing::error!("Emergency stop: Catastrophic loss detected");
            return Ok(true);
        }

        // Extreme volatility
        if portfolio_metrics.volatility > 0.1 { // 10% daily volatility
            tracing::warn!("Emergency stop: Extreme volatility detected");
            return Ok(true);
        }

        // System error conditions would be checked here

        Ok(false)
    }
}

impl VaRCalculator {
    /// Create new VaR calculator
    pub fn new(confidence_level: f64, lookback_period: usize, monte_carlo_simulations: usize) -> Self {
        Self {
            confidence_level,
            lookback_period,
            monte_carlo_simulations,
        }
    }

    /// Calculate Value at Risk using historical simulation
    pub fn calculate_historical_var(&self, returns: &[f64]) -> Result<f64> {
        if returns.len() < self.lookback_period {
            return Err(TengriError::Risk("Insufficient data for VaR calculation".to_string()));
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - self.confidence_level) * sorted_returns.len() as f64).floor() as usize;
        Ok(sorted_returns[index])
    }

    /// Calculate parametric VaR assuming normal distribution
    pub fn calculate_parametric_var(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Err(TengriError::Risk("No returns data for VaR calculation".to_string()));
        }

        let mean = returns.mean();
        let std_dev = returns.std_dev();
        
        // Z-score for confidence level (e.g., -1.645 for 95% confidence)
        let z_score = match self.confidence_level {
            0.95 => -1.645,
            0.99 => -2.326,
            0.90 => -1.282,
            _ => -1.645, // Default to 95%
        };

        Ok(mean + z_score * std_dev)
    }

    /// Calculate Monte Carlo VaR
    pub fn calculate_monte_carlo_var(&self, mean_return: f64, volatility: f64) -> Result<f64> {
        use rand::prelude::*;
        use statrs::distribution::{Normal, Continuous};

        let normal = Normal::new(mean_return, volatility)
            .map_err(|e| TengriError::Risk(format!("Failed to create normal distribution: {}", e)))?;

        let mut rng = thread_rng();
        let mut simulated_returns = Vec::with_capacity(self.monte_carlo_simulations);

        for _ in 0..self.monte_carlo_simulations {
            let random_return = normal.sample(&mut rng);
            simulated_returns.push(random_return);
        }

        self.calculate_historical_var(&simulated_returns)
    }
}

impl KellyCriterion {
    /// Calculate Kelly optimal position size
    pub fn calculate_kelly_fraction(win_probability: f64, win_amount: f64, loss_amount: f64) -> f64 {
        if win_amount <= 0.0 || loss_amount <= 0.0 {
            return 0.0;
        }

        let odds = win_amount / loss_amount;
        let kelly = (win_probability * (odds + 1.0) - 1.0) / odds;
        
        // Return conservative Kelly (typically 25% of full Kelly)
        (kelly * 0.25).max(0.0).min(0.1) // Cap at 10% of capital
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;

    #[tokio::test]
    async fn test_risk_manager_creation() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(config).await;
        assert!(risk_manager.is_ok());
    }

    #[test]
    fn test_var_calculation() {
        let calculator = VaRCalculator::new(0.95, 252, 10000);
        let returns = vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03];
        let var = calculator.calculate_historical_var(&returns);
        assert!(var.is_ok());
    }

    #[test]
    fn test_kelly_criterion() {
        let kelly_fraction = KellyCriterion::calculate_kelly_fraction(0.6, 2.0, 1.0);
        assert!(kelly_fraction > 0.0);
        assert!(kelly_fraction <= 0.1);
    }

    #[tokio::test]
    async fn test_position_size_calculation() {
        let config = RiskConfig::default();
        let risk_manager = RiskManager::new(config).await.unwrap();
        
        let signal = crate::types::TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: "BTCUSDT".to_string(),
            signal_type: crate::types::SignalType::Buy,
            strength: 0.8,
            confidence: 0.9,
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: HashMap::new(),
            expires_at: None,
        };

        let position_size = risk_manager.calculate_position_size(&signal).await;
        assert!(position_size.is_ok());
        assert!(position_size.unwrap() > 0.0);
    }
}