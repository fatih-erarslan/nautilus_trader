//! Quantitative researcher agent for strategy analysis.
//!
//! Operates in the slow path to perform sophisticated quantitative
//! analysis, backtesting, and strategy evaluation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

/// Configuration for the quant researcher agent.
#[derive(Debug, Clone)]
pub struct QuantResearcherConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Lookback period for analysis (days).
    pub lookback_days: u32,
    /// Confidence level for statistical tests.
    pub confidence_level: f64,
    /// Minimum Sharpe ratio threshold.
    pub min_sharpe_ratio: f64,
    /// Maximum acceptable drawdown.
    pub max_drawdown: f64,
}

impl Default for QuantResearcherConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "quant_researcher_agent".to_string(),
                enabled: true,
                priority: 5,
                max_latency_us: 100_000, // 100ms (slow path)
                verbose: false,
            },
            lookback_days: 252,
            confidence_level: 0.95,
            min_sharpe_ratio: 1.0,
            max_drawdown: 0.20,
        }
    }
}

/// Strategy performance metrics.
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Strategy identifier.
    pub strategy_id: String,
    /// Annualized return.
    pub annualized_return: f64,
    /// Annualized volatility.
    pub annualized_volatility: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Sortino ratio.
    pub sortino_ratio: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Calmar ratio (return / max drawdown).
    pub calmar_ratio: f64,
    /// Win rate.
    pub win_rate: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Number of trades.
    pub trade_count: u32,
    /// Analysis timestamp.
    pub analyzed_at: Timestamp,
}

/// Factor exposure analysis.
#[derive(Debug, Clone)]
pub struct FactorExposure {
    /// Factor name.
    pub factor: String,
    /// Beta to factor.
    pub beta: f64,
    /// T-statistic.
    pub t_stat: f64,
    /// P-value.
    pub p_value: f64,
    /// R-squared contribution.
    pub r_squared: f64,
}

/// Research finding/insight.
#[derive(Debug, Clone)]
pub struct ResearchFinding {
    /// Finding type.
    pub finding_type: FindingType,
    /// Symbol (if applicable).
    pub symbol: Option<Symbol>,
    /// Description.
    pub description: String,
    /// Confidence level.
    pub confidence: f64,
    /// Actionable recommendation.
    pub recommendation: Option<String>,
    /// Discovery timestamp.
    pub discovered_at: Timestamp,
}

/// Type of research finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingType {
    /// Alpha opportunity identified.
    AlphaOpportunity,
    /// Risk factor identified.
    RiskFactor,
    /// Correlation change detected.
    CorrelationShift,
    /// Regime change detected.
    RegimeChange,
    /// Strategy degradation.
    StrategyDegradation,
    /// Optimization opportunity.
    OptimizationOpportunity,
}

/// Quantitative researcher agent.
#[derive(Debug)]
pub struct QuantResearcherAgent {
    config: QuantResearcherConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Strategy metrics cache.
    strategy_metrics: RwLock<HashMap<String, StrategyMetrics>>,
    /// Factor exposures.
    factor_exposures: RwLock<Vec<FactorExposure>>,
    /// Research findings.
    findings: RwLock<Vec<ResearchFinding>>,
    /// Return series by symbol.
    return_series: RwLock<HashMap<Symbol, Vec<f64>>>,
}

impl QuantResearcherAgent {
    /// Create a new quant researcher agent.
    pub fn new(config: QuantResearcherConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            strategy_metrics: RwLock::new(HashMap::new()),
            factor_exposures: RwLock::new(Vec::new()),
            findings: RwLock::new(Vec::new()),
            return_series: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(QuantResearcherConfig::default())
    }

    /// Add return observation for a symbol.
    pub fn add_return(&self, symbol: &Symbol, ret: f64) {
        let mut series = self.return_series.write();
        series.entry(symbol.clone()).or_default().push(ret);
    }

    /// Get strategy metrics.
    pub fn get_strategy_metrics(&self, strategy_id: &str) -> Option<StrategyMetrics> {
        self.strategy_metrics.read().get(strategy_id).cloned()
    }

    /// Get all research findings.
    pub fn get_findings(&self) -> Vec<ResearchFinding> {
        self.findings.read().clone()
    }

    /// Get factor exposures.
    pub fn get_factor_exposures(&self) -> Vec<FactorExposure> {
        self.factor_exposures.read().clone()
    }

    /// Analyze a strategy's performance.
    pub fn analyze_strategy(&self, strategy_id: &str, returns: &[f64]) -> StrategyMetrics {
        let n = returns.len() as f64;
        if n < 2.0 {
            return StrategyMetrics {
                strategy_id: strategy_id.to_string(),
                annualized_return: 0.0,
                annualized_volatility: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                calmar_ratio: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                trade_count: 0,
                analyzed_at: Timestamp::now(),
            };
        }

        // Calculate basic statistics
        let mean_return: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        // Annualize (assuming daily returns)
        let annualized_return = mean_return * 252.0;
        let annualized_volatility = std_dev * (252.0_f64).sqrt();

        // Sharpe ratio (assuming 5% risk-free rate)
        let risk_free_daily = 0.05 / 252.0;
        let sharpe_ratio = if annualized_volatility > 0.0 {
            (annualized_return - 0.05) / annualized_volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < risk_free_daily)
            .map(|&r| (r - risk_free_daily).powi(2))
            .collect();
        let downside_dev = if !downside_returns.is_empty() {
            (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt() * (252.0_f64).sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_dev > 0.0 {
            (annualized_return - 0.05) / downside_dev
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak: f64 = 1.0;
        let mut max_dd: f64 = 0.0;
        let mut value: f64 = 1.0;
        for &ret in returns {
            value *= 1.0 + ret;
            peak = peak.max(value);
            let dd = (peak - value) / peak;
            max_dd = max_dd.max(dd);
        }

        // Calmar ratio
        let calmar_ratio = if max_dd > 0.0 {
            annualized_return / max_dd
        } else {
            0.0
        };

        // Win rate and profit factor
        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let win_rate = wins.len() as f64 / n;
        let profit_factor = if !losses.is_empty() && losses.iter().sum::<f64>().abs() > 0.0 {
            wins.iter().sum::<f64>() / losses.iter().sum::<f64>().abs()
        } else {
            0.0
        };

        let metrics = StrategyMetrics {
            strategy_id: strategy_id.to_string(),
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: max_dd,
            calmar_ratio,
            win_rate,
            profit_factor,
            trade_count: returns.len() as u32,
            analyzed_at: Timestamp::now(),
        };

        // Store metrics
        self.strategy_metrics.write().insert(strategy_id.to_string(), metrics.clone());

        // Generate findings based on analysis
        self.generate_findings(&metrics);

        metrics
    }

    /// Generate research findings from metrics.
    fn generate_findings(&self, metrics: &StrategyMetrics) {
        let mut findings = self.findings.write();

        // Check for strategy degradation
        if metrics.sharpe_ratio < self.config.min_sharpe_ratio {
            findings.push(ResearchFinding {
                finding_type: FindingType::StrategyDegradation,
                symbol: None,
                description: format!(
                    "Strategy {} has Sharpe ratio {:.2} below threshold {:.2}",
                    metrics.strategy_id, metrics.sharpe_ratio, self.config.min_sharpe_ratio
                ),
                confidence: 0.95,
                recommendation: Some("Consider reducing position sizes or reviewing strategy parameters".to_string()),
                discovered_at: Timestamp::now(),
            });
        }

        // Check for excessive drawdown
        if metrics.max_drawdown > self.config.max_drawdown {
            findings.push(ResearchFinding {
                finding_type: FindingType::RiskFactor,
                symbol: None,
                description: format!(
                    "Strategy {} has max drawdown {:.1}% exceeding limit {:.1}%",
                    metrics.strategy_id, metrics.max_drawdown * 100.0, self.config.max_drawdown * 100.0
                ),
                confidence: 1.0,
                recommendation: Some("Implement tighter stop losses or reduce leverage".to_string()),
                discovered_at: Timestamp::now(),
            });
        }

        // Identify alpha opportunity if metrics are strong
        if metrics.sharpe_ratio > 2.0 && metrics.calmar_ratio > 1.0 {
            findings.push(ResearchFinding {
                finding_type: FindingType::AlphaOpportunity,
                symbol: None,
                description: format!(
                    "Strategy {} shows strong risk-adjusted returns (Sharpe: {:.2}, Calmar: {:.2})",
                    metrics.strategy_id, metrics.sharpe_ratio, metrics.calmar_ratio
                ),
                confidence: 0.8,
                recommendation: Some("Consider increasing allocation to this strategy".to_string()),
                discovered_at: Timestamp::now(),
            });
        }
    }

    /// Convert u8 to AgentStatus.
    fn status_from_u8(value: u8) -> AgentStatus {
        match value {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            4 => AgentStatus::ShuttingDown,
            _ => AgentStatus::Error,
        }
    }
}

impl Agent for QuantResearcherAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, _portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Analyze each symbol's return series
        let return_series = self.return_series.read();
        for (symbol, returns) in return_series.iter() {
            if returns.len() >= 20 {
                self.analyze_strategy(&symbol.as_str(), returns);
            }
        }

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency_ns);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

        Ok(None)
    }

    fn start(&self) -> Result<()> {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_researcher_creation() {
        let agent = QuantResearcherAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_strategy_analysis() {
        let agent = QuantResearcherAgent::with_defaults();

        // Generate sample returns (daily)
        let returns: Vec<f64> = (0..252).map(|i| {
            0.0005 + 0.01 * ((i as f64 * 0.1).sin())
        }).collect();

        let metrics = agent.analyze_strategy("test_strategy", &returns);

        assert!(!metrics.strategy_id.is_empty());
        assert!(metrics.trade_count > 0);
    }

    #[test]
    fn test_findings_generation() {
        let agent = QuantResearcherAgent::with_defaults();

        // Generate poor returns to trigger findings
        let returns: Vec<f64> = vec![
            -0.05, -0.03, 0.01, -0.04, -0.02, 0.005, -0.03, -0.02, 0.01, -0.01,
            -0.02, -0.01, 0.005, -0.015, -0.02, 0.01, -0.025, -0.01, 0.005, -0.02,
        ];

        agent.analyze_strategy("poor_strategy", &returns);

        let findings = agent.get_findings();
        assert!(!findings.is_empty());
    }

    #[test]
    fn test_return_series() {
        let agent = QuantResearcherAgent::with_defaults();
        let symbol = Symbol::new("AAPL");

        for i in 0..30 {
            agent.add_return(&symbol, 0.001 * (i as f64));
        }

        let series = agent.return_series.read();
        assert_eq!(series.get(&symbol).map(|v| v.len()), Some(30));
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = QuantResearcherAgent::with_defaults();

        agent.start().unwrap();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.pause();
        assert_eq!(agent.status(), AgentStatus::Paused);

        agent.resume();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.stop().unwrap();
        assert_eq!(agent.status(), AgentStatus::ShuttingDown);
    }
}
