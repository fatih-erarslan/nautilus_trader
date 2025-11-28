//! Macro researcher agent for macroeconomic analysis.
//!
//! Operates in the slow path to analyze macroeconomic indicators,
//! geopolitical risks, and their impact on portfolio positioning.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, RiskDecision, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

/// Configuration for the macro researcher agent.
#[derive(Debug, Clone)]
pub struct MacroResearcherConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Economic indicators to track.
    pub tracked_indicators: Vec<String>,
    /// Regime detection sensitivity.
    pub regime_sensitivity: f64,
    /// Lookback period for trend analysis (months).
    pub lookback_months: u32,
}

impl Default for MacroResearcherConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "macro_researcher_agent".to_string(),
                enabled: true,
                priority: 5,
                max_latency_us: 100_000, // 100ms (slow path)
                verbose: false,
            },
            tracked_indicators: vec![
                "GDP".to_string(),
                "CPI".to_string(),
                "UNEMPLOYMENT".to_string(),
                "FED_FUNDS_RATE".to_string(),
                "YIELD_CURVE".to_string(),
                "PMI".to_string(),
            ],
            regime_sensitivity: 0.7,
            lookback_months: 12,
        }
    }
}

/// Economic indicator reading.
#[derive(Debug, Clone)]
pub struct EconomicIndicator {
    /// Indicator name.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Previous value.
    pub previous_value: f64,
    /// Year-over-year change.
    pub yoy_change: f64,
    /// Trend direction.
    pub trend: Trend,
    /// Reading timestamp.
    pub as_of: Timestamp,
}

/// Trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// Strongly increasing.
    StronglyUp,
    /// Moderately increasing.
    Up,
    /// Flat/stable.
    Flat,
    /// Moderately decreasing.
    Down,
    /// Strongly decreasing.
    StronglyDown,
}

/// Macroeconomic regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroRegime {
    /// Expansion with low inflation.
    GoldilockExpansion,
    /// Expansion with high inflation.
    InflationaryExpansion,
    /// Slowing growth.
    Slowdown,
    /// Recession.
    Recession,
    /// Recovery from recession.
    Recovery,
    /// Stagflation.
    Stagflation,
}

/// Macro research insight.
#[derive(Debug, Clone)]
pub struct MacroInsight {
    /// Insight category.
    pub category: InsightCategory,
    /// Title.
    pub title: String,
    /// Detailed analysis.
    pub analysis: String,
    /// Confidence level.
    pub confidence: f64,
    /// Recommended portfolio adjustment.
    pub recommendation: PortfolioAdjustment,
    /// Generated timestamp.
    pub generated_at: Timestamp,
}

/// Category of macro insight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightCategory {
    /// Monetary policy insight.
    MonetaryPolicy,
    /// Fiscal policy insight.
    FiscalPolicy,
    /// Growth outlook.
    GrowthOutlook,
    /// Inflation outlook.
    InflationOutlook,
    /// Geopolitical risk.
    GeopoliticalRisk,
    /// Sector rotation.
    SectorRotation,
}

/// Portfolio adjustment recommendation.
#[derive(Debug, Clone)]
pub struct PortfolioAdjustment {
    /// Risk level adjustment.
    pub risk_adjustment: f64,
    /// Equity allocation adjustment.
    pub equity_adjustment: f64,
    /// Fixed income duration adjustment.
    pub duration_adjustment: f64,
    /// Commodity allocation adjustment.
    pub commodity_adjustment: f64,
    /// Cash allocation adjustment.
    pub cash_adjustment: f64,
}

/// Macro researcher agent.
#[derive(Debug)]
pub struct MacroResearcherAgent {
    config: MacroResearcherConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Current economic indicators.
    indicators: RwLock<HashMap<String, EconomicIndicator>>,
    /// Current macro regime assessment.
    current_regime: RwLock<MacroRegime>,
    /// Research insights.
    insights: RwLock<Vec<MacroInsight>>,
    /// Indicator history.
    indicator_history: RwLock<HashMap<String, Vec<(Timestamp, f64)>>>,
}

impl MacroResearcherAgent {
    /// Create a new macro researcher agent.
    pub fn new(config: MacroResearcherConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            indicators: RwLock::new(HashMap::new()),
            current_regime: RwLock::new(MacroRegime::GoldilockExpansion),
            insights: RwLock::new(Vec::new()),
            indicator_history: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MacroResearcherConfig::default())
    }

    /// Update an economic indicator.
    pub fn update_indicator(&self, indicator: EconomicIndicator) {
        let name = indicator.name.clone();
        let value = indicator.value;
        let timestamp = indicator.as_of;

        // Store current indicator
        self.indicators.write().insert(name.clone(), indicator);

        // Update history
        let mut history = self.indicator_history.write();
        history.entry(name).or_default().push((timestamp, value));
    }

    /// Get current macro regime.
    pub fn get_regime(&self) -> MacroRegime {
        *self.current_regime.read()
    }

    /// Get all research insights.
    pub fn get_insights(&self) -> Vec<MacroInsight> {
        self.insights.read().clone()
    }

    /// Get current indicators.
    pub fn get_indicators(&self) -> HashMap<String, EconomicIndicator> {
        self.indicators.read().clone()
    }

    /// Assess macro regime based on indicators.
    fn assess_regime(&self) -> MacroRegime {
        let indicators = self.indicators.read();

        // Get key indicators
        let gdp_growth = indicators.get("GDP").map(|i| i.yoy_change).unwrap_or(0.0);
        let inflation = indicators.get("CPI").map(|i| i.yoy_change).unwrap_or(0.0);
        let unemployment = indicators.get("UNEMPLOYMENT").map(|i| i.value).unwrap_or(5.0);
        let yield_curve = indicators.get("YIELD_CURVE").map(|i| i.value).unwrap_or(0.5);

        // Simple regime classification logic
        if gdp_growth > 2.0 && inflation < 3.0 && unemployment < 5.0 {
            MacroRegime::GoldilockExpansion
        } else if gdp_growth > 2.0 && inflation >= 3.0 {
            MacroRegime::InflationaryExpansion
        } else if gdp_growth < 0.0 && inflation > 4.0 {
            MacroRegime::Stagflation
        } else if gdp_growth < 0.0 {
            MacroRegime::Recession
        } else if gdp_growth > 0.0 && gdp_growth < 2.0 && yield_curve < 0.0 {
            MacroRegime::Slowdown
        } else if unemployment > 6.0 && gdp_growth > 0.0 {
            MacroRegime::Recovery
        } else {
            MacroRegime::Slowdown
        }
    }

    /// Generate portfolio adjustment based on regime.
    fn generate_adjustment(&self, regime: MacroRegime) -> PortfolioAdjustment {
        match regime {
            MacroRegime::GoldilockExpansion => PortfolioAdjustment {
                risk_adjustment: 0.1,
                equity_adjustment: 0.05,
                duration_adjustment: 0.0,
                commodity_adjustment: 0.0,
                cash_adjustment: -0.05,
            },
            MacroRegime::InflationaryExpansion => PortfolioAdjustment {
                risk_adjustment: 0.0,
                equity_adjustment: 0.0,
                duration_adjustment: -0.1,
                commodity_adjustment: 0.05,
                cash_adjustment: 0.0,
            },
            MacroRegime::Slowdown => PortfolioAdjustment {
                risk_adjustment: -0.05,
                equity_adjustment: -0.05,
                duration_adjustment: 0.05,
                commodity_adjustment: -0.02,
                cash_adjustment: 0.05,
            },
            MacroRegime::Recession => PortfolioAdjustment {
                risk_adjustment: -0.15,
                equity_adjustment: -0.15,
                duration_adjustment: 0.1,
                commodity_adjustment: -0.05,
                cash_adjustment: 0.1,
            },
            MacroRegime::Recovery => PortfolioAdjustment {
                risk_adjustment: 0.05,
                equity_adjustment: 0.1,
                duration_adjustment: -0.05,
                commodity_adjustment: 0.02,
                cash_adjustment: -0.05,
            },
            MacroRegime::Stagflation => PortfolioAdjustment {
                risk_adjustment: -0.1,
                equity_adjustment: -0.1,
                duration_adjustment: -0.1,
                commodity_adjustment: 0.1,
                cash_adjustment: 0.1,
            },
        }
    }

    /// Generate insights based on current indicators and regime.
    fn generate_insights(&self, regime: MacroRegime) {
        let mut insights = self.insights.write();
        let adjustment = self.generate_adjustment(regime);

        // Clear old insights
        insights.clear();

        // Generate regime-based insight
        let regime_insight = MacroInsight {
            category: InsightCategory::GrowthOutlook,
            title: format!("Macro Regime: {:?}", regime),
            analysis: self.regime_analysis(regime),
            confidence: self.config.regime_sensitivity,
            recommendation: adjustment.clone(),
            generated_at: Timestamp::now(),
        };
        insights.push(regime_insight);

        // Generate inflation outlook if CPI data available
        let indicators = self.indicators.read();
        if let Some(cpi) = indicators.get("CPI") {
            let inflation_insight = MacroInsight {
                category: InsightCategory::InflationOutlook,
                title: format!("Inflation Trend: {:?}", cpi.trend),
                analysis: format!(
                    "CPI YoY change: {:.1}%, trend is {:?}. {}",
                    cpi.yoy_change,
                    cpi.trend,
                    if cpi.yoy_change > 3.0 {
                        "Elevated inflation may pressure central bank to tighten."
                    } else if cpi.yoy_change < 2.0 {
                        "Subdued inflation provides room for accommodative policy."
                    } else {
                        "Inflation within target range."
                    }
                ),
                confidence: 0.8,
                recommendation: adjustment,
                generated_at: Timestamp::now(),
            };
            insights.push(inflation_insight);
        }
    }

    /// Generate regime analysis text.
    fn regime_analysis(&self, regime: MacroRegime) -> String {
        match regime {
            MacroRegime::GoldilockExpansion => {
                "Economy in ideal state with solid growth and contained inflation. \
                 Favorable environment for risk assets."
                    .to_string()
            }
            MacroRegime::InflationaryExpansion => {
                "Strong growth but elevated inflation pressures. \
                 Consider inflation hedges and shorter duration bonds."
                    .to_string()
            }
            MacroRegime::Slowdown => {
                "Growth decelerating from peak. Reduce cyclical exposure \
                 and increase defensive positioning."
                    .to_string()
            }
            MacroRegime::Recession => {
                "Economic contraction underway. Prioritize capital preservation \
                 with defensive assets and high quality bonds."
                    .to_string()
            }
            MacroRegime::Recovery => {
                "Economy emerging from downturn. Gradually increase risk exposure \
                 to capture recovery gains."
                    .to_string()
            }
            MacroRegime::Stagflation => {
                "Challenging environment with weak growth and high inflation. \
                 Focus on real assets and maintain elevated cash levels."
                    .to_string()
            }
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

impl Agent for MacroResearcherAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, _portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Assess macro regime
        let regime = self.assess_regime();
        *self.current_regime.write() = regime;

        // Generate insights
        self.generate_insights(regime);

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
    fn test_macro_researcher_creation() {
        let agent = MacroResearcherAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_indicator_update() {
        let agent = MacroResearcherAgent::with_defaults();

        agent.update_indicator(EconomicIndicator {
            name: "GDP".to_string(),
            value: 2.5,
            previous_value: 2.3,
            yoy_change: 2.5,
            trend: Trend::Up,
            as_of: Timestamp::now(),
        });

        let indicators = agent.get_indicators();
        assert!(indicators.contains_key("GDP"));
    }

    #[test]
    fn test_regime_assessment() {
        let agent = MacroResearcherAgent::with_defaults();

        // Set indicators for goldilocks scenario
        agent.update_indicator(EconomicIndicator {
            name: "GDP".to_string(),
            value: 3.0,
            previous_value: 2.8,
            yoy_change: 3.0,
            trend: Trend::Up,
            as_of: Timestamp::now(),
        });
        agent.update_indicator(EconomicIndicator {
            name: "CPI".to_string(),
            value: 2.0,
            previous_value: 1.9,
            yoy_change: 2.0,
            trend: Trend::Flat,
            as_of: Timestamp::now(),
        });
        agent.update_indicator(EconomicIndicator {
            name: "UNEMPLOYMENT".to_string(),
            value: 4.0,
            previous_value: 4.1,
            yoy_change: -0.1,
            trend: Trend::Down,
            as_of: Timestamp::now(),
        });

        let portfolio = Portfolio::default();
        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        assert_eq!(agent.get_regime(), MacroRegime::GoldilockExpansion);
    }

    #[test]
    fn test_insights_generation() {
        let agent = MacroResearcherAgent::with_defaults();

        agent.update_indicator(EconomicIndicator {
            name: "GDP".to_string(),
            value: 3.0,
            previous_value: 2.8,
            yoy_change: 3.0,
            trend: Trend::Up,
            as_of: Timestamp::now(),
        });

        let portfolio = Portfolio::default();
        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let insights = agent.get_insights();
        assert!(!insights.is_empty());
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = MacroResearcherAgent::with_defaults();

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
