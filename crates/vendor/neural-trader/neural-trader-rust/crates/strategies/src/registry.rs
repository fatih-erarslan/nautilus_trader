//! Strategy registry module providing metadata access for MCP tools
//!
//! This module provides a simpler API for accessing strategy metadata,
//! complementing the full strategy implementations in the main crate.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple strategy metadata for MCP tool compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetadataSimple {
    pub name: String,
    pub description: String,
    pub sharpe_ratio: f64,
    pub status: String,
    pub gpu_capable: bool,
    pub risk_level: String,
}

/// Registry of strategy metadata
pub struct StrategyRegistry {
    strategies: HashMap<String, StrategyMetadataSimple>,
}

impl StrategyRegistry {
    /// Create a new registry with all available strategies
    pub fn new() -> Self {
        let mut strategies = HashMap::new();

        // Mirror Trading - Highest Sharpe (6.01)
        strategies.insert("mirror_trading".to_string(), StrategyMetadataSimple {
            name: "mirror_trading".to_string(),
            description: "High-frequency mirror trading with neural pattern matching".to_string(),
            sharpe_ratio: 6.01,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "high".to_string(),
        });

        // Statistical Arbitrage (3.89)
        strategies.insert("statistical_arbitrage".to_string(), StrategyMetadataSimple {
            name: "statistical_arbitrage".to_string(),
            description: "High-frequency statistical arbitrage using cross-asset correlations".to_string(),
            sharpe_ratio: 3.89,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "medium".to_string(),
        });

        // Adaptive Multi-Strategy (3.42)
        strategies.insert("adaptive".to_string(), StrategyMetadataSimple {
            name: "adaptive".to_string(),
            description: "Machine learning-based strategy allocation across market regimes".to_string(),
            sharpe_ratio: 3.42,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "medium".to_string(),
        });

        // Momentum Trading (2.84)
        strategies.insert("momentum_trading".to_string(), StrategyMetadataSimple {
            name: "momentum_trading".to_string(),
            description: "Momentum-based trading with technical indicators".to_string(),
            sharpe_ratio: 2.84,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "medium".to_string(),
        });

        // Breakout Strategy (2.68)
        strategies.insert("breakout".to_string(), StrategyMetadataSimple {
            name: "breakout".to_string(),
            description: "Trade breakouts from support/resistance with volume confirmation".to_string(),
            sharpe_ratio: 2.68,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "high".to_string(),
        });

        // Options Delta-Neutral (2.57)
        strategies.insert("options_delta_neutral".to_string(), StrategyMetadataSimple {
            name: "options_delta_neutral".to_string(),
            description: "Volatility trading with delta-hedged options positions".to_string(),
            sharpe_ratio: 2.57,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "medium".to_string(),
        });

        // Pairs Trading (2.31)
        strategies.insert("pairs_trading".to_string(), StrategyMetadataSimple {
            name: "pairs_trading".to_string(),
            description: "Trade cointegrated pairs for market-neutral returns".to_string(),
            sharpe_ratio: 2.31,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "low".to_string(),
        });

        // Trend Following (2.15)
        strategies.insert("trend_following".to_string(), StrategyMetadataSimple {
            name: "trend_following".to_string(),
            description: "Follow established trends using moving average crossovers".to_string(),
            sharpe_ratio: 2.15,
            status: "available".to_string(),
            gpu_capable: false,
            risk_level: "medium".to_string(),
        });

        // Mean Reversion (1.95)
        strategies.insert("mean_reversion".to_string(), StrategyMetadataSimple {
            name: "mean_reversion".to_string(),
            description: "Statistical mean reversion strategy".to_string(),
            sharpe_ratio: 1.95,
            status: "available".to_string(),
            gpu_capable: true,
            risk_level: "low".to_string(),
        });

        Self { strategies }
    }

    /// Get a strategy by name
    pub fn get(&self, name: &str) -> Option<&StrategyMetadataSimple> {
        self.strategies.get(name)
    }

    /// List all available strategies
    pub fn list_all(&self) -> Vec<&StrategyMetadataSimple> {
        self.strategies.values().collect()
    }

    /// Get strategies by risk level
    pub fn get_by_risk_level(&self, risk_level: &str) -> Vec<&StrategyMetadataSimple> {
        self.strategies
            .values()
            .filter(|s| s.risk_level == risk_level)
            .collect()
    }

    /// Get GPU-capable strategies
    pub fn get_gpu_capable(&self) -> Vec<&StrategyMetadataSimple> {
        self.strategies
            .values()
            .filter(|s| s.gpu_capable)
            .collect()
    }

    /// Get strategies sorted by Sharpe ratio
    pub fn get_by_sharpe_ratio(&self, descending: bool) -> Vec<&StrategyMetadataSimple> {
        let mut strategies: Vec<&StrategyMetadataSimple> = self.strategies.values().collect();

        if descending {
            strategies.sort_by(|a, b| b.sharpe_ratio.partial_cmp(&a.sharpe_ratio).unwrap());
        } else {
            strategies.sort_by(|a, b| a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap());
        }

        strategies
    }

    /// Check if a strategy exists
    pub fn contains(&self, name: &str) -> bool {
        self.strategies.contains_key(name)
    }

    /// Get strategy count
    pub fn count(&self) -> usize {
        self.strategies.len()
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = StrategyRegistry::new();
        assert_eq!(registry.count(), 9);
    }

    #[test]
    fn test_get_strategy() {
        let registry = StrategyRegistry::new();
        let mirror = registry.get("mirror_trading");

        assert!(mirror.is_some());
        assert_eq!(mirror.unwrap().name, "mirror_trading");
        assert_eq!(mirror.unwrap().sharpe_ratio, 6.01);
    }

    #[test]
    fn test_sharpe_ratio_sorting() {
        let registry = StrategyRegistry::new();
        let sorted = registry.get_by_sharpe_ratio(true);

        assert!(sorted.len() > 0);
        assert_eq!(sorted[0].name, "mirror_trading"); // Highest Sharpe ratio
        assert_eq!(sorted[0].sharpe_ratio, 6.01);
    }

    #[test]
    fn test_gpu_capable_strategies() {
        let registry = StrategyRegistry::new();
        let gpu_strategies = registry.get_gpu_capable();

        assert!(gpu_strategies.len() >= 7); // Most strategies support GPU
        assert!(gpu_strategies.iter().all(|s| s.gpu_capable));
    }

    #[test]
    fn test_risk_level_filtering() {
        let registry = StrategyRegistry::new();
        let low_risk = registry.get_by_risk_level("low");
        let medium_risk = registry.get_by_risk_level("medium");
        let high_risk = registry.get_by_risk_level("high");

        assert!(low_risk.len() >= 2); // mean_reversion, pairs_trading
        assert!(medium_risk.len() >= 4);
        assert!(high_risk.len() >= 2); // mirror_trading, breakout
    }
}
