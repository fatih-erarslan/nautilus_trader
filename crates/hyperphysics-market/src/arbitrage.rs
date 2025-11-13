//! Multi-exchange arbitrage detection and analysis
//!
//! This module provides tools for detecting arbitrage opportunities across
//! multiple cryptocurrency exchanges, including:
//! - Cross-exchange arbitrage (price differences between exchanges)
//! - Triangular arbitrage (price inefficiencies within an exchange)
//! - Statistical arbitrage (mean reversion opportunities)

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::MarketResult;
use crate::providers::MarketDataProvider;

/// Type of arbitrage opportunity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArbitrageType {
    /// Cross-exchange arbitrage (buy on one exchange, sell on another)
    CrossExchange,
    /// Triangular arbitrage (trading cycle within one exchange)
    Triangular,
    /// Statistical arbitrage (mean reversion)
    Statistical,
}

/// Detected arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    /// Type of arbitrage
    pub arbitrage_type: ArbitrageType,

    /// Timestamp when detected
    pub timestamp: DateTime<Utc>,

    /// Expected profit percentage (before fees)
    pub profit_pct: f64,

    /// Trading symbols involved
    pub symbols: Vec<String>,

    /// Exchanges involved (for cross-exchange arbitrage)
    pub exchanges: Vec<String>,

    /// Suggested trade actions
    pub actions: Vec<TradeAction>,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Estimated execution window in seconds
    pub window_secs: u64,
}

/// Suggested trade action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeAction {
    /// Exchange to execute on
    pub exchange: String,

    /// Trading symbol
    pub symbol: String,

    /// Buy or sell
    pub side: TradeSide,

    /// Suggested price
    pub price: f64,

    /// Suggested quantity
    pub quantity: f64,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Multi-exchange arbitrage detector
pub struct ArbitrageDetector {
    /// Market data providers by name
    providers: HashMap<String, Arc<dyn MarketDataProvider>>,

    /// Latest prices cache
    prices: Arc<RwLock<HashMap<(String, String), f64>>>, // (exchange, symbol) -> price

    /// Minimum profit threshold percentage
    min_profit_pct: f64,

    /// Maximum latency tolerance in milliseconds
    max_latency_ms: u64,

    /// Detected opportunities
    opportunities: Arc<RwLock<Vec<ArbitrageOpportunity>>>,
}

impl ArbitrageDetector {
    /// Create new arbitrage detector
    ///
    /// # Arguments
    /// * `min_profit_pct` - Minimum profit percentage to detect (e.g., 0.5 for 0.5%)
    /// * `max_latency_ms` - Maximum acceptable latency in milliseconds
    pub fn new(min_profit_pct: f64, max_latency_ms: u64) -> Self {
        Self {
            providers: HashMap::new(),
            prices: Arc::new(RwLock::new(HashMap::new())),
            min_profit_pct,
            max_latency_ms,
            opportunities: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a market data provider
    pub fn add_provider(&mut self, name: String, provider: Arc<dyn MarketDataProvider>) {
        self.providers.insert(name, provider);
    }

    /// Update price for a symbol on an exchange
    pub async fn update_price(&self, exchange: &str, symbol: &str, price: f64) {
        let mut prices = self.prices.write().await;
        prices.insert((exchange.to_string(), symbol.to_string()), price);
    }

    /// Detect cross-exchange arbitrage opportunities
    ///
    /// Finds price differences for the same symbol across multiple exchanges
    pub async fn detect_cross_exchange(&self, symbol: &str) -> MarketResult<Vec<ArbitrageOpportunity>> {
        let prices = self.prices.read().await;
        let mut opportunities = Vec::new();

        // Collect prices for this symbol across all exchanges
        let mut exchange_prices: Vec<(String, f64)> = prices
            .iter()
            .filter_map(|((ex, sym), price)| {
                if sym == symbol {
                    Some((ex.clone(), *price))
                } else {
                    None
                }
            })
            .collect();

        // Need at least 2 exchanges
        if exchange_prices.len() < 2 {
            return Ok(opportunities);
        }

        // Sort by price
        exchange_prices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Find opportunities: buy low, sell high
        let (low_exchange, low_price) = &exchange_prices[0];
        let (high_exchange, high_price) = &exchange_prices[exchange_prices.len() - 1];

        let profit_pct = ((high_price - low_price) / low_price) * 100.0;

        if profit_pct >= self.min_profit_pct {
            opportunities.push(ArbitrageOpportunity {
                arbitrage_type: ArbitrageType::CrossExchange,
                timestamp: Utc::now(),
                profit_pct,
                symbols: vec![symbol.to_string()],
                exchanges: vec![low_exchange.clone(), high_exchange.clone()],
                actions: vec![
                    TradeAction {
                        exchange: low_exchange.clone(),
                        symbol: symbol.to_string(),
                        side: TradeSide::Buy,
                        price: *low_price,
                        quantity: 1.0, // Would calculate based on available liquidity
                    },
                    TradeAction {
                        exchange: high_exchange.clone(),
                        symbol: symbol.to_string(),
                        side: TradeSide::Sell,
                        price: *high_price,
                        quantity: 1.0,
                    },
                ],
                confidence: 0.8, // Would calculate based on order book depth
                window_secs: 5, // Typical execution window
            });
        }

        Ok(opportunities)
    }

    /// Detect triangular arbitrage opportunities
    ///
    /// Finds price inefficiencies in trading cycles (e.g., BTC->ETH->USDT->BTC)
    pub async fn detect_triangular(
        &self,
        exchange: &str,
        base: &str,
        quote1: &str,
        quote2: &str,
    ) -> MarketResult<Vec<ArbitrageOpportunity>> {
        let prices = self.prices.read().await;
        let mut opportunities = Vec::new();

        // Build symbol pairs
        let pair1 = format!("{}-{}", base, quote1); // BTC-ETH
        let pair2 = format!("{}-{}", quote1, quote2); // ETH-USDT
        let pair3 = format!("{}-{}", base, quote2); // BTC-USDT

        // Get prices
        let price1 = prices.get(&(exchange.to_string(), pair1.clone()));
        let price2 = prices.get(&(exchange.to_string(), pair2.clone()));
        let price3 = prices.get(&(exchange.to_string(), pair3.clone()));

        if let (Some(&p1), Some(&p2), Some(&p3)) = (price1, price2, price3) {
            // Calculate cycle profit
            // Start with 1 unit of base currency
            // Trade: base -> quote1 -> quote2 -> base
            let amount_quote1 = 1.0 / p1; // BTC -> ETH
            let amount_quote2 = amount_quote1 * p2; // ETH -> USDT
            let amount_base = amount_quote2 / p3; // USDT -> BTC

            let profit_pct = ((amount_base - 1.0) / 1.0) * 100.0;

            if profit_pct >= self.min_profit_pct {
                opportunities.push(ArbitrageOpportunity {
                    arbitrage_type: ArbitrageType::Triangular,
                    timestamp: Utc::now(),
                    profit_pct,
                    symbols: vec![pair1.clone(), pair2.clone(), pair3.clone()],
                    exchanges: vec![exchange.to_string()],
                    actions: vec![
                        TradeAction {
                            exchange: exchange.to_string(),
                            symbol: pair1,
                            side: TradeSide::Sell,
                            price: p1,
                            quantity: 1.0,
                        },
                        TradeAction {
                            exchange: exchange.to_string(),
                            symbol: pair2,
                            side: TradeSide::Buy,
                            price: p2,
                            quantity: amount_quote1,
                        },
                        TradeAction {
                            exchange: exchange.to_string(),
                            symbol: pair3,
                            side: TradeSide::Buy,
                            price: p3,
                            quantity: amount_quote2,
                        },
                    ],
                    confidence: 0.7,
                    window_secs: 2, // Triangular arb requires fast execution
                });
            }
        }

        Ok(opportunities)
    }

    /// Scan all providers for arbitrage opportunities
    pub async fn scan_all(&self, symbols: &[String]) -> MarketResult<Vec<ArbitrageOpportunity>> {
        let mut all_opportunities = Vec::new();

        // Update prices from all providers
        for (exchange, provider) in &self.providers {
            for symbol in symbols {
                if let Ok(bar) = provider.fetch_latest_bar(symbol).await {
                    self.update_price(exchange, symbol, bar.close).await;
                }
            }
        }

        // Detect cross-exchange opportunities
        for symbol in symbols {
            if let Ok(mut opps) = self.detect_cross_exchange(symbol).await {
                all_opportunities.append(&mut opps);
            }
        }

        // Store detected opportunities
        let mut opportunities = self.opportunities.write().await;
        *opportunities = all_opportunities.clone();

        Ok(all_opportunities)
    }

    /// Get all detected opportunities
    pub async fn get_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        self.opportunities.read().await.clone()
    }

    /// Clear detected opportunities
    pub async fn clear_opportunities(&self) {
        self.opportunities.write().await.clear();
    }

    /// Get current price spread for a symbol
    pub async fn get_spread(&self, symbol: &str) -> Option<(f64, f64, f64)> {
        let prices = self.prices.read().await;

        let mut symbol_prices: Vec<f64> = prices
            .iter()
            .filter_map(|((_, sym), price)| {
                if sym == symbol {
                    Some(*price)
                } else {
                    None
                }
            })
            .collect();

        if symbol_prices.is_empty() {
            return None;
        }

        symbol_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = symbol_prices[0];
        let max = symbol_prices[symbol_prices.len() - 1];
        let spread_pct = ((max - min) / min) * 100.0;

        Some((min, max, spread_pct))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detector_creation() {
        let detector = ArbitrageDetector::new(0.5, 100);
        assert_eq!(detector.min_profit_pct, 0.5);
        assert_eq!(detector.max_latency_ms, 100);
    }

    #[tokio::test]
    async fn test_price_update() {
        let detector = ArbitrageDetector::new(0.5, 100);
        detector.update_price("binance", "BTC-USDT", 50000.0).await;

        let prices = detector.prices.read().await;
        assert_eq!(
            prices.get(&("binance".to_string(), "BTC-USDT".to_string())),
            Some(&50000.0)
        );
    }

    #[tokio::test]
    async fn test_cross_exchange_detection() {
        let detector = ArbitrageDetector::new(0.5, 100);

        // Set up price difference
        detector.update_price("binance", "BTC-USDT", 50000.0).await;
        detector.update_price("okx", "BTC-USDT", 50300.0).await; // 0.6% higher

        let opportunities = detector.detect_cross_exchange("BTC-USDT").await.unwrap();

        assert_eq!(opportunities.len(), 1);
        assert_eq!(opportunities[0].arbitrage_type, ArbitrageType::CrossExchange);
        assert!(opportunities[0].profit_pct >= 0.5);
    }

    #[tokio::test]
    async fn test_spread_calculation() {
        let detector = ArbitrageDetector::new(0.5, 100);

        detector.update_price("binance", "ETH-USDT", 3000.0).await;
        detector.update_price("okx", "ETH-USDT", 3015.0).await;
        detector.update_price("coinbase", "ETH-USDT", 3010.0).await;

        let spread = detector.get_spread("ETH-USDT").await;
        assert!(spread.is_some());

        let (min, max, spread_pct) = spread.unwrap();
        assert_eq!(min, 3000.0);
        assert_eq!(max, 3015.0);
        assert!((spread_pct - 0.5).abs() < 0.01);
    }
}
