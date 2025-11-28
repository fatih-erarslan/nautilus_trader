//! Arbitrage detection agent for cross-market opportunities.
//!
//! Operates in the medium path (<800μs) to identify and evaluate
//! arbitrage opportunities across multiple venues and instruments.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, Price, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

/// Configuration for the arbitrage detection agent.
#[derive(Debug, Clone)]
pub struct ArbitrageConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Minimum spread threshold for arbitrage detection (basis points).
    pub min_spread_bps: f64,
    /// Maximum position size for arbitrage trades.
    pub max_position_size: f64,
    /// Venues to monitor for arbitrage.
    pub venues: Vec<String>,
    /// Execution latency budget in nanoseconds.
    pub latency_budget_ns: u64,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "arbitrage_agent".to_string(),
                enabled: true,
                priority: 2,
                max_latency_us: 800, // 800μs
                verbose: false,
            },
            min_spread_bps: 5.0,
            max_position_size: 100_000.0,
            venues: vec!["venue_a".to_string(), "venue_b".to_string()],
            latency_budget_ns: 500_000,
        }
    }
}

/// Market quote from a specific venue.
#[derive(Debug, Clone)]
pub struct VenueQuote {
    /// Symbol being quoted.
    pub symbol: Symbol,
    /// Venue identifier.
    pub venue: String,
    /// Best bid price.
    pub bid: Price,
    /// Best ask price.
    pub ask: Price,
    /// Quote timestamp.
    pub timestamp: Timestamp,
}

/// Detected arbitrage opportunity.
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    /// Symbol with arbitrage opportunity.
    pub symbol: Symbol,
    /// Buy venue.
    pub buy_venue: String,
    /// Sell venue.
    pub sell_venue: String,
    /// Buy price (ask at buy venue).
    pub buy_price: Price,
    /// Sell price (bid at sell venue).
    pub sell_price: Price,
    /// Spread in basis points.
    pub spread_bps: f64,
    /// Estimated profit.
    pub estimated_profit: f64,
    /// Detection timestamp.
    pub detected_at: Timestamp,
}

/// Arbitrage detection agent.
#[derive(Debug)]
pub struct ArbitrageAgent {
    config: ArbitrageConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Current venue quotes by symbol and venue.
    quotes: RwLock<HashMap<(Symbol, String), VenueQuote>>,
    /// Active arbitrage opportunities.
    opportunities: RwLock<Vec<ArbitrageOpportunity>>,
}

impl ArbitrageAgent {
    /// Create a new arbitrage agent with the given configuration.
    pub fn new(config: ArbitrageConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            quotes: RwLock::new(HashMap::new()),
            opportunities: RwLock::new(Vec::new()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ArbitrageConfig::default())
    }

    /// Update a venue quote.
    pub fn update_quote(&self, quote: VenueQuote) {
        let key = (quote.symbol.clone(), quote.venue.clone());
        self.quotes.write().insert(key, quote);
    }

    /// Get current arbitrage opportunities.
    pub fn get_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        self.opportunities.read().clone()
    }

    /// Scan for arbitrage opportunities across venues.
    fn scan_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        let quotes = self.quotes.read();
        let mut opportunities = Vec::new();

        // Group quotes by symbol
        let mut by_symbol: HashMap<Symbol, Vec<&VenueQuote>> = HashMap::new();
        for quote in quotes.values() {
            by_symbol
                .entry(quote.symbol.clone())
                .or_default()
                .push(quote);
        }

        // Find cross-venue arbitrage
        for (_symbol, symbol_quotes) in by_symbol.iter() {
            if symbol_quotes.len() < 2 {
                continue;
            }

            for i in 0..symbol_quotes.len() {
                for j in 0..symbol_quotes.len() {
                    if i == j {
                        continue;
                    }

                    let buy_quote = symbol_quotes[i];
                    let sell_quote = symbol_quotes[j];

                    // Check if we can buy at ask and sell at bid for profit
                    let buy_price = buy_quote.ask.as_f64();
                    let sell_price = sell_quote.bid.as_f64();

                    if sell_price > buy_price && buy_price > 0.0 {
                        let spread_bps = ((sell_price - buy_price) / buy_price) * 10_000.0;

                        if spread_bps >= self.config.min_spread_bps {
                            let estimated_profit =
                                (sell_price - buy_price) * self.config.max_position_size;

                            opportunities.push(ArbitrageOpportunity {
                                symbol: buy_quote.symbol.clone(),
                                buy_venue: buy_quote.venue.clone(),
                                sell_venue: sell_quote.venue.clone(),
                                buy_price: buy_quote.ask,
                                sell_price: sell_quote.bid,
                                spread_bps,
                                estimated_profit,
                                detected_at: Timestamp::now(),
                            });
                        }
                    }
                }
            }
        }

        opportunities
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

impl Agent for ArbitrageAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, _portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Scan for arbitrage opportunities
        let new_opportunities = self.scan_opportunities();

        // Update stored opportunities
        *self.opportunities.write() = new_opportunities;

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
    fn test_arbitrage_agent_creation() {
        let agent = ArbitrageAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_arbitrage_detection() {
        let agent = ArbitrageAgent::with_defaults();
        agent.start().unwrap();

        // Add quotes from two venues with price discrepancy
        let symbol = Symbol::new("BTC-USD");

        agent.update_quote(VenueQuote {
            symbol: symbol.clone(),
            venue: "venue_a".to_string(),
            bid: Price::from_f64(100.0),
            ask: Price::from_f64(100.10),
            timestamp: Timestamp::now(),
        });

        agent.update_quote(VenueQuote {
            symbol: symbol.clone(),
            venue: "venue_b".to_string(),
            bid: Price::from_f64(100.20),
            ask: Price::from_f64(100.30),
            timestamp: Timestamp::now(),
        });

        // Process to detect arbitrage
        let portfolio = Portfolio::default();
        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let opportunities = agent.get_opportunities();
        assert!(!opportunities.is_empty());

        // Should find opportunity: buy at venue_a ask (100.10), sell at venue_b bid (100.20)
        let opp = &opportunities[0];
        assert!(opp.spread_bps > 0.0);
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = ArbitrageAgent::with_defaults();

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
