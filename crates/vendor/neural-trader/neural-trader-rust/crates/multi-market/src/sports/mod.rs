//! Sports Betting Module
//!
//! Provides comprehensive sports betting functionality including:
//! - The Odds API integration for live odds
//! - Kelly Criterion for optimal stake sizing
//! - Arbitrage opportunity detection
//! - Syndicate management for pooled betting
//! - Real-time odds streaming

pub mod odds_api;
pub mod kelly;
pub mod arbitrage;
pub mod syndicate;
pub mod streaming;

// Re-exports
pub use odds_api::{OddsApiClient, Sport, Market as OddsMarket, BookmakerOdds, Event};
pub use kelly::{KellyOptimizer, BettingOpportunity, KellyResult};
pub use arbitrage::{ArbitrageDetector, SportsArbitrageOpportunity, ArbitrageBet};
pub use syndicate::{Syndicate, Member, MemberRole, SyndicateBet, BetStatus};
pub use streaming::{OddsStreamClient, PollingOddsStreamer, WebSocketOddsStreamer, OddsUpdate, StreamFilter};
