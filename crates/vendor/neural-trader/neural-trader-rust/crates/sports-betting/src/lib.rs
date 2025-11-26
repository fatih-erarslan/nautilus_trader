//! Sports Betting Integration with Syndicate Management
//!
//! This crate provides comprehensive sports betting functionality including:
//! - Risk management and portfolio optimization
//! - Syndicate collaboration and governance
//! - Capital management and profit distribution
//! - Multi-member coordination with RBAC
//!
//! # Example
//!
//! ```rust,no_run
//! use nt_sports_betting::{SyndicateManager, RiskFramework};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new syndicate
//!     let syndicate = SyndicateManager::new("elite-bettors".to_string());
//!
//!     // Add members with capital contributions
//!     syndicate.add_member("alice", 10000.0).await?;
//!     syndicate.add_member("bob", 5000.0).await?;
//!
//!     // Initialize risk framework
//!     let risk = RiskFramework::new();
//!     risk.set_max_bet_size(1000.0)?;
//!
//!     Ok(())
//! }
//! ```

pub mod risk;
pub mod syndicate;
pub mod odds_api;
pub mod models;
pub mod error;

// Re-exports for convenience
pub use risk::{
    PortfolioRiskManager,
    BettingLimitsController,
    MarketRiskAnalyzer,
    SyndicateRiskController,
    PerformanceMonitor,
    RiskFramework,
};

pub use syndicate::{
    CapitalManager,
    VotingSystem,
    MemberManager,
    CollaborationManager,
    SyndicateManager,
};

pub use models::{
    Member,
    MemberRole,
    SyndicateConfig,
    BetPosition,
    RiskMetrics,
    BetStatus,
};

pub use error::{Error, Result};

/// Version of the sports betting crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
