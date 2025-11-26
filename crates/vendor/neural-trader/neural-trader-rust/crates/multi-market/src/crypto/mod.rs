//! Cryptocurrency Trading Module
//!
//! Provides comprehensive cryptocurrency trading functionality including:
//! - DeFi integration (Beefy Finance, yield farming)
//! - Cross-exchange arbitrage
//! - Liquidity pool strategies
//! - Gas optimization and MEV protection

pub mod defi;
pub mod arbitrage;
pub mod yield_farming;
pub mod gas;
pub mod strategies;

// Re-exports
pub use defi::{DefiManager, YieldVault, VaultStrategy};
pub use arbitrage::{ArbitrageEngine, CrossExchangeOpportunity};
pub use yield_farming::{YieldFarmingStrategy, FarmingPool, RewardCalculator};
pub use gas::{GasOptimizer, GasEstimate, MevProtection};
pub use strategies::{DexArbitrageStrategy, LiquidityPoolStrategy};
