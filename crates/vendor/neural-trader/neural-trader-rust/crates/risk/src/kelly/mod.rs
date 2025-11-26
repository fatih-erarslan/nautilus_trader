//! Kelly Criterion position sizing
//!
//! Optimal position sizing based on Kelly Criterion:
//! - Single asset: f* = (p × w - (1-p)) / w
//! - Multi-asset: Mean-variance optimization
//! - Fractional Kelly: Conservative multipliers (0.25x-0.5x)

pub mod single_asset;
pub mod multi_asset;

pub use single_asset::{KellySingleAsset, RiskTolerance};
pub use multi_asset::{KellyMultiAsset, KellyMultiAssetBuilder};
