//! Trading engine implementations

pub mod autopoietic;
pub mod market_making;
pub mod arbitrage;
pub mod trend_following;

pub use autopoietic::AutopoieticEngine;
pub use market_making::MarketMakingEngine;
pub use arbitrage::ArbitrageEngine;
pub use trend_following::TrendFollowingEngine;