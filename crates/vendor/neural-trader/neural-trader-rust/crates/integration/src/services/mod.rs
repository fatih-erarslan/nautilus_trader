//! High-level services layer.

pub mod trading;
pub mod analytics;
pub mod risk;
pub mod neural;

pub use trading::TradingService;
pub use analytics::AnalyticsService;
pub use risk::RiskService;
pub use neural::NeuralService;
