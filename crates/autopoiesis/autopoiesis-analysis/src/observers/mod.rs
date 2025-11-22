//! Observer implementations for the autopoiesis system

pub mod momentum;
pub mod volatility;
pub mod volume;
pub mod sentiment;
pub mod pattern;

pub use momentum::MomentumObserver;
pub use volatility::VolatilityObserver;
pub use volume::VolumeObserver;
pub use sentiment::SentimentObserver;
pub use pattern::PatternObserver;