//! Trading system integrations
//!
//! CWTS-Ultra, ATS-Core, and Intelligence system bridges

pub mod active_inference_env;

pub use active_inference_env::{
    AITradingAgent, MarketState, TradingAction, TradingEnvironment, TradingMetrics,
};
