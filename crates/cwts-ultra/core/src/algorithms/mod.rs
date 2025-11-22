pub mod bayesian_var_engine;
pub mod cascade_networks;
pub mod cuckoo_simd;
pub mod fee_optimizer;
pub mod hft_algorithms;
pub mod liquidation_engine;
pub mod lockfree_orderbook;
pub mod order_matching;
pub mod probabilistic_risk_engine;
pub mod risk_management;
pub mod safe_orderbook;
pub mod slippage_calculator;
pub mod wasp_lockfree;

#[cfg(test)]
pub mod tests;

// Fixed ambiguous re-exports by being specific about what we export
pub use bayesian_var_engine::{
    BayesianPriors, BayesianVaREngine, BayesianVaRError, BayesianVaRResult, BinanceMarketData,
    E2BTrainingConfig, E2BTrainingResults, EmergenceProperties, KupiecTestResult,
    MonteCarloSamples,
};
pub use cascade_networks::*;
pub use cuckoo_simd::{SimdWhaleDetector, WhaleDetectorStats, WhaleMovement, WhalePatternType};
pub use fee_optimizer::{FeeCalculation, FeeOptimizer};
pub use hft_algorithms::*;
pub use liquidation_engine::*;
pub use lockfree_orderbook::{AtomicOrder, LockFreeOrderBook};
pub use order_matching::{MatchingAlgorithm, OrderBook as MatchingOrderBook, OrderMatchingEngine};
pub use probabilistic_risk_engine::{
    BayesianParameters, HeavyTailDistribution, ProbabilisticRiskEngine, ProbabilisticRiskError,
    ProbabilisticRiskMetrics,
};
pub use risk_management::*;
pub use slippage_calculator::{SlippageAnalysis, SlippageCalculator};
pub use wasp_lockfree::*;
