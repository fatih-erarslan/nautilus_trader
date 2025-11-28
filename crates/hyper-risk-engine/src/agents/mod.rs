//! Agent framework for medium-path risk processing.
//!
//! Agents are active processors that run in the medium path (100μs-1ms)
//! to perform sophisticated analysis and decision-making.
//!
//! ## Agent Taxonomy
//!
//! | Agent | Latency | Function |
//! |-------|---------|----------|
//! | PortfolioManagerAgent | <500μs | Position orchestration |
//! | AlphaGeneratorAgent | <800μs | Signal generation |
//! | RegimeDetectionAgent | <1ms | HMM/MS-GARCH regime |
//! | ExecutionAgent | <500μs | Order management |
//! | ResearcherAgent | <1ms | Strategy analysis |
//! | ArbitrageAgent | <800μs | Cross-market arbitrage |
//! | AnomalyDetectionAgent | <1ms | Market anomaly detection |
//! | AssetAllocationAgent | <1ms | Portfolio optimization |
//! | ReconciliationAgent | <10ms | Position reconciliation |
//! | NAVCalculationAgent | <1ms | NAV computation |
//! | QuantResearcherAgent | <100ms | Quantitative analysis |
//! | MacroResearcherAgent | <100ms | Macro analysis |

pub mod base;
pub mod portfolio_manager;
pub mod alpha_generator;
pub mod regime_detection;
pub mod market_maker;
pub mod execution;
pub mod arbitrage;
pub mod anomaly_detection;
pub mod asset_allocation;
pub mod reconciliation;
pub mod nav_calculation;
pub mod quant_researcher;
pub mod macro_researcher;

pub use base::{Agent, AgentId, AgentStatus, AgentConfig};
pub use portfolio_manager::PortfolioManagerAgent;
pub use alpha_generator::AlphaGeneratorAgent;
pub use regime_detection::RegimeDetectionAgent;
pub use market_maker::{MarketMakerAgent, MarketMakerConfig, Quote, InventoryState, ToxicityScore};
pub use execution::ExecutionAgent;
pub use arbitrage::{ArbitrageAgent, ArbitrageConfig, ArbitrageOpportunity, VenueQuote};
pub use anomaly_detection::{AnomalyDetectionAgent, AnomalyDetectionConfig, Anomaly, AnomalyType};
pub use asset_allocation::{AssetAllocationAgent, AssetAllocationConfig, AssetTarget, AllocationAction, AllocationRecommendation, AssetStats};
pub use reconciliation::{ReconciliationAgent, ReconciliationConfig, ReconciliationResult, Discrepancy, DiscrepancyType, ExternalPosition, ExternalCashBalance};
pub use nav_calculation::{NAVCalculationAgent, NAVCalculationConfig, NAVResult, NAVComponent, PositionValuation};
pub use quant_researcher::{QuantResearcherAgent, QuantResearcherConfig, StrategyMetrics, FactorExposure, ResearchFinding, FindingType};
pub use macro_researcher::{MacroResearcherAgent, MacroResearcherConfig, EconomicIndicator, Trend, MacroRegime, MacroInsight, InsightCategory, PortfolioAdjustment};
