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

pub mod base;
pub mod portfolio_manager;
pub mod alpha_generator;
pub mod regime_detection;

pub use base::{Agent, AgentId, AgentStatus, AgentConfig};
pub use portfolio_manager::PortfolioManagerAgent;
pub use alpha_generator::AlphaGeneratorAgent;
pub use regime_detection::RegimeDetectionAgent;
