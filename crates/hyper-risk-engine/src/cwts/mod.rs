//! CWTS - Complex Weighted Trading System Integration
//!
//! This module consolidates five major crate integrations into a unified
//! risk management framework based on Complex Adaptive Systems theory.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         CWTS Integration Layer                           │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
//! │  │   Autopoiesis   │  │  Game Theory    │  │   Physics Bridge        │  │
//! │  │ ─────────────── │  │ ─────────────── │  │ ─────────────────────── │  │
//! │  │ • Boundary Mgmt │  │ • Nash Equilib  │  │ • Market→Physics Map   │  │
//! │  │ • Emergence Det │  │ • Multi-Agent   │  │ • Order Flow Dynamics  │  │
//! │  │ • Dissipative   │  │ • Machiavellian │  │ • Rapier Simulation    │  │
//! │  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
//! │           │                    │                        │               │
//! │           └────────────────────┼────────────────────────┘               │
//! │                                │                                        │
//! │                    ┌───────────▼───────────┐                            │
//! │                    │    CWTS Coordinator   │                            │
//! │                    │ ───────────────────── │                            │
//! │                    │ • Risk Aggregation    │                            │
//! │                    │ • Decision Synthesis  │                            │
//! │                    │ • Consensus Protocol  │                            │
//! │                    └───────────┬───────────┘                            │
//! │                                │                                        │
//! │           ┌────────────────────┼────────────────────────┐               │
//! │           │                    │                        │               │
//! │  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────────▼────────────┐  │
//! │  │    Nautilus     │  │  Neural Trader  │  │   Hyper Risk Engine    │  │
//! │  │ ─────────────── │  │ ─────────────── │  │ ───────────────────────│  │
//! │  │ • Exec Bridge   │  │ • 8 NN Models   │  │ • Sentinels            │  │
//! │  │ • Backtest      │  │ • Ensemble Pred │  │ • Agents               │  │
//! │  │ • Live Trading  │  │ • Conformal UQ  │  │ • Fast/Med/Slow Path   │  │
//! │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `cwts-autopoiesis` | Self-organizing systems, boundary maintenance |
//! | `cwts-game-theory` | Nash equilibrium, multi-agent coordination |
//! | `cwts-physics` | Market-physics bridge via Rapier |
//! | `cwts-nautilus` | NautilusTrader execution bridge |
//! | `cwts-neural` | Neural forecasting, ensemble prediction |
//! | `cwts-full` | All CWTS features enabled |
//!
//! ## Scientific References
//!
//! - Maturana & Varela (1980): "Autopoiesis and Cognition"
//! - Prigogine (1977): "Self-Organization in Nonequilibrium Systems"
//! - von Neumann & Morgenstern (1944): "Theory of Games and Economic Behavior"
//! - Nash (1950): "Equilibrium Points in N-Person Games"
//! - Bak et al. (1987): "Self-Organized Criticality"
//! - Castro & Liskov (1999): "Practical Byzantine Fault Tolerance"

#[cfg(feature = "cwts-autopoiesis")]
pub mod autopoiesis_integration;

#[cfg(feature = "cwts-game-theory")]
pub mod game_theory_integration;

#[cfg(feature = "cwts-physics")]
pub mod physics_integration;

#[cfg(feature = "cwts-nautilus")]
pub mod nautilus_integration;

#[cfg(feature = "cwts-neural")]
pub mod neural_integration;

pub mod coordinator;
pub mod bft_consensus;

// Re-exports based on enabled features
#[cfg(feature = "cwts-autopoiesis")]
pub use autopoiesis_integration::{
    AutopoiesisRiskAdapter, BoundaryState, EmergenceAlert, SystemHealth,
};

#[cfg(feature = "cwts-game-theory")]
pub use game_theory_integration::{
    GameTheoryRiskAdapter, NashAnalysis, StrategicPosition, MultiAgentRisk,
};

#[cfg(feature = "cwts-physics")]
pub use physics_integration::{
    PhysicsRiskAdapter, MarketPhysicsState, OrderFlowDynamics, SimulationSignal,
};

#[cfg(feature = "cwts-nautilus")]
pub use nautilus_integration::{
    NautilusRiskAdapter, ExecutionRisk, BacktestRiskMetrics, LiveTradingGuard,
};

#[cfg(feature = "cwts-neural")]
pub use neural_integration::{
    NeuralRiskAdapter, EnsembleForecast, ConfidenceBounds, ModelDisagreement,
};

pub use coordinator::{
    CWTSCoordinator, CWTSConfig, CWTSDecision, IntegratedRiskMetrics,
    SubsystemRisk, SubsystemId,
};

pub use bft_consensus::{
    BftConsensusEngine, BftConsensusConfig, BftConsensusResult,
    BftRiskMessage, ConsensusPhase, ConsensusProof, ConsensusRound,
    RiskProposal, ProposalContext, SubsystemVote, ViewChangeReason,
};
