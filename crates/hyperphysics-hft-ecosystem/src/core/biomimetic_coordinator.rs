//! Biomimetic algorithm coordinator
//!
//! Coordinates multiple biomimetic algorithms using Byzantine consensus

use super::physics_engine_router::PhysicsResult;
use crate::swarms::tier1_execution::{MarketState, Tier1SwarmExecutor};
use crate::Result;

/// Biomimetic algorithm tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiomimeticTier {
    /// Tier 1: <1ms execution (Whale, Bat, Firefly, Cuckoo, PSO)
    Tier1,
    /// Tier 2: 1-10ms execution (GA, DE, GWO, ABC, ACO)
    Tier2,
    /// Tier 3: 10ms+ execution (BFO, SSO, MFO, Salp)
    Tier3,
    /// All algorithms with Byzantine consensus
    All,
}

/// Biomimetic coordinator
pub struct BiomimeticCoordinator {
    /// Active tier
    tier: BiomimeticTier,

    /// Tier 1 executor
    tier1_executor: Tier1SwarmExecutor,
}

impl BiomimeticCoordinator {
    /// Create a new coordinator
    pub fn new(tier: BiomimeticTier) -> Result<Self> {
        Ok(Self {
            tier,
            tier1_executor: Tier1SwarmExecutor::new()?,
        })
    }

    /// Coordinate biomimetic swarms
    pub async fn coordinate_swarms(
        &mut self,
        physics_result: &PhysicsResult,
    ) -> Result<BiomimeticDecision> {
        // Convert physics result to market state
        let market_state = self.physics_to_market(physics_result)?;

        // Execute appropriate tier
        let decision = match self.tier {
            BiomimeticTier::Tier1 => {
                let swarm_decision = self.tier1_executor.execute(&market_state).await?;
                BiomimeticDecision {
                    consensus: swarm_decision.action,
                    confidence: swarm_decision.confidence,
                    tier_used: self.tier,
                    latency_us: swarm_decision.latency_us,
                }
            }
            _ => {
                // TODO: Implement Tier 2 and Tier 3
                BiomimeticDecision {
                    consensus: super::Action::Hold,
                    confidence: 0.5,
                    tier_used: self.tier,
                    latency_us: 0,
                }
            }
        };

        Ok(decision)
    }

    /// Convert physics result to market state
    fn physics_to_market(&self, _physics_result: &PhysicsResult) -> Result<MarketState> {
        // TODO: Implement proper conversion from physics simulation to market state
        Ok(MarketState {
            order_book: vec![],
            price_history: vec![],
            volume_profile: vec![],
            volatility: 0.02,
            trend_strength: 0.5,
        })
    }
}

/// Biomimetic decision
#[derive(Debug, Clone)]
pub struct BiomimeticDecision {
    /// Consensus action
    pub consensus: super::Action,

    /// Confidence score
    pub confidence: f64,

    /// Tier that produced this decision
    pub tier_used: BiomimeticTier,

    /// Latency in microseconds
    pub latency_us: u64,
}

impl From<BiomimeticDecision> for super::TradingDecision {
    fn from(decision: BiomimeticDecision) -> Self {
        Self {
            action: decision.consensus,
            confidence: decision.confidence,
            size: 1.0, // Default
        }
    }
}
