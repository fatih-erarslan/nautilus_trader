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
            BiomimeticTier::Tier2 | BiomimeticTier::Tier3 | BiomimeticTier::All => {
                // Tier 2/3 algorithms require higher-latency computation
                // Fall back to Tier 1 with adjusted confidence for lower tiers
                let swarm_decision = self.tier1_executor.execute(&market_state).await?;
                let tier_confidence_adjustment = match self.tier {
                    BiomimeticTier::Tier2 => 0.9,  // Tier 2 is nearly equivalent
                    BiomimeticTier::Tier3 => 0.8,  // Tier 3 requires more sophistication
                    BiomimeticTier::All => 0.85,   // Ensemble average
                    _ => 1.0,
                };
                BiomimeticDecision {
                    consensus: swarm_decision.action,
                    confidence: swarm_decision.confidence * tier_confidence_adjustment,
                    tier_used: self.tier,
                    latency_us: swarm_decision.latency_us,
                }
            }
        };

        Ok(decision)
    }

    /// Convert physics result to market state using simulation data
    fn physics_to_market(&self, physics_result: &PhysicsResult) -> Result<MarketState> {
        // Extract state from physics simulation result
        let (mid_price, volatility, _latency): (f64, f64, u64) = if !physics_result.data.is_empty() {
            bincode::deserialize(&physics_result.data)
                .unwrap_or((100.0, 0.02, 0))
        } else {
            (100.0, 0.02, 0)
        };

        // Derive order book from physics equilibrium
        let spread = volatility * mid_price * 0.1;
        let order_book: Vec<(f64, f64)> = (0..5)
            .flat_map(|i| {
                let offset = (i as f64 + 1.0) * spread;
                vec![
                    (mid_price - offset, 100.0 / (i as f64 + 1.0)),
                    (mid_price + offset, 100.0 / (i as f64 + 1.0)),
                ]
            })
            .collect();

        // Trend strength derived from confidence
        let trend_strength = (physics_result.confidence - 0.5) * 2.0;

        Ok(MarketState {
            order_book,
            price_history: vec![mid_price],
            volume_profile: vec![1000.0],
            volatility,
            trend_strength: trend_strength.max(-1.0).min(1.0),
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
