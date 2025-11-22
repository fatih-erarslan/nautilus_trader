//! Ecosystem builder
//!
//! Fluent API for constructing the HFT ecosystem

use super::*;
use crate::Result;

/// Builder for HFT ecosystem
pub struct EcosystemBuilder {
    physics_engine: Option<PhysicsEngine>,
    biomimetic_tier: Option<BiomimeticTier>,
    config: EcosystemConfig,
}

impl EcosystemBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            physics_engine: None,
            biomimetic_tier: None,
            config: EcosystemConfig::default(),
        }
    }
    
    /// Set the physics engine
    pub fn with_physics_engine(mut self, engine: PhysicsEngine) -> Self {
        self.physics_engine = Some(engine);
        self
    }
    
    /// Set the biomimetic tier
    pub fn with_biomimetic_tier(mut self, tier: BiomimeticTier) -> Self {
        self.biomimetic_tier = Some(tier);
        self
    }
    
    /// Enable or disable formal verification
    pub fn with_formal_verification(mut self, enabled: bool) -> Self {
        self.config.formal_verification = enabled;
        self
    }
    
    /// Set target latency
    pub fn with_target_latency_us(mut self, latency_us: u64) -> Self {
        self.config.target_latency_us = latency_us;
        self
    }
    
    /// Build the ecosystem
    pub async fn build(self) -> Result<HFTEcosystem> {
        let engine = self.physics_engine.unwrap_or(PhysicsEngine::Rapier);
        let tier = self.biomimetic_tier.unwrap_or(BiomimeticTier::Tier1);
        
        let physics_router = Arc::new(PhysicsEngineRouter::new(
            engine,
            self.config.formal_verification
        ));
        
        let biomimetic_coord = Arc::new(tokio::sync::RwLock::new(BiomimeticCoordinator::new(tier)?));
        
        Ok(HFTEcosystem {
            physics_router,
            biomimetic_coord,
            config: self.config,
        })
    }
}

impl Default for EcosystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}
