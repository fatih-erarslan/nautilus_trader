//! Cortical bus integration (ultra-low-latency messaging)
//!
//! Integrates cognition system with the HyperPhysics cortical bus:
//! - **Spike routing** (<50ns latency)
//! - **Message translation** between cognition and bus formats
//! - **Pattern memory** integration (HNSW + LSH)
//! - **pBit fabric** coordination
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │      CORTICAL BUS INTEGRATION                          │
//! │                                                        │
//! │   ┌──────────────┐        ┌──────────────┐            │
//! │   │  Cognition   │◄──────►│ Cortical Bus │            │
//! │   │  Messages    │        │  (Spikes)    │            │
//! │   └──────────────┘        └──────────────┘            │
//! │         │                        │                    │
//! │         ▼                        ▼                    │
//! │   ┌──────────────┐        ┌──────────────┐            │
//! │   │  Message     │        │  Spike       │            │
//! │   │  Router      │        │  Injector    │            │
//! │   └──────────────┘        └──────────────┘            │
//! │         │                        │                    │
//! │         └────────────┬───────────┘                    │
//! │                      ▼                                │
//! │               ┌─────────────┐                         │
//! │               │  Pattern    │                         │
//! │               │  Memory     │                         │
//! │               │  (HNSW+LSH) │                         │
//! │               └─────────────┘                         │
//! └────────────────────────────────────────────────────────┘
//! ```

use crate::error::{CognitionError, Result};
use crate::types::{CognitionPhase, NodeId};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, trace};

#[cfg(feature = "cortical-bus")]
use hyperphysics_cortical_bus::prelude::*;

/// Route configuration
#[derive(Debug, Clone)]
pub struct RouteConfig {
    /// Enable spike injection
    pub enable_spike_injection: bool,

    /// Enable pattern memory
    pub enable_pattern_memory: bool,

    /// Enable pBit integration
    pub enable_pbit: bool,

    /// Default routing hint (0-255)
    pub default_routing_hint: u8,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            enable_spike_injection: true,
            enable_pattern_memory: true,
            enable_pbit: true,
            default_routing_hint: 0,
        }
    }
}

/// Message router (cognition ↔ cortical bus)
pub struct MessageRouter {
    /// Configuration
    config: RouteConfig,

    /// Message count
    message_count: Arc<RwLock<u64>>,

    /// Spike count
    spike_count: Arc<RwLock<u64>>,

    #[cfg(feature = "cortical-bus")]
    /// Cortical bus reference
    bus: Option<Arc<CorticalBus>>,
}

impl MessageRouter {
    /// Create new message router
    pub fn new(config: RouteConfig) -> Self {
        debug!("🔌 Message router initialized");

        Self {
            config,
            message_count: Arc::new(RwLock::new(0)),
            spike_count: Arc::new(RwLock::new(0)),
            #[cfg(feature = "cortical-bus")]
            bus: None,
        }
    }

    /// Attach cortical bus
    #[cfg(feature = "cortical-bus")]
    pub fn attach_bus(&mut self, bus: Arc<CorticalBus>) {
        self.bus = Some(bus);
        debug!("Cortical bus attached to router");
    }

    /// Route cognition message to bus (as spike)
    #[cfg(feature = "cortical-bus")]
    pub fn route_to_bus(&self, source: NodeId, phase: CognitionPhase, strength: i8) -> Result<()> {
        let bus = self.bus.as_ref()
            .ok_or_else(|| CognitionError::Integration("No bus attached".to_string()))?;

        if !self.config.enable_spike_injection {
            return Ok(());
        }

        // Create spike from cognition message
        let spike = Spike::new(
            source as u32,
            0, // timestamp (relative)
            strength,
            self.routing_hint_for_phase(phase),
        );

        // Inject into bus
        bus.inject_spike(spike)
            .map_err(|e| CognitionError::CorticalBus(format!("Spike injection failed: {}", e)))?;

        let mut count = self.spike_count.write();
        *count += 1;

        trace!("Spike injected: phase={}, strength={}", phase.name(), strength);

        Ok(())
    }

    /// Route bus spike to cognition
    #[cfg(feature = "cortical-bus")]
    pub fn route_from_bus(&self, spike: &Spike) -> Result<(NodeId, CognitionPhase, i8)> {
        let node_id = spike.source_id as NodeId;
        let phase = self.phase_from_routing_hint(spike.routing_hint);
        let strength = spike.strength;

        let mut count = self.message_count.write();
        *count += 1;

        trace!("Spike received: node={}, phase={}, strength={}", node_id, phase.name(), strength);

        Ok((node_id, phase, strength))
    }

    /// Get routing hint for cognition phase
    fn routing_hint_for_phase(&self, phase: CognitionPhase) -> u8 {
        match phase {
            CognitionPhase::Perceiving => 0x10,
            CognitionPhase::Cognizing => 0x20,
            CognitionPhase::Deliberating => 0x30,
            CognitionPhase::Intending => 0x40,
            CognitionPhase::Integrating => 0x50,
            CognitionPhase::Acting => 0x60,
        }
    }

    /// Get cognition phase from routing hint
    fn phase_from_routing_hint(&self, hint: u8) -> CognitionPhase {
        match hint & 0xF0 {
            0x10 => CognitionPhase::Perceiving,
            0x20 => CognitionPhase::Cognizing,
            0x30 => CognitionPhase::Deliberating,
            0x40 => CognitionPhase::Intending,
            0x50 => CognitionPhase::Integrating,
            0x60 => CognitionPhase::Acting,
            _ => CognitionPhase::Perceiving, // Default
        }
    }

    /// Get message count
    pub fn message_count(&self) -> u64 {
        *self.message_count.read()
    }

    /// Get spike count
    pub fn spike_count(&self) -> u64 {
        *self.spike_count.read()
    }

    /// Reset counters
    pub fn reset_counters(&self) {
        *self.message_count.write() = 0;
        *self.spike_count.write() = 0;
        debug!("Router counters reset");
    }
}

/// Cortical bus integration
pub struct CorticalBusIntegration {
    /// Message router
    router: Arc<MessageRouter>,

    /// Configuration
    config: RouteConfig,
}

impl CorticalBusIntegration {
    /// Create new cortical bus integration
    pub fn new() -> Result<Self> {
        let config = RouteConfig::default();
        let router = Arc::new(MessageRouter::new(config.clone()));

        debug!("🧠 Cortical bus integration initialized");

        Ok(Self {
            router,
            config,
        })
    }

    /// Get message router
    pub fn router(&self) -> Arc<MessageRouter> {
        Arc::clone(&self.router)
    }

    /// Attach cortical bus
    /// Note: This is a placeholder. In production, router should be RefCell or RwLock
    #[cfg(feature = "cortical-bus")]
    pub fn attach(&self, _bus: Arc<CorticalBus>) {
        // TODO: Implement proper interior mutability for router
        // For now, router attachment should happen at construction time
        tracing::warn!("CorticalBusIntegration::attach is not yet implemented");
    }

    /// Send cognition message to bus
    #[cfg(feature = "cortical-bus")]
    pub fn send(&self, source: NodeId, phase: CognitionPhase, strength: i8) -> Result<()> {
        self.router.route_to_bus(source, phase, strength)
    }

    /// Receive spike from bus
    #[cfg(feature = "cortical-bus")]
    pub fn receive(&self, spike: &Spike) -> Result<(NodeId, CognitionPhase, i8)> {
        self.router.route_from_bus(spike)
    }

    /// Get statistics
    pub fn stats(&self) -> IntegrationStats {
        IntegrationStats {
            messages_routed: self.router.message_count(),
            spikes_injected: self.router.spike_count(),
        }
    }
}

/// Integration statistics
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Messages routed from bus to cognition
    pub messages_routed: u64,

    /// Spikes injected from cognition to bus
    pub spikes_injected: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_config() {
        let config = RouteConfig::default();
        assert!(config.enable_spike_injection);
        assert!(config.enable_pattern_memory);
        assert_eq!(config.default_routing_hint, 0);
    }

    #[test]
    fn test_message_router() {
        let router = MessageRouter::new(RouteConfig::default());
        assert_eq!(router.message_count(), 0);
        assert_eq!(router.spike_count(), 0);
    }

    #[test]
    fn test_routing_hints() {
        let router = MessageRouter::new(RouteConfig::default());

        let hint_perceiving = router.routing_hint_for_phase(CognitionPhase::Perceiving);
        assert_eq!(hint_perceiving, 0x10);

        let phase = router.phase_from_routing_hint(0x10);
        assert_eq!(phase, CognitionPhase::Perceiving);

        let hint_acting = router.routing_hint_for_phase(CognitionPhase::Acting);
        assert_eq!(hint_acting, 0x60);

        let phase = router.phase_from_routing_hint(0x60);
        assert_eq!(phase, CognitionPhase::Acting);
    }

    #[test]
    fn test_integration() {
        let integration = CorticalBusIntegration::new().unwrap();
        let stats = integration.stats();

        assert_eq!(stats.messages_routed, 0);
        assert_eq!(stats.spikes_injected, 0);
    }
}
