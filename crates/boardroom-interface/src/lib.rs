#![doc = include_str!("../README.md")]

pub mod agent;
pub mod boardroom;
pub mod consensus;
pub mod discovery;
pub mod message;
pub mod routing;
pub mod transport;

pub use agent::{Agent, AgentId, AgentInfo, AgentCapability};
pub use boardroom::{Boardroom, BoardroomConfig};
pub use consensus::{ConsensusProtocol, ConsensusResult, VotingPolicy};
pub use discovery::{DiscoveryService, ServiceRegistry};
pub use message::{Message, MessageType, MessageRouter};
pub use routing::{RoutingStrategy, LoadBalancer};
pub use transport::{Transport, RedisTransport, ZmqTransport};

// Re-export commonly used types
pub use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Ensure all modules are accessible
        let _ = AgentId::new();
    }
}