//! # Cryptographic Systems Module
//!
//! Provides Ed25519 agent identity management, payment mandate structures,
//! and Byzantine fault-tolerant consensus for multi-agent coordination.
//!
//! ## Components
//!
//! - **identity**: Ed25519 keypair generation, signing, and verification
//! - **mandate**: Payment authorization structures with spend caps and merchant rules
//! - **consensus**: Byzantine fault-tolerant voting for multi-agent agreement
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use hyperphysics_core::crypto::{
//!     identity::AgentIdentity,
//!     mandate::{PaymentMandate, Period, MandateKind},
//!     consensus::{ByzantineConsensus, Vote},
//! };
//! use chrono::{Utc, Duration};
//!
//! // 1. Generate agent identities
//! let queen = AgentIdentity::generate("queen-seraphina".to_string());
//! let agent1 = AgentIdentity::generate("agent-001".to_string());
//! let agent2 = AgentIdentity::generate("agent-002".to_string());
//!
//! // 2. Create payment mandate
//! let mandate = PaymentMandate::new(
//!     agent1.agent_id().to_string(),
//!     queen.agent_id().to_string(),
//!     10000,  // $100.00
//!     "USD".to_string(),
//!     Period::Daily,
//!     MandateKind::Intent,
//!     Utc::now() + Duration::hours(24),
//!     vec!["trusted-merchant.com".to_string()],
//! );
//!
//! // 3. Queen signs the mandate
//! let mandate_bytes = mandate.to_bytes();
//! let signature = queen.sign(&mandate_bytes).unwrap();
//!
//! // 4. Byzantine consensus voting
//! let consensus = ByzantineConsensus::new(2.0/3.0);  // 67% threshold
//!
//! let message = b"approve mandate";
//! let vote1_sig = agent1.sign(message).unwrap();
//! let vote2_sig = agent2.sign(message).unwrap();
//!
//! let votes = vec![
//!     Vote {
//!         agent_id: agent1.agent_id().to_string(),
//!         approve: true,
//!         signature: hex::encode(vote1_sig.to_bytes()),
//!         public_key: agent1.export_public_key(),
//!     },
//!     Vote {
//!         agent_id: agent2.agent_id().to_string(),
//!         approve: true,
//!         signature: hex::encode(vote2_sig.to_bytes()),
//!         public_key: agent2.export_public_key(),
//!     },
//! ];
//!
//! // Verify signatures
//! consensus.verify_votes(&votes, message).unwrap();
//!
//! // Check consensus
//! let result = consensus.check_consensus(&votes);
//! assert!(result.approved);
//! ```

pub mod identity;
pub mod mandate;
pub mod consensus;
pub mod signed_state;

// Re-export key types
pub use identity::AgentIdentity;
pub use mandate::{PaymentMandate, Period, MandateKind, SignedMandate};
pub use consensus::{ByzantineConsensus, Vote, ConsensusResult, ConsensusTracker};
pub use signed_state::{
    AuditRecord, ConsciousnessMetrics, SignedConsciousnessState, StateMetadata,
};
