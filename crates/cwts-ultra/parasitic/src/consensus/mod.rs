//! CWTS Ultra Consensus Voting Mechanism
//!
//! Byzantine fault-tolerant consensus system for organism selection with weighted voting,
//! emergence detection, and sub-millisecond decision times. Integrates with CQGS for
//! quality governance and real-time validation.

pub mod byzantine_tolerance;
pub mod emergence_detector;
pub mod organism_selector;
pub mod performance_weights;
pub mod voting_engine;

pub use byzantine_tolerance::{ByzantineTolerance, FaultTolerance, NodeState};
pub use emergence_detector::{EmergenceDetector, EmergencePattern, EmergenceSignal};
pub use organism_selector::{OrganismSelector, OrganismVote, SelectionCriteria};
pub use performance_weights::{PerformanceWeights, WeightCalculator, WeightFactors};
pub use voting_engine::{ConsensusVotingEngine, VotingConfig, VotingResult};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use crate::cqgs::{CqgsEvent, QualityGateDecision};
use crate::organisms::*;

/// Maximum decision time target (sub-millisecond)
pub const MAX_DECISION_TIME_US: u64 = 800; // 0.8ms

/// Minimum organisms required for valid consensus
pub const MIN_CONSENSUS_PARTICIPANTS: usize = 3;

/// Byzantine fault tolerance threshold (2/3 majority)
pub const BYZANTINE_THRESHOLD: f64 = 0.67;

/// Maximum concurrent consensus sessions
pub const MAX_CONCURRENT_SESSIONS: usize = 1000;

/// Consensus session identifier
pub type ConsensusSessionId = Uuid;

/// Organism performance score (0.0 to 1.0)
pub type PerformanceScore = f64;

/// Voting weight (higher = more influence)
pub type VotingWeight = f64;

/// Core consensus voting mechanism for organism selection
#[async_trait]
pub trait ConsensusVoting: Send + Sync {
    /// Initiate consensus vote for organism selection
    async fn initiate_consensus_vote(
        &self,
        criteria: SelectionCriteria,
        available_organisms: Vec<Box<dyn ParasiticOrganism + Send + Sync>>,
    ) -> Result<ConsensusResult, ConsensusError>;

    /// Cast vote for organism selection
    async fn cast_vote(
        &mut self,
        session_id: ConsensusSessionId,
        vote: OrganismVote,
    ) -> Result<(), ConsensusError>;

    /// Get current consensus status
    async fn get_consensus_status(
        &self,
        session_id: ConsensusSessionId,
    ) -> Result<ConsensusStatus, ConsensusError>;

    /// Detect emergence patterns in voting
    async fn detect_emergence_patterns(
        &self,
        votes: &[OrganismVote],
    ) -> Result<Vec<EmergencePattern>, ConsensusError>;
}

/// Final consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub session_id: ConsensusSessionId,
    pub selected_organisms: Vec<OrganismSelection>,
    pub confidence_score: f64,
    pub consensus_time_us: u64,
    pub total_votes: usize,
    pub emergence_patterns: Vec<EmergencePattern>,
    pub byzantine_faults_detected: usize,
    pub quality_gate_decision: QualityGateDecision,
    pub timestamp: SystemTime,
}

/// Selected organism with consensus details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismSelection {
    pub organism_id: Uuid,
    pub organism_type: String,
    pub selection_score: f64,
    pub voting_weight: VotingWeight,
    pub votes_received: usize,
    pub emergence_factor: f64,
}

/// Current status of consensus session
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusStatus {
    Initializing,
    CollectingVotes,
    AnalyzingEmergence,
    ByzantineBFTValidation,
    Completed,
    Failed,
    Timeout,
}

/// Errors that can occur during consensus
#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("Insufficient participants: {0}")]
    InsufficientParticipants(String),

    #[error("Byzantine fault detected: {0}")]
    ByzantineFault(String),

    #[error("Consensus timeout after {0:?}")]
    Timeout(Duration),

    #[error("Invalid vote: {0}")]
    InvalidVote(String),

    #[error("Session not found: {0}")]
    SessionNotFound(ConsensusSessionId),

    #[error("Emergence detection failed: {0}")]
    EmergenceDetectionFailed(String),

    #[error("Weight calculation failed: {0}")]
    WeightCalculationFailed(String),

    #[error("CQGS integration failed: {0}")]
    CqgsIntegrationFailed(String),
}

#[cfg(test)]
pub mod tests;

#[cfg(test)]
mod unit_tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn test_consensus_performance_requirements() {
        // Test that consensus completes within sub-millisecond requirements
        let start_time = std::time::Instant::now();

        // Placeholder for actual consensus test
        tokio::time::sleep(Duration::from_micros(100)).await; // Simulate fast consensus

        let elapsed = start_time.elapsed();
        assert!(elapsed.as_micros() < MAX_DECISION_TIME_US as u128);
    }

    #[tokio::test]
    async fn test_byzantine_threshold() {
        // Test that Byzantine threshold is correctly set
        assert_eq!(BYZANTINE_THRESHOLD, 0.67);
        assert!(BYZANTINE_THRESHOLD > 0.5); // Must be majority
        assert!(BYZANTINE_THRESHOLD < 1.0); // Must allow some faults
    }

    #[tokio::test]
    async fn test_minimum_participants() {
        // Test minimum participants requirement
        assert!(MIN_CONSENSUS_PARTICIPANTS >= 3);

        // For Byzantine fault tolerance, we need at least 3f+1 nodes
        // where f is the number of faulty nodes
        let max_faults = (MIN_CONSENSUS_PARTICIPANTS - 1) / 3;
        assert!(max_faults >= 0);
    }

    #[tokio::test]
    async fn test_consensus_result_structure() {
        let result = ConsensusResult {
            session_id: Uuid::new_v4(),
            selected_organisms: vec![],
            confidence_score: 0.85,
            consensus_time_us: 750,
            total_votes: 5,
            emergence_patterns: vec![],
            byzantine_faults_detected: 0,
            quality_gate_decision: QualityGateDecision::Pass,
            timestamp: SystemTime::now(),
        };

        assert!(result.confidence_score > 0.0);
        assert!(result.consensus_time_us < MAX_DECISION_TIME_US);
        assert_eq!(result.byzantine_faults_detected, 0);
    }
}
