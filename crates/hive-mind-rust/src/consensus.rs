//! Distributed consensus mechanisms for the hive mind system
//! 
//! This module provides the main entry point for the enhanced Byzantine fault tolerance
//! consensus system with RAFT, PBFT, financial consensus, and performance optimizations.

pub mod pbft;
pub mod raft;
pub mod byzantine_detector;
pub mod financial_consensus;
pub mod performance_optimizer;
pub mod fault_tolerance;
pub mod message_ordering;
pub mod crypto_verifier;

// Re-export the main consensus engine
pub use pbft::PbftConsensus;
pub use raft::OptimizedRaft;
pub use byzantine_detector::ByzantineDetector;
pub use financial_consensus::FinancialConsensus;
pub use performance_optimizer::PerformanceOptimizer;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, oneshot};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug, warn};

use crate::{
    config::{ConsensusConfig, ConsensusAlgorithm},
    network::P2PNetwork,
    metrics::MetricsCollector,
    error::{ConsensusError, HiveMindError, Result},
};

/// Main consensus engine for distributed decision making
#[derive(Debug)]
pub struct ConsensusEngine {
    /// Configuration
    config: ConsensusConfig,
    
    /// Network layer for communication
    network: Arc<P2PNetwork>,
    
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    
    /// Current consensus state
    state: Arc<RwLock<ConsensusState>>,
    
    /// Pending proposals
    proposals: Arc<RwLock<HashMap<Uuid, ProposalState>>>,
    
    /// Message channel for internal communication
    message_tx: mpsc::UnboundedSender<ConsensusMessage>,
    message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<ConsensusMessage>>>>,
}

/// Current state of the consensus system
#[derive(Debug, Clone)]
pub struct ConsensusState {
    /// Current leader node (if any)
    pub current_leader: Option<Uuid>,
    
    /// Current term/epoch
    pub current_term: u64,
    
    /// Last voted term
    pub voted_for: Option<Uuid>,
    
    /// Whether this node is the leader
    pub is_leader: bool,
    
    /// List of known peers
    pub peers: Vec<Uuid>,
    
    /// Last heartbeat time
    pub last_heartbeat: Option<Instant>,
    
    /// Node's current role
    pub role: NodeRole,
}

/// Roles a node can have in consensus
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRole {
    Follower,
    Candidate,
    Leader,
}

/// State of a proposal in the consensus process
#[derive(Debug, Clone)]
pub struct ProposalState {
    /// Unique proposal ID
    pub id: Uuid,
    
    /// Proposal content
    pub content: serde_json::Value,
    
    /// Proposer node ID
    pub proposer: Uuid,
    
    /// Current status
    pub status: ProposalStatus,
    
    /// Votes received
    pub votes: HashMap<Uuid, Vote>,
    
    /// When the proposal was created
    pub created_at: Instant,
    
    /// Deadline for voting
    pub deadline: Instant,
}

/// Status of a consensus proposal
#[derive(Debug, Clone, PartialEq)]
pub enum ProposalStatus {
    Proposed,
    VotingInProgress,
    Accepted,
    Rejected,
    TimedOut,
}

/// A vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Voter node ID
    pub voter: Uuid,
    
    /// Vote decision
    pub decision: VoteDecision,
    
    /// Optional reasoning
    pub reasoning: Option<String>,
    
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Vote decision types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VoteDecision {
    Accept,
    Reject,
    Abstain,
}

/// Messages used in consensus protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Request for votes (Raft)
    RequestVote {
        term: u64,
        candidate_id: Uuid,
        last_log_index: u64,
        last_log_term: u64,
    },
    
    /// Vote response (Raft)
    VoteResponse {
        term: u64,
        vote_granted: bool,
        voter_id: Uuid,
    },
    
    /// Heartbeat from leader (Raft)
    Heartbeat {
        term: u64,
        leader_id: Uuid,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    },
    
    /// Heartbeat response
    HeartbeatResponse {
        term: u64,
        success: bool,
        follower_id: Uuid,
    },
    
    /// Proposal submission
    ProposeDecision {
        proposal_id: Uuid,
        content: serde_json::Value,
        proposer: Uuid,
    },
    
    /// Vote on proposal
    VoteOnProposal {
        proposal_id: Uuid,
        vote: Vote,
    },
    
    /// Proposal result
    ProposalResult {
        proposal_id: Uuid,
        status: ProposalStatus,
        votes: HashMap<Uuid, Vote>,
    },
}

/// Log entry for Raft consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub content: serde_json::Value,
}

/// Result of a consensus operation
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub proposal_id: Uuid,
    pub status: ProposalStatus,
    pub votes: HashMap<Uuid, Vote>,
    pub majority_decision: VoteDecision,
}

impl ConsensusEngine {
    /// Create a new consensus engine
    pub async fn new(
        config: &ConsensusConfig,
        network: Arc<P2PNetwork>,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        
        let state = Arc::new(RwLock::new(ConsensusState {
            current_leader: None,
            current_term: 0,
            voted_for: None,
            is_leader: false,
            peers: Vec::new(),
            last_heartbeat: None,
            role: NodeRole::Follower,
        }));
        
        let proposals = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            config: config.clone(),
            network,
            metrics,
            state,
            proposals,
            message_tx,
            message_rx: Arc::new(RwLock::new(Some(message_rx))),
        })
    }
    
    /// Start the consensus engine
    pub async fn start(&self) -> Result<()> {
        info!("Starting consensus engine with algorithm: {:?}", self.config.algorithm);
        
        // Start message processing loop
        self.start_message_processing().await?;
        
        // Start algorithm-specific processes
        match self.config.algorithm {
            ConsensusAlgorithm::Raft => self.start_raft().await?,
            ConsensusAlgorithm::Pbft => self.start_pbft().await?,
            ConsensusAlgorithm::Gossip => self.start_gossip().await?,
            ConsensusAlgorithm::Hybrid => self.start_hybrid().await?,
        }
        
        info!("Consensus engine started successfully");
        Ok(())
    }
    
    /// Stop the consensus engine
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping consensus engine");
        // Implementation would stop all consensus processes
        Ok(())
    }
    
    /// Submit a proposal for consensus
    pub async fn submit_proposal(&self, content: serde_json::Value) -> Result<Uuid> {
        let proposal_id = Uuid::new_v4();
        let node_id = self.network.get_node_id().await?;
        
        debug!("Submitting proposal {}", proposal_id);
        
        let proposal = ProposalState {
            id: proposal_id,
            content: content.clone(),
            proposer: node_id,
            status: ProposalStatus::Proposed,
            votes: HashMap::new(),
            created_at: Instant::now(),
            deadline: Instant::now() + self.config.timeout,
        };
        
        // Store proposal
        {
            let mut proposals = self.proposals.write().await;
            proposals.insert(proposal_id, proposal);
        }
        
        // Broadcast proposal to network
        let message = ConsensusMessage::ProposeDecision {
            proposal_id,
            content,
            proposer: node_id,
        };
        
        self.broadcast_message(message).await?;
        
        // Start voting process
        self.start_voting_process(proposal_id).await?;
        
        Ok(proposal_id)
    }
    
    /// Vote on a proposal
    pub async fn vote_on_proposal(
        &self,
        proposal_id: Uuid,
        decision: VoteDecision,
        reasoning: Option<String>,
    ) -> Result<()> {
        let node_id = self.network.get_node_id().await?;
        
        debug!("Voting on proposal {} with decision: {:?}", proposal_id, decision);
        
        let vote = Vote {
            voter: node_id,
            decision,
            reasoning,
            timestamp: chrono::Utc::now(),
        };
        
        // Update local proposal state
        {
            let mut proposals = self.proposals.write().await;
            if let Some(proposal) = proposals.get_mut(&proposal_id) {
                proposal.votes.insert(node_id, vote.clone());
                proposal.status = ProposalStatus::VotingInProgress;
            }
        }
        
        // Broadcast vote
        let message = ConsensusMessage::VoteOnProposal {
            proposal_id,
            vote,
        };
        
        self.broadcast_message(message).await?;
        
        Ok(())
    }
    
    /// Get current consensus leader
    pub async fn get_current_leader(&self) -> Result<Option<Uuid>> {
        let state = self.state.read().await;
        Ok(state.current_leader)
    }
    
    /// Check if this node is the current leader
    pub async fn is_leader(&self) -> Result<bool> {
        let state = self.state.read().await;
        Ok(state.is_leader)
    }
    
    /// Get consensus statistics
    pub async fn get_statistics(&self) -> Result<ConsensusStatistics> {
        let proposals = self.proposals.read().await;
        let state = self.state.read().await;
        
        let total_proposals = proposals.len();
        let accepted_proposals = proposals.values()
            .filter(|p| p.status == ProposalStatus::Accepted)
            .count();
        
        Ok(ConsensusStatistics {
            current_term: state.current_term,
            total_proposals,
            accepted_proposals,
            current_leader: state.current_leader,
            peer_count: state.peers.len(),
            role: state.role.clone(),
        })
    }
    
    /// Start message processing loop
    async fn start_message_processing(&self) -> Result<()> {
        let mut receiver = {
            let mut rx_guard = self.message_rx.write().await;
            rx_guard.take().ok_or_else(|| HiveMindError::InvalidState {
                message: "Message receiver already taken".to_string(),
            })?
        };
        
        let state = self.state.clone();
        let proposals = self.proposals.clone();
        let network = self.network.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            while let Some(message) = receiver.recv().await {
                if let Err(e) = Self::process_consensus_message(
                    &message,
                    &state,
                    &proposals,
                    &network,
                    &config,
                ).await {
                    error!("Failed to process consensus message: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Process incoming consensus messages
    async fn process_consensus_message(
        message: &ConsensusMessage,
        state: &Arc<RwLock<ConsensusState>>,
        proposals: &Arc<RwLock<HashMap<Uuid, ProposalState>>>,
        network: &Arc<P2PNetwork>,
        config: &ConsensusConfig,
    ) -> Result<()> {
        match message {
            ConsensusMessage::ProposeDecision { proposal_id, content, proposer } => {
                Self::handle_proposal(
                    *proposal_id,
                    content.clone(),
                    *proposer,
                    proposals,
                    config,
                ).await?;
            }
            
            ConsensusMessage::VoteOnProposal { proposal_id, vote } => {
                Self::handle_vote(*proposal_id, vote.clone(), proposals, config).await?;
            }
            
            ConsensusMessage::RequestVote { term, candidate_id, .. } => {
                Self::handle_vote_request(*term, *candidate_id, state, network).await?;
            }
            
            ConsensusMessage::Heartbeat { term, leader_id, .. } => {
                Self::handle_heartbeat(*term, *leader_id, state).await?;
            }
            
            _ => {
                debug!("Received unhandled consensus message type");
            }
        }
        
        Ok(())
    }
    
    /// Handle incoming proposal
    async fn handle_proposal(
        proposal_id: Uuid,
        content: serde_json::Value,
        proposer: Uuid,
        proposals: &Arc<RwLock<HashMap<Uuid, ProposalState>>>,
        config: &ConsensusConfig,
    ) -> Result<()> {
        debug!("Handling proposal {} from {}", proposal_id, proposer);
        
        let proposal = ProposalState {
            id: proposal_id,
            content,
            proposer,
            status: ProposalStatus::VotingInProgress,
            votes: HashMap::new(),
            created_at: Instant::now(),
            deadline: Instant::now() + config.timeout,
        };
        
        let mut proposals_guard = proposals.write().await;
        proposals_guard.insert(proposal_id, proposal);
        
        Ok(())
    }
    
    /// Handle incoming vote
    async fn handle_vote(
        proposal_id: Uuid,
        vote: Vote,
        proposals: &Arc<RwLock<HashMap<Uuid, ProposalState>>>,
        config: &ConsensusConfig,
    ) -> Result<()> {
        debug!("Handling vote on proposal {} from {}", proposal_id, vote.voter);
        
        let mut proposals_guard = proposals.write().await;
        if let Some(proposal) = proposals_guard.get_mut(&proposal_id) {
            proposal.votes.insert(vote.voter, vote);
            
            // Check if we have enough votes to make a decision
            Self::check_proposal_completion(proposal, config)?;
        }
        
        Ok(())
    }
    
    /// Check if a proposal has enough votes to complete
    fn check_proposal_completion(
        proposal: &mut ProposalState,
        config: &ConsensusConfig,
    ) -> Result<()> {
        let total_votes = proposal.votes.len();
        let min_votes = config.min_nodes;
        
        if total_votes >= min_votes {
            // Count votes
            let accept_votes = proposal.votes.values()
                .filter(|v| v.decision == VoteDecision::Accept)
                .count();
            
            let majority = (min_votes / 2) + 1;
            
            if accept_votes >= majority {
                proposal.status = ProposalStatus::Accepted;
                info!("Proposal {} accepted with {} votes", proposal.id, accept_votes);
            } else {
                proposal.status = ProposalStatus::Rejected;
                info!("Proposal {} rejected", proposal.id);
            }
        } else if proposal.deadline <= Instant::now() {
            proposal.status = ProposalStatus::TimedOut;
            warn!("Proposal {} timed out", proposal.id);
        }
        
        Ok(())
    }
    
    /// Handle vote request (Raft)
    async fn handle_vote_request(
        term: u64,
        candidate_id: Uuid,
        state: &Arc<RwLock<ConsensusState>>,
        network: &Arc<P2PNetwork>,
    ) -> Result<()> {
        let mut state_guard = state.write().await;
        let vote_granted = if term > state_guard.current_term {
            state_guard.current_term = term;
            state_guard.voted_for = Some(candidate_id);
            true
        } else {
            false
        };
        
        let node_id = network.get_node_id().await?;
        let response = ConsensusMessage::VoteResponse {
            term,
            vote_granted,
            voter_id: node_id,
        };
        
        // Send vote response back to candidate
        // Implementation would send response through network
        
        Ok(())
    }
    
    /// Handle heartbeat from leader (Raft)
    async fn handle_heartbeat(
        term: u64,
        leader_id: Uuid,
        state: &Arc<RwLock<ConsensusState>>,
    ) -> Result<()> {
        let mut state_guard = state.write().await;
        
        if term >= state_guard.current_term {
            state_guard.current_term = term;
            state_guard.current_leader = Some(leader_id);
            state_guard.last_heartbeat = Some(Instant::now());
            state_guard.role = NodeRole::Follower;
        }
        
        Ok(())
    }
    
    /// Start Raft consensus algorithm
    async fn start_raft(&self) -> Result<()> {
        info!("Starting Raft consensus algorithm");
        
        // Start election timeout monitoring
        let state = self.state.clone();
        let message_tx = self.message_tx.clone();
        let election_timeout = self.config.leader_election_timeout;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(election_timeout / 2);
            
            loop {
                interval.tick().await;
                
                let should_start_election = {
                    let state_guard = state.read().await;
                    match state_guard.role {
                        NodeRole::Follower => {
                            if let Some(last_heartbeat) = state_guard.last_heartbeat {
                                last_heartbeat.elapsed() > election_timeout
                            } else {
                                true // No heartbeat received yet
                            }
                        }
                        _ => false,
                    }
                };
                
                if should_start_election {
                    // Start leader election
                    // Implementation would trigger election process
                    debug!("Starting leader election");
                }
            }
        });
        
        Ok(())
    }
    
    /// Start PBFT consensus algorithm
    async fn start_pbft(&self) -> Result<()> {
        info!("Starting PBFT consensus algorithm");
        // Implementation for PBFT
        Ok(())
    }
    
    /// Start Gossip consensus algorithm
    async fn start_gossip(&self) -> Result<()> {
        info!("Starting Gossip consensus algorithm");
        // Implementation for Gossip
        Ok(())
    }
    
    /// Start Hybrid consensus algorithm
    async fn start_hybrid(&self) -> Result<()> {
        info!("Starting Hybrid consensus algorithm");
        // Implementation for Hybrid approach
        Ok(())
    }
    
    /// Start voting process for a proposal
    async fn start_voting_process(&self, proposal_id: Uuid) -> Result<()> {
        let proposals = self.proposals.clone();
        let timeout = self.config.timeout;
        
        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            
            // Check if proposal is still pending
            let mut proposals_guard = proposals.write().await;
            if let Some(proposal) = proposals_guard.get_mut(&proposal_id) {
                if proposal.status == ProposalStatus::VotingInProgress {
                    proposal.status = ProposalStatus::TimedOut;
                    warn!("Proposal {} timed out during voting", proposal_id);
                }
            }
        });
        
        Ok(())
    }
    
    /// Broadcast message to all peers
    async fn broadcast_message(&self, message: ConsensusMessage) -> Result<()> {
        // Implementation would broadcast through network layer
        debug!("Broadcasting consensus message: {:?}", message);
        Ok(())
    }
}

/// Consensus statistics
#[derive(Debug, Clone)]
pub struct ConsensusStatistics {
    pub current_term: u64,
    pub total_proposals: usize,
    pub accepted_proposals: usize,
    pub current_leader: Option<Uuid>,
    pub peer_count: usize,
    pub role: NodeRole,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConsensusConfig;

    #[tokio::test]
    async fn test_consensus_engine_creation() {
        // This would require mocking the network and metrics
        // For now, we'll test the basic structures
        
        let vote = Vote {
            voter: Uuid::new_v4(),
            decision: VoteDecision::Accept,
            reasoning: Some("Test vote".to_string()),
            timestamp: chrono::Utc::now(),
        };
        
        assert_eq!(vote.decision, VoteDecision::Accept);
    }
    
    #[test]
    fn test_proposal_state() {
        let proposal = ProposalState {
            id: Uuid::new_v4(),
            content: serde_json::json!({"test": "data"}),
            proposer: Uuid::new_v4(),
            status: ProposalStatus::Proposed,
            votes: HashMap::new(),
            created_at: Instant::now(),
            deadline: Instant::now() + Duration::from_secs(30),
        };
        
        assert_eq!(proposal.status, ProposalStatus::Proposed);
        assert!(proposal.votes.is_empty());
    }
    
    #[test]
    fn test_node_roles() {
        assert_ne!(NodeRole::Leader, NodeRole::Follower);
        assert_ne!(NodeRole::Candidate, NodeRole::Leader);
    }
}