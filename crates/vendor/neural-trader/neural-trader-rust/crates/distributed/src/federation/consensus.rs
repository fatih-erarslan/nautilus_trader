// Consensus protocols for distributed decision-making

use super::{AgentId, MessageBus, Message, MessageType};
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Consensus protocol type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusProtocol {
    /// Simple majority voting
    Majority,

    /// Weighted voting based on agent capabilities
    Weighted,

    /// Byzantine fault tolerant (2/3 majority)
    Byzantine,

    /// Raft consensus
    Raft,

    /// Unanimous agreement required
    Unanimous,
}

/// Consensus proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    /// Unique proposal ID
    pub id: Uuid,

    /// Proposer agent
    pub proposer: AgentId,

    /// Proposal description
    pub description: String,

    /// Proposal data
    pub data: serde_json::Value,

    /// Required consensus protocol
    pub protocol: ConsensusProtocol,

    /// Minimum votes required
    pub quorum: usize,

    /// Deadline for voting
    pub deadline: chrono::DateTime<chrono::Utc>,
}

/// Vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Voter agent ID
    pub voter: AgentId,

    /// Proposal being voted on
    pub proposal_id: Uuid,

    /// Vote decision
    pub decision: VoteDecision,

    /// Vote weight (for weighted consensus)
    pub weight: f64,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Vote decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteDecision {
    /// Approve the proposal
    Approve,

    /// Reject the proposal
    Reject,

    /// Abstain from voting
    Abstain,
}

/// Consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Proposal ID
    pub proposal_id: Uuid,

    /// Consensus reached
    pub consensus_reached: bool,

    /// Final decision
    pub decision: Option<VoteDecision>,

    /// Vote counts
    pub vote_counts: VoteCounts,

    /// Participating agents
    pub participants: Vec<AgentId>,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Vote counts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VoteCounts {
    /// Number of approve votes
    pub approve: usize,

    /// Number of reject votes
    pub reject: usize,

    /// Number of abstain votes
    pub abstain: usize,

    /// Total votes
    pub total: usize,

    /// Weighted approve (for weighted consensus)
    pub weighted_approve: f64,

    /// Weighted reject (for weighted consensus)
    pub weighted_reject: f64,
}

/// Consensus manager
pub struct ConsensusManager {
    /// Active proposals
    proposals: Arc<RwLock<HashMap<Uuid, ConsensusProposal>>>,

    /// Votes per proposal
    votes: Arc<RwLock<HashMap<Uuid, Vec<Vote>>>>,

    /// Completed consensus results
    results: Arc<RwLock<HashMap<Uuid, ConsensusResult>>>,

    /// Message bus for communication
    message_bus: Arc<MessageBus>,

    /// Agent weights (for weighted consensus)
    agent_weights: Arc<RwLock<HashMap<AgentId, f64>>>,
}

impl ConsensusManager {
    /// Create new consensus manager
    pub fn new(message_bus: Arc<MessageBus>) -> Self {
        Self {
            proposals: Arc::new(RwLock::new(HashMap::new())),
            votes: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            message_bus,
            agent_weights: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a consensus proposal
    pub async fn create_proposal(&self, proposal: ConsensusProposal) -> Result<Uuid> {
        let proposal_id = proposal.id;

        // Store proposal
        self.proposals.write().await.insert(proposal_id, proposal.clone());
        self.votes.write().await.insert(proposal_id, Vec::new());

        // Broadcast proposal to all agents
        let message = Message::new(
            MessageType::ConsensusRequest,
            proposal.proposer.clone(),
            None, // Broadcast
            serde_json::to_value(&proposal)?,
        );

        self.message_bus.send(message).await?;

        tracing::info!("Created consensus proposal: {}", proposal_id);
        Ok(proposal_id)
    }

    /// Submit a vote
    pub async fn submit_vote(&self, vote: Vote) -> Result<()> {
        // Verify proposal exists
        {
            let proposals = self.proposals.read().await;
            if !proposals.contains_key(&vote.proposal_id) {
                return Err(DistributedError::FederationError(
                    format!("Proposal not found: {}", vote.proposal_id),
                ));
            }
        }

        // Add vote
        {
            let mut votes = self.votes.write().await;
            let proposal_votes = votes.get_mut(&vote.proposal_id).ok_or_else(|| {
                DistributedError::FederationError("Vote storage error".to_string())
            })?;

            // Check for duplicate vote
            if proposal_votes.iter().any(|v| v.voter == vote.voter) {
                return Err(DistributedError::FederationError(
                    "Duplicate vote".to_string(),
                ));
            }

            proposal_votes.push(vote.clone());
        }

        // Broadcast vote
        let message = Message::new(
            MessageType::ConsensusVote,
            vote.voter.clone(),
            None,
            serde_json::to_value(&vote)?,
        );

        self.message_bus.send(message).await?;

        // Check if consensus is reached
        self.check_consensus(&vote.proposal_id).await?;

        Ok(())
    }

    /// Check if consensus is reached for a proposal
    async fn check_consensus(&self, proposal_id: &Uuid) -> Result<()> {
        let proposals = self.proposals.read().await;
        let proposal = proposals
            .get(proposal_id)
            .ok_or_else(|| DistributedError::FederationError("Proposal not found".to_string()))?;

        let votes = self.votes.read().await;
        let proposal_votes = votes.get(proposal_id).unwrap();

        // Check if quorum is met
        if proposal_votes.len() < proposal.quorum {
            return Ok(()); // Not enough votes yet
        }

        // Calculate vote counts
        let mut counts = VoteCounts::default();
        for vote in proposal_votes {
            counts.total += 1;
            match vote.decision {
                VoteDecision::Approve => {
                    counts.approve += 1;
                    counts.weighted_approve += vote.weight;
                }
                VoteDecision::Reject => {
                    counts.reject += 1;
                    counts.weighted_reject += vote.weight;
                }
                VoteDecision::Abstain => counts.abstain += 1,
            }
        }

        // Determine consensus based on protocol
        let (consensus_reached, decision) = match proposal.protocol {
            ConsensusProtocol::Majority => {
                let approved = counts.approve > counts.reject;
                (approved || counts.reject > counts.approve, Some(if approved { VoteDecision::Approve } else { VoteDecision::Reject }))
            }
            ConsensusProtocol::Weighted => {
                let approved = counts.weighted_approve > counts.weighted_reject;
                (true, Some(if approved { VoteDecision::Approve } else { VoteDecision::Reject }))
            }
            ConsensusProtocol::Byzantine => {
                // Requires 2/3 majority
                let threshold = (counts.total * 2) / 3;
                let approved = counts.approve >= threshold;
                (approved || counts.reject >= threshold, Some(if approved { VoteDecision::Approve } else { VoteDecision::Reject }))
            }
            ConsensusProtocol::Unanimous => {
                let approved = counts.approve == counts.total;
                let rejected = counts.reject > 0;
                (approved || rejected, Some(if approved { VoteDecision::Approve } else { VoteDecision::Reject }))
            }
            ConsensusProtocol::Raft => {
                // Simple majority for Raft
                let approved = counts.approve > counts.total / 2;
                (approved || counts.reject > counts.total / 2, Some(if approved { VoteDecision::Approve } else { VoteDecision::Reject }))
            }
        };

        if consensus_reached {
            // Store result
            let result = ConsensusResult {
                proposal_id: *proposal_id,
                consensus_reached,
                decision,
                vote_counts: counts,
                participants: proposal_votes.iter().map(|v| v.voter.clone()).collect(),
                timestamp: chrono::Utc::now(),
            };

            self.results.write().await.insert(*proposal_id, result.clone());

            tracing::info!(
                "Consensus reached for proposal {}: {:?}",
                proposal_id,
                decision
            );
        }

        Ok(())
    }

    /// Get consensus result
    pub async fn get_result(&self, proposal_id: &Uuid) -> Option<ConsensusResult> {
        self.results.read().await.get(proposal_id).cloned()
    }

    /// Set agent weight for weighted consensus
    pub async fn set_agent_weight(&self, agent_id: AgentId, weight: f64) {
        self.agent_weights.write().await.insert(agent_id, weight);
    }

    /// Get statistics
    pub async fn stats(&self) -> ConsensusStats {
        let active_proposals = self.proposals.read().await.len();
        let completed_consensus = self.results.read().await.len();
        let total_votes: usize = self.votes.read().await.values().map(|v| v.len()).sum();

        ConsensusStats {
            active_proposals,
            completed_consensus,
            total_votes,
        }
    }
}

/// Consensus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusStats {
    /// Number of active proposals
    pub active_proposals: usize,

    /// Number of completed consensus results
    pub completed_consensus: usize,

    /// Total votes cast
    pub total_votes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vote_counts() {
        let mut counts = VoteCounts::default();
        counts.approve = 5;
        counts.reject = 3;
        counts.total = 8;

        assert_eq!(counts.approve, 5);
        assert_eq!(counts.reject, 3);
    }

    #[test]
    fn test_consensus_protocol() {
        assert_eq!(
            ConsensusProtocol::Majority,
            ConsensusProtocol::Majority
        );
    }

    #[tokio::test]
    async fn test_consensus_manager_creation() {
        let bus = Arc::new(MessageBus::new());
        let manager = ConsensusManager::new(bus);

        let stats = manager.stats().await;
        assert_eq!(stats.active_proposals, 0);
    }
}
